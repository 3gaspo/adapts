"""Evaluate the official pretrained Cross-RAG checkpoint on our held-out T3 payload."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
import types
from typing import Any, Sequence

import torch
from einops import rearrange
from transformers import AutoConfig

from src.experiments.runtime import log_experiment_separator, setup_logging


LOGGER = logging.getLogger(__name__)
EXPECTED_LAGS = 512
EXPECTED_HORIZON = 64
EXPECTED_NEIGHBORS = 15


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _code_root(root: Path) -> Path:
    candidates = (root, root / "cross-rag")
    for candidate in candidates:
        if (candidate / "models" / "CrossRAG.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot find models/CrossRAG.py below {root}; clone the official Cross-RAG repository"
    )


def _load_model(
    crossrag_root: Path,
    base_checkpoint: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    code_root = _code_root(crossrag_root)
    os.environ["INPUT_LEN"] = str(EXPECTED_LAGS)
    os.environ["RETRIEVE_SPACE"] = "X"
    package_name = "_adaptation_crossrag_release_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(code_root / "models")]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    sys.path.insert(0, str(code_root))
    try:
        module = importlib.import_module(f"{package_name}.CrossRAG")
    finally:
        sys.path.pop(0)
    model_class = module.ChronosBoltModelForForecastingWithRetrieval
    config = AutoConfig.from_pretrained(
        str(base_checkpoint),
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        str(base_checkpoint),
        config=config,
        augment="moe",
        local_files_only=True,
    )
    state = _torch_load(checkpoint)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    cleaned = OrderedDict(
        (str(key).removeprefix("module."), value)
        for key, value in state.items()
    )
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def _flatten_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    prefix = "eval"
    return {
        "x": rearrange(
            payload[f"{prefix}_X_values"].float(),
            "date user lags -> (date user) lags",
        ),
        "xc": rearrange(
            payload[f"{prefix}_Xc_values"].float(),
            "date user neighbor lags -> (date user) neighbor lags",
        ),
        "yc": rearrange(
            payload[f"{prefix}_Yc_values"].float(),
            "date user neighbor horizon -> (date user) neighbor horizon",
        ),
        "distance": rearrange(
            payload[f"{prefix}_distance_x_xc"].float(),
            "date user neighbor -> (date user) neighbor",
        ),
        "y": rearrange(
            payload[f"{prefix}_Y_values"].float(),
            "date user horizon -> (date user) horizon",
        ),
        "vanilla": rearrange(
            payload[f"{prefix}_preds"].float(),
            "date user horizon -> (date user) horizon",
        ),
    }


def _validate_shapes(arrays: dict[str, torch.Tensor]) -> None:
    if arrays["x"].shape[-1] != EXPECTED_LAGS:
        raise ValueError(
            f"Cross-RAG requires L={EXPECTED_LAGS}, found {arrays['x'].shape[-1]}"
        )
    if arrays["y"].shape[-1] != EXPECTED_HORIZON:
        raise ValueError(
            f"Cross-RAG requires H={EXPECTED_HORIZON}, found {arrays['y'].shape[-1]}"
        )
    if arrays["xc"].shape[1] != EXPECTED_NEIGHBORS:
        raise ValueError(
            f"Cross-RAG requires K={EXPECTED_NEIGHBORS}, found {arrays['xc'].shape[1]}"
        )


def _metrics(
    prediction: torch.Tensor,
    arrays: dict[str, torch.Tensor],
) -> list[dict[str, float | str]]:
    target = arrays["y"]
    error = prediction - target
    scale = arrays["x"].std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    vanilla_nmse = (((arrays["vanilla"] - target) / scale) ** 2).mean()
    nmse = ((error / scale) ** 2).mean()
    return [
        {
            "baseline": "crossrag",
            "split": "eval",
            "mse": float(error.square().mean().item()),
            "mae": float(error.abs().mean().item()),
            "nmse": float(nmse.item()),
            "relative_nmse_improvement_pct": float(
                100.0 * (vanilla_nmse - nmse) / vanilla_nmse.clamp_min(1e-12)
            ),
        }
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crossrag-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    setup_logging()
    log_experiment_separator(LOGGER)
    started = perf_counter()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    crossrag_root = Path(args.crossrag_root).expanduser().resolve()
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path, kind in (
        (input_dir / "eval_prediction_payload.pt", "evaluation payload"),
        (base_checkpoint, "Chronos-Bolt base checkpoint"),
        (checkpoint, "Cross-RAG checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {kind}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_payload(_torch_load(input_dir / "eval_prediction_payload.pt"))
    _validate_shapes(arrays)
    device = torch.device(args.device)
    model_started = perf_counter()
    model = _load_model(crossrag_root, base_checkpoint, checkpoint, device)
    model_load_seconds = perf_counter() - model_started
    quantiles = model.quantiles.detach().float().cpu()
    median_index = int(torch.abs(quantiles - 0.5).argmin().item())
    predictions: list[torch.Tensor] = []
    inference_started = perf_counter()
    with torch.inference_mode():
        for start in range(0, arrays["x"].shape[0], args.batch_size):
            stop = min(start + args.batch_size, arrays["x"].shape[0])
            context = arrays["x"][start:stop].to(device)
            retrieved = torch.cat(
                (arrays["xc"][start:stop], arrays["yc"][start:stop]),
                dim=-1,
            ).to(device)
            distances = arrays["distance"][start:stop].to(device)
            outputs = model(
                context=context,
                retrieved_seq=retrieved,
                distances=distances,
            )
            predictions.append(
                outputs.quantile_preds[:, median_index, :].detach().cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = perf_counter() - inference_started
    prediction = torch.cat(predictions, dim=0)
    rows = _metrics(prediction, arrays)
    metrics_path = output_dir / "crossrag_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    predictions_path = output_dir / "crossrag_predictions.pt"
    torch.save(
        {
            "prediction": prediction,
            "target": arrays["y"],
            "vanilla": arrays["vanilla"],
        },
        predictions_path,
    )
    timing_path = output_dir / "crossrag_timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "elapsed_seconds": perf_counter() - started,
                "examples": int(prediction.shape[0]),
                "batch_size": int(args.batch_size),
                "lags": EXPECTED_LAGS,
                "horizon": EXPECTED_HORIZON,
                "neighbors": EXPECTED_NEIGHBORS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Cross-RAG done examples=%s nmse=%.6f seconds=%.2f",
        prediction.shape[0],
        rows[0]["nmse"],
        perf_counter() - started,
    )
    log_experiment_separator(LOGGER)
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "timing": timing_path,
    }


if __name__ == "__main__":
    main()
