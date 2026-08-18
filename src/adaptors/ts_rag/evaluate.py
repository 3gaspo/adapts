"""Evaluate the released TS-RAG checkpoint on a held-out T3 payload."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch
from einops import rearrange
from transformers import AutoConfig

from src.experiments.runtime import log_experiment_separator, setup_logging
from src.models.ts_rag import ChronosBoltModelForForecastingWithRetrieval


LOGGER = logging.getLogger(__name__)
EXPECTED_LAGS = 512
EXPECTED_HORIZON = 64


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("best.pth")) if path.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one TS-RAG best.pth below {path}, found {len(matches)}"
        )
    return matches[0]


def _load_model(
    base_checkpoint: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        str(base_checkpoint),
        local_files_only=True,
    )
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
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
    return {
        "x": rearrange(
            payload["eval_X_values"].float(),
            "date user lags -> (date user) lags",
        ),
        "xc": rearrange(
            payload["eval_Xc_values"].float(),
            "date user neighbor lags -> (date user) neighbor lags",
        ),
        "yc": rearrange(
            payload["eval_Yc_values"].float(),
            "date user neighbor horizon -> (date user) neighbor horizon",
        ),
        "y": rearrange(
            payload["eval_Y_values"].float(),
            "date user horizon -> (date user) horizon",
        ),
        "vanilla": rearrange(
            payload["eval_preds"].float(),
            "date user horizon -> (date user) horizon",
        ),
    }


def _validate_shapes(
    arrays: dict[str, torch.Tensor],
    neighbors: int,
) -> None:
    if arrays["x"].shape[-1] != EXPECTED_LAGS:
        raise ValueError(
            f"TS-RAG requires L={EXPECTED_LAGS}, found {arrays['x'].shape[-1]}"
        )
    if arrays["y"].shape[-1] != EXPECTED_HORIZON:
        raise ValueError(
            f"TS-RAG requires H={EXPECTED_HORIZON}, found {arrays['y'].shape[-1]}"
        )
    if arrays["xc"].shape[1] != neighbors:
        raise ValueError(
            f"TS-RAG expected K={neighbors}, found {arrays['xc'].shape[1]}"
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
    nmae = (error.abs() / scale).mean()
    candidate_window_mse = error.square().mean(dim=-1)
    vanilla_window_mse = (arrays["vanilla"] - target).square().mean(dim=-1)
    return [
        {
            "baseline": "tsrag",
            "split": "eval",
            "mse": float(error.square().mean().item()),
            "mae": float(error.abs().mean().item()),
            "nmse": float(nmse.item()),
            "nmae": float(nmae.item()),
            "positive_window_pct": float(
                100.0
                * (candidate_window_mse < vanilla_window_mse).float().mean().item()
            ),
            "relative_nmse_improvement_pct": float(
                100.0 * (vanilla_nmse - nmse) / vanilla_nmse.clamp_min(1e-12)
            ),
        }
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chronos-bolt-weights", required=True)
    parser.add_argument("--ts-rag-weights", required=True)
    parser.add_argument("--neighbors", type=int, required=True)
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
    if args.neighbors <= 0:
        raise ValueError("--neighbors must be positive")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_checkpoint = Path(args.chronos_bolt_weights).expanduser().resolve()
    checkpoint = _checkpoint_file(Path(args.ts_rag_weights).expanduser().resolve())
    for path, kind in (
        (input_dir / "eval_prediction_payload.pt", "evaluation payload"),
        (base_checkpoint, "Chronos-Bolt base checkpoint"),
        (checkpoint, "TS-RAG checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {kind}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_payload(_torch_load(input_dir / "eval_prediction_payload.pt"))
    _validate_shapes(arrays, args.neighbors)
    device = torch.device(args.device)
    model_started = perf_counter()
    model = _load_model(base_checkpoint, checkpoint, device)
    model_load_seconds = perf_counter() - model_started
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    quantiles = model.quantiles.detach().float().cpu()
    median_index = int(torch.abs(quantiles - 0.5).argmin().item())
    predictions: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = perf_counter()
    with torch.inference_mode():
        for start in range(0, arrays["x"].shape[0], args.batch_size):
            stop = min(start + args.batch_size, arrays["x"].shape[0])
            retrieved = torch.cat(
                (arrays["xc"][start:stop], arrays["yc"][start:stop]),
                dim=-1,
            ).to(device)
            outputs = model(
                context=arrays["x"][start:stop].to(device),
                retrieved_seq=retrieved,
            )
            predictions.append(
                outputs.quantile_preds[:, median_index, :].detach().cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = perf_counter() - inference_started
    prediction = torch.cat(predictions, dim=0)
    rows = _metrics(prediction, arrays)
    metrics_path = output_dir / "tsrag_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    predictions_path = output_dir / "tsrag_predictions.pt"
    torch.save(
        {
            "prediction": prediction,
            "target": arrays["y"],
            "vanilla": arrays["vanilla"],
        },
        predictions_path,
    )
    timing_path = output_dir / "tsrag_timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "inference_ms_per_example": (
                    1000.0 * inference_seconds / max(int(prediction.shape[0]), 1)
                ),
                "elapsed_seconds": perf_counter() - started,
                "examples": int(prediction.shape[0]),
                "batch_size": int(args.batch_size),
                "lags": EXPECTED_LAGS,
                "horizon": EXPECTED_HORIZON,
                "neighbors": int(args.neighbors),
                "total_parameters": int(total_parameters),
                "trainable_parameters": int(trainable_parameters),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "result_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "adaptation_tsrag_result",
                "metric_fields": list(rows[0]),
                "protocol": {
                    "lags": EXPECTED_LAGS,
                    "horizon": EXPECTED_HORIZON,
                    "neighbors": int(args.neighbors),
                },
                "files": {
                    "metrics": metrics_path.name,
                    "predictions": predictions_path.name,
                    "timing": timing_path.name,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "TS-RAG done examples=%s nmse=%.6f seconds=%.2f",
        prediction.shape[0],
        rows[0]["nmse"],
        perf_counter() - started,
    )
    log_experiment_separator(LOGGER)
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "timing": timing_path,
        "manifest": manifest_path,
    }


if __name__ == "__main__":
    main()
