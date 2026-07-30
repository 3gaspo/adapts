"""Train TS-IFA branches on T1, then train and compare rooters on T2."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from src.data.load_dataset import set_seed
from src.data.neighbors import neighbor_to_query_scale
from src.experiments.prediction_store import PredictionStore
from src.experiments.runtime import log_experiment_separator, setup_logging
from src.experiments.splits import chronological_date_slices
from src.models.models import resolve_device

from .model import TSIFAConfig, TimeSeriesInformedForecastingAdapter


LOGGER = logging.getLogger(__name__)


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(Path(path), map_location="cpu")


def flatten_time_user(value: torch.Tensor) -> torch.Tensor:
    return rearrange(value, "date user ... -> (date user) ...").float()


class PredictionPayloadDataset(Dataset):
    """Flatten ``(date, user, ...)`` payload tensors into examples."""

    required = (
        "preds",
        "preds_context",
        "E_values",
        "X_values",
        "Xc_values",
        "Y_values",
        "Yc_values",
    )

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        prefix: str,
        date_slice: slice | None = None,
        max_samples: int | None = None,
        use_transformed_prediction: bool = False,
    ):
        self.prefix = prefix
        missing = [f"{prefix}_{name}" for name in self.required if f"{prefix}_{name}" not in payload]
        if missing:
            raise KeyError(f"payload is missing required TS-IFA keys {missing}")

        date_slice = slice(None) if date_slice is None else date_slice
        transformed_key = f"{prefix}_preds_transformed"
        self.has_transformed_prediction = transformed_key in payload
        x = payload[f"{prefix}_X_values"][date_slice].float().clone()
        self.n_dates = int(x.shape[0])
        self.n_users = int(x.shape[1])
        x_c_raw = payload[f"{prefix}_Xc_values"][date_slice].float()
        x_c = neighbor_to_query_scale(x, x_c_raw, x_c_raw)
        if x_c.shape[2] <= 0:
            raise ValueError("TS-IFA training requires payloads extracted with neighbors > 0")

        y_c_raw = payload[f"{prefix}_Yc_values"][date_slice].float()
        residual_c_raw = payload[f"{prefix}_E_values"][date_slice].float()
        pred_neighbors_raw = y_c_raw - residual_c_raw
        y_c = neighbor_to_query_scale(x, x_c_raw, y_c_raw)
        residual_c = neighbor_to_query_scale(x, x_c_raw, residual_c_raw, residual=True)
        pred_neighbors = neighbor_to_query_scale(x, x_c_raw, pred_neighbors_raw)

        pred = payload[f"{prefix}_preds"][date_slice].float().clone()
        pred_transformed = (
            payload[transformed_key][date_slice].float().clone()
            if self.has_transformed_prediction and use_transformed_prediction
            else pred
        )
        self.tensors = {
            "x": flatten_time_user(x),
            "x_c": flatten_time_user(x_c),
            "y": flatten_time_user(
                payload[f"{prefix}_Y_values"][date_slice].float().clone()
            ),
            "y_c": flatten_time_user(y_c),
            "pred": flatten_time_user(pred),
            "pred_context": flatten_time_user(
                payload[f"{prefix}_preds_context"][date_slice].float().clone()
            ),
            "pred_transformed": flatten_time_user(pred_transformed),
            "pred_neighbors": flatten_time_user(pred_neighbors),
            "residual_c": flatten_time_user(residual_c),
        }
        n_examples = self.tensors["x"].shape[0]
        if max_samples is not None:
            n_examples = min(n_examples, int(max_samples))
            self.tensors = {key: value[:n_examples] for key, value in self.tensors.items()}

        self.lags = int(self.tensors["x"].shape[-1])
        self.horizon = int(self.tensors["y"].shape[-1])
        self.neighbors = int(self.tensors["x_c"].shape[1])

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


class RandomPredictionPayloadDataset(Dataset):
    """Draw random examples from one chronological split."""

    def __init__(self, source: PredictionPayloadDataset, *, virtual_size: int):
        if len(source) == 0:
            raise ValueError("cannot sample from an empty payload")
        if int(virtual_size) <= 0:
            raise ValueError("virtual_size must be positive")
        self.source = source
        self.virtual_size = int(virtual_size)
        self.lags = source.lags
        self.horizon = source.horizon
        self.neighbors = source.neighbors

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        if len(self.source) == self.source.n_dates * self.source.n_users:
            date_index = int(torch.randint(self.source.n_dates, ()).item())
            user_index = int(torch.randint(self.source.n_users, ()).item())
            source_index = date_index * self.source.n_users + user_index
        else:
            source_index = int(torch.randint(len(self.source), ()).item())
        return self.source[source_index]


def log_scale_diagnostics(name: str, dataset: PredictionPayloadDataset | None) -> None:
    if dataset is None:
        return
    scale = dataset.tensors["x"].std(dim=-1, unbiased=False).float()
    if scale.numel() == 0:
        LOGGER.info("payload scale split=%s samples=0", name)
        return
    quantiles = torch.quantile(
        scale,
        torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.5], dtype=torch.float32),
    )
    LOGGER.info(
        "payload scale split=%s samples=%s std_min=%.6g std_q001=%.6g "
        "std_q01=%.6g std_q05=%.6g std_q10=%.6g std_median=%.6g "
        "below_1e-8=%s below_1e-6=%s below_1e-3=%s",
        name,
        len(dataset),
        *(float(value) for value in quantiles),
        int((scale < 1e-8).sum().item()),
        int((scale < 1e-6).sum().item()),
        int((scale < 1e-3).sum().item()),
    )


def ensure_compatible(
    reference: PredictionPayloadDataset,
    candidate: PredictionPayloadDataset | None,
    *,
    name: str,
) -> None:
    if candidate is None:
        return
    if (candidate.lags, candidate.horizon, candidate.neighbors) != (
        reference.lags,
        reference.horizon,
        reference.neighbors,
    ):
        raise ValueError(f"{name} payload shape is incompatible with train payload")


def query_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def prepare_batch(
    raw: dict[str, torch.Tensor],
    *,
    normalization: str,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    q_mean, q_std = query_stats(raw["x"], eps)
    neighbor_mean = q_mean.unsqueeze(-2)
    neighbor_std = q_std.unsqueeze(-2)
    if normalization == "instance":
        batch = {
            "x": (raw["x"] - q_mean) / q_std,
            "y": (raw["y"] - q_mean) / q_std,
            "pred": (raw["pred"] - q_mean) / q_std,
            "pred_context": (raw["pred_context"] - q_mean) / q_std,
            "pred_transformed": (raw["pred_transformed"] - q_mean) / q_std,
            "x_c": (raw["x_c"] - neighbor_mean) / neighbor_std,
            "y_c": (raw["y_c"] - neighbor_mean) / neighbor_std,
            "pred_neighbors": (raw["pred_neighbors"] - neighbor_mean) / neighbor_std,
        }
        batch["residual_c"] = raw["residual_c"] / neighbor_std
        loss_scale = torch.ones_like(q_std)
    elif normalization == "none":
        batch = dict(raw)
        loss_scale = q_std
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return batch, {"mean": q_mean, "std": q_std, "loss_scale": loss_scale}


def denormalize(
    value: torch.Tensor,
    state: dict[str, torch.Tensor],
    normalization: str,
) -> torch.Tensor:
    if normalization == "instance":
        return value * state["std"] + state["mean"]
    return value


def normalized_square(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (value / scale).pow(2).mean()


def branch_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    vanilla_anchor: float,
) -> dict[str, torch.Tensor]:
    scale = state["loss_scale"]
    residual = normalized_square(outputs["residual_prediction"] - batch["y"], scale)
    memory = normalized_square(outputs["memory_prediction"] - batch["y"], scale)
    anchoring = (
        normalized_square(outputs["residual_prediction"] - batch["pred"], scale)
        + normalized_square(outputs["memory_prediction"] - batch["pred"], scale)
    )
    total = residual + memory
    result = {
        "loss": total,
        "residual": residual,
        "memory": memory,
        "vanilla_anchoring": anchoring,
    }
    if "transformed_delta" in outputs:
        transformed = normalized_square(
            outputs["transformed_prediction"] - batch["y"],
            scale,
        )
        transformed_anchoring = normalized_square(
            outputs["transformed_prediction"] - batch["pred"],
            scale,
        )
        total = total + transformed
        anchoring = anchoring + transformed_anchoring
        result["transformed"] = transformed
    result["vanilla_anchoring"] = anchoring
    result["loss"] = total + float(vanilla_anchor) * anchoring
    return result


def rooter_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    vanilla_anchor: float,
    coefficient_l2: float,
    horizon_smoothness: float,
) -> dict[str, torch.Tensor]:
    scale = state["loss_scale"]
    prediction = normalized_square(outputs["prediction"] - batch["y"], scale)
    anchoring = normalized_square(outputs["prediction"] - batch["pred"], scale)
    coefficients = outputs["coefficients"]
    ridge = coefficients.pow(2).mean()
    if coefficients.shape[-1] > 1:
        smoothness = torch.diff(coefficients, dim=-1).pow(2).mean()
    else:
        smoothness = coefficients.new_zeros(())
    total = (
        prediction
        + float(vanilla_anchor) * anchoring
        + float(coefficient_l2) * ridge
        + float(horizon_smoothness) * smoothness
    )
    return {
        "loss": total,
        "prediction": prediction,
        "vanilla_anchoring": anchoring,
        "coefficient_l2": ridge,
        "horizon_smoothness": smoothness,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def nmse_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    lookback: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    scale = lookback.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((pred - target) / scale).pow(2).mean(dim=-1)


def ridge_prediction(
    outputs: dict[str, torch.Tensor],
    vanilla: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return vanilla + (
        outputs["candidates"] * coefficients.to(outputs["candidates"]).unsqueeze(0)
    ).sum(dim=1)


def fit_horizon_ridge_rooter(
    model: TimeSeriesInformedForecastingAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: str,
    eps: float,
    alpha: float,
) -> torch.Tensor:
    """Fit a no-intercept, horizon-wise ridge rooter from exact T2 statistics."""
    if alpha < 0:
        raise ValueError("ridge rooter alpha cannot be negative")
    model.eval()
    horizon = model.config.horizon
    candidates = len(model.candidate_names)
    xtx = torch.zeros(horizon, candidates, candidates, dtype=torch.float64)
    xty = torch.zeros(horizon, candidates, dtype=torch.float64)
    n_samples = 0
    with torch.inference_mode():
        for raw_cpu in loader:
            raw = move_batch(raw_cpu, device)
            batch, _ = prepare_batch(raw, normalization=normalization, eps=eps)
            outputs = model.forward_branches(batch)
            design = outputs["candidates"].detach().cpu().double()
            target = (batch["y"] - batch["pred"]).detach().cpu().double()
            xtx += torch.einsum("bjh,bkh->hjk", design, design)
            xty += torch.einsum("bjh,bh->hj", design, target)
            n_samples += int(design.shape[0])
    if n_samples == 0:
        raise ValueError("cannot fit ridge rooter on an empty T2 split")

    scale = torch.sqrt(
        torch.diagonal(xtx, dim1=-2, dim2=-1) / float(n_samples)
    ).clamp_min(1e-12)
    scaled_xtx = xtx / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    scaled_xty = xty / scale
    regularized = scaled_xtx + float(alpha) * torch.eye(
        candidates,
        dtype=torch.float64,
    ).unsqueeze(0)
    try:
        scaled_coefficients = torch.linalg.solve(regularized, scaled_xty.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        scaled_coefficients = torch.einsum(
            "hjk,hk->hj",
            torch.linalg.pinv(regularized),
            scaled_xty,
        )
    coefficients = scaled_coefficients / scale
    return rearrange(coefficients.float(), "horizon candidate -> candidate horizon")


def evaluate(
    model: TimeSeriesInformedForecastingAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: str,
    eps: float,
    ridge_coefficients: torch.Tensor | None = None,
    prediction_store: PredictionStore | None = None,
) -> dict[str, float]:
    model.eval()
    variants = (
        "adapted",
        *(f"{name}_branch" for name in model.candidate_names),
    )
    if ridge_coefficients is not None:
        variants = (*variants, "ridge_rooter")
    sums = {
        f"{variant}_{metric}": 0.0
        for variant in variants
        for metric in ("nmse", "mse", "mae")
    }
    prediction_arrays = (
        {
            variant: prediction_store.open(
                "eval",
                "predictions",
                f"ts_ifa_{variant}",
                shape=(len(loader.dataset), model.config.horizon),
                dtype=np.float32,
            )
            for variant in variants
        }
        if prediction_store is not None
        else {}
    )
    coefficient_array = (
        prediction_store.open(
            "eval",
            "gate_diagnostics",
            "neural_rooter_coefficients",
            shape=(
                len(loader.dataset),
                len(model.candidate_names),
                model.config.horizon,
            ),
            dtype=np.float32,
        )
        if prediction_store is not None
        else None
    )
    count = 0
    with torch.inference_mode():
        for raw_cpu in loader:
            raw = move_batch(raw_cpu, device)
            batch, state = prepare_batch(raw, normalization=normalization, eps=eps)
            outputs = model(batch)
            normalized_predictions = {
                "adapted": outputs["prediction"],
                **{
                    f"{name}_branch": outputs["candidates"][:, index]
                    for index, name in enumerate(model.candidate_names)
                },
            }
            if ridge_coefficients is not None:
                normalized_predictions["ridge_rooter"] = ridge_prediction(
                    outputs,
                    batch["pred"],
                    ridge_coefficients,
                )
            predictions = {
                name: denormalize(value, state, normalization)
                for name, value in normalized_predictions.items()
            }
            y = raw["y"]
            n = int(y.shape[0])
            for name, prediction in predictions.items():
                sums[f"{name}_nmse"] += nmse_mean(prediction, y, raw["x"], eps).sum().item()
                sums[f"{name}_mse"] += F.mse_loss(
                    prediction,
                    y,
                    reduction="sum",
                ).item() / y.shape[-1]
                sums[f"{name}_mae"] += F.l1_loss(
                    prediction,
                    y,
                    reduction="sum",
                ).item() / y.shape[-1]
                if prediction_store is not None:
                    prediction_arrays[name][count : count + n] = (
                        prediction.detach().cpu().numpy()
                    )
            if coefficient_array is not None:
                coefficient_array[count : count + n] = (
                    outputs["coefficients"].detach().cpu().numpy()
                )
            count += n

    for prediction in prediction_arrays.values():
        prediction.flush()
    if coefficient_array is not None:
        coefficient_array.flush()
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_step_frequency(value: int | None, *, default: int, name: str) -> int:
    frequency = default if value is None else int(value)
    if frequency <= 0:
        raise ValueError(f"{name} must be positive")
    return frequency


def _set_stage_mode(model: TimeSeriesInformedForecastingAdapter, stage: str) -> None:
    model.train()
    inactive = model.rooter_modules() if stage == "branches" else model.branch_modules()
    for module in inactive:
        module.eval()


def train_stage(
    model: TimeSeriesInformedForecastingAdapter,
    *,
    stage: str,
    loader: DataLoader,
    diagnostic_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[
        [dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ],
    epochs: int,
    device: torch.device,
    normalization: str,
    eps: float,
    grad_clip: float,
    valid_eval_freq: int,
    logging_eval_freq: int,
    ridge_coefficients: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    train_steps: list[dict[str, Any]] = []
    recent_totals: dict[str, float] = {}
    recent_seen = 0
    step = 0
    total_steps = int(epochs) * max(1, len(loader))
    LOGGER.info(
        "stage start stage=%s epochs=%s total_steps=%s valid_eval_freq=%s logging_eval_freq=%s",
        stage,
        epochs,
        total_steps,
        valid_eval_freq,
        logging_eval_freq,
    )
    for epoch in range(1, int(epochs) + 1):
        _set_stage_mode(model, stage)
        for raw_cpu in loader:
            step += 1
            raw = move_batch(raw_cpu, device)
            batch, state = prepare_batch(raw, normalization=normalization, eps=eps)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_branches(batch) if stage == "branches" else model(batch)
            losses = loss_function(outputs, batch, state)
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.branch_parameters() if stage == "branches" else model.rooter_parameters(),
                    grad_clip,
                )
            optimizer.step()

            batch_size = int(raw["x"].shape[0])
            loss_values = {key: float(value.detach().item()) for key, value in losses.items()}
            train_steps.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "step": step,
                    **{f"train_{key}": value for key, value in loss_values.items()},
                }
            )
            if not recent_totals:
                recent_totals = {key: 0.0 for key in loss_values}
            recent_seen += batch_size
            for key, value in loss_values.items():
                recent_totals[key] += value * batch_size

            final_step = step == total_steps
            should_evaluate = step % valid_eval_freq == 0 or final_step
            should_log = step % logging_eval_freq == 0 or final_step
            if not should_evaluate:
                continue
            row = {
                "stage": stage,
                "epoch": epoch,
                "step": step,
                **{
                    f"train_batch_{key}": value / max(recent_seen, 1)
                    for key, value in recent_totals.items()
                },
            }
            diagnostic = evaluate(
                model,
                diagnostic_loader,
                device=device,
                normalization=normalization,
                eps=eps,
                ridge_coefficients=ridge_coefficients,
            )
            row.update({f"t2_{key}": value for key, value in diagnostic.items()})
            history.append(row)
            if should_log:
                focus = (
                    diagnostic["adapted_nmse"]
                    if stage == "rooter"
                    else sum(
                        diagnostic[key]
                        for key in (
                            "residual_branch_nmse",
                            "memory_branch_nmse",
                            "transformed_branch_nmse",
                        )
                        if key in diagnostic
                    )
                    / sum(
                        key in diagnostic
                        for key in (
                            "residual_branch_nmse",
                            "memory_branch_nmse",
                            "transformed_branch_nmse",
                        )
                    )
                )
                LOGGER.info(
                    "stage progress stage=%s step=%s/%s train_interval_loss=%.6f "
                    "t2_focus_nmse=%.6f",
                    stage,
                    step,
                    total_steps,
                    row["train_batch_loss"],
                    focus,
                )
            recent_totals = {key: 0.0 for key in recent_totals}
            recent_seen = 0
    return history, train_steps


def plot_training(
    branch_history: list[dict[str, Any]],
    rooter_history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if branch_history:
        steps = [row["step"] for row in branch_history]
        axes[0].plot(
            steps,
            [row["train_batch_residual"] for row in branch_history],
            label="T1 residual train",
        )
        axes[0].plot(
            steps,
            [row["train_batch_memory"] for row in branch_history],
            label="T1 memory train",
        )
        axes[0].plot(
            steps,
            [row["t2_residual_branch_nmse"] for row in branch_history],
            label="T2 residual",
        )
        axes[0].plot(
            steps,
            [row["t2_memory_branch_nmse"] for row in branch_history],
            label="T2 memory",
        )
        if "train_batch_transformed" in branch_history[0]:
            axes[0].plot(
                steps,
                [row["train_batch_transformed"] for row in branch_history],
                label="T1 transformed train",
            )
            axes[0].plot(
                steps,
                [row["t2_transformed_branch_nmse"] for row in branch_history],
                label="T2 transformed",
            )
    axes[0].set_title("T1 branch training")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("nMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if rooter_history:
        steps = [row["step"] for row in rooter_history]
        axes[1].plot(
            steps,
            [row["train_batch_prediction"] for row in rooter_history],
            label="T2 rooter train",
        )
        axes[1].plot(
            steps,
            [row["t2_adapted_nmse"] for row in rooter_history],
            label="T2 neural rooter",
        )
        if "t2_ridge_rooter_nmse" in rooter_history[0]:
            axes[1].plot(
                steps,
                [row["t2_ridge_rooter_nmse"] for row in rooter_history],
                label="T2 ridge rooter",
            )
    axes[1].set_title("T2 rooter training")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("nMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None, help="Directory with adapt/eval prediction payloads")
    parser.add_argument("--adapt-payload", default=None)
    parser.add_argument("--eval-payload", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--branch-epochs", type=int, default=10000)
    parser.add_argument("--rooter-epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--valid-eval-freq",
        type=int,
        default=None,
        help="Run deterministic T2 diagnostics every N optimizer steps in each stage.",
    )
    parser.add_argument(
        "--logging-eval-freq",
        type=int,
        default=None,
        help="Log interval-average train and T2 metrics every N optimizer steps.",
    )
    parser.add_argument("--branch-lr", type=float, default=1e-5)
    parser.add_argument("--rooter-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--vanilla-anchor",
        type=float,
        default=1e-2,
        help="Vanilla-anchoring regularizer used in both training stages; 0 disables it.",
    )
    parser.add_argument(
        "--coefficient-l2",
        type=float,
        default=1e-2,
        help="Ridge penalty on neural rooter coefficients; 0 disables it.",
    )
    parser.add_argument(
        "--horizon-smoothness",
        type=float,
        default=1e-2,
        help="First-order horizon penalty on rooter coefficients; 0 disables it.",
    )
    parser.add_argument("--ridge-rooter-alpha", type=float, default=1e-2)
    parser.add_argument("--normalization", default="instance", choices=["instance", "none"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction of pooled T1+T2 query dates assigned to T2/rooter fitting",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--residual-heads", type=int, default=4)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--rooter-heads", type=int, default=None)
    parser.add_argument("--residual-attn-dim", type=int, default=32)
    parser.add_argument("--memory-attn-dim", type=int, default=32)
    parser.add_argument("--rooter-attn-dim", type=int, default=None)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--memory-hidden", type=int, default=128)
    parser.add_argument("--rooter-hidden", type=int, default=None)
    parser.add_argument("--transformed-hidden", type=int, default=128)
    parser.add_argument(
        "--precomputed-transformed-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the extracted sign/sqrt frozen expert; disabled by default.",
    )
    parser.add_argument(
        "--learnable-transformed-covariate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add u=MLP(x) to the transformed expert and train its forecast head on T1. "
            "This also enables the transformed candidate."
        ),
    )
    parser.add_argument(
        "--vanilla-anchoring-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize both learned branches and the rooter to return vanilla exactly.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    base = Path(args.input_dir).expanduser() if args.input_dir else None
    adapt_payload = Path(args.adapt_payload).expanduser() if args.adapt_payload else None
    eval_payload = Path(args.eval_payload).expanduser() if args.eval_payload else None
    if adapt_payload is None:
        if base is None:
            raise ValueError("pass --input-dir or --adapt-payload")
        adapt_payload = base / "adapt_prediction_payload.pt"
    if eval_payload is None and base is not None:
        candidate = base / "eval_prediction_payload.pt"
        eval_payload = candidate if candidate.exists() else None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    elif base is not None:
        output_dir = base / "ts_ifa"
    else:
        output_dir = adapt_payload.parent / "ts_ifa"
    return adapt_payload, eval_payload, output_dir


def _resolved_epochs(args: argparse.Namespace) -> tuple[int, int]:
    branch_epochs = int(args.branch_epochs)
    rooter_epochs = int(args.rooter_epochs)
    if branch_epochs <= 0 or rooter_epochs <= 0:
        raise ValueError("branch and rooter epochs must be positive")
    return branch_epochs, rooter_epochs


def _state_dict_cpu(model: TimeSeriesInformedForecastingAdapter) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _branch_state_dict_cpu(
    model: TimeSeriesInformedForecastingAdapter,
) -> dict[str, torch.Tensor]:
    prefixes = (
        "residual_attention.",
        "residual_head.",
        "memory_attention.",
        "memory_skip.",
        "memory_head.",
        "transformed_covariate.",
        "transformed_head.",
    )
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    experiment_start = perf_counter()
    set_seed(args.seed)
    adapt_payload_path, eval_payload_path, output_dir = resolve_paths(args)
    if not adapt_payload_path.is_file():
        raise FileNotFoundError(adapt_payload_path)
    if eval_payload_path is not None and not eval_payload_path.is_file():
        raise FileNotFoundError(eval_payload_path)
    input_directories = {adapt_payload_path.resolve().parent}
    if eval_payload_path is not None:
        input_directories.add(eval_payload_path.resolve().parent)
    if output_dir.resolve() in input_directories:
        raise ValueError("output directory must differ from extraction directories")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_epochs, rooter_epochs = _resolved_epochs(args)
    LOGGER.info(
        "experiment start kind=ts_ifa_train input=%s branch_epochs=%s rooter_epochs=%s "
        "batch_size=%s",
        adapt_payload_path.parent,
        branch_epochs,
        rooter_epochs,
        args.batch_size,
    )

    adapt_payload = torch_load(adapt_payload_path)
    n_adapt_dates = int(adapt_payload["adapt_X_values"].shape[0])
    train_dates, rooter_dates = chronological_date_slices(
        n_adapt_dates,
        args.validation_fraction,
    )
    train_dataset = PredictionPayloadDataset(
        adapt_payload,
        prefix="adapt",
        date_slice=train_dates,
        max_samples=args.max_train_samples,
        use_transformed_prediction=args.precomputed_transformed_expert,
    )
    rooter_dataset = PredictionPayloadDataset(
        adapt_payload,
        prefix="adapt",
        date_slice=rooter_dates,
        max_samples=args.max_valid_samples,
        use_transformed_prediction=args.precomputed_transformed_expert,
    )
    del adapt_payload
    gc.collect()
    eval_dataset = None
    if eval_payload_path is not None and eval_payload_path.exists():
        eval_payload = torch_load(eval_payload_path)
        eval_dataset = PredictionPayloadDataset(
            eval_payload,
            prefix="eval",
            use_transformed_prediction=args.precomputed_transformed_expert,
        )
        del eval_payload
        gc.collect()
    ensure_compatible(train_dataset, rooter_dataset, name="rooter")
    ensure_compatible(train_dataset, eval_dataset, name="evaluation")
    if args.precomputed_transformed_expert:
        missing_transformed = [
            name
            for name, dataset in (
                ("T1", train_dataset),
                ("T2", rooter_dataset),
                ("T3", eval_dataset),
            )
            if dataset is not None and not dataset.has_transformed_prediction
        ]
        if missing_transformed:
            raise ValueError(
                "the precomputed transformed expert was requested but is absent "
                f"from {missing_transformed}; rerun extraction with "
                "--compute-transformed-prediction or disable the expert"
            )
    LOGGER.info(
        "payload load done t1_samples=%s t2_samples=%s t3_samples=%s",
        len(train_dataset),
        len(rooter_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    log_scale_diagnostics("T1", train_dataset)
    log_scale_diagnostics("T2", rooter_dataset)
    log_scale_diagnostics("T3", eval_dataset)

    eval_batch_size = args.eval_batch_size or args.batch_size
    branch_loader = DataLoader(
        RandomPredictionPayloadDataset(train_dataset, virtual_size=args.batch_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    rooter_train_loader = DataLoader(
        RandomPredictionPayloadDataset(rooter_dataset, virtual_size=args.batch_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    rooter_full_loader = DataLoader(
        rooter_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if eval_dataset is not None
        else None
    )

    rooter_heads = args.rooter_heads or 4
    rooter_attn_dim = args.rooter_attn_dim or 32
    rooter_hidden = args.rooter_hidden or 128
    config = TSIFAConfig(
        lags=train_dataset.lags,
        horizon=train_dataset.horizon,
        neighbors=train_dataset.neighbors,
        residual_heads=args.residual_heads,
        memory_heads=args.memory_heads,
        rooter_heads=rooter_heads,
        residual_attn_dim=args.residual_attn_dim,
        memory_attn_dim=args.memory_attn_dim,
        rooter_attn_dim=rooter_attn_dim,
        residual_hidden=args.residual_hidden,
        memory_hidden=args.memory_hidden,
        rooter_hidden=rooter_hidden,
        transformed_hidden=args.transformed_hidden,
        precomputed_transformed_expert=bool(args.precomputed_transformed_expert),
        learnable_transformed_covariate=bool(args.learnable_transformed_covariate),
        vanilla_anchoring_init=bool(args.vanilla_anchoring_init),
        dropout=args.dropout,
    )
    device = resolve_device(args.device)
    model = TimeSeriesInformedForecastingAdapter(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    branch_parameters = sum(parameter.numel() for parameter in model.branch_parameters())
    neural_rooter_parameters = sum(parameter.numel() for parameter in model.rooter_parameters())
    ridge_rooter_parameters = len(model.candidate_names) * config.horizon
    LOGGER.info(
        "model ready name=TS-IFA device=%s parameters_total=%s branches=%s "
        "neural_rooter=%s ridge_rooter=%s",
        device,
        f"{total_parameters:,}",
        f"{branch_parameters:,}",
        f"{neural_rooter_parameters:,}",
        f"{ridge_rooter_parameters:,}",
    )

    eps = 1e-8
    valid_eval_freq = resolve_step_frequency(
        args.valid_eval_freq,
        default=1,
        name="valid_eval_freq",
    )
    logging_eval_freq = resolve_step_frequency(
        args.logging_eval_freq,
        default=valid_eval_freq,
        name="logging_eval_freq",
    )
    if logging_eval_freq % valid_eval_freq != 0:
        raise ValueError("logging_eval_freq must be a multiple of valid_eval_freq")

    training_start = perf_counter()
    model.set_trainable_stage("branches")
    branch_optimizer = torch.optim.AdamW(
        model.branch_parameters(),
        lr=args.branch_lr,
        weight_decay=args.weight_decay,
    )
    branch_history, branch_steps = train_stage(
        model,
        stage="branches",
        loader=branch_loader,
        diagnostic_loader=rooter_full_loader,
        optimizer=branch_optimizer,
        loss_function=lambda outputs, batch, state: branch_loss_components(
            outputs,
            batch,
            state,
            vanilla_anchor=args.vanilla_anchor,
        ),
        epochs=branch_epochs,
        device=device,
        normalization=args.normalization,
        eps=eps,
        grad_clip=args.grad_clip,
        valid_eval_freq=valid_eval_freq,
        logging_eval_freq=logging_eval_freq,
    )

    branches_path = output_dir / "branches.pt"
    torch.save(
        {
            "branch_state_dict": _branch_state_dict_cpu(model),
            "config": asdict(config),
            "stage": "T1_branches",
            "candidate_names": model.candidate_names,
            "parameter_counts": {"branches": branch_parameters},
        },
        branches_path,
    )

    LOGGER.info("ridge rooter fit start split=T2 alpha=%s", args.ridge_rooter_alpha)
    ridge_coefficients = fit_horizon_ridge_rooter(
        model,
        rooter_full_loader,
        device=device,
        normalization=args.normalization,
        eps=eps,
        alpha=args.ridge_rooter_alpha,
    )
    ridge_path = output_dir / "ridge_rooter.pt"
    torch.save(
        {
            "coefficients": ridge_coefficients,
            "candidate_names": model.candidate_names,
            "alpha": args.ridge_rooter_alpha,
            "fit_split": "T2",
            "parameter_count": ridge_rooter_parameters,
        },
        ridge_path,
    )

    model.set_trainable_stage("rooter")
    rooter_optimizer = torch.optim.AdamW(
        model.rooter_parameters(),
        lr=args.rooter_lr,
        weight_decay=args.weight_decay,
    )
    rooter_history, rooter_steps = train_stage(
        model,
        stage="rooter",
        loader=rooter_train_loader,
        diagnostic_loader=rooter_full_loader,
        optimizer=rooter_optimizer,
        loss_function=lambda outputs, batch, state: rooter_loss_components(
            outputs,
            batch,
            state,
            vanilla_anchor=args.vanilla_anchor,
            coefficient_l2=args.coefficient_l2,
            horizon_smoothness=args.horizon_smoothness,
        ),
        epochs=rooter_epochs,
        device=device,
        normalization=args.normalization,
        eps=eps,
        grad_clip=args.grad_clip,
        valid_eval_freq=valid_eval_freq,
        logging_eval_freq=logging_eval_freq,
        ridge_coefficients=ridge_coefficients,
    )
    model.set_trainable_stage("all")
    LOGGER.info("training done seconds=%.2f", perf_counter() - training_start)

    final_eval: dict[str, float] = {}
    prediction_store = PredictionStore(output_dir)
    if eval_loader is not None:
        LOGGER.info("evaluation start split=T3")
        final_eval = evaluate(
            model,
            eval_loader,
            device=device,
            normalization=args.normalization,
            eps=eps,
            ridge_coefficients=ridge_coefficients,
            prediction_store=prediction_store,
        )
        LOGGER.info(
            "evaluation done neural_nmse=%.6f ridge_nmse=%.6f residual_nmse=%.6f "
            "memory_nmse=%.6f transformed_nmse=%s",
            final_eval["adapted_nmse"],
            final_eval["ridge_rooter_nmse"],
            final_eval["residual_branch_nmse"],
            final_eval["memory_branch_nmse"],
            (
                f"{final_eval['transformed_branch_nmse']:.6f}"
                if "transformed_branch_nmse" in final_eval
                else "disabled"
            ),
        )
    prediction_manifest = prediction_store.finalize(
        metadata={
            "family": "ts_ifa",
            "candidate_names": list(model.candidate_names),
            "gate_diagnostics": ["neural_rooter_coefficients"],
        }
    )

    checkpoint_path = output_dir / "ts_ifa.pt"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "eval_metrics.json"
    config_path = output_dir / "config.json"
    plot_path = output_dir / "training_nmse.pdf"
    torch.save(
        {
            "model_state_dict": _state_dict_cpu(model),
            "config": asdict(config),
            "model_name": "TS-IFA",
            "candidate_names": model.candidate_names,
            "parameter_counts": {
                "total": total_parameters,
                "trainable": total_parameters,
                "branches": branch_parameters,
                "neural_rooter": neural_rooter_parameters,
                "ridge_rooter": ridge_rooter_parameters,
                "branches_plus_ridge_rooter": branch_parameters + ridge_rooter_parameters,
            },
            "normalization": args.normalization,
            "adapt_payload": str(adapt_payload_path),
            "eval_payload": str(eval_payload_path) if eval_payload_path else None,
            "branch_epochs": branch_epochs,
            "rooter_epochs": rooter_epochs,
        },
        checkpoint_path,
    )
    save_json(
        history_path,
        {
            "history": [*branch_history, *rooter_history],
            "branch_history": branch_history,
            "rooter_history": rooter_history,
            "train_steps": [*branch_steps, *rooter_steps],
        },
    )
    save_json(metrics_path, final_eval)
    plot_training(branch_history, rooter_history, plot_path)
    save_json(
        config_path,
        {
            "name": "TS-IFA",
            "model": asdict(config),
            "candidate_names": model.candidate_names,
            "parameters": {
                "total": total_parameters,
                "trainable": total_parameters,
                "branches": branch_parameters,
                "neural_rooter": neural_rooter_parameters,
                "ridge_rooter": ridge_rooter_parameters,
                "branches_plus_ridge_rooter": branch_parameters + ridge_rooter_parameters,
            },
            "training": {
                "pipeline": ["T1_branches", "T2_rooters", "T3_evaluation"],
                "branch_optimizer": "AdamW",
                "rooter_optimizer": "AdamW",
                "branch_lr": args.branch_lr,
                "rooter_lr": args.rooter_lr,
                "weight_decay": args.weight_decay,
                "normalization": args.normalization,
                "regularizers": {
                    "vanilla_anchor": args.vanilla_anchor,
                    "coefficient_l2": args.coefficient_l2,
                    "first_order_horizon_smoothness": args.horizon_smoothness,
                },
                "ridge_rooter_alpha": args.ridge_rooter_alpha,
                "branch_train_split": "T1",
                "rooter_train_split": "T2",
                "final_eval_split": "T3",
                "branch_epochs": branch_epochs,
                "rooter_epochs": rooter_epochs,
                "random_epoch_size": args.batch_size,
                "valid_eval_freq": valid_eval_freq,
                "logging_eval_freq": logging_eval_freq,
                "seconds": perf_counter() - training_start,
            },
        },
    )
    result_manifest = output_dir / "result_manifest.json"
    save_json(
        result_manifest,
        {
            "format": "adaptation_ts_ifa_result",
            "files": {
                "checkpoint": checkpoint_path.name,
                "branches": branches_path.name,
                "ridge_rooter": ridge_path.name,
                "history": history_path.name,
                "metrics": metrics_path.name,
                "predictions": prediction_manifest.name,
                "config": config_path.name,
                "plot": plot_path.name,
            },
        },
    )
    LOGGER.info("outputs saved dir=%s", output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - experiment_start)
    log_experiment_separator(LOGGER)
    return {
        "checkpoint": checkpoint_path,
        "branches": branches_path,
        "ridge_rooter": ridge_path,
        "history": history_path,
        "metrics": metrics_path,
        "predictions": prediction_manifest,
        "config": config_path,
        "plot": plot_path,
        "manifest": result_manifest,
    }


if __name__ == "__main__":
    main()
