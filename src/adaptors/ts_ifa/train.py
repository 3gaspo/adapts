"""Train one joint or meta-learned TS-IFA router and score it once on T3."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset

from src.data.load_dataset import set_seed
from src.data.neighbors import neighbor_to_query_scale
from src.experiment_runs import prepare_run_output
from src.experiments.prediction_store import PredictionStore
from src.experiments.runtime import log_experiment_separator, setup_logging
from src.experiments.splits import chronological_date_slices
from src.models.models import resolve_device

from .model import (
    TSIFA_ARCHITECTURE,
    TSIFA_VARIANTS,
    TSIFAConfig,
    TimeSeriesInformedForecastingAdapter,
)


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
    ):
        self.prefix = prefix
        missing = [f"{prefix}_{name}" for name in self.required if f"{prefix}_{name}" not in payload]
        if missing:
            raise KeyError(f"payload is missing required TS-IFA keys {missing}")

        date_slice = slice(None) if date_slice is None else date_slice
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
        self.tensors = {
            "x": flatten_time_user(x),
            "x_c": flatten_time_user(x_c),
            "y": flatten_time_user(
                payload[f"{prefix}_Y_values"][date_slice].float().clone()
            ),
            "y_c": flatten_time_user(y_c),
            "pred": flatten_time_user(pred),
            "pred_cov": flatten_time_user(
                payload[f"{prefix}_preds_context"][date_slice].float().clone()
            ),
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
            "pred_cov": (raw["pred_cov"] - q_mean) / q_std,
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
    zero = batch["y"].new_zeros(())
    losses = {
        name: normalized_square(outputs[f"{name}_prediction"] - batch["y"], scale)
        if f"{name}_prediction" in outputs
        else zero
        for name in ("residual", "memory")
    }
    anchors = [
        normalized_square(outputs[f"{name}_prediction"] - batch["pred"], scale)
        for name in ("residual", "memory")
        if f"{name}_prediction" in outputs
    ]
    anchoring = sum(anchors, zero)
    total = losses["residual"] + losses["memory"]
    result = {
        "loss": total,
        **losses,
        "vanilla_anchoring": anchoring,
    }
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
    regularized_coefficients = (
        coefficients[:, 1:]
        if coefficients.shape[1] == outputs["candidates"].shape[1]
        else coefficients
    )
    ridge = regularized_coefficients.pow(2).mean()
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
    if coefficients.shape[-1] == 1:
        coefficients = coefficients.expand(-1, vanilla.shape[-1])
    candidate_deltas = outputs["candidates"][:, 1:] - vanilla.unsqueeze(1)
    return vanilla + (
        candidate_deltas * coefficients.to(candidate_deltas).unsqueeze(0)
    ).sum(dim=1)


def differentiable_horizon_ridge(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    eps: float,
    scope: str = "horizon",
) -> torch.Tensor:
    """Solve standardized no-intercept ridge while retaining its autograd graph."""
    if alpha <= 0:
        raise ValueError("ridge rooter alpha must be positive")
    if design.ndim != 3 or target.ndim != 2:
        raise ValueError("ridge design and target must have shapes (B,C,H) and (B,H)")
    if design.shape[0] != target.shape[0] or design.shape[-1] != target.shape[-1]:
        raise ValueError("ridge design and target shapes are incompatible")

    solve_design = design.double()
    solve_target = target.double()
    if scope == "shared":
        solve_design = rearrange(solve_design, "batch candidate horizon -> (batch horizon) candidate 1")
        solve_target = rearrange(solve_target, "batch horizon -> (batch horizon) 1")
    elif scope != "horizon":
        raise ValueError(f"unknown ridge routing scope {scope!r}")
    scale = (solve_design.pow(2).mean(dim=0) + float(eps) ** 2).sqrt()
    standardized = solve_design / scale.unsqueeze(0)
    xtx = torch.einsum("bjh,bkh->hjk", standardized, standardized)
    xty = torch.einsum("bjh,bh->hj", standardized, solve_target)
    candidates = int(design.shape[1])
    regularized = xtx + float(alpha) * torch.eye(
        candidates,
        dtype=xtx.dtype,
        device=xtx.device,
    ).unsqueeze(0)
    standardized_coefficients = torch.linalg.solve(
        regularized,
        xty.unsqueeze(-1),
    ).squeeze(-1)
    coefficients = standardized_coefficients / rearrange(
        scale,
        "candidate horizon -> horizon candidate",
    )
    return rearrange(
        coefficients,
        "horizon candidate -> candidate horizon",
    ).to(design)


def ridge_query_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    coefficients: torch.Tensor,
    *,
    vanilla_anchor: float,
) -> dict[str, torch.Tensor]:
    prediction_value = ridge_prediction(outputs, batch["pred"], coefficients)
    prediction = normalized_square(prediction_value - batch["y"], state["loss_scale"])
    anchoring = normalized_square(
        prediction_value - batch["pred"],
        state["loss_scale"],
    )
    return {
        "loss": prediction + float(vanilla_anchor) * anchoring,
        "prediction": prediction,
        "vanilla_anchoring": anchoring,
    }


def fit_horizon_ridge_rooter(
    model: TimeSeriesInformedForecastingAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: str,
    eps: float,
    alpha: float,
    scope: str,
) -> torch.Tensor:
    """Fit a no-intercept, horizon-wise ridge rooter from exact split statistics."""
    if alpha <= 0:
        raise ValueError("ridge rooter alpha must be positive")
    model.eval()
    horizon = 1 if scope == "shared" else model.config.horizon
    candidates = len(model.rooter_candidate_names)
    xtx = torch.zeros(horizon, candidates, candidates, dtype=torch.float64)
    xty = torch.zeros(horizon, candidates, dtype=torch.float64)
    n_samples = 0
    with torch.inference_mode():
        for raw_cpu in loader:
            raw = move_batch(raw_cpu, device)
            batch, _ = prepare_batch(raw, normalization=normalization, eps=eps)
            outputs = model.forward_branches(batch)
            design = (
                outputs["candidates"][:, 1:] - batch["pred"].unsqueeze(1)
            ).detach().cpu().double()
            target = (batch["y"] - batch["pred"]).detach().cpu().double()
            if scope == "shared":
                design = rearrange(design, "batch candidate step -> (batch step) candidate 1")
                target = rearrange(target, "batch step -> (batch step) 1")
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
    prediction_store: PredictionStore | None = None,
) -> dict[str, float]:
    model.eval()
    variants = (
        "adapted",
        *(f"{name}_branch" for name in model.candidate_names),
    )
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
            "rooter_coefficients",
            shape=(
                len(loader.dataset),
                len(model.rooter_candidate_names),
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
    if stage == "branches":
        inactive = model.rooter_modules()
    elif stage == "rooter":
        inactive = model.branch_modules()
    elif stage == "all":
        inactive = ()
    else:
        raise ValueError(f"unknown training stage {stage!r}")
    for module in inactive:
        module.eval()


def named_rooter_parameters(
    model: TimeSeriesInformedForecastingAdapter,
) -> dict[str, torch.Tensor]:
    selected = {id(parameter) for parameter in model.rooter_parameters()}
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if id(parameter) in selected
    }


def adapt_gradient_rooter(
    model: TimeSeriesInformedForecastingAdapter,
    support_batch: dict[str, torch.Tensor],
    support_state: dict[str, torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
    first_order: bool,
    vanilla_anchor: float,
    coefficient_l2: float,
    horizon_smoothness: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Take differentiable support updates from a neural or softmax-ridge router."""
    fast_parameters = named_rooter_parameters(model)
    support_losses: dict[str, torch.Tensor] = {}
    for _ in range(int(steps)):
        outputs = functional_call(model, fast_parameters, (support_batch,), strict=False)
        support_losses = rooter_loss_components(
            outputs,
            support_batch,
            support_state,
            vanilla_anchor=vanilla_anchor,
            coefficient_l2=coefficient_l2,
            horizon_smoothness=horizon_smoothness,
        )
        gradients = torch.autograd.grad(
            support_losses["loss"],
            tuple(fast_parameters.values()),
            create_graph=not first_order,
        )
        fast_parameters = {
            name: parameter - float(learning_rate) * gradient
            for (name, parameter), gradient in zip(fast_parameters.items(), gradients)
        }
    return fast_parameters, support_losses


def train_bilevel_stage(
    model: TimeSeriesInformedForecastingAdapter,
    *,
    support_loader: DataLoader,
    query_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    normalization: str,
    eps: float,
    grad_clip: float,
    valid_eval_freq: int,
    logging_eval_freq: int,
    variant: str,
    neural_inner_steps: int,
    neural_inner_lr: float,
    neural_first_order: bool,
    ridge_alpha: float,
    branch_aux_weight: float,
    vanilla_anchor: float,
    coefficient_l2: float,
    horizon_smoothness: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Optimize branches for adaptation after a support-set rooter update."""
    if variant not in {"meta_ridge", "meta_neural"}:
        raise ValueError(f"bilevel training requires a meta variant, found {variant!r}")
    history: list[dict[str, Any]] = []
    train_steps: list[dict[str, Any]] = []
    recent_totals: dict[str, float] = {}
    recent_seen = 0
    step = 0
    steps_per_epoch = min(len(support_loader), len(query_loader))
    total_steps = int(epochs) * max(1, steps_per_epoch)
    LOGGER.info(
        "stage start stage=bilevel_meta epochs=%s total_steps=%s "
        "neural_inner_steps=%s neural_first_order=%s",
        epochs,
        total_steps,
        neural_inner_steps,
        neural_first_order,
    )
    for epoch in range(1, int(epochs) + 1):
        _set_stage_mode(model, "all")
        for support_cpu, query_cpu in zip(support_loader, query_loader):
            step += 1
            support_raw = move_batch(support_cpu, device)
            query_raw = move_batch(query_cpu, device)
            support_batch, support_state = prepare_batch(
                support_raw,
                normalization=normalization,
                eps=eps,
            )
            query_batch, query_state = prepare_batch(
                query_raw,
                normalization=normalization,
                eps=eps,
            )

            optimizer.zero_grad(set_to_none=True)
            if variant == "meta_ridge" and model.config.routing_constraint == "unconstrained":
                support_outputs = model.forward_branches(support_batch)
                support_design = (
                    support_outputs["candidates"][:, 1:]
                    - support_batch["pred"].unsqueeze(1)
                )
                ridge_coefficients = differentiable_horizon_ridge(
                    support_design,
                    support_batch["y"] - support_batch["pred"],
                    alpha=ridge_alpha,
                    eps=eps,
                    scope=model.config.routing_scope,
                )
                query_outputs = model.forward_branches(query_batch)
                rooter_query = ridge_query_loss_components(
                    query_outputs,
                    query_batch,
                    query_state,
                    ridge_coefficients,
                    vanilla_anchor=vanilla_anchor,
                )
                support_prediction = normalized_square(
                    ridge_prediction(support_outputs, support_batch["pred"], ridge_coefficients)
                    - support_batch["y"],
                    support_state["loss_scale"],
                )
            else:
                fast_rooter, neural_support = adapt_gradient_rooter(
                    model,
                    support_batch,
                    support_state,
                    steps=neural_inner_steps,
                    learning_rate=neural_inner_lr,
                    first_order=neural_first_order,
                    vanilla_anchor=vanilla_anchor,
                    coefficient_l2=coefficient_l2,
                    horizon_smoothness=horizon_smoothness,
                )
                query_outputs = functional_call(
                    model,
                    fast_rooter,
                    (query_batch,),
                    strict=False,
                )
                rooter_query = rooter_loss_components(
                    query_outputs,
                    query_batch,
                    query_state,
                    vanilla_anchor=vanilla_anchor,
                    coefficient_l2=coefficient_l2,
                    horizon_smoothness=horizon_smoothness,
                )
                support_prediction = neural_support["prediction"]
            branch_aux = branch_loss_components(
                query_outputs,
                query_batch,
                query_state,
                vanilla_anchor=vanilla_anchor,
            )
            outer_loss = (
                rooter_query["loss"] + float(branch_aux_weight) * branch_aux["loss"]
            )
            outer_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses = {
                "loss": outer_loss,
                "rooter_query": rooter_query["prediction"],
                "rooter_support": support_prediction,
                "branch_aux": branch_aux["loss"],
                "residual_aux": branch_aux["residual"],
                "memory_aux": branch_aux["memory"],
            }
            batch_size = int(query_raw["x"].shape[0])
            loss_values = {key: float(value.detach().item()) for key, value in losses.items()}
            train_steps.append(
                {
                    "stage": "bilevel_meta",
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
            should_record = step % valid_eval_freq == 0 or final_step
            should_log = step % logging_eval_freq == 0 or final_step
            if not should_record:
                continue
            row = {
                "stage": "bilevel_meta",
                "epoch": epoch,
                "step": step,
                **{
                    f"t1_query_{key}": value / max(recent_seen, 1)
                    for key, value in recent_totals.items()
                },
            }
            history.append(row)
            if should_log:
                LOGGER.info(
                    "stage progress stage=bilevel_meta step=%s/%s "
                    "variant=%s query_rooter_nmse=%.6f branch_aux=%.6f",
                    step,
                    total_steps,
                    variant,
                    row["t1_query_rooter_query"],
                    row["t1_query_branch_aux"],
                )
            recent_totals = {key: 0.0 for key in recent_totals}
            recent_seen = 0
    return history, train_steps


def joint_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    branch_aux_weight: float,
    vanilla_anchor: float,
    coefficient_l2: float,
    horizon_smoothness: float,
) -> dict[str, torch.Tensor]:
    rooter = rooter_loss_components(
        outputs,
        batch,
        state,
        vanilla_anchor=vanilla_anchor,
        coefficient_l2=coefficient_l2,
        horizon_smoothness=horizon_smoothness,
    )
    branches = branch_loss_components(
        outputs,
        batch,
        state,
        vanilla_anchor=vanilla_anchor,
    )
    return {
        "loss": rooter["loss"] + float(branch_aux_weight) * branches["loss"],
        "prediction": rooter["prediction"],
        "rooter_regularization": rooter["loss"] - rooter["prediction"],
        "branch_aux": branches["loss"],
    }


def train_joint_stage(
    model: TimeSeriesInformedForecastingAdapter,
    *,
    loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    normalization: str,
    eps: float,
    grad_clip: float,
    valid_eval_freq: int,
    logging_eval_freq: int,
    branch_aux_weight: float,
    vanilla_anchor: float,
    coefficient_l2: float,
    horizon_smoothness: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Jointly update branches and the active rooter on T1; select on T2."""
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    recent: dict[str, float] = {}
    recent_seen = 0
    step = 0
    total_steps = int(epochs) * max(1, len(loader))
    best_nmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(epochs) + 1):
        _set_stage_mode(model, "all")
        for raw_cpu in loader:
            step += 1
            raw = move_batch(raw_cpu, device)
            batch, state = prepare_batch(raw, normalization=normalization, eps=eps)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = joint_loss_components(
                outputs,
                batch,
                state,
                branch_aux_weight=branch_aux_weight,
                vanilla_anchor=vanilla_anchor,
                coefficient_l2=coefficient_l2,
                horizon_smoothness=horizon_smoothness,
            )
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            values = {key: float(value.detach()) for key, value in losses.items()}
            batch_size = int(raw["x"].shape[0])
            steps.append({"stage": "joint", "epoch": epoch, "step": step, **values})
            if not recent:
                recent = {key: 0.0 for key in values}
            recent_seen += batch_size
            for key, value in values.items():
                recent[key] += value * batch_size

            final_step = step == total_steps
            if step % valid_eval_freq != 0 and not final_step:
                continue
            diagnostic = evaluate(
                model,
                validation_loader,
                device=device,
                normalization=normalization,
                eps=eps,
            )
            row = {
                "stage": "joint",
                "epoch": epoch,
                "step": step,
                **{f"train_{key}": value / max(recent_seen, 1) for key, value in recent.items()},
                **{f"t2_{key}": value for key, value in diagnostic.items()},
            }
            history.append(row)
            if diagnostic["adapted_nmse"] < best_nmse:
                best_nmse = diagnostic["adapted_nmse"]
                best_state = _state_dict_cpu(model)
                row["selected"] = True
            if step % logging_eval_freq == 0 or final_step:
                LOGGER.info(
                    "stage progress stage=joint step=%s/%s train_nmse=%.6f t2_nmse=%.6f",
                    step,
                    total_steps,
                    row["train_prediction"],
                    diagnostic["adapted_nmse"],
                )
            recent = {key: 0.0 for key in recent}
            recent_seen = 0
    if best_state is None:
        raise RuntimeError("joint training produced no T2 checkpoint")
    model.load_state_dict(best_state)
    return history, steps


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
            )
            row.update({f"t2_{key}": value for key, value in diagnostic.items()})
            history.append(row)
            if should_log:
                focus = (
                    diagnostic["adapted_nmse"]
                    if stage == "rooter"
                    else sum(
                        diagnostic[key]
                        for key in ("residual_branch_nmse", "memory_branch_nmse")
                        if key in diagnostic
                    )
                    / sum(
                        key in diagnostic
                        for key in ("residual_branch_nmse", "memory_branch_nmse")
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
    train_history: list[dict[str, Any]],
    rooter_history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if train_history:
        steps = [row["step"] for row in train_history]
        if train_history[0]["stage"] == "joint":
            axes[0].plot(steps, [row["train_prediction"] for row in train_history], label="T1")
            axes[0].plot(steps, [row["t2_adapted_nmse"] for row in train_history], label="T2")
        else:
            axes[0].plot(
                steps,
                [row["t1_query_rooter_query"] for row in train_history],
                label="T1 query rooter",
            )
            axes[0].plot(
                steps,
                [row["t1_query_branch_aux"] for row in train_history],
                label="T1 branch auxiliary",
            )
    axes[0].set_title("T1 optimization")
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
    axes[1].set_title("T2 neural fit (meta only)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("nMSE")
    axes[1].grid(True, alpha=0.3)
    if rooter_history:
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
    parser.add_argument("--method-id", default=None)
    parser.add_argument(
        "--run-signature",
        default="direct_cli",
        help="Exact launcher configuration signature recorded for completion checks.",
    )
    parser.add_argument("--variant", choices=TSIFA_VARIANTS, default="joint_ridge")
    parser.add_argument("--train-epochs", type=int, default=20000)
    parser.add_argument("--rooter-epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--valid-eval-freq",
        type=int,
        default=None,
        help="Record T1-query or T2 diagnostics every N optimizer steps.",
    )
    parser.add_argument(
        "--logging-eval-freq",
        type=int,
        default=None,
        help="Log interval-average meta-query and T2 metrics every N optimizer steps.",
    )
    parser.add_argument("--branch-lr", type=float, default=1e-5)
    parser.add_argument("--rooter-lr", type=float, default=1e-5)
    parser.add_argument("--neural-inner-lr", type=float, default=1e-3)
    parser.add_argument("--neural-inner-steps", type=int, default=1)
    parser.add_argument(
        "--neural-first-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the first-order MAML approximation for neural inner updates.",
    )
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
    parser.add_argument("--branch-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--branches",
        default="cov,residual,memory",
        help="Comma-separated non-vanilla branches selected from cov,residual,memory.",
    )
    parser.add_argument("--routing-scope", choices=("shared", "horizon"), default="horizon")
    parser.add_argument(
        "--routing-constraint",
        choices=("unconstrained", "softmax"),
        default="unconstrained",
    )
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
    parser.add_argument(
        "--meta-query-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction of T1 query dates assigned to the outer meta-query subset.",
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
    parser.add_argument(
        "--vanilla-anchoring-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize learned corrections at vanilla and route from vanilla "
            "exactly (unconstrained) or nearly exactly (softmax)."
        ),
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
    train_epochs = int(args.train_epochs)
    rooter_epochs = int(args.rooter_epochs)
    if train_epochs <= 0 or rooter_epochs <= 0:
        raise ValueError("train and rooter epochs must be positive")
    uses_gradient_inner = args.variant == "meta_neural" or (
        args.variant == "meta_ridge" and args.routing_constraint == "softmax"
    )
    if uses_gradient_inner and int(args.neural_inner_steps) <= 0:
        raise ValueError("neural inner steps must be positive")
    if uses_gradient_inner and float(args.neural_inner_lr) <= 0:
        raise ValueError("neural inner learning rate must be positive")
    if args.variant == "meta_ridge" and float(args.ridge_rooter_alpha) <= 0:
        raise ValueError("ridge rooter alpha must be positive for closed-form solves")
    if float(args.branch_aux_weight) < 0:
        raise ValueError("branch_aux_weight cannot be negative")
    return train_epochs, rooter_epochs


def _resolved_branches(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in str(value).replace("+", ",").split(",") if item.strip()]
    if "full" in requested:
        if len(requested) != 1:
            raise ValueError("full cannot be combined with explicit TS-IFA branch names")
        requested = ["cov", "residual", "memory"]
    allowed = ("cov", "residual", "memory")
    unknown = set(requested) - set(allowed)
    if not requested or unknown:
        raise ValueError(f"branches must be a non-empty subset of {allowed}; got {requested}")
    if len(set(requested)) != len(requested):
        raise ValueError("branches cannot contain duplicates")
    return tuple(name for name in allowed if name in requested)


def _state_dict_cpu(model: TimeSeriesInformedForecastingAdapter) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _branch_state_dict_cpu(
    model: TimeSeriesInformedForecastingAdapter,
) -> dict[str, torch.Tensor]:
    prefixes = (
        "residual_attention.",
        "residual_head.",
        "memory_attention.",
        "memory_head.",
    )
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def _rooter_state_dict_cpu(
    model: TimeSeriesInformedForecastingAdapter,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in named_rooter_parameters(model).items()
    }


def nested_date_slice(parent: slice, child: slice) -> slice:
    start = int(parent.start or 0)
    return slice(start + int(child.start or 0), start + int(child.stop))


def split_sample_limit(
    maximum: int | None,
    *,
    query_fraction: float,
) -> tuple[int | None, int | None]:
    if maximum is None:
        return None, None
    maximum = int(maximum)
    if maximum < 2:
        raise ValueError("max train samples must be at least two for support/query meta-training")
    query = min(maximum - 1, max(1, int(round(maximum * float(query_fraction)))))
    return maximum - query, query


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
    prepare_run_output(output_dir)
    train_epochs, rooter_epochs = _resolved_epochs(args)
    branches = _resolved_branches(args.branches)
    variant = str(args.variant)
    optimization, rooter_form = variant.split("_", maxsplit=1)
    branch_label = "full" if branches == ("cov", "residual", "memory") else "+".join(branches)
    method_id = args.method_id or "_".join(
        (variant, args.routing_scope, args.routing_constraint, branch_label)
    )
    LOGGER.info(
        "experiment start kind=ts_ifa method=%s variant=%s train_epochs=%s rooter_epochs=%s",
        method_id,
        variant,
        train_epochs,
        rooter_epochs,
    )

    adapt_payload = torch_load(adapt_payload_path)
    train_dates, t2_dates = chronological_date_slices(
        int(adapt_payload["adapt_X_values"].shape[0]), args.validation_fraction
    )
    t2_dataset = PredictionPayloadDataset(
        adapt_payload, prefix="adapt", date_slice=t2_dates, max_samples=args.max_valid_samples
    )
    train_dataset: PredictionPayloadDataset | None = None
    support_dataset: PredictionPayloadDataset | None = None
    query_dataset: PredictionPayloadDataset | None = None
    if optimization == "joint":
        train_dataset = PredictionPayloadDataset(
            adapt_payload, prefix="adapt", date_slice=train_dates, max_samples=args.max_train_samples
        )
        reference_dataset = train_dataset
    else:
        n_train_dates = int(train_dates.stop) - int(train_dates.start or 0)
        support_local, query_local = chronological_date_slices(n_train_dates, args.meta_query_fraction)
        support_max, query_max = split_sample_limit(
            args.max_train_samples, query_fraction=args.meta_query_fraction
        )
        support_dataset = PredictionPayloadDataset(
            adapt_payload,
            prefix="adapt",
            date_slice=nested_date_slice(train_dates, support_local),
            max_samples=support_max,
        )
        query_dataset = PredictionPayloadDataset(
            adapt_payload,
            prefix="adapt",
            date_slice=nested_date_slice(train_dates, query_local),
            max_samples=query_max,
        )
        reference_dataset = support_dataset
        ensure_compatible(reference_dataset, query_dataset, name="meta query")
    del adapt_payload
    gc.collect()

    eval_dataset = None
    if eval_payload_path is not None:
        eval_payload = torch_load(eval_payload_path)
        eval_dataset = PredictionPayloadDataset(eval_payload, prefix="eval")
        del eval_payload
        gc.collect()
    ensure_compatible(reference_dataset, t2_dataset, name="T2")
    ensure_compatible(reference_dataset, eval_dataset, name="T3")
    log_scale_diagnostics("T1", train_dataset)
    log_scale_diagnostics("T1 support", support_dataset)
    log_scale_diagnostics("T1 query", query_dataset)
    log_scale_diagnostics("T2", t2_dataset)
    log_scale_diagnostics("T3", eval_dataset)

    eval_batch_size = args.eval_batch_size or args.batch_size
    random_loader = lambda dataset: DataLoader(
        RandomPredictionPayloadDataset(dataset, virtual_size=args.batch_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    full_loader = lambda dataset: DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    t2_full_loader = full_loader(t2_dataset)
    eval_loader = full_loader(eval_dataset) if eval_dataset is not None else None

    config = TSIFAConfig(
        lags=reference_dataset.lags,
        horizon=reference_dataset.horizon,
        neighbors=reference_dataset.neighbors,
        rooter_form=rooter_form,
        branches=branches,
        routing_scope=args.routing_scope,
        routing_constraint=args.routing_constraint,
        residual_heads=args.residual_heads,
        memory_heads=args.memory_heads,
        rooter_heads=args.rooter_heads or 4,
        residual_attn_dim=args.residual_attn_dim,
        memory_attn_dim=args.memory_attn_dim,
        rooter_attn_dim=args.rooter_attn_dim or 32,
        residual_hidden=args.residual_hidden,
        memory_hidden=args.memory_hidden,
        rooter_hidden=args.rooter_hidden or 128,
        vanilla_anchoring_init=bool(args.vanilla_anchoring_init),
        dropout=args.dropout,
    )
    device = resolve_device(args.device)
    model = TimeSeriesInformedForecastingAdapter(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    branch_parameter_count = sum(parameter.numel() for parameter in model.branch_parameters())
    rooter_parameter_count = sum(parameter.numel() for parameter in model.rooter_parameters())
    valid_eval_freq = resolve_step_frequency(args.valid_eval_freq, default=1, name="valid_eval_freq")
    logging_eval_freq = resolve_step_frequency(
        args.logging_eval_freq, default=valid_eval_freq, name="logging_eval_freq"
    )
    if logging_eval_freq % valid_eval_freq:
        raise ValueError("logging_eval_freq must be a multiple of valid_eval_freq")

    eps = 1e-8
    training_start = perf_counter()
    rooter_history: list[dict[str, Any]] = []
    rooter_steps: list[dict[str, Any]] = []
    if optimization == "joint":
        model.set_trainable_stage("all")
        parameter_groups = []
        if model.branch_parameters():
            parameter_groups.append({"params": model.branch_parameters(), "lr": args.branch_lr})
        parameter_groups.append({"params": model.rooter_parameters(), "lr": args.rooter_lr})
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
        train_history, train_steps = train_joint_stage(
            model,
            loader=random_loader(train_dataset),
            validation_loader=t2_full_loader,
            optimizer=optimizer,
            epochs=train_epochs,
            device=device,
            normalization=args.normalization,
            eps=eps,
            grad_clip=args.grad_clip,
            valid_eval_freq=valid_eval_freq,
            logging_eval_freq=logging_eval_freq,
            branch_aux_weight=args.branch_aux_weight,
            vanilla_anchor=args.vanilla_anchor,
            coefficient_l2=args.coefficient_l2,
            horizon_smoothness=args.horizon_smoothness,
        )
        t1_description = "T1_joint_gradient_updates_with_T2_checkpoint_selection"
        t1_dates_count, t1_samples_count = train_dataset.n_dates, len(train_dataset)
    else:
        model.set_trainable_stage("all")
        parameter_groups = []
        if model.branch_parameters():
            parameter_groups.append({"params": model.branch_parameters(), "lr": args.branch_lr})
        if rooter_form == "neural" or args.routing_constraint == "softmax":
            parameter_groups.append({"params": model.rooter_parameters(), "lr": args.rooter_lr})
        if parameter_groups:
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
            train_history, train_steps = train_bilevel_stage(
                model,
                support_loader=random_loader(support_dataset),
                query_loader=random_loader(query_dataset),
                optimizer=optimizer,
                epochs=train_epochs,
                device=device,
                normalization=args.normalization,
                eps=eps,
                grad_clip=args.grad_clip,
                valid_eval_freq=valid_eval_freq,
                logging_eval_freq=logging_eval_freq,
                variant=variant,
                neural_inner_steps=args.neural_inner_steps,
                neural_inner_lr=args.neural_inner_lr,
                neural_first_order=bool(args.neural_first_order),
                ridge_alpha=args.ridge_rooter_alpha,
                branch_aux_weight=args.branch_aux_weight,
                vanilla_anchor=args.vanilla_anchor,
                coefficient_l2=args.coefficient_l2,
                horizon_smoothness=args.horizon_smoothness,
            )
        else:
            LOGGER.info(
                "stage skip stage=bilevel_meta reason=no_trainable_branch_parameters"
            )
            train_history, train_steps = [], []
        if rooter_form == "ridge" and args.routing_constraint == "unconstrained":
            coefficients = fit_horizon_ridge_rooter(
                model,
                t2_full_loader,
                device=device,
                normalization=args.normalization,
                eps=eps,
                alpha=args.ridge_rooter_alpha,
                scope=args.routing_scope,
            )
            with torch.no_grad():
                model.ridge_coefficients.copy_(coefficients.to(model.ridge_coefficients))
        else:
            model.set_trainable_stage("rooter")
            rooter_optimizer = torch.optim.AdamW(
                model.rooter_parameters(), lr=args.rooter_lr, weight_decay=args.weight_decay
            )
            rooter_history, rooter_steps = train_stage(
                model,
                stage="rooter",
                loader=random_loader(t2_dataset),
                diagnostic_loader=t2_full_loader,
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
            )
        model.set_trainable_stage("all")
        t1_description = "T1_chronological_support_query_meta_updates_then_T2_rooter_fit"
        t1_dates_count = support_dataset.n_dates + query_dataset.n_dates
        t1_samples_count = len(support_dataset) + len(query_dataset)

    branches_path = output_dir / "branches.pt"
    rooter_path = output_dir / "rooter.pt"
    torch.save(
        {
            "branch_state_dict": _branch_state_dict_cpu(model),
            "variant": variant,
            "architecture": TSIFA_ARCHITECTURE,
            "run_signature": args.run_signature,
            "config": asdict(config),
        },
        branches_path,
    )
    rooter_payload: dict[str, Any] = {
        "rooter_state_dict": _rooter_state_dict_cpu(model),
        "variant": variant,
        "method": method_id,
        "rooter_form": rooter_form,
        "optimization": optimization,
        "routing_scope": args.routing_scope,
        "routing_constraint": args.routing_constraint,
        "architecture": TSIFA_ARCHITECTURE,
        "run_signature": args.run_signature,
        "candidate_names": model.rooter_candidate_names,
        "parameter_count": rooter_parameter_count,
    }
    if rooter_form == "ridge":
        routing_values = model.ridge_coefficients.detach().cpu()
        if args.routing_scope == "shared":
            routing_values = routing_values.expand(-1, model.config.horizon)
        if args.routing_constraint == "softmax":
            vanilla_logits = torch.zeros(1, model.config.horizon)
            rooter_payload["coefficients"] = torch.softmax(
                torch.cat((vanilla_logits, routing_values), dim=0), dim=0
            )
            rooter_payload["routing_values"] = routing_values
        else:
            rooter_payload["coefficients"] = routing_values
        rooter_payload["fit"] = (
            "closed_form_on_T2"
            if optimization == "meta" and args.routing_constraint == "unconstrained"
            else "gradient_updates"
        )
        rooter_payload["alpha"] = args.ridge_rooter_alpha
    torch.save(rooter_payload, rooter_path)

    final_eval: dict[str, float] = {}
    prediction_store = PredictionStore(output_dir)
    if eval_loader is not None:
        final_eval = evaluate(
            model,
            eval_loader,
            device=device,
            normalization=args.normalization,
            eps=eps,
            prediction_store=prediction_store,
        )
    prediction_manifest = prediction_store.finalize(
        metadata={
            "family": "ts_ifa",
            "variant": variant,
            "method": method_id,
            "routing_scope": args.routing_scope,
            "routing_constraint": args.routing_constraint,
            "candidate_names": list(model.candidate_names),
            "rooter_candidate_names": list(model.rooter_candidate_names),
            "gate_diagnostics": ["rooter_coefficients"],
        }
    )

    checkpoint_path = output_dir / "ts_ifa.pt"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "eval_metrics.json"
    config_path = output_dir / "config.json"
    plot_path = output_dir / "training_nmse.pdf"
    common = {
        "name": "TS-IFA",
        "variant": variant,
        "method": method_id,
        "rooter_form": rooter_form,
        "optimization": optimization,
        "routing_scope": args.routing_scope,
        "routing_constraint": args.routing_constraint,
        "architecture": TSIFA_ARCHITECTURE,
        "run_signature": args.run_signature,
        "model": asdict(config),
        "candidate_names": model.candidate_names,
        "rooter_candidate_names": model.rooter_candidate_names,
        "parameters": {
            "total": total_parameters,
            "branches": branch_parameter_count,
            "rooter": rooter_parameter_count,
        },
    }
    torch.save(
        {
            **common,
            "model_state_dict": _state_dict_cpu(model),
            "adapt_payload": str(adapt_payload_path),
            "eval_payload": str(eval_payload_path) if eval_payload_path else None,
        },
        checkpoint_path,
    )
    save_json(
        history_path,
        {
            "history": [*train_history, *rooter_history],
            "train_history": train_history,
            "rooter_history": rooter_history,
            "train_steps": [*train_steps, *rooter_steps],
        },
    )
    save_json(metrics_path, final_eval)
    plot_training(train_history, rooter_history, plot_path)
    training = {
        "pipeline": t1_description,
        "train_split": "T1",
        "validation_or_fit_split": "T2",
        "final_eval_split": "T3",
        "train_epochs": train_epochs,
        "rooter_epochs": rooter_epochs
        if optimization == "meta" and (rooter_form == "neural" or args.routing_constraint == "softmax")
        else 0,
        "t1_dates": t1_dates_count,
        "t1_samples": t1_samples_count,
        "t2_dates": t2_dataset.n_dates,
        "t2_samples": len(t2_dataset),
        "meta_query_fraction": args.meta_query_fraction if optimization == "meta" else None,
        "branch_lr": args.branch_lr,
        "rooter_lr": args.rooter_lr,
        "neural_inner_lr": args.neural_inner_lr
        if optimization == "meta" and (rooter_form == "neural" or args.routing_constraint == "softmax")
        else None,
        "neural_inner_steps": args.neural_inner_steps
        if optimization == "meta" and (rooter_form == "neural" or args.routing_constraint == "softmax")
        else None,
        "neural_first_order": bool(args.neural_first_order)
        if optimization == "meta" and (rooter_form == "neural" or args.routing_constraint == "softmax")
        else None,
        "regularizers": {
            "vanilla_anchor": args.vanilla_anchor,
            "coefficient_l2": args.coefficient_l2,
            "horizon_smoothness": args.horizon_smoothness,
            "closed_form_ridge_alpha": args.ridge_rooter_alpha
            if variant == "meta_ridge" and args.routing_constraint == "unconstrained"
            else None,
            "branch_auxiliary_weight": args.branch_aux_weight,
        },
        "seconds": perf_counter() - training_start,
    }
    save_json(config_path, {**common, "training": training})
    result_manifest = output_dir / "result_manifest.json"
    save_json(
        result_manifest,
        {
            "format": "adaptation_ts_ifa_result",
            "variant": variant,
            "method": method_id,
            "rooter_form": rooter_form,
            "optimization": optimization,
            "routing_scope": args.routing_scope,
            "routing_constraint": args.routing_constraint,
            "candidate_names": list(model.candidate_names),
            "architecture": TSIFA_ARCHITECTURE,
            "run_signature": args.run_signature,
            "files": {
                "checkpoint": checkpoint_path.name,
                "branches": branches_path.name,
                "rooter": rooter_path.name,
                "history": history_path.name,
                "metrics": metrics_path.name,
                "predictions": prediction_manifest.name,
                "config": config_path.name,
                "plot": plot_path.name,
            },
        },
    )
    LOGGER.info("outputs saved variant=%s dir=%s", variant, output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - experiment_start)
    log_experiment_separator(LOGGER)
    return {
        "checkpoint": checkpoint_path,
        "branches": branches_path,
        "rooter": rooter_path,
        "history": history_path,
        "metrics": metrics_path,
        "predictions": prediction_manifest,
        "config": config_path,
        "plot": plot_path,
        "manifest": result_manifest,
    }


if __name__ == "__main__":
    main()
