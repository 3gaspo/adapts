"""Fit and evaluate anchored ridge baselines and retrieval gates."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from src.data.neighbors import neighbor_to_query_scale
from src.experiments.prediction_store import PredictionStore
from src.experiments.runtime import log_experiment_separator, setup_logging
from src.experiments.splits import chronological_resplit_arrays


LOGGER = logging.getLogger(__name__)
RIDGE_CHUNK_ROWS = 65_536
GATE_FEATURE_CHUNK_ROWS = 2_048
METRIC_CHUNK_ROWS = 16_384
AGGREGATION_CHUNK_ROWS = 16_384
DEFAULT_L2_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _solve_system(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, target, rcond=None)[0]


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def flatten_payload(
    payload: dict[str, Any],
    prefix: str,
    *,
    family: str | None = None,
) -> dict[str, np.ndarray]:
    if family not in {None, "baselines", "gates"}:
        raise ValueError(f"unknown payload family {family!r}")
    x = payload[f"{prefix}_X_values"].float()
    x_c = payload[f"{prefix}_Xc_values"].float()
    y_c_raw = payload[f"{prefix}_Yc_values"].float()
    e_raw = payload[f"{prefix}_E_values"].float()
    y_c = neighbor_to_query_scale(x, x_c, y_c_raw)
    e = (
        neighbor_to_query_scale(x, x_c, e_raw, residual=True)
        if family != "baselines"
        else None
    )
    pred_neighbors = (
        neighbor_to_query_scale(x, x_c, y_c_raw - e_raw)
        if family != "gates"
        else None
    )
    query_t = payload[f"{prefix}_query_t"]
    query_user = payload[f"{prefix}_query_user_idx"]
    neighbor_t = payload[f"{prefix}_neighbor_t"]
    neighbor_user = payload[f"{prefix}_neighbor_user_idx"]
    arrays = {
        "pred": rearrange(
            payload[f"{prefix}_preds"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "pred_c": rearrange(
            payload[f"{prefix}_preds_context"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "y": rearrange(
            payload[f"{prefix}_Y_values"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "x": rearrange(x, "date user lags -> (date user) lags").numpy(),
        "y_c": rearrange(
            y_c,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy(),
        "distance": rearrange(
            payload[f"{prefix}_distance_x_xc"].float(),
            "date user neighbor -> (date user) neighbor",
        ).numpy(),
        "query_t": rearrange(query_t, "date user -> (date user)").numpy(),
        "neighbor_lookback_mean": rearrange(
            x_c.mean(dim=-1).mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_mean_std": rearrange(
            x_c.mean(dim=-1).std(dim=-1, unbiased=False),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_std": rearrange(
            x_c.std(dim=-1, unbiased=False).mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_std_std": rearrange(
            x_c.std(dim=-1, unbiased=False).std(dim=-1, unbiased=False),
            "date user -> (date user)",
        ).numpy(),
        "same_user_ratio": rearrange(
            (neighbor_user == query_user.unsqueeze(-1)).float().mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_age_mean": rearrange(
            (query_t.unsqueeze(-1) - neighbor_t).float().mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
    }
    if e is not None:
        arrays["e"] = rearrange(
            e,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy()
    if pred_neighbors is not None:
        arrays["pred_neighbors"] = rearrange(
            pred_neighbors,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy()
    return arrays


def distance_weights(arrays: dict[str, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    d = arrays["distance"].astype(np.float64)
    d_std = d.std(axis=-1, keepdims=True)
    d_norm = (d - d.min(axis=-1, keepdims=True)) / np.maximum(d_std, eps)
    return softmax_np(-d_norm, axis=-1)


def weighted_neighbor_horizon(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "aggr_y" in arrays:
        return arrays["aggr_y"]
    n_samples, _, horizon = arrays["y_c"].shape
    aggregate = np.empty((n_samples, horizon), dtype=np.float64)
    for start in range(0, n_samples, AGGREGATION_CHUNK_ROWS):
        stop = min(start + AGGREGATION_CHUNK_ROWS, n_samples)
        chunk = {name: value[start:stop] for name, value in arrays.items()}
        weights = distance_weights(chunk)
        aggregate[start:stop] = (
            weights[:, :, None] * arrays["y_c"][start:stop]
        ).sum(axis=1)
    return aggregate


def _ridge_no_intercept_from_chunks(
    chunks,
    *,
    n_samples: int,
    n_features: int,
    target_shape: tuple[int, ...],
    l2: float,
) -> np.ndarray:
    """Exact chunked ridge retained as a public smoke-test helper."""
    if n_samples == 0:
        raise ValueError("cannot fit ridge regression without observations")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    sum_squares = np.zeros(n_features, dtype=np.float64)
    xtx = np.zeros((n_features, n_features), dtype=np.float64)
    xty = np.zeros((n_features, *target_shape), dtype=np.float64)
    seen = 0
    for x_chunk, y_chunk in chunks:
        x_chunk = np.asarray(x_chunk, dtype=np.float64)
        y_chunk = np.asarray(y_chunk, dtype=np.float64)
        sum_squares += np.einsum("ij,ij->j", x_chunk, x_chunk)
        xtx += x_chunk.T @ x_chunk
        xty += x_chunk.T @ y_chunk
        seen += x_chunk.shape[0]
    if seen != n_samples:
        raise ValueError(f"ridge chunks contain {seen} samples, expected {n_samples}")
    scale = np.maximum(np.sqrt(sum_squares / n_samples), 1e-12)
    xtx = xtx / np.outer(scale, scale) / n_samples
    scale_shape = (n_features, *([1] * len(target_shape)))
    xty = xty / scale.reshape(scale_shape) / n_samples
    standardized = _solve_system(
        xtx + float(l2) * np.eye(n_features, dtype=np.float64),
        xty,
    )
    return standardized / scale.reshape(scale_shape)


def ridge_no_intercept(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    *,
    chunk_rows: int = RIDGE_CHUNK_ROWS,
) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("ridge features and targets must align")
    chunks = (
        (x[start : start + chunk_rows], y[start : start + chunk_rows])
        for start in range(0, x.shape[0], chunk_rows)
    )
    return _ridge_no_intercept_from_chunks(
        chunks,
        n_samples=x.shape[0],
        n_features=x.shape[1],
        target_shape=y.shape[1:],
        l2=l2,
    )


def subsample_fit_arrays(
    arrays: dict[str, np.ndarray],
    max_samples: int | None,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select reproducible fitting rows while leaving scoring arrays untouched."""
    n_samples = arrays["y"].shape[0]
    if max_samples is None or n_samples <= max_samples:
        return arrays
    if max_samples <= 0:
        raise ValueError("fit sample maxima must be positive")
    indices = np.random.default_rng(seed).choice(
        n_samples,
        size=max_samples,
        replace=False,
    )
    indices.sort()
    return {name: value[indices] for name, value in arrays.items()}


def _contiguous_fit_partition(
    fit_arrays: dict[str, dict[str, np.ndarray]],
) -> bool:
    train = fit_arrays["T1"]["y"]
    valid = fit_arrays["T2"]["y"]
    refit = fit_arrays["T1+T2"]["y"]
    n_train = train.shape[0]
    if n_train + valid.shape[0] != refit.shape[0]:
        return False
    return (
        int(train.__array_interface__["data"][0])
        == int(refit.__array_interface__["data"][0])
        and int(valid.__array_interface__["data"][0])
        == int(refit[n_train:].__array_interface__["data"][0])
    )


RIDGE_DESIGNS: dict[str, tuple[str, ...]] = {
    "context": ("V", "C"),
    "aggr_y": ("V", "aggr_y"),
    "y": ("V", "Y"),
    "cov_y": ("V", "C", "Y"),
    "cov_horizon": ("V", "C", "aggr_y"),
    "residual": ("V", "Y", "N"),
    "full": ("V", "C", "Y", "N"),
}

RIDGE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("context_ridge_shared", "context", "shared"),
    ("context_ridge_horizon", "context", "horizon"),
    ("aggr_y_ridge_shared", "aggr_y", "shared"),
    ("aggr_y_ridge_horizon", "aggr_y", "horizon"),
    ("y_ridge_shared", "y", "shared"),
    ("y_ridge_horizon", "y", "horizon"),
    ("cov_y_ridge_shared", "cov_y", "shared"),
    ("cov_horizon_ridge_shared", "cov_horizon", "shared"),
    ("cov_horizon_ridge_horizon", "cov_horizon", "horizon"),
    ("residual_ridge_shared", "residual", "shared"),
    ("residual_ridge_horizon", "residual", "horizon"),
    ("full_ridge_shared", "full", "shared"),
    ("full_ridge_horizon", "full", "horizon"),
)

TRAINABLE_BASELINES = (
    "aggr_y_mix_shared",
    "aggr_y_mix_horizon",
    *(name for name, _, _ in RIDGE_MODELS),
)
DIRECT_BASELINES = ("context_forecast", "aggr_y", "y_mean")
BASELINE_METHODS = (*DIRECT_BASELINES, *TRAINABLE_BASELINES)


def _design_chunk(
    arrays: dict[str, np.ndarray],
    design: str,
    start: int,
    stop: int,
) -> np.ndarray:
    pred = arrays["pred"][start:stop]
    parts: list[np.ndarray] = []
    for signal in RIDGE_DESIGNS[design]:
        if signal == "V":
            parts.append(pred[:, :, None])
        elif signal == "C":
            parts.append(arrays["pred_c"][start:stop, :, None])
        elif signal == "aggr_y":
            weights = distance_weights(
                {name: value[start:stop] for name, value in arrays.items()}
            )
            aggregate = (
                weights[:, :, None] * arrays["y_c"][start:stop]
            ).sum(axis=1)
            parts.append(aggregate[:, :, None])
        elif signal == "Y":
            parts.append(np.moveaxis(arrays["y_c"][start:stop], 1, 2))
        elif signal == "N":
            parts.append(
                np.moveaxis(arrays["pred_neighbors"][start:stop], 1, 2)
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown ridge signal {signal!r}")
    return np.concatenate(parts, axis=-1).astype(np.float64, copy=False)


def _ridge_statistics(
    arrays: dict[str, np.ndarray],
    design: str,
    mode: str,
) -> dict[str, Any]:
    n_samples, horizon = arrays["y"].shape
    if n_samples == 0:
        raise ValueError("cannot fit ridge from an empty split")
    neighbors = arrays["y_c"].shape[1]
    feature_count = sum(
        neighbors if signal in {"Y", "N"} else 1
        for signal in RIDGE_DESIGNS[design]
    )
    chunk_samples = max(1, RIDGE_CHUNK_ROWS // max(horizon, 1))
    if mode == "shared":
        sum_squares = np.zeros(feature_count, dtype=np.float64)
        xtx = np.zeros((feature_count, feature_count), dtype=np.float64)
        xty = np.zeros(feature_count, dtype=np.float64)
    elif mode == "horizon":
        sum_squares = np.zeros((horizon, feature_count), dtype=np.float64)
        xtx = np.zeros((horizon, feature_count, feature_count), dtype=np.float64)
        xty = np.zeros((horizon, feature_count), dtype=np.float64)
    else:
        raise ValueError(f"unknown ridge mode {mode!r}")
    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        x = _design_chunk(arrays, design, start, stop)
        target = (
            arrays["y"][start:stop].astype(np.float64)
            - arrays["pred"][start:stop].astype(np.float64)
        )
        if mode == "shared":
            x_flat = x.reshape(-1, feature_count)
            target_flat = target.reshape(-1)
            sum_squares += np.einsum("ij,ij->j", x_flat, x_flat)
            xtx += x_flat.T @ x_flat
            xty += x_flat.T @ target_flat
        else:
            sum_squares += np.einsum("shf,shf->hf", x, x)
            xtx += np.einsum("shf,shg->hfg", x, x)
            xty += np.einsum("shf,sh->hf", x, target)
    return {
        "mode": mode,
        "sum_squares": sum_squares,
        "xtx": xtx,
        "xty": xty,
        "n_samples": n_samples,
        "horizon": horizon,
        "feature_count": feature_count,
    }


def _solve_ridge_statistics(statistics: dict[str, Any], l2: float) -> np.ndarray:
    mode = statistics["mode"]
    n_samples = int(statistics["n_samples"])
    horizon = int(statistics["horizon"])
    feature_count = int(statistics["feature_count"])
    if mode == "shared":
        n_observations = n_samples * horizon
        scale = np.maximum(
            np.sqrt(statistics["sum_squares"] / n_observations),
            1e-12,
        )
        xtx = statistics["xtx"] / np.outer(scale, scale) / n_observations
        xty = statistics["xty"] / scale / n_observations
        standardized = _solve_system(
            xtx + float(l2) * np.eye(feature_count),
            xty,
        )
        return standardized / scale
    coefficients = np.empty((horizon, feature_count), dtype=np.float64)
    for h in range(horizon):
        scale = np.maximum(
            np.sqrt(statistics["sum_squares"][h] / n_samples),
            1e-12,
        )
        xtx = statistics["xtx"][h] / np.outer(scale, scale) / n_samples
        xty = statistics["xty"][h] / scale / n_samples
        standardized = _solve_system(
            xtx + float(l2) * np.eye(feature_count),
            xty,
        )
        coefficients[h] = standardized / scale
    return coefficients


def _predict_anchored_ridge(
    arrays: dict[str, np.ndarray],
    *,
    design: str,
    mode: str,
    coefficients: np.ndarray,
) -> np.ndarray:
    n_samples, horizon = arrays["pred"].shape
    out = np.empty((n_samples, horizon), dtype=np.float64)
    chunk_samples = max(1, RIDGE_CHUNK_ROWS // max(horizon, 1))
    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        x = _design_chunk(arrays, design, start, stop)
        if mode == "shared":
            correction = np.einsum("shf,f->sh", x, coefficients)
        else:
            correction = np.einsum("shf,hf->sh", x, coefficients)
        out[start:stop] = arrays["pred"][start:stop] + correction
    return out


def _normalized_mse(arrays: dict[str, np.ndarray], prediction: np.ndarray) -> float:
    scale = (
        arrays["scale"]
        if "scale" in arrays
        else np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
    )
    return float(np.mean(((prediction - arrays["y"]) / scale) ** 2))


def _fit_convex_lambda(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
) -> np.ndarray | float:
    pred = arrays["pred"].astype(np.float64)
    direction = weighted_neighbor_horizon(arrays).astype(np.float64) - pred
    target = arrays["y"].astype(np.float64) - pred
    if mode == "shared":
        denominator = float(np.sum(direction**2))
        unconstrained = (
            float(np.sum(direction * target)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        return float(np.clip(unconstrained, 0.0, 1.0))
    denominator = np.sum(direction**2, axis=0)
    numerator = np.sum(direction * target, axis=0)
    unconstrained = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return np.clip(unconstrained, 0.0, 1.0)


def _predict_convex_lambda(
    arrays: dict[str, np.ndarray],
    value: np.ndarray | float,
) -> np.ndarray:
    pred = arrays["pred"]
    aggregate = weighted_neighbor_horizon(arrays)
    return (1.0 - value) * pred + value * aggregate


def fit_baseline_adapters(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray] | None = None,
    refit: dict[str, np.ndarray] | None = None,
    l2_grid: Sequence[float] | float = DEFAULT_L2_GRID,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Tune ridge alpha on T2, then refit selected models on pooled T1+T2."""
    valid = train if valid is None else valid
    refit = train if refit is None else refit
    grid = (
        (float(l2_grid),)
        if isinstance(l2_grid, (int, float))
        else tuple(float(value) for value in l2_grid)
    )
    if not grid or any(value < 0 for value in grid):
        raise ValueError("l2_grid must contain non-negative values")
    artifacts: dict[str, Any] = {
        "protocol": "tune_on_T2_then_refit_on_T1_plus_T2",
        "l2_grid": grid,
        "models": {},
    }
    selected = set(TRAINABLE_BASELINES if methods is None else methods)
    for mode in ("shared", "horizon"):
        name = f"aggr_y_mix_{mode}"
        if name not in selected:
            continue
        train_lambda = _fit_convex_lambda(train, mode=mode)
        valid_prediction = _predict_convex_lambda(valid, train_lambda)
        final_lambda = _fit_convex_lambda(refit, mode=mode)
        artifacts["models"][name] = {
            "kind": "lambda",
            "mode": mode,
            "lambda": final_lambda,
            "t1_lambda": train_lambda,
            "t2_nmse": _normalized_mse(valid, valid_prediction),
            "constraint": "[0,1] by clipping the closed-form least-squares estimate",
        }
    for name, design, mode in RIDGE_MODELS:
        if name not in selected:
            continue
        train_statistics = _ridge_statistics(train, design, mode)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for alpha in grid:
            coefficients = _solve_ridge_statistics(train_statistics, alpha)
            prediction = _predict_anchored_ridge(
                valid,
                design=design,
                mode=mode,
                coefficients=coefficients,
            )
            candidates.append(
                (alpha, _normalized_mse(valid, prediction), coefficients)
            )
        selected_alpha, validation_nmse, t1_coefficients = min(
            candidates,
            key=lambda item: (item[1], item[0]),
        )
        final_statistics = _ridge_statistics(refit, design, mode)
        final_coefficients = _solve_ridge_statistics(
            final_statistics,
            selected_alpha,
        )
        artifacts["models"][name] = {
            "kind": "ridge",
            "design": design,
            "signals": RIDGE_DESIGNS[design],
            "mode": mode,
            "alpha": selected_alpha,
            "coef": final_coefficients,
            "t1_coef": t1_coefficients,
            "t2_nmse": validation_nmse,
        }
    return artifacts


def iter_baseline_predictions(
    arrays: dict[str, np.ndarray],
    artifacts: dict[str, Any],
    *,
    methods: Sequence[str],
):
    """Yield one complete prediction at a time so callers can release it."""
    selected = set(methods)
    yield "vanilla", arrays["pred"]
    if "context_forecast" in selected:
        yield "context_forecast", arrays["pred_c"]
    if "aggr_y" in selected:
        yield "aggr_y", weighted_neighbor_horizon(arrays)
    if "y_mean" in selected:
        yield "y_mean", arrays["y_c"].mean(axis=1)
    for name, model in artifacts["models"].items():
        if name not in selected:
            continue
        if model["kind"] == "lambda":
            prediction = _predict_convex_lambda(arrays, model["lambda"])
        else:
            prediction = _predict_anchored_ridge(
                arrays,
                design=model["design"],
                mode=model["mode"],
                coefficients=model["coef"],
            )
        yield name, prediction


def run_streamed_baselines(
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    fit_arrays: dict[str, dict[str, np.ndarray]],
    *,
    output_dir: Path,
    selected_methods: Sequence[str],
    l2_grid: Sequence[float],
    fit_on_eval: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Fit baseline coefficients once and persist one prediction at a time."""
    artifacts = fit_baseline_adapters(
        fit_arrays["T1"],
        fit_arrays["T2"],
        fit_arrays["T1+T2"],
        l2_grid,
        methods=selected_methods,
    )
    eval_fit_artifacts = (
        fit_baseline_adapters(
            fit_arrays["T3_oracle"],
            fit_arrays["T3_oracle"],
            fit_arrays["T3_oracle"],
            l2_grid,
            methods=selected_methods,
        )
        if fit_on_eval
        else None
    )
    store = PredictionStore(output_dir)
    rows: list[dict[str, Any]] = []
    reference_nmse = vanilla_nmse(arrays_by_split["eval"])
    for split, arrays in arrays_by_split.items():
        for name, prediction in iter_baseline_predictions(
            arrays,
            artifacts,
            methods=selected_methods,
        ):
            path = store.write(split, "predictions", name, prediction)
            if split == "eval":
                saved = np.load(path, mmap_mode="r", allow_pickle=False)
                rows.append(
                    evaluate_prediction(
                        "eval",
                        arrays,
                        name,
                        saved,
                        vanilla_nmse=reference_nmse,
                    )
                )
                del saved
            if prediction is not arrays.get("pred") and prediction is not arrays.get(
                "pred_c"
            ):
                del prediction
        gc.collect()
    if eval_fit_artifacts is not None:
        arrays = arrays_by_split["eval"]
        for name, prediction in iter_baseline_predictions(
            arrays,
            eval_fit_artifacts,
            methods=selected_methods,
        ):
            if name not in TRAINABLE_BASELINES:
                continue
            output_name = f"{name}_eval_fit"
            path = store.write("eval", "predictions", output_name, prediction)
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            rows.append(
                evaluate_prediction(
                    "eval",
                    arrays,
                    output_name,
                    saved,
                    vanilla_nmse=reference_nmse,
                )
            )
            del saved, prediction
        gc.collect()
    saved_artifacts = {
        "format": "adaptation_baseline_models",
        "protocol": artifacts["protocol"],
        "l2_grid": list(artifacts["l2_grid"]),
        "models": artifacts["models"],
        "eval_fit_models": (
            None if eval_fit_artifacts is None else eval_fit_artifacts["models"]
        ),
    }
    artifact_path = output_dir / "baseline_artifacts.pt"
    torch.save(saved_artifacts, artifact_path)
    prediction_manifest = store.finalize(
        metadata={
            "family": "baselines",
            "fit_on_eval": bool(fit_on_eval),
        }
    )
    return rows, saved_artifacts, prediction_manifest


STATIC_GATE_FEATURE_NAMES = (
    "neighbor_y_minus_vanilla_mean",
    "neighbor_y_minus_vanilla_between_std",
    "neighbor_y_minus_vanilla_within_std_mean",
    "neighbor_y_minus_neighbor_pred_mean",
    "neighbor_y_minus_neighbor_pred_between_std",
    "neighbor_y_minus_neighbor_pred_within_std_mean",
    "query_mean",
    "query_std",
    "neighbor_lookback_means_mean_raw",
    "neighbor_lookback_means_std_raw",
    "neighbor_lookback_stds_mean_raw",
    "neighbor_lookback_stds_std_raw",
    "same_user_ratio",
    "neighbor_age_mean",
    "neighbor_weight_std",
    "neighbor_weight_max",
    "distance_mean",
)

SCALAR_GATE_FEATURE_NAMES = (
    "candidate_minus_vanilla_mean",
    "candidate_minus_vanilla_std",
    *STATIC_GATE_FEATURE_NAMES,
)

HORIZON_GATE_FEATURE_NAMES = (
    "candidate_minus_vanilla_h",
    "neighbor_y_minus_vanilla_mean_h",
    "neighbor_y_minus_vanilla_std_h",
    "neighbor_y_minus_neighbor_pred_mean_h",
    "neighbor_y_minus_neighbor_pred_std_h",
    *STATIC_GATE_FEATURE_NAMES,
)
GATE_CANDIDATES = ("context", "aggr_y")
GATE_DIRECT_METHODS = ("context_forecast", "aggr_y")
GATE_ADAPTIVE_METHODS = tuple(
    f"{prefix}_{candidate}_{suffix}"
    for candidate in GATE_CANDIDATES
    for prefix, suffixes in (
        ("bayes", ("shared", "horizon")),
        (
            "catboost",
            (
                "classifier_shared",
                "classifier_horizon",
                "regressor_shared",
                "regressor_horizon",
            ),
        ),
    )
    for suffix in suffixes
)
GATE_ORACLE_METHODS = tuple(
    f"oracle_{candidate}_{shape}"
    for candidate in GATE_CANDIDATES
    for shape in ("shared", "horizon")
)
GATE_METHODS = (*GATE_DIRECT_METHODS, *GATE_ADAPTIVE_METHODS, *GATE_ORACLE_METHODS)


def horizon_gate_feature_names(horizon: int | None = None) -> tuple[str, ...]:
    del horizon
    return HORIZON_GATE_FEATURE_NAMES


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(weights * values, axis=1)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = _weighted_mean(values, weights)
    variance = _weighted_mean((values - mean[:, None]) ** 2, weights)
    return np.sqrt(np.maximum(variance, 0.0))


def _candidate_prediction(
    arrays: dict[str, np.ndarray],
    candidate: str,
) -> np.ndarray:
    if candidate == "context":
        return arrays["pred_c"]
    if candidate == "aggr_y":
        return weighted_neighbor_horizon(arrays)
    raise ValueError(f"unknown gate candidate {candidate!r}")


def _static_gate_features_dense(
    arrays: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    pred = arrays["pred"]
    weights = distance_weights(arrays)
    y_minus_v = arrays["y_c"] - pred[:, None, :]
    y_minus_n = (
        arrays["e"]
        if "e" in arrays
        else arrays["y_c"] - arrays["pred_neighbors"]
    )
    yv_horizon_mean = y_minus_v.mean(axis=2)
    yn_horizon_mean = y_minus_n.mean(axis=2)
    values = [
        _weighted_mean(yv_horizon_mean, weights),
        _weighted_std(yv_horizon_mean, weights),
        _weighted_mean(y_minus_v.std(axis=2), weights),
        _weighted_mean(yn_horizon_mean, weights),
        _weighted_std(yn_horizon_mean, weights),
        _weighted_mean(y_minus_n.std(axis=2), weights),
        arrays["x"].mean(axis=1),
        arrays["x"].std(axis=1),
        arrays["neighbor_lookback_mean"],
        arrays["neighbor_lookback_mean_std"],
        arrays["neighbor_lookback_std"],
        arrays["neighbor_lookback_std_std"],
        arrays["same_user_ratio"],
        arrays["neighbor_age_mean"],
        weights.std(axis=1),
        weights.max(axis=1),
        arrays["distance"].mean(axis=1),
    ]
    local = {
        "aggr_y": np.sum(weights[:, :, None] * arrays["y_c"], axis=1),
        "yv_mean": np.sum(weights[:, :, None] * y_minus_v, axis=1),
        "yv_std": np.sqrt(
            np.maximum(
                np.sum(
                    weights[:, :, None]
                    * (
                        y_minus_v
                        - np.sum(weights[:, :, None] * y_minus_v, axis=1)[:, None, :]
                    )
                    ** 2,
                    axis=1,
                ),
                0.0,
            )
        ),
        "yn_mean": np.sum(weights[:, :, None] * y_minus_n, axis=1),
        "yn_std": np.sqrt(
            np.maximum(
                np.sum(
                    weights[:, :, None]
                    * (
                        y_minus_n
                        - np.sum(weights[:, :, None] * y_minus_n, axis=1)[:, None, :]
                    )
                    ** 2,
                    axis=1,
                ),
                0.0,
            )
        ),
    }
    return values, local


def compact_gate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    chunk_rows: int = GATE_FEATURE_CHUNK_ROWS,
) -> dict[str, np.ndarray]:
    """Retain only gate inputs and compute neighbor summaries in bounded chunks."""
    n_samples, horizon = arrays["y"].shape
    static = np.empty((n_samples, len(STATIC_GATE_FEATURE_NAMES)), dtype=np.float32)
    local = np.empty((n_samples, horizon, 4), dtype=np.float32)
    aggregate = np.empty((n_samples, horizon), dtype=np.float32)
    for start in range(0, n_samples, int(chunk_rows)):
        stop = min(start + int(chunk_rows), n_samples)
        chunk = {name: value[start:stop] for name, value in arrays.items()}
        values, horizon_values = _static_gate_features_dense(chunk)
        aggregate[start:stop] = horizon_values["aggr_y"]
        static[start:stop] = np.stack(values, axis=1).astype(
            np.float32,
            copy=False,
        )
        local[start:stop] = np.stack(
            [
                horizon_values["yv_mean"],
                horizon_values["yv_std"],
                horizon_values["yn_mean"],
                horizon_values["yn_std"],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
    return {
        "pred": arrays["pred"].astype(np.float32, copy=False),
        "pred_c": arrays["pred_c"].astype(np.float32, copy=False),
        "aggr_y": aggregate,
        "y": arrays["y"].astype(np.float32, copy=False),
        "scale": np.maximum(
            arrays["x"].std(axis=1, keepdims=True),
            1e-8,
        ).astype(np.float32, copy=False),
        "query_t": arrays["query_t"],
        "gate_static": static,
        "gate_local": local,
    }


def compact_baseline_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Drop extraction tensors that baseline fitting and scoring never consume."""
    return {
        "pred": arrays["pred"].astype(np.float32, copy=False),
        "pred_c": arrays["pred_c"].astype(np.float32, copy=False),
        "y": arrays["y"].astype(np.float32, copy=False),
        "y_c": arrays["y_c"].astype(np.float32, copy=False),
        "pred_neighbors": arrays["pred_neighbors"].astype(np.float32, copy=False),
        "distance": arrays["distance"].astype(np.float32, copy=False),
        "scale": np.maximum(
            arrays["x"].std(axis=1, keepdims=True),
            1e-8,
        ).astype(np.float32, copy=False),
        "query_t": arrays["query_t"],
    }


def _static_gate_features(
    arrays: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    if "gate_static" not in arrays:
        return _static_gate_features_dense(arrays)
    static = arrays["gate_static"]
    local = arrays["gate_local"]
    return (
        [static[:, index] for index in range(static.shape[1])],
        {
            "yv_mean": local[:, :, 0],
            "yv_std": local[:, :, 1],
            "yn_mean": local[:, :, 2],
            "yn_std": local[:, :, 3],
        },
    )


def scalar_gate_features(
    arrays: dict[str, np.ndarray],
    candidate: str = "context",
) -> np.ndarray:
    candidate_delta = _candidate_prediction(arrays, candidate) - arrays["pred"]
    static, _ = _static_gate_features(arrays)
    return np.stack(
        [
            candidate_delta.mean(axis=1),
            candidate_delta.std(axis=1),
            *static,
        ],
        axis=1,
    ).astype(np.float32)


class HorizonGateFeatureView(Sequence[np.ndarray]):
    """Lazily materialize one local feature matrix at a time."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        candidate: str,
    ):
        self.candidate_delta = (
            _candidate_prediction(arrays, candidate) - arrays["pred"]
        )
        if "gate_static" in arrays:
            self.common = arrays["gate_static"]
            local = arrays["gate_local"]
            self.local = {
                "yv_mean": local[:, :, 0],
                "yv_std": local[:, :, 1],
                "yn_mean": local[:, :, 2],
                "yn_std": local[:, :, 3],
            }
        else:
            static, self.local = _static_gate_features(arrays)
            self.common = np.stack(static, axis=1).astype(
                np.float32,
                copy=False,
            )

    def __len__(self) -> int:
        return int(self.candidate_delta.shape[1])

    def __getitem__(self, horizon: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(horizon, slice):
            return [
                self[index]
                for index in range(*horizon.indices(len(self)))
            ]
        index = int(horizon)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return np.column_stack(
            [
                self.candidate_delta[:, index],
                self.local["yv_mean"][:, index],
                self.local["yv_std"][:, index],
                self.local["yn_mean"][:, index],
                self.local["yn_std"][:, index],
                self.common,
            ]
        ).astype(np.float32, copy=False)


def horizon_gate_features(
    arrays: dict[str, np.ndarray],
    candidate: str = "context",
) -> Sequence[np.ndarray]:
    """Return lazy local features; no model sees another horizon's C_h-V_h."""
    return HorizonGateFeatureView(arrays, candidate)


def _catboost_regressor(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    task_type: str = "CPU",
    devices: str | None = None,
    thread_count: int | None = None,
):
    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
    execution = _catboost_execution_kwargs(
        task_type=task_type,
        devices=devices,
        thread_count=thread_count,
    )
    return CatBoostRegressor(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        **execution,
    )


def _catboost_classifier(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    task_type: str = "CPU",
    devices: str | None = None,
    thread_count: int | None = None,
):
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
    execution = _catboost_execution_kwargs(
        task_type=task_type,
        devices=devices,
        thread_count=thread_count,
    )
    return CatBoostClassifier(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        **execution,
    )


def _catboost_execution_kwargs(
    *,
    task_type: str,
    devices: str | None,
    thread_count: int | None,
) -> dict[str, Any]:
    mode = str(task_type).upper()
    if mode not in {"CPU", "GPU"}:
        raise ValueError("CatBoost task_type must be CPU or GPU")
    if thread_count is not None and int(thread_count) <= 0:
        raise ValueError("CatBoost thread_count must be positive")
    if mode == "CPU" and devices not in {None, ""}:
        raise ValueError("CatBoost devices is only valid for GPU training")
    out: dict[str, Any] = {"task_type": mode}
    if thread_count is not None:
        out["thread_count"] = int(thread_count)
    if mode == "GPU" and devices not in {None, ""}:
        out["devices"] = str(devices)
    return out


def _selected_iterations(estimator: Any, fallback: int) -> int:
    best = estimator.get_best_iteration()
    return int(best) + 1 if best is not None and int(best) >= 0 else int(fallback)


def fit_loss_difference_regressor(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    valid_x_np: np.ndarray | None = None,
    valid_y_np: np.ndarray | None = None,
    refit_x_np: np.ndarray | None = None,
    refit_y_np: np.ndarray | None = None,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    early_stopping_rounds: int = 50,
    task_type: str = "CPU",
    devices: str | None = None,
    thread_count: int | None = None,
) -> dict[str, Any]:
    target = np.asarray(y_np, dtype=np.float32).reshape(-1)
    if np.ptp(target) <= 1e-12:
        return {"constant": float(target.mean()), "selected_iterations": 0}
    selected = int(iterations)
    if valid_x_np is not None and valid_y_np is not None:
        selector = _catboost_regressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed,
            task_type=task_type,
            devices=devices,
            thread_count=thread_count,
        )
        selector.fit(
            x_np,
            target,
            eval_set=(valid_x_np, np.asarray(valid_y_np).reshape(-1)),
            early_stopping_rounds=max(1, int(early_stopping_rounds)),
            use_best_model=True,
        )
        selected = _selected_iterations(selector, iterations)
        del selector
        gc.collect()
    if refit_x_np is not None and refit_y_np is not None:
        final_x = refit_x_np
        final_y = np.asarray(refit_y_np).reshape(-1)
    elif valid_x_np is not None and valid_y_np is not None:
        final_x = np.concatenate([x_np, valid_x_np], axis=0)
        final_y = np.concatenate([target, np.asarray(valid_y_np).reshape(-1)])
    else:
        final_x, final_y = x_np, target
    model = _catboost_regressor(
        iterations=selected,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed,
        task_type=task_type,
        devices=devices,
        thread_count=thread_count,
    )
    model.fit(final_x, final_y)
    return {"regressor": model, "selected_iterations": selected}


def fit_improvement_classifier(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    valid_x_np: np.ndarray | None = None,
    valid_y_np: np.ndarray | None = None,
    refit_x_np: np.ndarray | None = None,
    refit_y_np: np.ndarray | None = None,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    early_stopping_rounds: int = 50,
    task_type: str = "CPU",
    devices: str | None = None,
    thread_count: int | None = None,
) -> dict[str, Any]:
    target = np.asarray(y_np).reshape(-1) > 0.0
    valid_target = (
        None
        if valid_y_np is None
        else np.asarray(valid_y_np).reshape(-1) > 0.0
    )
    if refit_y_np is not None:
        combined_target = np.asarray(refit_y_np).reshape(-1) > 0.0
    else:
        combined_target = (
            target
            if valid_target is None
            else np.concatenate([target, valid_target])
        )
    if np.unique(combined_target).size == 1:
        return {
            "constant": float(combined_target[0]) - 0.5,
            "selected_iterations": 0,
        }
    selected = int(iterations)
    if (
        valid_x_np is not None
        and valid_target is not None
        and np.unique(target).size > 1
    ):
        selector = _catboost_classifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed,
            task_type=task_type,
            devices=devices,
            thread_count=thread_count,
        )
        selector.fit(
            x_np,
            target.astype(np.int8),
            eval_set=(valid_x_np, valid_target.astype(np.int8)),
            early_stopping_rounds=max(1, int(early_stopping_rounds)),
            use_best_model=True,
        )
        selected = _selected_iterations(selector, iterations)
        del selector
        gc.collect()
    if refit_x_np is not None:
        final_x = refit_x_np
    else:
        final_x = (
            x_np
            if valid_x_np is None
            else np.concatenate([x_np, valid_x_np], axis=0)
        )
    model = _catboost_classifier(
        iterations=selected,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed,
        task_type=task_type,
        devices=devices,
        thread_count=thread_count,
    )
    model.fit(final_x, combined_target.astype(np.int8))
    return {"classifier": model, "selected_iterations": selected}


def fit_gate(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    valid_x_np: np.ndarray | None = None,
    valid_y_np: np.ndarray | None = None,
    refit_x_np: np.ndarray | None = None,
    refit_y_np: np.ndarray | None = None,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    objective: str = "regressor",
    early_stopping_rounds: int = 50,
    task_type: str = "CPU",
    devices: str | None = None,
    thread_count: int | None = None,
) -> dict[str, Any]:
    if x_np.shape[0] == 0:
        raise ValueError("cannot train gates from an empty T1 slice")
    targets = np.asarray(y_np).reshape(-1)
    valid_targets = (
        None if valid_y_np is None else np.asarray(valid_y_np).reshape(-1)
    )
    refit_targets = (
        None if refit_y_np is None else np.asarray(refit_y_np).reshape(-1)
    )
    if targets.shape[0] != x_np.shape[0]:
        raise ValueError("gate features and targets must align")
    if (
        valid_x_np is not None
        and valid_targets is not None
        and valid_targets.shape[0] != valid_x_np.shape[0]
    ):
        raise ValueError("validation gate features and targets must align")
    if (
        refit_x_np is not None
        and refit_targets is not None
        and refit_targets.shape[0] != refit_x_np.shape[0]
    ):
        raise ValueError("refit gate features and targets must align")
    fit_one = (
        fit_improvement_classifier
        if objective == "classifier"
        else fit_loss_difference_regressor
    )
    return fit_one(
        x_np,
        targets,
        valid_x_np=valid_x_np,
        valid_y_np=valid_targets,
        refit_x_np=refit_x_np,
        refit_y_np=refit_targets,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        task_type=task_type,
        devices=devices,
        thread_count=thread_count,
    )


def _fit_no_feature_gates(arrays: dict[str, np.ndarray], candidate: str) -> dict[str, Any]:
    candidate_prediction = _candidate_prediction(arrays, candidate)
    base_loss = (arrays["y"] - arrays["pred"]) ** 2
    candidate_loss = (arrays["y"] - candidate_prediction) ** 2
    improvement = base_loss - candidate_loss
    return {
        "shared_score": float(improvement.mean()),
        "horizon_score": improvement.mean(axis=0).astype(np.float64),
    }


def _gate_targets(
    arrays: dict[str, np.ndarray],
    candidate: str,
) -> dict[str, np.ndarray]:
    candidate_prediction = _candidate_prediction(arrays, candidate)
    improvement = (
        (arrays["y"] - arrays["pred"]) ** 2
        - (arrays["y"] - candidate_prediction) ** 2
    )
    return {
        "shared": improvement.mean(axis=1, keepdims=True),
        "horizon": improvement,
    }


def predict_gate(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if "constant" in model:
        return np.full(features.shape[0], model["constant"], dtype=np.float32)
    if "classifier" in model:
        return (
            model["classifier"].predict_proba(features)[:, 1] - 0.5
        ).astype(np.float32, copy=False)
    return np.asarray(
        model["regressor"].predict(features),
        dtype=np.float32,
    )


def _save_gate_model(
    model: dict[str, Any],
    *,
    output_dir: Path,
    model_dir: Path,
    stem: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "selected_iterations": int(model.get("selected_iterations", 0)),
    }
    if "constant" in model:
        entry.update({"kind": "constant", "value": float(model["constant"])})
        return entry
    kind = "classifier" if "classifier" in model else "regressor"
    path = model_dir / f"{stem}.cbm"
    path.parent.mkdir(parents=True, exist_ok=True)
    model[kind].save_model(str(path))
    entry.update(
        {
            "kind": kind,
            "path": path.relative_to(output_dir).as_posix(),
        }
    )
    return entry


def _model_feature_importance(model: dict[str, Any]) -> np.ndarray | None:
    estimator = model.get("classifier", model.get("regressor"))
    if estimator is None:
        return None
    return np.asarray(estimator.get_feature_importance(), dtype=np.float64)


def _write_gated_prediction(
    store: PredictionStore,
    *,
    split: str,
    method: str,
    arrays: dict[str, np.ndarray],
    candidate: str,
    score: np.ndarray,
    shared: bool,
) -> np.memmap:
    prediction = store.open(
        split,
        "predictions",
        method,
        shape=arrays["pred"].shape,
        dtype=np.float32,
    )
    candidate_prediction = _candidate_prediction(arrays, candidate)
    for start in range(0, arrays["pred"].shape[0], METRIC_CHUNK_ROWS):
        stop = min(start + METRIC_CHUNK_ROWS, arrays["pred"].shape[0])
        decision = np.asarray(score[start:stop]) > 0.0
        if shared:
            decision = decision.reshape(-1, 1)
        prediction[start:stop] = np.where(
            decision,
            candidate_prediction[start:stop],
            arrays["pred"][start:stop],
        )
    prediction.flush()
    return prediction


def _save_importance_outputs(
    importances: dict[str, tuple[tuple[str, ...], list[np.ndarray]]],
    output_dir: Path,
    *,
    top_k: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not any(values for _, values in importances.values()):
        return []
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover
        LOGGER.warning("matplotlib unavailable; skipping feature importance plots")
        return []
    saved: list[Path] = []
    for method, (names, values) in importances.items():
        if not values:
            continue
        importance = np.mean(np.stack(values), axis=0)
        order = np.argsort(importance)[::-1][: min(int(top_k), len(names))]
        frame = pd.DataFrame(
            {
                "feature": np.asarray(names, dtype=object)[order],
                "importance": importance[order],
            }
        )
        stem = f"feature_importance_{method.removeprefix('catboost_')}"
        csv_path = output_dir / f"{stem}.csv"
        frame.to_csv(csv_path, index=False)
        saved.append(csv_path)
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(frame))))
        ax.barh(frame["feature"][::-1], frame["importance"][::-1])
        ax.set_xlabel("Mean CatBoost feature importance")
        ax.set_title(method)
        fig.tight_layout()
        png_path = output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        saved.append(png_path)
    return saved


def run_streamed_gates(
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    fit_arrays: dict[str, dict[str, np.ndarray]],
    *,
    output_dir: Path,
    selected_methods: Sequence[str],
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
    seed: int,
    task_type: str,
    devices: str | None,
    thread_count: int | None,
    feature_importance_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Fit and score each gate model once, releasing it immediately after use."""
    selected = set(selected_methods)
    store = PredictionStore(output_dir)
    model_root = output_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    reference_nmse = vanilla_nmse(arrays_by_split["eval"])
    model_manifest: dict[str, Any] = {
        "format": "adaptation_gate_models",
        "protocol": "fit_T1_validate_iterations_T2_refit_T1_plus_T2",
        "config": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "early_stopping_rounds": int(early_stopping_rounds),
            "seed": int(seed),
            "task_type": str(task_type).upper(),
            "devices": devices,
            "thread_count": thread_count,
            "horizon_fits": "serial",
            "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
            "horizon_feature_names": HORIZON_GATE_FEATURE_NAMES,
        },
        "models": {},
        "bayes": {},
    }
    importances: dict[str, tuple[tuple[str, ...], list[np.ndarray]]] = {}

    def save_prediction(
        split: str,
        method: str,
        prediction: np.ndarray,
    ) -> None:
        path = store.write(split, "predictions", method, prediction)
        if split == "eval":
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            metric_rows.append(
                evaluate_prediction(
                    "eval",
                    arrays_by_split["eval"],
                    method,
                    saved,
                    vanilla_nmse=reference_nmse,
                )
            )
            del saved

    for split, arrays in arrays_by_split.items():
        save_prediction(split, "vanilla", arrays["pred"])
        if "context_forecast" in selected:
            save_prediction(split, "context_forecast", arrays["pred_c"])
        if "aggr_y" in selected:
            save_prediction(split, "aggr_y", arrays["aggr_y"])

    for candidate_index, candidate in enumerate(GATE_CANDIDATES):
        candidate_methods = {
            method for method in selected
            if f"_{candidate}_" in method
        }
        if not candidate_methods:
            continue
        LOGGER.info("gate fitting start candidate=%s", candidate)
        catboost_required = any(
            method.startswith("catboost_") for method in candidate_methods
        )
        contiguous_partition = (
            catboost_required and _contiguous_fit_partition(fit_arrays)
        )
        refit_is_adapt = (
            fit_arrays["T1+T2"] is arrays_by_split["adapt"]
        )
        train_targets: dict[str, np.ndarray] = {}
        valid_targets: dict[str, np.ndarray] = {}
        refit_targets: dict[str, np.ndarray] = {}
        scalar_features: dict[str, np.ndarray] = {}
        horizon_features: dict[str, Sequence[np.ndarray]] = {}
        if catboost_required:
            n_train = fit_arrays["T1"]["y"].shape[0]
            refit_targets = _gate_targets(fit_arrays["T1+T2"], candidate)
            if contiguous_partition:
                train_targets = {
                    name: value[:n_train]
                    for name, value in refit_targets.items()
                }
                valid_targets = {
                    name: value[n_train:]
                    for name, value in refit_targets.items()
                }
            else:
                train_targets = _gate_targets(fit_arrays["T1"], candidate)
                valid_targets = _gate_targets(fit_arrays["T2"], candidate)

            refit_scalar = scalar_gate_features(
                fit_arrays["T1+T2"],
                candidate,
            )
            scalar_features["T1+T2"] = refit_scalar
            if contiguous_partition:
                scalar_features["T1"] = refit_scalar[:n_train]
                scalar_features["T2"] = refit_scalar[n_train:]
            else:
                scalar_features["T1"] = scalar_gate_features(
                    fit_arrays["T1"],
                    candidate,
                )
                scalar_features["T2"] = scalar_gate_features(
                    fit_arrays["T2"],
                    candidate,
                )
            scalar_features["eval"] = scalar_gate_features(
                arrays_by_split["eval"],
                candidate,
            )
            scalar_features["adapt"] = (
                refit_scalar
                if refit_is_adapt
                else scalar_gate_features(arrays_by_split["adapt"], candidate)
            )

            horizon_features["T1+T2"] = horizon_gate_features(
                fit_arrays["T1+T2"],
                candidate,
            )
            if not contiguous_partition:
                horizon_features["T1"] = horizon_gate_features(
                    fit_arrays["T1"],
                    candidate,
                )
                horizon_features["T2"] = horizon_gate_features(
                    fit_arrays["T2"],
                    candidate,
                )
            horizon_features["eval"] = horizon_gate_features(
                arrays_by_split["eval"],
                candidate,
            )
            horizon_features["adapt"] = (
                horizon_features["T1+T2"]
                if refit_is_adapt
                else horizon_gate_features(arrays_by_split["adapt"], candidate)
            )

            for split, arrays in arrays_by_split.items():
                shared_target = store.open(
                    split,
                    "gate_diagnostics",
                    f"{candidate}_shared_target",
                    shape=(arrays["y"].shape[0],),
                    dtype=np.float32,
                )
                horizon_target = store.open(
                    split,
                    "gate_diagnostics",
                    f"{candidate}_horizon_target",
                    shape=arrays["y"].shape,
                    dtype=np.float32,
                )
                candidate_prediction = _candidate_prediction(arrays, candidate)
                for start in range(0, arrays["y"].shape[0], METRIC_CHUNK_ROWS):
                    stop = min(start + METRIC_CHUNK_ROWS, arrays["y"].shape[0])
                    improvement = (
                        (arrays["y"][start:stop] - arrays["pred"][start:stop]) ** 2
                        - (
                            arrays["y"][start:stop]
                            - candidate_prediction[start:stop]
                        )
                        ** 2
                    )
                    shared_target[start:stop] = improvement.mean(axis=1)
                    horizon_target[start:stop] = improvement
                shared_target.flush()
                horizon_target.flush()
                del shared_target, horizon_target

        bayes = (
            _fit_no_feature_gates(fit_arrays["T1+T2"], candidate)
            if any(method.startswith("bayes_") for method in candidate_methods)
            else None
        )
        if bayes is not None:
            model_manifest["bayes"][candidate] = {
                "shared_score": float(bayes["shared_score"]),
                "horizon_score": np.asarray(bayes["horizon_score"]).tolist(),
            }
        for shape in ("shared", "horizon"):
            method = f"bayes_{candidate}_{shape}"
            if method not in candidate_methods:
                continue
            assert bayes is not None
            score = (
                np.asarray([bayes["shared_score"]], dtype=np.float32)
                if shape == "shared"
                else np.asarray(bayes["horizon_score"], dtype=np.float32)
            )
            store.write("fit", "gate_diagnostics", f"{method}_score", score)
            for split, arrays in arrays_by_split.items():
                expanded = (
                    np.full(arrays["pred"].shape[0], score[0], dtype=np.float32)
                    if shape == "shared"
                    else np.broadcast_to(score, arrays["pred"].shape)
                )
                prediction = _write_gated_prediction(
                    store,
                    split=split,
                    method=method,
                    arrays=arrays,
                    candidate=candidate,
                    score=expanded,
                    shared=shape == "shared",
                )
                if split == "eval":
                    metric_rows.append(
                        evaluate_prediction(
                            "eval",
                            arrays,
                            method,
                            prediction,
                            vanilla_nmse=reference_nmse,
                        )
                    )
                del prediction

        for shape in ("shared", "horizon"):
            method = f"oracle_{candidate}_{shape}"
            if method not in candidate_methods:
                continue
            for split, arrays in arrays_by_split.items():
                prediction = store.open(
                    split,
                    "predictions",
                    method,
                    shape=arrays["pred"].shape,
                    dtype=np.float32,
                )
                candidate_prediction = _candidate_prediction(arrays, candidate)
                for start in range(0, arrays["pred"].shape[0], METRIC_CHUNK_ROWS):
                    stop = min(start + METRIC_CHUNK_ROWS, arrays["pred"].shape[0])
                    base_loss = (
                        arrays["y"][start:stop] - arrays["pred"][start:stop]
                    ) ** 2
                    candidate_loss = (
                        arrays["y"][start:stop]
                        - candidate_prediction[start:stop]
                    ) ** 2
                    decision = candidate_loss < base_loss
                    if shape == "shared":
                        decision = (
                            candidate_loss.mean(axis=1, keepdims=True)
                            < base_loss.mean(axis=1, keepdims=True)
                        )
                    prediction[start:stop] = np.where(
                        decision,
                        candidate_prediction[start:stop],
                        arrays["pred"][start:stop],
                    )
                prediction.flush()
                if split == "eval":
                    metric_rows.append(
                        evaluate_prediction(
                            "eval",
                            arrays,
                            method,
                            prediction,
                            vanilla_nmse=reference_nmse,
                        )
                    )
                del prediction

        for objective_index, objective in enumerate(("classifier", "regressor")):
            for shape in ("shared", "horizon"):
                method = f"catboost_{candidate}_{objective}_{shape}"
                if method not in candidate_methods:
                    continue
                names = (
                    SCALAR_GATE_FEATURE_NAMES
                    if shape == "shared"
                    else HORIZON_GATE_FEATURE_NAMES
                )
                importance_values: list[np.ndarray] = []
                entries: list[dict[str, Any]] = []
                if shape == "shared":
                    model = fit_gate(
                        scalar_features["T1"],
                        train_targets["shared"],
                        valid_x_np=scalar_features["T2"],
                        valid_y_np=valid_targets["shared"],
                        refit_x_np=scalar_features["T1+T2"],
                        refit_y_np=refit_targets["shared"],
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        early_stopping_rounds=early_stopping_rounds,
                        seed=seed + candidate_index * 100_000
                        + objective_index * 10_000,
                        objective=objective,
                        task_type=task_type,
                        devices=devices,
                        thread_count=thread_count,
                    )
                    entries.append(
                        _save_gate_model(
                            model,
                            output_dir=output_dir,
                            model_dir=model_root / method,
                            stem="shared",
                        )
                    )
                    importance = _model_feature_importance(model)
                    if importance is not None:
                        importance_values.append(importance)
                    for split, arrays in arrays_by_split.items():
                        score = predict_gate(model, scalar_features[split])
                        store.write(
                            split,
                            "gate_diagnostics",
                            f"{method}_score",
                            score,
                        )
                        prediction = _write_gated_prediction(
                            store,
                            split=split,
                            method=method,
                            arrays=arrays,
                            candidate=candidate,
                            score=score,
                            shared=True,
                        )
                        if split == "eval":
                            metric_rows.append(
                                evaluate_prediction(
                                    "eval",
                                    arrays,
                                    method,
                                    prediction,
                                    vanilla_nmse=reference_nmse,
                                )
                            )
                        del prediction, score
                    del model
                else:
                    score_arrays = {
                        split: store.open(
                            split,
                            "gate_diagnostics",
                            f"{method}_score",
                            shape=arrays["pred"].shape,
                            dtype=np.float32,
                        )
                        for split, arrays in arrays_by_split.items()
                    }
                    horizon = fit_arrays["T1"]["y"].shape[1]
                    for h in range(horizon):
                        refit_x = horizon_features["T1+T2"][h]
                        if contiguous_partition:
                            train_x = refit_x[:n_train]
                            valid_x = refit_x[n_train:]
                        else:
                            train_x = horizon_features["T1"][h]
                            valid_x = horizon_features["T2"][h]
                        model = fit_gate(
                            train_x,
                            train_targets["horizon"][:, h],
                            valid_x_np=valid_x,
                            valid_y_np=valid_targets["horizon"][:, h],
                            refit_x_np=refit_x,
                            refit_y_np=refit_targets["horizon"][:, h],
                            iterations=iterations,
                            learning_rate=learning_rate,
                            depth=depth,
                            early_stopping_rounds=early_stopping_rounds,
                            seed=seed + candidate_index * 100_000
                            + objective_index * 10_000 + 1_000 + h,
                            objective=objective,
                            task_type=task_type,
                            devices=devices,
                            thread_count=thread_count,
                        )
                        entries.append(
                            _save_gate_model(
                                model,
                                output_dir=output_dir,
                                model_dir=model_root / method,
                                stem=f"horizon_{h:04d}",
                            )
                        )
                        importance = _model_feature_importance(model)
                        if importance is not None:
                            importance_values.append(importance)
                        for split in arrays_by_split:
                            scoring_x = (
                                refit_x
                                if split == "adapt" and refit_is_adapt
                                else horizon_features[split][h]
                            )
                            score_arrays[split][:, h] = predict_gate(
                                model,
                                scoring_x,
                            )
                            del scoring_x
                        del model, train_x, valid_x, refit_x
                        if (h + 1) % 8 == 0:
                            gc.collect()
                    for score in score_arrays.values():
                        score.flush()
                    for split, arrays in arrays_by_split.items():
                        prediction = _write_gated_prediction(
                            store,
                            split=split,
                            method=method,
                            arrays=arrays,
                            candidate=candidate,
                            score=score_arrays[split],
                            shared=False,
                        )
                        if split == "eval":
                            metric_rows.append(
                                evaluate_prediction(
                                    "eval",
                                    arrays,
                                    method,
                                    prediction,
                                    vanilla_nmse=reference_nmse,
                                )
                            )
                        del prediction
                    del score_arrays
                model_manifest["models"][method] = entries
                importances[method] = (names, importance_values)
                gc.collect()

        del scalar_features, horizon_features
        del train_targets, valid_targets, refit_targets
        gc.collect()
        LOGGER.info("gate fitting done candidate=%s", candidate)

    importance_paths = _save_importance_outputs(
        importances,
        output_dir / "plots",
        top_k=feature_importance_top_k,
    )
    model_manifest["feature_importance_files"] = [
        path.relative_to(output_dir).as_posix() for path in importance_paths
    ]
    artifact_path = output_dir / "gate_artifacts.json"
    artifact_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    prediction_manifest = store.finalize(
        metadata={
            "family": "gates",
            "diagnostics": "scores_only",
        }
    )
    return metric_rows, model_manifest, prediction_manifest


def _metric_sums(
    arrays: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> tuple[float, float, float, int]:
    y = arrays["y"]
    scale = (
        arrays["scale"]
        if "scale" in arrays
        else np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
    )
    mse_sum = 0.0
    mae_sum = 0.0
    nmse_sum = 0.0
    count = int(y.size)
    for start in range(0, y.shape[0], METRIC_CHUNK_ROWS):
        stop = min(start + METRIC_CHUNK_ROWS, y.shape[0])
        error = np.asarray(prediction[start:stop], dtype=np.float64) - y[
            start:stop
        ].astype(np.float64)
        mse_sum += float(np.sum(error * error))
        mae_sum += float(np.sum(np.abs(error)))
        normalized = error / scale[start:stop]
        nmse_sum += float(np.sum(normalized * normalized))
    return mse_sum, mae_sum, nmse_sum, count


def evaluate_prediction(
    split: str,
    arrays: dict[str, np.ndarray],
    name: str,
    prediction: np.ndarray,
    *,
    vanilla_nmse: float,
) -> dict[str, Any]:
    mse_sum, mae_sum, nmse_sum, count = _metric_sums(arrays, prediction)
    nmse = nmse_sum / max(count, 1)
    return {
        "split": split,
        "baseline": name,
        "mse": mse_sum / max(count, 1),
        "mae": mae_sum / max(count, 1),
        "nmse": nmse,
        "relative_nmse_improvement_pct": (
            100.0 * (vanilla_nmse - nmse) / max(vanilla_nmse, 1e-12)
        ),
    }


def vanilla_nmse(arrays: dict[str, np.ndarray]) -> float:
    _, _, nmse_sum, count = _metric_sums(arrays, arrays["pred"])
    return nmse_sum / max(count, 1)


def write_metric_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    metrics_stem: str,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{metrics_stem}.csv"
    json_path = output_dir / f"{metrics_stem}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    for baseline, group in frame.groupby("baseline", sort=False):
        method_dir = output_dir / str(baseline)
        method_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(method_dir / f"{metrics_stem}.csv", index=False)
        (method_dir / f"{metrics_stem}.json").write_text(
            json.dumps(group.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    return csv_path, json_path


def _parse_float_grid(value: str) -> tuple[float, ...]:
    grid = tuple(
        float(item.strip())
        for item in value.replace(";", ",").split(",")
        if item.strip()
    )
    if not grid:
        raise ValueError("l2 grid cannot be empty")
    return grid


def _parse_methods(value: str | None, family: str) -> tuple[str, ...]:
    allowed = {
        "baselines": set(BASELINE_METHODS),
        "gates": set(GATE_METHODS),
    }[family]
    if value is None:
        return tuple(sorted(allowed))
    methods = tuple(
        item.strip()
        for item in value.replace(";", ",").split(",")
        if item.strip()
    )
    if not methods:
        raise ValueError("--methods cannot be empty")
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(
            f"methods not available for family={family}: {unknown}; "
            f"choose from {sorted(allowed)}"
        )
    return tuple(dict.fromkeys(methods))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--family",
        choices=("baselines", "gates"),
        default="baselines",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods to fit/score; defaults to every method in the family",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--l2-grid",
        default=",".join(str(value) for value in DEFAULT_L2_GRID),
    )
    parser.add_argument(
        "--fit-baselines-on-eval",
        action="store_true",
        help="Append explicitly optimistic T3 in-sample diagnostics",
    )
    parser.add_argument("--gate-iterations", type=int, default=300)
    parser.add_argument("--gate-learning-rate", type=float, default=3e-2)
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--gate-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--gate-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--gate-devices", default=None)
    parser.add_argument("--gate-thread-count", type=int, default=None)
    parser.add_argument("--feature-importance-top-k", type=int, default=20)
    parser.add_argument("--max-t1-fit-samples", type=int, default=None)
    parser.add_argument("--max-t2-valid-samples", type=int, default=None)
    parser.add_argument("--max-adapt-refit-samples", type=int, default=None)
    parser.add_argument("--max-eval-fit-samples", type=int, default=None)
    parser.add_argument("--fit-sample-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    started = perf_counter()
    selected_methods = _parse_methods(args.methods, args.family)
    selected_baselines = tuple(
        method for method in selected_methods if method in BASELINE_METHODS
    )
    selected_gates = tuple(
        method for method in selected_methods if method in GATE_METHODS
    )
    LOGGER.info("selected methods=%s", selected_methods)
    input_dir = Path(args.input_dir).expanduser()
    default_subdir = {"baselines": "baselines", "gates": "gates"}[args.family]
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_dir / default_subdir
    )
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output directory must differ from the extraction directory")
    payload_paths = [
        input_dir / f"{prefix}_prediction_payload.pt"
        for prefix in ("adapt", "eval")
    ]
    missing_payloads = [path for path in payload_paths if not path.is_file()]
    if missing_payloads:
        raise FileNotFoundError(f"missing extraction payloads: {missing_payloads}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_by_split: dict[str, dict[str, np.ndarray]] = {}
    for prefix, payload_path in zip(("adapt", "eval"), payload_paths, strict=True):
        payload = torch_load(payload_path)
        dense = flatten_payload(payload, prefix, family=args.family)
        del payload
        arrays_by_split[prefix] = (
            compact_gate_arrays(dense)
            if args.family == "gates"
            else compact_baseline_arrays(dense)
        )
        del dense
        gc.collect()
    t1_arrays, t2_arrays, resplit = chronological_resplit_arrays(
        arrays_by_split["adapt"],
        args.validation_fraction,
    )
    fit_seed = args.seed if args.fit_sample_seed is None else args.fit_sample_seed
    fit_arrays = {
        "T1": subsample_fit_arrays(
            t1_arrays,
            args.max_t1_fit_samples,
            seed=fit_seed,
        ),
        "T2": subsample_fit_arrays(
            t2_arrays,
            args.max_t2_valid_samples,
            seed=fit_seed + 1,
        ),
        "T1+T2": subsample_fit_arrays(
            arrays_by_split["adapt"],
            args.max_adapt_refit_samples,
            seed=fit_seed + 2,
        ),
        "T3_oracle": subsample_fit_arrays(
            arrays_by_split["eval"],
            args.max_eval_fit_samples,
            seed=fit_seed + 3,
        ),
    }
    fit_sampling = {
        name: {
            "available_samples": int(
                {
                    "T1": t1_arrays,
                    "T2": t2_arrays,
                    "T1+T2": arrays_by_split["adapt"],
                    "T3_oracle": arrays_by_split["eval"],
                }[name]["y"].shape[0]
            ),
            "used_samples": int(arrays["y"].shape[0]),
        }
        for name, arrays in fit_arrays.items()
    }
    l2_grid = _parse_float_grid(args.l2_grid)
    if args.family == "baselines":
        LOGGER.info("baseline selection start l2_grid=%s", l2_grid)
        rows, _, prediction_manifest = run_streamed_baselines(
            arrays_by_split,
            fit_arrays,
            output_dir=output_dir,
            selected_methods=selected_baselines,
            l2_grid=l2_grid,
            fit_on_eval=args.fit_baselines_on_eval,
        )
        LOGGER.info("baseline selection done")
    else:
        rows, _, prediction_manifest = run_streamed_gates(
            arrays_by_split,
            fit_arrays,
            output_dir=output_dir,
            selected_methods=selected_gates,
            iterations=args.gate_iterations,
            learning_rate=args.gate_learning_rate,
            depth=args.gate_depth,
            early_stopping_rounds=args.gate_early_stopping_rounds,
            seed=args.seed,
            task_type=args.gate_task_type,
            devices=args.gate_devices,
            thread_count=args.gate_thread_count,
            feature_importance_top_k=args.feature_importance_top_k,
        )
    frame = pd.DataFrame(rows)
    metrics_stem = "gate_metrics" if args.family == "gates" else "baseline_metrics"
    artifact_path = output_dir / (
        "gate_artifacts.json" if args.family == "gates" else "baseline_artifacts.pt"
    )
    csv_path, json_path = write_metric_outputs(frame, output_dir, metrics_stem)
    elapsed_seconds = perf_counter() - started
    timing_path = output_dir / (
        "gate_timing.json" if args.family == "gates" else "baseline_timing.json"
    )
    timing_path.write_text(
        json.dumps(
            {
                "family": args.family,
                "methods": list(selected_methods),
                "elapsed_seconds": elapsed_seconds,
                "adapt_samples": int(arrays_by_split["adapt"]["y"].shape[0]),
                "eval_samples": int(arrays_by_split["eval"]["y"].shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result_manifest = output_dir / "result_manifest.json"
    result_manifest.write_text(
        json.dumps(
            {
                "format": "adaptation_evaluation_result",
                "family": args.family,
                "methods": list(selected_methods),
                "split_protocol": resplit,
                "fit_sampling": fit_sampling,
                "files": {
                    "metrics_csv": csv_path.name,
                    "metrics_json": json_path.name,
                    "artifacts": artifact_path.name,
                    "predictions": prediction_manifest.name,
                    "timing": timing_path.name,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("experiment done seconds=%.2f", elapsed_seconds)
    log_experiment_separator(LOGGER)
    return {
        "csv": csv_path,
        "json": json_path,
        "artifacts": artifact_path,
        "predictions": prediction_manifest,
        "timing": timing_path,
        "manifest": result_manifest,
    }


if __name__ == "__main__":
    main()
