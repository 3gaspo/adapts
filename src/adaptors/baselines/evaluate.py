"""Fit and evaluate anchored ridge baselines and retrieval gates."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from src.data.neighbors import neighbor_to_query_scale
from src.experiments.runtime import log_experiment_separator, setup_logging
from src.experiments.splits import chronological_resplit_arrays


LOGGER = logging.getLogger(__name__)
RIDGE_CHUNK_ROWS = 65_536
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


def flatten_payload(payload: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    x = payload[f"{prefix}_X_values"].float()
    x_c = payload[f"{prefix}_Xc_values"].float()
    y_c_raw = payload[f"{prefix}_Yc_values"].float()
    e_raw = payload[f"{prefix}_E_values"].float()
    pred_neighbors_raw = y_c_raw - e_raw
    y_c = neighbor_to_query_scale(x, x_c, y_c_raw)
    e = neighbor_to_query_scale(x, x_c, e_raw, residual=True)
    pred_neighbors = neighbor_to_query_scale(x, x_c, pred_neighbors_raw)
    query_t = payload[f"{prefix}_query_t"]
    query_user = payload[f"{prefix}_query_user_idx"]
    neighbor_t = payload[f"{prefix}_neighbor_t"]
    neighbor_user = payload[f"{prefix}_neighbor_user_idx"]
    return {
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
        "e": rearrange(
            e,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy(),
        "pred_neighbors": rearrange(
            pred_neighbors,
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


def distance_weights(arrays: dict[str, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    d = arrays["distance"].astype(np.float64)
    d_std = d.std(axis=-1, keepdims=True)
    d_norm = (d - d.min(axis=-1, keepdims=True)) / np.maximum(d_std, eps)
    return softmax_np(-d_norm, axis=-1)


def weighted_neighbor_horizon(arrays: dict[str, np.ndarray]) -> np.ndarray:
    weights = distance_weights(arrays)
    return (weights[:, :, None] * arrays["y_c"]).sum(axis=1)


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
    scale = np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
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
    for mode in ("shared", "horizon"):
        train_lambda = _fit_convex_lambda(train, mode=mode)
        valid_prediction = _predict_convex_lambda(valid, train_lambda)
        final_lambda = _fit_convex_lambda(refit, mode=mode)
        name = f"aggr_y_mix_{mode}"
        artifacts["models"][name] = {
            "kind": "lambda",
            "mode": mode,
            "lambda": final_lambda,
            "t1_lambda": train_lambda,
            "t2_nmse": _normalized_mse(valid, valid_prediction),
            "constraint": "[0,1] by clipping the closed-form least-squares estimate",
        }
    for name, design, mode in RIDGE_MODELS:
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


def predict_baseline_adapters(
    arrays: dict[str, np.ndarray],
    artifacts: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {
        "vanilla": arrays["pred"],
        "context_forecast": arrays["pred_c"],
        "aggr_y": weighted_neighbor_horizon(arrays),
        "y_mean": arrays["y_c"].mean(axis=1),
    }
    for name, model in artifacts["models"].items():
        if model["kind"] == "lambda":
            predictions[name] = _predict_convex_lambda(
                arrays,
                model["lambda"],
            )
        else:
            predictions[name] = _predict_anchored_ridge(
                arrays,
                design=model["design"],
                mode=model["mode"],
                coefficients=model["coef"],
            )
    return predictions


def add_eval_fitted_baselines(
    predictions_by_split: dict[str, dict[str, np.ndarray]],
    eval_fit_arrays: dict[str, np.ndarray],
    *,
    eval_scoring_arrays: dict[str, np.ndarray] | None = None,
    l2_grid: Sequence[float] | float = DEFAULT_L2_GRID,
) -> dict[str, Any]:
    """Add explicitly optimistic T3 in-sample appendix diagnostics."""
    artifacts = fit_baseline_adapters(
        eval_fit_arrays,
        eval_fit_arrays,
        eval_fit_arrays,
        l2_grid,
    )
    scoring = eval_fit_arrays if eval_scoring_arrays is None else eval_scoring_arrays
    predictions = predict_baseline_adapters(scoring, artifacts)
    predictions_by_split["eval"].update(
        {
            f"{name}_eval_fit": predictions[name]
            for name in TRAINABLE_BASELINES
        }
    )
    return artifacts


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


def _static_gate_features(
    arrays: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    pred = arrays["pred"]
    weights = distance_weights(arrays)
    y_minus_v = arrays["y_c"] - pred[:, None, :]
    y_minus_n = arrays["y_c"] - arrays["pred_neighbors"]
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


def horizon_gate_features(
    arrays: dict[str, np.ndarray],
    candidate: str = "context",
) -> list[np.ndarray]:
    """Return one feature matrix per horizon; no model sees other C_h-V_h."""
    candidate_delta = _candidate_prediction(arrays, candidate) - arrays["pred"]
    static, local = _static_gate_features(arrays)
    common = np.stack(static, axis=1)
    return [
        np.column_stack(
            [
                candidate_delta[:, h],
                local["yv_mean"][:, h],
                local["yv_std"][:, h],
                local["yn_mean"][:, h],
                local["yn_std"][:, h],
                common,
            ]
        ).astype(np.float32)
        for h in range(candidate_delta.shape[1])
    ]


def _catboost_regressor(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
):
    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
    return CatBoostRegressor(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )


def _catboost_classifier(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
):
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
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
    )


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
) -> dict[str, Any]:
    target = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if np.ptp(target) <= 1e-12:
        return {"constant": float(target.mean()), "selected_iterations": 0}
    selected = int(iterations)
    if valid_x_np is not None and valid_y_np is not None:
        selector = _catboost_regressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed,
        )
        selector.fit(
            x_np,
            target,
            eval_set=(valid_x_np, np.asarray(valid_y_np).reshape(-1)),
            early_stopping_rounds=max(1, int(early_stopping_rounds)),
            use_best_model=True,
        )
        selected = _selected_iterations(selector, iterations)
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
        )
        selector.fit(
            x_np,
            target.astype(np.int8),
            eval_set=(valid_x_np, valid_target.astype(np.int8)),
            early_stopping_rounds=max(1, int(early_stopping_rounds)),
            use_best_model=True,
        )
        selected = _selected_iterations(selector, iterations)
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
) -> list[dict[str, Any]]:
    if x_np.shape[0] == 0:
        raise ValueError("cannot train gates from an empty T1 slice")
    targets = np.asarray(y_np)
    if targets.ndim == 1:
        targets = targets[:, None]
    valid_targets = None if valid_y_np is None else np.asarray(valid_y_np)
    if valid_targets is not None and valid_targets.ndim == 1:
        valid_targets = valid_targets[:, None]
    refit_targets = None if refit_y_np is None else np.asarray(refit_y_np)
    if refit_targets is not None and refit_targets.ndim == 1:
        refit_targets = refit_targets[:, None]
    fit_one = (
        fit_improvement_classifier
        if objective == "classifier"
        else fit_loss_difference_regressor
    )
    return [
        fit_one(
            x_np,
            targets[:, output_idx],
            valid_x_np=valid_x_np,
            valid_y_np=(
                None
                if valid_targets is None
                else valid_targets[:, output_idx]
            ),
            refit_x_np=refit_x_np,
            refit_y_np=(
                None
                if refit_targets is None
                else refit_targets[:, output_idx]
            ),
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed + output_idx,
            early_stopping_rounds=early_stopping_rounds,
        )
        for output_idx in range(targets.shape[1])
    ]


def predict_gate(
    models: list[dict[str, Any]],
    features: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    per_model_features = (
        list(features)
        if isinstance(features, (list, tuple))
        else [features] * len(models)
    )
    if len(per_model_features) != len(models):
        raise ValueError("one horizon feature matrix is required per gate model")
    columns = []
    for model, x_np in zip(models, per_model_features):
        if "constant" in model:
            score = np.full(x_np.shape[0], model["constant"], dtype=np.float64)
        elif "classifier" in model:
            score = model["classifier"].predict_proba(x_np)[:, 1] - 0.5
        else:
            score = model["regressor"].predict(x_np)
        columns.append(score)
    return np.column_stack(columns)


def _fit_no_feature_gates(arrays: dict[str, np.ndarray], candidate: str) -> dict[str, Any]:
    candidate_prediction = _candidate_prediction(arrays, candidate)
    base_loss = (arrays["y"] - arrays["pred"]) ** 2
    candidate_loss = (arrays["y"] - candidate_prediction) ** 2
    improvement = base_loss - candidate_loss
    return {
        "shared_score": float(improvement.mean()),
        "horizon_score": improvement.mean(axis=0).astype(np.float64),
    }


def _add_no_feature_gates(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    candidate: str,
    artifacts: dict[str, Any],
) -> dict[str, np.ndarray]:
    pred = arrays["pred"]
    candidate_prediction = _candidate_prediction(arrays, candidate)
    shared_score = float(artifacts["shared_score"])
    horizon_score = np.asarray(artifacts["horizon_score"])
    predictions[f"bayes_{candidate}_shared"] = (
        candidate_prediction if shared_score > 0.0 else pred
    )
    predictions[f"bayes_{candidate}_horizon"] = np.where(
        horizon_score[None, :] > 0.0,
        candidate_prediction,
        pred,
    )
    return {
        f"{candidate}_bayes_shared_score": np.full(
            pred.shape[0],
            shared_score,
        ),
        f"{candidate}_bayes_horizon_score": np.repeat(
            horizon_score[None, :],
            pred.shape[0],
            axis=0,
        ),
    }


def _add_true_oracles(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    candidate: str,
) -> None:
    pred = arrays["pred"]
    candidate_prediction = _candidate_prediction(arrays, candidate)
    target = arrays["y"]
    base_loss = (target - pred) ** 2
    candidate_loss = (target - candidate_prediction) ** 2
    shared = candidate_loss.mean(axis=1, keepdims=True) < base_loss.mean(
        axis=1,
        keepdims=True,
    )
    predictions[f"oracle_{candidate}_shared"] = np.where(
        shared,
        candidate_prediction,
        pred,
    )
    predictions[f"oracle_{candidate}_horizon"] = np.where(
        candidate_loss < base_loss,
        candidate_prediction,
        pred,
    )


def add_true_context_oracles(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> None:
    """Compatibility wrapper using the new shared/horizon naming."""
    _add_true_oracles(predictions, arrays, "context")


def add_candidate_gate_predictions(
    base_predictions_by_split: dict[str, dict[str, np.ndarray]],
    train_arrays: dict[str, np.ndarray],
    valid_arrays: dict[str, np.ndarray],
    refit_arrays: dict[str, np.ndarray],
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    *,
    candidate: str,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    def targets(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        candidate_prediction = _candidate_prediction(arrays, candidate)
        base_loss = (arrays["y"] - arrays["pred"]) ** 2
        candidate_loss = (arrays["y"] - candidate_prediction) ** 2
        improvement = base_loss - candidate_loss
        return {
            "shared": improvement.mean(axis=1, keepdims=True),
            "horizon": improvement,
        }

    train_targets = targets(train_arrays)
    valid_targets = targets(valid_arrays)
    refit_targets = targets(refit_arrays)
    train_features: dict[str, Any] = {
        "shared": scalar_gate_features(train_arrays, candidate),
        "horizon": horizon_gate_features(train_arrays, candidate),
    }
    valid_features: dict[str, Any] = {
        "shared": scalar_gate_features(valid_arrays, candidate),
        "horizon": horizon_gate_features(valid_arrays, candidate),
    }
    refit_features: dict[str, Any] = {
        "shared": scalar_gate_features(refit_arrays, candidate),
        "horizon": horizon_gate_features(refit_arrays, candidate),
    }
    models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for objective_index, objective in enumerate(("classifier", "regressor")):
        models[objective] = {}
        models[objective]["shared"] = fit_gate(
            train_features["shared"],
            train_targets["shared"],
            valid_x_np=valid_features["shared"],
            valid_y_np=valid_targets["shared"],
            refit_x_np=refit_features["shared"],
            refit_y_np=refit_targets["shared"],
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed + objective_index * 10_000,
            objective=objective,
        )
        horizon_models = []
        for h, (train_x, valid_x) in enumerate(
            zip(train_features["horizon"], valid_features["horizon"])
        ):
            horizon_models.extend(
                fit_gate(
                    train_x,
                    train_targets["horizon"][:, h],
                    valid_x_np=valid_x,
                    valid_y_np=valid_targets["horizon"][:, h],
                    refit_x_np=refit_features["horizon"][h],
                    refit_y_np=refit_targets["horizon"][:, h],
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    early_stopping_rounds=early_stopping_rounds,
                    seed=seed + objective_index * 10_000 + 1_000 + h,
                    objective=objective,
                )
            )
        models[objective]["horizon"] = horizon_models

    no_feature = _fit_no_feature_gates(refit_arrays, candidate)
    out: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for split, arrays in arrays_by_split.items():
        split_predictions = dict(base_predictions_by_split[split])
        candidate_prediction = _candidate_prediction(arrays, candidate)
        diagnostics[split] = _add_no_feature_gates(
            split_predictions,
            arrays,
            candidate,
            no_feature,
        )
        split_features = {
            "shared": scalar_gate_features(arrays, candidate),
            "horizon": horizon_gate_features(arrays, candidate),
        }
        split_targets = targets(arrays)
        for objective in ("classifier", "regressor"):
            for shape in ("shared", "horizon"):
                score = predict_gate(
                    models[objective][shape],
                    split_features[shape],
                )
                decision = score > 0.0
                if shape == "shared":
                    decision = decision[:, :1]
                name = f"catboost_{candidate}_{objective}_{shape}"
                split_predictions[name] = np.where(
                    decision,
                    candidate_prediction,
                    arrays["pred"],
                )
                diagnostics[split][f"{candidate}_{objective}_{shape}_score"] = (
                    score[:, 0] if shape == "shared" else score
                )
                diagnostics[split][f"{candidate}_{objective}_{shape}_target"] = (
                    split_targets[shape][:, 0]
                    if shape == "shared"
                    else split_targets[shape]
                )
        _add_true_oracles(split_predictions, arrays, candidate)
        out[split] = split_predictions
    artifacts = {
        "candidate": candidate,
        "backend": "catboost",
        "protocol": "fit_T1_validate_iterations_T2_refit_T1_plus_T2",
        "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
        "horizon_feature_names": HORIZON_GATE_FEATURE_NAMES,
        "models": models,
        "no_feature": no_feature,
    }
    return out, artifacts, diagnostics


def add_context_gate_predictions(
    base_predictions_by_split: dict[str, dict[str, np.ndarray]],
    train_arrays: dict[str, np.ndarray],
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    valid_arrays: dict[str, np.ndarray] | None = None,
    refit_arrays: dict[str, np.ndarray] | None = None,
    early_stopping_rounds: int = 50,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Compatibility wrapper for the context candidate."""
    return add_candidate_gate_predictions(
        base_predictions_by_split,
        train_arrays,
        train_arrays if valid_arrays is None else valid_arrays,
        train_arrays if refit_arrays is None else refit_arrays,
        arrays_by_split,
        candidate="context",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )


def visualization_payload(
    predictions_by_split: dict[str, dict[str, np.ndarray]],
    gate_diagnostics: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "description": "Predictions and gate diagnostics for pooled adaptation and held-out evaluation.",
        "splits": {
            split: {
                "predictions": {
                    name: torch.as_tensor(value, dtype=torch.float32)
                    for name, value in predictions.items()
                },
                "gate_diagnostics": {
                    name: torch.as_tensor(value, dtype=torch.float32)
                    for name, value in gate_diagnostics.get(split, {}).items()
                },
            }
            for split, predictions in predictions_by_split.items()
        },
    }


def _model_feature_importance(model: dict[str, Any]) -> np.ndarray | None:
    estimator = model.get("classifier") or model.get("regressor")
    if estimator is None or not hasattr(estimator, "get_feature_importance"):
        return None
    return np.asarray(estimator.get_feature_importance(), dtype=np.float64)


def _mean_feature_importance(
    models: list[dict[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray | None:
    values = [
        importance
        for model in models
        if (importance := _model_feature_importance(model)) is not None
        and importance.shape[0] == len(feature_names)
    ]
    return None if not values else np.mean(np.stack(values), axis=0)


def save_gate_feature_importance_plots(
    gate_artifacts: dict[str, Any],
    output_dir: Path,
    *,
    top_k: int = 20,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover
        LOGGER.warning("matplotlib unavailable; skipping feature importance")
        return []
    saved: list[Path] = []
    for candidate, candidate_artifacts in gate_artifacts.items():
        for objective, by_shape in candidate_artifacts["models"].items():
            for shape, names in (
                ("shared", SCALAR_GATE_FEATURE_NAMES),
                ("horizon", HORIZON_GATE_FEATURE_NAMES),
            ):
                importance = _mean_feature_importance(by_shape[shape], names)
                if importance is None:
                    continue
                order = np.argsort(importance)[::-1][
                    : max(1, min(int(top_k), len(importance)))
                ]
                selected_names = np.asarray(names, dtype=object)[order]
                selected_values = importance[order]
                stem = f"feature_importance_{candidate}_{objective}_{shape}"
                csv_path = output_dir / f"{stem}.csv"
                pd.DataFrame(
                    {"feature": selected_names, "importance": selected_values}
                ).to_csv(csv_path, index=False)
                saved.append(csv_path)
                fig, ax = plt.subplots(
                    figsize=(8, max(4, 0.32 * len(order)))
                )
                y = np.arange(len(order))
                ax.barh(y, selected_values[::-1])
                ax.set_yticks(y, labels=list(selected_names[::-1]))
                ax.set_xlabel("mean CatBoost feature importance")
                ax.set_title(f"{candidate} {objective} {shape} gate")
                ax.grid(True, axis="x", alpha=0.25)
                fig.tight_layout()
                png_path = output_dir / f"{stem}.png"
                fig.savefig(png_path, dpi=180)
                plt.close(fig)
                saved.append(png_path)
    return saved


def evaluate_predictions(
    split: str,
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    y = arrays["y"]
    scale = np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
    vanilla_nmse = np.mean(((arrays["pred"] - y) / scale) ** 2)
    for name, prediction in predictions.items():
        error = prediction - y
        nmse = np.mean((error / scale) ** 2)
        rows.append(
            {
                "split": split,
                "baseline": name,
                "mse": float(np.mean(error**2)),
                "mae": float(np.mean(np.abs(error))),
                "nmse": float(nmse),
                "relative_nmse_improvement_pct": float(
                    100.0
                    * (vanilla_nmse - nmse)
                    / max(vanilla_nmse, 1e-12)
                ),
            }
        )
    return rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--family", choices=("all", "baselines", "gates"), default="all")
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
    parser.add_argument("--gate-iterations", "--gate-epochs", dest="gate_iterations", type=int, default=300)
    parser.add_argument("--gate-learning-rate", "--gate-lr", dest="gate_learning_rate", type=float, default=3e-2)
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--gate-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--feature-importance-top-k", type=int, default=20)
    parser.add_argument("--max-t1-fit-samples", "--max-train-fit-samples", dest="max_t1_fit_samples", type=int, default=None)
    parser.add_argument("--max-t2-valid-samples", "--max-oracle-fit-samples", dest="max_t2_valid_samples", type=int, default=None)
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
    input_dir = Path(args.input_dir).expanduser()
    default_subdir = {
        "all": "baseline_adapters",
        "baselines": "baselines",
        "gates": "gates",
    }[args.family]
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_dir / default_subdir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_by_split = {
        prefix: flatten_payload(
            torch_load(input_dir / f"{prefix}_prediction_payload.pt"),
            prefix,
        )
        for prefix in ("adapt", "eval")
    }
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
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in arrays_by_split
    }
    baseline_artifacts = None
    eval_fit_artifacts = None
    if args.family in {"all", "baselines"}:
        LOGGER.info("baseline selection start l2_grid=%s", l2_grid)
        baseline_artifacts = fit_baseline_adapters(
            fit_arrays["T1"],
            fit_arrays["T2"],
            fit_arrays["T1+T2"],
            l2_grid,
        )
        predictions_by_split = {
            split: predict_baseline_adapters(arrays, baseline_artifacts)
            for split, arrays in arrays_by_split.items()
        }
        if args.fit_baselines_on_eval:
            eval_fit_artifacts = add_eval_fitted_baselines(
                predictions_by_split,
                fit_arrays["T3_oracle"],
                eval_scoring_arrays=arrays_by_split["eval"],
                l2_grid=l2_grid,
            )
        LOGGER.info("baseline selection done")
    else:
        predictions_by_split = {
            split: {
                "vanilla": arrays["pred"],
                "context_forecast": arrays["pred_c"],
                "aggr_y": weighted_neighbor_horizon(arrays),
            }
            for split, arrays in arrays_by_split.items()
        }

    gate_artifacts: dict[str, Any] = {}
    gate_diagnostics = {split: {} for split in arrays_by_split}
    if args.family in {"all", "gates"}:
        for candidate_index, candidate in enumerate(("context", "aggr_y")):
            LOGGER.info("gate fitting start candidate=%s", candidate)
            candidate_predictions, artifacts, diagnostics = add_candidate_gate_predictions(
                predictions_by_split,
                fit_arrays["T1"],
                fit_arrays["T2"],
                fit_arrays["T1+T2"],
                arrays_by_split,
                candidate=candidate,
                iterations=args.gate_iterations,
                learning_rate=args.gate_learning_rate,
                depth=args.gate_depth,
                early_stopping_rounds=args.gate_early_stopping_rounds,
                seed=args.seed + candidate_index * 100_000,
            )
            predictions_by_split = candidate_predictions
            gate_artifacts[candidate] = artifacts
            for split in gate_diagnostics:
                gate_diagnostics[split].update(diagnostics[split])
            LOGGER.info("gate fitting done candidate=%s", candidate)

    rows = evaluate_predictions(
        "eval",
        arrays_by_split["eval"],
        predictions_by_split["eval"],
    )
    frame = pd.DataFrame(rows)
    metrics_stem = "gate_metrics" if args.family == "gates" else "baseline_metrics"
    artifact_path = output_dir / (
        "gate_artifacts.pt" if args.family == "gates" else "baseline_artifacts.pt"
    )
    visualization_path = output_dir / "visualization_payload.pt"
    csv_path, json_path = write_metric_outputs(frame, output_dir, metrics_stem)
    saved_artifacts: dict[str, Any] = {
        "family": args.family,
        "split_protocol": resplit,
        "fit_sampling": fit_sampling,
    }
    if baseline_artifacts is not None:
        saved_artifacts["baseline_artifacts"] = baseline_artifacts
    if eval_fit_artifacts is not None:
        saved_artifacts["eval_fit_baseline_artifacts"] = eval_fit_artifacts
    if gate_artifacts:
        saved_artifacts["gate_artifacts"] = gate_artifacts
        saved_artifacts["gate_config"] = {
            "candidates": ["context", "aggr_y"],
            "protocol": "T1 fit, T2 early stopping, T1+T2 refit, T3 score",
            "iterations": args.gate_iterations,
            "learning_rate": args.gate_learning_rate,
            "depth": args.gate_depth,
            "early_stopping_rounds": args.gate_early_stopping_rounds,
            "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
            "horizon_feature_names": HORIZON_GATE_FEATURE_NAMES,
        }
        importance = save_gate_feature_importance_plots(
            gate_artifacts,
            output_dir / "plots",
            top_k=args.feature_importance_top_k,
        )
        LOGGER.info("feature-importance outputs=%s", len(importance))
    torch.save(saved_artifacts, artifact_path)
    torch.save(
        visualization_payload(predictions_by_split, gate_diagnostics),
        visualization_path,
    )
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - started)
    log_experiment_separator(LOGGER)
    return {
        "csv": csv_path,
        "json": json_path,
        "artifacts": artifact_path,
        "visualization": visualization_path,
    }


if __name__ == "__main__":
    main()
