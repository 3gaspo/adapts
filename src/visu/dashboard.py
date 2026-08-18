"""Artifact-only helpers for the extraction and adaptation dashboards."""

from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
import re
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.artifacts import validate_extraction
from src.experiments.prediction_store import load_prediction_store


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(Path(path), map_location="cpu")


def _flatten(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().reshape(-1, *value.shape[2:]).numpy()


def _flatten_optional(payload: dict[str, Any], key: str) -> np.ndarray | None:
    value = payload.get(key)
    if value is None or not torch.is_tensor(value):
        return None
    return _flatten(value)


def _load_extraction(extraction_dir: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = Path(extraction_dir).expanduser()
    extraction_complete, extraction_reason = validate_extraction(root)
    if not extraction_complete:
        raise ValueError(f"Extraction is not complete: {extraction_reason}")
    extracted: dict[str, dict[str, Any]] = {}
    for split in ("adapt", "eval"):
        path = root / f"{split}_prediction_payload.pt"
        if path.exists():
            extracted[split] = torch_load(path)
    if not extracted:
        raise FileNotFoundError(f"No *_prediction_payload.pt files found under {root}")
    return root, extracted


def _empty_dashboard_data(root: Path, extracted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_dir": root,
        "result_dirs": [],
        "extracted": extracted,
        "baseline": {"splits": {}},
        "baseline_artifacts": {"models": {}},
        "gate_importances": {},
        "ts_ifa_artifacts": {},
        "paths": {
            "extraction_manifest": root / "extraction_manifest.json",
            "results": [],
        },
    }


def _validate_dashboard_alignment(data: dict[str, Any]) -> dict[str, Any]:
    for split in data["extracted"]:
        split_arrays(data, split)
    return data


def load_extraction_dashboard_data(extraction_dir: str | Path) -> dict[str, Any]:
    """Load one completed extraction run without adaptation results."""
    root, extracted = _load_extraction(extraction_dir)
    return _validate_dashboard_alignment(_empty_dashboard_data(root, extracted))


def _merge_array(
    destination: dict[str, np.ndarray],
    owners: dict[str, Path],
    name: str,
    value: Any,
    source: Path,
) -> None:
    array = np.asarray(value)
    if name not in destination:
        destination[name] = array
        owners[name] = source
        return
    current = destination[name]
    if current.shape == array.shape and np.array_equal(current, array, equal_nan=True):
        return
    raise ValueError(
        f"Conflicting dashboard array {name!r} from {owners[name]} and {source}; "
        "select result paths for the same extraction and only one run per named model"
    )


def load_adaptation_dashboard_data(
    extraction_dir: str | Path,
    result_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Load one extraction and an explicit list of completed adaptation runs."""
    root, extracted = _load_extraction(extraction_dir)
    data = _empty_dashboard_data(root, extracted)
    baseline = data["baseline"]
    baseline_artifacts = data["baseline_artifacts"]
    gate_importances = data["gate_importances"]
    ts_ifa_artifacts = data["ts_ifa_artifacts"]
    prediction_owners: dict[str, dict[str, Path]] = {}
    diagnostic_owners: dict[str, dict[str, Path]] = {}

    for selected_dir in result_dirs:
        current_dir = Path(selected_dir).expanduser()
        result_manifest = current_dir / "result_manifest.json"
        if not result_manifest.exists():
            raise FileNotFoundError(f"Missing selected result manifest: {result_manifest}")
        completion = json.loads(result_manifest.read_text(encoding="utf-8"))
        result_format = completion.get("format")
        expected_family = completion.get("family")
        if result_format not in {
            "adaptation_evaluation_result",
            "adaptation_ts_ifa_result",
        }:
            raise ValueError(f"{result_manifest} is not a current result manifest")
        if result_format == "adaptation_evaluation_result" and expected_family not in {"baselines", "gates"}:
            raise ValueError(f"{result_manifest} has the wrong result family")
        if completion.get("files", {}).get("predictions") != "prediction_manifest.json":
            raise ValueError(f"{result_manifest} does not index current predictions")
        payload = load_prediction_store(current_dir)
        result_label = str(
            completion.get("method")
            or (completion.get("methods") or [None])[0]
            or completion.get("variant")
            or current_dir.parent.name
        )
        data["result_dirs"].append(current_dir)
        data["paths"]["results"].append(current_dir / "prediction_manifest.json")
        for split, split_payload in payload["splits"].items():
            merged = baseline["splits"].setdefault(
                split, {"predictions": {}, "gate_diagnostics": {}}
            )
            split_prediction_owners = prediction_owners.setdefault(split, {})
            split_diagnostic_owners = diagnostic_owners.setdefault(split, {})
            for name, value in split_payload.get("predictions", {}).items():
                display_name = (
                    result_label
                    if result_format == "adaptation_ts_ifa_result" and name == "ts_ifa_adapted"
                    else name
                )
                _merge_array(
                    merged["predictions"],
                    split_prediction_owners,
                    display_name,
                    value,
                    current_dir,
                )
            for name, value in split_payload.get("gate_diagnostics", {}).items():
                if result_format == "adaptation_ts_ifa_result" and name == "rooter_coefficients":
                    continue
                _merge_array(
                    merged["gate_diagnostics"],
                    split_diagnostic_owners,
                    name,
                    value,
                    current_dir,
                )
        files = completion.get("files", {})
        if expected_family == "baselines":
            artifact_path = current_dir / str(files.get("artifacts", ""))
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing baseline artifacts: {artifact_path}")
            current_artifacts = torch_load(artifact_path)
            if current_artifacts.get("format") != "adaptation_baseline_models":
                raise ValueError(f"{artifact_path} is not a current baseline artifact")
            current_models = current_artifacts.get("models", {})
            overlap = set(baseline_artifacts["models"]) & set(current_models)
            if overlap:
                raise ValueError(f"Duplicate baseline models {sorted(overlap)} from {current_dir}")
            baseline_artifacts["models"].update(current_models)
            baseline_artifacts.setdefault("eval_fit_models", {}).update(
                current_artifacts.get("eval_fit_models") or {}
            )
        elif expected_family == "gates":
            artifact_path = current_dir / str(files.get("artifacts", ""))
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing gate artifacts: {artifact_path}")
            gate_artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
            if gate_artifacts.get("format") != "adaptation_gate_models":
                raise ValueError(f"{artifact_path} is not a current gate artifact")
            for relative in gate_artifacts.get("feature_importance_files", []):
                importance_path = current_dir / str(relative)
                if importance_path.suffix.lower() != ".csv":
                    continue
                method_suffix = importance_path.stem.removeprefix("feature_importance_")
                with importance_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                gate_name = f"catboost_{method_suffix}"
                current_importance = {
                    "feature": np.asarray([row["feature"] for row in rows], dtype=object),
                    "importance": np.asarray(
                        [float(row["importance"]) for row in rows],
                        dtype=np.float64,
                    ),
                }
                if gate_name in gate_importances:
                    previous = gate_importances[gate_name]
                    identical = (
                        np.array_equal(previous["feature"], current_importance["feature"])
                        and np.array_equal(
                            previous["importance"],
                            current_importance["importance"],
                            equal_nan=True,
                        )
                    )
                    if not identical:
                        raise ValueError(
                            f"Conflicting gate importance {gate_name!r} from {current_dir}"
                        )
                else:
                    gate_importances[gate_name] = current_importance
        elif result_format == "adaptation_ts_ifa_result":
            if result_label in ts_ifa_artifacts:
                raise ValueError(f"Duplicate TS-IFA model label {result_label!r} from {current_dir}")
            rooter_path = current_dir / str(files.get("rooter", ""))
            if not rooter_path.is_file():
                raise FileNotFoundError(f"Missing TS-IFA rooter: {rooter_path}")
            rooter_payload = torch_load(rooter_path)
            current_ts_ifa = {
                "candidate_names": list(
                    rooter_payload.get(
                        "candidate_names",
                        payload.get("metadata", {}).get("candidate_names", []),
                    )
                ),
                "variant": completion.get("variant"),
            }
            if "coefficients" in rooter_payload:
                current_ts_ifa["ridge_rooter_coefficients"] = np.asarray(
                    rooter_payload["coefficients"].detach().cpu(), dtype=np.float64
                )
            eval_diagnostics = payload.get("splits", {}).get("eval", {}).get("gate_diagnostics", {})
            if "rooter_coefficients" in eval_diagnostics:
                current_ts_ifa["active_rooter_coefficients"] = np.asarray(
                    eval_diagnostics["rooter_coefficients"], dtype=np.float64
                )
            ts_ifa_artifacts[result_label] = current_ts_ifa

    return _validate_dashboard_alignment(data)


def available_splits(data: dict[str, Any]) -> list[str]:
    return [name for name in ("adapt", "eval") if name in data["extracted"]]


def split_arrays(data: dict[str, Any], split: str) -> dict[str, Any]:
    payload = data["extracted"][split]
    prefix = f"{split}_"
    x_tensor = payload[prefix + "X_values"].float()
    y_tensor = payload[prefix + "Y_values"].float()
    x_c_tensor = payload[prefix + "Xc_values"].float()
    y_c_tensor = payload[prefix + "Yc_values"].float()
    e_tensor = payload.get(prefix + "E_values")
    dates, users = x_tensor.shape[:2]
    arrays: dict[str, Any] = {
        "x": _flatten(x_tensor),
        "y": _flatten(y_tensor),
        "x_c": _flatten(x_c_tensor),
        "y_c": _flatten(y_c_tensor),
        "e": _flatten(e_tensor.float()) if torch.is_tensor(e_tensor) else None,
        "query_t": payload[prefix + "query_t"].reshape(-1).numpy(),
        "query_user_idx": payload[prefix + "query_user_idx"].reshape(-1).numpy(),
        "neighbor_t": _flatten(payload[prefix + "neighbor_t"]),
        "neighbor_user_idx": _flatten(payload[prefix + "neighbor_user_idx"]),
        "distance": _flatten_optional(payload, prefix + "distance_x_xc"),
        "dates": dates,
        "users": users,
        "datetimes": payload.get(prefix + "datetimes", []),
    }
    predictions = {
        "vanilla": _flatten(payload[prefix + "preds"]),
        "cov_forecast": _flatten(payload[prefix + "preds_context"]),
    }
    baseline_split = data["baseline"].get("splits", {}).get(split, {})
    for name, value in baseline_split.get("predictions", {}).items():
        predictions[name] = np.asarray(value)
    n_samples = dates * users
    invalid = {name: value.shape for name, value in predictions.items() if value.shape[0] != n_samples}
    if invalid:
        raise ValueError(f"Prediction alignment mismatch for {split}: {invalid}; expected {n_samples} samples")
    arrays["predictions"] = predictions
    arrays["gate_diagnostics"] = {
        name: np.asarray(value)
        for name, value in baseline_split.get("gate_diagnostics", {}).items()
    }
    return arrays


def _normalize_pair(lookback: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(lookback))
    std = max(float(np.std(lookback)), 1e-8)
    return (lookback - mean) / std, (future - mean) / std


def plot_query_example(
    data: dict[str, Any],
    split: str,
    sample_index: int,
    *,
    instance_normalized: bool,
    hide_axes: bool,
) -> plt.Figure:
    arrays = split_arrays(data, split)
    n_samples = len(arrays["x"])
    if not 0 <= sample_index < n_samples:
        raise IndexError(f"sample_index must be in [0, {n_samples})")
    x = arrays["x"][sample_index].copy()
    y = arrays["y"][sample_index].copy()
    x_c = arrays["x_c"][sample_index].copy()
    y_c = arrays["y_c"][sample_index].copy()
    if instance_normalized:
        x, y = _normalize_pair(x, y)
        for neighbor in range(len(x_c)):
            x_c[neighbor], y_c[neighbor] = _normalize_pair(x_c[neighbor], y_c[neighbor])

    lags, horizon = len(x), len(y)
    past_axis = np.arange(-lags, 0)
    future_axis = np.arange(horizon)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, max(len(x_c), 1)))
    for neighbor, color in enumerate(colors):
        label = (
            f"neighbor {neighbor + 1} "
            f"(user {int(arrays['neighbor_user_idx'][sample_index, neighbor])}, "
            f"t={int(arrays['neighbor_t'][sample_index, neighbor])})"
        )
        ax.plot(past_axis, x_c[neighbor], color=color, alpha=0.72, linewidth=1.2, label=label)
        ax.plot(
            np.r_[past_axis[-1], future_axis],
            np.r_[x_c[neighbor, -1], y_c[neighbor]],
            color=color,
            alpha=0.72,
            linewidth=1.2,
            linestyle="--",
        )
    ax.plot(past_axis, x, color="black", linewidth=2.6, label="query lookback")
    ax.plot(
        np.r_[past_axis[-1], future_axis],
        np.r_[x[-1], y],
        color="black",
        linewidth=2.6,
        linestyle="--",
        label="query future",
    )
    ax.axvline(-0.5, color="0.45", linewidth=1, linestyle=":")
    ax.set_xlabel("Time relative to forecast origin")
    ax.set_ylabel("Instance-normalized value" if instance_normalized else "Value")
    query_t = int(arrays["query_t"][sample_index])
    user = int(arrays["query_user_idx"][sample_index])
    ax.set_title(f"{split}: query user {user}, t={query_t}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    if hide_axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def prediction_names(data: dict[str, Any], split: str) -> list[str]:
    return sorted(split_arrays(data, split)["predictions"])


SCALAR_FEATURE_ORDER = (
    "cov_minus_vanilla_mean",
    "cov_minus_vanilla_std",
    "weighted_neighbor_minus_vanilla_mean",
    "weighted_neighbor_residual_mean",
    "query_mean",
    "query_std",
    "neighbor_lookback_means_mean_raw",
    "neighbor_lookback_means_std_raw",
    "neighbor_lookback_stds_mean_raw",
    "neighbor_lookback_stds_std_raw",
    "same_user_ratio",
    "neighbor_age_mean",
    "neighbor_age_std",
    "neighbor_weight_std",
    "neighbor_weight_max",
    "distance_mean",
)


def _distance_weights(distance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    shifted = -np.asarray(distance, dtype=np.float64)
    shifted = shifted - np.nanmax(shifted, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.nansum(exp, axis=1, keepdims=True), eps)


def _neighbor_to_query_scale_np(
    query_lookback: np.ndarray,
    neighbor_lookback: np.ndarray,
    neighbor_value: np.ndarray,
    *,
    residual: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    query_mean = query_lookback.mean(axis=-1, keepdims=True)[:, None, :]
    query_std = np.maximum(query_lookback.std(axis=-1, keepdims=True), eps)[:, None, :]
    neighbor_mean = neighbor_lookback.mean(axis=-1, keepdims=True)
    neighbor_std = np.maximum(neighbor_lookback.std(axis=-1, keepdims=True), eps)
    if residual:
        return neighbor_value / neighbor_std * query_std
    return (neighbor_value - neighbor_mean) / neighbor_std * query_std + query_mean


def scalar_feature_values(data: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    x = np.asarray(arrays["x"], dtype=np.float64)
    x_c = np.asarray(arrays["x_c"], dtype=np.float64)
    pred = np.asarray(predictions["vanilla"], dtype=np.float64)
    pred_c = np.asarray(predictions["cov_forecast"], dtype=np.float64)
    cov_delta = pred_c - pred
    features: dict[str, np.ndarray] = {
        "cov_minus_vanilla_mean": np.nanmean(cov_delta, axis=1),
        "cov_minus_vanilla_std": np.nanstd(cov_delta, axis=1),
        "query_mean": np.nanmean(x, axis=1),
        "query_std": np.nanstd(x, axis=1),
        "neighbor_lookback_means_mean_raw": np.nanmean(np.nanmean(x_c, axis=-1), axis=1),
        "neighbor_lookback_means_std_raw": np.nanstd(np.nanmean(x_c, axis=-1), axis=1),
        "neighbor_lookback_stds_mean_raw": np.nanmean(np.nanstd(x_c, axis=-1), axis=1),
        "neighbor_lookback_stds_std_raw": np.nanstd(np.nanstd(x_c, axis=-1), axis=1),
        "same_user_ratio": np.nanmean(
            arrays["neighbor_user_idx"] == arrays["query_user_idx"][:, None],
            axis=1,
        ),
        "neighbor_age_mean": np.nanmean(
            arrays["query_t"][:, None] - arrays["neighbor_t"],
            axis=1,
        ),
        "neighbor_age_std": np.nanstd(
            arrays["query_t"][:, None] - arrays["neighbor_t"],
            axis=1,
        ),
    }

    distance = arrays.get("distance")
    if distance is not None:
        distance = np.asarray(distance, dtype=np.float64)
        weights = _distance_weights(distance)
        features.update(
            {
                "neighbor_weight_std": np.nanstd(weights, axis=1),
                "neighbor_weight_max": np.nanmax(weights, axis=1),
                "distance_mean": np.nanmean(distance, axis=1),
            }
        )
        if arrays.get("e") is not None:
            y_c_scaled = _neighbor_to_query_scale_np(x, x_c, np.asarray(arrays["y_c"], dtype=np.float64))
            e_scaled = _neighbor_to_query_scale_np(
                x,
                x_c,
                np.asarray(arrays["e"], dtype=np.float64),
                residual=True,
            )
            weighted = np.nansum(weights[:, :, None] * y_c_scaled, axis=1)
            weighted_e = np.nansum(weights[:, :, None] * e_scaled, axis=1)
            features.update(
                {
                    "weighted_neighbor_minus_vanilla_mean": np.nanmean(weighted - pred, axis=1),
                    "weighted_neighbor_residual_mean": np.nanmean(weighted_e, axis=1),
                }
            )

    return {name: features[name] for name in SCALAR_FEATURE_ORDER if name in features}


def scalar_feature_names(data: dict[str, Any], split: str) -> list[str]:
    return list(scalar_feature_values(data, split))


def _prediction_metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    lookback: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, str]:
    if metric == "difference":
        return prediction - target, "Difference"
    if metric == "mse":
        return (prediction - target) ** 2, "MSE"
    if metric == "nmse":
        scale = np.maximum(lookback.std(axis=1, keepdims=True), 1e-8)
        return ((prediction - target) / scale) ** 2, "nMSE"
    raise ValueError(f"Unknown metric: {metric}")


def _safe_relative_delta(selected: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(reference) > 1e-12, reference, np.nan)
    return (selected - reference) / denominator


def _format_metric_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e4:
        return f"{value:.4g}"
    return f"{value:.3e}"


def _symlog_linthresh(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    return max(float(np.nanpercentile(finite, 10)), 1e-8) if finite.size else 1.0


def horizon_values(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> tuple[np.ndarray, str, float, float | None]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    selected = np.asarray(predictions[prediction_name], dtype=np.float64)
    reference = np.asarray(predictions[reference_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)

    selected_metric, metric_label = _prediction_metric_values(selected, target, x, metric)
    reference_metric, _ = _prediction_metric_values(reference, target, x, metric)
    selected_horizon = np.nanmean(selected_metric, axis=0)
    reference_horizon = np.nanmean(reference_metric, axis=0)
    if view == "direct":
        values = selected_horizon
        ylabel = metric_label
    elif view == "improvement":
        values = selected_horizon - reference_horizon
        ylabel = f"{metric_label} delta vs {reference_name}"
    elif view == "relative":
        values = _safe_relative_delta(selected_horizon, reference_horizon)
        ylabel = f"Relative {metric_label} delta vs {reference_name}"
    else:
        raise ValueError(f"Unknown view: {view}")

    average = float(np.nanmean(values))
    window_average = None
    if view == "relative":
        selected_window = np.nanmean(selected_metric, axis=1)
        reference_window = np.nanmean(reference_metric, axis=1)
        window_average = float(np.nanmean(_safe_relative_delta(selected_window, reference_window)))
    return values, ylabel, average, window_average


def window_metric_values(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> tuple[np.ndarray, str, float]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    selected = np.asarray(predictions[prediction_name], dtype=np.float64)
    reference = np.asarray(predictions[reference_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)

    selected_metric, metric_label = _prediction_metric_values(selected, target, x, metric)
    reference_metric, _ = _prediction_metric_values(reference, target, x, metric)
    selected_window = np.nanmean(selected_metric, axis=1)
    reference_window = np.nanmean(reference_metric, axis=1)
    if view == "direct":
        values = selected_window
        ylabel = metric_label
    elif view == "improvement":
        values = selected_window - reference_window
        ylabel = f"{metric_label} delta vs {reference_name}"
    elif view == "relative":
        values = _safe_relative_delta(selected_window, reference_window)
        ylabel = f"Relative {metric_label} delta vs {reference_name}"
    else:
        raise ValueError(f"Unknown view: {view}")
    return values, ylabel, float(np.nanmean(values))


def plot_window_metric_scatter(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
    scalar_feature_name: str,
    *,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    max_points: int = 5000,
) -> plt.Figure:
    feature_map = scalar_feature_values(data, split)
    if scalar_feature_name not in feature_map:
        raise KeyError(f"Unknown scalar feature {scalar_feature_name!r}")
    x_values = np.asarray(feature_map[scalar_feature_name], dtype=np.float64)
    y_values, ylabel, average = window_metric_values(
        data,
        split,
        prediction_name,
        reference_name,
        metric,
        view,
    )
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(x_values), size=max_points, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
    correlation = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if len(x_values) > 1 and np.std(x_values) > 0.0 and np.std(y_values) > 0.0
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=10, alpha=0.35, linewidths=0)
    if view != "direct" or metric == "difference":
        ax.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel(scalar_feature_name)
    ax.set_ylabel(ylabel)
    if x_log_scale:
        ax.set_xscale("symlog", linthresh=_symlog_linthresh(x_values))
    if y_log_scale:
        ax.set_yscale("symlog", linthresh=_symlog_linthresh(y_values))
    title = f"{split}: {prediction_name} - {metric} ({view})"
    if view != "direct":
        title += f" vs {reference_name}"
    title += (
        f"\navg={_format_metric_value(average)}; "
        f"r={_format_metric_value(correlation)}; n={len(x_values):,}"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_horizon(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> plt.Figure:
    values, ylabel, average, window_average = horizon_values(
        data,
        split,
        prediction_name,
        reference_name,
        metric,
        view,
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=2.2)
    if view != "direct" or metric == "difference":
        ax.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    title = f"{split}: {prediction_name} - {metric} ({view})"
    if view != "direct":
        title += f" vs {reference_name}"
    if view == "relative":
        title += (
            f"\nhorizon avg={_format_metric_value(average)}; "
            f"window avg={_format_metric_value(float(window_average))}"
        )
    else:
        title += f"\navg={_format_metric_value(average)}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _relative_improvement_pct(reference: float, current: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(current) or abs(reference) <= 1e-12:
        return float("nan")
    return float(100.0 * (reference - current) / reference)


def _gate_candidate_prediction_name(gate_name: str) -> str:
    match = re.fullmatch(
        r"(?:bayes|catboost|oracle)_(cov|avgy)(?:_[a-z]+)*_(?:shared|horizon)(?:_soft)?",
        gate_name,
    )
    if match is None:
        raise ValueError(f"Cannot identify the gated candidate from {gate_name!r}")
    return "cov_forecast" if match.group(1) == "cov" else "avgy"


def gate_options(data: dict[str, Any], split: str) -> list[tuple[str, str]]:
    diagnostics = split_arrays(data, split)["gate_diagnostics"]
    options: list[tuple[str, str]] = []
    for key in sorted(diagnostics):
        match = re.fullmatch(
            r"(catboost_(cov|avgy)_(classifier|regressor)_(shared|horizon)(?:_soft)?)_(score|probability)",
            key,
        )
        if match is None:
            continue
        stem, candidate, _, shape, _ = match.groups()
        if f"{candidate}_{shape}_target" not in diagnostics:
            continue
        label = stem.replace("_", " ")
        if shape == "horizon":
            label += " (all horizons)"
        options.append((label, stem))
    return options


def _gate_score_target(data: dict[str, Any], split: str, gate_name: str) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = split_arrays(data, split)["gate_diagnostics"]
    match = re.fullmatch(
        r"catboost_(cov|avgy)_(classifier|regressor)_(shared|horizon)(?:(_soft))?",
        gate_name,
    )
    if match is None:
        raise ValueError(gate_name)
    candidate, _, shape, soft = match.groups()
    diagnostic_name = "probability" if soft else "score"
    score = np.asarray(
        diagnostics[f"{gate_name}_{diagnostic_name}"],
        dtype=np.float64,
    )
    if soft:
        score = score - 0.5
    target = diagnostics[f"{candidate}_{shape}_target"]
    return score, np.asarray(target, dtype=np.float64)


def gate_roc(
    data: dict[str, Any],
    split: str,
    gate_name: str,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    score, target = _gate_score_target(data, split, gate_name)
    score = score.reshape(-1)
    target = target.reshape(-1)
    finite = np.isfinite(score) & np.isfinite(target)
    score = np.asarray(score[finite], dtype=np.float64)
    label = np.asarray(target[finite] > 0.0, dtype=bool)
    accuracy = float(np.mean((score > 0.0) == label)) if len(label) else float("nan")
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan"), accuracy, len(label)
    order = np.argsort(-score, kind="stable")
    sorted_label = label[order]
    threshold_indices = np.r_[np.where(np.diff(score[order]))[0], len(score) - 1]
    tp = np.cumsum(sorted_label)[threshold_indices]
    fp = np.cumsum(~sorted_label)[threshold_indices]
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))
    return fpr, tpr, auc, accuracy, len(label)


def plot_gate_roc(data: dict[str, Any], split: str, gate_name: str) -> tuple[plt.Figure, dict[str, float]]:
    fpr, tpr, auc, accuracy, count = gate_roc(data, split, gate_name)
    fig, ax = plt.subplots(figsize=(6, 5))
    label = f"ROC (AUC={auc:.3f})" if np.isfinite(auc) else "ROC (one class only)"
    ax.plot(fpr, tpr, linewidth=2.2, label=label)
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{split}: {gate_name}; accuracy={accuracy:.3f}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, {"auc": auc, "accuracy": accuracy, "count": float(count)}


def gate_prediction_names(data: dict[str, Any], split: str) -> list[str]:
    names = prediction_names(data, split)
    prefixes = (
        "bayes_cov_",
        "catboost_cov_",
        "oracle_cov_",
        "bayes_avgy_",
        "catboost_avgy_",
        "oracle_avgy_",
    )
    return [name for name in names if name.startswith(prefixes)]


def _gate_shape(name: str) -> str:
    return "horizon" if name.endswith("_horizon") else "shared"


def _gate_right_percent(arrays: dict[str, Any], prediction_name: str) -> float:
    predictions = arrays["predictions"]
    prediction = np.asarray(predictions[prediction_name], dtype=np.float64)
    vanilla = np.asarray(predictions["vanilla"], dtype=np.float64)
    candidate_name = "avgy" if "_avgy_" in prediction_name else "cov_forecast"
    candidate = np.asarray(predictions[candidate_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    base_loss = (vanilla - target) ** 2
    candidate_loss = (candidate - target) ** 2
    distance_to_candidate = np.abs(prediction - candidate)
    distance_to_vanilla = np.abs(prediction - vanilla)
    if _gate_shape(prediction_name) == "shared":
        decision = np.nanmean(distance_to_candidate, axis=1) <= np.nanmean(distance_to_vanilla, axis=1)
        target_candidate = np.nanmean(candidate_loss, axis=1) < np.nanmean(base_loss, axis=1)
        non_tie = np.abs(np.nanmean(base_loss - candidate_loss, axis=1)) > 1e-12
    else:
        decision = distance_to_candidate <= distance_to_vanilla
        target_candidate = candidate_loss < base_loss
        non_tie = np.abs(base_loss - candidate_loss) > 1e-12
    finite = np.isfinite(decision) & np.isfinite(target_candidate) & non_tie
    return float(100.0 * np.mean(decision[finite] == target_candidate[finite])) if np.any(finite) else float("nan")


def _nmse_for_prediction(arrays: dict[str, Any], prediction_name: str) -> float:
    prediction = np.asarray(arrays["predictions"][prediction_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)
    values, _ = _prediction_metric_values(prediction, target, x, "nmse")
    return float(np.nanmean(values))


def gate_summary_rows(data: dict[str, Any], split: str) -> list[dict[str, float | str]]:
    arrays = split_arrays(data, split)
    vanilla_nmse = _nmse_for_prediction(arrays, "vanilla")
    cov_nmse = _nmse_for_prediction(arrays, "cov_forecast")
    rows: list[dict[str, float | str]] = []
    for name in gate_prediction_names(data, split):
        nmse = _nmse_for_prediction(arrays, name)
        rows.append(
            {
                "name": name,
                "shape": _gate_shape(name),
                "right_pct": _gate_right_percent(arrays, name),
                "nmse": nmse,
                "relative_nmse_pct": (
                    float(100.0 * nmse / vanilla_nmse)
                    if abs(vanilla_nmse) > 1e-12
                    else float("nan")
                ),
                "improvement_vanilla_pct": _relative_improvement_pct(vanilla_nmse, nmse),
                "improvement_cov_pct": _relative_improvement_pct(cov_nmse, nmse),
            }
        )
    return rows


def gate_summary_html(
    rows: list[dict[str, float | str]],
    *,
    vanilla_nmse: float,
    cov_nmse: float,
) -> str:
    summary = (
        f"<p><b>References:</b> vanilla nMSE={_format_metric_value(vanilla_nmse)}; "
        f"cov nMSE={_format_metric_value(cov_nmse)}.</p>"
    )
    if not rows:
        return summary + "<b>No gate or oracle predictions found.</b>"
    header = (
        "<tr><th>gate/oracle</th><th>shape</th><th>% right</th>"
        "<th>nMSE to y</th><th>relative nMSE vs vanilla (%)</th>"
        "<th>improvement vs vanilla (%)</th>"
        "<th>improvement vs cov (%)</th></tr>"
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row['name']))}</td>"
            f"<td>{escape(str(row['shape']))}</td>"
            f"<td>{_format_metric_value(float(row['right_pct']))}</td>"
            f"<td>{_format_metric_value(float(row['nmse']))}</td>"
            f"<td>{_format_metric_value(float(row['relative_nmse_pct']))}</td>"
            f"<td>{_format_metric_value(float(row['improvement_vanilla_pct']))}</td>"
            f"<td>{_format_metric_value(float(row['improvement_cov_pct']))}</td>"
            "</tr>"
        )
    return (
        summary
        + "<table>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}</style>"
        + header
        + "".join(body)
        + "</table>"
    )


def gate_threshold_sweep(
    data: dict[str, Any],
    split: str,
    gate_name: str,
    *,
    points: int = 101,
) -> dict[str, np.ndarray]:
    arrays = split_arrays(data, split)
    score, target = _gate_score_target(data, split, gate_name)
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise ValueError(f"No finite gate scores for {gate_name}")
    thresholds = np.linspace(float(np.nanmin(finite_score)), float(np.nanmax(finite_score)), points)
    vanilla = np.asarray(arrays["predictions"]["vanilla"], dtype=np.float64)
    candidate_name = _gate_candidate_prediction_name(gate_name)
    candidate = np.asarray(arrays["predictions"][candidate_name], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)
    right_pct = np.empty_like(thresholds)
    true_positive_rate_pct = np.empty_like(thresholds)
    nmse = np.empty_like(thresholds)
    relative_improvement_vanilla_pct = np.empty_like(thresholds)
    target_label = target > 0.0
    finite_target = np.isfinite(score) & np.isfinite(target)
    scalar_gate = score.ndim == 1
    vanilla_values, _ = _prediction_metric_values(vanilla, y, x, "nmse")
    vanilla_nmse = float(np.nanmean(vanilla_values))
    for index, threshold in enumerate(thresholds):
        decision = score > threshold
        if scalar_gate:
            right_mask = finite_target
            prediction = np.where(decision[:, None], candidate, vanilla)
        else:
            right_mask = finite_target
            prediction = np.where(decision, candidate, vanilla)
        right_pct[index] = (
            100.0 * np.mean(decision[right_mask] == target_label[right_mask])
            if np.any(right_mask)
            else np.nan
        )
        positive_mask = right_mask & target_label
        true_positive_rate_pct[index] = (
            100.0 * np.mean(decision[positive_mask])
            if np.any(positive_mask)
            else np.nan
        )
        metric_values, _ = _prediction_metric_values(prediction, y, x, "nmse")
        nmse[index] = np.nanmean(metric_values)
        relative_improvement_vanilla_pct[index] = _relative_improvement_pct(
            vanilla_nmse,
            float(nmse[index]),
        )
    return {
        "threshold": thresholds,
        "right_pct": right_pct,
        "true_positive_rate_pct": true_positive_rate_pct,
        "nmse": nmse,
        "relative_improvement_vanilla_pct": relative_improvement_vanilla_pct,
    }


def plot_gate_threshold_sweep(data: dict[str, Any], split: str, gate_name: str) -> plt.Figure:
    values = gate_threshold_sweep(data, split, gate_name)
    thresholds = values["threshold"]
    fig, (ax_right, ax_nmse, ax_improvement) = plt.subplots(1, 3, figsize=(17, 4.5))
    ax_right.plot(thresholds, values["right_pct"], linewidth=2.1, label="% right")
    ax_right.plot(
        thresholds,
        values["true_positive_rate_pct"],
        linewidth=2.1,
        label="true-positive rate",
    )
    ax_right.axvline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_right.set_xlabel("Decision threshold")
    ax_right.set_ylabel("Percent")
    ax_right.legend(loc="best")
    ax_right.grid(True, alpha=0.25)
    ax_nmse.plot(thresholds, values["nmse"], linewidth=2.1, color="tab:orange")
    ax_nmse.axvline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_nmse.set_xlabel("Decision threshold")
    ax_nmse.set_ylabel("nMSE")
    ax_nmse.grid(True, alpha=0.25)
    ax_improvement.plot(
        thresholds,
        values["relative_improvement_vanilla_pct"],
        linewidth=2.1,
        color="tab:green",
    )
    ax_improvement.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_improvement.axvline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_improvement.set_xlabel("Decision threshold")
    ax_improvement.set_ylabel("nMSE improvement vs vanilla (%)")
    ax_improvement.grid(True, alpha=0.25)
    fig.suptitle(f"{split}: {gate_name} threshold sweep")
    fig.tight_layout()
    return fig


def baseline_importance_options(data: dict[str, Any]) -> list[str]:
    return sorted(data.get("baseline_artifacts", {}).get("models", {}))


def baseline_feature_importance(
    data: dict[str, Any],
    baseline_name: str,
) -> tuple[list[str], np.ndarray, str]:
    model = data["baseline_artifacts"]["models"][baseline_name]
    if model["kind"] == "ridge":
        coefficients = np.asarray(model["coef"], dtype=np.float64)
        importance = (
            np.abs(coefficients)
            if coefficients.ndim == 1
            else np.nanmean(np.abs(coefficients), axis=0)
        )
        names = [str(name) for name in model["signals"]]
        detail = (
            "absolute shared coefficient"
            if coefficients.ndim == 1
            else "mean absolute coefficient over horizons"
        )
    elif model["kind"] == "lambda":
        coefficients = np.asarray(model["lambda"], dtype=np.float64)
        importance = np.asarray([float(np.nanmean(np.abs(coefficients)))])
        names = ["avgy"]
        detail = (
            "absolute mixing coefficient"
            if coefficients.ndim == 0
            else "mean absolute mixing coefficient over horizons"
        )
    else:
        raise ValueError(f"Unsupported baseline artifact kind: {model['kind']!r}")
    return names, importance, detail


def _plot_importance_bars(
    names: list[str] | np.ndarray,
    importance: np.ndarray,
    *,
    title: str,
    xlabel: str,
) -> plt.Figure:
    names_array = np.asarray(names, dtype=object)
    values = np.asarray(importance, dtype=np.float64)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(values))))
    ax.barh(names_array[order], values[order])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_baseline_feature_importance(
    data: dict[str, Any],
    baseline_name: str,
) -> plt.Figure:
    names, importance, detail = baseline_feature_importance(data, baseline_name)
    return _plot_importance_bars(
        names,
        importance,
        title=f"{baseline_name}: coefficient importance",
        xlabel=detail,
    )


def gate_importance_options(data: dict[str, Any]) -> list[str]:
    return sorted(data.get("gate_importances", {}))


def plot_gate_feature_importance(
    data: dict[str, Any],
    gate_name: str,
) -> plt.Figure:
    values = data["gate_importances"][gate_name]
    return _plot_importance_bars(
        values["feature"],
        values["importance"],
        title=f"{gate_name}: CatBoost feature importance",
        xlabel="Mean CatBoost feature importance",
    )


def ts_ifa_coefficient_options(data: dict[str, Any]) -> list[tuple[str, tuple[str, str]]]:
    options: list[tuple[str, tuple[str, str]]] = []
    for model_name, artifacts in sorted(data.get("ts_ifa_artifacts", {}).items()):
        if "ridge_rooter_coefficients" in artifacts:
            options.append((f"{model_name}: ridge rooter", (model_name, "ridge_rooter")))
        if "active_rooter_coefficients" in artifacts:
            options.append(
                (
                    f"{model_name}: active rooter (mean over T3 windows)",
                    (model_name, "rooter_mean"),
                )
            )
    return options


def ts_ifa_coefficients(
    data: dict[str, Any],
    coefficient_selection: tuple[str, str],
) -> tuple[np.ndarray, list[str]]:
    model_name, coefficient_name = coefficient_selection
    artifacts = data["ts_ifa_artifacts"][model_name]
    names = [str(name) for name in artifacts.get("candidate_names", [])]
    if coefficient_name == "ridge_rooter":
        values = np.asarray(artifacts["ridge_rooter_coefficients"], dtype=np.float64)
    elif coefficient_name == "rooter_mean":
        values = np.nanmean(
            np.asarray(artifacts["active_rooter_coefficients"], dtype=np.float64),
            axis=0,
        )
    else:
        raise ValueError(f"Unknown TS-IFA coefficient view: {coefficient_name}")
    if values.ndim != 2:
        raise ValueError(f"TS-IFA coefficients must be candidate x horizon, found {values.shape}")
    if not names:
        names = [f"candidate {index + 1}" for index in range(values.shape[0])]
    if len(names) != values.shape[0]:
        raise ValueError(
            f"TS-IFA candidate-name mismatch: {len(names)} names for {values.shape[0]} rows"
        )
    return values, names


def plot_ts_ifa_coefficients(
    data: dict[str, Any],
    coefficient_selection: tuple[str, str],
) -> plt.Figure:
    model_name, coefficient_name = coefficient_selection
    values, names = ts_ifa_coefficients(data, coefficient_selection)
    maximum = max(float(np.nanmax(np.abs(values))), 1e-8)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.08 * values.shape[1]), max(3.5, 0.55 * len(names))))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Candidate")
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_title(f"{model_name}: {coefficient_name.replace('_', ' ')} coefficients")
    fig.colorbar(image, ax=ax, label="Coefficient")
    fig.tight_layout()
    return fig


def _preview_names(names: list[str], limit: int = 6) -> str:
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"


def data_summary(data: dict[str, Any]) -> str:
    manifests = data["paths"]["results"]
    lines = [
        "Loaded splits: " + ", ".join(available_splits(data)),
        "Current prediction manifests: "
        + (", ".join(str(path) for path in manifests) if manifests else "none"),
        f"Baseline coefficient models: {len(baseline_importance_options(data))}",
        f"Gate feature-importance models: {len(gate_importance_options(data))}",
        f"TS-IFA coefficient views: {len(ts_ifa_coefficient_options(data))}",
    ]
    for split in available_splits(data):
        arrays = split_arrays(data, split)
        names = list(arrays["predictions"])
        lines.append(
            f"{split}: {len(arrays['x']):,} queries; "
            f"{len(names)} quantity options: {_preview_names(names)}"
        )
    return "\n".join(lines)


def _notebook_runtime() -> tuple[Any, Any, Any]:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    return widgets, clear_output, display


def _default_split(splits: list[str]) -> str:
    return "eval" if "eval" in splits else splits[0]


def _default_prediction_name(names: list[str]) -> str:
    return "vanilla" if "vanilla" in names else names[0]


def _default_scalar_feature(names: list[str]) -> str:
    return "distance_mean" if "distance_mean" in names else names[0]


def query_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    query_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    query_sample = widgets.Button(description="random query", icon="random", button_style="primary")
    query_normalized = widgets.ToggleButton(value=False, description="instance normalized", icon="exchange")
    query_hide_axes = widgets.ToggleButton(value=False, description="remove axes", icon="eye-slash")
    query_output = widgets.Output()
    query_rng = np.random.default_rng()
    query_state = {"index": 0}

    def draw_query(*_: Any) -> None:
        with query_output:
            clear_output(wait=True)
            fig = plot_query_example(
                data,
                query_split.value,
                query_state["index"],
                instance_normalized=query_normalized.value,
                hide_axes=query_hide_axes.value,
            )
            display(fig)
            plt.close(fig)

    def sample_query(*_: Any) -> None:
        count = len(split_arrays(data, query_split.value)["x"])
        query_state["index"] = int(query_rng.integers(count))
        draw_query()

    query_sample.on_click(sample_query)
    query_split.observe(sample_query, names="value")
    query_normalized.observe(draw_query, names="value")
    query_hide_axes.observe(draw_query, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([query_split, query_sample]),
            widgets.HBox([query_normalized, query_hide_axes]),
            query_output,
        ]
    )
    sample_query()
    return section


def window_scatter_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    scatter_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    scatter_names = prediction_names(data, scatter_split.value)
    scatter_features = scalar_feature_names(data, scatter_split.value)
    scatter_prediction = widgets.Dropdown(
        options=scatter_names,
        value=_default_prediction_name(scatter_names),
        description="quantity:",
    )
    scatter_reference = widgets.Dropdown(
        options=scatter_names,
        value=_default_prediction_name(scatter_names),
        description="relative to:",
    )
    scatter_reference_box = widgets.HBox([scatter_reference])
    scatter_metric = widgets.Dropdown(options=["mse", "nmse", "difference"], value="mse", description="metric:")
    scatter_view = widgets.Dropdown(
        options=["direct", "improvement", "relative"],
        value="direct",
        description="view:",
    )
    scatter_feature = widgets.Dropdown(
        options=scatter_features or [""],
        value=_default_scalar_feature(scatter_features) if scatter_features else "",
        description="x:",
    )
    scatter_log_x = widgets.ToggleButton(value=False, description="log x", icon="arrows-h")
    scatter_log_y = widgets.ToggleButton(value=False, description="log y", icon="arrows-v")
    scatter_output = widgets.Output()

    def update_scatter_controls(*_: Any, redraw: bool = True) -> None:
        scatter_reference_box.layout.display = (
            "" if scatter_view.value in {"improvement", "relative"} else "none"
        )
        if redraw:
            draw_scatter()

    def draw_scatter(*_: Any) -> None:
        update_scatter_controls(redraw=False)
        with scatter_output:
            clear_output(wait=True)
            if not scatter_feature.value:
                print("No scalar features available for this split.")
                return
            fig = plot_window_metric_scatter(
                data,
                scatter_split.value,
                scatter_prediction.value,
                scatter_reference.value,
                scatter_metric.value,
                scatter_view.value,
                scatter_feature.value,
                x_log_scale=scatter_log_x.value,
                y_log_scale=scatter_log_y.value,
            )
            display(fig)
            plt.close(fig)

    def update_scatter_names(*_: Any) -> None:
        names = prediction_names(data, scatter_split.value)
        features = scalar_feature_names(data, scatter_split.value)
        scatter_prediction.options = names
        scatter_reference.options = names
        scatter_feature.options = features or [""]
        scatter_prediction.value = _default_prediction_name(names)
        scatter_reference.value = _default_prediction_name(names)
        scatter_feature.value = _default_scalar_feature(features) if features else ""
        draw_scatter()

    scatter_split.observe(update_scatter_names, names="value")
    for control in [
        scatter_prediction,
        scatter_reference,
        scatter_metric,
        scatter_view,
        scatter_feature,
        scatter_log_x,
        scatter_log_y,
    ]:
        control.observe(draw_scatter, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([scatter_split, scatter_prediction]),
            widgets.HBox([scatter_metric, scatter_view, scatter_feature]),
            widgets.HBox([scatter_log_x, scatter_log_y]),
            scatter_reference_box,
            scatter_output,
        ]
    )
    draw_scatter()
    return section


def horizon_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    horizon_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    initial_names = prediction_names(data, horizon_split.value)
    horizon_prediction = widgets.Dropdown(
        options=initial_names,
        value=_default_prediction_name(initial_names),
        description="quantity:",
    )
    horizon_reference = widgets.Dropdown(
        options=initial_names,
        value=_default_prediction_name(initial_names),
        description="relative to:",
    )
    horizon_reference_box = widgets.HBox([horizon_reference])
    horizon_metric = widgets.Dropdown(options=["mse", "nmse", "difference"], value="mse", description="metric:")
    horizon_view = widgets.Dropdown(options=["direct", "improvement", "relative"], value="direct", description="view:")
    horizon_output = widgets.Output()

    def update_horizon_controls(*_: Any, redraw: bool = True) -> None:
        horizon_reference_box.layout.display = "" if horizon_view.value in {"improvement", "relative"} else "none"
        if redraw:
            draw_horizon()

    def draw_horizon(*_: Any) -> None:
        update_horizon_controls(redraw=False)
        with horizon_output:
            clear_output(wait=True)
            fig = plot_horizon(
                data,
                horizon_split.value,
                horizon_prediction.value,
                horizon_reference.value,
                horizon_metric.value,
                horizon_view.value,
            )
            display(fig)
            plt.close(fig)

    def update_horizon_names(*_: Any) -> None:
        names = prediction_names(data, horizon_split.value)
        horizon_prediction.options = names
        horizon_reference.options = names
        horizon_prediction.value = _default_prediction_name(names)
        horizon_reference.value = _default_prediction_name(names)
        draw_horizon()

    horizon_split.observe(update_horizon_names, names="value")
    for control in [horizon_prediction, horizon_reference, horizon_metric, horizon_view]:
        control.observe(draw_horizon, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([horizon_split, horizon_prediction]),
            widgets.HBox([horizon_metric, horizon_view]),
            horizon_reference_box,
            horizon_output,
        ]
    )
    draw_horizon()
    return section


def gates_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    gate_splits = [split for split in splits if gate_summary_rows(data, split) or gate_options(data, split)]
    gate_split = widgets.Dropdown(
        options=gate_splits or [""],
        value=(gate_splits or [""])[0],
        description="split:",
    )
    initial_gate_options = gate_options(data, gate_split.value) if gate_split.value else [("no scored gate", "")]
    gate_choice = widgets.Dropdown(
        options=initial_gate_options or [("no scored gate", "")],
        description="gate:",
    )
    gate_table = widgets.HTML()
    roc_summary = widgets.HTML()
    roc_output = widgets.Output()
    threshold_output = widgets.Output()
    importance_output = widgets.Output()

    def refresh_gate_table() -> None:
        rows = gate_summary_rows(data, gate_split.value) if gate_split.value else []
        if not gate_split.value:
            gate_table.value = "<b>No gate or oracle predictions found.</b>"
            return
        arrays = split_arrays(data, gate_split.value)
        gate_table.value = gate_summary_html(
            rows,
            vanilla_nmse=_nmse_for_prediction(arrays, "vanilla"),
            cov_nmse=_nmse_for_prediction(arrays, "cov_forecast"),
        )

    def draw_gates(*_: Any) -> None:
        refresh_gate_table()
        with roc_output:
            clear_output(wait=True)
            if not gate_split.value or not gate_choice.value:
                roc_summary.value = "<b>No saved gate diagnostics.</b> Run evaluate_baselines first."
            else:
                fig, metrics = plot_gate_roc(data, gate_split.value, gate_choice.value)
                auc_text = f"{metrics['auc']:.4f}" if np.isfinite(metrics["auc"]) else "undefined (one class)"
                roc_summary.value = (
                    f"<b>Threshold=0 % right:</b> {100 * metrics['accuracy']:.2f}% &nbsp; "
                    f"<b>ROC AUC:</b> {auc_text} &nbsp; "
                    f"<b>Decisions:</b> {int(metrics['count']):,}"
                )
                display(fig)
                plt.close(fig)
        with threshold_output:
            clear_output(wait=True)
            if gate_split.value and gate_choice.value:
                fig = plot_gate_threshold_sweep(data, gate_split.value, gate_choice.value)
                display(fig)
                plt.close(fig)
        with importance_output:
            clear_output(wait=True)
            if gate_choice.value in data.get("gate_importances", {}):
                fig = plot_gate_feature_importance(data, gate_choice.value)
                display(fig)
                plt.close(fig)
            elif gate_choice.value:
                print(f"No saved feature importance for {gate_choice.value}.")

    def update_gate_options(*_: Any) -> None:
        options = gate_options(data, gate_split.value) if gate_split.value else [("no scored gate", "")]
        gate_choice.options = options or [("no scored gate", "")]
        draw_gates()

    gate_split.observe(update_gate_options, names="value")
    gate_choice.observe(draw_gates, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([gate_split, gate_choice]),
            gate_table,
            roc_summary,
            roc_output,
            threshold_output,
            importance_output,
        ]
    )
    draw_gates()
    return section


def baseline_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    names = baseline_importance_options(data)
    baseline_choice = widgets.Dropdown(
        options=names or [""],
        value=(names or [""])[0],
        description="baseline:",
    )
    output = widgets.Output()

    def draw_baseline(*_: Any) -> None:
        with output:
            clear_output(wait=True)
            if not baseline_choice.value:
                print("No fitted baseline coefficient artifacts were loaded.")
                return
            fig = plot_baseline_feature_importance(data, baseline_choice.value)
            display(fig)
            plt.close(fig)

    baseline_choice.observe(draw_baseline, names="value")
    draw_baseline()
    return widgets.VBox([baseline_choice, output])


def ts_ifa_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    options = ts_ifa_coefficient_options(data)
    coefficient_choice = widgets.Dropdown(
        options=options or [("no TS-IFA coefficients", "")],
        description="rooter:",
    )
    output = widgets.Output()

    def draw_coefficients(*_: Any) -> None:
        with output:
            clear_output(wait=True)
            if not coefficient_choice.value:
                print("No completed TS-IFA coefficient artifacts were loaded.")
                return
            fig = plot_ts_ifa_coefficients(data, coefficient_choice.value)
            display(fig)
            plt.close(fig)

    coefficient_choice.observe(draw_coefficients, names="value")
    draw_coefficients()
    return widgets.VBox([coefficient_choice, output])
