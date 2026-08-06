"""Export fitted baseline coefficients as numeric tables and heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch


RESULT_FORMAT = "adaptation_evaluation_result"
ARTIFACT_FORMAT = "adaptation_baseline_models"
REPEATED_SIGNALS = {"Y", "N", "Y-V", "N-V"}
DIRECT_BASELINES = {"cov_forecast", "avgy", "y_mean"}
INDEX_FIELDS = (
    "dataset",
    "setting",
    "model",
    "retrieval",
    "baseline",
    "kind",
    "mode",
    "coefficients_csv",
    "coefficients_plot",
)


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(Path(path), map_location="cpu")


def _split_names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _run_names(
    spaces: Sequence[str],
    distance_metrics: Sequence[str],
    neighbors: Sequence[int],
    retrieval_mode: str,
) -> list[str]:
    return [
        f"{space}_{metric}_{neighbors_value}_{retrieval_mode}"
        for space in spaces
        for metric in distance_metrics
        for neighbors_value in neighbors
    ]


def _pipeline_parts(pipeline: str) -> tuple[str, str, str]:
    parts = pipeline.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"invalid pipeline {pipeline!r}; expected family/retrieval_run/method"
        )
    return parts[0], parts[1], parts[2]


def _neighbors_from_run(run: str) -> int:
    parts = run.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit() or parts[2] not in {"online", "fixed"}:
        raise ValueError(f"cannot resolve K from retrieval run {run!r}")
    return int(parts[1])


def _expanded_signal_names(signals: Sequence[str], neighbors: int) -> list[str]:
    names: list[str] = []
    for signal in signals:
        signal = str(signal)
        if signal in REPEATED_SIGNALS:
            names.extend(f"{signal}_{rank}" for rank in range(1, neighbors + 1))
        else:
            names.append(signal)
    return names


def _coefficient_payload(model: dict[str, Any]) -> tuple[np.ndarray, str]:
    kind = str(model.get("kind", ""))
    if kind == "ridge":
        return np.asarray(model["coef"], dtype=np.float64), "Coefficient"
    if kind == "convex":
        return np.asarray(model["weights"], dtype=np.float64), "Convex weight"
    raise ValueError(f"unsupported fitted baseline kind {kind!r}")


def _coefficient_models(artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = dict(artifacts.get("models", {}))
    eval_fit_models = artifacts.get("eval_fit_models") or {}
    models.update({f"{name}_eval_fit": model for name, model in eval_fit_models.items()})
    return models


def _write_coefficient_csv(
    path: Path,
    coefficients: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    values = coefficients[None, :] if coefficients.ndim == 1 else coefficients
    if values.ndim != 2:
        raise ValueError(f"baseline coefficients must be 1D or 2D, found {values.shape}")
    index = ["shared"] if coefficients.ndim == 1 else list(range(1, values.shape[0] + 1))
    frame = pd.DataFrame(values, index=index, columns=feature_names)
    frame.index.name = "horizon"
    frame.to_csv(path)


def _plot_coefficients(
    path: Path,
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
) -> None:
    values = coefficients[None, :] if coefficients.ndim == 1 else coefficients
    finite = np.abs(values[np.isfinite(values)])
    scale = float(finite.max()) if finite.size else 1.0
    scale = max(scale, 1e-12)
    if colorbar_label == "Convex weight" and np.nanmin(values) >= 0:
        norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)), 1e-12))
        cmap = "viridis"
    else:
        norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
        cmap = "coolwarm"

    height = 3.4 if values.shape[0] == 1 else min(10.0, max(4.0, 0.055 * values.shape[0]))
    width = min(20.0, max(8.0, 0.52 * len(feature_names)))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    if len(feature_names) <= 30:
        feature_ticks = np.arange(len(feature_names))
    else:
        feature_ticks = np.unique(
            np.linspace(0, len(feature_names) - 1, 25, dtype=int)
        )
    ax.set_xticks(feature_ticks)
    ax.set_xticklabels(
        [feature_names[index] for index in feature_ticks], rotation=45, ha="right"
    )
    if values.shape[0] == 1:
        ax.set_yticks([0], ["shared"])
    else:
        tick_count = min(9, values.shape[0])
        ticks = np.unique(np.linspace(0, values.shape[0] - 1, tick_count, dtype=int))
        ax.set_yticks(ticks, [str(tick + 1) for tick in ticks])
    ax.set_xlabel("Input signal")
    ax.set_ylabel("Horizon step")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_current_artifacts(baseline_dir: Path) -> dict[str, Any]:
    manifest_path = baseline_dir / "result_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing current baseline result manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RESULT_FORMAT or manifest.get("family") != "baselines":
        raise ValueError(f"{manifest_path} is not a current baseline result")
    artifact_name = manifest.get("files", {}).get("artifacts")
    if not artifact_name:
        raise ValueError(f"{manifest_path} does not index baseline artifacts")
    artifact_path = baseline_dir / str(artifact_name)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing fitted baseline artifacts: {artifact_path}")
    artifacts = torch_load(artifact_path)
    if artifacts.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{artifact_path} is not a current baseline artifact")
    return artifacts


def export_baseline_coefficient_plots(
    experiment_dir: str | Path,
    output_dir: str | Path,
    *,
    datasets: Sequence[str],
    settings: Sequence[str],
    models: Sequence[str],
    families: Sequence[str] = ("baselines",),
    spaces: Sequence[str] = ("raw", "instance"),
    distance_metrics: Sequence[str] = ("euclidean",),
    neighbors: Sequence[int] = (1, 3),
    retrieval_mode: str = "online",
    variants: Sequence[str] | None = None,
    pipelines: Sequence[str] | None = None,
) -> list[Path]:
    """Write one exact signed-coefficient heatmap per fitted result configuration."""
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    requested_families = set(families)
    selected_baseline_pipelines: dict[str, set[str]] = {}
    exact_pipeline_selection = bool(pipelines)
    if pipelines:
        for pipeline in pipelines:
            family, run, method = _pipeline_parts(pipeline)
            if family == "baselines":
                selected_baseline_pipelines.setdefault(run, set()).add(method)
    elif requested_families & {"baselines", "full", "comparison"}:
        selected_baseline_pipelines = {
            run: set(variants or ())
            for run in _run_names(spaces, distance_metrics, neighbors, retrieval_mode)
        }

    if not selected_baseline_pipelines:
        destination.mkdir(parents=True, exist_ok=True)
        index_path = destination / "coefficient_index.csv"
        if not index_path.exists():
            with index_path.open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=INDEX_FIELDS).writeheader()
        return [index_path]

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    outputs: list[Path] = []
    for model_name in models:
        for dataset in datasets:
            for setting in settings:
                for run, selected_methods in selected_baseline_pipelines.items():
                    baseline_dir = root / dataset / setting / model_name / run / "baselines"
                    artifacts = _load_current_artifacts(baseline_dir)
                    fitted = _coefficient_models(artifacts)
                    methods = (
                        sorted(selected_methods)
                        if exact_pipeline_selection
                        else sorted(set(fitted) & selected_methods)
                        if selected_methods
                        else sorted(fitted)
                    )
                    neighbors_value = _neighbors_from_run(run)
                    for method in methods:
                        if method in DIRECT_BASELINES:
                            continue
                        if method not in fitted:
                            raise ValueError(
                                f"selected fitted baseline {method!r} is missing from "
                                f"{baseline_dir / 'baseline_artifacts.pt'}"
                            )
                        baseline = fitted[method]
                        coefficients, colorbar_label = _coefficient_payload(baseline)
                        feature_names = _expanded_signal_names(
                            baseline.get("signals", ()), neighbors_value
                        )
                        feature_count = coefficients.shape[-1] if coefficients.ndim else 0
                        if feature_count != len(feature_names):
                            raise ValueError(
                                f"{method} has {feature_count} coefficients but "
                                f"{len(feature_names)} expanded signal names"
                            )
                        method_dir = destination / dataset / setting / run
                        method_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = method_dir / f"{method}.csv"
                        png_path = method_dir / f"{method}.png"
                        _write_coefficient_csv(csv_path, coefficients, feature_names)
                        _plot_coefficients(
                            png_path,
                            coefficients,
                            feature_names,
                            title=f"{dataset} | {setting} | {run}\n{method}",
                            colorbar_label=colorbar_label,
                        )
                        outputs.extend((csv_path, png_path))
                        rows.append(
                            {
                                "dataset": dataset,
                                "setting": setting,
                                "model": model_name,
                                "retrieval": run,
                                "baseline": method,
                                "kind": str(baseline.get("kind", "")),
                                "mode": str(baseline.get("mode", "")),
                                "coefficients_csv": csv_path.relative_to(destination).as_posix(),
                                "coefficients_plot": png_path.relative_to(destination).as_posix(),
                            }
                        )

    index_path = destination / "coefficient_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(index_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--families", default="baselines")
    parser.add_argument("--spaces", default="raw,instance")
    parser.add_argument("--distance-metrics", default="euclidean")
    parser.add_argument("--neighbors", default="1,3")
    parser.add_argument("--retrieval-mode", default="online")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--pipelines", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    outputs = export_baseline_coefficient_plots(
        args.experiment_dir,
        args.output_dir,
        datasets=_split_names(args.datasets),
        settings=_split_names(args.settings),
        models=_split_names(args.models),
        families=_split_names(args.families),
        spaces=_split_names(args.spaces),
        distance_metrics=_split_names(args.distance_metrics),
        neighbors=[int(value) for value in _split_names(args.neighbors)],
        retrieval_mode=args.retrieval_mode,
        variants=_split_names(args.variants),
        pipelines=_split_names(args.pipelines),
    )
    print(f"Baseline coefficient outputs written: {len(outputs) - 1}")
    print(f"Coefficient index written to {outputs[-1]}")
    return outputs


if __name__ == "__main__":
    main()
