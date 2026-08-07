"""Export fitted baseline coefficients as numeric tables and heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from experiment_runs import load_manifest, select_identity_runs, write_report_manifest


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
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Sequence[str] = (),
) -> list[Path]:
    """Write one exact signed-coefficient heatmap per fitted result configuration."""
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    requested_families = set(families)
    if not requested_families & {"baselines", "full", "comparison"}:
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
    used_choices = []
    allowed_pipelines = {
        (run, method)
        for family, run, method in map(_pipeline_parts, pipelines or ())
        if family == "baselines"
    }
    allowed_runs = set(_run_names(spaces, distance_metrics, neighbors, retrieval_mode))
    allowed_variants = set(variants or ())
    identity_roots = sorted(
        {
            path.parent.parent
            for path in root.rglob("manifest.json")
            if path.parent.name.startswith("run_")
            and "archive" not in path.relative_to(root).parts
        }
    )
    selected = []
    for identity_root in identity_roots:
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        if not any(manifest["status"] == "completed" for manifest in manifests):
            continue
        selected.extend(
            select_identity_runs(
                identity_root,
                requested_pipeline=pipeline_config,
                config_policy=config_policy,
                repeat_policy=repeat_policy,
                purposes=purposes,
            )
        )
    for choice in selected:
        identity = choice.manifest["identity"]
        config = identity["model_config"]
        formula = str(config.get("formula", ""))
        if not formula or not (choice.run_dir / "baseline_artifacts.pt").is_file():
            continue
        dataset = str(identity["dataset"])
        setting = f"{identity['lookback']}_{identity['horizon']}"
        model_name = str(identity["backbone"])
        run = "_".join(str(config[name]) for name in ("space", "metric", "k", "mode"))
        if dataset not in datasets or setting not in settings or model_name not in models:
            continue
        if pipelines and (run, formula) not in allowed_pipelines:
            continue
        if not pipelines and (run not in allowed_runs or allowed_variants and formula not in allowed_variants):
            continue
        if formula in DIRECT_BASELINES:
            continue
        artifacts = _load_current_artifacts(choice.run_dir)
        fitted = _coefficient_models(artifacts)
        if formula not in fitted:
            raise ValueError(
                f"selected fitted baseline {formula!r} is missing from "
                f"{choice.run_dir / 'baseline_artifacts.pt'}"
            )
        baseline = fitted[formula]
        coefficients, colorbar_label = _coefficient_payload(baseline)
        feature_names = _expanded_signal_names(baseline.get("signals", ()), int(config["k"]))
        feature_count = coefficients.shape[-1] if coefficients.ndim else 0
        if feature_count != len(feature_names):
            raise ValueError(
                f"{formula} has {feature_count} coefficients but "
                f"{len(feature_names)} expanded signal names"
            )
        method = choice.label
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
        used_choices.append(choice)

    index_path = destination / "coefficient_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(index_path)
    write_report_manifest(
        destination / "report_manifest.json",
        inputs=used_choices,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        filters={
            "pipeline": dict(pipeline_config or {}),
            "purposes": list(purposes),
            "datasets": list(datasets),
            "settings": list(settings),
            "models": list(models),
            "families": list(families),
            "pipelines": list(pipelines or ()),
        },
    )
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
    parser.add_argument("--pipeline-config", action="append", default=[])
    parser.add_argument(
        "--config-policy",
        choices=("distinct", "latest", "selected", "average"),
        default="distinct",
    )
    parser.add_argument(
        "--repeat-policy",
        choices=("selected", "latest", "distinct", "average"),
        default="selected",
    )
    parser.add_argument("--purpose", action="append", default=[])
    return parser.parse_args(argv)


def _pipeline_config(values: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"pipeline config must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        lowered = value.casefold()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
        else:
            try:
                parsed[key] = int(value)
            except ValueError:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
    return parsed


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
        pipeline_config=_pipeline_config(args.pipeline_config),
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        purposes=args.purpose,
    )
    print(f"Baseline coefficient outputs written: {len(outputs) - 1}")
    print(f"Coefficient index written to {outputs[-1]}")
    return outputs


if __name__ == "__main__":
    main()
