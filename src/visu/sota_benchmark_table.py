"""Combine selected exact-split MSE runs with published Cross-RAG paper rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_runs import SelectedRun, write_report_manifest
from visu.results_table import selected_manifest_runs


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Exchange")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _single_or_average(values: list[float], *, name: str, allow_average: bool) -> float:
    if not values:
        raise ValueError(f"missing {name}")
    if len(values) > 1 and not allow_average:
        raise ValueError(
            f"multiple selected values for {name}; narrow --pipeline-config "
            "or use an average policy"
        )
    return sum(values) / len(values)


def _our_rows(
    results_root: Path,
    config: Mapping[str, Any],
    selected_pipeline: Mapping[str, Any],
    *,
    pipeline_config: Mapping[str, Any],
    config_policy: str,
    repeat_policy: str,
    purposes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[SelectedRun]]:
    project_to_paper = {
        values["project_name"]: name for name, values in config["datasets"].items()
    }
    collected: dict[tuple[str, str], list[float]] = {}
    obtained: list[SelectedRun] = []
    for selected in selected_manifest_runs(
        results_root,
        pipeline_config=pipeline_config,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        purposes=purposes,
    ):
        identity = selected.manifest["identity"]
        project_dataset = identity.get("dataset")
        model_config = identity.get("model_config", {})
        if (
            project_dataset not in project_to_paper
            or identity.get("lookback") != 512
            or identity.get("horizon") != 64
            or identity.get("backbone") != "chronos-bolt"
            or any(model_config.get(key) != value for key, value in selected_pipeline.items())
        ):
            continue
        metrics_path = selected.run_dir / "baseline_metrics.json"
        if not metrics_path.is_file():
            continue
        dataset = project_to_paper[project_dataset]
        used = False
        for row in _load_json(metrics_path):
            if row.get("split") != "eval":
                continue
            method = str(row["baseline"])
            collected.setdefault((method, dataset), []).append(float(row["mse"]))
            used = True
        if used:
            obtained.append(selected)

    allow_average = config_policy == "average" or repeat_policy == "average"
    methods = {method for method, _ in collected}
    if "vanilla" not in methods:
        raise ValueError("no selected Chronos-Bolt evaluation runs found")
    rows: list[dict[str, Any]] = []
    for method in sorted(methods, key=lambda item: (item != "vanilla", item)):
        values = {
            dataset: _single_or_average(
                collected.get((method, dataset), []),
                name=f"{method}/{dataset}",
                allow_average=allow_average,
            )
            for dataset in DATASETS
        }
        label = "Chronos-Bolt (our evaluation)" if method == "vanilla" else method
        rows.append({"method": label, **values, "source": "computed"})
    return rows, obtained


def build_table(
    config_path: Path,
    results_root: Path,
    output_dir: Path,
    selected_pipeline: Mapping[str, Any],
    *,
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Sequence[str] = (),
) -> dict[str, Path]:
    config = _load_json(config_path)
    if config.get("metric") != "mse" or tuple(config.get("datasets", {})) != DATASETS:
        raise ValueError("unexpected SOTA benchmark configuration")
    requested_pipeline = dict(pipeline_config or {})
    rows, obtained = _our_rows(
        results_root,
        config,
        selected_pipeline,
        pipeline_config=requested_pipeline,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        purposes=purposes,
    )
    for method, values in config["published_results"].items():
        rows.append(
            {
                "method": f"{method} (published)",
                **{dataset: float(values[dataset]) for dataset in DATASETS},
                "average": float(values["Average"]),
                "source": config["source_table"],
            }
        )
    for row in rows:
        row.setdefault("average", sum(float(row[name]) for name in DATASETS) / len(DATASETS))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sota_benchmark_mse.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Method", *DATASETS, "Average", "Source"))
        for row in rows:
            writer.writerow(
                (
                    row["method"],
                    *(f"{float(row[name]):.6f}" for name in DATASETS),
                    f"{float(row['average']):.6f}",
                    row["source"],
                )
            )

    tex_path = output_dir / "sota_benchmark_mse.tex"
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & ETTh1 & ETTh2 & ETTm1 & ETTm2 & Weather & Electricity & Exchange & Average \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["method"]).replace("_", r"\_")
        values = " & ".join(f"{float(row[name]):.3f}" for name in (*DATASETS, "average"))
        lines.append(f"{label} & {values} " + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = write_report_manifest(
        output_dir / "report_manifest.json",
        inputs=obtained,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        filters={
            "datasets": list(DATASETS),
            "setting": "512_64",
            "backbone": "chronos-bolt",
            "selected_pipeline": dict(selected_pipeline),
            "pipeline": requested_pipeline,
            "purposes": list(purposes),
            "static_metrics": str(config_path),
            "published_source": config["source"],
        },
    )
    return {"csv": csv_path, "latex": tex_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--distance-metric", required=True)
    parser.add_argument("--neighbors", required=True, type=int)
    parser.add_argument("--retrieval-mode", required=True)
    parser.add_argument("--pipeline-config", action="append", default=[])
    parser.add_argument(
        "--config-policy", choices=("distinct", "latest", "average"), default="distinct"
    )
    parser.add_argument(
        "--repeat-policy",
        choices=("selected", "latest", "distinct", "average"),
        default="selected",
    )
    parser.add_argument("--purpose", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    return build_table(
        Path(args.config).expanduser().resolve(),
        Path(args.results_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        {
            "formula": args.formula,
            "space": args.space,
            "metric": args.distance_metric,
            "k": args.neighbors,
            "mode": args.retrieval_mode,
        },
        pipeline_config=_pipeline_config(args.pipeline_config),
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        purposes=args.purpose,
    )


if __name__ == "__main__":
    main()
