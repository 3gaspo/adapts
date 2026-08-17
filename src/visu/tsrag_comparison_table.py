"""Build the project-dataset TS-RAG and matched-control comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_runs import SelectedRun, write_report_manifest
from visu.results_table import selected_manifest_runs


DATASETS = ("Electricity", "Traffic", "Solar", "exchange_rate")
METRICS = ("mse", "mae", "nmse", "positive_window_pct")
TSRAG_CONFIG = {
    "formula": "tsrag",
    "space": "tsrag",
    "metric": "euclidean",
    "k": 10,
    "mode": "fixed",
}


def _json(path: Path) -> Any:
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


def _matches_protocol(
    selected: SelectedRun, *, formula: str, backbones: set[str]
) -> bool:
    identity = selected.manifest["identity"]
    model_config = identity.get("model_config", {})
    expected = {**TSRAG_CONFIG, "formula": formula}
    return (
        identity.get("dataset") in DATASETS
        and identity.get("lookback") == 512
        and identity.get("horizon") == 64
        and identity.get("backbone") in backbones
        and all(model_config.get(key) == value for key, value in expected.items())
    )


def _rows(
    controls_root: Path,
    tsrag_root: Path,
    control_method: str,
    *,
    pipeline_config: Mapping[str, Any],
    config_policy: str,
    repeat_policy: str,
    purposes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[SelectedRun]]:
    collected: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    obtained: list[SelectedRun] = []
    selection = {
        "pipeline_config": pipeline_config,
        "config_policy": config_policy,
        "repeat_policy": repeat_policy,
        "purposes": purposes,
    }
    for selected in selected_manifest_runs(controls_root, **selection):
        if not _matches_protocol(
            selected, formula=control_method, backbones={"chronos2", "chronos-bolt"}
        ):
            continue
        metrics_path = selected.run_dir / "baseline_metrics.json"
        if not metrics_path.is_file():
            continue
        identity = selected.manifest["identity"]
        used = False
        for row in _json(metrics_path):
            if row.get("split") != "eval":
                continue
            key = (str(identity["backbone"]), str(row["baseline"]), str(identity["dataset"]))
            collected.setdefault(key, []).append({name: float(row[name]) for name in METRICS})
            used = True
        if used:
            obtained.append(selected)

    for selected in selected_manifest_runs(tsrag_root, **selection):
        if not _matches_protocol(selected, formula="tsrag", backbones={"chronos-bolt"}):
            continue
        metrics_path = selected.run_dir / "tsrag_metrics.json"
        if not metrics_path.is_file():
            continue
        metric_rows = _json(metrics_path)
        if len(metric_rows) != 1:
            raise ValueError(f"expected one TS-RAG metric row in {metrics_path}")
        dataset = str(selected.manifest["identity"]["dataset"])
        collected.setdefault(("chronos-bolt", "tsrag", dataset), []).append(
            {name: float(metric_rows[0][name]) for name in METRICS}
        )
        obtained.append(selected)

    allow_average = config_policy == "average" or repeat_policy == "average"
    rows: list[dict[str, Any]] = []
    for (backbone, method, dataset), values in sorted(collected.items()):
        if len(values) > 1 and not allow_average:
            raise ValueError(
                f"multiple selected values for {backbone}/{method}/{dataset}; "
                "narrow --pipeline-config or use an average policy"
            )
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "dataset": dataset,
                **{
                    name: sum(value[name] for value in values) / len(values)
                    for name in METRICS
                },
            }
        )

    groups = {(row["backbone"], row["method"]) for row in rows}
    for backbone, method in groups:
        present = {
            row["dataset"]
            for row in rows
            if (row["backbone"], row["method"]) == (backbone, method)
        }
        if present != set(DATASETS):
            raise ValueError(f"incomplete {backbone}/{method}: {sorted(present)}")
    if not any(method == "tsrag" for _, method in groups):
        raise ValueError("no selected TS-RAG results found")
    return rows, obtained


def build(
    controls_root: Path,
    tsrag_root: Path,
    output_dir: Path,
    control_method: str,
    *,
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Sequence[str] = (),
) -> dict[str, Path]:
    requested_pipeline = dict(pipeline_config or {})
    rows, obtained = _rows(
        controls_root,
        tsrag_root,
        control_method,
        pipeline_config=requested_pipeline,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        purposes=purposes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tsrag_comparison.csv"
    fields = ("backbone", "method", "dataset", *METRICS)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row["backbone"], row["method"]), {})[row["dataset"]] = row["mse"]
    tex_path = output_dir / "tsrag_comparison_mse.tex"
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Backbone & Method & Electricity & Traffic & Solar & Exchange & Average \\",
        r"\midrule",
    ]
    for (backbone, method), values in sorted(grouped.items()):
        average = sum(values[name] for name in DATASETS) / len(DATASETS)
        label = method.replace("_", r"\_")
        numbers = " & ".join(f"{values[name]:.3f}" for name in DATASETS)
        lines.append(f"{backbone} & {label} & {numbers} & {average:.3f} " + r"\\")
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
            "selected_control_method": control_method,
            "pipeline": requested_pipeline,
            "purposes": list(purposes),
            "protocol": {
                "neighbors": 10,
                "retrieval": "chronos-t5-base/euclidean/same_user/fixed",
            },
        },
    )
    return {"csv": csv_path, "latex": tex_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", required=True)
    parser.add_argument("--tsrag-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-method", required=True)
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
    return build(
        Path(args.controls_root).expanduser().resolve(),
        Path(args.tsrag_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.control_method,
        pipeline_config=_pipeline_config(args.pipeline_config),
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        purposes=args.purpose,
    )


if __name__ == "__main__":
    main()
