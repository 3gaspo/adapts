"""Combine our exact-split MSE with published Cross-RAG paper rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Exchange")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _our_rows(
    results_root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    project_to_paper = {
        values["project_name"]: name
        for name, values in config["datasets"].items()
    }
    values: dict[str, dict[str, float]] = {}
    vanilla: dict[str, float] = {}
    obtained: list[dict[str, Any]] = []
    for manifest_path in results_root.rglob("manifest.json"):
        if manifest_path.parent.parent.name == "manifest_history":
            continue
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        identity = manifest.get("identity", {})
        project_dataset = identity.get("dataset")
        if project_dataset not in project_to_paper:
            continue
        metrics_path = manifest_path.parent / "baseline_metrics.json"
        if not metrics_path.is_file():
            continue
        dataset = project_to_paper[project_dataset]
        used = False
        for row in _load_json(metrics_path):
            if row.get("split") != "eval":
                continue
            method = str(row["baseline"])
            if method == "vanilla":
                vanilla[dataset] = float(row["mse"])
            else:
                values.setdefault(method, {})[dataset] = float(row["mse"])
            used = True
        if used:
            obtained.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "launch_id": manifest.get("launch", {}).get("launch_id"),
                    "dataset": project_dataset,
                    "backbone": identity.get("backbone"),
                    "formula": identity.get("model_config", {}).get("formula"),
                }
            )
    if set(vanilla) != set(DATASETS):
        raise ValueError(f"incomplete Chronos-Bolt row: {sorted(vanilla)}")
    rows = [{"method": "Chronos-Bolt (our evaluation)", **vanilla, "source": "computed"}]
    for method, dataset_values in sorted(values.items()):
        if set(dataset_values) != set(DATASETS):
            raise ValueError(f"incomplete method={method}: {sorted(dataset_values)}")
        rows.append({"method": method, **dataset_values, "source": "computed"})
    return rows, sorted(obtained, key=lambda item: item["dataset"])


def build_table(config_path: Path, results_root: Path, output_dir: Path) -> dict[str, Path]:
    config = _load_json(config_path)
    if config.get("metric") != "mse" or tuple(config.get("datasets", {})) != DATASETS:
        raise ValueError("unexpected SOTA benchmark configuration")
    rows, obtained = _our_rows(results_root, config)
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

    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "adaptation_sota_benchmark_report",
                "metric": "mse",
                "published_source": config["source"],
                "obtained_manifests": obtained,
                "files": {"csv": csv_path.name, "latex": tex_path.name},
                "methods": [row["method"] for row in rows],
                "datasets": list(DATASETS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "latex": tex_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    return build_table(
        Path(args.config).expanduser().resolve(),
        Path(args.results_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
