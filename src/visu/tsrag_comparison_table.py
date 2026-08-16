"""Build the project-dataset TS-RAG and matched-control comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DATASETS = ("Electricity", "Traffic", "Solar", "exchange_rate")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_manifests(root: Path) -> list[tuple[dict[str, Any], Path]]:
    selected = []
    for path in root.rglob("manifest.json"):
        manifest = _json(path)
        if manifest.get("status") == "completed":
            selected.append((manifest, path.parent))
    return selected


def _manifest_ref(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = manifest["identity"]
    return {
        "manifest_id": manifest["manifest_id"],
        "launch_id": manifest.get("launch", {}).get("launch_id"),
        "dataset": identity["dataset"],
        "backbone": identity["backbone"],
        "formula": identity.get("model_config", {}).get("formula"),
    }


def _rows(
    controls_root: Path, tsrag_root: Path, control_method: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    obtained: list[dict[str, Any]] = []
    for manifest, directory in _selected_manifests(controls_root):
        metrics = directory / "baseline_metrics.json"
        if not metrics.is_file():
            continue
        identity = manifest["identity"]
        if identity.get("model_config", {}).get("formula") != control_method:
            continue
        dataset = str(identity["dataset"])
        backbone = str(identity["backbone"])
        used = False
        for row in _json(metrics):
            if row.get("split") != "eval":
                continue
            method = str(row["baseline"])
            rows[(backbone, method, dataset)] = {
                "backbone": backbone,
                "method": method,
                "dataset": dataset,
                **{name: float(row[name]) for name in ("mse", "mae", "nmse", "positive_window_pct")},
            }
            used = True
        if used:
            obtained.append(_manifest_ref(manifest))
    for manifest, directory in _selected_manifests(tsrag_root):
        metrics = directory / "tsrag_metrics.json"
        if not metrics.is_file():
            continue
        dataset = str(manifest["identity"]["dataset"])
        row = _json(metrics)[0]
        rows[("chronos-bolt", "tsrag", dataset)] = {
            "backbone": "chronos-bolt",
            "method": "tsrag",
            "dataset": dataset,
            **{name: float(row[name]) for name in ("mse", "mae", "nmse", "positive_window_pct")},
        }
        obtained.append(_manifest_ref(manifest))
    groups = {(backbone, method) for backbone, method, _ in rows}
    for backbone, method in groups:
        present = {dataset for b, m, dataset in rows if (b, m) == (backbone, method)}
        if present != set(DATASETS):
            raise ValueError(f"incomplete {backbone}/{method}: {sorted(present)}")
    if not any(method == "tsrag" for _, method in groups):
        raise ValueError("no completed TS-RAG results found")
    return [rows[key] for key in sorted(rows)], sorted(
        obtained, key=lambda item: (item["backbone"], item["dataset"], item["formula"] or "")
    )


def build(
    controls_root: Path, tsrag_root: Path, output_dir: Path, control_method: str
) -> dict[str, Path]:
    rows, obtained = _rows(controls_root, tsrag_root, control_method)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tsrag_comparison.csv"
    fields = ("backbone", "method", "dataset", "mse", "mae", "nmse", "positive_window_pct")
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
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "adaptation_tsrag_comparison_report",
                "protocol": {
                    "lags": 512,
                    "horizon": 64,
                    "neighbors": 10,
                    "retrieval": "chronos-t5-base/euclidean/same_user/fixed",
                },
                "selected_control_method": control_method,
                "obtained_manifests": obtained,
                "files": {"csv": csv_path.name, "latex": tex_path.name},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "latex": tex_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", required=True)
    parser.add_argument("--tsrag-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-method", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    return build(
        Path(args.controls_root).expanduser().resolve(),
        Path(args.tsrag_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.control_method,
    )


if __name__ == "__main__":
    main()
