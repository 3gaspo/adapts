"""Build the selected MSE-versus-nMSE fitting-objective ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from experiment_runs import SelectedRun, write_report_manifest
from visu.results_table import selected_manifest_runs


METRICS = ("mse", "nmse", "mae", "nmae")
FIT_LOSSES = ("mse", "nmse")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())


def _pipeline(value: str) -> tuple[str, str, str]:
    parts = value.split("/")
    if len(parts) != 3 or parts[0] not in {"baselines", "gates"}:
        raise ValueError(f"invalid selected pipeline {value!r}")
    return parts[0], parts[1], parts[2]


def _setting(identity: dict[str, Any]) -> str:
    return f"{identity['lookback']}:{identity['horizon']}"


def _run_name(model_config: dict[str, Any]) -> str:
    return "_".join(
        str(model_config[name]) for name in ("space", "metric", "k", "mode")
    )


def build(
    results_root: Path,
    output_dir: Path,
    *,
    pipelines: Sequence[str],
    datasets: Sequence[str],
    settings: Sequence[str],
    model: str,
    purposes: Sequence[str] = (),
) -> dict[str, Path]:
    selected_pipelines = {_pipeline(value) for value in pipelines}
    expected = {
        (family, run, method, fit_loss, dataset, setting)
        for family, run, method in selected_pipelines
        for fit_loss in FIT_LOSSES
        for dataset in datasets
        for setting in settings
    }
    found: set[tuple[str, str, str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    obtained: list[SelectedRun] = []
    for selected in selected_manifest_runs(
        results_root,
        config_policy="distinct",
        repeat_policy="selected",
        purposes=purposes,
    ):
        manifest = selected.manifest
        identity = manifest["identity"]
        if identity.get("backbone") != model:
            continue
        model_config = identity.get("model_config", {})
        method = str(model_config.get("formula", ""))
        run = _run_name(model_config)
        family = "baselines" if (selected.run_dir / "baseline_metrics.json").is_file() else "gates"
        pipeline = (family, run, method)
        if pipeline not in selected_pipelines:
            continue
        fit_loss = str(manifest.get("config", {}).get("pipeline", {}).get("fit_loss", ""))
        dataset = str(identity["dataset"])
        setting = _setting(identity)
        key = (*pipeline, fit_loss, dataset, setting)
        if key not in expected:
            continue
        metrics_path = selected.run_dir / f"{'baseline' if family == 'baselines' else 'gate'}_metrics.json"
        metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        matches = [row for row in metric_rows if row.get("split") == "eval" and row.get("baseline") == method]
        references = [row for row in metric_rows if row.get("split") == "eval" and row.get("baseline") == "vanilla"]
        if len(matches) != 1 or len(references) != 1:
            raise ValueError(f"expected one method and vanilla row in {metrics_path}")
        row = matches[0]
        reference = references[0]
        values = {name: float(row[name]) for name in METRICS}
        rows.append(
            {
                "family": family,
                "method": method,
                "retrieval": run,
                "fit_loss": fit_loss,
                "dataset": dataset,
                "setting": setting,
                **values,
                "mse_improvement_pct": 100.0 * (float(reference["mse"]) - values["mse"]) / max(float(reference["mse"]), 1e-12),
                "nmse_improvement_pct": 100.0 * (float(reference["nmse"]) - values["nmse"]) / max(float(reference["nmse"]), 1e-12),
            }
        )
        found.add(key)
        obtained.append(selected)
    missing = sorted(expected - found)
    if missing:
        preview = ", ".join("/".join(item) for item in missing[:8])
        raise ValueError(f"incomplete training-loss ablation: {preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_loss_ablation.csv"
    fields = (
        "family", "method", "retrieval", "fit_loss", "dataset", "setting",
        *METRICS, "mse_improvement_pct", "nmse_improvement_pct",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row[name]) for name in fields[:6])))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["family"], row["method"], row["retrieval"], row["fit_loss"]), []
        ).append(row)
    averages = []
    for (family, method, retrieval, fit_loss), group in sorted(grouped.items()):
        averages.append(
            {
                "family": family,
                "method": method,
                "retrieval": retrieval,
                "fit_loss": fit_loss,
                **{
                    name: sum(float(row[name]) for row in group) / len(group)
                    for name in (*METRICS, "mse_improvement_pct", "nmse_improvement_pct")
                },
            }
        )
    average_path = output_dir / "training_loss_ablation_average.csv"
    average_fields = ("family", "method", "retrieval", "fit_loss", *METRICS, "mse_improvement_pct", "nmse_improvement_pct")
    with average_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=average_fields)
        writer.writeheader()
        writer.writerows(averages)

    tex_path = output_dir / "training_loss_ablation.tex"
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Family & Method & Retrieval & Fit loss & MSE & nMSE & $\Delta$MSE & $\Delta$nMSE \\",
        r"\midrule",
    ]
    for row in averages:
        labels = [str(row[name]).replace("_", r"\_") for name in ("family", "method", "retrieval", "fit_loss")]
        lines.append(
            " & ".join(labels)
            + f" & {row['mse']:.4f} & {row['nmse']:.4f} & {row['mse_improvement_pct']:.2f}\\% & {row['nmse_improvement_pct']:.2f}\\% "
            + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = write_report_manifest(
        output_dir / "report_manifest.json",
        inputs=obtained,
        config_policy="distinct",
        repeat_policy="selected",
        filters={
            "pipelines": list(pipelines),
            "datasets": list(datasets),
            "settings": list(settings),
            "model": model,
            "fit_losses": list(FIT_LOSSES),
            "purposes": list(purposes),
        },
    )
    return {"csv": csv_path, "average": average_path, "latex": tex_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipelines", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--purpose", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    return build(
        Path(args.results_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        pipelines=_split_csv(args.pipelines),
        datasets=_split_csv(args.datasets),
        settings=_split_csv(args.settings),
        model=args.model,
        purposes=args.purpose,
    )


if __name__ == "__main__":
    main()
