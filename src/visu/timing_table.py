"""Build a compute-time table for the fixed Chronos-Bolt comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_runs import load_manifest, select_identity_runs, write_report_manifest
from .results_table import _latex


def _seconds(path: Path) -> float:
    if not path.is_file():
        return math.nan
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["elapsed_seconds"])


def _cell(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value:.1f}"


def build_timing_table(
    root: Path,
    *,
    datasets: Sequence[str],
    setting: str,
    model: str,
    candidate_run: str,
    crossrag_run: str,
    candidate_family: str,
    candidate_formula: str,
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Sequence[str] = (),
    selected_inputs: list | None = None,
) -> str:
    adaptation_file = {"baselines": "baseline_timing.json", "gates": "gate_timing.json"}[candidate_family]
    selected = []
    identity_roots = sorted(
        {
            path.parent.parent
            for path in root.rglob("manifest.json")
            if path.parent.name.startswith("run_")
            and "archive" not in path.relative_to(root).parts
        }
    )
    for identity_root in identity_roots:
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        if any(manifest["status"] == "completed" for manifest in manifests):
            selected.extend(
                select_identity_runs(
                    identity_root,
                    requested_pipeline=pipeline_config,
                    config_policy=config_policy,
                    repeat_policy=repeat_policy,
                    purposes=purposes,
                )
            )

    def matching(dataset: str, formula: str, retrieval: str, artifact: str):
        matches = []
        for choice in selected:
            identity = choice.manifest["identity"]
            config = identity["model_config"]
            run = "_".join(str(config[name]) for name in ("space", "metric", "k", "mode"))
            if (
                identity["dataset"] == dataset
                and f"{identity['lookback']}_{identity['horizon']}" == setting
                and identity["backbone"] == model
                and config.get("formula") == formula
                and run == retrieval
                and (choice.run_dir / artifact).is_file()
            ):
                matches.append(choice)
        if len(matches) != 1:
            raise ValueError(
                f"expected one selected {formula}/{retrieval} timing run for {dataset}; "
                f"found {len(matches)}. Narrow the pipeline/repeat selection."
            )
        return matches[0]

    def upstream_seconds(choice, name: str) -> float:
        record = choice.manifest.get("inputs", {}).get(name, {})
        manifest_path = Path(str(record.get("path", "")))
        return _seconds(manifest_path.parent / "extraction_timing.json")

    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for dataset in datasets:
        candidate = matching(dataset, candidate_formula, candidate_run, adaptation_file)
        crossrag_choice = matching(dataset, "crossrag", crossrag_run, "crossrag_timing.json")
        if selected_inputs is not None:
            for choice in (candidate, crossrag_choice):
                if choice not in selected_inputs:
                    selected_inputs.append(choice)
        vanilla = upstream_seconds(candidate, "vanilla_manifest")
        retrieval = upstream_seconds(candidate, "upstream_manifest")
        adaptation = _seconds(candidate.run_dir / adaptation_file)
        crossrag_retrieval = upstream_seconds(crossrag_choice, "upstream_manifest")
        crossrag = _seconds(crossrag_choice.run_dir / "crossrag_timing.json")
        candidate_total = retrieval + adaptation
        crossrag_total = crossrag_retrieval + crossrag
        rows.append(
            (
                dataset,
                vanilla,
                retrieval,
                adaptation,
                candidate_total,
                crossrag_retrieval,
                crossrag_total,
            )
        )
    means = tuple(
        sum(values) / len(values) if values and all(math.isfinite(v) for v in values) else math.nan
        for values in zip(*(row[1:] for row in rows))
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wall-clock seconds for the fixed Chronos-Bolt comparison. "
        r"Each total includes that pipeline's own retrieval extraction pass.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Vanilla & Candidate retrieval & Adaptation & Candidate total "
        r"& Cross-RAG retrieval & Cross-RAG total \\",
        r"\midrule",
    ]
    for dataset, *values in rows:
        lines.append(
            " & ".join([_latex(dataset), *(_cell(value) for value in values)]) + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            " & ".join(["Mean", *(_cell(value) for value in means)]) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:crossrag-compute}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--setting", default="512_64")
    parser.add_argument("--model", default="chronos-bolt")
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--crossrag-run", default="minmax_cosine_15_online")
    parser.add_argument("--candidate-family", choices=("baselines", "gates"), required=True)
    parser.add_argument("--candidate-formula", required=True)
    parser.add_argument("--pipeline-config", action="append", default=[])
    parser.add_argument("--config-policy", choices=("distinct", "latest", "average"), default="distinct")
    parser.add_argument("--repeat-policy", choices=("selected", "latest", "distinct", "average"), default="selected")
    parser.add_argument("--purpose", action="append", default=[])
    return parser.parse_args()


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


def main() -> Path:
    args = parse_args()
    datasets = tuple(
        value.strip()
        for value in args.datasets.replace(";", ",").split(",")
        if value.strip()
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_inputs = []
    pipeline = _pipeline_config(args.pipeline_config)
    output.write_text(
        build_timing_table(
            Path(args.results_root).expanduser().resolve(),
            datasets=datasets,
            setting=args.setting,
            model=args.model,
            candidate_run=args.candidate_run,
            crossrag_run=args.crossrag_run,
            candidate_family=args.candidate_family,
            candidate_formula=args.candidate_formula,
            pipeline_config=pipeline,
            config_policy=args.config_policy,
            repeat_policy=args.repeat_policy,
            purposes=args.purpose,
            selected_inputs=selected_inputs,
        ),
        encoding="utf-8",
    )
    write_report_manifest(
        output.with_name("report_manifest.json"),
        inputs=selected_inputs,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        filters={
            "pipeline": pipeline,
            "purposes": args.purpose,
            "datasets": list(datasets),
            "setting": args.setting,
            "model": args.model,
            "candidate_formula": args.candidate_formula,
        },
    )
    print(f"LaTeX timing table written to {output}")
    return output


if __name__ == "__main__":
    main()
