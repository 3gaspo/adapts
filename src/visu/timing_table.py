"""Build a compute-time table for the fixed Chronos-Bolt comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

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
) -> str:
    adaptation_file = {
        "baselines": ("baselines", "baseline_timing.json"),
        "gates": ("gates", "gate_timing.json"),
    }[candidate_family]
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for dataset in datasets:
        model_root = root / dataset / setting / model
        candidate_root = model_root / candidate_run
        crossrag_root = model_root / crossrag_run
        vanilla = _seconds(model_root / "vanilla" / "extraction_timing.json")
        retrieval = _seconds(candidate_root / "extraction_timing.json")
        adaptation = _seconds(candidate_root / adaptation_file[0] / adaptation_file[1])
        crossrag_retrieval = _seconds(crossrag_root / "extraction_timing.json")
        crossrag = _seconds(crossrag_root / "crossrag" / "crossrag_timing.json")
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
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    datasets = tuple(
        value.strip()
        for value in args.datasets.replace(";", ",").split(",")
        if value.strip()
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_timing_table(
            Path(args.results_root).expanduser().resolve(),
            datasets=datasets,
            setting=args.setting,
            model=args.model,
            candidate_run=args.candidate_run,
            crossrag_run=args.crossrag_run,
            candidate_family=args.candidate_family,
        ),
        encoding="utf-8",
    )
    print(f"LaTeX timing table written to {output}")
    return output


if __name__ == "__main__":
    main()
