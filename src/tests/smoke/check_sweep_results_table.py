"""Smoke-check adaptation LaTeX table generation."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visu.sweep_results_table import (  # noqa: E402
    generate_average_results_tables,
    generate_full_results_tables,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset, offset in [("electricity", 0.0), ("solar", 0.2)]:
            setting = root / dataset / "168_24" / "chronos"
            _write(
                setting / "vanilla" / "vanilla_metrics.json",
                [{"split": "eval", "baseline": "vanilla", "nmse": 1.0 + offset, "mse": 0.01}],
            )
            _write(
                setting / "vanilla" / "univariate_summary.json",
                {"eval": {"nmse": {"mean": 9.0 + offset}, "mse": {"mean": 0.09}}},
            )
            for run, ridge, bayes, ts_ifa, residual_branch, memory_branch in [
                (
                    "raw_euclidean_1_online",
                    0.92 + offset,
                    0.88 + offset,
                    0.84 + offset,
                    0.88 + offset,
                    0.90 + offset,
                ),
                (
                    "instance_euclidean_3_online",
                    0.72 + offset,
                    0.62 + offset,
                    0.58 + offset,
                    0.64 + offset,
                    0.66 + offset,
                ),
            ]:
                _write(
                    setting / run / "baselines" / "baseline_metrics.json",
                    [
                        {"split": "eval", "baseline": "context_forecast", "nmse": 0.95 + offset, "mse": 0.009},
                        {"split": "eval", "baseline": "horizon_ridge_shared", "nmse": ridge, "mse": 0.007},
                        {
                            "split": "eval",
                            "baseline": "residual_ridge_horizon_eval_fit",
                            "nmse": 0.1 + offset,
                            "mse": 0.001,
                        },
                    ],
                )
                _write(
                    setting / run / "gates" / "gate_metrics.json",
                    [
                        {"split": "eval", "baseline": "bayes_context_scalar", "nmse": bayes, "mse": 0.006},
                        {
                            "split": "eval",
                            "baseline": "catboost_context_classifier_scalar",
                            "nmse": bayes + 0.05,
                            "mse": 0.0065,
                        },
                        {"split": "eval", "baseline": "oracle_context_horizon", "nmse": 0.4 + offset, "mse": 0.004},
                    ],
                )
                _write(
                    setting / run / "ts_ifa" / "TS-IFA" / "eval_metrics.json",
                    {
                        "adapted_nmse": ts_ifa,
                        "adapted_mse": 0.005,
                        "vanilla_nmse": 1.0 + offset,
                        "residual_branch_nmse": residual_branch,
                        "memory_branch_nmse": memory_branch,
                    },
                )

        full_outputs = generate_full_results_tables(
            root,
            root / "tables" / "full",
            datasets=["electricity", "solar"],
            settings=["168_24"],
            spaces=["raw", "instance"],
            neighbors=[1, 3],
        )
        assert {output.name for output in full_outputs} == {
            "full_results.tex",
            "baselines_results.tex",
            "gates_results.tex",
            "ts_ifa_results.tex",
        }
        full = (root / "tables" / "full" / "full_results.tex").read_text(encoding="utf-8")
        assert full.count(r"\begin{table}") == 1
        assert "vanilla" in full
        assert r"raw\_L2\_1/Y-ridge" in full
        assert r"IN\_L2\_3/TS-IFA" in full
        assert "Overall improvement" not in full

        average_outputs = generate_average_results_tables(
            root,
            root / "tables" / "average",
            datasets=["electricity", "solar"],
            settings=["168_24"],
            spaces=["raw", "instance"],
            neighbors=[1, 3],
        )
        assert {output.name for output in average_outputs} == {
            "full_results.tex",
            "baselines_results.tex",
            "gates_results.tex",
            "ts_ifa_results.tex",
        }

        baselines = (root / "tables" / "average" / "baselines_results.tex").read_text(encoding="utf-8")
        assert baselines.count(r"\begin{table}") == 1
        assert "Vanilla Chronos NMSE: 1.10" in baselines
        assert "9.10" not in baselines
        assert r"\textcolor{green!50!black}{\textbf{25.45\%}}" in baselines
        assert r"R-ridge-h-fit-T3" in baselines
        assert baselines.count(r"\midrule") >= 2

        gates = (root / "tables" / "average" / "gates_results.tex").read_text(encoding="utf-8")
        assert r"bayes-s & \begin{tabular}{@{}c@{}}\textcolor{green!50!black}{10.91\%}" in gates
        assert r"oracle-h" in gates

        ts_ifa = (root / "tables" / "average" / "ts_ifa_results.tex").read_text(encoding="utf-8")
        assert r"TS-IFA & " in ts_ifa
        assert r"TS-IFA-R & " in ts_ifa
        assert r"TS-IFA-M & " in ts_ifa

    print("adaptation table checks passed")


if __name__ == "__main__":
    main()
