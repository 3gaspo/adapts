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
            setting = root / dataset / "168_24" / "chronos2"
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
                run_bonus = 10.0 if run.startswith("instance") else 0.0
                dataset_bonus = 10.0 * offset
                _write(
                    setting / run / "baselines" / "baseline_metrics.json",
                    [
                        {
                            "split": "eval",
                            "baseline": "cov_forecast",
                            "nmse": 0.95 + offset,
                            "mse": 0.009,
                            "positive_window_pct": 55.0 + dataset_bonus,
                        },
                        {
                            "split": "eval",
                            "baseline": "y_ridge_shared",
                            "nmse": ridge,
                            "mse": 0.007,
                            "positive_window_pct": 60.0 + run_bonus + dataset_bonus,
                        },
                        {
                            "split": "eval",
                            "baseline": "residual_ridge_horizon_eval_fit",
                            "nmse": 0.1 + offset,
                            "mse": 0.001,
                            "positive_window_pct": 90.0 + dataset_bonus,
                        },
                    ],
                )
                _write(
                    setting / run / "gates" / "gate_metrics.json",
                    [
                        {
                            "split": "eval",
                            "baseline": "bayes_cov_shared",
                            "nmse": bayes,
                            "mse": 0.006,
                            "positive_window_pct": 65.0 + run_bonus + dataset_bonus,
                        },
                        {
                            "split": "eval",
                            "baseline": "catboost_cov_classifier_shared",
                            "nmse": bayes + 0.05,
                            "mse": 0.0065,
                            "positive_window_pct": 62.0 + run_bonus + dataset_bonus,
                        },
                        {
                            "split": "eval",
                            "baseline": "oracle_cov_horizon",
                            "nmse": 0.4 + offset,
                            "mse": 0.004,
                            "positive_window_pct": 95.0 + dataset_bonus,
                        },
                    ],
                )
                _write(
                    setting / run / "ts_ifa" / "joint_ridge" / "eval_metrics.json",
                    {
                        "adapted_nmse": ts_ifa,
                        "adapted_mse": 0.005,
                        "vanilla_branch_nmse": 1.0 + offset,
                        "cov_branch_nmse": 0.95 + offset,
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
        assert r"raw\_L2\_1/Y-ridge-s" in full
        assert r"IN\_L2\_3/TS-IFA joint ridge" in full
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
            "positive_windows_results.tex",
        }

        baselines = (root / "tables" / "average" / "baselines_results.tex").read_text(encoding="utf-8")
        assert baselines.count(r"\begin{table}") == 1
        assert "Vanilla Chronos-2 NMSE: 1.10" in baselines
        assert "9.10" not in baselines
        assert "25.67" in baselines
        assert r"residual-ridge-h-fit-T3" in baselines
        assert baselines.count(r"\midrule") >= 2
        assert " & Mean" not in baselines
        positive_windows = (
            root / "tables" / "average" / "positive_windows_results.tex"
        ).read_text(encoding="utf-8")
        assert "lower horizon-averaged MSE than the vanilla forecast" in positive_windows
        assert r"71.00\%" in positive_windows
        ranking = json.loads(
            (root / "tables" / "average" / "pipeline_ranking.json").read_text(
                encoding="utf-8"
            )
        )
        assert ranking
        assert ranking[0]["winner_name"].count("/") == 2
        assert ranking[0]["winner_name"].startswith(
            ("baselines/", "gates/", "full/", "ts_ifa/")
        )

        tabpfn_root = root / "tabpfn_only"
        tabpfn_setting = tabpfn_root / "electricity" / "168_24" / "tabpfnts"
        _write(
            tabpfn_setting / "vanilla" / "vanilla_metrics.json",
            [{"split": "eval", "baseline": "vanilla", "nmse": 1.0}],
        )
        _write(
            tabpfn_setting / "raw_euclidean_1_online" / "baselines" / "baseline_metrics.json",
            [{"split": "eval", "baseline": "cov_forecast", "nmse": 0.9}],
        )
        generate_average_results_tables(
            tabpfn_root,
            tabpfn_root / "tables",
            models=["tabpfnts"],
            families=["baselines"],
            spaces=["raw"],
            neighbors=[1],
        )
        tabpfn_table = (tabpfn_root / "tables" / "baselines_results.tex").read_text(encoding="utf-8")
        assert "Vanilla TabPFN-TS NMSE: 1.00" in tabpfn_table

        gates = (root / "tables" / "average" / "gates_results.tex").read_text(encoding="utf-8")
        assert r"bayes-cov-s & " in gates
        assert r"oracle-cov-h" in gates

        ts_ifa = (root / "tables" / "average" / "ts_ifa_results.tex").read_text(encoding="utf-8")
        assert r"TS-IFA joint ridge & " in ts_ifa
        assert r"TS-IFA JR-R & " in ts_ifa
        assert r"TS-IFA JR-M & " in ts_ifa

    print("adaptation table checks passed")


if __name__ == "__main__":
    main()
