"""Smoke-check sweep tables over current manifest-backed adaptation runs."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_results_table import _evaluation_run, _ts_ifa_run
from visu.sweep_results_table import (
    generate_average_results_tables,
    generate_full_results_tables,
)


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        retrievals = {
            "raw_euclidean_1_online": ("raw", "euclidean", 1, "online"),
            "instance_euclidean_3_online": ("instance", "euclidean", 3, "online"),
        }
        for dataset, offset in (("electricity", 0.0), ("solar", 0.2)):
            for run, retrieval in retrievals.items():
                better = 0.2 if run.startswith("instance") else 0.0
                for index, (formula, nmse) in enumerate(
                    (
                        ("cov_forecast", 0.95 + offset),
                        ("y_ridge_shared", 0.92 + offset - better),
                        ("residual_ridge_horizon_eval_fit", 0.10 + offset),
                    )
                ):
                    _evaluation_run(
                        root,
                        family="baselines",
                        formula=formula,
                        row={
                            "split": "eval",
                            "mse": nmse / 100,
                            "mae": 0.01,
                            "nmse": nmse,
                            "positive_window_pct": 60.0 + 10.0 * better,
                        },
                        include_vanilla=index == 0,
                        retrieval=retrieval,
                        dataset=dataset,
                    )
                for formula, nmse in (
                    ("bayes_cov_shared", 0.88 + offset - better),
                    ("catboost_cov_classifier_shared", 0.93 + offset - better),
                    ("oracle_cov_horizon", 0.40 + offset),
                ):
                    _evaluation_run(
                        root,
                        family="gates",
                        formula=formula,
                        row={
                            "split": "eval",
                            "mse": nmse / 100,
                            "mae": 0.01,
                            "nmse": nmse,
                            "positive_window_pct": 65.0 + 10.0 * better,
                        },
                        retrieval=retrieval,
                        dataset=dataset,
                    )
                _ts_ifa_run(root, dataset=dataset, retrieval=retrieval)

        method = "joint_ridge_horizon_unconstrained_full"
        pipelines = [
            f"{family}/{run}/{formula}"
            for run in retrievals
            for family, formulas in (
                ("baselines", ("cov_forecast", "y_ridge_shared", "residual_ridge_horizon_eval_fit")),
                ("gates", ("bayes_cov_shared", "catboost_cov_classifier_shared", "oracle_cov_horizon")),
                (
                    "ts_ifa",
                    (
                        method,
                        f"{method}_vanilla_branch",
                        f"{method}_cov_branch",
                        f"{method}_residual_branch",
                        f"{method}_memory_branch",
                    ),
                ),
            )
            for formula in formulas
        ]

        full_outputs = generate_full_results_tables(
            root,
            root / "tables/full",
            datasets=["electricity", "solar"],
            settings=["168_24"],
            models=["chronos2"],
            spaces=["raw", "instance"],
            neighbors=[1, 3],
            pipelines=pipelines,
        )
        assert {path.name for path in full_outputs} == {
            "full_results.tex",
            "baselines_results.tex",
            "gates_results.tex",
            "ts_ifa_results.tex",
        }
        full = (root / "tables/full/full_results.tex").read_text(encoding="utf-8")
        assert "vanilla" in full
        assert r"raw\_L2\_1/Y-ridge-s" in full
        assert r"IN\_L2\_3/TS-IFA JR-H-U-full" in full

        average_outputs = generate_average_results_tables(
            root,
            root / "tables/average",
            datasets=["electricity", "solar"],
            settings=["168_24"],
            models=["chronos2"],
            spaces=["raw", "instance"],
            neighbors=[1, 3],
            pipelines=pipelines,
        )
        assert {path.name for path in average_outputs} == {
            "full_results.tex",
            "baselines_results.tex",
            "gates_results.tex",
            "ts_ifa_results.tex",
            "positive_windows_results.tex",
        }
        ranking = json.loads(
            (root / "tables/average/pipeline_ranking.json").read_text(encoding="utf-8")
        )
        assert ranking
        assert ranking[0]["winner_name"].count("/") == 2

        try:
            generate_average_results_tables(
                root,
                root / "tables/incomplete",
                datasets=["electricity", "solar", "traffic"],
                settings=["168_24"],
                models=["chronos2"],
                families=["baselines"],
                spaces=["instance"],
                neighbors=[3],
                pipelines=["baselines/instance_euclidean_3_online/y_ridge_shared"],
            )
        except ValueError as error:
            assert "incomplete table inputs" in str(error)
        else:
            raise AssertionError("incomplete selected pipeline was accepted")

    print("adaptation sweep table checks passed")


if __name__ == "__main__":
    main()
