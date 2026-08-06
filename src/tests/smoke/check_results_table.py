"""Smoke-check TS-IFA result discovery and LaTeX rendering."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visu.results_table import discover_results, generate_results_table


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_evaluation(path: Path, family: str, rows) -> None:
    complete_rows = []
    for row in rows:
        complete = {
            "positive_window_pct": 50.0,
            "relative_nmse_improvement_pct": 0.0,
            **row,
        }
        complete_rows.append(complete)
    _write(path, complete_rows)
    _write(path.parent / "prediction_manifest.json", {"format": "test"})
    _write(
        path.parent / "result_manifest.json",
        {
            "format": "adaptation_evaluation_result",
            "family": family,
            "methods": [row["baseline"] for row in complete_rows if row["baseline"] != "vanilla"],
            "metric_fields": list(complete_rows[0]),
            "files": {
                "metrics_json": path.name,
                "predictions": "prediction_manifest.json",
            },
        },
    )


def _write_ts_ifa(path: Path, variant: str, metrics) -> None:
    complete = {
        f"{branch}_{metric}": 0.5
        for branch in ("adapted", "vanilla_branch", "cov_branch", "residual_branch", "memory_branch")
        for metric in ("mse", "mae", "nmse")
    }
    complete.update(metrics)
    _write(path, complete)
    _write(path.parent / "prediction_manifest.json", {"format": "test"})
    _write(
        path.parent / "result_manifest.json",
        {
            "format": "adaptation_ts_ifa_result",
            "variant": variant,
            "architecture": "shared_delta_branches_four_rooters_v3",
            "run_signature": "test",
            "files": {
                "metrics": path.name,
                "predictions": "prediction_manifest.json",
            },
        },
    )


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        setting = root / "electricity" / "168_24"
        _write(setting / "direct" / "chronos2" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 0.0012}, "nmse": {"mean": 0.4}}})
        run = "instance_euclidean_3_online"
        _write_evaluation(setting / run / "baselines" / "baseline_metrics.json", "baselines",
               [{"split": "eval", "baseline": "vanilla", "mse": 0.0012, "mae": 0.03, "nmse": 0.4},
                {"split": "eval", "baseline": "avgy_ridge_shared", "mse": 0.0009, "mae": 0.02, "nmse": 0.3},
                {"split": "eval", "baseline": "cov_delta_ridge_shared", "mse": 0.00085,
                 "mae": 0.019, "nmse": 0.28},
                {"split": "eval", "baseline": "avgy_ridge_shared_eval_fit", "mse": 0.0005,
                 "mae": 0.015, "nmse": 0.18}])
        _write_evaluation(setting / run / "gates" / "gate_metrics.json", "gates",
               [{"split": "eval", "baseline": "bayes_cov_shared", "mse": 0.00075,
                 "mae": 0.019, "nmse": 0.24},
                {"split": "eval", "baseline": "catboost_cov_classifier_shared", "mse": 0.0007,
                 "mae": 0.018, "nmse": 0.22},
                {"split": "eval", "baseline": "catboost_cov_classifier_shared_soft", "mse": 0.00065,
                 "mae": 0.017, "nmse": 0.21},
                {"split": "eval", "baseline": "catboost_cov_regressor_horizon", "mse": 0.0006,
                 "mae": 0.016, "nmse": 0.2},
                {"split": "eval", "baseline": "oracle_cov_shared", "mse": 0.0002, "mae": 0.01, "nmse": 0.1},
                {"split": "eval", "baseline": "oracle_cov_horizon", "mse": 0.0001, "mae": 0.005, "nmse": 0.05}])
        _write_ts_ifa(setting / run / "ts_ifa" / "joint_ridge" / "eval_metrics.json", "joint_ridge",
               {"adapted_mse": 0.0008, "adapted_mae": 0.018, "adapted_nmse": 0.25,
                "vanilla_branch_nmse": 0.4,
                "residual_branch_nmse": 0.28, "memory_branch_nmse": 0.32})

        records = discover_results(root)
        methods = {record.method for record in records if record.metric == "mse"}
        assert methods == {
            "chronos2",
            f"{run}/vanilla",
            f"{run}/avgy_ridge_shared",
            f"{run}/cov_delta_ridge_shared",
            f"{run}/avgy_ridge_shared_eval_fit",
            f"{run}/bayes_cov_shared",
            f"{run}/catboost_cov_classifier_shared",
            f"{run}/catboost_cov_classifier_shared_soft",
            f"{run}/catboost_cov_regressor_horizon",
            f"{run}/oracle_cov_shared",
            f"{run}/oracle_cov_horizon",
            f"{run}/joint_ridge",
            f"{run}/joint_ridge_vanilla_branch",
            f"{run}/joint_ridge_cov_branch",
            f"{run}/joint_ridge_residual_branch",
            f"{run}/joint_ridge_memory_branch",
        }, methods
        output = generate_results_table(
            root,
            methods=["chronos2", f"{run}/avgy_ridge_shared", f"{run}/joint_ridge",
                     f"{run}/oracle_cov_shared", f"{run}/oracle_cov_horizon"],
            reference="chronos2",
        )
        latex = output.read_text(encoding="utf-8")
        assert r"$\times 10^{-3}$" in latex
        assert r"\textbf{0.80}" in latex
        assert "33.33\\%" in latex
        assert r"IN\_L2\_3/TS-IFA joint ridge" in latex
        assert "online" not in latex
        assert r"\begin{tabular}{llcrrr|rr}" in latex
        assert r"\textbf{0.10}" not in latex

        default_output = generate_results_table(root, output=root / "default.tex", datasets=["electricity"])
        default_latex = default_output.read_text(encoding="utf-8")
        assert "vanilla" not in default_latex
        assert r"IN\_L2\_3/oracle-cov-s" in default_latex
        assert r"IN\_L2\_3/bayes-cov-s" in default_latex
        assert r"IN\_L2\_3/cb-cov-cls-s" in default_latex
        assert r"IN\_L2\_3/cb-cov-cls-s-soft" in default_latex
        assert r"IN\_L2\_3/cov-delta-ridge-s" in default_latex
        assert r"IN\_L2\_3/cb-cov-reg-h" in default_latex

        baseline_output = generate_results_table(
            root,
            output=root / "baselines.tex",
            methods=["chronos2", f"{run}/avgy_ridge_shared", f"{run}/avgy_ridge_shared_eval_fit"],
            reference="chronos2",
            excluded_from_bold=["avgy_ridge_shared_eval_fit"],
        )
        baseline_latex = baseline_output.read_text(encoding="utf-8")
        assert r"IN\_L2\_3/avgy-ridge-s-fit-T3" in baseline_latex
        assert r"\begin{tabular}{llcrr|r}" in baseline_latex

        ts_ifa_output = generate_results_table(
            root,
            output=root / "ts_ifa.tex",
            metric="nmse",
            methods=["chronos2", f"{run}/joint_ridge", f"{run}/joint_ridge_residual_branch", f"{run}/joint_ridge_memory_branch"],
            reference="chronos2",
        )
        ts_ifa_latex = ts_ifa_output.read_text(encoding="utf-8")
        assert r"IN\_L2\_3/TS-IFA joint ridge" in ts_ifa_latex
        assert r"IN\_L2\_3/TS-IFA JR-R" in ts_ifa_latex
        assert r"IN\_L2\_3/TS-IFA JR-M" in ts_ifa_latex

        fixed_run = "raw_euclidean_3_fixed"
        _write_evaluation(setting / fixed_run / "baselines" / "baseline_metrics.json", "baselines",
               [{"split": "eval", "baseline": "avgy_convex_shared", "mse": 0.001, "mae": 0.02, "nmse": 0.35}])
        fixed_output = generate_results_table(root, output=root / "fixed.tex", datasets=["electricity"])
        assert r"raw\_L2\_3\_fixed/avgy-convex-s" in fixed_output.read_text(encoding="utf-8")

        _write(root / "toy" / "1_1" / "direct" / "reference" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 1.0}}})
        _write(root / "toy" / "1_1" / "direct" / "candidate" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 0.5}}})
        _write(root / "toy" / "2_1" / "direct" / "reference" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 9.0}}})
        _write(root / "toy" / "2_1" / "direct" / "candidate" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 8.1}}})
        averaged_output = generate_results_table(
            root,
            output=root / "averaged.tex",
            datasets=["toy"],
            methods=["reference", "candidate"],
            reference="reference",
            setting_improvements=False,
        )
        averaged_latex = averaged_output.read_text(encoding="utf-8")
        assert "14.00\\%" in averaged_latex
        assert "30.00\\%" not in averaged_latex

        positive_output = generate_results_table(
            root,
            output=root / "positive.tex",
            datasets=["toy"],
            methods=["reference", "candidate"],
            reference="reference",
            positive_only=True,
        )
        positive_latex = positive_output.read_text(encoding="utf-8")
        assert "candidate" in positive_latex

        negative_output = generate_results_table(
            root,
            output=root / "negative.tex",
            datasets=["toy"],
            methods=["candidate", "reference"],
            reference="candidate",
            positive_only=True,
        )
        negative_latex = negative_output.read_text(encoding="utf-8")
        assert "reference" not in negative_latex

        obsolete_root = root / "obsolete"
        _write(
            obsolete_root
            / "electricity"
            / "168_24"
            / run
            / "ts_ifa"
            / "TS-IFA"
            / "eval_metrics.json",
            {"adapted_nmse": 0.1},
        )
        try:
            discover_results(obsolete_root)
        except ValueError as error:
            assert "obsolete TS-IFA result directory" in str(error)
        else:
            raise AssertionError("obsolete TS-IFA output was accepted")

    print("results table checks passed")


if __name__ == "__main__":
    main()
