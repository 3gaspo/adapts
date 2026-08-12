"""Smoke-check current-manifest adaptation result discovery and rendering."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_runs import allocate_run, mark_status
from visu.results_table import discover_results, generate_results_table


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation_run(
    root: Path,
    *,
    family: str,
    formula: str,
    row: dict,
    include_vanilla: bool = False,
    retrieval: tuple[str, str, int, str] = ("instance", "euclidean", 3, "online"),
    dataset: str = "electricity",
    setting: str = "168_24",
    model: str = "chronos2",
    pipeline_config: dict | None = None,
) -> None:
    space, metric, neighbors, mode = retrieval
    config = {
        "formula": formula,
        "space": space,
        "metric": metric,
        "k": neighbors,
        "mode": mode,
    }
    identity = (
        root / dataset / setting / model / formula
        / space / metric / str(neighbors) / mode
    )
    lookback, horizon = map(int, setting.split("_"))
    allocation = allocate_run(
        identity,
        project="adaptation",
        workflow=family,
        dataset=dataset,
        lookback=lookback,
        horizon=horizon,
        backbone=model,
        model_config_order=list(config),
        model_config=config,
        pipeline_config=pipeline_config or {"iterations": 30_000},
        display_name=formula,
    )
    complete = {
        "positive_window_pct": 50.0,
        "relative_nmse_improvement_pct": 0.0,
        **row,
        "baseline": formula,
    }
    rows = [complete]
    if include_vanilla:
        rows.insert(
            0,
            {
                "split": "eval",
                "baseline": "vanilla",
                "mse": 0.0012,
                "mae": 0.03,
                "nmse": 0.4,
                "positive_window_pct": 50.0,
                "relative_nmse_improvement_pct": 0.0,
            },
        )
    filename = "baseline_metrics.json" if family == "baselines" else "gate_metrics.json"
    _write(allocation.run_dir / filename, rows)
    _write(allocation.run_dir / "prediction_manifest.json", {"format": "test"})
    _write(
        allocation.run_dir / "result_manifest.json",
        {
            "format": "adaptation_evaluation_result",
            "family": family,
            "methods": [formula],
            "metric_fields": list(rows[0]),
            "files": {
                "metrics_json": filename,
                "predictions": "prediction_manifest.json",
            },
        },
    )
    mark_status(
        allocation.run_dir,
        "completed",
        required_artifacts=[filename, "prediction_manifest.json", "result_manifest.json"],
    )


def _ts_ifa_run(
    root: Path,
    *,
    dataset: str = "electricity",
    setting: str = "168_24",
    model: str = "chronos2",
    retrieval: tuple[str, str, int, str] = ("instance", "euclidean", 3, "online"),
) -> str:
    method = "joint_ridge_horizon_unconstrained_full"
    config = {
        "variant": "joint_ridge",
        "routing_scope": "horizon",
        "routing_constraint": "unconstrained",
        "branch_set": "full",
        "space": retrieval[0],
        "metric": retrieval[1],
        "k": retrieval[2],
        "mode": retrieval[3],
    }
    identity = root / dataset / setting / model
    for name in config:
        identity /= str(config[name])
    allocation = allocate_run(
        identity,
        project="adaptation",
        workflow="ts_ifa",
        dataset=dataset,
        lookback=int(setting.split("_")[0]),
        horizon=int(setting.split("_")[1]),
        backbone=model,
        model_config_order=list(config),
        model_config=config,
        pipeline_config={"training.epochs": 20000},
        display_name=method,
    )
    payload = {
        f"{branch}_{metric_name}": 0.5
        for branch in (
            "adapted",
            "vanilla_branch",
            "cov_branch",
            "residual_branch",
            "memory_branch",
        )
        for metric_name in ("mse", "mae", "nmse")
    }
    payload.update(
        adapted_mse=0.0008,
        adapted_mae=0.018,
        adapted_nmse=0.25,
        vanilla_branch_nmse=0.4,
        residual_branch_nmse=0.28,
        memory_branch_nmse=0.32,
    )
    _write(allocation.run_dir / "eval_metrics.json", payload)
    _write(allocation.run_dir / "prediction_manifest.json", {"format": "test"})
    _write(
        allocation.run_dir / "result_manifest.json",
        {
            "format": "adaptation_ts_ifa_result",
            "variant": "joint_ridge",
            "method": method,
            "candidate_names": ["vanilla", "cov", "residual", "memory"],
            "architecture": "configurable_delta_branches_routing_v4",
            "run_signature": "test",
            "files": {"metrics": "eval_metrics.json", "predictions": "prediction_manifest.json"},
        },
    )
    mark_status(
        allocation.run_dir,
        "completed",
        required_artifacts=["eval_metrics.json", "prediction_manifest.json", "result_manifest.json"],
    )
    return method


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        baselines = {
            "avgy_ridge_shared": (0.0009, 0.02, 0.30),
            "cov_delta_ridge_shared": (0.00085, 0.019, 0.28),
            "avgy_ridge_shared_eval_fit": (0.0005, 0.015, 0.18),
        }
        for index, (formula, (mse, mae, nmse)) in enumerate(baselines.items()):
            _evaluation_run(
                root,
                family="baselines",
                formula=formula,
                row={"split": "eval", "mse": mse, "mae": mae, "nmse": nmse},
                include_vanilla=index == 0,
            )
        gates = {
            "bayes_cov_shared": (0.00075, 0.019, 0.24),
            "catboost_cov_classifier_shared": (0.0007, 0.018, 0.22),
            "catboost_cov_regressor_horizon": (0.0006, 0.016, 0.20),
            "oracle_cov_shared": (0.0002, 0.01, 0.10),
        }
        for formula, (mse, mae, nmse) in gates.items():
            _evaluation_run(
                root,
                family="gates",
                formula=formula,
                row={"split": "eval", "mse": mse, "mae": mae, "nmse": nmse},
            )
        method = _ts_ifa_run(root)
        retrieval = "instance_euclidean_3_online"

        # Direct formulas occur in both workflows with different pipeline
        # configurations. Their metric rows must retain the canonical formula
        # rather than a configuration-disambiguated manifest label.
        _evaluation_run(
            root,
            family="baselines",
            formula="cov_forecast",
            row={"split": "eval", "mse": 0.0008, "mae": 0.02, "nmse": 0.27},
            pipeline_config={"l2_grid": "0,1e-6"},
        )
        _evaluation_run(
            root,
            family="gates",
            formula="cov_forecast",
            row={"split": "eval", "mse": 0.0008, "mae": 0.02, "nmse": 0.27},
            pipeline_config={"gate.iterations": 300},
        )

        records = discover_results(root)
        methods = {record.method for record in records if record.metric == "mse"}
        assert "vanilla" in methods
        assert f"{retrieval}/avgy_ridge_shared" in methods
        assert f"{retrieval}/cov_forecast" in methods
        assert f"{retrieval}/oracle_cov_shared" in methods
        assert f"{retrieval}/{method}" in methods
        assert f"{retrieval}/{method}_memory_branch" in methods
        direct_sources = {
            record.path.name
            for record in records
            if record.metric == "mse" and record.method == f"{retrieval}/cov_forecast"
        }
        assert direct_sources == {"baseline_metrics.json", "gate_metrics.json"}

        output = generate_results_table(
            root,
            methods=[
                "vanilla",
                f"{retrieval}/avgy_ridge_shared",
                f"{retrieval}/{method}",
                f"{retrieval}/oracle_cov_shared",
            ],
            reference="vanilla",
        )
        latex = output.read_text(encoding="utf-8")
        assert r"$\times 10^{-3}$" in latex
        assert r"IN\_L2\_3/TS-IFA JR-H-U-full" in latex
        assert "online" not in latex
        assert r"\textbf{0.20}" not in latex  # oracle columns are excluded from bolding

        fixed = ("raw", "euclidean", 3, "fixed")
        _evaluation_run(
            root,
            family="baselines",
            formula="avgy_convex_shared",
            row={"split": "eval", "mse": 0.001, "mae": 0.02, "nmse": 0.35},
            retrieval=fixed,
        )
        fixed_output = generate_results_table(root, output=root / "fixed.tex")
        assert r"raw\_L2\_3\_fixed/avgy-convex-s" in fixed_output.read_text(encoding="utf-8")

        obsolete_root = root / "obsolete"
        _write(obsolete_root / "electricity" / "168_24" / "legacy" / "metrics.json", {"mse": 0.1})
        assert discover_results(obsolete_root) == []

    print("results table checks passed")


if __name__ == "__main__":
    main()
