"""Smoke-check fitted baseline coefficient CSV and heatmap exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiment_runs import allocate_run, mark_status
from visu.baseline_coefficients import export_baseline_coefficient_plots


def _write_run(
    root: Path,
    formula: str,
    model: dict,
    *,
    pipeline_config: dict | None = None,
) -> None:
    config = {
        "formula": formula,
        "space": "instance",
        "metric": "euclidean",
        "k": 3,
        "mode": "online",
    }
    identity = root / "Electricity" / "168_24" / "chronos2"
    for name in config:
        identity /= str(config[name])
    allocation = allocate_run(
        identity,
        project="adaptation",
        workflow="baselines",
        dataset="Electricity",
        lookback=168,
        horizon=24,
        backbone="chronos2",
        model_config_order=list(config),
        model_config=config,
        pipeline_config=pipeline_config or {},
        display_name=formula,
    )
    torch.save(
        {
            "format": "adaptation_baseline_models",
            "models": {formula: model},
            "eval_fit_models": None,
        },
        allocation.run_dir / "baseline_artifacts.pt",
    )
    (allocation.run_dir / "result_manifest.json").write_text(
        json.dumps(
            {
                "format": "adaptation_evaluation_result",
                "family": "baselines",
                "files": {"artifacts": "baseline_artifacts.pt"},
            }
        ),
        encoding="utf-8",
    )
    mark_status(
        allocation.run_dir,
        "completed",
        required_artifacts=["baseline_artifacts.pt", "result_manifest.json"],
    )


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_run(
            root,
            "cov_y_ridge_shared",
            {
                "kind": "ridge",
                "mode": "shared",
                "signals": ("V", "C", "Y"),
                "coef": np.asarray([0.1, -0.2, 0.3, 0.4, 0.5]),
            },
        )
        _write_run(
            root,
            "cov_convex_horizon",
            {
                "kind": "convex",
                "mode": "horizon",
                "signals": ("V", "C"),
                "weights": np.asarray([[0.7, 0.3], [0.6, 0.4]]),
            },
        )
        _write_run(
            root,
            "full_ridge_shared",
            {
                "kind": "ridge",
                "mode": "shared",
                "signals": ("V",),
                "coef": np.asarray([0.2]),
            },
            pipeline_config={"retrieval.scope": "other_users"},
        )

        output_dir = root / "tables" / "chronos2" / "coefficients"
        outputs = export_baseline_coefficient_plots(
            root,
            output_dir,
            datasets=["Electricity"],
            settings=["168_24"],
            models=["chronos2"],
            pipelines=[
                "baselines/instance_euclidean_3_online/cov_y_ridge_shared",
                "baselines/instance_euclidean_3_online/cov_convex_horizon",
                "baselines/instance_euclidean_3_online_other_users/full_ridge_shared",
            ],
        )
        assert len(outputs) == 7
        coefficient_csv = (
            output_dir
            / "Electricity"
            / "168_24"
            / "instance_euclidean_3_online"
            / "cov_y_ridge_shared.csv"
        )
        header = coefficient_csv.read_text(encoding="utf-8").splitlines()[0]
        assert header == "horizon,V,C,Y_1,Y_2,Y_3"
        assert coefficient_csv.with_suffix(".png").is_file()
        assert coefficient_csv.with_name("cov_convex_horizon.csv").is_file()
        assert coefficient_csv.with_name("cov_convex_horizon.png").is_file()
        assert (
            output_dir
            / "Electricity"
            / "168_24"
            / "instance_euclidean_3_online_other_users"
            / "full_ridge_shared.csv"
        ).is_file()
        with (output_dir / "coefficient_index.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert {row["baseline"] for row in rows} == {
            "cov_y_ridge_shared",
            "cov_convex_horizon",
            "full_ridge_shared",
        }
        assert (output_dir / "report_manifest.json").is_file()

        export_baseline_coefficient_plots(
            root,
            output_dir,
            datasets=["Electricity"],
            settings=["168_24"],
            models=["chronos2"],
            families=["baselines", "gates"],
            spaces=["instance"],
            neighbors=[3],
            variants=[
                "cov_y_ridge_shared",
                "cov_forecast",
                "catboost_cov_regressor_shared",
            ],
        )
        with (output_dir / "coefficient_index.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert [row["baseline"] for row in rows] == ["cov_y_ridge_shared"]

        export_baseline_coefficient_plots(
            root,
            output_dir,
            datasets=["Electricity"],
            settings=["168_24"],
            models=["chronos2"],
            families=["ts_ifa"],
            pipelines=[
                "ts_ifa/instance_euclidean_3_online/"
                "joint_ridge_horizon_unconstrained_full"
            ],
        )
        assert coefficient_csv.is_file()
        assert coefficient_csv.with_suffix(".png").is_file()

    print("baseline coefficient plot checks passed")


if __name__ == "__main__":
    main()
