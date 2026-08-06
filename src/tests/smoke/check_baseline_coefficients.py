"""Smoke-check fitted baseline coefficient CSV and heatmap exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.visu.baseline_coefficients import export_baseline_coefficient_plots


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline_dir = (
            root
            / "Electricity"
            / "168_24"
            / "chronos2"
            / "instance_euclidean_3_online"
            / "baselines"
        )
        baseline_dir.mkdir(parents=True)
        torch.save(
            {
                "format": "adaptation_baseline_models",
                "models": {
                    "cov_y_ridge_shared": {
                        "kind": "ridge",
                        "mode": "shared",
                        "signals": ("V", "C", "Y"),
                        "coef": np.asarray([0.1, -0.2, 0.3, 0.4, 0.5]),
                    },
                    "cov_convex_horizon": {
                        "kind": "convex",
                        "mode": "horizon",
                        "signals": ("V", "C"),
                        "weights": np.asarray([[0.7, 0.3], [0.6, 0.4]]),
                    },
                },
                "eval_fit_models": None,
            },
            baseline_dir / "baseline_artifacts.pt",
        )
        (baseline_dir / "result_manifest.json").write_text(
            json.dumps(
                {
                    "format": "adaptation_evaluation_result",
                    "family": "baselines",
                    "files": {"artifacts": "baseline_artifacts.pt"},
                }
            ),
            encoding="utf-8",
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
                "baselines/instance_euclidean_3_online/cov_forecast",
            ],
        )
        assert len(outputs) == 5
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
        with (output_dir / "coefficient_index.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert {row["baseline"] for row in rows} == {
            "cov_y_ridge_shared",
            "cov_convex_horizon",
        }

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
            pipelines=["ts_ifa/instance_euclidean_3_online/joint_ridge"],
        )
        assert coefficient_csv.is_file()
        assert coefficient_csv.with_suffix(".png").is_file()

    print("baseline coefficient plot checks passed")


if __name__ == "__main__":
    main()
