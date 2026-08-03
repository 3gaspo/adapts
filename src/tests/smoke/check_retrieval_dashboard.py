"""Smoke-test artifact loading and dashboard calculations without a notebook kernel."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visu.dashboard import (  # noqa: E402
    baseline_feature_importance,
    gate_importance_options,
    gate_roc,
    gate_threshold_sweep,
    horizon_values,
    load_dashboard_data,
    plot_baseline_feature_importance,
    plot_gate_feature_importance,
    plot_query_example,
    plot_ts_ifa_coefficients,
    plot_window_metric_scatter,
    prediction_names,
    split_arrays,
    ts_ifa_coefficients,
)
from src.experiments.artifacts import write_extraction_manifest  # noqa: E402
from src.experiments.prediction_store import PredictionStore  # noqa: E402


def extraction_payload(prefix: str) -> dict[str, torch.Tensor]:
    dates, users, neighbors, lags, horizon = 3, 2, 2, 4, 3
    x = torch.arange(dates * users * lags, dtype=torch.float32).reshape(dates, users, lags)
    x_c = torch.stack([x - 2.0, x + 2.0], dim=2)
    y = x[..., -1:].repeat(1, 1, horizon) + torch.arange(horizon)
    y_c = x_c[..., -1:].repeat(1, 1, 1, horizon) + torch.arange(horizon)
    query_t = torch.arange(10, 10 + dates).unsqueeze(1).repeat(1, users)
    query_user = torch.arange(users).unsqueeze(0).repeat(dates, 1)
    return {
        f"{prefix}_X_values": x,
        f"{prefix}_Y_values": y,
        f"{prefix}_Xc_values": x_c,
        f"{prefix}_Yc_values": y_c,
        f"{prefix}_E_values": y_c - x_c[..., -1:].repeat(1, 1, 1, horizon),
        f"{prefix}_preds": y + 1.0,
        f"{prefix}_preds_context": y + 0.25,
        f"{prefix}_query_t": query_t,
        f"{prefix}_query_user_idx": query_user,
        f"{prefix}_neighbor_t": query_t.unsqueeze(-1).repeat(1, 1, neighbors) - 1,
        f"{prefix}_neighbor_user_idx": query_user.unsqueeze(-1).repeat(1, 1, neighbors),
        f"{prefix}_distance_x_xc": torch.ones(dates, users, neighbors),
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extraction_root = root / "extraction"
        result_root = root / "results"
        extraction_root.mkdir()
        payloads = {split: extraction_payload(split) for split in ("adapt", "eval")}
        for split, payload in payloads.items():
            torch.save(payload, extraction_root / f"{split}_prediction_payload.pt")
            torch.save({"split": split}, extraction_root / f"{split}_features_payload.pt")
        (extraction_root / "extraction_timing.json").write_text("{}", encoding="utf-8")
        write_extraction_manifest(
            extraction_root,
            signature={"neighbors": 2, "dataset": "synthetic"},
            required_files=[
                "adapt_prediction_payload.pt",
                "eval_prediction_payload.pt",
                "adapt_features_payload.pt",
                "eval_features_payload.pt",
                "extraction_timing.json",
            ],
        )

        baseline_predictions = {}
        gate_predictions = {}
        diagnostics_by_split = {}
        for split, payload in payloads.items():
            target = payload[f"{split}_Y_values"].reshape(-1, 3)
            score = torch.linspace(-1.0, 1.0, len(target))
            baseline_predictions[split] = {
                "vanilla": target + 1.0,
                "cov_forecast": target + 0.25,
                "avgy": target + 0.5,
            }
            gate_predictions[split] = {
                "catboost_cov_classifier_shared": target + 0.2,
                "oracle_cov_shared": target + 0.1,
            }
            diagnostics_by_split[split] = {
                "catboost_cov_classifier_shared_score": score,
                "cov_shared_target": score,
            }
        baseline_dir = result_root / "baselines"
        baseline_dir.mkdir(parents=True)
        baseline_store = PredictionStore(baseline_dir)
        for split, predictions in baseline_predictions.items():
            for name, value in predictions.items():
                baseline_store.write(split, "predictions", name, value.numpy())
        baseline_store.finalize(metadata={"family": "baselines"})
        torch.save(
            {
                "format": "adaptation_baseline_models",
                "models": {
                    "cov_ridge_shared": {
                        "kind": "ridge",
                        "mode": "shared",
                        "signals": ("V", "C"),
                        "coef": np.asarray([0.75, 0.25]),
                    },
                    "avgy_mix_horizon": {
                        "kind": "lambda",
                        "mode": "horizon",
                        "lambda": np.asarray([0.2, 0.4, 0.6]),
                    },
                },
            },
            baseline_dir / "baseline_artifacts.pt",
        )
        (baseline_dir / "result_manifest.json").write_text(
            json.dumps(
                {
                    "format": "adaptation_evaluation_result",
                    "family": "baselines",
                    "files": {
                        "predictions": "prediction_manifest.json",
                        "artifacts": "baseline_artifacts.pt",
                    },
                }
            ),
            encoding="utf-8",
        )
        gate_dir = result_root / "gates"
        gate_dir.mkdir()
        gate_store = PredictionStore(gate_dir)
        for split, predictions in gate_predictions.items():
            for name, value in predictions.items():
                gate_store.write(split, "predictions", name, value.numpy())
            for name, value in diagnostics_by_split[split].items():
                gate_store.write(
                    split,
                    "gate_diagnostics",
                    name,
                    value.numpy(),
                )
        gate_store.finalize(metadata={"family": "gates"})
        gate_plot_dir = gate_dir / "plots"
        gate_plot_dir.mkdir()
        importance_path = gate_plot_dir / "feature_importance_cov_classifier_shared.csv"
        importance_path.write_text(
            "feature,importance\nquery_mean,0.7\ndistance_mean,0.3\n",
            encoding="utf-8",
        )
        (gate_dir / "gate_artifacts.json").write_text(
            json.dumps(
                {
                    "format": "adaptation_gate_models",
                    "feature_importance_files": [
                        importance_path.relative_to(gate_dir).as_posix(),
                    ],
                }
            ),
            encoding="utf-8",
        )
        (gate_dir / "result_manifest.json").write_text(
            json.dumps(
                {
                    "format": "adaptation_evaluation_result",
                    "family": "gates",
                    "files": {
                        "predictions": "prediction_manifest.json",
                        "artifacts": "gate_artifacts.json",
                    },
                }
            ),
            encoding="utf-8",
        )

        ts_ifa_dir = result_root / "ts_ifa" / "TS-IFA"
        ts_ifa_dir.mkdir(parents=True)
        target = payloads["eval"]["eval_Y_values"].reshape(-1, 3)
        ts_ifa_store = PredictionStore(ts_ifa_dir)
        ts_ifa_store.write(
            "eval",
            "predictions",
            "ts_ifa_adapted",
            (target + 0.05).numpy(),
        )
        neural_coefficients = np.linspace(
            -0.5,
            0.5,
            len(target) * 2 * 3,
            dtype=np.float32,
        ).reshape(len(target), 2, 3)
        ts_ifa_store.write(
            "eval",
            "gate_diagnostics",
            "neural_rooter_coefficients",
            neural_coefficients,
        )
        ts_ifa_store.finalize(
            metadata={"family": "ts_ifa", "candidate_names": ["cov", "memory"]}
        )
        torch.save(
            {
                "coefficients": torch.tensor(
                    [[0.1, 0.2, 0.3], [-0.2, -0.1, 0.0]],
                    dtype=torch.float32,
                ),
                "candidate_names": ["cov", "memory"],
            },
            ts_ifa_dir / "ridge_rooter.pt",
        )
        (ts_ifa_dir / "result_manifest.json").write_text(
            json.dumps(
                {
                    "format": "adaptation_ts_ifa_result",
                    "files": {
                        "predictions": "prediction_manifest.json",
                        "ridge_rooter": "ridge_rooter.pt",
                    },
                }
            ),
            encoding="utf-8",
        )

        data = load_dashboard_data(extraction_root, result_root)
        arrays = split_arrays(data, "eval")
        assert arrays["x"].shape == (6, 4)
        assert "avgy" in prediction_names(data, "eval")
        assert "catboost_cov_classifier_shared" in prediction_names(data, "eval")
        assert "ts_ifa_adapted" in prediction_names(data, "eval")

        values, _, _, window_average = horizon_values(
            data,
            "eval",
            "cov_forecast",
            "vanilla",
            "mse",
            "relative",
        )
        assert values.shape == (3,)
        assert (values < 0).all()
        assert window_average is not None

        _, _, auc, accuracy, count = gate_roc(
            data,
            "eval",
            "catboost_cov_classifier_shared",
        )
        assert auc == 1.0
        assert accuracy == 1.0
        assert count == 6
        threshold_values = gate_threshold_sweep(
            data,
            "eval",
            "catboost_cov_classifier_shared",
            points=7,
        )
        assert set(threshold_values) == {
            "threshold",
            "right_pct",
            "true_positive_rate_pct",
            "nmse",
            "relative_improvement_vanilla_pct",
        }
        assert threshold_values["threshold"].shape == (7,)

        feature_names, importance, _ = baseline_feature_importance(
            data,
            "cov_ridge_shared",
        )
        assert feature_names == ["V", "C"]
        np.testing.assert_allclose(importance, [0.75, 0.25])
        assert gate_importance_options(data) == [
            "catboost_cov_classifier_shared"
        ]
        ridge_values, candidates = ts_ifa_coefficients(data, "ridge_rooter")
        assert candidates == ["cov", "memory"]
        assert ridge_values.shape == (2, 3)
        neural_values, _ = ts_ifa_coefficients(data, "neural_rooter_mean")
        assert neural_values.shape == (2, 3)

        figure = plot_query_example(
            data,
            "eval",
            0,
            instance_normalized=True,
            hide_axes=False,
        )
        plt.close(figure)
        for figure in (
            plot_window_metric_scatter(
                data,
                "eval",
                "cov_forecast",
                "vanilla",
                "mse",
                "relative",
                "distance_mean",
                x_log_scale=True,
                y_log_scale=True,
            ),
            plot_baseline_feature_importance(data, "cov_ridge_shared"),
            plot_gate_feature_importance(
                data,
                "catboost_cov_classifier_shared",
            ),
            plot_ts_ifa_coefficients(data, "ridge_rooter"),
            plot_ts_ifa_coefficients(data, "neural_rooter_mean"),
        ):
            plt.close(figure)
    print("retrieval dashboard smoke check passed")


if __name__ == "__main__":
    main()
