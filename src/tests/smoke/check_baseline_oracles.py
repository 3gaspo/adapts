"""Smoke-check target-aware cov oracle baselines."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.neighbors import neighbor_to_query_scale  # noqa: E402
from src.adaptors.baselines.evaluate import (  # noqa: E402
    TRAINABLE_BASELINES,
    compact_baseline_arrays,
    compact_gate_arrays,
    fit_gate,
    flatten_payload,
    horizon_gate_feature_names,
    horizon_gate_features,
    predict_gate,
    ridge_no_intercept,
    run_streamed_baselines,
    run_streamed_gates,
    scalar_gate_features,
    subsample_fit_arrays,
)
from src.experiments.prediction_store import load_prediction_store  # noqa: E402
from src.experiments.splits import chronological_resplit_arrays  # noqa: E402


def has_catboost() -> bool:
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def main() -> None:
    coefficient = ridge_no_intercept(
        np.ones((2, 1), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        l2=1.0,
    )
    np.testing.assert_allclose(coefficient, np.asarray([0.5]))

    ridge_x = np.asarray(
        [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0], [3.0, 5.0]],
        dtype=np.float64,
    )
    ridge_y = np.asarray([2.0, -1.0, 3.0, 4.0], dtype=np.float64)
    coefficient = ridge_no_intercept(ridge_x, ridge_y, l2=0.4, chunk_rows=2)
    rescaled_coefficient = ridge_no_intercept(
        100.0 * ridge_x,
        100.0 * ridge_y,
        l2=0.4,
        chunk_rows=2,
    )
    np.testing.assert_allclose(coefficient, rescaled_coefficient)

    fit_arrays = {
        "y": np.arange(20, dtype=np.float32).reshape(10, 2),
        "pred": np.arange(20, dtype=np.float32).reshape(10, 2),
    }
    sampled_a = subsample_fit_arrays(fit_arrays, 4, seed=7)
    sampled_b = subsample_fit_arrays(fit_arrays, 4, seed=7)
    assert sampled_a["y"].shape[0] == 4
    np.testing.assert_array_equal(sampled_a["y"], sampled_b["y"])
    assert subsample_fit_arrays(fit_arrays, None, seed=7) is fit_arrays
    chronological = {
        "query_t": np.repeat(np.arange(5), 2),
        "y": np.arange(20, dtype=np.float32).reshape(10, 2),
    }
    chronological_train, chronological_valid, _ = chronological_resplit_arrays(
        chronological,
        0.4,
    )
    assert np.shares_memory(chronological_train["y"], chronological["y"])
    assert np.shares_memory(chronological_valid["y"], chronological["y"])

    query = np.asarray([[3.0, 7.0]], dtype=np.float32)
    neighbor = np.asarray([[[8.0, 12.0]]], dtype=np.float32)
    horizon = np.asarray([[[14.0, 16.0]]], dtype=np.float32)
    residual = np.asarray([[[2.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        neighbor_to_query_scale(query, neighbor, horizon),
        np.asarray([[[9.0, 11.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        neighbor_to_query_scale(query, neighbor, residual, residual=True),
        np.asarray([[[2.0, 4.0]]], dtype=np.float32),
    )

    payload = {
        "train_preds": torch.tensor([[[7.0, 7.0]]]),
        "train_preds_context": torch.tensor([[[8.0, 8.0]]]),
        "train_X_values": torch.tensor([[[3.0, 7.0]]]),
        "train_Xc_values": torch.tensor([[[[8.0, 12.0]]]]),
        "train_Y_values": torch.tensor([[[9.0, 11.0]]]),
        "train_Yc_values": torch.tensor([[[[14.0, 16.0]]]]),
        "train_E_values": torch.tensor([[[[2.0, 4.0]]]]),
        "train_distance_x_xc": torch.tensor([[[0.5]]]),
        "train_query_t": torch.tensor([[42]]),
        "train_query_user_idx": torch.tensor([[3]]),
        "train_neighbor_t": torch.tensor([[[30]]]),
        "train_neighbor_user_idx": torch.tensor([[[3]]]),
    }
    flattened = flatten_payload(payload, "train")
    gate_flattened = flatten_payload(payload, "train", family="gates")
    baseline_flattened = flatten_payload(payload, "train", family="baselines")
    assert "pred_neighbors" not in gate_flattened
    assert "e" not in baseline_flattened
    np.testing.assert_allclose(flattened["y_c"], np.asarray([[[9.0, 11.0]]]))
    np.testing.assert_allclose(flattened["e"], np.asarray([[[2.0, 4.0]]]))
    np.testing.assert_allclose(flattened["pred_neighbors"], np.asarray([[[7.0, 7.0]]]))
    np.testing.assert_allclose(flattened["neighbor_lookback_mean"], np.asarray([10.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_mean_std"], np.asarray([0.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_std"], np.asarray([2.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_std_std"], np.asarray([0.0]))
    np.testing.assert_allclose(flattened["same_user_ratio"], np.asarray([1.0]))
    np.testing.assert_allclose(flattened["neighbor_age_mean"], np.asarray([12.0]))
    np.testing.assert_allclose(flattened["neighbor_age_std"], np.asarray([0.0]))
    scalar_features = scalar_gate_features(flattened)
    horizon_features = horizon_gate_features(flattened)
    np.testing.assert_allclose(
        scalar_features[0, :6],
        np.asarray([1.0, 0.0, 3.0, 0.0, 1.0, 3.0]),
    )
    np.testing.assert_allclose(
        horizon_features[0][0, :5],
        np.asarray([1.0, 2.0, 0.0, 2.0, 0.0]),
    )
    np.testing.assert_allclose(
        horizon_features[1][0, :5],
        np.asarray([1.0, 4.0, 0.0, 4.0, 0.0]),
    )
    assert scalar_features.shape[1] == 20
    assert len(horizon_features) == 2
    assert horizon_features[0].shape[1] == len(horizon_gate_feature_names(2))

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        gate_arrays = compact_gate_arrays(flattened)
        gate_rows, gate_artifacts, _ = run_streamed_gates(
            {"adapt": gate_arrays, "eval": gate_arrays},
            {
                "T1": gate_arrays,
                "T2": gate_arrays,
                "T1+T2": gate_arrays,
                "T3_oracle": gate_arrays,
            },
            output_dir=root / "gates",
            selected_methods=(
                "oracle_cov_shared",
                "oracle_cov_horizon",
            ),
            iterations=1,
            learning_rate=0.1,
            depth=1,
            early_stopping_rounds=1,
            seed=1,
            task_type="CPU",
            devices=None,
            thread_count=1,
            feature_importance_top_k=1,
        )
        gate_store = load_prediction_store(root / "gates")
        gate_predictions = gate_store["splits"]["eval"]["predictions"]
        assert set(gate_predictions) == {
            "vanilla",
            "oracle_cov_shared",
            "oracle_cov_horizon",
        }
        np.testing.assert_array_equal(
            gate_predictions["oracle_cov_shared"],
            gate_arrays["pred_c"],
        )
        np.testing.assert_array_equal(
            gate_predictions["oracle_cov_horizon"],
            gate_arrays["pred_c"],
        )
        assert gate_artifacts["format"] == "adaptation_gate_models"
        gate_metrics = {row["baseline"]: row for row in gate_rows}
        assert gate_metrics["vanilla"]["positive_window_pct"] == 0.0
        assert gate_metrics["oracle_cov_shared"]["positive_window_pct"] == 100.0

        baseline_arrays = compact_baseline_arrays(flattened)
        _, baseline_artifacts, _ = run_streamed_baselines(
            {"adapt": baseline_arrays, "eval": baseline_arrays},
            {
                "T1": baseline_arrays,
                "T2": baseline_arrays,
                "T1+T2": baseline_arrays,
                "T3_oracle": baseline_arrays,
            },
            output_dir=root / "baselines",
            selected_methods=TRAINABLE_BASELINES,
            l2_grid=(1e-3,),
            fit_on_eval=True,
        )
        baseline_store = load_prediction_store(root / "baselines")
        baseline_predictions = baseline_store["splits"]["eval"]["predictions"]
        assert {
            f"{name}_eval_fit" for name in TRAINABLE_BASELINES
        } <= set(baseline_predictions)
        assert set(baseline_artifacts["models"]) == set(TRAINABLE_BASELINES)
        del baseline_predictions, baseline_store, gate_predictions, gate_store
        gc.collect()

    gate_x = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    gate_y = np.asarray([[-4.0], [-1.0], [1.0], [4.0]], dtype=np.float32)
    if has_catboost():
        gate = fit_gate(
            gate_x,
            gate_y,
            iterations=50,
            learning_rate=0.1,
            depth=2,
            seed=1,
        )
        differences = predict_gate(gate, gate_x)
        assert differences.shape == (gate_y.shape[0],)
        assert differences[:2].mean() < 0.0 < differences[2:].mean()

        classifier = fit_gate(
            gate_x,
            gate_y,
            iterations=50,
            learning_rate=0.1,
            depth=2,
            seed=1,
            objective="classifier",
        )
        classifier_scores = predict_gate(classifier, gate_x)
        assert classifier_scores.shape == (gate_y.shape[0],)
        assert classifier_scores[:2].mean() < 0.0 < classifier_scores[2:].mean()

    constant_classifier = fit_gate(
        gate_x,
        np.ones_like(gate_y),
        iterations=1,
        learning_rate=0.1,
        depth=1,
        seed=1,
        objective="classifier",
    )
    np.testing.assert_array_equal(
        predict_gate(constant_classifier, gate_x),
        np.full(gate_y.shape[0], 0.5, dtype=np.float32),
    )
    print("baseline oracle checks passed")


if __name__ == "__main__":
    main()
