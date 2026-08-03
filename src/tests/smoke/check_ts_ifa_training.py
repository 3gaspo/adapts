"""Smoke-test the staged TS-IFA training path on synthetic payloads."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from einops import repeat


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptors.ts_ifa.model import (  # noqa: E402
    CANDIDATE_NAMES,
    TSIFAConfig,
    TimeSeriesInformedForecastingAdapter,
)
from src.adaptors.ts_ifa.train import (  # noqa: E402
    PredictionPayloadDataset,
    branch_loss_components,
    main,
    prepare_batch,
    rooter_loss_components,
)
from src.experiments.prediction_store import load_prediction_store  # noqa: E402


def make_payload(prefix: str) -> dict[str, torch.Tensor]:
    n_dates, n_users, neighbors, lags, horizon = 5, 3, 2, 6, 2
    x = torch.randn(n_dates, n_users, lags)
    x_c = torch.randn(n_dates, n_users, neighbors, lags)
    y = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    y = y + 0.1 * torch.randn(n_dates, n_users, horizon)
    y_c = repeat(
        x_c[..., -1],
        "date user neighbor -> date user neighbor horizon",
        horizon=horizon,
    )
    y_c = y_c + 0.1 * torch.randn(n_dates, n_users, neighbors, horizon)
    preds = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    preds_context = preds + 0.05 * torch.randn_like(preds)
    preds_transformed = preds + 0.03 * torch.randn_like(preds)
    pred_neighbors = repeat(
        x_c[..., -1],
        "date user neighbor -> date user neighbor horizon",
        horizon=horizon,
    )
    return {
        f"{prefix}_preds": preds,
        f"{prefix}_preds_context": preds_context,
        f"{prefix}_preds_transformed": preds_transformed,
        f"{prefix}_E_values": y_c - pred_neighbors,
        f"{prefix}_X_values": x,
        f"{prefix}_Xc_values": x_c,
        f"{prefix}_Y_values": y,
        f"{prefix}_Yc_values": y_c,
    }


def check_query_scale_transfer() -> None:
    payload = {
        "train_preds": torch.tensor([[[7.0]]]),
        "train_preds_context": torch.tensor([[[7.0]]]),
        "train_preds_transformed": torch.tensor([[[8.0]]]),
        "train_E_values": torch.tensor([[[[2.0]]]]),
        "train_X_values": torch.tensor([[[3.0, 7.0]]]),
        "train_Xc_values": torch.tensor([[[[8.0, 12.0]]]]),
        "train_Y_values": torch.tensor([[[7.0]]]),
        "train_Yc_values": torch.tensor([[[[14.0]]]]),
    }
    dataset = PredictionPayloadDataset(
        payload,
        prefix="train",
        use_transformed_prediction=True,
    )
    expected_values = {
        "x_c": torch.tensor([[[3.0, 7.0]]]),
        "y_c": torch.tensor([[[9.0]]]),
        "pred_neighbors": torch.tensor([[[7.0]]]),
        "residual_c": torch.tensor([[[2.0]]]),
    }
    for name, expected in expected_values.items():
        torch.testing.assert_close(dataset.tensors[name], expected)

    batch, _ = prepare_batch(dataset.tensors, normalization="instance", eps=1e-8)
    torch.testing.assert_close(batch["x_c"], torch.tensor([[[-1.0, 1.0]]]))
    torch.testing.assert_close(batch["y_c"], torch.tensor([[[2.0]]]))
    torch.testing.assert_close(batch["pred_neighbors"], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(batch["residual_c"], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(batch["pred_transformed"], torch.tensor([[1.5]]))
    payload_without_transformed = {
        key: value
        for key, value in payload.items()
        if key != "train_preds_transformed"
    }
    default_dataset = PredictionPayloadDataset(payload_without_transformed, prefix="train")
    assert default_dataset.has_transformed_prediction is False
    torch.testing.assert_close(
        default_dataset.tensors["pred_transformed"],
        default_dataset.tensors["pred"],
    )


def small_model() -> TimeSeriesInformedForecastingAdapter:
    return TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=6,
            horizon=2,
            neighbors=2,
            residual_heads=2,
            memory_heads=2,
            rooter_heads=2,
            residual_attn_dim=8,
            memory_attn_dim=8,
            rooter_attn_dim=8,
            residual_hidden=16,
            memory_hidden=16,
            rooter_hidden=16,
        )
    )


def random_batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(4, 6),
        "x_c": torch.randn(4, 2, 6),
        "y": torch.randn(4, 2),
        "y_c": torch.randn(4, 2, 2),
        "pred": torch.randn(4, 2),
        "pred_cov": torch.randn(4, 2),
        "pred_transformed": torch.randn(4, 2),
        "pred_neighbors": torch.randn(4, 2, 2),
        "residual_c": torch.randn(4, 2, 2),
    }


def check_vanilla_anchoring_initialization() -> None:
    torch.manual_seed(1)
    model = small_model()
    batch = random_batch()
    outputs = model(batch)
    assert model.candidate_names == ("vanilla", "cov", "residual", "memory")
    assert outputs["candidates"].shape == (4, len(model.candidate_names), 2)
    torch.testing.assert_close(outputs["coefficients"], torch.zeros_like(outputs["coefficients"]))
    torch.testing.assert_close(outputs["residual_delta"], torch.zeros_like(outputs["residual_delta"]))
    torch.testing.assert_close(outputs["residual_prediction"], batch["pred"])
    torch.testing.assert_close(outputs["memory_prediction"], batch["pred"])
    torch.testing.assert_close(outputs["prediction"], batch["pred"])

    model.set_trainable_stage("branches")
    assert all(parameter.requires_grad for parameter in model.branch_parameters())
    assert not any(parameter.requires_grad for parameter in model.rooter_parameters())
    model.set_trainable_stage("rooter")
    assert not any(parameter.requires_grad for parameter in model.branch_parameters())
    assert all(parameter.requires_grad for parameter in model.rooter_parameters())

    fixed_model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=6,
            horizon=2,
            neighbors=2,
            residual_heads=2,
            memory_heads=2,
            rooter_heads=2,
            residual_attn_dim=8,
            memory_attn_dim=8,
            rooter_attn_dim=8,
            residual_hidden=16,
            memory_hidden=16,
            rooter_hidden=16,
            precomputed_transformed_expert=True,
        )
    )
    assert fixed_model.candidate_names == CANDIDATE_NAMES
    fixed_outputs = fixed_model(batch)
    torch.testing.assert_close(
        fixed_outputs["transformed_prediction"],
        batch["pred_transformed"],
    )

    learned_batch = {**batch, "pred_transformed": batch["pred"]}
    learned_model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=6,
            horizon=2,
            neighbors=2,
            residual_heads=2,
            memory_heads=2,
            rooter_heads=2,
            residual_attn_dim=8,
            memory_attn_dim=8,
            rooter_attn_dim=8,
            residual_hidden=16,
            memory_hidden=16,
            rooter_hidden=16,
            transformed_hidden=16,
            learnable_transformed_covariate=True,
        )
    )
    learned_outputs = learned_model(learned_batch)
    assert "transformed_delta" in learned_outputs
    torch.testing.assert_close(
        learned_outputs["transformed_prediction"],
        learned_batch["pred"],
    )


def check_loss_components() -> None:
    model = small_model()
    batch = random_batch()
    outputs = model(batch)
    state = {"loss_scale": torch.ones((4, 1))}
    branch = branch_loss_components(
        outputs,
        batch,
        state,
        vanilla_anchor=0.5,
    )
    expected_branch = branch["residual"] + branch["memory"] + 0.5 * branch["vanilla_anchoring"]
    torch.testing.assert_close(branch["loss"], expected_branch)

    rooter = rooter_loss_components(
        outputs,
        batch,
        state,
        vanilla_anchor=0.5,
        coefficient_l2=0.25,
        horizon_smoothness=0.125,
    )
    expected_rooter = (
        rooter["prediction"]
        + 0.5 * rooter["vanilla_anchoring"]
        + 0.25 * rooter["coefficient_l2"]
        + 0.125 * rooter["horizon_smoothness"]
    )
    torch.testing.assert_close(rooter["loss"], expected_rooter)

    learned_model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=6,
            horizon=2,
            neighbors=2,
            residual_heads=2,
            memory_heads=2,
            rooter_heads=2,
            residual_attn_dim=8,
            memory_attn_dim=8,
            rooter_attn_dim=8,
            residual_hidden=16,
            memory_hidden=16,
            rooter_hidden=16,
            transformed_hidden=16,
            learnable_transformed_covariate=True,
        )
    )
    learned_batch = {**batch, "pred_transformed": batch["pred"]}
    learned_branch = branch_loss_components(
        learned_model.forward_branches(learned_batch),
        learned_batch,
        state,
        vanilla_anchor=0.5,
    )
    assert "transformed" in learned_branch


def check_reference_parameter_count() -> None:
    model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(lags=512, horizon=64, neighbors=15)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    branches = sum(parameter.numel() for parameter in model.branch_parameters())
    rooter = sum(parameter.numel() for parameter in model.rooter_parameters())
    assert branches == 768_064
    assert rooter == 471_232
    assert total == 1_239_296

    fixed_model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=512,
            horizon=64,
            neighbors=15,
            precomputed_transformed_expert=True,
        )
    )
    assert sum(parameter.numel() for parameter in fixed_model.parameters()) == 1_239_424

    learned_model = TimeSeriesInformedForecastingAdapter(
        TSIFAConfig(
            lags=512,
            horizon=64,
            neighbors=15,
            learnable_transformed_covariate=True,
        )
    )
    assert sum(parameter.numel() for parameter in learned_model.branch_parameters()) == 866_752
    assert sum(parameter.numel() for parameter in learned_model.parameters()) == 1_338_112


def run() -> None:
    check_query_scale_transfer()
    check_vanilla_anchoring_initialization()
    check_loss_components()
    check_reference_parameter_count()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        adapt_path = base / "adapt_prediction_payload.pt"
        eval_path = base / "eval_prediction_payload.pt"
        out = base / "ts_ifa"
        torch.save(make_payload("adapt"), adapt_path)
        torch.save(make_payload("eval"), eval_path)
        old_argv = sys.argv
        try:
            sys.argv = [
                "src.adaptors.ts_ifa.train",
                "--adapt-payload",
                str(adapt_path),
                "--eval-payload",
                str(eval_path),
                "--output-dir",
                str(out),
                "--branch-epochs",
                "1",
                "--rooter-epochs",
                "1",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--residual-heads",
                "2",
                "--memory-heads",
                "2",
                "--rooter-heads",
                "2",
                "--residual-attn-dim",
                "8",
                "--memory-attn-dim",
                "8",
                "--rooter-attn-dim",
                "8",
                "--residual-hidden",
                "16",
                "--memory-hidden",
                "16",
                "--rooter-hidden",
                "16",
            ]
            paths = main()
        finally:
            sys.argv = old_argv
        for path in paths.values():
            assert Path(path).exists(), path
        prediction_store = load_prediction_store(out)
        assert set(prediction_store["splits"]["eval"]["predictions"]) == {
            "ts_ifa_adapted",
            "ts_ifa_cov_branch",
            "ts_ifa_memory_branch",
            "ts_ifa_residual_branch",
            "ts_ifa_ridge_rooter",
            "ts_ifa_vanilla_branch",
        }
        coefficient_diagnostics = prediction_store["splits"]["eval"]["gate_diagnostics"]
        assert set(coefficient_diagnostics) == {"neural_rooter_coefficients"}
        assert coefficient_diagnostics["neural_rooter_coefficients"].shape == (
            15,
            4,
            2,
        )
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        assert config["parameters"]["total"] > 0
        assert config["parameters"]["trainable"] == config["parameters"]["total"]
        assert config["parameters"]["ridge_rooter"] == 2 * 4
        assert config["training"]["branch_train_split"] == "T1"
        assert config["training"]["rooter_train_split"] == "T2"
        assert config["training"]["final_eval_split"] == "T3"
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        for name in (
            "adapted_nmse",
            "ridge_rooter_nmse",
            "vanilla_branch_nmse",
            "cov_branch_nmse",
            "residual_branch_nmse",
            "memory_branch_nmse",
        ):
            assert name in metrics
        assert "transformed_branch_nmse" not in metrics
        history = json.loads(paths["history"].read_text(encoding="utf-8"))
        assert history["branch_history"][0]["stage"] == "branches"
        assert history["rooter_history"][0]["stage"] == "rooter"
    print("TS-IFA training smoke checks passed")


if __name__ == "__main__":
    run()
