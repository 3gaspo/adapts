"""Smoke-test all four TS-IFA router/optimization contracts."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch
from einops import repeat


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptors.ts_ifa.model import (  # noqa: E402
    TSIFA_ARCHITECTURE,
    TSIFA_VARIANTS,
    TSIFAConfig,
    TimeSeriesInformedForecastingAdapter,
)
from src.adaptors.ts_ifa.train import differentiable_horizon_ridge, main  # noqa: E402
from src.experiments.prediction_store import load_prediction_store  # noqa: E402


def make_payload(prefix: str) -> dict[str, torch.Tensor]:
    n_dates, n_users, neighbors, lags, horizon = 5, 2, 2, 6, 2
    x = torch.randn(n_dates, n_users, lags)
    x_c = torch.randn(n_dates, n_users, neighbors, lags)
    y = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    y = y + 0.1 * torch.randn_like(y)
    y_c = repeat(x_c[..., -1], "date user neighbor -> date user neighbor horizon", horizon=horizon)
    pred_neighbors = y_c - 0.1 * torch.randn_like(y_c)
    preds = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    return {
        f"{prefix}_preds": preds,
        f"{prefix}_preds_context": preds + 0.05 * torch.randn_like(preds),
        f"{prefix}_E_values": y_c - pred_neighbors,
        f"{prefix}_X_values": x,
        f"{prefix}_Xc_values": x_c,
        f"{prefix}_Y_values": y,
        f"{prefix}_Yc_values": y_c,
    }


def random_batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(4, 6),
        "x_c": torch.randn(4, 2, 6),
        "y": torch.randn(4, 2),
        "y_c": torch.randn(4, 2, 2),
        "pred": torch.randn(4, 2),
        "pred_cov": torch.randn(4, 2),
        "pred_neighbors": torch.randn(4, 2, 2),
        "residual_c": torch.randn(4, 2, 2),
    }


def check_models() -> None:
    batch = random_batch()
    for rooter_form in ("ridge", "neural"):
        model = TimeSeriesInformedForecastingAdapter(
            TSIFAConfig(
                lags=6,
                horizon=2,
                neighbors=2,
                rooter_form=rooter_form,
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
        outputs = model(batch)
        torch.testing.assert_close(outputs["prediction"], batch["pred"])
        torch.testing.assert_close(outputs["residual_prediction"], batch["pred"])
        torch.testing.assert_close(outputs["memory_prediction"], batch["pred"])
        torch.testing.assert_close(outputs["coefficients"], torch.zeros_like(outputs["coefficients"]))
        rooter_parameters = sum(parameter.numel() for parameter in model.rooter_parameters())
        assert rooter_parameters == 6 if rooter_form == "ridge" else rooter_parameters > 6

    design = torch.randn(7, 3, 2, requires_grad=True)
    differentiable_horizon_ridge(design, torch.randn(7, 2), alpha=0.1, eps=1e-8).square().mean().backward()
    assert design.grad is not None and torch.isfinite(design.grad).all()


def run_variant(base: Path, variant: str, adapt_path: Path, eval_path: Path) -> None:
    out = base / variant
    old_argv = sys.argv
    try:
        sys.argv = [
            "src.adaptors.ts_ifa.train",
            "--adapt-payload", str(adapt_path),
            "--eval-payload", str(eval_path),
            "--output-dir", str(out),
            "--variant", variant,
            "--train-epochs", "1",
            "--rooter-epochs", "1",
            "--batch-size", "4",
            "--device", "cpu",
            "--residual-heads", "2",
            "--memory-heads", "2",
            "--rooter-heads", "2",
            "--residual-attn-dim", "8",
            "--memory-attn-dim", "8",
            "--rooter-attn-dim", "8",
            "--residual-hidden", "16",
            "--memory-hidden", "16",
            "--rooter-hidden", "16",
        ]
        paths = main()
    finally:
        sys.argv = old_argv
    assert all(Path(path).exists() for path in paths.values())
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert config["architecture"] == TSIFA_ARCHITECTURE
    assert config["variant"] == manifest["variant"] == variant
    assert config["rooter_form"] == variant.split("_")[1]
    assert config["optimization"] == variant.split("_")[0]
    rooter = torch.load(paths["rooter"], map_location="cpu", weights_only=False)
    if variant == "joint_ridge":
        assert rooter["fit"] == "gradient_updates_on_T1"
        assert torch.count_nonzero(rooter["coefficients"]).item() > 0
    if variant == "meta_ridge":
        assert rooter["fit"] == "closed_form_on_T2"
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert math.isfinite(metrics["adapted_nmse"])
    store = load_prediction_store(out)
    assert set(store["splits"]["eval"]["predictions"]) == {
        "ts_ifa_adapted",
        "ts_ifa_cov_branch",
        "ts_ifa_memory_branch",
        "ts_ifa_residual_branch",
        "ts_ifa_vanilla_branch",
    }
    assert set(store["splits"]["eval"]["gate_diagnostics"]) == {"rooter_coefficients"}


def run() -> None:
    check_models()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        adapt_path = base / "adapt_prediction_payload.pt"
        eval_path = base / "eval_prediction_payload.pt"
        torch.save(make_payload("adapt"), adapt_path)
        torch.save(make_payload("eval"), eval_path)
        for variant in TSIFA_VARIANTS:
            run_variant(base, variant, adapt_path, eval_path)
    print("TS-IFA four-variant smoke checks passed")


if __name__ == "__main__":
    run()
