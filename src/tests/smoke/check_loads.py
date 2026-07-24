"""Small local/cluster smoke checks for data and model loading."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_dataset import load_csv_dataset, resolve_csv_path, split_bounds  # noqa: E402
from src.data.neighbors import aligned_store_dates, build_window_batch, period_eval_dates  # noqa: E402
from src.experiments.extraction import context_on_query_scale  # noqa: E402
from src.models.chronos_model import Chronos  # noqa: E402
from src.models.models import ForecastModel, Linear, load_pretrained_model, parameter_counts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=Path(__file__).with_name("tiny_timeseries.csv"))
    parser.add_argument("--check-patchtst", action="store_true")
    parser.add_argument("--chronos-weights", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "electricity.CSV"
        csv.write_text("date,user_a\n2020-01-01,1\n", encoding="utf-8")
        assert resolve_csv_path(tmp, "Electricity") == csv.resolve()
        assert resolve_csv_path(Path(tmp) / "ELECTRICITY.csv") == csv.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tiny.csv").write_text(
            "date,user_a,user_b,user_c,user_d\n2020-01-01,1,2,3,4\n2020-01-02,5,6,7,8\n",
            encoding="utf-8",
        )
        (root / "config.json").write_text(
            '{"drop_users": [0], "adaptation": {"drop_users": [1]}}',
            encoding="utf-8",
        )
        configured = load_csv_dataset(
            root,
            dataset_name="tiny",
            date_col="date",
            drop_users=[2],
        )
        assert configured.user_names == ["user_d"]

    dataset = load_csv_dataset(
        args.csv,
        date_col="date",
        target_cols="user_a,user_b",
    )
    assert dataset.n_dates == 12
    assert dataset.n_users == 2

    lags = 4
    horizon = 2
    x, y = dataset.window_tensor(3, lags, horizon)
    assert x.shape == (2, 1, lags)
    assert y.shape == (2, 1, horizon)
    torch.testing.assert_close(x[0, 0], torch.tensor([10.0, 11.0, 12.0, 13.0]))
    torch.testing.assert_close(y[0, 0], torch.tensor([14.0, 15.0]))

    persistence = load_pretrained_model(
        "persistence",
        lags=lags,
        horizon=horizon,
        device="cpu",
    )
    pred = persistence(x)
    assert pred.shape == y.shape

    raw_windows = build_window_batch(
        dataset,
        np.asarray([3, 4]),
        lags=lags,
        horizon=horizon,
        distance_space="raw",
    )
    instance_windows = build_window_batch(
        dataset,
        np.asarray([3, 4]),
        lags=lags,
        horizon=horizon,
        distance_space="instance",
    )
    np.testing.assert_allclose(instance_windows.features.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(instance_windows.features.std(axis=1), 1.0, atol=1e-6)
    assert not np.allclose(raw_windows.features, instance_windows.features)

    class EncoderStub(torch.nn.Module):
        def representation(self, values: torch.Tensor, *, pool: bool = False) -> torch.Tensor:
            del pool
            return values.mean(dim=-1)

    encoder_windows = build_window_batch(
        dataset,
        np.asarray([3, 4]),
        lags=lags,
        horizon=horizon,
        distance_space="encoder",
        model=EncoderStub(),
        device="cpu",
    )
    np.testing.assert_allclose(
        encoder_windows.features[:, 0],
        raw_windows.features.mean(axis=1),
    )

    linear = ForecastModel(Linear(lags=lags, horizon=horizon))
    assert parameter_counts(linear) == (lags * horizon + horizon, lags * horizon + horizon)

    chronos_stub = Chronos.__new__(Chronos)
    torch.nn.Module.__init__(chronos_stub)
    chronos_stub.lags = lags
    chronos_stub.horizon = horizon
    chronos_stub.context_mode = "future_included"
    chronos_stub.shared_context = False
    chronos_stub.pipeline = SimpleNamespace(model=torch.nn.Linear(3, 2))
    assert parameter_counts(chronos_stub) == (8, 8)
    context = torch.zeros(2, 3, lags + horizon)
    chronos_inputs = chronos_stub._prepare_inputs(
        x,
        context,
        past_covariates=None,
        future_covariates=None,
    )
    for item in chronos_inputs:
        past_keys = set(item["past_covariates"])
        future_keys = set(item["future_covariates"])
        assert future_keys <= past_keys
        assert {"context_0", "context_1", "context_2"} == future_keys

    scaled_context = context_on_query_scale(
        torch.tensor([[3.0, 7.0]]),
        torch.tensor([[[8.0, 12.0, 14.0, 16.0]]]),
        lags=2,
    )
    torch.testing.assert_close(scaled_context, torch.tensor([[[3.0, 7.0, 9.0, 11.0]]]))

    assert split_bounds(100, "0.3,0.5,0.2") == (30, 80, 100)
    eval_dates = period_eval_dates(
        80,
        100,
        n_dates=100,
        lags=4,
        horizon=2,
        stride=1,
    )
    assert eval_dates[0] == 79
    assert eval_dates[-1] == 97
    fixed_dates = aligned_store_dates(
        80,
        lags=4,
        horizon=2,
        train_stride=1,
        n_users=2,
        period=1,
        store_start=0,
        store_end=30,
        online=False,
    )
    online_dates = aligned_store_dates(
        80,
        lags=4,
        horizon=2,
        train_stride=1,
        n_users=2,
        period=1,
        store_start=0,
        store_end=30,
        min_store_dates=len(fixed_dates),
        max_store_dates=len(fixed_dates),
    )
    assert len(online_dates) == len(fixed_dates)
    assert online_dates[-1] == 80 - 2
    assert np.all((80 - online_dates) % 1 == 0)
    assert np.all(online_dates + horizon <= 80)
    warmup_dates = aligned_store_dates(
        20,
        lags=4,
        horizon=2,
        train_stride=1,
        n_users=2,
        period=1,
        store_start=0,
        store_end=30,
        min_store_dates=len(fixed_dates),
        max_store_dates=len(fixed_dates),
    )
    assert len(warmup_dates) == 0
    try:
        aligned_store_dates(
            80,
            lags=4,
            horizon=2,
            datastore_stride=5,
            n_users=2,
            period=2,
            store_start=0,
            store_end=30,
            online=False,
        )
    except ValueError as exc:
        assert "multiple of period" in str(exc)
    else:
        raise AssertionError("unaligned datastore stride should be rejected")

    if args.check_patchtst:
        patchtst = load_pretrained_model(
            "patchtst",
            lags=lags,
            horizon=horizon,
            device="cpu",
            model_kwargs={"patch_len": 2, "stride": 1, "n_heads": 4},
        )
        assert patchtst(x).shape == y.shape
        assert patchtst.representation(x).shape[0] == x.shape[0]

    if args.chronos_weights:
        chronos = load_pretrained_model(
            "chronos",
            lags=lags,
            horizon=horizon,
            device="cpu",
            model_kwargs={
                "weights_path": args.chronos_weights,
                "device_map": "cpu",
                "context_mode": "future_included",
            },
        )
        assert chronos(x).shape == y.shape

    print("smoke checks passed")


if __name__ == "__main__":
    main()
