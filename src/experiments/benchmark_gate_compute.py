"""Benchmark CatBoost execution strategies for an H=504 horizon gate."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.adaptors.baselines.evaluate import (
    fit_horizon_gate_models,
    flatten_payload,
    horizon_gate_features,
    subsample_fit_arrays,
    torch_load,
    weighted_neighbor_horizon,
)
from src.experiments.runtime import log_experiment_separator, setup_logging
from src.experiments.splits import chronological_resplit_arrays


LOGGER = logging.getLogger(__name__)
MODES = ("cpu_serial", "gpu_serial", "cpu_parallel")
SUMMARY_COLUMNS = (
    "fit_rank",
    "mode",
    "task_type",
    "devices",
    "thread_count",
    "horizon_jobs",
    "horizon",
    "feature_count",
    "t1_samples",
    "t2_samples",
    "refit_samples",
    "iterations",
    "constant_models",
    "mean_selected_iterations",
    "fit_seconds",
    "total_seconds",
    "horizons_per_fit_second",
    "speedup_vs_cpu_serial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time one complete H=504 horizon-gate family.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--candidate", choices=("context", "aggr_y"), default="context")
    parser.add_argument(
        "--objective",
        choices=("classifier", "regressor"),
        default="regressor",
    )
    parser.add_argument("--expected-horizon", type=int, default=504)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--max-t1-fit-samples", type=int, default=50_000)
    parser.add_argument("--max-t2-valid-samples", type=int, default=10_000)
    parser.add_argument("--max-refit-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-serial-threads", type=int, default=16)
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--gpu-data-threads", type=int, default=4)
    parser.add_argument("--cpu-parallel-jobs", type=int, default=8)
    parser.add_argument("--cpu-parallel-threads", type=int, default=2)
    return parser.parse_args()


def _execution_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "cpu_serial":
        return {
            "task_type": "CPU",
            "devices": None,
            "thread_count": args.cpu_serial_threads,
            "horizon_jobs": 1,
        }
    if args.mode == "gpu_serial":
        return {
            "task_type": "GPU",
            "devices": args.gpu_devices,
            "thread_count": args.gpu_data_threads,
            "horizon_jobs": 1,
        }
    return {
        "task_type": "CPU",
        "devices": None,
        "thread_count": args.cpu_parallel_threads,
        "horizon_jobs": args.cpu_parallel_jobs,
    }


def _candidate_prediction(
    arrays: dict[str, np.ndarray],
    candidate: str,
) -> np.ndarray:
    if candidate == "context":
        return arrays["pred_c"]
    return weighted_neighbor_horizon(arrays)


def _horizon_targets(
    arrays: dict[str, np.ndarray],
    candidate: str,
) -> np.ndarray:
    candidate_prediction = _candidate_prediction(arrays, candidate)
    vanilla_loss = (arrays["y"] - arrays["pred"]) ** 2
    candidate_loss = (arrays["y"] - candidate_prediction) ** 2
    return vanilla_loss - candidate_loss


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _write_summary(output_dir: Path) -> Path:
    rows = []
    for mode in MODES:
        path = output_dir / f"{mode}.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    cpu_serial = next(
        (row["fit_seconds"] for row in rows if row["mode"] == "cpu_serial"),
        None,
    )
    rows.sort(key=lambda row: row["fit_seconds"])
    for rank, row in enumerate(rows, start=1):
        row["fit_rank"] = rank
        row["speedup_vs_cpu_serial"] = (
            None if cpu_serial is None else float(cpu_serial / row["fit_seconds"])
        )
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {name: result.get(name) for name in SUMMARY_COLUMNS}
            for result in rows
        )
    return summary_path


def _load_adapt_arrays(input_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    adapt_path = input_dir / "adapt_prediction_payload.pt"
    if adapt_path.is_file():
        return flatten_payload(torch_load(adapt_path), "adapt"), "adapt"
    legacy_paths = (
        input_dir / "train_prediction_payload.pt",
        input_dir / "oracle_prediction_payload.pt",
    )
    if not all(path.is_file() for path in legacy_paths):
        raise FileNotFoundError(
            "benchmark needs adapt_prediction_payload.pt, or both legacy "
            "train_prediction_payload.pt and oracle_prediction_payload.pt"
        )
    LOGGER.warning(
        "using legacy train+oracle payloads for timing only; official experiments "
        "still require a format-v2 extraction"
    )
    train = flatten_payload(torch_load(legacy_paths[0]), "train")
    oracle = flatten_payload(torch_load(legacy_paths[1]), "oracle")
    return (
        {
            name: np.concatenate([train[name], oracle[name]], axis=0)
            for name in train
        },
        "legacy_train_plus_oracle",
    )


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    LOGGER.info(
        "gate compute benchmark mode=%s candidate=%s objective=%s",
        args.mode,
        args.candidate,
        args.objective,
    )
    started = perf_counter()
    execution = _execution_config(args)
    for name in ("thread_count", "horizon_jobs"):
        execution[name] = _positive_int(name, execution[name])
    if execution["task_type"] == "GPU":
        from catboost.utils import get_gpu_device_count

        gpu_count = int(get_gpu_device_count())
        if gpu_count < 1:
            raise RuntimeError("CatBoost did not detect a GPU")
    else:
        gpu_count = 0

    arrays, payload_source = _load_adapt_arrays(args.input_dir)
    horizon = int(arrays["y"].shape[1])
    if horizon != int(args.expected_horizon):
        raise ValueError(
            f"expected horizon {args.expected_horizon}, found {horizon} "
            f"in {args.input_dir}"
        )
    t1_arrays, t2_arrays, resplit = chronological_resplit_arrays(
        arrays,
        args.validation_fraction,
    )
    t1_fit = subsample_fit_arrays(
        t1_arrays,
        args.max_t1_fit_samples,
        seed=args.seed,
    )
    t2_fit = subsample_fit_arrays(
        t2_arrays,
        args.max_t2_valid_samples,
        seed=args.seed + 1,
    )
    refit = subsample_fit_arrays(
        arrays,
        args.max_refit_samples,
        seed=args.seed + 2,
    )
    train_features = horizon_gate_features(t1_fit, args.candidate)
    valid_features = horizon_gate_features(t2_fit, args.candidate)
    refit_features = horizon_gate_features(refit, args.candidate)
    feature_count = int(train_features[0].shape[1])
    train_targets = _horizon_targets(t1_fit, args.candidate)
    valid_targets = _horizon_targets(t2_fit, args.candidate)
    refit_targets = _horizon_targets(refit, args.candidate)

    LOGGER.info(
        "fit start mode=%s execution=%s horizon=%d features=%d "
        "t1_samples=%d t2_samples=%d refit_samples=%d",
        args.mode,
        execution,
        horizon,
        feature_count,
        train_targets.shape[0],
        valid_targets.shape[0],
        refit_targets.shape[0],
    )
    fit_started = perf_counter()
    models = fit_horizon_gate_models(
        train_features,
        train_targets,
        valid_features=valid_features,
        valid_targets=valid_targets,
        refit_features=refit_features,
        refit_targets=refit_targets,
        iterations=_positive_int("iterations", args.iterations),
        learning_rate=args.learning_rate,
        depth=_positive_int("depth", args.depth),
        seed=args.seed,
        objective=args.objective,
        early_stopping_rounds=_positive_int(
            "early_stopping_rounds",
            args.early_stopping_rounds,
        ),
        **execution,
    )
    fit_seconds = perf_counter() - fit_started
    selected = np.asarray(
        [model["selected_iterations"] for model in models],
        dtype=np.float64,
    )
    result = {
        "mode": args.mode,
        **execution,
        "gpu_device_count": gpu_count,
        "candidate": args.candidate,
        "objective": args.objective,
        "horizon": horizon,
        "feature_count": feature_count,
        "t1_samples": int(train_targets.shape[0]),
        "t2_samples": int(valid_targets.shape[0]),
        "refit_samples": int(refit_targets.shape[0]),
        "iterations": int(args.iterations),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "constant_models": int(sum("constant" in model for model in models)),
        "min_selected_iterations": float(selected.min()),
        "mean_selected_iterations": float(selected.mean()),
        "max_selected_iterations": float(selected.max()),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(perf_counter() - started),
        "horizons_per_fit_second": float(horizon / fit_seconds),
        "validation_split": resplit,
        "input_directory": str(args.input_dir),
        "payload_source": payload_source,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.mode}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = _write_summary(args.output_dir)
    LOGGER.info(
        "fit done mode=%s fit_seconds=%.3f horizons_per_second=%.4f output=%s",
        args.mode,
        fit_seconds,
        result["horizons_per_fit_second"],
        result_path,
    )
    del models
    gc.collect()
    return {"result": result_path, "summary": summary_path}


if __name__ == "__main__":
    main()
