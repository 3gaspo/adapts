"""Build selected-method tables and evolution plots for multi-backbone sweeps."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .results_table import (
    Result,
    _method_label,
    _parse_dataset_settings,
    _setting_key,
    _short_run_name,
    _split_names,
    build_table,
    discover_results,
)
from .sweep_results_table import (
    REFERENCE_METHOD,
    _run_names,
    build_average_matrix_table,
)


SELECTED_VARIANTS = (
    "aggr_y_ridge_shared",
    "bayes_context_shared",
    "full_ridge_horizon",
)


def _setting_parts(setting: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)[_-](\d+)", str(setting))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _filter_results(
    results: Sequence[Result],
    *,
    model: str | None = None,
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
) -> list[Result]:
    dataset_set = set(datasets or ())
    setting_set = set(settings or ())
    per_dataset = dataset_settings or {}
    out = []
    for result in results:
        if model is not None and result.model != model:
            continue
        if dataset_set and result.dataset not in dataset_set:
            continue
        if result.dataset in per_dataset:
            if result.setting not in per_dataset[result.dataset]:
                continue
        elif setting_set and result.setting not in setting_set:
            continue
        out.append(result)
    return out


def _average_value(
    results: Sequence[Result],
    *,
    method: str,
    metric: str,
    split: str,
    setting: str,
    datasets: Sequence[str] | None,
) -> float:
    dataset_set = set(datasets or ())
    values = [
        result.value
        for result in results
        if result.method == method
        and result.metric.casefold() == metric.casefold()
        and result.split.casefold() == split.casefold()
        and result.setting == setting
        and (not dataset_set or result.dataset in dataset_set)
        and math.isfinite(result.value)
    ]
    return sum(values) / len(values) if values else math.nan


def _relative_improvement(reference: float, value: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or reference == 0:
        return math.nan
    direction = 1.0 if lower_is_better else -1.0
    return direction * (reference - value) / abs(reference) * 100.0


def _axis_settings(
    settings: Sequence[str],
    *,
    fixed_l: int | None = None,
    fixed_h: int | None = None,
) -> list[tuple[int, str]]:
    pairs = []
    for setting in settings:
        parts = _setting_parts(setting)
        if parts is None:
            continue
        lags, horizon = parts
        if fixed_l is not None and lags == fixed_l:
            pairs.append((horizon, setting))
        elif fixed_h is not None and horizon == fixed_h:
            pairs.append((lags, setting))
    return sorted(pairs, key=lambda item: item[0])


def _plot_evolution(
    results: Sequence[Result],
    output_dir: Path,
    *,
    filename: str,
    settings: Sequence[str],
    runs: Sequence[str],
    datasets: Sequence[str] | None,
    metric: str,
    split: str,
    lower_is_better: bool,
    plot_value: str,
    fixed_l: int | None = None,
    fixed_h: int | None = None,
) -> list[Path]:
    axis_pairs = _axis_settings(settings, fixed_l=fixed_l, fixed_h=fixed_h)
    if not axis_pairs:
        return []

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(SELECTED_VARIANTS),
        1,
        figsize=(11, 3.3 * len(SELECTED_VARIANTS)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    x_values = [axis for axis, _ in axis_pairs]
    x_label = "Horizon H" if fixed_l is not None else "Lookback L"
    y_label = "Relative improvement over vanilla (%)" if plot_value == "improvement" else metric.upper()

    for ax, variant in zip(axes, SELECTED_VARIANTS):
        for run in runs:
            method = f"{run}/{variant}"
            y_values = []
            for _, setting in axis_pairs:
                value = _average_value(
                    results,
                    method=method,
                    metric=metric,
                    split=split,
                    setting=setting,
                    datasets=datasets,
                )
                if plot_value == "improvement":
                    reference = _average_value(
                        results,
                        method=REFERENCE_METHOD,
                        metric=metric,
                        split=split,
                        setting=setting,
                        datasets=datasets,
                    )
                    value = _relative_improvement(reference, value, lower_is_better)
                y_values.append(value)
            values = np.asarray(y_values, dtype=float)
            if np.isfinite(values).any():
                ax.plot(x_values, values, marker="o", linewidth=1.6, label=_short_run_name(run))
        if plot_value == "improvement":
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_ylabel(y_label)
        ax.set_title(_method_label(f"run/{variant}", True).rsplit("/", 1)[-1])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
        fig.subplots_adjust(top=0.88)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{filename}.{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(fig)
    return outputs


def build_selected_outputs(
    experiment_dir: str | Path,
    output_dir: str | Path,
    *,
    datasets: Sequence[str] | None,
    settings: Sequence[str],
    dataset_settings: Mapping[str, set[str]] | None = None,
    models: Sequence[str],
    spaces: Sequence[str],
    neighbors: Sequence[int],
    retrieval_mode: str,
    metric: str,
    split: str,
    decimals: int,
    lower_is_better: bool,
    fixed_l: int,
    fixed_h: int,
    plot_value: str,
) -> list[Path]:
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = discover_results(root)
    runs = _run_names(spaces, neighbors, retrieval_mode)
    outputs: list[Path] = []

    for model in models:
        model_results = _filter_results(
            records,
            model=model,
            datasets=datasets,
            settings=settings,
            dataset_settings=dataset_settings,
        )
        model_dir = destination / model
        model_dir.mkdir(parents=True, exist_ok=True)

        average_table = build_average_matrix_table(
            model_results,
            variants=SELECTED_VARIANTS,
            diagnostic_variants=(),
            runs=runs,
            metric=metric,
            split=split,
            datasets=datasets,
            settings=settings,
            dataset_settings=dataset_settings,
            decimals=decimals,
            lower_is_better=lower_is_better,
            caption=f"{model} selected adaptation results, averaged over selected datasets and settings",
            label=f"tab:{model}-selected-methods-average",
        )
        average_path = model_dir / "selected_methods_average.tex"
        average_path.write_text(average_table, encoding="utf-8")
        outputs.append(average_path)

        bayes_methods = [
            REFERENCE_METHOD,
            *(f"{run}/bayes_context_shared" for run in runs),
        ]
        try:
            bayes_table = build_table(
                model_results,
                metric=metric,
                split=split,
                datasets=datasets,
                settings=settings,
                dataset_settings=dataset_settings,
                methods=bayes_methods,
                reference=REFERENCE_METHOD,
                decimals=decimals,
                lower_is_better=lower_is_better,
                dataset_improvements=False,
                setting_improvements=False,
                overall_improvement=False,
                caption=f"{model} bayes-s {metric.upper()} by dataset and horizon setting",
                label=f"tab:{model}-bayes-s-full",
            )
        except ValueError as exc:
            bayes_table = f"% {exc}\n"
        bayes_path = model_dir / "bayes_s_full_results.tex"
        bayes_path.write_text(bayes_table, encoding="utf-8")
        outputs.append(bayes_path)

        plot_dir = model_dir / "plots"
        outputs.extend(
            _plot_evolution(
                model_results,
                plot_dir,
                filename=f"selected_methods_fixed_L_{fixed_l}",
                settings=settings,
                runs=runs,
                datasets=datasets,
                metric=metric,
                split=split,
                lower_is_better=lower_is_better,
                plot_value=plot_value,
                fixed_l=fixed_l,
            )
        )
        outputs.extend(
            _plot_evolution(
                model_results,
                plot_dir,
                filename=f"selected_methods_fixed_H_{fixed_h}",
                settings=settings,
                runs=runs,
                datasets=datasets,
                metric=metric,
                split=split,
                lower_is_better=lower_is_better,
                plot_value=plot_value,
                fixed_h=fixed_h,
            )
        )
    return outputs


def _parse_neighbors(value: str | Sequence[str] | None) -> list[int]:
    if value is None:
        return [1, 3, 10]
    return [int(item) for item in (_split_names(value) or [])]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--models", required=True)
    parser.add_argument("--spaces", default="raw,instance")
    parser.add_argument("--neighbors", default="1,3,10")
    parser.add_argument("--retrieval-mode", default="online")
    parser.add_argument("--metric", default="nmse")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--fixed-l", type=int, default=672)
    parser.add_argument("--fixed-h", type=int, default=24)
    parser.add_argument("--plot-value", choices=("improvement", "metric"), default="improvement")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    settings = _split_names(args.settings) or []
    outputs = build_selected_outputs(
        args.experiment_dir,
        args.output_dir,
        datasets=_split_names(args.datasets),
        settings=sorted(settings, key=_setting_key),
        dataset_settings=_parse_dataset_settings(args.dataset_settings),
        models=_split_names(args.models) or [],
        spaces=_split_names(args.spaces) or ["raw", "instance"],
        neighbors=_parse_neighbors(args.neighbors),
        retrieval_mode=args.retrieval_mode,
        metric=args.metric,
        split=args.split,
        decimals=args.decimals,
        lower_is_better=not args.higher_is_better,
        fixed_l=args.fixed_l,
        fixed_h=args.fixed_h,
        plot_value=args.plot_value,
    )
    for output in outputs:
        print(f"Selected-method artifact written to {output}")
    return outputs


if __name__ == "__main__":
    main()
