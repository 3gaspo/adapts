"""Build full and averaged adaptation LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .results_table import (
    Result,
    _latex,
    _method_label,
    _parse_dataset_settings,
    _short_run_name,
    _split_names,
    build_table,
    discover_results,
)


REFERENCE_METHOD = "vanilla"

BASELINE_HELDOUT_VARIANTS = (
    "context_forecast",
    "aggr_y",
    "y_mean",
    "aggr_y_mix_shared",
    "aggr_y_mix_horizon",
    "context_ridge_shared",
    "context_ridge_horizon",
    "aggr_y_ridge_shared",
    "aggr_y_ridge_horizon",
    "y_ridge_shared",
    "y_ridge_horizon",
    "cov_y_ridge_shared",
    "cov_y_ridge_horizon",
    "cov_horizon_ridge_shared",
    "cov_horizon_ridge_horizon",
    "residual_ridge_shared",
    "residual_ridge_horizon",
    "full_ridge_shared",
    "full_ridge_horizon",
)

BASELINE_DIAGNOSTIC_VARIANTS = tuple(
    f"{name}_eval_fit"
    for name in BASELINE_HELDOUT_VARIANTS
    if name not in {"context_forecast", "aggr_y", "y_mean"}
)

GATE_HELDOUT_VARIANTS = (
    "context_forecast",
    "aggr_y",
    "bayes_context_shared",
    "bayes_context_horizon",
    "catboost_context_classifier_shared",
    "catboost_context_classifier_horizon",
    "catboost_context_regressor_shared",
    "catboost_context_regressor_horizon",
    "bayes_aggr_y_shared",
    "bayes_aggr_y_horizon",
    "catboost_aggr_y_classifier_shared",
    "catboost_aggr_y_classifier_horizon",
    "catboost_aggr_y_regressor_shared",
    "catboost_aggr_y_regressor_horizon",
)

GATE_DIAGNOSTIC_VARIANTS = (
    "oracle_context_shared",
    "oracle_context_horizon",
    "oracle_aggr_y_shared",
    "oracle_aggr_y_horizon",
)

TS_IFA_MAIN_VARIANTS = ("TS-IFA",)
TS_IFA_BRANCH_VARIANTS = (
    "vanilla_branch",
    "context_branch",
    "transformed_branch",
    "residual_branch",
    "memory_branch",
    "ridge_rooter",
)

FULL_VARIANTS = (
    *BASELINE_HELDOUT_VARIANTS,
    "bayes_context_shared",
    "bayes_context_horizon",
    "catboost_context_classifier_shared",
    "catboost_context_classifier_horizon",
    "catboost_context_regressor_shared",
    "catboost_context_regressor_horizon",
    "bayes_aggr_y_shared",
    "bayes_aggr_y_horizon",
    "catboost_aggr_y_classifier_shared",
    "catboost_aggr_y_classifier_horizon",
    "catboost_aggr_y_regressor_shared",
    "catboost_aggr_y_regressor_horizon",
    *TS_IFA_MAIN_VARIANTS,
)
CROSSRAG_VARIANTS = ("crossrag",)
COMPARISON_VARIANTS = tuple(
    dict.fromkeys(
        (*BASELINE_HELDOUT_VARIANTS, *GATE_HELDOUT_VARIANTS, *CROSSRAG_VARIANTS)
    )
)


@dataclass(frozen=True)
class Family:
    name: str
    full_variants: tuple[str, ...]
    average_variants: tuple[str, ...]
    diagnostic_variants: tuple[str, ...]
    output_name: str
    caption: str
    label: str


FAMILIES = (
    Family(
        "full",
        FULL_VARIANTS,
        FULL_VARIANTS,
        (),
        "full_results.tex",
        "Adaptation nMSE results across retrieval settings",
        "tab:adaptation-results",
    ),
    Family(
        "baselines",
        BASELINE_HELDOUT_VARIANTS,
        BASELINE_HELDOUT_VARIANTS,
        BASELINE_DIAGNOSTIC_VARIANTS,
        "baselines_results.tex",
        "Baseline nMSE results across retrieval settings",
        "tab:baselines-results",
    ),
    Family(
        "gates",
        GATE_HELDOUT_VARIANTS,
        GATE_HELDOUT_VARIANTS,
        GATE_DIAGNOSTIC_VARIANTS,
        "gates_results.tex",
        "Gate nMSE results across retrieval settings",
        "tab:gates-results",
    ),
    Family(
        "ts_ifa",
        (*TS_IFA_MAIN_VARIANTS, *TS_IFA_BRANCH_VARIANTS),
        (*TS_IFA_MAIN_VARIANTS, *TS_IFA_BRANCH_VARIANTS),
        (),
        "ts_ifa_results.tex",
        "TS-IFA nMSE results across retrieval settings",
        "tab:ts-ifa-results",
    ),
    Family(
        "crossrag",
        CROSSRAG_VARIANTS,
        CROSSRAG_VARIANTS,
        (),
        "crossrag_results.tex",
        "Pretrained Cross-RAG nMSE on the fixed Chronos-Bolt setting",
        "tab:crossrag-results",
    ),
    Family(
        "comparison",
        COMPARISON_VARIANTS,
        COMPARISON_VARIANTS,
        (),
        "comparison_results.tex",
        "Chronos-Bolt candidate and pretrained Cross-RAG comparison",
        "tab:crossrag-comparison",
    ),
)


def _selected_families(names: Sequence[str] | None) -> tuple[Family, ...]:
    if not names:
        return tuple(
            family
            for family in FAMILIES
            if family.name not in {"crossrag", "comparison"}
        )
    by_name = {family.name: family for family in FAMILIES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown table families: {missing}")
    return tuple(by_name[name] for name in names)


def _select_variants(
    family: Family,
    available: Sequence[str],
    selected: Sequence[str] | None,
) -> tuple[str, ...]:
    if not selected:
        return tuple(available)
    selected_set = set(selected)
    if family.name in {"crossrag", "comparison"}:
        selected_set.add("crossrag")
    return tuple(variant for variant in available if variant in selected_set)


def _present_variants(
    family: Family,
    variants: Sequence[str],
    results: Sequence[Result],
) -> tuple[str, ...]:
    """Drop optional TS-IFA branches that were disabled for the selected run."""
    if family.name != "ts_ifa":
        return tuple(variants)
    present = {result.method.rsplit("/", 1)[-1] for result in results}
    return tuple(variant for variant in variants if variant in present)


def _filter_models(results: Sequence[Result], models: Sequence[str] | None) -> list[Result]:
    if not models:
        return list(results)
    selected = set(models)
    return [result for result in results if result.model in selected]


def _run_name(space: str, distance_metric: str, neighbors: int, retrieval_mode: str) -> str:
    return f"{space}_{distance_metric}_{neighbors}_{retrieval_mode}"


def _run_names(
    spaces: Sequence[str],
    distance_metrics: Sequence[str],
    neighbors: Sequence[int],
    retrieval_mode: str,
) -> list[str]:
    return [
        _run_name(space, metric, k, retrieval_mode)
        for space in spaces
        for metric in distance_metrics
        for k in neighbors
    ]


def _methods_for_variants(runs: Sequence[str], variants: Sequence[str]) -> list[str]:
    return [f"{run}/{variant}" for run in runs for variant in variants]


def _selected_runs(
    runs: Sequence[str],
    pipelines: Sequence[str] | None,
) -> list[str]:
    if not pipelines:
        return list(runs)
    requested = {
        pipeline.rsplit("/", 1)[0]
        for pipeline in pipelines
        if "/" in pipeline
    }
    return [run for run in runs if run in requested]


def _filters_match(
    result: Result,
    dataset_order: Sequence[str] | None,
    setting_filter: set[str],
    dataset_settings: Mapping[str, set[str]],
) -> bool:
    if dataset_order is not None and result.dataset not in dataset_order:
        return False
    if result.dataset in dataset_settings:
        return result.setting in dataset_settings[result.dataset]
    return not setting_filter or result.setting in setting_filter


def _average_metric(
    results: Sequence[Result],
    *,
    method: str,
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
) -> float:
    dataset_order = list(datasets) if datasets else None
    setting_filter = set(settings or ())
    values = [
        result.value
        for result in results
        if result.method == method
        and result.metric.casefold() == metric.casefold()
        and result.split.casefold() == split.casefold()
        and _filters_match(result, dataset_order, setting_filter, dataset_settings)
        and math.isfinite(result.value)
    ]
    return sum(values) / len(values) if values else math.nan


def _average_method_statistics(
    results: Sequence[Result],
    *,
    method: str,
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    lower_is_better: bool,
) -> tuple[float, float]:
    """Average the metric and per-configuration percentage improvement.

    Averaging percentages (rather than taking a ratio of two pooled metrics)
    gives every dataset/horizon configuration equal weight.
    """
    dataset_order = list(datasets) if datasets else None
    setting_filter = set(settings or ())
    references = {
        (result.dataset, result.setting, result.model): result.value
        for result in results
        if result.method == REFERENCE_METHOD
        and result.metric.casefold() == metric.casefold()
        and result.split.casefold() == split.casefold()
        and _filters_match(result, dataset_order, setting_filter, dataset_settings)
        and math.isfinite(result.value)
    }
    values: list[float] = []
    improvements: list[float] = []
    for result in results:
        if (
            result.method != method
            or result.metric.casefold() != metric.casefold()
            or result.split.casefold() != split.casefold()
            or not _filters_match(result, dataset_order, setting_filter, dataset_settings)
            or not math.isfinite(result.value)
        ):
            continue
        reference = references.get((result.dataset, result.setting, result.model))
        if reference is None:
            continue
        values.append(result.value)
        improvements.append(
            _relative_improvement(reference, result.value, lower_is_better)
        )
    return (
        sum(values) / len(values) if values else math.nan,
        sum(improvements) / len(improvements) if improvements else math.nan,
    )


def _reference_label(results: Sequence[Result]) -> str:
    models = sorted({result.model for result in results if result.model}, key=str.casefold)
    if len(models) == 1:
        display = {
            "chronos2": "Chronos-2",
            "chronos-bolt": "Chronos-Bolt",
            "tabpfnts": "TabPFN-TS",
        }.get(models[0], models[0])
        return f"Vanilla {display}"
    return "Vanilla backbone"


def _caption_with_reference(
    caption: str,
    metric: str,
    reference: float,
    decimals: int,
    reference_label: str,
) -> str:
    separator = " " if caption.rstrip().endswith((".", "?", "!")) else ". "
    if math.isfinite(reference):
        reference_text = f"{reference_label} {metric.upper()}: {reference:.{decimals}f}."
    else:
        reference_text = f"{reference_label} {metric.upper()}: unavailable."
    return caption + separator + reference_text


def _relative_improvement(reference: float, value: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or reference == 0:
        return math.nan
    direction = 1.0 if lower_is_better else -1.0
    return direction * (reference - value) / abs(reference) * 100.0


def _matrix_row_label(variant: str) -> str:
    return _method_label(f"run/{variant}", True).rsplit("/", 1)[-1]


def _colored_improvement(text: str, improvement: float, decimals: int) -> str:
    rounded = round(improvement, decimals)
    if rounded > 0.0:
        return rf"\textcolor{{green!50!black}}{{{text}}}"
    if rounded < 0.0:
        return rf"\textcolor{{red!70!black}}{{{text}}}"
    return text


def _matrix_cell(value: float, improvement: float, decimals: int, bold: bool) -> str:
    if not math.isfinite(value) or not math.isfinite(improvement):
        return "--"
    top = f"{improvement:.{decimals}f}" + r"\%"
    if bold:
        top = rf"\textbf{{{top}}}"
    top = _colored_improvement(top, improvement, decimals)
    bottom = rf"{{\scriptsize {value:.{decimals}f}}}"
    return rf"\begin{{tabular}}{{@{{}}c@{{}}}}{top}\\{bottom}\end{{tabular}}"


def build_average_matrix_table(
    results: Sequence[Result],
    *,
    variants: Sequence[str],
    diagnostic_variants: Sequence[str],
    runs: Sequence[str],
    metric: str = "nmse",
    split: str = "eval",
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    decimals: int = 2,
    lower_is_better: bool = True,
    caption: str | None = None,
    label: str = "tab:sweep-matrix",
    allowed_methods: set[str] | None = None,
) -> str:
    dataset_settings = dataset_settings or {}
    reference = _average_metric(
        results,
        method=REFERENCE_METHOD,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
    )
    row_variants = [*variants, *diagnostic_variants]
    values: dict[tuple[str, str], float] = {}
    improvements: dict[tuple[str, str], float] = {}
    for variant in row_variants:
        for run in runs:
            method = f"{run}/{variant}"
            if allowed_methods is not None and method not in allowed_methods:
                values[(variant, run)] = math.nan
                improvements[(variant, run)] = math.nan
                continue
            value, improvement = _average_method_statistics(
                results,
                method=method,
                metric=metric,
                split=split,
                datasets=datasets,
                settings=settings,
                dataset_settings=dataset_settings,
                lower_is_better=lower_is_better,
            )
            values[(variant, run)] = value
            improvements[(variant, run)] = improvement
    diagnostic_set = set(diagnostic_variants)
    finite = [
        improvement
        for (variant, _), improvement in improvements.items()
        if variant not in diagnostic_set
        and not variant.startswith("oracle_")
        and math.isfinite(improvement)
    ]
    best = max(finite) if finite else None
    caption_text = caption or f"Average {metric.upper()} by retrieval setting."
    display_runs = tuple(runs)
    headers = [_latex(_short_run_name(run)) for run in runs]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex(_caption_with_reference(caption_text, metric, reference, decimals, _reference_label(results)))}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{'l' + 'c' * len(display_runs)}}}",
        r"\toprule",
        "Model & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    inserted_diagnostic_rule = False
    for variant in row_variants:
        if variant in diagnostic_set and not inserted_diagnostic_rule:
            lines.append(r"\midrule")
            inserted_diagnostic_rule = True
        cells = []
        for run in display_runs:
            improvement = improvements[(variant, run)]
            is_best = (
                variant not in diagnostic_set
                and not variant.startswith("oracle_")
                and best is not None
                and math.isclose(improvement, best, rel_tol=1e-12, abs_tol=1e-15)
            )
            cells.append(_matrix_cell(values[(variant, run)], improvement, decimals, is_best))
        lines.append(" & ".join([_latex(_matrix_row_label(variant)), *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", rf"\label{{{_latex(label)}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_average_positive_window_table(
    results: Sequence[Result],
    *,
    variants: Sequence[str],
    diagnostic_variants: Sequence[str],
    runs: Sequence[str],
    split: str = "eval",
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    decimals: int = 2,
    allowed_methods: set[str] | None = None,
) -> str:
    """Build absolute win-rate averages from persisted per-run aggregates."""
    dataset_settings = dataset_settings or {}
    row_variants = [*variants, *diagnostic_variants]
    values: dict[tuple[str, str], float] = {}
    for variant in row_variants:
        for run in runs:
            method = f"{run}/{variant}"
            values[(variant, run)] = (
                math.nan
                if allowed_methods is not None and method not in allowed_methods
                else _average_metric(
                    results,
                    method=method,
                    metric="positive_window_pct",
                    split=split,
                    datasets=datasets,
                    settings=settings,
                    dataset_settings=dataset_settings,
                )
            )
    diagnostic_set = set(diagnostic_variants)
    finite = [
        value
        for (variant, _), value in values.items()
        if variant not in diagnostic_set
        and not variant.startswith("oracle_")
        and math.isfinite(value)
    ]
    best = max(finite) if finite else None
    headers = [_latex(_short_run_name(run)) for run in runs]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean percentage of evaluation windows with lower horizon-averaged MSE than the vanilla forecast, averaged equally over the selected dataset and horizon configurations.}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{'l' + 'c' * len(runs)}}}",
        r"\toprule",
        "Model & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    inserted_diagnostic_rule = False
    for variant in row_variants:
        if variant in diagnostic_set and not inserted_diagnostic_rule:
            lines.append(r"\midrule")
            inserted_diagnostic_rule = True
        cells: list[str] = []
        for run in runs:
            value = values[(variant, run)]
            if not math.isfinite(value):
                cells.append("--")
                continue
            cell = f"{value:.{decimals}f}" + r"\%"
            if (
                variant not in diagnostic_set
                and not variant.startswith("oracle_")
                and best is not None
                and math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-15)
            ):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join([_latex(_matrix_row_label(variant)), *cells]) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\label{tab:positive-window-percentage-average}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_full_family_table(
    results: Sequence[Result],
    output_dir: Path,
    family: Family,
    *,
    runs: Sequence[str],
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    decimals: int,
    lower_is_better: bool,
    allowed_methods: set[str] | None,
) -> Path:
    candidate_methods = _methods_for_variants(runs, family.full_variants)
    if allowed_methods is not None:
        candidate_methods = [
            method for method in candidate_methods if method in allowed_methods
        ]
    methods = [REFERENCE_METHOD, *candidate_methods]
    table = build_table(
        results,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
        methods=methods,
        reference=REFERENCE_METHOD,
        decimals=decimals,
        lower_is_better=lower_is_better,
        dataset_improvements=False,
        setting_improvements=False,
        overall_improvement=False,
        caption=family.caption + " by dataset and horizon setting",
        label=family.label,
    )
    output = output_dir / family.output_name
    output.write_text(table, encoding="utf-8")
    return output


def _write_average_family_table(
    results: Sequence[Result],
    output_dir: Path,
    family: Family,
    *,
    runs: Sequence[str],
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    decimals: int,
    lower_is_better: bool,
    allowed_methods: set[str] | None,
) -> Path:
    table = build_average_matrix_table(
        results,
        variants=family.average_variants,
        diagnostic_variants=family.diagnostic_variants,
        runs=runs,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
        decimals=decimals,
        lower_is_better=lower_is_better,
        caption=family.caption + ", averaged over selected datasets and horizon settings",
        label=f"{family.label}-average",
        allowed_methods=allowed_methods,
    )
    output = output_dir / family.output_name
    output.write_text(table, encoding="utf-8")
    return output


def _write_average_positive_window_table(
    results: Sequence[Result],
    output_dir: Path,
    *,
    families: Sequence[Family],
    runs: Sequence[str],
    selected_variants: Sequence[str] | None,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    decimals: int,
    allowed_methods: set[str] | None,
) -> Path | None:
    available = {
        result.method.rsplit("/", 1)[-1]
        for result in results
        if result.metric == "positive_window_pct"
    }
    variants = tuple(
        dict.fromkeys(
            variant
            for family in families
            for variant in _select_variants(
                family,
                family.average_variants,
                selected_variants,
            )
            if variant in available
        )
    )
    diagnostics = tuple(
        dict.fromkeys(
            variant
            for family in families
            for variant in _select_variants(
                family,
                family.diagnostic_variants,
                selected_variants,
            )
            if variant in available
        )
    )
    diagnostics_set = set(diagnostics)
    variants = tuple(variant for variant in variants if variant not in diagnostics_set)
    if not variants and not diagnostics:
        return None
    table = build_average_positive_window_table(
        results,
        variants=variants,
        diagnostic_variants=diagnostics,
        runs=runs,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
        decimals=decimals,
        allowed_methods=allowed_methods,
    )
    output = output_dir / "positive_windows_results.tex"
    output.write_text(table, encoding="utf-8")
    return output


def _write_pipeline_ranking(
    results: Sequence[Result],
    output_dir: Path,
    *,
    families: Sequence[Family],
    runs: Sequence[str],
    selected_variants: Sequence[str] | None,
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    lower_is_better: bool,
    allowed_methods: set[str] | None,
) -> None:
    rows: list[dict[str, object]] = []
    for family in families:
        variants = _select_variants(
            family,
            family.average_variants,
            selected_variants,
        )
        for run in runs:
            for variant in variants:
                method = f"{run}/{variant}"
                if allowed_methods is not None and method not in allowed_methods:
                    continue
                if variant.startswith("oracle_") or variant.endswith("_eval_fit"):
                    continue
                value, improvement = _average_method_statistics(
                    results,
                    method=method,
                    metric=metric,
                    split=split,
                    datasets=datasets,
                    settings=settings,
                    dataset_settings=dataset_settings,
                    lower_is_better=lower_is_better,
                )
                if not math.isfinite(improvement):
                    continue
                rows.append(
                    {
                        "winner_name": f"{family.name}/{run}/{variant}",
                        "family": family.name,
                        "retrieval": run,
                        "method": variant,
                        f"average_{metric}": value,
                        "average_improvement_pct": improvement,
                    }
                )
    rows.sort(
        key=lambda row: float(row["average_improvement_pct"]),
        reverse=True,
    )
    json_path = output_dir / "pipeline_ranking.json"
    csv_path = output_dir / "pipeline_ranking.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "winner_name",
            "family",
            "retrieval",
            "method",
            f"average_{metric}",
            "average_improvement_pct",
        ]
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_full_results_tables(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    models: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    spaces: Sequence[str] = ("raw", "instance"),
    distance_metrics: Sequence[str] = ("euclidean",),
    neighbors: Sequence[int] = (1, 3, 10),
    retrieval_mode: str = "online",
    metric: str = "nmse",
    split: str = "eval",
    decimals: int = 2,
    lower_is_better: bool = True,
    variants: Sequence[str] | None = None,
    pipelines: Sequence[str] | None = None,
) -> list[Path]:
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else root / "full_tables"
    destination.mkdir(parents=True, exist_ok=True)
    records = _filter_models(discover_results(root), models)
    runs = _selected_runs(
        _run_names(spaces, distance_metrics, neighbors, retrieval_mode),
        pipelines,
    )
    allowed_methods = set(pipelines) if pipelines else None
    return [
        _write_full_family_table(
            records,
            destination,
            (
                Family(
                    family.name,
                    _present_variants(
                        family,
                        _select_variants(family, family.full_variants, variants),
                        records,
                    ),
                    family.average_variants,
                    family.diagnostic_variants,
                    family.output_name,
                    family.caption,
                    family.label,
                )
            ),
            runs=runs,
            metric=metric,
            split=split,
            datasets=datasets,
            settings=settings,
            dataset_settings=dataset_settings or {},
            decimals=decimals,
            lower_is_better=lower_is_better,
            allowed_methods=allowed_methods,
        )
        for family in _selected_families(families)
    ]


def generate_average_results_tables(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    models: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    spaces: Sequence[str] = ("raw", "instance"),
    distance_metrics: Sequence[str] = ("euclidean",),
    neighbors: Sequence[int] = (1, 3, 10),
    retrieval_mode: str = "online",
    metric: str = "nmse",
    split: str = "eval",
    decimals: int = 2,
    lower_is_better: bool = True,
    variants: Sequence[str] | None = None,
    pipelines: Sequence[str] | None = None,
) -> list[Path]:
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else root / "average_tables"
    destination.mkdir(parents=True, exist_ok=True)
    records = _filter_models(discover_results(root), models)
    runs = _selected_runs(
        _run_names(spaces, distance_metrics, neighbors, retrieval_mode),
        pipelines,
    )
    allowed_methods = set(pipelines) if pipelines else None
    selected_families = _selected_families(families)
    outputs = [
        _write_average_family_table(
            records,
            destination,
            (
                Family(
                    family.name,
                    family.full_variants,
                    _present_variants(
                        family,
                        _select_variants(family, family.average_variants, variants),
                        records,
                    ),
                    _present_variants(
                        family,
                        _select_variants(family, family.diagnostic_variants, variants),
                        records,
                    ),
                    family.output_name,
                    family.caption,
                    family.label,
                )
            ),
            runs=runs,
            metric=metric,
            split=split,
            datasets=datasets,
            settings=settings,
            dataset_settings=dataset_settings or {},
            decimals=decimals,
            lower_is_better=lower_is_better,
            allowed_methods=allowed_methods,
        )
        for family in selected_families
    ]
    _write_pipeline_ranking(
        records,
        destination,
        families=selected_families,
        runs=runs,
        selected_variants=variants,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings or {},
        lower_is_better=lower_is_better,
        allowed_methods=allowed_methods,
    )
    positive_window_output = _write_average_positive_window_table(
        records,
        destination,
        families=selected_families,
        runs=runs,
        selected_variants=variants,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings or {},
        decimals=decimals,
        allowed_methods=allowed_methods,
    )
    if positive_window_output is not None:
        outputs.append(positive_window_output)
    return outputs


def _parse_neighbors(value: str | Sequence[str] | None) -> list[int]:
    if value is None:
        return [1, 3, 10]
    return [int(item) for item in _split_names(value)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table-kind", choices=("full", "average"), default="average")
    parser.add_argument("--metric", default="nmse")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--models", default=None)
    parser.add_argument("--families", default=None, help="Comma/semicolon-separated table families")
    parser.add_argument("--spaces", default="raw,instance")
    parser.add_argument("--distance-metrics", default="euclidean")
    parser.add_argument("--neighbors", default="1,3")
    parser.add_argument("--variants", default=None, help="Only include these method variants")
    parser.add_argument(
        "--pipelines",
        default=None,
        help="Only include exact retrieval_run/method pipeline names",
    )
    parser.add_argument("--retrieval-mode", default="online")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    generator = generate_full_results_tables if args.table_kind == "full" else generate_average_results_tables
    outputs = generator(
        args.experiment_dir,
        args.output_dir,
        metric=args.metric,
        split=args.split,
        datasets=_split_names(args.datasets),
        settings=_split_names(args.settings),
        dataset_settings=_parse_dataset_settings(args.dataset_settings),
        models=_split_names(args.models),
        families=_split_names(args.families),
        spaces=_split_names(args.spaces) or ("raw", "instance"),
        distance_metrics=_split_names(args.distance_metrics) or ("euclidean",),
        neighbors=_parse_neighbors(args.neighbors),
        retrieval_mode=args.retrieval_mode,
        decimals=args.decimals,
        lower_is_better=not args.higher_is_better,
        variants=_split_names(args.variants),
        pipelines=_split_names(args.pipelines),
    )
    for output in outputs:
        print(f"LaTeX table written to {output}")
    return outputs


if __name__ == "__main__":
    main()
