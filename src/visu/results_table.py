"""Build publication-ready LaTeX tables from current adaptation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment_runs import (
    SelectedRun,
    load_manifest,
    manifest_is_selectable,
    select_identity_runs,
)


@dataclass(frozen=True)
class Result:
    dataset: str
    setting: str
    method: str
    split: str
    metric: str
    value: float
    path: Path
    model: str = ""
    run: str = ""


_RUN_NAME_RE = re.compile(
    r"(in|raw|instance|minmax|encoder|fourier|chronos|patchtst|model|representation)"
    r"_(euclidean|cosine|pearson)_(\d+)_(online|fixed)"
    r"(?:_(all|same_user|other_users))?"
)
_EVALUATION_RESULT_FORMAT = "adaptation_evaluation_result"
_TS_IFA_RESULT_FORMAT = "adaptation_ts_ifa_result"
_TS_IFA_ARCHITECTURE = "configurable_delta_branches_routing_v4"
_TS_IFA_VARIANTS = {"joint_ridge", "joint_neural", "meta_ridge", "meta_neural"}
_REQUIRED_METRIC_FIELDS = {
    "split",
    "baseline",
    "mse",
    "mae",
    "nmse",
    "positive_window_pct",
}
_PIPELINE_INDEPENDENT_METHODS = {"cov_forecast", "avgy", "y_mean"}
_RUN_SELECTION = {
    "pipeline_config": {},
    "config_policy": "distinct",
    "repeat_policy": "selected",
    "purposes": [],
}
_LAST_SELECTED_RUNS: list[SelectedRun] = []


def configure_run_selection(
    *, pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Sequence[str] = (),
) -> None:
    _RUN_SELECTION.update(
        pipeline_config=dict(pipeline_config or {}),
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        purposes=list(purposes),
    )


def selected_manifest_runs(root: str | Path) -> list[SelectedRun]:
    base = Path(root).expanduser().resolve()
    active_launch = os.environ.get("EXPERIMENT_LAUNCH_ID")
    identity_roots = sorted(
        {path.parent.parent for path in base.rglob("manifest.json") if path.parent.name.startswith("run_") and "archive" not in path.relative_to(base).parts}
    )
    selected: list[SelectedRun] = []
    for identity_root in identity_roots:
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        if any(manifest_is_selectable(manifest, allow_ready_launch_id=active_launch) for manifest in manifests):
            selected.extend(
                select_identity_runs(
                    identity_root,
                    requested_pipeline=_RUN_SELECTION["pipeline_config"],
                    config_policy=_RUN_SELECTION["config_policy"],
                    repeat_policy=_RUN_SELECTION["repeat_policy"],
                    purposes=_RUN_SELECTION["purposes"],
                    allow_ready_launch_id=active_launch,
                )
            )
    return selected


def _load_json(path: Path, expected_type: type, description: str) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    if not isinstance(payload, expected_type):
        raise ValueError(
            f"invalid {description} at {path}: expected {expected_type.__name__}"
        )
    return payload


def _require_file(directory: Path, filename: Any, description: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f"invalid {description} filename in {directory}: {filename!r}")
    path = directory / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {description}: {path}")
    return path


def _load_evaluation_metrics(path: Path, family: str) -> list[Mapping[str, Any]]:
    manifest_path = path.parent / "result_manifest.json"
    manifest = _load_json(manifest_path, dict, "evaluation result manifest")
    if manifest.get("format") != _EVALUATION_RESULT_FORMAT:
        raise ValueError(f"obsolete evaluation result format at {manifest_path}")
    if manifest.get("family") != family:
        raise ValueError(
            f"evaluation family mismatch at {manifest_path}: expected {family!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or files.get("metrics_json") != path.name:
        raise ValueError(f"metrics file is not indexed by {manifest_path}")
    _require_file(path.parent, files.get("predictions"), "prediction manifest")
    fields = manifest.get("metric_fields")
    if not isinstance(fields, list) or not _REQUIRED_METRIC_FIELDS <= set(fields):
        raise ValueError(f"incomplete metric contract at {manifest_path}")
    methods = manifest.get("methods")
    if not isinstance(methods, list) or any(not isinstance(item, str) for item in methods):
        raise ValueError(f"invalid method list at {manifest_path}")
    rows = _load_json(path, list, f"{family} metrics")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not _REQUIRED_METRIC_FIELDS <= set(row):
            raise ValueError(f"invalid metric row {index} at {path}")
        method = row["baseline"]
        if method != "vanilla" and method not in methods:
            raise ValueError(f"unindexed method {method!r} in {path}")
        for metric in ("mse", "mae", "nmse", "positive_window_pct"):
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} in row {index} at {path}")
    row_methods = {str(row["baseline"]) for row in rows if row["baseline"] != "vanilla"}
    if row_methods != set(methods):
        raise ValueError(f"metric methods do not match {manifest_path}")
    return rows


def _load_ts_ifa_metrics(path: Path, method: str | None = None) -> Mapping[str, Any]:
    manifest_path = path.parent / "result_manifest.json"
    manifest = _load_json(manifest_path, dict, "TS-IFA result manifest")
    variant = manifest.get("variant")
    if variant not in _TS_IFA_VARIANTS:
        raise ValueError(f"invalid TS-IFA variant at {manifest_path}: {variant!r}")
    if manifest.get("format") != _TS_IFA_RESULT_FORMAT:
        raise ValueError(f"obsolete TS-IFA result format at {manifest_path}")
    if method is not None and manifest.get("method") != method:
        raise ValueError(f"TS-IFA method mismatch at {manifest_path}")
    if manifest.get("architecture") != _TS_IFA_ARCHITECTURE:
        raise ValueError(f"obsolete TS-IFA architecture at {manifest_path}")
    if not isinstance(manifest.get("run_signature"), str) or not manifest["run_signature"]:
        raise ValueError(f"missing TS-IFA run signature at {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or files.get("metrics") != path.name:
        raise ValueError(f"metrics file is not indexed by {manifest_path}")
    _require_file(path.parent, files.get("predictions"), "prediction manifest")
    payload = _load_json(path, dict, "TS-IFA metrics")
    candidates = manifest.get("candidate_names")
    if not isinstance(candidates, list) or not candidates or candidates[0] != "vanilla":
        raise ValueError(f"invalid TS-IFA candidates at {manifest_path}")
    required = {
        f"{branch}_{metric}"
        for branch in ("adapted", *(f"{name}_branch" for name in candidates))
        for metric in ("mse", "mae", "nmse")
    }
    if not required <= set(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"incomplete TS-IFA metrics at {path}: missing {missing}")
    for key in required:
        if not math.isfinite(float(payload[key])):
            raise ValueError(f"non-finite {key} at {path}")
    return payload


def _load_crossrag_metrics(path: Path) -> list[Mapping[str, Any]]:
    manifest_path = path.parent / "result_manifest.json"
    manifest = _load_json(manifest_path, dict, "Cross-RAG result manifest")
    if manifest.get("format") != "adaptation_crossrag_result":
        raise ValueError(f"obsolete Cross-RAG result format at {manifest_path}")
    if manifest.get("protocol") != {"lags": 512, "horizon": 64, "neighbors": 15}:
        raise ValueError(f"unexpected Cross-RAG protocol at {manifest_path}")
    fields = manifest.get("metric_fields")
    if not isinstance(fields, list) or not _REQUIRED_METRIC_FIELDS <= set(fields):
        raise ValueError(f"incomplete Cross-RAG metric contract at {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or files.get("metrics") != path.name:
        raise ValueError(f"metrics file is not indexed by {manifest_path}")
    _require_file(path.parent, files.get("timing"), "Cross-RAG timing")
    rows = _load_json(path, list, "Cross-RAG metrics")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not _REQUIRED_METRIC_FIELDS <= set(row):
            raise ValueError(f"invalid Cross-RAG metric row {index} at {path}")
    return rows


def _setting_ancestor(path: Path, root: Path) -> tuple[str, str] | None:
    relative = path.relative_to(root)
    parts = relative.parts
    for index in range(len(parts) - 1, 0, -1):
        if re.fullmatch(r"\d+[_-]\d+", parts[index]):
            return parts[index - 1], parts[index]
    return None


def _after_setting_parts(path: Path, root: Path, setting: str) -> tuple[str, ...]:
    parts = path.relative_to(root).parts
    index = parts.index(setting)
    return parts[index + 1 : -1]


def _looks_like_run_name(value: str) -> bool:
    return _RUN_NAME_RE.fullmatch(value) is not None


def _path_model(path: Path, root: Path, setting: str) -> str:
    after_setting = _after_setting_parts(path, root, setting)
    if not after_setting:
        return ""
    first = after_setting[0]
    if _looks_like_run_name(first) or first == "vanilla":
        return ""
    return first


def _relative_run(path: Path, root: Path, setting: str) -> str:
    after_setting = _after_setting_parts(path, root, setting)
    if (
        len(after_setting) >= 2
        and not _looks_like_run_name(after_setting[0])
        and after_setting[0] != "vanilla"
    ):
        return after_setting[1]
    return after_setting[0] if after_setting else path.parent.name


def discover_results(experiment_dir: str | Path) -> list[Result]:
    """Discover metrics only from selected, completed current manifests."""
    root = Path(experiment_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    results: list[Result] = []
    selected = selected_manifest_runs(root)
    _LAST_SELECTED_RUNS[:] = selected
    for choice in selected:
        identity = choice.manifest["identity"]
        config = identity["model_config"]
        dataset = str(identity["dataset"])
        setting = f"{identity['lookback']}_{identity['horizon']}"
        model = str(identity["backbone"])
        run = "_".join(str(config[name]) for name in ("space", "metric", "k", "mode") if name in config)
        retrieval_scope = str(
            choice.manifest.get("config", {}).get("pipeline", {}).get(
                "retrieval.scope", ""
            )
        )
        if retrieval_scope:
            run = f"{run}_{retrieval_scope}"
        if (choice.run_dir / "baseline_metrics.json").is_file():
            path = choice.run_dir / "baseline_metrics.json"
            family = "baselines"
            payload = _load_evaluation_metrics(path, family)
        elif (choice.run_dir / "gate_metrics.json").is_file():
            path = choice.run_dir / "gate_metrics.json"
            family = "gates"
            payload = _load_evaluation_metrics(path, family)
        elif (choice.run_dir / "crossrag_metrics.json").is_file():
            path = choice.run_dir / "crossrag_metrics.json"
            family = "crossrag"
            payload = _load_crossrag_metrics(path)
        else:
            path = choice.run_dir / "eval_metrics.json"
            if not path.is_file():
                continue
            method_name = "_".join(
                str(config[name])
                for name in ("variant", "routing_scope", "routing_constraint", "branch_set")
            )
            payload = _load_ts_ifa_metrics(path, method_name)
            for key, value in payload.items():
                match = re.fullmatch(r"(.+)_(mse|mae|nmse)", str(key).lower())
                if match is None:
                    continue
                variant, metric = match.groups()
                method = f"{run}/{choice.label}" if variant == "adapted" else f"{run}/{choice.label}_{variant}"
                results.append(
                    Result(
                        dataset,
                        setting,
                        method,
                        "eval",
                        metric,
                        float(value),
                        path,
                        model,
                        run,
                    )
                )
            continue
        for row in payload:
            formula = str(row["baseline"])
            if formula == "vanilla":
                method = formula
            elif formula in _PIPELINE_INDEPENDENT_METHODS:
                method = f"{run}/{formula}"
            elif retrieval_scope:
                method = f"{run}/{formula}"
            else:
                method = f"{run}/{choice.label}"
            for metric in ("mse", "mae", "nmse", "positive_window_pct"):
                if metric in row:
                    results.append(
                        Result(
                            dataset,
                            setting,
                            method,
                            str(row.get("split", "eval")),
                            metric,
                            float(row[metric]),
                            path,
                            model,
                            run,
                        )
                    )
    return results


def _split_names(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    values = re.split(r"[,;]", value) if isinstance(value, str) else [str(item) for item in value]
    return [item.strip() for item in values if item.strip()]


def _setting_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"[_-]", value))


def _parse_dataset_settings(values: Iterable[str] | None) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"dataset setting must be DATASET=L_H[,L_H], got {item!r}")
        dataset, settings = item.split("=", 1)
        selected.setdefault(dataset.strip(), set()).update(_split_names(settings) or ())
    return selected


def _parse_scale_exponents(values: Iterable[str] | None) -> dict[tuple[str, str], int]:
    exponents: dict[tuple[str, str], int] = {}
    for item in values or ():
        if "=" not in item or "/" not in item.split("=", 1)[0]:
            raise ValueError(f"scale must be DATASET/L_H=EXPONENT, got {item!r}")
        row, exponent = item.split("=", 1)
        dataset, setting = row.split("/", 1)
        exponents[(dataset.strip(), setting.strip())] = int(exponent)
    return exponents


def _latex(text: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(text))


def _latex_setting(setting: str) -> str:
    return "--".join(_latex(part) for part in re.split(r"[_-]", setting))


_METHOD_LABELS = {
    "vanilla": "vanilla",
    "cov_forecast": "cov",
    "avgy": "avgy",
    "y_mean": "mean-Y",
    "cov_ridge_shared": "cov-ridge-s",
    "cov_ridge_horizon": "cov-ridge-h",
    "avgy_ridge_shared": "avgy-ridge-s",
    "avgy_ridge_horizon": "avgy-ridge-h",
    "y_ridge_shared": "Y-ridge-s",
    "y_ridge_horizon": "Y-ridge-h",
    "cov_y_ridge_shared": "cov-Y-ridge-s",
    "cov_y_ridge_horizon": "cov-Y-ridge-h",
    "cov_avgy_ridge_shared": "cov-avgy-ridge-s",
    "cov_avgy_ridge_horizon": "cov-avgy-ridge-h",
    "residual_ridge_shared": "residual-ridge-s",
    "residual_ridge_horizon": "residual-ridge-h",
    "full_ridge_shared": "full-ridge-s",
    "full_ridge_horizon": "full-ridge-h",
    "cov_ridge_shared_eval_fit": "cov-ridge-s-fit-T3",
    "cov_ridge_horizon_eval_fit": "cov-ridge-h-fit-T3",
    "avgy_ridge_shared_eval_fit": "avgy-ridge-s-fit-T3",
    "avgy_ridge_horizon_eval_fit": "avgy-ridge-h-fit-T3",
    "y_ridge_shared_eval_fit": "Y-ridge-s-fit-T3",
    "y_ridge_horizon_eval_fit": "Y-ridge-h-fit-T3",
    "cov_y_ridge_shared_eval_fit": "cov-Y-ridge-s-fit-T3",
    "cov_y_ridge_horizon_eval_fit": "cov-Y-ridge-h-fit-T3",
    "cov_avgy_ridge_shared_eval_fit": "cov-avgy-ridge-s-fit-T3",
    "cov_avgy_ridge_horizon_eval_fit": "cov-avgy-ridge-h-fit-T3",
    "residual_ridge_shared_eval_fit": "residual-ridge-s-fit-T3",
    "residual_ridge_horizon_eval_fit": "residual-ridge-h-fit-T3",
    "full_ridge_shared_eval_fit": "full-ridge-s-fit-T3",
    "full_ridge_horizon_eval_fit": "full-ridge-h-fit-T3",
    "bayes_cov_shared": "bayes-cov-s",
    "bayes_cov_horizon": "bayes-cov-h",
    "catboost_cov_classifier_shared": "cb-cov-cls-s",
    "catboost_cov_classifier_horizon": "cb-cov-cls-h",
    "catboost_cov_regressor_shared": "cb-cov-reg-s",
    "catboost_cov_regressor_horizon": "cb-cov-reg-h",
    "catboost_cov_classifier_shared_soft": "cb-cov-cls-s-soft",
    "catboost_cov_classifier_horizon_soft": "cb-cov-cls-h-soft",
    "catboost_cov_regressor_shared_soft": "cb-cov-reg-s-soft",
    "catboost_cov_regressor_horizon_soft": "cb-cov-reg-h-soft",
    "bayes_avgy_shared": "bayes-avgy-s",
    "bayes_avgy_horizon": "bayes-avgy-h",
    "catboost_avgy_classifier_shared": "cb-avgy-cls-s",
    "catboost_avgy_classifier_horizon": "cb-avgy-cls-h",
    "catboost_avgy_regressor_shared": "cb-avgy-reg-s",
    "catboost_avgy_regressor_horizon": "cb-avgy-reg-h",
    "catboost_avgy_classifier_shared_soft": "cb-avgy-cls-s-soft",
    "catboost_avgy_classifier_horizon_soft": "cb-avgy-cls-h-soft",
    "catboost_avgy_regressor_shared_soft": "cb-avgy-reg-s-soft",
    "catboost_avgy_regressor_horizon_soft": "cb-avgy-reg-h-soft",
    "oracle_cov_shared": "oracle-cov-s",
    "oracle_cov_horizon": "oracle-cov-h",
    "oracle_avgy_shared": "oracle-avgy-s",
    "oracle_avgy_horizon": "oracle-avgy-h",
    "joint_ridge": "TS-IFA joint ridge",
    "joint_neural": "TS-IFA joint neural",
    "meta_ridge": "TS-IFA meta ridge",
    "meta_neural": "TS-IFA meta neural",
    "crossrag": "Cross-RAG",
}

_BASELINE_DESIGN_LABELS = {
    "cov": "cov",
    "avgy": "avgy",
    "y": "Y",
    "cov_y": "cov-Y",
    "cov_avgy": "cov-avgy",
    "residual": "residual",
    "full": "full",
}
for _variant, _variant_label in (
    ("joint_ridge", "JR"),
    ("joint_neural", "JN"),
    ("meta_ridge", "MR"),
    ("meta_neural", "MN"),
):
    for _branch, _branch_label in (
        ("vanilla", "V"),
        ("cov", "C"),
        ("residual", "R"),
        ("memory", "M"),
    ):
        _METHOD_LABELS[f"{_variant}_{_branch}_branch"] = (
            f"TS-IFA {_variant_label}-{_branch_label}"
        )
for _variant, _variant_label in (
    ("joint_ridge", "JR"),
    ("joint_neural", "JN"),
    ("meta_ridge", "MR"),
    ("meta_neural", "MN"),
):
    for _scope, _scope_label in (("shared", "S"), ("horizon", "H")):
        for _constraint, _constraint_label in (
            ("unconstrained", "U"),
            ("softmax", "SM"),
        ):
            for _branches in ("cov", "residual", "memory", "full"):
                _method = (
                    f"{_variant}_{_scope}_{_constraint}_{_branches}"
                )
                _METHOD_LABELS[_method] = (
                    f"TS-IFA {_variant_label}-{_scope_label}-"
                    f"{_constraint_label}-{_branches}"
                )
                for _branch, _branch_label in (
                    ("vanilla", "V"),
                    ("cov", "C"),
                    ("residual", "R"),
                    ("memory", "M"),
                ):
                    _METHOD_LABELS[f"{_method}_{_branch}_branch"] = (
                        f"TS-IFA {_variant_label}-{_scope_label}-"
                        f"{_constraint_label}-{_branches}-{_branch_label}"
                    )
for _design, _label in _BASELINE_DESIGN_LABELS.items():
    for _mode, _mode_label in (("shared", "s"), ("horizon", "h")):
        for _family, _family_label in (
            ("convex", "convex"),
            ("delta_ridge", "delta-ridge"),
        ):
            _method = f"{_design}_{_family}_{_mode}"
            _METHOD_LABELS[_method] = f"{_label}-{_family_label}-{_mode_label}"
            _METHOD_LABELS[f"{_method}_eval_fit"] = (
                f"{_label}-{_family_label}-{_mode_label}-fit-T3"
            )


def _short_run_name(run: str) -> str:
    match = _RUN_NAME_RE.fullmatch(run)
    if match is not None:
        space, metric, neighbors, mode, scope = match.groups()
        space = {
            "in": "IN",
            "instance": "IN",
            "minmax": "MM",
            "fourier": "FFT",
            "encoder": "ENC",
        }.get(space, space)
        metric = "L2" if metric == "euclidean" else metric
        parts = [space, metric, neighbors]
        if mode == "fixed":
            parts.append("fixed")
        if scope:
            parts.append({"same_user": "same", "other_users": "other"}.get(scope, scope))
        return "_".join(parts)
    short = run.replace("_euclidean_", "_L2_")
    return short.removesuffix("_online")


def _method_label(method: str, short_names: bool) -> str:
    if not short_names or "/" not in method:
        return method
    run, variant = method.rsplit("/", 1)
    return f"{_short_run_name(run)}/{_METHOD_LABELS.get(variant, variant)}"


def _method_selected(method: str, selectors: set[str]) -> bool:
    return method in selectors or method.rsplit("/", 1)[-1] in selectors


def _auto_exponent(values: Sequence[float], lower_is_better: bool) -> int:
    del lower_is_better
    finite = [abs(value) for value in values if math.isfinite(value) and value != 0]
    if not finite:
        return 0
    finite.sort()
    middle = len(finite) // 2
    anchor = finite[middle] if len(finite) % 2 else (finite[middle - 1] + finite[middle]) / 2.0
    return math.floor(math.log10(anchor))


def _improvement(reference: float, current: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(current) or reference == 0:
        return math.nan
    return (1.0 if lower_is_better else -1.0) * (reference - current) / abs(reference) * 100.0


def _improvements_of_averages(rows: Sequence[Mapping[str, float]], methods: Sequence[str], reference: str,
                              lower_is_better: bool) -> dict[str, float]:
    averages = {}
    for method in methods:
        finite = [row.get(method, math.nan) for row in rows]
        finite = [value for value in finite if math.isfinite(value)]
        averages[method] = sum(finite) / len(finite) if finite else math.nan
    reference_average = averages.get(reference, math.nan)
    return {
        method: _improvement(reference_average, averages[method], lower_is_better)
        for method in methods
    }


def _format_cells(values: Mapping[str, float], methods: Sequence[str], decimals: int, *, lower_is_better: bool,
                  bold: bool, divisor: float = 1.0, percent: bool = False,
                  bold_methods: Sequence[str] | None = None) -> list[str]:
    eligible = set(methods if bold_methods is None else bold_methods)
    finite = [values.get(method, math.nan) for method in methods if method in eligible]
    finite = [value for value in finite if math.isfinite(value)]
    best = (min(finite) if lower_is_better else max(finite)) if finite else None
    cells = []
    for method in methods:
        raw = values.get(method, math.nan)
        if not math.isfinite(raw):
            cells.append("--")
            continue
        cell = f"{raw / divisor:.{decimals}f}" + (r"\%" if percent else "")
        if bold and method in eligible and best is not None and math.isclose(raw, best, rel_tol=1e-12, abs_tol=1e-15):
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


def build_table(results: Sequence[Result], *, metric: str = "mse", split: str = "eval",
                datasets: Sequence[str] | None = None, settings: Sequence[str] | None = None,
                dataset_settings: Mapping[str, set[str]] | None = None, methods: Sequence[str] | None = None,
                reference: str | None = None, decimals: int = 2, lower_is_better: bool = True,
                bold: bool = True, dataset_improvements: bool = True, setting_improvements: bool = True,
                overall_improvement: bool = True, positive_only: bool = False,
                auto_scale: bool = True, scale_exponent: int | None = None,
                scale_exponents: Mapping[tuple[str, str], int] | None = None,
                caption: str | None = None, label: str = "tab:results",
                excluded_from_bold: Sequence[str] | None = None, short_names: bool = True) -> str:
    """Render selected records as a complete LaTeX table environment."""
    filtered = [result for result in results if result.metric.casefold() == metric.casefold()
                and result.split.casefold() == split.casefold()]
    dataset_order = list(datasets) if datasets else sorted({result.dataset for result in filtered}, key=str.casefold)
    filtered = [result for result in filtered if result.dataset in set(dataset_order)]
    global_settings, per_dataset = set(settings or ()), dataset_settings or {}
    if global_settings or per_dataset:
        filtered = [
            result for result in filtered
            if (result.setting in per_dataset[result.dataset] if result.dataset in per_dataset
                else not global_settings or result.setting in global_settings)
        ]
    if methods:
        method_order = list(methods)
    else:
        method_order = sorted(
            {result.method for result in filtered if result.method.rsplit("/", 1)[-1] != "vanilla"},
            key=str.casefold,
        )
    filtered = [result for result in filtered if result.method in set(method_order)]
    if not filtered:
        raise ValueError(f"no results match metric={metric!r}, split={split!r}, and the selected filters")
    reference = reference or method_order[0]
    if reference not in method_order:
        raise ValueError(f"reference {reference!r} is not in selected methods {method_order}")

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for result in filtered:
        grouped.setdefault((result.dataset, result.setting, result.method), []).append(result.value)
    table: dict[tuple[str, str], dict[str, float]] = {}
    for (dataset, setting, method), values in grouped.items():
        finite = [value for value in values if math.isfinite(value)]
        table.setdefault((dataset, setting), {})[method] = sum(finite) / len(finite) if finite else math.nan
    if positive_only:
        improvements = _improvements_of_averages(
            list(table.values()),
            method_order,
            reference,
            lower_is_better,
        )
        keep_methods = {
            method
            for method in method_order
            if method == reference or improvements.get(method, math.nan) > 0.0
        }
        method_order = [method for method in method_order if method in keep_methods]
        table = {
            key: {method: value for method, value in row.items() if method in keep_methods}
            for key, row in table.items()
        }
    excluded_selectors = set(excluded_from_bold or ())
    excluded_methods = [
        method for method in method_order
        if method.rsplit("/", 1)[-1].startswith("oracle_")
        or _method_selected(method, excluded_selectors)
    ]
    regular_methods = [method for method in method_order if method not in set(excluded_methods)]
    method_order = [*regular_methods, *excluded_methods]
    dataset_order = [dataset for dataset in dataset_order if any(key[0] == dataset for key in table)]
    settings_by_dataset = {dataset: sorted((key[1] for key in table if key[0] == dataset), key=_setting_key)
                           for dataset in dataset_order}
    observed_settings = sorted({setting for _, setting in table}, key=_setting_key)
    exponent_overrides = scale_exponents or {}

    column_spec = "llc" + "r" * len(regular_methods)
    if excluded_methods:
        column_spec += "|" + "r" * len(excluded_methods)
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{{_latex(caption or f'{metric.upper()} results on {split}.')}}}",
             r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule",
             "Dataset & $L$--$H$ & Scale & "
             + " & ".join(_latex(_method_label(method, short_names)) for method in method_order) + r" \\",
             r"\midrule"]
    for dataset_index, dataset in enumerate(dataset_order):
        row_settings = settings_by_dataset[dataset]
        for setting_index, setting in enumerate(row_settings):
            row = table[(dataset, setting)]
            scale_values = [row[reference]] if math.isfinite(row.get(reference, math.nan)) else list(row.values())
            exponent = (exponent_overrides[(dataset, setting)] if (dataset, setting) in exponent_overrides
                        else scale_exponent if scale_exponent is not None
                        else _auto_exponent(scale_values, lower_is_better) if auto_scale else 0)
            dataset_cell = rf"\multirow{{{len(row_settings)}}}{{*}}{{{_latex(dataset)}}}" if setting_index == 0 else ""
            cells = _format_cells(row, method_order, decimals, lower_is_better=lower_is_better, bold=bold,
                                  divisor=10.0**exponent, bold_methods=regular_methods)
            lines.append(" & ".join([dataset_cell, _latex_setting(setting),
                                      rf"$\times 10^{{{exponent}}}$", *cells]) + r" \\")
        if dataset_improvements:
            improvements = _improvements_of_averages([table[(dataset, setting)] for setting in row_settings],
                                                      method_order, reference, lower_is_better)
            cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                                  percent=True, bold_methods=regular_methods)
            lines.append(" & ".join(["", r"\textit{Improvement}", "", *cells]) + r" \\")
        if dataset_index < len(dataset_order) - 1:
            lines.append(r"\midrule")
    if setting_improvements:
        lines.extend([r"\midrule", r"\multicolumn{%d}{l}{\textit{Improvements by setting}} \\" % (3 + len(method_order))])
        for setting in observed_settings:
            rows = [row for (_, row_setting), row in table.items() if row_setting == setting]
            improvements = _improvements_of_averages(rows, method_order, reference, lower_is_better)
            cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                                  percent=True, bold_methods=regular_methods)
            lines.append(" & ".join(["", _latex_setting(setting), "", *cells]) + r" \\")
    if overall_improvement:
        improvements = _improvements_of_averages(list(table.values()), method_order, reference, lower_is_better)
        cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                              percent=True, bold_methods=regular_methods)
        lines.extend([r"\midrule", " & ".join([r"\multicolumn{2}{l}{Overall improvement}", "", *cells]) + r" \\"])
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", rf"\label{{{_latex(label)}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def generate_results_table(experiment_dir: str | Path, output: str | Path | None = None, **kwargs: Any) -> Path:
    root = Path(experiment_dir).expanduser().resolve()
    default_name = f"results_{str(kwargs.get('metric', 'mse')).lower()}.tex"
    destination = Path(output).expanduser().resolve() if output else root / default_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_table(discover_results(root), **kwargs), encoding="utf-8", newline="\n"
    )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--methods", default=None, help="Comma/semicolon-separated ordered columns")
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--exclude-from-bold",
        default=None,
        help="Comma/semicolon-separated method IDs or variant names to move right and exclude from bolding",
    )
    parser.add_argument("--long-method-names", action="store_true")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--no-dataset-improvements", action="store_true")
    parser.add_argument("--no-setting-improvements", action="store_true")
    parser.add_argument("--no-overall-improvement", action="store_true")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Keep only methods with positive overall improvement versus --reference",
    )
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--scale-exponent", type=int, default=None)
    parser.add_argument("--row-scale", action="append", default=[], metavar="DATASET/L_H=EXPONENT")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:results")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = generate_results_table(
        args.experiment_dir, args.output, metric=args.metric, split=args.split,
        datasets=_split_names(args.datasets), settings=_split_names(args.settings),
        dataset_settings=_parse_dataset_settings(args.dataset_settings), methods=_split_names(args.methods),
        reference=args.reference, decimals=args.decimals, lower_is_better=not args.higher_is_better,
        bold=not args.no_bold, dataset_improvements=not args.no_dataset_improvements,
        setting_improvements=not args.no_setting_improvements, overall_improvement=not args.no_overall_improvement,
        positive_only=args.positive_only, auto_scale=not args.no_auto_scale, scale_exponent=args.scale_exponent,
        scale_exponents=_parse_scale_exponents(args.row_scale), caption=args.caption, label=args.label,
        excluded_from_bold=_split_names(args.exclude_from_bold), short_names=not args.long_method_names,
    )
    print(f"LaTeX table written to {output}")
    return output


if __name__ == "__main__":
    main()
