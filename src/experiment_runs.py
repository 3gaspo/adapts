"""Thesis-wide experiment directory, manifest, and run-selection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
SELECTION_NAME = "SELECTED_RUNS.txt"
VALID_STATUSES = {"not_run", "running", "interrupted", "completed"}
RUN_PATTERN = re.compile(r"run_(\d+)$")


class ManifestError(RuntimeError):
    """Raised when a run does not satisfy the current artifact contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration keys into stable dotted names."""
    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(flatten_mapping(item, name))
        else:
            output[name] = plain(item)
    return output


def slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.").lower()
    if not text or text in {".", ".."}:
        raise ValueError(f"cannot construct a path component from {value!r}")
    return text


def canonical_json(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _variant_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest.get("config", {})
    values = dict(config.get("pipeline", {}))
    values.update({f"experiment.{key}": value for key, value in config.get("experiment", {}).items()})
    return values


def identity_path(
    output_root: str | Path,
    dataset: str,
    lookback: int,
    horizon: int,
    backbone: str,
    model_config_order: Sequence[str],
    model_config: Mapping[str, Any],
) -> Path:
    missing = [name for name in model_config_order if name not in model_config]
    extra = [name for name in model_config if name not in model_config_order]
    if missing or extra:
        raise ValueError(f"model config/order mismatch missing={missing} extra={extra}")
    path = Path(output_root).expanduser()
    path /= slug(dataset)
    path /= f"{int(lookback)}_{int(horizon)}"
    path /= slug(backbone)
    for name in model_config_order:
        path /= slug(model_config[name])
    return path.resolve()


def computation_signature(
    identity: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    seeds: Sequence[int],
) -> str:
    """Identify a run only by its declared scientific parameters."""
    return signature(
        {
            "schema_version": SCHEMA_VERSION,
            "identity": dict(identity),
            "pipeline": dict(pipeline_config),
            "experiment": dict(experiment_config),
            "seeds": sorted(int(seed) for seed in seeds),
        }
    )


def manifest_computation_signature(manifest: Mapping[str, Any]) -> str:
    config = manifest.get("config", {})
    return computation_signature(
        manifest["identity"],
        config.get("pipeline", {}),
        config.get("experiment", {}),
        manifest.get("seeds", []),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path_or_run: str | Path) -> dict[str, Any]:
    path = Path(path_or_run)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"missing current manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"{path} uses schema_version={payload.get('schema_version')!r}; "
            f"migrate it to {SCHEMA_VERSION} or move it under archive/"
        )
    if payload.get("status") not in VALID_STATUSES:
        raise ManifestError(f"{path} has invalid status={payload.get('status')!r}")
    return payload


def discover_manifests(root: str | Path, *, completed_only: bool = False) -> list[tuple[Path, dict[str, Any]]]:
    base = Path(root).expanduser().resolve()
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(base.rglob(MANIFEST_NAME)):
        if "archive" in path.relative_to(base).parts:
            continue
        payload = load_manifest(path)
        if completed_only and payload["status"] != "completed":
            continue
        found.append((path.parent, payload))
    return found


def _run_dirs(identity_root: Path) -> list[Path]:
    if not identity_root.exists():
        return []
    return sorted(
        (path for path in identity_root.iterdir() if path.is_dir() and RUN_PATTERN.fullmatch(path.name)),
        key=lambda path: int(RUN_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )


def _archive_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    history = run_dir / "manifest_history"
    history.mkdir(parents=True, exist_ok=True)
    launch_id = slug(manifest.get("launch", {}).get("launch_id", "unknown"))
    stamp = re.sub(r"[^0-9]", "", utc_now())[:20]
    _atomic_json(history / f"{stamp}_{launch_id}.json", manifest)


def _parse_selections(identity_root: Path) -> dict[str, tuple[str, str]]:
    path = identity_root / SELECTION_NAME
    selected: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return selected
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[1] not in {"auto", "pinned"}:
            raise ManifestError(f"invalid selection line in {path}: {line!r}")
        selected[fields[0]] = (fields[1], fields[2])
    return selected


def _write_selections(identity_root: Path, selected: Mapping[str, tuple[str, str]]) -> None:
    lines = ["# pipeline_signature mode run"]
    lines.extend(f"{key}\t{mode}\t{run}" for key, (mode, run) in sorted(selected.items()))
    path = identity_root / SELECTION_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _update_auto_selection(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    selected = _parse_selections(run_dir.parent)
    pipeline_signature = manifest["signatures"]["pipeline"]
    current = selected.get(pipeline_signature)
    if current is None or current[0] == "auto":
        selected[pipeline_signature] = ("auto", run_dir.name)
        _write_selections(run_dir.parent, selected)


def set_selected_run(identity_root: str | Path, pipeline_signature: str, run_name: str, *, pinned: bool = True) -> None:
    root = Path(identity_root).expanduser().resolve()
    run_dir = root / run_name
    manifest = load_manifest(run_dir)
    if manifest["status"] != "completed":
        raise ManifestError(f"cannot select incomplete run: {run_dir}")
    if manifest["signatures"]["pipeline"] != pipeline_signature:
        raise ManifestError(f"{run_dir} does not have pipeline signature {pipeline_signature}")
    selected = _parse_selections(root)
    selected[pipeline_signature] = ("pinned" if pinned else "auto", run_name)
    _write_selections(root, selected)


def _latest_key(item: tuple[Path, Mapping[str, Any]]) -> str:
    manifest = item[1]
    launch = manifest.get("launch", {})
    return str(launch.get("finished_at_utc") or launch.get("started_at_utc") or launch.get("launched_at_utc") or "")


@dataclass(frozen=True)
class Allocation:
    run_dir: Path
    action: str
    computation_signature: str


def _allocate_run_unlocked(
    identity_root: str | Path,
    *,
    project: str,
    workflow: str,
    dataset: str,
    lookback: int,
    horizon: int,
    backbone: str,
    model_config_order: Sequence[str],
    model_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (),
    purpose: str = "development",
    mode: str | None = None,
    display_name: str | None = None,
    row_config: Sequence[str] = (),
    column_config: Sequence[str] = (),
    inputs: Mapping[str, Any] | None = None,
    policy: str = "overwrite_exact",
    skip_completed: bool = True,
    force: bool = False,
    run_index: int | None = None,
    launch_id: str | None = None,
) -> Allocation:
    if policy not in {"overwrite_exact", "overwrite_path", "new"}:
        raise ValueError(f"unknown conflict policy: {policy}")
    root = Path(identity_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_config = dict(model_config)
    identity = {
        "dataset": str(dataset),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "backbone": str(backbone),
        "model_config_order": list(model_config_order),
        "model_config": model_config,
    }
    path_signature = signature(identity)
    pipeline_signature = signature(
        {"pipeline": dict(pipeline_config), "experiment": dict(experiment_config or {})}
    )
    run_signature = computation_signature(
        identity, pipeline_config, experiment_config or {}, seeds
    )
    existing: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in _run_dirs(root):
        existing.append((run_dir, load_manifest(run_dir)))
    exact = [item for item in existing if manifest_computation_signature(item[1]) == run_signature]

    action = "new"
    target: Path | None = None
    old_manifest: dict[str, Any] | None = None
    if run_index is not None:
        target = root / f"run_{int(run_index)}"
        if target.exists():
            old_manifest = load_manifest(target)
            same_computation = manifest_computation_signature(old_manifest) == run_signature
            if not same_computation and policy != "overwrite_path":
                raise ManifestError(f"{target} contains a different computation; use overwrite_path or another RUN_INDEX")
            if same_computation:
                status = old_manifest["status"]
                if status == "completed" and skip_completed and not force:
                    purposes = sorted(set(old_manifest.get("purposes", [])) | {purpose})
                    if purposes != old_manifest.get("purposes", []):
                        old_manifest["purposes"] = purposes
                        _atomic_json(target / MANIFEST_NAME, old_manifest)
                    return Allocation(target, "skip", run_signature)
                if status == "running" and old_manifest.get("launch", {}).get("launch_id") != launch_id:
                    raise ManifestError(f"matching run is already running: {target}")
                action = "resume" if status in {"not_run", "interrupted"} and not force else "overwrite"
            else:
                action = "overwrite"
    elif exact and policy != "new":
        target, old_manifest = max(exact, key=_latest_key)
        status = old_manifest["status"]
        if status == "completed" and skip_completed and not force:
            purposes = sorted(set(old_manifest.get("purposes", [])) | {purpose})
            if purposes != old_manifest.get("purposes", []):
                old_manifest["purposes"] = purposes
                _atomic_json(target / MANIFEST_NAME, old_manifest)
            return Allocation(target, "skip", run_signature)
        if status == "running" and old_manifest.get("launch", {}).get("launch_id") != launch_id:
            raise ManifestError(f"matching run is already running: {target}")
        action = "resume" if status in {"not_run", "interrupted"} and not force else "overwrite"
    elif policy == "overwrite_path" and existing:
        target, old_manifest = max(existing, key=_latest_key)
        action = "overwrite"
    if target is None:
        next_index = 0 if not existing else max(int(RUN_PATTERN.fullmatch(path.name).group(1)) for path, _ in existing) + 1  # type: ignore[union-attr]
        target = root / f"run_{next_index}"
        action = "new"

    if old_manifest is not None and action == "overwrite":
        _archive_manifest(target, old_manifest)
    now = utc_now()
    effective_launch_id = str(launch_id or os.environ.get("SLURM_JOB_ID") or uuid.uuid4())
    attempt = {
        "launch_id": effective_launch_id,
        "launched_at_utc": now,
        "mode": mode,
        "purpose": purpose,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    attempts = []
    if old_manifest is not None and action == "resume":
        attempts = list(old_manifest.get("launch", {}).get("attempts", []))
    attempts.append(attempt)
    seed_status = {str(int(seed)): {"status": "not_run", "artifacts": []} for seed in seeds}
    if old_manifest is not None and action == "resume":
        for seed, state in old_manifest.get("seed_status", {}).items():
            if seed in seed_status and state.get("status") == "completed":
                seed_status[seed] = state
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "workflow": {"path": [part for part in workflow.replace("\\", "/").split("/") if part]},
        "manifest_id": str(uuid.uuid4()),
        "run_index": int(RUN_PATTERN.fullmatch(target.name).group(1)),  # type: ignore[union-attr]
        "status": "not_run",
        "purposes": sorted(set((old_manifest or {}).get("purposes", [])) | {purpose}),
        "identity": identity,
        "config": {
            "model": model_config,
            "pipeline": dict(pipeline_config),
            "runtime": dict(runtime_config or {}),
            "experiment": dict(experiment_config or {}),
        },
        "inputs": dict(inputs or {}),
        "signatures": {
            "path": path_signature,
            "pipeline": pipeline_signature,
            "computation": run_signature,
        },
        "table": {
            "display_name": display_name or backbone,
            "row_config": list(row_config),
            "column_config": list(column_config),
        },
        "seeds": sorted(int(seed) for seed in seeds),
        "seed_status": seed_status,
        "launch": {
            "launch_id": effective_launch_id,
            "launched_at_utc": now,
            "started_at_utc": None,
            "finished_at_utc": None,
            "mode": mode,
            "attempts": attempts,
        },
        "artifacts": {"required": [], "files": []},
    }
    target.mkdir(parents=True, exist_ok=True)
    _atomic_json(target / MANIFEST_NAME, manifest)
    return Allocation(target, action, run_signature)


def allocate_run(*args: Any, **kwargs: Any) -> Allocation:
    """Allocate a run while serializing concurrent writers per identity path."""
    identity_root = args[0] if args else kwargs["identity_root"]
    root = Path(identity_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".allocate.lock"
    deadline = time.monotonic() + 30.0
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} time={utc_now()}\n".encode("utf-8"))
            os.close(descriptor)
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 3600:
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise ManifestError(f"timed out waiting for run allocation lock: {lock}")
            time.sleep(0.05)
    try:
        return _allocate_run_unlocked(*args, **kwargs)
    finally:
        lock.unlink(missing_ok=True)


def mark_status(
    run_dir: str | Path,
    status: str,
    *,
    required_artifacts: Sequence[str] = (),
    seed: int | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    now = utc_now()
    if status == "running":
        manifest["launch"]["started_at_utc"] = manifest["launch"].get("started_at_utc") or now
    if status == "completed":
        missing = [name for name in required_artifacts if not (root / name).is_file()]
        empty = [name for name in required_artifacts if (root / name).is_file() and (root / name).stat().st_size == 0]
        if missing or empty:
            raise ManifestError(f"cannot complete {root}: missing={missing} empty={empty}")
        records = [
            {"path": name, "size": (root / name).stat().st_size}
            for name in required_artifacts
        ]
        manifest["artifacts"] = {"required": list(required_artifacts), "files": records}
        manifest["launch"]["finished_at_utc"] = now
    if seed is None:
        manifest["status"] = status
        for item in manifest.get("seed_status", {}).values():
            if item.get("status") != "completed" or status == "completed":
                item["status"] = status
    else:
        key = str(int(seed))
        if key not in manifest.get("seed_status", {}):
            raise ManifestError(f"seed {seed} is not declared by {root}")
        manifest["seed_status"][key]["status"] = status
        manifest["seed_status"][key]["artifacts"] = list(required_artifacts)
        states = {item["status"] for item in manifest["seed_status"].values()}
        manifest["status"] = "completed" if states == {"completed"} else ("running" if "running" in states else "interrupted")
        if manifest["status"] == "completed":
            manifest["launch"]["finished_at_utc"] = now
    _atomic_json(root / MANIFEST_NAME, manifest)
    if manifest["status"] == "completed":
        _update_auto_selection(root, manifest)
    return manifest


def validate_completed(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    if manifest["status"] != "completed":
        raise ManifestError(f"run is not completed: {root}")
    missing = [name for name in manifest.get("artifacts", {}).get("required", []) if not (root / name).is_file()]
    if missing:
        raise ManifestError(f"completed run is missing required artifacts: {missing}")
    return manifest


def interrupt_launch(root: str | Path, launch_id: str) -> list[Path]:
    """Mark running manifests owned by a failed or cancelled launch as interrupted."""
    changed: list[Path] = []
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return changed
    for manifest_path in sorted(base.rglob(MANIFEST_NAME)):
        if "archive" in manifest_path.relative_to(base).parts:
            continue
        manifest = load_manifest(manifest_path)
        if manifest["status"] == "running" and str(manifest.get("launch", {}).get("launch_id")) == str(launch_id):
            mark_status(manifest_path.parent, "interrupted")
            changed.append(manifest_path.parent)
    return changed


@dataclass(frozen=True)
class SelectedRun:
    run_dir: Path
    manifest: Mapping[str, Any]
    label: str


def _pipeline_matches(manifest: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    config = _variant_config(manifest)
    return all(key in config and plain(config[key]) == plain(value) for key, value in requested.items())


def select_identity_runs(
    identity_root: str | Path,
    *,
    requested_pipeline: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: Iterable[str] | None = None,
) -> list[SelectedRun]:
    if config_policy not in {"distinct", "latest", "selected", "average"}:
        raise ValueError(f"unknown config policy: {config_policy}")
    if repeat_policy not in {"distinct", "latest", "selected", "average"}:
        raise ValueError(f"unknown repeat policy: {repeat_policy}")
    root = Path(identity_root).expanduser().resolve()
    candidates = [(path, manifest) for path, manifest in ((path, load_manifest(path)) for path in _run_dirs(root)) if manifest["status"] == "completed"]
    wanted_purposes = set(purposes or ())
    if wanted_purposes:
        candidates = [item for item in candidates if wanted_purposes & set(item[1].get("purposes", []))]
    requested = dict(requested_pipeline or {})
    if requested:
        candidates = [item for item in candidates if _pipeline_matches(item[1], requested)]
    if not candidates:
        raise ManifestError(f"no completed run in {root} matches pipeline={requested or 'any'}")

    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for item in candidates:
        groups.setdefault(item[1]["signatures"]["pipeline"], []).append(item)
    selections = _parse_selections(root)
    if config_policy in {"latest", "selected"} and len(groups) > 1:
        if config_policy == "latest":
            chosen = max(candidates, key=_latest_key)[1]["signatures"]["pipeline"]
        else:
            selected_names = {run for _, run in selections.values()}
            selected_candidates = [item for item in candidates if item[0].name in selected_names]
            if not selected_candidates:
                raise ManifestError(f"{root} has no selected completed configuration")
            chosen = max(selected_candidates, key=_latest_key)[1]["signatures"]["pipeline"]
        groups = {chosen: groups[chosen]}

    differing_keys: set[str] = set()
    if len(groups) > 1:
        all_configs = [_variant_config(items[0][1]) for items in groups.values()]
        for key in set().union(*(config.keys() for config in all_configs)):
            if len({canonical_json(config.get(key)) for config in all_configs}) > 1:
                differing_keys.add(key)

    output: list[SelectedRun] = []
    for pipeline_sig, items in sorted(groups.items()):
        items = sorted(items, key=_latest_key)
        if repeat_policy == "latest":
            chosen_items = [items[-1]]
        elif repeat_policy == "selected":
            selected = selections.get(pipeline_sig)
            if selected is None:
                chosen_items = [items[-1]]
            else:
                chosen_items = [item for item in items if item[0].name == selected[1]]
                if not chosen_items:
                    raise ManifestError(f"{root}/{selected[1]} is selected but unavailable")
        else:
            chosen_items = items
        for path, manifest in chosen_items:
            label = str(manifest.get("table", {}).get("display_name") or manifest["identity"]["backbone"])
            if differing_keys and config_policy != "average":
                config = _variant_config(manifest)
                suffix = "__".join(f"{slug(key)}-{slug(config.get(key))}" for key in sorted(differing_keys))
                label = f"{label}__{suffix}"
            if repeat_policy == "distinct" and len(items) > 1:
                label = f"{label}_{path.name}"
            output.append(SelectedRun(path, manifest, label))
    return output


def write_report_manifest(
    destination: str | Path,
    *,
    inputs: Sequence[SelectedRun],
    config_policy: str,
    repeat_policy: str,
    filters: Mapping[str, Any],
) -> Path:
    path = Path(destination).expanduser().resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "config_policy": config_policy,
        "repeat_policy": repeat_policy,
        "filters": dict(filters),
        "inputs": [
            {
                "manifest_id": item.manifest["manifest_id"],
                "path": str(item.run_dir / MANIFEST_NAME),
                "label": item.label,
                "pipeline_signature": item.manifest["signatures"]["pipeline"],
            }
            for item in inputs
        ],
    }
    _atomic_json(path, payload)
    return path


def _value(text: str) -> Any:
    lowered = text.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _pairs(values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key] = _value(value)
    return result


def _bool(value: str) -> bool:
    return value.casefold() in {"1", "true", "yes", "on"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    allocate = commands.add_parser("allocate")
    allocate.add_argument("--identity-root", required=True)
    allocate.add_argument("--project", required=True)
    allocate.add_argument("--workflow", required=True)
    allocate.add_argument("--dataset", required=True)
    allocate.add_argument("--lookback", type=int, required=True)
    allocate.add_argument("--horizon", type=int, required=True)
    allocate.add_argument("--backbone", required=True)
    allocate.add_argument("--model-config-order", default="")
    allocate.add_argument("--model-config", action="append", default=[])
    allocate.add_argument("--pipeline-config", action="append", default=[])
    allocate.add_argument("--runtime-config", action="append", default=[])
    allocate.add_argument("--experiment-config", action="append", default=[])
    allocate.add_argument("--seed", action="append", type=int, default=[])
    allocate.add_argument("--purpose", default="development")
    allocate.add_argument("--mode")
    allocate.add_argument("--display-name")
    allocate.add_argument("--row-config", default="")
    allocate.add_argument("--column-config", default="")
    allocate.add_argument("--input", action="append", default=[])
    allocate.add_argument("--policy", default="overwrite_exact")
    allocate.add_argument("--skip-completed", default="true")
    allocate.add_argument("--force", default="false")
    allocate.add_argument("--run-index", type=int)
    allocate.add_argument("--launch-id")

    status = commands.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    status.add_argument("--artifact", action="append", default=[])
    status.add_argument("--seed", type=int)

    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", required=True)

    select = commands.add_parser("select")
    select.add_argument("--identity-root", required=True)
    select.add_argument("--pipeline-signature", required=True)
    select.add_argument("--run", required=True)
    select.add_argument("--auto", action="store_true")

    interrupt = commands.add_parser("interrupt-launch")
    interrupt.add_argument("--root", required=True)
    interrupt.add_argument("--launch-id", required=True)

    pending = commands.add_parser("pending-seeds")
    pending.add_argument("--run-dir", required=True)

    resolve = commands.add_parser("resolve")
    resolve.add_argument("--identity-root", required=True)
    resolve.add_argument("--pipeline-config", action="append", default=[])
    resolve.add_argument("--config-policy", default="selected")
    resolve.add_argument("--repeat-policy", default="selected")
    resolve.add_argument("--purpose", action="append", default=[])

    args = parser.parse_args(argv)
    if args.command == "allocate":
        model_order = [item for item in args.model_config_order.split(",") if item]
        inputs = {key: {"path": str(value)} for key, value in _pairs(args.input).items()}
        result = allocate_run(
            args.identity_root,
            project=args.project,
            workflow=args.workflow,
            dataset=args.dataset,
            lookback=args.lookback,
            horizon=args.horizon,
            backbone=args.backbone,
            model_config_order=model_order,
            model_config=_pairs(args.model_config),
            pipeline_config=_pairs(args.pipeline_config),
            runtime_config=_pairs(args.runtime_config),
            experiment_config=_pairs(args.experiment_config),
            seeds=args.seed,
            purpose=args.purpose,
            mode=args.mode,
            display_name=args.display_name,
            row_config=[item for item in args.row_config.split(",") if item],
            column_config=[item for item in args.column_config.split(",") if item],
            inputs=inputs,
            policy=args.policy,
            skip_completed=_bool(args.skip_completed),
            force=_bool(args.force),
            run_index=args.run_index,
            launch_id=args.launch_id,
        )
        print(f"{result.run_dir}\t{result.action}\t{result.computation_signature}")
    elif args.command == "status":
        mark_status(args.run_dir, args.status, required_artifacts=args.artifact, seed=args.seed)
    elif args.command == "validate":
        validate_completed(args.run_dir)
        print(Path(args.run_dir).expanduser().resolve())
    elif args.command == "select":
        set_selected_run(args.identity_root, args.pipeline_signature, args.run, pinned=not args.auto)
    elif args.command == "interrupt-launch":
        for run_dir in interrupt_launch(args.root, args.launch_id):
            print(run_dir)
    elif args.command == "pending-seeds":
        manifest = load_manifest(args.run_dir)
        print(",".join(seed for seed, state in manifest.get("seed_status", {}).items() if state.get("status") != "completed"))
    else:
        for selected_run in select_identity_runs(
            args.identity_root,
            requested_pipeline=_pairs(args.pipeline_config),
            config_policy=args.config_policy,
            repeat_policy=args.repeat_policy,
            purposes=args.purpose,
        ):
            print(f"{selected_run.run_dir}\t{selected_run.label}\t{selected_run.manifest['manifest_id']}")


if __name__ == "__main__":
    main()
