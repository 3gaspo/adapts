"""Audit and migrate adaptation runs to explicit scientific input dependencies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment_runs import (
    MANIFEST_NAME,
    SELECTION_NAME,
    computation_signature,
    plain,
    signature,
    utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _manifest_paths(outputs: Path) -> list[Path]:
    return [
        path
        for path in outputs.rglob(MANIFEST_NAME)
        if "archive" not in path.relative_to(outputs).parts
        and "manifest_history" not in path.relative_to(outputs).parts
    ]


def _local_input(outputs: Path, recorded_path: str) -> Path | None:
    normalized = recorded_path.replace("\\", "/")
    if "/outputs/" not in normalized:
        return None
    return outputs / Path(normalized.split("/outputs/", 1)[1])


def _recorded_input(outputs: Path, local_path: Path, original: str) -> str:
    prefix = original.replace("\\", "/").split("/outputs/", 1)[0]
    return f"{prefix}/outputs/{local_path.relative_to(outputs).as_posix()}"


def _schedule(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.get("config", {}).get("pipeline", {}).items()
        if key != "retrieval.scope"
    }


def _dependency(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config", {})
    return {
        "schema_version": manifest["schema_version"],
        "identity": plain(manifest["identity"]),
        "pipeline": plain(config.get("pipeline", {})),
        "experiment": plain(config.get("experiment", {})),
        "seeds": sorted(int(seed) for seed in manifest.get("seeds", [])),
    }


def _is_vanilla_extraction(manifest: dict[str, Any]) -> bool:
    model = manifest.get("identity", {}).get("model_config", {})
    return (
        manifest.get("workflow", {}).get("path") == ["extraction"]
        and model.get("space") is None
        and model.get("metric") is None
        and model.get("k") == 0
        and model.get("mode") is None
    )


def _same_panel(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_identity = left["identity"]
    right_identity = right["identity"]
    return all(
        left_identity[key] == right_identity[key]
        for key in ("dataset", "lookback", "horizon", "backbone")
    )


def _repair_selections(
    touched: dict[Path, list[tuple[str, str | None, str, str, str]]]
) -> int:
    repaired = 0
    for identity_root, changes in touched.items():
        selection_path = identity_root / SELECTION_NAME
        entries: dict[str, tuple[str, str]] = {}
        if selection_path.exists():
            for line in selection_path.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                pipeline_signature, mode, run = line.split("\t")
                entries[pipeline_signature] = (mode, run)

        for old_signature, _, run, _, _ in changes:
            if entries.get(old_signature, (None, None))[1] == run:
                entries.pop(old_signature, None)

        completed_by_signature: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for old_signature, new_signature, run, status, finished_at in changes:
            if new_signature is not None and status == "completed":
                completed_by_signature[new_signature].append(
                    (run, old_signature, finished_at)
                )
        for new_signature, candidates in completed_by_signature.items():
            current = entries.get(new_signature)
            if current is not None:
                current_manifest_path = identity_root / current[1] / MANIFEST_NAME
                if current_manifest_path.exists():
                    current_manifest = _read(current_manifest_path)
                    if (
                        current_manifest.get("status") == "completed"
                        and current_manifest.get("signatures", {}).get("pipeline")
                        == new_signature
                    ):
                        continue
            chosen = max(candidates, key=lambda item: item[2])
            mode = "auto"
            for run, old_signature, _ in candidates:
                old_selection = next(
                    (
                        change
                        for change in changes
                        if change[0] == old_signature and change[2] == run
                    ),
                    None,
                )
                if old_selection is not None:
                    previous_lines = selection_path.read_text(encoding="utf-8").splitlines() if selection_path.exists() else []
                    for line in previous_lines:
                        if line == f"{old_signature}\tpinned\t{run}":
                            chosen = (run, old_signature, old_selection[4])
                            mode = "pinned"
                            break
                if mode == "pinned":
                    break
            entries[new_signature] = (mode, chosen[0])

        lines = ["# pipeline_signature mode run"]
        lines.extend(
            f"{pipeline_signature}\t{mode}\t{run}"
            for pipeline_signature, (mode, run) in sorted(entries.items())
        )
        with selection_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
        repaired += 1
    return repaired


def audit(outputs: Path, *, apply: bool) -> dict[str, int]:
    paths = _manifest_paths(outputs)
    manifests = {path: _read(path) for path in paths}
    touched: dict[Path, list[tuple[str, str | None, str, str, str]]] = defaultdict(list)
    extraction_defaults = {
        "data.split_bounds": "ratios",
        "data.standardize_train_boundary": None,
        "retrieval.scope": "all",
    }
    extraction_migrations = 0
    for manifest_path, manifest in manifests.items():
        if manifest.get("workflow", {}).get("path") != ["extraction"]:
            continue
        pipeline = manifest.get("config", {}).get("pipeline", {})
        missing = [key for key in extraction_defaults if key not in pipeline]
        if not missing:
            continue
        extraction_manifest_path = manifest_path.parent / "extraction_manifest.json"
        if not extraction_manifest_path.exists():
            raise RuntimeError(f"cannot reconstruct extraction defaults for {manifest_path}")
        extraction_signature = _read(extraction_manifest_path).get("signature", {})
        if "retrieval_scope" in extraction_signature or "split_bounds" in extraction_signature:
            raise RuntimeError(f"non-default historical extraction requires manual audit: {manifest_path}")
        extraction_migrations += 1
        old_signature = manifest["signatures"]["pipeline"]
        pipeline.update(extraction_defaults)
        experiment = manifest["config"].get("experiment", {})
        new_signature = signature({"pipeline": pipeline, "experiment": experiment})
        manifest["signatures"]["pipeline"] = new_signature
        manifest["signatures"]["computation"] = computation_signature(
            manifest["identity"], pipeline, experiment, manifest.get("seeds", [])
        )
        if apply:
            _write(manifest_path, manifest)
            touched[manifest_path.parent.parent].append(
                (
                    old_signature,
                    new_signature,
                    manifest_path.parent.name,
                    str(manifest.get("status")),
                    str(manifest.get("launch", {}).get("finished_at_utc") or ""),
                )
            )
    vanilla = [
        (path, manifest)
        for path, manifest in manifests.items()
        if manifest.get("status") == "completed"
        and "publication" in manifest.get("purposes", [])
        and _is_vanilla_extraction(manifest)
    ]
    counts = {
        "manifests": len(manifests),
        "extraction_manifest_migrations": extraction_migrations,
        "scientifically_invalid": 0,
        "provenance_repairs": 0,
        "dependency_migrations": 0,
        "missing_inputs": 0,
        "selection_indexes": 0,
    }
    for manifest_path, manifest in manifests.items():
        if manifest.get("launch", {}).get("invalidated_at_utc"):
            continue
        upstream_ref = manifest.get("inputs", {}).get("upstream_manifest")
        if not isinstance(upstream_ref, dict) or not upstream_ref.get("path"):
            continue
        if not str(upstream_ref["path"]).replace("\\", "/").endswith("/manifest.json"):
            continue
        upstream_path = _local_input(outputs, str(upstream_ref["path"]))
        if upstream_path is None or upstream_path not in manifests:
            counts["missing_inputs"] += 1
            continue
        upstream = manifests[upstream_path]
        old_signature = manifest["signatures"]["pipeline"]
        status = str(manifest.get("status"))
        publication = "publication" in manifest.get("purposes", [])
        invalid = (
            status == "completed"
            and publication
            and "publication" not in upstream.get("purposes", [])
        )
        if invalid:
            counts["scientifically_invalid"] += 1
            if apply:
                manifest["status"] = "interrupted"
                manifest.setdefault("launch", {})["invalidated_at_utc"] = utc_now()
                _write(manifest_path, manifest)
                touched[manifest_path.parent.parent].append(
                    (
                        old_signature,
                        None,
                        manifest_path.parent.name,
                        "interrupted",
                        str(manifest.get("launch", {}).get("finished_at_utc") or ""),
                    )
                )
            continue

        vanilla_ref = manifest.get("inputs", {}).get("vanilla_manifest")
        if isinstance(vanilla_ref, dict) and vanilla_ref.get("path"):
            vanilla_path = _local_input(outputs, str(vanilla_ref["path"]))
            vanilla_manifest = manifests.get(vanilla_path) if vanilla_path else None
            if publication and vanilla_manifest is not None and "publication" not in vanilla_manifest.get("purposes", []):
                candidates = [
                    (path, candidate)
                    for path, candidate in vanilla
                    if _same_panel(upstream, candidate)
                    and _schedule(upstream) == _schedule(candidate)
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"expected one publication vanilla match for {manifest_path}, got {len(candidates)}"
                    )
                replacement_path, _ = candidates[0]
                counts["provenance_repairs"] += 1
                if apply:
                    vanilla_ref["path"] = _recorded_input(
                        outputs, replacement_path, str(vanilla_ref["path"])
                    )

        pipeline = manifest.setdefault("config", {}).setdefault("pipeline", {})
        dependency = _dependency(upstream)
        if pipeline.get("dependency.extraction") == dependency:
            continue
        counts["dependency_migrations"] += 1
        if not apply:
            continue
        pipeline["dependency.extraction"] = dependency
        experiment = manifest["config"].get("experiment", {})
        new_signature = signature({"pipeline": pipeline, "experiment": experiment})
        manifest["signatures"]["pipeline"] = new_signature
        manifest["signatures"]["computation"] = computation_signature(
            manifest["identity"], pipeline, experiment, manifest.get("seeds", [])
        )
        _write(manifest_path, manifest)
        touched[manifest_path.parent.parent].append(
            (
                old_signature,
                new_signature,
                manifest_path.parent.name,
                status,
                str(manifest.get("launch", {}).get("finished_at_utc") or ""),
            )
        )

    if apply:
        counts["selection_indexes"] = _repair_selections(touched)
        normalized = 0
        contract_paths = [*manifests]
        contract_paths.extend(
            path
            for path in outputs.rglob(SELECTION_NAME)
            if "archive" not in path.relative_to(outputs).parts
        )
        for path in contract_paths:
            content = path.read_bytes()
            if b"\r\n" not in content:
                continue
            path.write_bytes(content.replace(b"\r\n", b"\n"))
            normalized += 1
        counts["normalized_line_endings"] = normalized
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    counts = audit(args.outputs.resolve(), apply=args.apply)
    print(json.dumps({"applied": args.apply, **counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
