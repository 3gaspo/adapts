"""Validate atomic completion markers for adaptation extraction payloads."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


MANIFEST_NAME = "extraction_manifest.json"
PREDICTION_FILES = tuple(
    f"{split}_prediction_payload.pt" for split in ("adapt", "eval")
)
FEATURE_FILES = tuple(
    f"{split}_features_payload.pt" for split in ("adapt", "eval")
)


def required_extraction_files(*, vanilla: bool) -> tuple[str, ...]:
    files = (*PREDICTION_FILES, *FEATURE_FILES)
    return (*files, "vanilla_metrics.json") if vanilla else files


def manifest_path(directory: str | Path) -> Path:
    return Path(directory).expanduser() / MANIFEST_NAME


def invalidate_extraction(directory: str | Path) -> None:
    """Remove the success marker before any payload can be overwritten."""
    manifest_path(directory).unlink(missing_ok=True)


def write_extraction_manifest(
    directory: str | Path,
    *,
    signature: Mapping[str, Any],
    required_files: Sequence[str],
) -> Path:
    """Atomically mark an extraction complete after all required files exist."""
    root = Path(directory).expanduser()
    sizes: dict[str, int] = {}
    for name in required_files:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"cannot complete extraction; missing or empty artifact: {path}")
        sizes[name] = path.stat().st_size
    payload = {
        "format_version": 2,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "signature": dict(signature),
        "files": sizes,
    }
    destination = manifest_path(root)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def validate_extraction(
    directory: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Check marker contents and exact artifact sizes without loading large tensors."""
    root = Path(directory).expanduser()
    marker = manifest_path(root)
    if not marker.is_file():
        return False, f"missing completion marker: {marker}"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion marker {marker}: {exc}"
    if payload.get("format_version") != 2 or payload.get("status") != "complete":
        return False, f"unsupported or incomplete marker: {marker}"
    signature = payload.get("signature")
    if not isinstance(signature, Mapping):
        return False, f"marker has no extraction signature: {marker}"
    if expected_signature is not None and dict(signature) != dict(expected_signature):
        return False, "completion marker belongs to a different extraction configuration"
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        return False, f"marker has no artifact inventory: {marker}"
    for name, expected_size in files.items():
        path = root / str(name)
        if not path.is_file():
            return False, f"missing artifact recorded by marker: {path}"
        actual_size = path.stat().st_size
        if actual_size <= 0 or actual_size != expected_size:
            return False, (
                f"artifact size changed after completion: {path} "
                f"(expected {expected_size}, found {actual_size})"
            )
    missing_predictions = [name for name in PREDICTION_FILES if name not in files]
    if missing_predictions:
        return False, f"marker is missing required prediction payloads: {missing_predictions}"
    return True, "complete"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.captureWarnings(True)
    args = parse_args(argv)
    complete, reason = validate_extraction(args.directory)
    LOGGER.info("extraction validation directory=%s status=%s", args.directory, reason)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
