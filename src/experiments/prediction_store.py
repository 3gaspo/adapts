"""Disk-backed prediction artifacts for bounded-memory evaluation and dashboards."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np


FORMAT = "adaptation_prediction_store"


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not name:
        raise ValueError("artifact names must contain at least one safe character")
    return name


class PredictionStore:
    """Write one NumPy array at a time and publish one final manifest."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "predictions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict[str, dict[str, str]]] = {}

    def _path(self, split: str, kind: str, name: str) -> Path:
        split_name = _safe_name(split)
        kind_name = _safe_name(kind)
        target = self.root / split_name / kind_name / f"{_safe_name(name)}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = target.relative_to(self.output_dir).as_posix()
        self.entries.setdefault(split, {}).setdefault(kind, {})[name] = relative
        return target

    def write(
        self,
        split: str,
        kind: str,
        name: str,
        value: np.ndarray,
    ) -> Path:
        path = self._path(split, kind, name)
        np.save(path, np.asarray(value), allow_pickle=False)
        return path

    def open(
        self,
        split: str,
        kind: str,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type[np.generic] = np.float32,
    ) -> np.memmap:
        path = self._path(split, kind, name)
        return np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )

    def finalize(self, *, metadata: dict[str, Any] | None = None) -> Path:
        manifest = self.output_dir / "prediction_manifest.json"
        payload = {
            "format": FORMAT,
            "splits": self.entries,
            "metadata": {} if metadata is None else metadata,
        }
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest


def load_prediction_store(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = output / "prediction_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"{manifest} is not a current prediction store")
    splits: dict[str, Any] = {}
    for split, kinds in payload.get("splits", {}).items():
        splits[split] = {
            kind: {
                name: np.load(output / relative, mmap_mode="r", allow_pickle=False)
                for name, relative in entries.items()
            }
            for kind, entries in kinds.items()
        }
    return {
        "format": FORMAT,
        "splits": splits,
        "metadata": payload.get("metadata", {}),
        "manifest": manifest,
    }
