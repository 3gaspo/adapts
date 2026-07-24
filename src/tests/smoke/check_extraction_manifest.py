"""Smoke-check safe extraction completion markers without loading tensors."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location("adaptation_artifacts", ROOT / "src" / "experiments" / "artifacts.py")
assert SPEC is not None and SPEC.loader is not None
ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARTIFACTS)
invalidate_extraction = ARTIFACTS.invalidate_extraction
validate_extraction = ARTIFACTS.validate_extraction
write_extraction_manifest = ARTIFACTS.write_extraction_manifest


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = ("adapt_prediction_payload.pt", "eval_prediction_payload.pt")
        for index, name in enumerate(files, start=1):
            (root / name).write_bytes(bytes([index]) * index)
        signature = {"dataset_name": "toy", "lags": 4, "horizon": 2}
        write_extraction_manifest(root, signature=signature, required_files=files)
        assert validate_extraction(root, expected_signature=signature) == (True, "complete")
        complete, reason = validate_extraction(root, expected_signature={**signature, "horizon": 3})
        assert not complete and "different extraction configuration" in reason
        (root / files[0]).write_bytes(b"changed")
        complete, reason = validate_extraction(root, expected_signature=signature)
        assert not complete and "size changed" in reason
        invalidate_extraction(root)
        complete, reason = validate_extraction(root)
        assert not complete and "missing completion marker" in reason
    print("extraction manifest checks passed")


if __name__ == "__main__":
    main()
