"""Check current-manifest timing-table input resolution."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiment_runs import allocate_run, mark_status
from visu.timing_table import build_timing_table


def _timing_input(root: Path, name: str, seconds: float) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    (directory / "extraction_timing.json").write_text(
        json.dumps({"elapsed_seconds": seconds}), encoding="utf-8"
    )
    return manifest


def _result_run(
    root: Path,
    *,
    formula: str,
    retrieval: tuple[str, str, int, str],
    artifact: str,
    seconds: float,
    inputs: dict,
):
    space, metric, neighbors, mode = retrieval
    config = {
        "formula": formula,
        "space": space,
        "metric": metric,
        "k": neighbors,
        "mode": mode,
    }
    identity = root / "Electricity/512_64/chronos-bolt"
    for name in config:
        identity /= str(config[name])
    allocation = allocate_run(
        identity,
        project="adaptation",
        workflow="crossrag",
        dataset="Electricity",
        lookback=512,
        horizon=64,
        backbone="chronos-bolt",
        model_config_order=list(config),
        model_config=config,
        pipeline_config={},
        inputs=inputs,
        display_name=formula,
    )
    (allocation.run_dir / artifact).write_text(
        json.dumps({"elapsed_seconds": seconds}), encoding="utf-8"
    )
    mark_status(allocation.run_dir, "completed", required_artifacts=[artifact])
    return allocation


def main() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        vanilla = _timing_input(root, "inputs/vanilla", 1.0)
        candidate_extraction = _timing_input(root, "inputs/candidate", 3.0)
        crossrag_extraction = _timing_input(root, "inputs/crossrag", 4.0)
        _result_run(
            root,
            formula="cov_ridge_shared",
            retrieval=("raw", "euclidean", 1, "online"),
            artifact="baseline_timing.json",
            seconds=2.0,
            inputs={
                "vanilla_manifest": {"path": str(vanilla)},
                "upstream_manifest": {"path": str(candidate_extraction)},
            },
        )
        _result_run(
            root,
            formula="crossrag",
            retrieval=("minmax", "cosine", 15, "online"),
            artifact="crossrag_timing.json",
            seconds=5.0,
            inputs={"upstream_manifest": {"path": str(crossrag_extraction)}},
        )
        selected = []
        latex = build_timing_table(
            root,
            datasets=["Electricity"],
            setting="512_64",
            model="chronos-bolt",
            candidate_run="raw_euclidean_1_online",
            crossrag_run="minmax_cosine_15_online",
            candidate_family="baselines",
            candidate_formula="cov_ridge_shared",
            selected_inputs=selected,
        )
        assert "Electricity & 1.0 & 3.0 & 2.0 & 5.0 & 4.0 & 9.0" in latex
        assert len(selected) == 2

    print("timing table checks passed")


if __name__ == "__main__":
    main()
