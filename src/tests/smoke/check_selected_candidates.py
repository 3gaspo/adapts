"""Check the shared manual candidate manifest and every ablation front."""

from __future__ import annotations

import subprocess
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _selected(filter_name: str, candidate_file: str | None = None) -> list[str]:
    bash = shutil.which("bash")
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt" and git_bash.is_file():
        bash = str(git_bash)
    if bash is None:
        raise RuntimeError("bash is required for the selected candidate contract check")
    prefix = (
        f'SELECTED_CANDIDATES_FILE="$PROJECT_ROOT/{candidate_file}"; '
        if candidate_file
        else ""
    )
    result = subprocess.run(
        [
            bash,
            "-c",
            'PROJECT_ROOT="$(pwd)"; '
            + prefix
            + 'source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"; '
            + f"selected_candidates_csv {filter_name}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [item for item in result.stdout.split(",") if item]


def _selected_first(filter_name: str, candidate_file: str | None = None) -> str:
    bash = shutil.which("bash")
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt" and git_bash.is_file():
        bash = str(git_bash)
    if bash is None:
        raise RuntimeError("bash is required for the selected candidate contract check")
    prefix = (
        f'SELECTED_CANDIDATES_FILE="$PROJECT_ROOT/{candidate_file}"; '
        if candidate_file
        else ""
    )
    result = subprocess.run(
        [
            bash,
            "-c",
            'PROJECT_ROOT="$(pwd)"; '
            + prefix
            + 'source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"; '
            + f"selected_candidate_first {filter_name}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> None:
    expected = [
        "baselines/instance_euclidean_3_online/full_ridge_shared",
        "baselines/raw_euclidean_1_online/full_ridge_shared",
        "baselines/instance_euclidean_3_online/cov_y_ridge_shared",
        "baselines/raw_euclidean_3_online/full_ridge_shared",
        "baselines/instance_euclidean_3_online/cov_avgy_ridge_shared",
        "gates/raw_euclidean_1_online/catboost_cov_regressor_shared",
        "gates/instance_euclidean_1_online/catboost_avgy_regressor_shared",
        "gates/raw_euclidean_1_online/bayes_cov_shared",
        "gates/instance_euclidean_1_online/bayes_avgy_shared",
    ]
    assert _selected("adaptation") == expected
    assert _selected("baseline_shared") == expected[:5]
    assert _selected("catboost_shared") == expected[5:7]
    assert _selected_first("baseline_shared") == expected[0]
    assert _selected("ts_ifa") == []
    second_generation = ROOT / "SECOND_GENERATION_CANDIDATES.txt"
    assert second_generation.is_file()
    second_expected = [
        "baselines/instance_euclidean_5_online/full_ridge_shared",
        "baselines/instance_euclidean_20_online/cov_y_ridge_shared",
        "baselines/instance_euclidean_20_online/cov_avgy_ridge_shared",
        "baselines/raw_euclidean_10_online/full_ridge_shared",
        "gates/raw_euclidean_15_online/catboost_cov_regressor_shared",
        "gates/raw_euclidean_1_online/bayes_cov_shared",
        "gates/instance_euclidean_1_online/bayes_avgy_shared",
        "gates/instance_euclidean_5_online/catboost_avgy_regressor_shared",
    ]
    second_entries = [
        line
        for line in second_generation.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert second_entries == second_expected
    assert _selected("adaptation", second_generation.name) == second_expected
    assert _selected("baseline_shared", second_generation.name) == second_expected[:4]
    assert (
        _selected_first("baseline_shared", second_generation.name)
        == "baselines/instance_euclidean_5_online/full_ridge_shared"
    )
    assert _selected("ts_ifa", second_generation.name) == []

    filters = {
        "benchmark.slurm": ("WINNERS_CSV", "adaptation"),
        "mixed_quantity_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "fourier_retrieval_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "offline_datastore_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "retrieval_scope_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "k_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "h_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "l_ablation.slurm": ("WINNERS_CSV", "adaptation"),
        "horizon_baselines_ablation.slurm": (
            "BASELINE_WINNERS_CSV",
            "baseline_shared",
        ),
        "convex_baselines_ablation.slurm": (
            "BASELINE_WINNERS_CSV",
            "baseline_shared",
        ),
        "delta_baselines_ablation.slurm": (
            "BASELINE_WINNERS_CSV",
            "baseline_shared",
        ),
        "catboost_ablation.slurm": ("CATBOOST_WINNERS_CSV", "catboost_shared"),
        "ts_ifa_h_ablation.slurm": ("TS_IFA_CANDIDATES_CSV", "ts_ifa"),
        "ts_ifa_l_ablation.slurm": ("TS_IFA_CANDIDATES_CSV", "ts_ifa"),
        "ts_ifa_meta_ridge.slurm": ("TS_IFA_CANDIDATES_CSV", "ts_ifa_ridge"),
        "ts_ifa_meta_neural.slurm": ("TS_IFA_CANDIDATES_CSV", "ts_ifa_neural"),
    }
    for filename, (variable, filter_name) in filters.items():
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert 'source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"' in text
        assert (
            f'{variable}="${{{variable}:-$(selected_candidates_csv {filter_name})}}"'
            in text
        )

    for filename in ("sota_benchmark.slurm", "tsrag.slurm"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert 'source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"' in text
        assert (
            'WINNERS_CSV="${WINNERS_CSV:-$(selected_candidate_first baseline_shared)}"'
            in text
        )

    second_generation_fronts = {
        "benchmark.slurm",
        "sota_benchmark.slurm",
        "tsrag.slurm",
        "ts_ifa_h_ablation.slurm",
        "ts_ifa_l_ablation.slurm",
        "ts_ifa_meta_ridge.slurm",
        "ts_ifa_meta_neural.slurm",
    }
    for path in ROOT.glob("*.slurm"):
        text = path.read_text(encoding="utf-8")
        has_second_generation = "SECOND_GENERATION_CANDIDATES.txt" in text
        assert has_second_generation == (path.name in second_generation_fronts)

    print("selected candidate contract checks passed")


if __name__ == "__main__":
    main()
