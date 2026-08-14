"""Check that primary and ablation sweeps keep disjoint expensive methods."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROFILES = ROOT / "src" / "slurm" / "profiles.sh"


def csv_assignment(text: str, name: str) -> tuple[str, ...]:
    match = re.search(rf'^{name}="([^"]+)"$', text, flags=re.MULTILINE)
    assert match is not None, name
    return tuple(match.group(1).split(","))


def main() -> None:
    text = PROFILES.read_text(encoding="utf-8")
    primary_datasets = csv_assignment(text, "PRIMARY_DATASETS_CSV")
    full_datasets = csv_assignment(text, "FULL_DATASETS_CSV")
    mixed_datasets = csv_assignment(text, "MIXED_QUANTITY_DATASETS_CSV")
    primary_settings = csv_assignment(text, "PRIMARY_SETTINGS_CSV")
    primary_baselines = csv_assignment(text, "PRIMARY_BASELINE_METHODS_CSV")
    primary_gates = csv_assignment(text, "PRIMARY_GATE_METHODS_CSV")
    neighbor_defaults = re.findall(r'^\s*DEFAULT_NEIGHBORS_CSV="([^"]*)"$', text, flags=re.MULTILINE)

    assert primary_datasets == ("Electricity", "Traffic", "Solar", "exchange_rate")
    assert full_datasets == (
        "Electricity",
        "Traffic",
        "Solar",
        "exchange_rate",
        "ETT_T_1H",
        "ETT_L_1H",
        "ETT_T_15T",
        "ETT_L_15T",
    )
    assert mixed_datasets == ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
    assert primary_settings == ("168:24", "336:48", "504:168")
    assert set(primary_datasets) < set(full_datasets)
    assert set(full_datasets).isdisjoint(mixed_datasets)

    assert primary_baselines
    assert len(primary_baselines) == 10
    assert len(primary_gates) == 8
    assert all(not method.endswith("_horizon") for method in primary_baselines)
    assert all("_convex_" not in method for method in primary_baselines)
    assert all("_delta_ridge_" not in method for method in primary_baselines)

    primary_catboost = {
        method for method in primary_gates if method.startswith("catboost_")
    }
    assert primary_catboost == {
        "catboost_cov_regressor_shared",
        "catboost_avgy_regressor_shared",
    }

    assert set(neighbor_defaults) == {
        "",
        "3",
        "1,3",
        "1,3,5,10,15,20,100",
        "10",
        "15",
    }
    assert neighbor_defaults.count("1,3") == 8
    assert neighbor_defaults.count("3") == 3
    assert 'DEFAULT_SETTINGS_CSV="504:168"' in text
    assert text.count('DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"') == 7
    assert "DEFAULT_DATASTORE_STRIDE=25" in text
    assert "DEFAULT_ADAPT_QUERY_STRIDE=25" in text
    assert "DEFAULT_EVAL_QUERY_STRIDE=127" in text
    assert "DEFAULT_ALIGN_PERIOD=false" in text
    assert "DEFAULT_ALIGN_PERIOD=true" in text
    assert 'permits only K=3' in text

    ts_ifa_fronts = (
        "ts_ifa.slurm",
        "ts_ifa_h_ablation.slurm",
        "ts_ifa_l_ablation.slurm",
        "ts_ifa_meta_ridge.slurm",
        "ts_ifa_meta_neural.slurm",
    )
    for scale_front in ("extraction.slurm", "tables.slurm"):
        front_text = (ROOT / scale_front).read_text(encoding="utf-8")
        assert "require_scale_experiment_mode || exit $?" in front_text
    for front in ts_ifa_fronts:
        front_text = (ROOT / front).read_text(encoding="utf-8")
        assert 'source "$PROJECT_ROOT/src/slurm/run_ts_ifa.sh"' in front_text
    assert "TS_IFA_GRID=complete" in (ROOT / "ts_ifa.slurm").read_text(encoding="utf-8")
    assert "TS_IFA_META_FORM=ridge" in (ROOT / "ts_ifa_meta_ridge.slurm").read_text(encoding="utf-8")
    assert "TS_IFA_META_FORM=neural" in (ROOT / "ts_ifa_meta_neural.slurm").read_text(encoding="utf-8")
    for test_front in ("baselines.slurm", "gates.slurm"):
        front_text = (ROOT / test_front).read_text(encoding="utf-8")
        assert "require_test_experiment_mode || exit $?" in front_text
    assert not (ROOT / "run_all.sh").exists()
    assert not (ROOT / "univariate.slurm").exists()
    assert not (ROOT / "ts_ifa_joint_ridge.slurm").exists()
    assert not (ROOT / "ts_ifa_joint_neural.slurm").exists()
    assert not (ROOT / "src" / "slurm" / "run_univariate.sh").exists()

    profile_runner = (ROOT / "src" / "slurm" / "run_profile_experiment.sh").read_text(
        encoding="utf-8"
    )
    extraction_runner = (ROOT / "src" / "slurm" / "extract_adaptation.sh").read_text(
        encoding="utf-8"
    )
    assert 'ALIGN_PERIOD="${ALIGN_PERIOD:-$DEFAULT_ALIGN_PERIOD}"' in extraction_runner
    assert '[ "$ALIGN_PERIOD" = true ] || alignment_args+=(--no-align-period)' in extraction_runner
    assert '"data.align_period=$ALIGN_PERIOD"' in extraction_runner
    assert "SELECTED_NEIGHBORS=()" in profile_runner
    assert 'append_unique SELECTED_NEIGHBORS "$group_neighbors"' in profile_runner
    assert 'append_unique SELECTED_NEIGHBORS "$k"' in profile_runner
    assert 'NEIGHBORS_CSV="$(join_csv_values "${SELECTED_NEIGHBORS[@]}")"' in profile_runner
    assert 'PIPELINES+=("$family/$run/$method")' in profile_runner
    assert 'PIPELINES+=("$family/${space}_${metric}_${k}_${retrieval}/$method")' in profile_runner
    assert 'run_groups chronos2' in profile_runner
    assert re.search(
        r'run_method_group \\\s+chronos-bolt "\$CANDIDATE_FAMILY"', profile_runner
    )
    assert 'CHRONOS2_PIPELINES_CSV="$CHRONOS2_PIPELINE"' in profile_runner
    assert 'CHRONOS_BOLT_PIPELINES_CSV="$CHRONOS_BOLT_CANDIDATE_PIPELINE,crossrag/' in profile_runner

    common = (ROOT / "src" / "slurm" / "common.sh").read_text(encoding="utf-8")
    assert "copy_if_needed()" in common
    assert 'mktemp "$destination_dir/.${destination##*/}.XXXXXX"' in common
    assert "small" not in re.search(
        r"require_experiment_mode\(\) \{.*?\n\}", common, flags=re.DOTALL
    ).group()
    benchmark_front = (ROOT / "benchmark.slurm").read_text(encoding="utf-8")
    assert 'EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"' in benchmark_front
    assert 'RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/benchmark}"' in benchmark_front
    assert 'WINNERS_CSV="${WINNERS_CSV:-}"' in benchmark_front
    assert "require_benchmark_experiment_mode || exit $?" in benchmark_front
    assert 'source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"' in benchmark_front
    screen_front = (ROOT / "screen.slurm").read_text(encoding="utf-8")
    assert 'EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"' in screen_front
    assert "EXPERIMENT_FAMILY=screen" in screen_front
    assert "MODELS_CSV=chronos2" in screen_front

    for front, implementation in (
        ("mixed_quantity_ablation.slurm", "run_profile_experiment.sh"),
        ("fourier_retrieval_ablation.slurm", "run_profile_experiment.sh"),
        ("offline_datastore_ablation.slurm", "run_profile_experiment.sh"),
        ("horizon_baselines_ablation.slurm", "run_baseline_family_ablation.sh"),
        ("convex_baselines_ablation.slurm", "run_baseline_family_ablation.sh"),
        ("delta_baselines_ablation.slurm", "run_baseline_family_ablation.sh"),
        ("catboost_ablation.slurm", "run_catboost_ablation.sh"),
    ):
        front_text = (ROOT / front).read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"' in front_text
        assert f'source "$PROJECT_ROOT/src/slurm/{implementation}"' in front_text

    mixed_front = (ROOT / "mixed_quantity_ablation.slurm").read_text(encoding="utf-8")
    assert (
        'WINNERS_CSV="${WINNERS_CSV:-$(selected_candidates_csv adaptation)}"'
        in mixed_front
    )
    fourier_front = (ROOT / "fourier_retrieval_ablation.slurm").read_text(
        encoding="utf-8"
    )
    assert "EXPERIMENT_FAMILY=fourier_retrieval_ablation" in fourier_front
    assert "selected_candidates_csv adaptation" in fourier_front
    offline_front = (ROOT / "offline_datastore_ablation.slurm").read_text(
        encoding="utf-8"
    )
    assert "EXPERIMENT_FAMILY=offline_datastore_ablation" in offline_front
    assert "selected_candidates_csv adaptation" in offline_front
    catboost_front = (ROOT / "catboost_ablation.slurm").read_text(encoding="utf-8")
    assert "selected_candidates_csv catboost_shared" in catboost_front
    for selected_front in (
        "k_ablation.slurm",
        "h_ablation.slurm",
        "l_ablation.slurm",
    ):
        assert 'WINNERS_CSV="${WINNERS_CSV:-$(selected_candidates_csv adaptation)}"' in (
            ROOT / selected_front
        ).read_text(encoding="utf-8")
    sota_front = (ROOT / "sota_benchmark.slurm").read_text(encoding="utf-8")
    assert (
        'WINNERS_CSV="${WINNERS_CSV:-baselines/instance_euclidean_10_online/'
        'full_ridge_shared}"'
    ) in sota_front
    assert 'source "$PROJECT_ROOT/src/slurm/run_sota_benchmark.sh"' in sota_front
    assert 'DEFAULT_MODELS_CSV="chronos-bolt"' in text
    tsrag_front = (ROOT / "tsrag.slurm").read_text(encoding="utf-8")
    assert 'source "$PROJECT_ROOT/src/slurm/run_tsrag_experiment.sh"' in tsrag_front
    for family_front in (
        "horizon_baselines_ablation.slurm",
        "convex_baselines_ablation.slurm",
        "delta_baselines_ablation.slurm",
    ):
        front_text = (ROOT / family_front).read_text(encoding="utf-8")
        assert "BASELINE_WINNERS_CSV" in front_text
        assert "selected_candidates_csv baseline_shared" in front_text
        assert "euclidean_10" not in front_text

    profile_runner = (ROOT / "src" / "slurm" / "run_profile_experiment.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [ "$EXPERIMENT_FAMILY" = k_ablation ]; then' in profile_runner
    assert 'elif [ "$EXPERIMENT_FAMILY" = fourier_retrieval_ablation ]; then' in profile_runner
    assert 'append_unique SPACES fourier' in profile_runner
    assert '"$family/fourier_${metric}_${neighbors}_${retrieval}/$method"' in profile_runner
    assert 'elif [ "$EXPERIMENT_FAMILY" = offline_datastore_ablation ]; then' in profile_runner
    assert '"$family/${space}_${metric}_${neighbors}_fixed/$method"' in profile_runner
    assert 'outside the primary K={1,3} policy' in profile_runner
    assert 'EXTRACTION_SKIP_COMPLETE="${EXTRACTION_SKIP_COMPLETE:-true}"' in profile_runner
    assert 'RESULT_SKIP_COMPLETE="${SKIP_COMPLETE:-true}"' in profile_runner
    assert 'SKIP_COMPLETE="$RESULT_SKIP_COMPLETE"' in profile_runner
    assert "SKIP_COMPLETE=false" not in profile_runner
    assert '[ "$existing" != "$value" ] || return 0' in profile_runner

    catboost_runner = (
        ROOT / "src" / "slurm" / "run_catboost_ablation.sh"
    ).read_text(encoding="utf-8")
    assert "CATBOOST_WINNERS_CSV" in catboost_runner
    for variant in (
        "regressor_shared",
        "regressor_shared_soft",
        "regressor_horizon",
        "classifier_shared",
    ):
        assert f'catboost_${{candidate}}_{variant}' in catboost_runner
    for excluded_variant in (
        "regressor_horizon_soft",
        "classifier_shared_soft",
        "classifier_horizon",
        "classifier_horizon_soft",
    ):
        assert f'catboost_${{candidate}}_{excluded_variant}' not in catboost_runner

    baseline_runner = (ROOT / "src" / "slurm" / "run_baselines.sh").read_text(
        encoding="utf-8"
    )
    gate_runner = (ROOT / "src" / "slurm" / "run_gates.sh").read_text(
        encoding="utf-8"
    )
    assert gate_runner.count("DEFAULT_GATE_ITERATIONS=2") == 1
    assert re.search(r"test:\*\)\s+DEFAULT_GATE_ITERATIONS=2", gate_runner)
    assert re.search(
        r"\*:k_ablation\|\*:h_ablation\|\*:l_ablation\|\*:crossrag\)\s+"
        r"DEFAULT_GATE_ITERATIONS=300",
        gate_runner,
    )
    ts_ifa_runner = (ROOT / "src" / "slurm" / "run_ts_ifa.sh").read_text(
        encoding="utf-8"
    )
    crossrag_runner = (ROOT / "src" / "slurm" / "run_crossrag.sh").read_text(
        encoding="utf-8"
    )
    table_runner = (ROOT / "src" / "slurm" / "build_tables.sh").read_text(
        encoding="utf-8"
    )
    for runner in (baseline_runner, gate_runner, crossrag_runner):
        assert 'OUT_ROOT="${OUT_ROOT:-outputs/extraction}"' in runner
        assert "copy_if_needed" not in runner
    assert "find_weight_path chronos-bolt-base" in crossrag_runner
    assert "find_weight_path cross-rag" in crossrag_runner
    assert '--chronos-bolt-weights "$CHRONOS_BOLT_WEIGHTS_PATH"' in crossrag_runner
    assert '--cross-rag-weights "$CROSSRAG_WEIGHTS_PATH"' in crossrag_runner
    assert "CROSSRAG_ROOT" not in crossrag_runner
    assert 'OUT_ROOT="${OUT_ROOT:-outputs/extraction}"' in ts_ifa_runner
    assert "copy_if_needed" not in ts_ifa_runner
    assert "TS_IFA_GRID" in ts_ifa_runner
    assert "TRAIN_EPOCHS=\"${TRAIN_EPOCHS:-20000}\"" in ts_ifa_runner
    assert 'OUT_ROOT="${OUT_ROOT:-outputs/extraction}"' in table_runner
    assert ': "${OUT_ROOT:=outputs/extraction}"' in (
        ROOT / "src" / "slurm" / "extract_adaptation.sh"
    ).read_text(encoding="utf-8")
    assert 'test:*|*:screen|*:baselines) DEFAULT_BASELINE_METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV"' in baseline_runner
    assert 'test:*|*:screen|*:gates) DEFAULT_GATE_METHODS_CSV="$PRIMARY_GATE_METHODS_CSV"' in gate_runner
    assert 'PROFILE_METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV,$PRIMARY_GATE_METHODS_CSV"' in table_runner
    assert 'test:*|*:screen)' in table_runner
    assert "src.visu.baseline_coefficients" in table_runner
    assert "coefficient_index.csv" in table_runner
    assert 'if [ "$EXPERIMENT_FAMILY" = k_ablation ]; then' in table_runner
    assert "src.visu.k_ablation_plot" in table_runner
    assert "k_ablation_average_${METRIC}_improvement" in table_runner
    assert "ts_ifa/TS-IFA" not in table_runner
    assert "validates every selected dataset/setting/model/" in table_runner
    assert 'MODEL_PIPELINES_CSV="$CHRONOS2_PIPELINES_CSV"' in table_runner
    assert 'chronos-bolt) MODEL_PIPELINES_CSV="$CHRONOS_BOLT_PIPELINES_CSV"' in table_runner
    assert 'MODEL_FAMILY_ARG="$CANDIDATE_FAMILY"' in table_runner
    assert '--candidate-formula "$CANDIDATE_METHOD"' in table_runner
    crossrag_evaluator = (
        ROOT / "src" / "adaptors" / "cross_rag" / "evaluate.py"
    ).read_text(encoding="utf-8")
    assert '"format": "adaptation_crossrag_result"' in crossrag_evaluator
    assert '"positive_window_pct"' in crossrag_evaluator
    assert "from src.models.cross_rag import" in crossrag_evaluator
    assert "importlib" not in crossrag_evaluator

    family_runner = (
        ROOT / "src" / "slurm" / "run_baseline_family_ablation.sh"
    ).read_text(encoding="utf-8")
    for shared in (
        "cov_ridge_shared",
        "avgy_ridge_shared",
        "y_ridge_shared",
        "cov_y_ridge_shared",
        "cov_avgy_ridge_shared",
        "residual_ridge_shared",
        "full_ridge_shared",
    ):
        assert f"{shared})" in family_runner
    assert 'variant="${design}_ridge_horizon"' in family_runner
    assert 'variant="${design}_convex_shared"' in family_runner
    assert 'variant="${design}_delta_ridge_shared"' in family_runner
    assert "positive_window_pct" in common
    assert 'python -m experiment_runs prepare --run-dir "$1"' in common
    baseline_evaluator = (
        ROOT / "src" / "adaptors" / "baselines" / "evaluate.py"
    ).read_text(encoding="utf-8")
    ts_ifa_trainer = (
        ROOT / "src" / "adaptors" / "ts_ifa" / "train.py"
    ).read_text(encoding="utf-8")
    for run_writer in (baseline_evaluator, ts_ifa_trainer):
        assert "prepare_run_output(output_dir)" in run_writer
        assert "shutil.rmtree(output_dir)" not in run_writer


if __name__ == "__main__":
    main()
