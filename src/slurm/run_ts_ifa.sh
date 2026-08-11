#!/bin/bash
# Train a complete or textually selected TS-IFA configuration grid.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-outputs/extraction}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-ts_ifa}"
require_experiment_mode
require_experiment_family
adaptation_profile_defaults
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/$EXPERIMENT_FAMILY}"
TS_IFA_GRID="${TS_IFA_GRID:-selected}"
TS_IFA_CANDIDATES_CSV="${TS_IFA_CANDIDATES_CSV:-}"
TS_IFA_META_FORM="${TS_IFA_META_FORM:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20000}"
ROOTER_EPOCHS="${ROOTER_EPOCHS:-20000}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES-}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES-}"
SKIP_COMPLETE="${SKIP_COMPLETE:-true}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BRANCH_LR="${BRANCH_LR:-0.00001}"
ROOTER_LR="${ROOTER_LR:-0.00001}"
NEURAL_INNER_LR="${NEURAL_INNER_LR:-0.001}"
NEURAL_INNER_STEPS="${NEURAL_INNER_STEPS:-1}"
NEURAL_FIRST_ORDER="${NEURAL_FIRST_ORDER:-true}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
VANILLA_ANCHOR="${VANILLA_ANCHOR:-0.01}"
COEFFICIENT_L2="${COEFFICIENT_L2:-0.01}"
HORIZON_SMOOTHNESS="${HORIZON_SMOOTHNESS:-0.01}"
RIDGE_ROOTER_ALPHA="${RIDGE_ROOTER_ALPHA:-0.01}"
BRANCH_AUX_WEIGHT="${BRANCH_AUX_WEIGHT:-0.1}"
DROPOUT="${DROPOUT:-0.0}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
ATTENTION_DIM="${ATTENTION_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
VANILLA_ANCHORING_INIT="${VANILLA_ANCHORING_INIT:-true}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
META_QUERY_FRACTION="${META_QUERY_FRACTION:-0.2}"

if [ "$TRAIN_EPOCHS" != 20000 ] || [ "$ROOTER_EPOCHS" != 20000 ]; then
  log_error "the current TS-IFA contract fixes every training stage at 20000 steps"
  return 2
fi
require_resolved_profile_grid
require_profile_neighbors "$NEIGHBORS_CSV"
csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$DISTANCE_METRICS_CSV" DISTANCE_METRICS
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

branch_csv() {
  case "$1" in
    cov|residual|memory) printf '%s\n' "$1" ;;
    full) printf 'cov,residual,memory\n' ;;
    *) printf '%s\n' "${1//+/,}" ;;
  esac
}

parse_method() {
  local method="$1" tail
  case "$method" in
    joint_ridge_*) PARSED_VARIANT=joint_ridge; tail="${method#joint_ridge_}" ;;
    joint_neural_*) PARSED_VARIANT=joint_neural; tail="${method#joint_neural_}" ;;
    meta_ridge_*) PARSED_VARIANT=meta_ridge; tail="${method#meta_ridge_}" ;;
    meta_neural_*) PARSED_VARIANT=meta_neural; tail="${method#meta_neural_}" ;;
    *) log_error "invalid TS-IFA method=$method"; return 2 ;;
  esac
  IFS='_' read -r PARSED_SCOPE PARSED_CONSTRAINT PARSED_BRANCH_SET PARSED_EXTRA <<< "$tail"
  case "$PARSED_SCOPE" in shared|horizon) ;; *) log_error "invalid routing scope in $method"; return 2 ;; esac
  case "$PARSED_CONSTRAINT" in unconstrained|softmax) ;; *) log_error "invalid routing constraint in $method"; return 2 ;; esac
  if [ -z "$PARSED_BRANCH_SET" ] || [ -n "${PARSED_EXTRA:-}" ]; then
    log_error "invalid branch set in TS-IFA method=$method"
    return 2
  fi
}

METHOD_SPECS=()
PIPELINE_SPECS=()
if [ "$TS_IFA_GRID" = complete ]; then
  for variant in joint_ridge joint_neural; do
    for scope in shared horizon; do
      for constraint in unconstrained softmax; do
        for branch_set in cov residual memory full; do
          METHOD_SPECS+=("$variant|$scope|$constraint|$branch_set")
        done
      done
    done
  done
elif [ "$TS_IFA_GRID" = selected ]; then
  [ -n "$TS_IFA_CANDIDATES_CSV" ] || {
    log_error "set TS_IFA_CANDIDATES_CSV to ts_ifa/retrieval_run/method entries"
    return 2
  }
  selected=()
  csv_to_array "$TS_IFA_CANDIDATES_CSV" selected
  for entry in "${selected[@]}"; do
    IFS='/' read -r family run method extra <<< "$entry"
    if [ "$family" != ts_ifa ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
      log_error "invalid TS-IFA candidate=$entry expected=ts_ifa/retrieval_run/method"
      return 2
    fi
    parse_method "$method"
    if [ -n "$TS_IFA_META_FORM" ]; then
      case "$TS_IFA_META_FORM|$PARSED_VARIANT" in
        ridge|joint_ridge) PARSED_VARIANT=meta_ridge ;;
        neural|joint_neural) PARSED_VARIANT=meta_neural ;;
        *) log_error "candidate $entry does not match meta form=$TS_IFA_META_FORM"; return 2 ;;
      esac
    fi
    PIPELINE_SPECS+=("$run|$PARSED_VARIANT|$PARSED_SCOPE|$PARSED_CONSTRAINT|$PARSED_BRANCH_SET")
  done
else
  log_error "TS_IFA_GRID must be complete or selected"
  return 2
fi

TASKS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      if [ "$TS_IFA_GRID" = complete ]; then
        for space in "${DISTANCE_SPACES[@]}"; do
          for metric in "${DISTANCE_METRICS[@]}"; do
            for neighbors in "${NEIGHBORS[@]}"; do
              run="${space}_${metric}_${neighbors}_${RETRIEVAL_MODE}"
              for spec in "${METHOD_SPECS[@]}"; do TASKS+=("$dataset|$model|$setting|$run|$spec"); done
            done
          done
        done
      else
        for spec in "${PIPELINE_SPECS[@]}"; do TASKS+=("$dataset|$model|$setting|$spec"); done
      fi
    done
  done
done

ts_ifa_complete() {
  local output="$1" expected_signature="$2" expected_method="$3"
  [ -s "$output/ts_ifa.pt" ] && [ -s "$output/branches.pt" ] &&
    [ -s "$output/rooter.pt" ] && [ -s "$output/training_history.json" ] &&
    [ -s "$output/eval_metrics.json" ] && [ -s "$output/prediction_manifest.json" ] &&
    [ -s "$output/config.json" ] && [ -s "$output/training_nmse.pdf" ] &&
    [ -s "$output/result_manifest.json" ] &&
    grep -Fq '"architecture": "configurable_delta_branches_routing_v4"' "$output/result_manifest.json" &&
    grep -Fq "\"method\": \"$expected_method\"" "$output/result_manifest.json" &&
    grep -Fq "\"run_signature\": \"$expected_signature\"" "$output/result_manifest.json"
}

for task_id in "${!TASKS[@]}"; do
  IFS='|' read -r dataset model setting run variant scope constraint branch_set <<< "${TASKS[$task_id]}"
  parse_setting "$setting"
  L="$SETTING_LAGS"; H="$SETTING_HORIZON"
  IFS='_' read -r space metric neighbors retrieval_mode run_extra <<< "$run"
  if [ -n "${run_extra:-}" ]; then log_error "invalid retrieval run=$run"; return 2; fi
  resolve_extraction_run "$dataset" "$L" "$H" "$model" "$space" "$metric" "$neighbors" "$retrieval_mode"
  INPUT_DIR="$EXTRACTION_RUN_DIR"
  require_extraction "$INPUT_DIR"
  method="${variant}_${scope}_${constraint}_${branch_set}"
  identity_root="$RESULTS_ROOT/$dataset/${L}_${H}/${model,,}/${variant,,}/${scope,,}/${constraint,,}/${branch_set,,}/${space,,}/${metric,,}/$neighbors/${retrieval_mode,,}"
  branches="$(branch_csv "$branch_set")"
  signature="architecture=configurable_delta_branches_routing_v4;variant=$variant;routing_scope=$scope;routing_constraint=$constraint;branches=$branches;train_epochs=20000;rooter_epochs=20000;batch_size=$BATCH_SIZE;validation_fraction=$VALIDATION_FRACTION;meta_query_fraction=$META_QUERY_FRACTION;valid_eval_freq=$VALID_EVAL_FREQ;logging_eval_freq=$LOGGING_EVAL_FREQ;max_train_samples=$MAX_TRAIN_SAMPLES;max_valid_samples=$MAX_VALID_SAMPLES;branch_lr=$BRANCH_LR;rooter_lr=$ROOTER_LR;neural_inner_lr=$NEURAL_INNER_LR;neural_inner_steps=$NEURAL_INNER_STEPS;neural_first_order=$NEURAL_FIRST_ORDER;weight_decay=$WEIGHT_DECAY;vanilla_anchor=$VANILLA_ANCHOR;coefficient_l2=$COEFFICIENT_L2;horizon_smoothness=$HORIZON_SMOOTHNESS;ridge_alpha=$RIDGE_ROOTER_ALPHA;branch_aux_weight=$BRANCH_AUX_WEIGHT;dropout=$DROPOUT;heads=$ATTENTION_HEADS;attention_dim=$ATTENTION_DIM;hidden_dim=$HIDDEN_DIM;vanilla_anchoring_init=$VANILLA_ANCHORING_INIT;normalization=instance;seed=$SEED"
  model_values=(
    "variant=$variant" "routing_scope=$scope" "routing_constraint=$constraint" "branch_set=$branch_set"
    "space=$space" "metric=$metric" "k=$neighbors" "mode=$retrieval_mode"
  )
  pipeline_values=(
    "train_epochs=$TRAIN_EPOCHS" "rooter_epochs=$ROOTER_EPOCHS" "batch_size=$BATCH_SIZE"
    "validation_fraction=$VALIDATION_FRACTION" "meta_query_fraction=$META_QUERY_FRACTION"
    "valid_eval_freq=$VALID_EVAL_FREQ" "logging_eval_freq=$LOGGING_EVAL_FREQ"
    "max_train_samples=${MAX_TRAIN_SAMPLES:-none}" "max_valid_samples=${MAX_VALID_SAMPLES:-none}"
    "branch_lr=$BRANCH_LR" "rooter_lr=$ROOTER_LR" "neural_inner_lr=$NEURAL_INNER_LR"
    "neural_inner_steps=$NEURAL_INNER_STEPS" "neural_first_order=$NEURAL_FIRST_ORDER"
    "weight_decay=$WEIGHT_DECAY" "vanilla_anchor=$VANILLA_ANCHOR" "coefficient_l2=$COEFFICIENT_L2"
    "horizon_smoothness=$HORIZON_SMOOTHNESS" "ridge_rooter_alpha=$RIDGE_ROOTER_ALPHA"
    "branch_aux_weight=$BRANCH_AUX_WEIGHT" "dropout=$DROPOUT" "attention_heads=$ATTENTION_HEADS"
    "attention_dim=$ATTENTION_DIM" "hidden_dim=$HIDDEN_DIM" "vanilla_anchoring_init=$VANILLA_ANCHORING_INIT"
  )
  allocate_manifest_run "$identity_root" "adaptation/$EXPERIMENT_FAMILY/ts_ifa" "$dataset" "$L" "$H" "$model" \
    variant,routing_scope,routing_constraint,branch_set,space,metric,k,mode "$method" \
    variant,routing_scope routing_constraint,branch_set,space,metric,k,mode \
    model_values pipeline_values "$SEED" "$INPUT_DIR/manifest.json"
  OUTPUT_DIR="$ALLOCATED_RUN_DIR"
  if [ "$ALLOCATED_ACTION" = skip ]; then
    log "skip complete family=ts_ifa method=$method dataset=$dataset setting=$setting retrieval=$run run=$OUTPUT_DIR"
    continue
  fi
  mark_manifest_running "$OUTPUT_DIR"
  optional_args=()
  [ -z "$MAX_TRAIN_SAMPLES" ] || optional_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  [ -z "$MAX_VALID_SAMPLES" ] || optional_args+=(--max-valid-samples "$MAX_VALID_SAMPLES")
  if is_true "$VANILLA_ANCHORING_INIT"; then optional_args+=(--vanilla-anchoring-init); else optional_args+=(--no-vanilla-anchoring-init); fi
  if is_true "$NEURAL_FIRST_ORDER"; then optional_args+=(--neural-first-order); else optional_args+=(--no-neural-first-order); fi
  log_section "training start configuration=$((task_id + 1))/${#TASKS[@]} method=$method dataset=$dataset setting=$setting retrieval=$run"
  srun --ntasks=1 python -m src.adaptors.ts_ifa.train \
    --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --method-id "$method" \
    --variant "$variant" --routing-scope "$scope" --routing-constraint "$constraint" \
    --branches "$branches" --run-signature "$signature" \
    --train-epochs 20000 --rooter-epochs 20000 --batch-size "$BATCH_SIZE" \
    --validation-fraction "$VALIDATION_FRACTION" --meta-query-fraction "$META_QUERY_FRACTION" \
    --valid-eval-freq "$VALID_EVAL_FREQ" --logging-eval-freq "$LOGGING_EVAL_FREQ" \
    --branch-lr "$BRANCH_LR" --rooter-lr "$ROOTER_LR" \
    --neural-inner-lr "$NEURAL_INNER_LR" --neural-inner-steps "$NEURAL_INNER_STEPS" \
    --weight-decay "$WEIGHT_DECAY" --vanilla-anchor "$VANILLA_ANCHOR" \
    --coefficient-l2 "$COEFFICIENT_L2" --horizon-smoothness "$HORIZON_SMOOTHNESS" \
    --ridge-rooter-alpha "$RIDGE_ROOTER_ALPHA" --branch-aux-weight "$BRANCH_AUX_WEIGHT" \
    --dropout "$DROPOUT" --residual-heads "$ATTENTION_HEADS" --memory-heads "$ATTENTION_HEADS" \
    --rooter-heads "$ATTENTION_HEADS" --residual-attn-dim "$ATTENTION_DIM" \
    --memory-attn-dim "$ATTENTION_DIM" --rooter-attn-dim "$ATTENTION_DIM" \
    --residual-hidden "$HIDDEN_DIM" --memory-hidden "$HIDDEN_DIM" --rooter-hidden "$HIDDEN_DIM" \
    --normalization instance --device gpu --seed "$SEED" "${optional_args[@]}"
  assert_files ts-ifa-output "$OUTPUT_DIR/ts_ifa.pt" "$OUTPUT_DIR/branches.pt" \
    "$OUTPUT_DIR/rooter.pt" "$OUTPUT_DIR/training_history.json" "$OUTPUT_DIR/eval_metrics.json" \
    "$OUTPUT_DIR/prediction_manifest.json" "$OUTPUT_DIR/config.json" \
    "$OUTPUT_DIR/training_nmse.pdf" "$OUTPUT_DIR/result_manifest.json"
  mark_manifest_ready "$OUTPUT_DIR" ts_ifa.pt branches.pt rooter.pt training_history.json eval_metrics.json prediction_manifest.json config.json training_nmse.pdf result_manifest.json
done
log_section "job done kind=ts_ifa_training family=$EXPERIMENT_FAMILY experiment_mode=$EXPERIMENT_MODE tasks=${#TASKS[@]} output=$RESULTS_ROOT"
