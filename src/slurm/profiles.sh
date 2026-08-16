#!/bin/bash
# Shared experiment grids. Source after common.sh, then call adaptation_profile_defaults.

PRIMARY_DATASETS_CSV="Electricity,Traffic,Solar,exchange_rate"
FULL_DATASETS_CSV="Electricity,Traffic,Solar,exchange_rate,ETT_T_1H,ETT_L_1H,ETT_T_15T,ETT_L_15T"
MIXED_QUANTITY_DATASETS_CSV="ETTh1,ETTh2,ETTm1,ETTm2,Weather"
PRIMARY_SETTINGS_CSV="168:24,336:48,504:168"

dataset_period() {
  case "$1" in
    ETTm1|ETTm2|ETT_T_15T|ETT_L_15T) printf '96\n' ;;
    *) printf '24\n' ;;
  esac
}

# The primary screen compares variable sets in the most flexible shared-ridge
# family. Convex, delta-ridge, and per-horizon forms are candidate-only ablations.
PRIMARY_BASELINE_METHODS_CSV="cov_forecast,avgy,y_mean,cov_ridge_shared,avgy_ridge_shared,y_ridge_shared,cov_y_ridge_shared,cov_avgy_ridge_shared,residual_ridge_shared,full_ridge_shared"

# The primary learned-gate comparison is the signed-advantage shared regressor.
# Cheap shared no-feature and oracle references remain beside it.
PRIMARY_GATE_METHODS_CSV="cov_forecast,avgy,bayes_cov_shared,bayes_avgy_shared,catboost_cov_regressor_shared,catboost_avgy_regressor_shared,oracle_cov_shared,oracle_avgy_shared"

adaptation_profile_defaults() {
  local mode="${EXPERIMENT_MODE:-test}"
  local profile="${EXPERIMENT_FAMILY:-}"
  case "$profile" in
    ""|extraction|baselines|gates|benchmark|tables|ts_ifa) profile="$mode" ;;
  esac
  DEFAULT_DATASETS_CSV=""
  DEFAULT_MODELS_CSV=""
  DEFAULT_SETTINGS_CSV=""
  DEFAULT_DISTANCE_SPACES_CSV=""
  DEFAULT_DISTANCE_METRICS_CSV="euclidean"
  DEFAULT_NEIGHBORS_CSV=""
  DEFAULT_DATASTORE_STRIDE=25
  DEFAULT_ADAPT_QUERY_STRIDE=25
  DEFAULT_EVAL_QUERY_STRIDE=127
  DEFAULT_ALIGN_PERIOD=false
  DEFAULT_MAX_STORE_WINDOWS=30000

  case "$profile" in
    test)
      DEFAULT_DATASETS_CSV="Electricity"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="504:168"
      DEFAULT_DISTANCE_SPACES_CSV="raw"
      DEFAULT_NEIGHBORS_CSV="3"
      DEFAULT_DATASTORE_STRIDE=168
      DEFAULT_ADAPT_QUERY_STRIDE=256
      DEFAULT_EVAL_QUERY_STRIDE=256
      DEFAULT_ALIGN_PERIOD=true
      DEFAULT_MAX_STORE_WINDOWS=2048
      ;;
    screen|horizon_baselines_ablation|convex_baselines_ablation|delta_baselines_ablation|catboost_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    fourier_retrieval_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance,fourier"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    offline_datastore_ablation|retrieval_scope_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    mixed_quantity_ablation)
      DEFAULT_DATASETS_CSV="$MIXED_QUANTITY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    k_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3,5,10,15,20"
      ;;
    h_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="504:24,504:168,504:504"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    l_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="24:24,168:24,504:24"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    ts_ifa_h_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="504:24,504:168,504:504"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="3"
      ;;
    ts_ifa_l_ablation)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="24:24,168:24,504:24"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="3"
      ;;
    crossrag)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2,chronos-bolt"
      DEFAULT_SETTINGS_CSV="512:64"
      DEFAULT_DISTANCE_SPACES_CSV="minmax"
      DEFAULT_DISTANCE_METRICS_CSV="cosine"
      DEFAULT_NEIGHBORS_CSV="15"
      ;;
    sota_benchmark)
      DEFAULT_DATASETS_CSV="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,exchange_rate"
      DEFAULT_MODELS_CSV="chronos-bolt"
      DEFAULT_SETTINGS_CSV="512:64"
      DEFAULT_DISTANCE_SPACES_CSV="instance"
      DEFAULT_DISTANCE_METRICS_CSV="euclidean"
      DEFAULT_NEIGHBORS_CSV="10"
      ;;
    tsrag)
      DEFAULT_DATASETS_CSV="$PRIMARY_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos-bolt"
      DEFAULT_SETTINGS_CSV="512:64"
      DEFAULT_DISTANCE_SPACES_CSV="tsrag"
      DEFAULT_DISTANCE_METRICS_CSV="euclidean"
      DEFAULT_NEIGHBORS_CSV="10"
      ;;
    full)
      DEFAULT_DATASETS_CSV="$FULL_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    ultra)
      DEFAULT_DATASETS_CSV="$FULL_DATASETS_CSV"
      DEFAULT_MODELS_CSV="chronos2,tabpfnts"
      DEFAULT_SETTINGS_CSV="$PRIMARY_SETTINGS_CSV"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3"
      ;;
    *)
      log_error "no defaults for experiment family=$profile mode=$mode"
      return 2
      ;;
  esac
}

require_resolved_profile_grid() {
  if [ -z "${DISTANCE_SPACES_CSV:-}" ]; then
    log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE has no distance-space grid"
    return 2
  fi
  if [ -z "${NEIGHBORS_CSV:-}" ]; then
    log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE has no neighbor-count grid"
    return 2
  fi
}

require_profile_neighbors() {
  local raw="$1" value
  local requested_neighbors=()
  case "${EXPERIMENT_FAMILY:-}" in
    k_ablation|crossrag|sota_benchmark|tsrag) return 0 ;;
  esac
  csv_to_array "$raw" requested_neighbors
  for value in "${requested_neighbors[@]}"; do
    if [ "${EXPERIMENT_MODE:-test}" = test ] ||
      [ "${EXPERIMENT_FAMILY:-}" = ts_ifa_h_ablation ] || [ "${EXPERIMENT_FAMILY:-}" = ts_ifa_l_ablation ]; then
      if [ "$value" != 3 ]; then
        log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE permits only K=3; got K=$value"
        return 2
      fi
    else
      case "$value" in
        1|3) ;;
        *)
          log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE permits only K=1,3; got K=$value"
          return 2
          ;;
      esac
    fi
  done
}

requires_selected_methods() {
  case "${EXPERIMENT_FAMILY:-}" in
    benchmark|mixed_quantity_ablation|fourier_retrieval_ablation|offline_datastore_ablation|retrieval_scope_ablation|horizon_baselines_ablation|convex_baselines_ablation|delta_baselines_ablation|catboost_ablation|k_ablation|h_ablation|l_ablation|crossrag|sota_benchmark) return 0 ;;
    *) return 1 ;;
  esac
}
