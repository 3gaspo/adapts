#!/bin/bash
# Shared experiment grids. Source after common.sh, then call adaptation_profile_defaults.

adaptation_profile_defaults() {
  local mode="${EXPERIMENT_MODE:-test}"
  DEFAULT_DATASETS_CSV=""
  DEFAULT_MODELS_CSV=""
  DEFAULT_SETTINGS_CSV=""
  DEFAULT_DISTANCE_SPACES_CSV=""
  DEFAULT_DISTANCE_METRICS_CSV="euclidean"
  DEFAULT_NEIGHBORS_CSV=""
  DEFAULT_DATASTORE_STRIDE=24
  DEFAULT_ADAPT_QUERY_STRIDE=24
  DEFAULT_EVAL_QUERY_STRIDE=128
  DEFAULT_MAX_STORE_WINDOWS=30000

  case "$mode" in
    test)
      DEFAULT_DATASETS_CSV="Electricity"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="168:24"
      DEFAULT_DISTANCE_SPACES_CSV="raw"
      DEFAULT_NEIGHBORS_CSV="3"
      DEFAULT_DATASTORE_STRIDE=168
      DEFAULT_ADAPT_QUERY_STRIDE=256
      DEFAULT_EVAL_QUERY_STRIDE=256
      DEFAULT_MAX_STORE_WINDOWS=2048
      ;;
    screen)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="168:24,336:48,504:168"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,10"
      ;;
    k_ablation)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="168:24,336:48,504:168"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3,5,10,15,20"
      ;;
    h_ablation)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="504:24,504:168,504:504"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="10"
      ;;
    l_ablation)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="24:24,168:24,504:24"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="10"
      ;;
    crossrag)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos-bolt"
      DEFAULT_SETTINGS_CSV="512:64"
      DEFAULT_DISTANCE_SPACES_CSV="minmax"
      DEFAULT_DISTANCE_METRICS_CSV="cosine"
      DEFAULT_NEIGHBORS_CSV="15"
      ;;
    small)
      DEFAULT_DATASETS_CSV="Traffic,Electricity,Solar"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="168:24,504:24,504:168,504:504"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3,10"
      ;;
    full|large)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2"
      DEFAULT_SETTINGS_CSV="168:24,504:24,504:168,504:504,512:64"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3,10"
      ;;
    ultra)
      DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,exchange_rate"
      DEFAULT_MODELS_CSV="chronos2,tabpfnts"
      DEFAULT_SETTINGS_CSV="168:24,504:24,504:168,504:504,512:64"
      DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
      DEFAULT_NEIGHBORS_CSV="1,3,10"
      ;;
    *)
      log_error "no defaults for experiment mode=$mode"
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

requires_selected_methods() {
  case "${EXPERIMENT_MODE:-test}" in
    k_ablation|h_ablation|l_ablation|crossrag) return 0 ;;
    *) return 1 ;;
  esac
}
