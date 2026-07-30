# Forecast adaptation and TS-IFA

This project studies retrieval-based adaptation of frozen time-series
forecasters.  The current paper-first path is to establish strong direct
retrieval baselines and learned gates; TS-IFA remains an architecture-tuning
track until its T2/T3 overfitting is controlled.

## Protocol, windows, and outputs

Every integer `s` stored as `query_t` is the last observed query date. Windows
are exactly

```text
X_s = (s-L, s] = {z_(s-L+1), ..., z_s}
Y_s = (s, s+H] = {z_(s+1), ..., z_(s+H)}
```

so `X_s` and `Y_s` contain exactly `L` and `H` values and never overlap. If a
target period starts at date `b`, its first eligible query date is `b-1`; the
lookback may cross a split boundary, but the entire target must lie inside the
selected target period. This keeps the same T3 target dates when `L` changes.

Extraction fixes only three chronological regions:

- T0 (30%) is the retrieval datastore.
- pooled T1+T2 (50%) is written once as the `adapt` payload.
- T3 (20%) is written as the untouched `eval` payload.

Each downstream model chronologically re-splits `adapt` by whole query dates;
the default assigns its last 20% of dates to T2. Users from the same date are
never separated. The model-specific protocols are:

- lambda mixtures fit on all T1+T2; their T1 fit is also scored on T2 as a
  diagnostic;
- ridge models fit all candidate alphas on T1, select alpha by T2 nMSE, and
  refit the selected model on T1+T2;
- fixed-candidate CatBoost gates fit on T1, use T2 early stopping to select the
  number of trees, then instantiate a fresh model and refit on T1+T2;
- TS-IFA trains on T1, selects/restores a checkpoint on T2, and does not refit
  after selection;
- a future gate over a trainable candidate must train that candidate on T1 and
  the gate on T2 without later changing the candidate, unless out-of-fold
  candidate predictions are introduced.

Nothing fitted on T3 belongs in the main comparison.

Extraction writes to
`outputs/adaptation/<dataset>/<L>_<H>/<model>/<retrieval>/extracted/`.
A usable extraction contains adapt/eval prediction and feature payloads
plus `extraction_manifest.json`.  The manifest is written atomically only after
all payloads exist and records the exact extraction signature, the resolved
dataset-config path and content hash, and file sizes.
`--skip-complete` therefore skips a matching complete run but re-runs a partial,
changed, or obsolete extraction.

Downstream results are profile-separated under
`outputs/adaptation_results/<experiment_mode>/`; extraction payloads remain
shared under `outputs/adaptation/`. Their contracts are:

```text
outputs/adaptation_results/<mode>/<dataset>/<L>_<H>/<model>/
  <retrieval>/baselines/{baseline_metrics.json,baseline_artifacts.pt,prediction_manifest.json,result_manifest.json,...}
  <retrieval>/gates/{gate_metrics.json,gate_artifacts.json,prediction_manifest.json,result_manifest.json,...}
  <retrieval>/crossrag/{crossrag_metrics.json,crossrag_predictions.pt,crossrag_timing.json}
  <retrieval>/ts_ifa/TS-IFA/{eval_metrics.json,config.json,ts_ifa.pt,prediction_manifest.json,result_manifest.json,...}
  tables/<model>/{full,average}/{baselines_results.tex,gates_results.tex,...}
```

Baseline, gate, and TS-IFA predictions use the sole current disk-backed
`prediction_manifest.json` contract. Each array is a separate `.npy` file below
`predictions/`; the dashboard reads only this contract. `result_manifest.json`
is written last and is the completion marker. Obsolete or partial result
folders are not accepted and are replaced when the run is launched again.
Gate runs also index their per-model CatBoost feature-importance CSV/PNG files
from `gate_artifacts.json`. TS-IFA stores T3 neural-rooter coefficients under
the prediction store's `gate_diagnostics` kind and its fixed coefficient matrix
in `ridge_rooter.pt`.

The baseline launcher retains `--fit-baselines-on-eval`.  Methods suffixed
`_eval_fit` are optimistic T3 in-sample oracle diagnostics for the appendix;
they are intentionally excluded from the deployable main comparison.
Ridge fits accumulate float64 sufficient statistics in bounded chunks, so they
use the complete selected fitting split without materializing the full design
matrix. Predictions are written, scored in chunks, and released one method at a
time. These changes bound memory without changing the fitted objective.

Baseline and gate fitting may optionally use reproducible subsets of the
already-extracted payloads through `MAX_T1_FIT_SAMPLES`,
`MAX_T2_VALID_SAMPLES`, `MAX_ADAPT_REFIT_SAMPLES`, and
`MAX_EVAL_FIT_SAMPLES`. All default to unlimited; `FIT_SAMPLE_SEED` defaults to
`SEED`. The first three limits affect only T1 fitting, T2 validation, and the
final T1+T2 refit respectively. The T3 maximum applies only to explicitly
optimistic `_eval_fit` methods. Final T3 scoring always uses every evaluation
sample.

For period-aligned retrieval, neighbor query dates `r_j` satisfy
`(s-r_j) mod P = 0`. In fixed mode, both the neighbor lookback and future lie
inside T0. In online mode, `r_j+H <= s`, so the complete retrieved future is
already observable at query date `s`. A neighbor future may overlap the
observed query lookback but can never overlap the query target.

## Data and weight locations

Submit from the project root.  Launchers search, in order, the project-local
folder (`datasets/` or `weights/`), the project parent, and an additional shared
parent candidate. The first folder containing the requested dataset or weight is
used.  When the repository is copied elsewhere, explicitly set the roots:

```bash
DATA_ROOT=/cluster/shared/datasets \
WEIGHTS_ROOT=/cluster/shared/weights \
sbatch extraction.slurm
```

`CHRONOS2_WEIGHTS_PATH`, `CHRONOS_BOLT_WEIGHTS_PATH`, and
`TABPFN_WEIGHTS_PATH` can override individual model paths. The model names are
`chronos2`, `chronos-bolt`, and `tabpfnts`.
TS-ICL is documented as a later extension and is rejected by the launcher until
it is implemented and registered.

Dataset directories may contain a sibling `config.json`. It is discovered by
the Python loader even for direct runs; `--dataset-config` accepts an explicit
JSON file or directory. Portable fields such as `drop_users`, `date_col`, and
aggregation settings live at the top level. Adaptation-only values belong under
`adaptation`. Project-scoped values
override other settings, while `drop_users` is merged additively with both the
top-level list and `--drop-users`. The loader logs the selected path and applied
keys.

ETTh1 uses every non-date variable in the source CSV in every profile. Ensure
the cluster copy is the complete seven-variable ETTh1 file rather than an
`OT`-only derivative.

The repository tracks the curated Electricity `config.json` while leaving its
CSV ignored, so the same exclusions—including source column 245—are carried to
cluster checkouts and shared with RevIN.

## Experiment profiles and required order

The main study uses

```text
D = {ETTh1, Electricity, Traffic, Solar, Weather, exchange_rate}
S = {168:24, 336:48, 504:168}
```

with Chronos-2. `src/slurm/profiles.sh` is the single source of truth:

- `screen`: every `(formula, retrieval normalization, K)` pipeline on `D x S`,
  with raw and instance-normalized retrieval and `K in {1,10}`.
- `k_ablation`: the manually named winning pipelines on `D x S`, varying only
  `K in {1,3,5,10,15,20}` while retaining each formula and normalization.
- `h_ablation`: manually named pipelines with `L=504` and
  `H in {24,168,504}`.
- `l_ablation`: manually named pipelines with `H=24` and
  `L in {24,168,504}`.
- `crossrag`: a separate Chronos-Bolt comparison at exactly `L=512`, `H=64`,
  `K=15`, per-window min-max X-space retrieval, and cosine distance. It is not
  crossed with `S`.

`test`, `small`, `full`, and `ultra` remain for smoke testing and historical
reproduction.
The generic `extraction.slurm`, `baselines.slurm`, `gates.slurm`,
`tables.slurm`, and `run_all.sh` launchers default to `test`; select a
publication profile explicitly.
The `exchange_rate` key resolves the shared
`datasets/exchange_rate/exchange_rate.csv` layout used by the other projects.

Each experiment is one sequential Slurm job. Run the screen first:

```bash
sbatch screen.slurm
```

For screening, a **setting** is one dataset plus one `L:H` pair, so there are
`|D| x |S| = 18` settings. Each complete pipeline is scored by the unweighted
mean of its 18 setting-level percentage improvements over vanilla Chronos-2.
There is no averaging over K or normalization: those identify different
pipelines. The average table also writes a sorted `pipeline_ranking.csv` whose
`winner_name` includes family, retrieval configuration, and formula.

Copy the selected complete names into `WINNERS_CSV` near the top of each later
Slurm file (or pass the same variable as an environment override):

```bash
WINNERS_CSV="${WINNERS_CSV:-baselines/instance_euclidean_10_online/y_ridge_horizon,gates/raw_euclidean_1_online/catboost_aggr_y_regressor_shared}"
```

Then submit exactly one front per experiment:

```bash
sbatch k_ablation.slurm
sbatch h_ablation.slurm
sbatch l_ablation.slurm
```

The K experiment expands each winner across the K grid while retaining its
formula and normalization. The H and L experiments retain each full winning
pipeline, including its K and normalization, while changing only the requested
lookback/horizon axis. Dataset–`L:H` pairs remain evaluation settings.

The Cross-RAG comparison is also one job. Enter one complete winning pipeline in
`crossrag.slurm`, configure the three released-code/checkpoint paths in the
environment, and submit it:

```bash
CROSSRAG_ROOT=/cluster/code/Cross-RAG \
CROSSRAG_BASE_CHECKPOINT=/cluster/code/Cross-RAG/cross-rag/checkpoints/base \
CROSSRAG_CHECKPOINT=/cluster/code/Cross-RAG/cross-rag/checkpoints/.../best.pth \
WINNERS_CSV=baselines/instance_euclidean_10_online/y_ridge_horizon \
sbatch crossrag.slurm
```

`crossrag.slurm` imports the official released model implementation and loads
its pretrained adapter checkpoint; it does not retrain Cross-RAG. It evaluates
the released model on the same T3 settings as our candidate. The candidate
retains its selected normalization while Cross-RAG uses its prescribed
min-max/cosine retrieval. The comparison tables include accuracy, and
`timing_comparison.tex` reports wall-clock vanilla, retrieval, adaptation, and
Cross-RAG totals, including each pipeline's own retrieval pass.

Extraction defaults to `SKIP_COMPLETE=true` and validates an atomic manifest
with the exact signature and timing artifact. Selected-candidate profiles
default downstream skipping to false because changing `*_METHODS_CSV` changes
the fitted output. Normal logs are under `logs/`; if a sequential job reaches
its time limit, resubmit the same mode and completed extractions will be reused.

All sweep dimensions remain overridable through `DATASETS_CSV`, `MODELS_CSV`,
`SETTINGS_CSV`, `DISTANCE_SPACES_CSV`, `DISTANCE_METRICS_CSV`, and
`NEIGHBORS_CSV`. Settings use `L:H`. `MAX_T1_FIT_SAMPLES`,
`MAX_T2_VALID_SAMPLES`, and `MAX_ADAPT_REFIT_SAMPLES` affect fitting only;
complete T3 scoring is never subsampled. Gate horizons are fitted serially with
two CatBoost threads by default. Each model is scored, saved in CatBoost's
native format, and released before the next model is fitted. Gate summaries are
computed in bounded chunks, horizon feature matrices are materialized lazily,
and predictions and diagnostic scores are disk-backed.

## Tables and averages

The only table front end is `tables.slurm`; it delegates to
`src/slurm/build_tables.sh`. It
checks every selected input rather than silently constructing a sparse table,
then writes separate Chronos and TabPFN-TS tables.  `full/` reports each
dataset/setting/retrieval result. `average/` reports, for each method, the
unweighted mean of its per-configuration percentage improvements from the
matching vanilla backbone (plus its mean metric below). This is deliberately
not the percentage computed from two pooled mean nMSE values. It prevents large
datasets or large-error configurations from receiving implicit extra weight.
Report it alongside, not instead of, per-dataset results and user-tail analyses.

Use `FAMILIES_CSV=baselines`, `FAMILIES_CSV=gates`, or include `ts_ifa` after
those outputs exist.  `METRIC=mse` produces the corresponding MSE tables;
`nmse` is the default.

## TS-IFA staged architecture

TS-IFA exposes four candidates by default: vanilla, context, a
residual-attention branch over analogue `[history, prediction]` states, and a
direct memory-attention branch. The past-only `sign(x)`/`sqrt(abs(x))` frozen
forecast is an optional fifth candidate and is disabled by default.
The final rooter computes unconstrained horizon-wise coefficients with one
candidate-conditioned scorer shared across the active type tokens. It does
not consume handcrafted retrieval distances or dispersion features.

`LEARNABLE_TRANSFORMED_COVARIATE=true` adds
`u=MLP(x)` and a T1-trained transformed forecast head. It automatically enables
the fifth candidate and works with old payloads by anchoring it to vanilla.
It can be combined with `TRANSFORMED_EXPERT=true`, in which case it is anchored
to the precomputed sign/root forecast instead. Producing that frozen forecast
requires `COMPUTE_TRANSFORMED_PREDICTION=true` during extraction.

Training is deliberately staged:

- the residual, memory, and optional learned transformed branches optimize
  their own losses on T1;
- the branches are frozen before both a 256-coefficient horizon-wise ridge
  rooter and the neural rooter are trained independently on T2;
- T3 is evaluated once, with no checkpoint selection or refit.

Vanilla-anchoring initialization makes both learned branches and the complete
neural model exactly equal to vanilla at step zero. Optional regularizers cover
branch/final vanilla anchoring, coefficient L2, and first-order horizon
smoothness. Set `VANILLA_ANCHOR`, `COEFFICIENT_L2`, or `HORIZON_SMOOTHNESS` to
zero to disable each term; set `VANILLA_ANCHORING_INIT=false` to disable the
initialization.

Each run saves `branches.pt`, `ridge_rooter.pt`, `ts_ifa.pt`, and T3 metrics and
disk-backed predictions for every candidate, the ridge rooter, and the neural
rooter. Evaluation writes and scores predictions batch by batch, while copied
T1/T2/T3 tensors let the original extraction payloads be released before
training. The extraction contract itself is unchanged, so completed extractions
whose manifest matches the current signature remain valid.

TS-IFA smoke submission:

```bash
EXPERIMENT_MODE=test sbatch ts_ifa.slurm
EXPERIMENT_MODE=test FAMILIES_CSV=ts_ifa sbatch tables.slurm

# Optional learned candidate; no re-extraction needed:
EXPERIMENT_MODE=test LEARNABLE_TRANSFORMED_COVARIATE=true sbatch ts_ifa.slurm

# Optional frozen sign/root candidate:
EXPERIMENT_MODE=test COMPUTE_TRANSFORMED_PREDICTION=true sbatch extraction.slurm
EXPERIMENT_MODE=test TRANSFORMED_EXPERT=true sbatch ts_ifa.slurm
```

Its input extraction must already have a valid completion manifest.
TS-IFA follows the shared small/full/ultra dataset, setting, and backbone
profiles, but remains outside the paper-critical baseline/gate path while its
architecture is being tuned.

## Executable files

Only the concise `.slurm` files in the project root are submitted. They contain
scheduler resources and the `EXPERIMENT_MODE` switch, while `src/slurm/*.sh` contains
enumeration, input checks, and command invocation:

- `extraction.slurm` -> `src/slurm/extract_adaptation.sh`.
- `baselines.slurm` -> `src/slurm/run_baselines.sh`.
- `gates.slurm` -> `src/slurm/run_gates.sh`.
- `tables.slurm` -> `src/slurm/build_tables.sh`.
- `ts_ifa.slurm` and `univariate.slurm` are optional model/reference jobs.

Implementation shells:

- `extract_adaptation.sh` builds vanilla and retrieval extraction tasks and
  calls `src.experiments.extraction`.
- `run_baselines.sh` checks extraction manifests and evaluates direct, ridge,
  horizon-ridge, and optimistic appendix references.
- `run_gates.sh` uses the same evaluator with `--family gates` to fit and score
  the candidate gates.
- `run_ts_ifa.sh` trains T1 branches, fits T2 ridge and neural rooters, and
  writes complete per-candidate and T3 comparison metrics.
- `run_univariate.sh` runs direct Chronos forecasts without retrieval.
- `build_tables.sh` verifies the selected sweep is complete before producing
  full and equal-configuration-average tables.
- `common.sh` provides resource lookup, setting parsing, manifest checks, and
  timestamped shell logging; it is sourced, not submitted.

The runnable Python modules are:

- `src.experiments.extraction`: frozen-backbone inference, features, neighbors,
  prediction payloads, and the atomic completion manifest.
- `src.experiments.experiment_univariate`: direct univariate backbone reference.
- `src.experiments.artifacts`: command-line validation of an extraction folder.
- `src.adaptors.baselines.evaluate`: both baseline and gate families, selected
  with `--family baselines` or `--family gates`.
- `src.adaptors.ts_ifa.train`: staged T1 branch training, T2 rooter fitting, and
  T3 evaluation.
- `src.visu.sweep_results_table`: full and averaged publication tables.
- `src.visu.results_table` and `src.visu.selected_methods`: focused table
  utilities for individual result folders and selected method subsets.
- `src.visu.dashboard`: interactive retrieval diagnostics. Library modules such
  as `features.py`, `runtime.py`, `models/*.py`, and `data/*.py` support these
  entry points and are not separate jobs.

The dashboard takes the extraction directory and the corresponding
profile-separated result directory as distinct inputs. It loads a result family
only when its current `result_manifest.json` is complete and points to the
current prediction store. Configure `CLUSTER_EXPERIMENT_MODE`, dataset,
`L_H` setting, model, and retrieval name in
`src/visu/retrieval_dashboard.ipynb`; the notebook derives both paths from that
single configuration. Its interactive sections cover retrieved examples,
window and horizon errors, gate summaries and threshold curves, CatBoost gate
importance, fitted baseline coefficient importance, and TS-IFA ridge/neural
rooter coefficient heatmaps.

TS-IFA outputs generated before neural-rooter coefficient diagnostics were
introduced remain sufficient for the fixed ridge heatmap, but rerun
`ts_ifa.slurm` for the affected configurations to populate the neural-rooter
heatmap. The launcher treats prediction stores without that diagnostic as
incomplete.

## Local checks

Full extraction and model inference run only on the remote cluster.  With the
user-prepared project environment, lightweight checks are:

```bash
python src/tests/smoke/check_extraction_manifest.py
python src/tests/smoke/check_loads.py
python src/tests/smoke/check_baseline_oracles.py
python src/tests/smoke/check_ts_ifa_training.py
python src/tests/smoke/check_results_table.py
python src/tests/smoke/check_sweep_results_table.py
python src/tests/smoke/check_retrieval_dashboard.py
```

The experiment guides and their compiled PDFs are under
`latex/experiment_guides/`: `01_univariate_control`, `02_retrieval_baselines`,
`03_learned_gates`, `04_ts_ifa`, and `05_related_methods`. The second and third
give the exact artifact names, formulas, feature definitions, and validation
protocols; the fifth records the dated retrieval/adaptation literature
comparison and source provenance. Source code, notebooks, tests, and Slurm
helpers remain under `src/`; generated artifacts stay under `outputs/`, and
runtime logs under `logs/`.
