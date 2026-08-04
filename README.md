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

- convex models fit on all T1+T2; their T1 fit is also scored on T2 as a
  diagnostic;
- ridge models fit all candidate alphas on T1, select alpha by T2 nMSE, and
  refit the selected model on T1+T2;
- fixed-candidate CatBoost gates fit on T1, use T2 early stopping to select the
  number of trees, then instantiate a fresh model and refit on T1+T2;
- every publication profile allows at most 300 CatBoost trees; only the
  `EXPERIMENT_MODE=test` smoke profile uses two iterations;
- joint TS-IFA variants optimize branches and one active rooter together on
  T1, select the checkpoint on T2, and evaluate it once on T3; meta variants
  use chronological T1 support/query episodes, then freeze the branches and
  fit only their active rooter on T2 before the single T3 evaluation;
- a future gate over a trainable candidate must train that candidate on T1 and
  the gate on T2 without later changing the candidate, unless out-of-fold
  candidate predictions are introduced.

Nothing fitted on T3 belongs in the main comparison.

Extraction writes to
`outputs/extractions/<dataset>/<L>_<H>/<model>/<retrieval>/extracted/`.
A usable extraction contains adapt/eval prediction and feature payloads
plus `extraction_manifest.json`.  The manifest is written atomically only after
all payloads exist and records the exact extraction signature, the resolved
dataset-config path and content hash, and file sizes.
`--skip-complete` therefore skips a matching complete run but re-runs a partial,
changed, or obsolete extraction.

This extraction root replaces `outputs/adaptation/` without a compatibility
reader. Git does not move ignored payloads already present on a persistent
cluster checkout, so migrate them once after pulling this change:

```bash
mkdir -p outputs/extractions
rsync -a --remove-source-files outputs/adaptation/ outputs/extractions/
```

After migration, downstream jobs reuse the extraction manifests and payloads
from `outputs/extractions/`; extraction does not need to be rerun.

Downstream results are profile-separated under
`outputs/adaptation_results/<experiment_mode>/`; extraction payloads remain
shared under `outputs/extractions/`. Their contracts are:

```text
outputs/adaptation_results/<mode>/<dataset>/<L>_<H>/<model>/
  <retrieval>/baselines/{baseline_metrics.json,baseline_artifacts.pt,prediction_manifest.json,result_manifest.json,...}
  <retrieval>/gates/{gate_metrics.json,gate_artifacts.json,prediction_manifest.json,result_manifest.json,...}
  <retrieval>/crossrag/{crossrag_metrics.json,crossrag_predictions.pt,crossrag_timing.json}
  <retrieval>/ts_ifa/<joint_ridge|joint_neural|meta_ridge|meta_neural>/{eval_metrics.json,config.json,ts_ifa.pt,rooter.pt,prediction_manifest.json,result_manifest.json,...}
  tables/<model>/{full,average}/{baselines_results.tex,gates_results.tex,positive_windows_results.tex,...}
```

Baseline, gate, and TS-IFA predictions use the sole current disk-backed
`prediction_manifest.json` contract. Each array is a separate `.npy` file below
`predictions/`; the dashboard reads only this contract. `result_manifest.json`
is written last and is the completion marker. Obsolete or partial result
folders are not accepted and are replaced when the run is launched again.
Gate runs also index their per-model CatBoost feature-importance CSV/PNG files
from `gate_artifacts.json`. Every TS-IFA variant stores its T3 active-rooter
coefficients under the prediction store's `gate_diagnostics` kind and its
rooter state in `rooter.pt`. Downstream launchers publish the shared vanilla metrics
and extraction timing only when the destination is absent or stale; this
publication is atomic, so baseline, gate, and TS-IFA jobs may start in parallel.

The baseline launcher retains `--fit-baselines-on-eval`.  Methods suffixed
`_eval_fit` are optimistic T3 in-sample oracle diagnostics for the appendix;
they are intentionally excluded from the deployable main comparison.
Ridge fits accumulate float64 sufficient statistics in bounded chunks, so they
use the complete selected fitting split without materializing the full design
matrix. Predictions are written, scored in chunks, and released one method at a
time. These changes bound memory without changing the fitted objective.
Baseline and gate metric rows also store `positive_window_pct`, the percentage
of complete evaluation windows whose horizon-averaged MSE is strictly below
the matching vanilla forecast. No per-window metric vector is persisted.
Baseline/gate results produced before this field and the neighbor-age-dispersion
gate feature are obsolete and must be rerun; their extraction payloads remain
reusable.

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

The extraction launcher resolves `P=96` for `ETTm1`, `ETTm2`, `ETT_T_15T`,
and `ETT_L_15T`, and `P=24` otherwise. Unless `DATASTORE_STRIDE` is explicitly
set, its profile default is rounded up to a multiple of the resolved period;
explicit `PERIOD` and `DATASTORE_STRIDE` overrides remain available.
Neighbor search accepts raw, instance-normalized, min--max, Fourier-amplitude,
and encoder representation spaces. Fourier retrieval standardizes each
lookback before taking the FFT magnitude; it is an override option and is not
part of the primary raw/instance screen.

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

The mixed-quantity ablation uses every non-date variable in each original ETT
CSV. Ensure the cluster copies of `ETTh1`, `ETTh2`, `ETTm1`, and `ETTm2` are
the complete seven-variable released files rather than `OT`-only derivatives.
The quantity-separated full-mode panels are named `ETT_T_1H`, `ETT_L_1H`,
`ETT_T_15T`, and `ETT_L_15T`; each folder contains a same-named CSV and its
`config.json`.
All original and quantity-separated ETT panels, Weather, and Exchange Rate
retain every non-date variate. Their configs keep `drop_users` empty; observed
constant windows are diagnostic metadata rather than a channel exclusion rule.

The repository tracks the curated Electricity `config.json` while leaving its
CSV ignored, so the same exclusions—including source column 245—are carried to
cluster checkouts and shared with RevIN.

## Experiment profiles and required order

The main study uses

```text
D = {Electricity, Traffic, Solar, exchange_rate}
S_default = {168:24, 336:48, 504:168}
```

with Chronos-2. `src/slurm/profiles.sh` is the single source of truth:

- `screen`: every `(formula, retrieval normalization, K)` pipeline on `D x S`,
  with raw and instance-normalized retrieval and `K in {1,3}`. Baselines use
  direct or shared coefficients only. Learned gates use the shared CatBoost
  signed-advantage regressor; shared no-feature and oracle gates remain as
  diagnostics.
- `mixed_quantity_ablation`: selected complete screen winners transferred to
  `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, and `Weather`. Their channels represent
  unlike physical quantities, so cross-variable nearest neighbors are studied
  separately and never enter the primary ranking.
- `horizon_baselines_ablation`: selected shared baseline winners and their
  per-horizon counterparts on the same dataset/setting grid and exact winner
  retrieval configuration.
- `convex_baselines_ablation`: selected shared-ridge winners and the shared
  simplex-constrained convex models using the same variables and retrieval.
- `delta_baselines_ablation`: selected shared-ridge winners and the shared
  delta-ridge models using the same variables and retrieval.
- `catboost_ablation`: selected shared-regressor screen winners, each expanded
  under the same retrieval pipeline by changing one axis at a time: classifier
  objective, soft mixture output, or per-horizon regression. Matching
  no-feature and oracle references are included.
- `k_ablation`: the manually named winning pipelines on `D x S`, varying only
  `K in {1,3,5,10,15,20}` while retaining each formula and normalization.
- `h_ablation`: manually named pipelines with `L=504` and
  `H in {24,168,504}`.
- `l_ablation`: manually named pipelines with `H=24` and
  `L in {24,168,504}`.
- `crossrag`: a separate Chronos-Bolt comparison at exactly `L=512`, `H=64`,
  `K=15`, per-window min-max X-space retrieval, and cosine distance. It is not
  crossed with `S`.

The publication screen uses `K in {1,3}`; the smoke-only `test` profile fixes
`K=3`. The only wider grids are the selected-winner `k_ablation` and
Cross-RAG's prescribed `K=15` setup. The final benchmark and every ablation
consume selected complete screen pipelines, so none reopens the complete
method/retrieval sweep.

The resolved grids are summarized below. `D_primary` is Electricity, Traffic,
Solar, and Exchange; `D_full` adds the four quantity-separated ETT panels;
`D_mixed` is the four original ETT panels plus Weather.

| Experiment mode | Datasets | `(L,H)` settings | Backbone | Retrieval / K | Adaptation methods |
|---|---|---|---|---|---|
| `test` | Electricity | `504:168` | Chronos-2 | Raw Euclidean / `3` | Primary baselines/gates or TS-IFA, according to front |
| `screen` | `D_primary` | `168:24`, `336:48`, `504:168` | Chronos-2 | Raw + instance Euclidean / `1,3` | 3 direct + 7 shared-ridge baselines; 8 shared gate/reference methods |
| `full` | `D_full` | Default three settings | Chronos-2 | Each selected winner's complete retrieval pipeline | `WINNERS_CSV` only |
| `ultra` | `D_full` | Default three settings | Chronos-2, TabPFN-TS | Each selected winner's complete retrieval pipeline | Same `WINNERS_CSV` as full |
| `mixed_quantity_ablation` | `D_mixed` | Screen settings | Chronos-2 | Selected pipeline / `1` or `3` | `WINNERS_CSV` only |
| `horizon_baselines_ablation` | `D_primary` | Screen settings | Chronos-2 | Selected pipeline / `1` or `3` | Shared ridge versus horizon-wise ridge |
| `convex_baselines_ablation` | `D_primary` | Screen settings | Chronos-2 | Selected pipeline / `1` or `3` | Shared ridge versus shared convex |
| `delta_baselines_ablation` | `D_primary` | Screen settings | Chronos-2 | Selected pipeline / `1` or `3` | Shared ridge versus shared delta-ridge |
| `catboost_ablation` | `D_primary` | Screen settings | Chronos-2 | Selected pipeline / `1` or `3` | Shared regressor versus classifier, soft mixture, and horizon regressor |
| `k_ablation` | `D_primary` | Screen settings | Chronos-2 | Winner's space/metric / `1,3,5,10,15,20` | `WINNERS_CSV` only |
| `h_ablation` | `D_primary` | `504:24`, `504:168`, `504:504` | Chronos-2 | Selected pipeline / `1` or `3` | `WINNERS_CSV` only |
| `l_ablation` | `D_primary` | `24:24`, `168:24`, `504:24` | Chronos-2 | Selected pipeline / `1` or `3` | `WINNERS_CSV` only |
| `crossrag` | `D_primary` | `512:64` | Chronos-Bolt | Min-max cosine / `15` | One selected winner versus released Cross-RAG |

For the primary methods, let `V` be the vanilla forecast, `C` the
covariate-conditioned forecast, `Y_j` the observed future of neighbor `j`,
`N_j` that neighbor's backbone forecast, and
`avgy = bar(Y) = sum_j w_j Y_j` the distance-weighted neighbor future. All formulas
below are evaluated horizon-wise; `shared` means that fitted coefficients or
the gate decision are shared across the complete horizon.

Each fitted baseline is identified by a variable design and a family. The
seven designs are:

| Design | Forecast variables |
|---|---|
| `cov` | `V, C` |
| `avgy` | `V, bar(Y)` |
| `y` | `V, Y_1, ..., Y_K` |
| `cov_y` | `V, C, Y_1, ..., Y_K` |
| `cov_avgy` | `V, C, bar(Y)` |
| `residual` | `V, Y_1, ..., Y_K, N_1, ..., N_K` |
| `full` | `V, C, Y_1, ..., Y_K, N_1, ..., N_K` |

For alternatives `Z_1, ..., Z_p` beside `V`, the three shared families are:

- ridge: `V + aV + sum_m b_m Z_m`, with all correction coefficients
  regularized toward zero;
- convex: `lambda_0 V + sum_m lambda_m Z_m`, with non-negative weights that
  sum to one;
- delta-ridge: `V + sum_m b_m (Z_m - V)`, with the delta coefficients
  regularized toward zero.

The same three definitions admit horizon-wise coefficients. The main screen
uses only the most flexible shared-ridge family; the three baseline ablation
fronts change respectively coefficient shape, family to convex, or family to
delta-ridge for selected variable designs.

The ten primary baseline artifacts are:

- direct: `cov_forecast = C`, `avgy = bar(Y)`, and
  `y_mean = K^-1 sum_j Y_j`;
- anchored ridge: `V + aV` plus, respectively, `bC`
  (`cov_ridge_shared`), `b bar(Y)` (`avgy_ridge_shared`),
  `sum_j b_jY_j` (`y_ridge_shared`), `bC + sum_j c_jY_j`
  (`cov_y_ridge_shared`), `bC + c bar(Y)`
  (`cov_avgy_ridge_shared`), `sum_j b_jY_j + sum_j c_jN_j`
  (`residual_ridge_shared`), or
  `bC + sum_j c_jY_j + sum_j d_jN_j` (`full_ridge_shared`).

Variant names replace `_ridge_shared` with `_ridge_horizon`, `_convex_shared`,
or `_delta_ridge_shared`. For example, the `cov` variants are
`cov_ridge_horizon`, `cov_convex_shared`, and `cov_delta_ridge_shared`.

The primary gates compare `V` with a candidate `A` in `{C, bar(Y)}`. Define
`Delta_h = (Y_h - V_h)^2 - (Y_h - A_h)^2` and
`Delta_bar = H^-1 sum_h Delta_h`; the shared decision outputs `A` when its
score is positive and `V` otherwise. For both candidates the screen includes
the direct candidate, `bayes_<A>_shared` (constant decision from mean training
advantage), `catboost_<A>_regressor_shared` (feature-based prediction of
`Delta_bar`), and `oracle_<A>_shared` (target-aware T3 upper bound). This gives
eight gate/reference artifacts in total. The CatBoost ablation adds the shared
classifier, shared soft regressor mixture, and horizon-wise regressor one at a
time. The soft regressor maps predicted advantage to `[0,1]` with a sigmoid
scaled by the refit-target standard deviation, then writes `(1-p)V + pA`.

Every TS-IFA variant reports vanilla, cov, residual, and memory candidates plus
one active rooter. The four models share the same branch architecture; only the
rooter form and optimization contract differ.

Modes provide default grids, including default backbone models; they are not
model-free labels. `test`, `screen`, and `full` default to `chronos2`, `ultra`
defaults to `chronos2,tabpfnts`, and `crossrag` fixes `chronos-bolt`.
`MODELS_CSV` explicitly overrides a model list where the launcher enumerates
backbones, for example `MODELS_CSV=chronos2,tabpfnts`. The primary screen is
defined on Chronos-2 and should retain that reference backbone. Every
extraction already writes the matching vanilla backbone forecast and metrics;
there is no separate univariate submission.

`test` is exactly one Electricity setting and one retrieval pipeline:
`L=504`, `H=168`, raw-space Euclidean online retrieval, and `K=3`. `screen`
is the only complete candidate sweep. `full` is the final selected-candidate
benchmark on `D_full` and the same three default settings; the selected winner
name already fixes formula, raw/instance normalization, metric, K, and online
or fixed retrieval. `ultra`
uses that identical final benchmark and adds the additional default backbones. Both modes write to
`outputs/adaptation_results/full`, allowing ultra to reuse completed full
results. The obsolete `small` profile has been removed.

The intended submission interface is:

| Slurm front | Accessible mode(s) | What it runs |
|---|---|---|
| `screen.slurm` | fixed `screen` | Primary extraction, 10 baselines, 8 gate/reference methods, and tables |
| `benchmark.slurm` | `full` or `ultra` | Selected screen winners on `D_full`, followed by tables |
| `extraction.slurm` | `test`, `full`, `ultra` | Standalone extraction, mainly for smoke or TS-IFA inputs |
| `baselines.slurm` | `test` | Primary baseline smoke run; requires test extraction |
| `gates.slurm` | `test` | Primary gate smoke run; requires test extraction |
| `tables.slurm` | `test`, `full`, `ultra` | Standalone tables, including optional TS-IFA tables |
| `ts_ifa_joint_ridge.slurm` | `test`, `full`, `ultra` | Joint-gradient ridge TS-IFA; requires matching extraction |
| `ts_ifa_joint_neural.slurm` | `test`, `full`, `ultra` | Joint-gradient neural TS-IFA; requires matching extraction |
| `ts_ifa_meta_ridge.slurm` | `test`, `full`, `ultra` | Closed-form-adapted ridge meta-TS-IFA; requires matching extraction |
| `ts_ifa_meta_neural.slurm` | `test`, `full`, `ultra` | Gradient-adapted neural meta-TS-IFA; requires matching extraction |
| `mixed_quantity_ablation.slurm` | fixed `mixed_quantity_ablation` | Selected winners on original ETT panels and Weather |
| `horizon_baselines_ablation.slurm` | fixed `horizon_baselines_ablation` | Selected shared ridge versus horizon ridge |
| `convex_baselines_ablation.slurm` | fixed `convex_baselines_ablation` | Selected shared ridge versus shared convex |
| `delta_baselines_ablation.slurm` | fixed `delta_baselines_ablation` | Selected shared ridge versus shared delta-ridge |
| `catboost_ablation.slurm` | fixed `catboost_ablation` | Selected shared-regressor gates versus three one-axis variants |
| `k_ablation.slurm` | fixed `k_ablation` | Selected winners over the K grid |
| `h_ablation.slurm` | fixed `h_ablation` | Selected winners over the H grid |
| `l_ablation.slurm` | fixed `l_ablation` | Selected winners over the L grid |
| `crossrag.slurm` | fixed `crossrag` | One selected winner versus Cross-RAG |

There is no aggregate smoke submitter. If needed, submit the `test` extraction,
then the baseline and gate smoke jobs, then tables after both complete. Final
baseline/gate comparisons use `benchmark.slurm`, which understands complete
selected pipeline names.
The `exchange_rate` key resolves the shared
`datasets/exchange_rate/exchange_rate.csv` layout used by the other projects.

The paper experiment order is:

1. Optionally validate the installation with the manual `test` stage sequence.
2. Run the complete Chronos-2 candidate screen.
3. Select complete winner names from
   `outputs/adaptation_results/screen/tables/chronos2/average/pipeline_ranking.csv`.
4. Run the final Chronos-2 benchmark with those winners.
5. Optionally run `ultra` with the same winners to add backbones.
6. Run selected-winner ablations; TS-IFA is an independent architecture track.

The optional smoke sequence is:

```bash
EXPERIMENT_MODE=test sbatch extraction.slurm
# After extraction succeeds, submit both:
EXPERIMENT_MODE=test sbatch baselines.slurm
EXPERIMENT_MODE=test sbatch gates.slurm
# After both succeed:
EXPERIMENT_MODE=test sbatch tables.slurm
```

Run the screen first:

```bash
sbatch screen.slurm
```

To deliberately regenerate every screen extraction after code changes, use:

```bash
EXTRACTION_SKIP_COMPLETE=false sbatch screen.slurm
```

The screen is the primary comparison. It intentionally excludes every convex,
delta-ridge, and per-horizon baseline, and every CatBoost formulation except
the hard shared regressor. Existing screen results may contain superseded extra
methods, but selected shared-ridge/shared-regressor pipelines remain valid;
new family comparisons belong only to their dedicated ablation profiles.

For screening, a **setting** is one dataset plus one `L:H` pair, so there are
`|D| x |S| = 12` settings. Each complete pipeline is scored by the unweighted
mean of its 12 setting-level percentage improvements over vanilla Chronos-2.
There is no averaging over K or normalization: those identify different
pipelines. The average table also writes a sorted `pipeline_ranking.csv` whose
`winner_name` includes family, retrieval configuration, and formula.

Copy the desired complete names into the final benchmark:

```bash
WINNERS_CSV=baselines/instance_euclidean_3_online/full_ridge_shared,gates/instance_euclidean_3_online/catboost_cov_regressor_shared \
sbatch benchmark.slurm
```

This defaults to `EXPERIMENT_MODE=full` and Chronos-2. To add the extra
backbones afterward without recomputing completed Chronos-2 results:

```bash
EXPERIMENT_MODE=ultra \
WINNERS_CSV=baselines/instance_euclidean_3_online/full_ridge_shared,gates/instance_euclidean_3_online/catboost_cov_regressor_shared \
sbatch benchmark.slurm
```

For an explicit custom backbone selection, pass `MODELS_CSV`, for example
`MODELS_CSV=tabpfnts`. `full` and `ultra` otherwise differ only in their
default model list.

The three baseline family studies default to the six baseline pipelines in
`SWEEP_CANDIDATES.txt`. Override `BASELINE_WINNERS_CSV` when a later screen
changes those selections. Every entry must name a primary shared-ridge method;
each launcher evaluates it beside exactly one transformed variant:

| Selected design / retrieval | Ridge control | Horizon variant | Convex variant | Delta-ridge variant |
|---|---|---|---|---|
| `full`, instance L2, `K=3` | `full_ridge_shared` | `full_ridge_horizon` | `full_convex_shared` | `full_delta_ridge_shared` |
| `y`, instance L2, `K=3` | `y_ridge_shared` | `y_ridge_horizon` | `y_convex_shared` | `y_delta_ridge_shared` |
| `cov_y`, instance L2, `K=3` | `cov_y_ridge_shared` | `cov_y_ridge_horizon` | `cov_y_convex_shared` | `cov_y_delta_ridge_shared` |
| `residual`, instance L2, `K=3` | `residual_ridge_shared` | `residual_ridge_horizon` | `residual_convex_shared` | `residual_delta_ridge_shared` |
| `cov_avgy`, raw L2, `K=3` | `cov_avgy_ridge_shared` | `cov_avgy_ridge_horizon` | `cov_avgy_convex_shared` | `cov_avgy_delta_ridge_shared` |
| `cov`, raw L2, `K=3` | `cov_ridge_shared` | `cov_ridge_horizon` | `cov_convex_shared` | `cov_delta_ridge_shared` |

```bash
sbatch horizon_baselines_ablation.slurm
sbatch convex_baselines_ablation.slurm
sbatch delta_baselines_ablation.slurm
```

The CatBoost formulation study is independent and can be submitted later:

```bash
sbatch catboost_ablation.slurm
```

Its defaults are the two selected CatBoost pipelines in `SWEEP_CANDIDATES.txt`.
Each fixes the candidate and retrieval pipeline; the runner evaluates the hard
shared regressor beside the shared classifier, shared soft-regressor mixture,
and horizon-wise regressor, plus matching references.

| Candidate / retrieval | Screen control | Classification | Mixture | Horizon |
|---|---|---|---|---|
| `avgy`, instance L2, `K=3` | `catboost_avgy_regressor_shared` | `catboost_avgy_classifier_shared` | `catboost_avgy_regressor_shared_soft` | `catboost_avgy_regressor_horizon` |
| `cov`, instance L2, `K=1` | `catboost_cov_regressor_shared` | `catboost_cov_classifier_shared` | `catboost_cov_regressor_shared_soft` | `catboost_cov_regressor_horizon` |

The mixed-quantity dataset study is also isolated and can be submitted later:

```bash
WINNERS_CSV=baselines/instance_euclidean_3_online/full_ridge_shared,gates/instance_euclidean_3_online/catboost_cov_regressor_shared \
sbatch mixed_quantity_ablation.slurm
```

All other publication profiles exclude the original ETTh1, ETTh2, ETTm1,
ETTm2, and Weather panels. Benchmark and TS-IFA full/ultra runs use the four
quantity-separated ETT panels instead.

Copy the selected complete names into `WINNERS_CSV` near the top of each later
Slurm file (or pass the same variable as an environment override):

```bash
WINNERS_CSV="${WINNERS_CSV:-baselines/instance_euclidean_3_online/full_ridge_shared,gates/instance_euclidean_3_online/catboost_cov_regressor_shared}"
```

Then submit exactly one front per experiment:

```bash
sbatch k_ablation.slurm
sbatch h_ablation.slurm
sbatch l_ablation.slurm
```

The K experiment expands each winner across the K grid while retaining its
formula and normalization. Run the formulation ablations first, filter their
outputs into the expanded candidate list, and pass that list to `k_ablation`.
Complete extraction payloads from an earlier K attempt remain reusable because
they are method-independent; fitted baseline/gate results are reused only when
their current method and training signatures match. The H and L experiments
retain each full winning pipeline, including its K and normalization, while
changing only the requested lookback/horizon axis. Dataset–`L:H` pairs remain
evaluation settings.

The Cross-RAG comparison is also one job. Enter one complete winning pipeline in
`crossrag.slurm`, configure the three released-code/checkpoint paths in the
environment, and submit it:

```bash
CROSSRAG_ROOT=/cluster/code/Cross-RAG \
CROSSRAG_BASE_CHECKPOINT=/cluster/code/Cross-RAG/cross-rag/checkpoints/base \
CROSSRAG_CHECKPOINT=/cluster/code/Cross-RAG/cross-rag/checkpoints/.../best.pth \
WINNERS_CSV=baselines/instance_euclidean_3_online/y_ridge_shared \
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
with the exact signature and timing artifact. The final benchmark also skips
only when the exact requested method set is complete and newer than its input
extraction; ultra can therefore reuse full. Ablations refit their selected
methods by default. Normal logs are under `logs/`; if a sequential job reaches
its time limit, resubmit the same mode and completed work will be reused where
its completion contract matches.

Outside the fixed Chronos-2 screen, sweep dimensions remain overridable through
`DATASETS_CSV`, `MODELS_CSV`,
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
For baseline and gate families, `average/positive_windows_results.tex` also
reports the unweighted mean of the saved per-configuration
`positive_window_pct` values. This is an absolute win-rate table, not a
relative improvement computed from pooled losses.

Use `FAMILIES_CSV=baselines`, `FAMILIES_CSV=gates`, or include `ts_ifa` after
those outputs exist.  `METRIC=mse` produces the corresponding MSE tables;
`nmse` is the default.

## TS-IFA variants and optimization

All four TS-IFA models use the same candidates. For vanilla prediction
`p in R^H`, the fixed covariate candidate is `c_c`; the trainable branches are

```text
c_r = p + delta_r(theta_r; x, p, Xc, Nc, Ec),
c_m = p + delta_m(theta_m; p, z_m),
theta_b = theta_r union theta_m.
```

Both correction heads are output-zero initialized. With
`I={c,r,m}` and `d_i=c_i-p`, every rooter has the delta form

```text
y_hat_h = p_h + sum_(i in I) a_(i,h) d_(i,h).
```

The ridge form has free `nu=(a_i) in R^(3 x H)`. The neural form has
`a_i=f_omega(z_g,p,d_i,tau_i)`, where `omega` includes the attention encoder,
candidate tokens `tau_i`, normalizers, and shared scorer. Ridge coefficients
and the neural scorer output are initialized to zero, so all four complete
models initially reproduce `p`. Neither rooter consumes handcrafted retrieval
features. Fixed and learned transformed-covariate gates are a separate project
documented in `../transformed_covariates/README.md`.

Let `L_R` be normalized rooter forecast loss plus vanilla anchoring,
coefficient L2, and first-order horizon smoothness, and let `L_B` be the sum of
residual- and memory-candidate forecast losses plus their vanilla anchors. The
joint variants optimize on all T1 by AdamW,

```text
min_(theta_b,nu)   L_R(theta_b,nu;T1)   + lambda_B L_B(theta_b;T1)  [joint_ridge]
min_(theta_b,omega)L_R(theta_b,omega;T1)+ lambda_B L_B(theta_b;T1)  [joint_neural].
```

For joint ridge, `nu` is an ordinary learned parameter updated by gradients;
the closed-form solver is never used. T2 selects the best joint checkpoint,
which is restored before the one-shot T3 evaluation. T2 is not used for a
second rooter fit in either joint variant.

The meta variants split T1 by whole query dates into earlier support `S1` and
later query `Q1`. Meta ridge solves independently at each horizon,

```text
nu*_S(theta_b)=argmin_nu ||r_S-D_S(theta_b)nu||^2+alpha||nu||^2,
r=y-p,
min_theta_b L_R(theta_b,nu*_S(theta_b);Q1)+lambda_B L_B(theta_b;Q1).
```

Gradients pass through the exact standardized `3 x 3` support solves. After
outer training, branches freeze and a final closed-form ridge is fit on all T2.
Meta neural instead takes `K_in` differentiable support updates from the
learned `omega` initialization, then updates `theta_b` and `omega` from the Q1
loss. First-order MAML is the default; `NEURAL_FIRST_ORDER=false` retains the
full higher-order graph. Its branches then freeze and only `omega` is fit on
T2. Both meta variants evaluate T3 exactly once.

`VANILLA_ANCHOR`, `COEFFICIENT_L2`, `HORIZON_SMOOTHNESS`, and
`BRANCH_AUX_WEIGHT` control the common regularizers; `RIDGE_ROOTER_ALPHA`
applies only to the meta-ridge closed-form solves. `META_QUERY_FRACTION`
controls the chronological meta split. Each run writes an isolated
`ts_ifa/<variant>/` folder with `branches.pt`, `rooter.pt`, `ts_ifa.pt`, exact
configuration/signature manifests, and disk-backed candidate, active-rooter,
and coefficient predictions.

The four smoke submissions are:

```bash
EXPERIMENT_MODE=test sbatch ts_ifa_joint_ridge.slurm
EXPERIMENT_MODE=test sbatch ts_ifa_joint_neural.slurm
EXPERIMENT_MODE=test sbatch ts_ifa_meta_ridge.slurm
EXPERIMENT_MODE=test sbatch ts_ifa_meta_neural.slurm
```

Each requires an already complete matching extraction. The launcher only
validates and reads extraction artifacts; it never reruns extraction.

## Executable files

Only the concise `.slurm` files in the project root are submitted. They contain
scheduler resources and the `EXPERIMENT_MODE` switch, while `src/slurm/*.sh` contains
enumeration, input checks, and command invocation:

- `extraction.slurm` -> `src/slurm/extract_adaptation.sh`.
- `benchmark.slurm` and `screen.slurm` ->
  `src/slurm/run_profile_experiment.sh`.
- `baselines.slurm` -> `src/slurm/run_baselines.sh`.
- `gates.slurm` -> `src/slurm/run_gates.sh`.
- `mixed_quantity_ablation.slurm` -> `src/slurm/run_profile_experiment.sh`.
- `horizon_baselines_ablation.slurm`, `convex_baselines_ablation.slurm`, and
  `delta_baselines_ablation.slurm` ->
  `src/slurm/run_baseline_family_ablation.sh`.
- `catboost_ablation.slurm` -> `src/slurm/run_catboost_ablation.sh`.
- `tables.slurm` -> `src/slurm/build_tables.sh`.
- `ts_ifa_{joint,meta}_{ridge,neural}.slurm` are the four optional TS-IFA jobs.

Implementation shells:

- `extract_adaptation.sh` builds vanilla and retrieval extraction tasks and
  calls `src.experiments.extraction`.
- `run_baselines.sh` checks extraction manifests and evaluates the explicitly
  selected direct/shared or ablation methods.
- `run_gates.sh` uses the same evaluator with `--family gates` to fit and score
  the candidate gates.
- `run_baseline_family_ablation.sh` maps selected shared-ridge winners to one
  horizon, convex, or delta-ridge counterpart and runs both in an isolated
  profile.
- `run_catboost_ablation.sh` validates selected shared-regressor winners and
  applies the classifier, soft-mixture, and horizon-regressor changes one at a
  time under their fixed retrieval pipelines.
- `run_ts_ifa.sh` dispatches the selected joint or chronological meta contract,
  reusing the matching extraction and writing one isolated active-rooter result.
- `build_tables.sh` verifies the selected sweep is complete before producing
  full and equal-configuration-average tables.
- `common.sh` provides resource lookup, setting parsing, manifest checks, and
  timestamped shell logging; it is sourced, not submitted.

The runnable Python modules are:

- `src.experiments.extraction`: frozen-backbone inference, features, neighbors,
  prediction payloads, and the atomic completion manifest.
- `src.experiments.artifacts`: command-line validation of an extraction folder.
- `src.adaptors.baselines.evaluate`: both baseline and gate families, selected
  with `--family baselines` or `--family gates`.
- `src.adaptors.ts_ifa.train`: joint T1 training with T2 checkpoint selection or
  chronological T1 support/query meta-training with a frozen-branch T2 rooter fit.
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
active-rooter coefficient heatmaps.

Every TS-IFA output must use the current result and prediction-store contracts
and include the active-rooter coefficient diagnostics. Its result manifest also
records the exact launcher training signature, so changing a bilevel split,
optimizer, regularizer, architecture, or sample cap invalidates completion.
Older outputs are incomplete and unsupported; rerun the required variant front
for every affected configuration.

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

The consolidated technical report and bibliography are
`latex/adaptation_report.tex` and `latex/adaptation_references.bib`. It records
the current notation, formulas, retrieval contract, split semantics, parameter
counts, related work, and complete experiment grid; it is the sole LaTeX model
and workflow specification. Source code, notebooks, tests, and Slurm helpers
remain under `src/`; generated artifacts stay under `outputs/`, and runtime
logs under `logs/`.
