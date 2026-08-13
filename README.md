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

## Result identity and manifests

Extraction is an independently reusable workflow. Its ordered identity is

```text
outputs/extraction/dataset/L_H/backbone/space/metric/k/mode/run_n/
```

The vanilla extraction uses `none/none/0/none`. Each completed run contains the
adapt/eval prediction and feature payloads plus the scientific extraction
manifest. Independently submitted downstream jobs select an explicit completed
extraction manifest; they never infer one from a similarly named directory.
Within one multi-stage Slurm launch, a later stage may select a ready manifest
only when it carries that same launch ID.

Every independently submitted comparison or ablation owns a workflow root
below `outputs/adaptation/`, for example `baselines`, `gates`, `benchmark`,
`screen`, `k_ablation`, `crossrag`, or `tsrag`. Baseline, gate, Cross-RAG, and
TS-RAG identities
are

```text
outputs/adaptation/<family>/dataset/L_H/backbone/formula/space/metric/k/mode/run_n/
```

TS-IFA identities are

```text
outputs/adaptation/<family>/dataset/L_H/backbone/
  variant/routing_scope/routing_constraint/branch_set/space/metric/k/mode/run_n/
```

Every model config has its own directory component in the declared order.
Train/validation split, iteration or epoch budget, optimizer, regularization,
fit caps, candidate list, and evaluation controls are pipeline configs in
`run_n/manifest.json`; device, worker, and scheduler placement are runtime
configs. The project currently uses a single manifest-recorded seed per run,
and that seed fixes every stochastic choice. Future repeated seeds use
`seed_n/` leaves under the run.

Run identity contains only the manifest schema, ordered identity/model configs,
pipeline and experiment parameters, and seeds. Source files, Slurm fronts,
datasets, weights, logs, outputs, checkpoints, and directories are never
fingerprinted or hashed. Plain upstream-manifest and checkpoint paths may be recorded for
provenance but do not affect reuse. Code and data changes are manual rerun
decisions; use `RUN_CONFLICT_POLICY=new` for another repeat with unchanged
parameters. Change `schema_version` only for a deliberate global
artifact-contract break.

The shared contract is `schema_version: 1` with status `not_run`, `running`,
`interrupted`, or `completed`. The default `RUN_CONFLICT_POLICY=overwrite_exact`
skips an identical completed run, resumes an identical interruption, and
allocates the next `run_n` when pipeline config differs. `overwrite_path`
explicitly replaces the latest path occupant; `new` always adds a run. A forced
exact overwrite preserves the prior manifest in `manifest_history/`.
Before a non-skipped attempt is marked running, the schema layer clears stale
generated artifacts without deleting `manifest.json`, `manifest_history/`, or
completed `seed_n/` leaves. Artifact writers must never delete the allocated
`run_n/` root. A manifest-less `run_n/` is an invalid partial directory, not a
current run; the next allocation safely reclaims its index and recomputes it.

`EXPERIMENT_MODE=test|full|ultra` selects delayed subsets of the same identity
trees; it is not a family, phase, path component, or computation-signature
field. Reports live under `outputs/reports/<family>/<mode>/`. They default to
distinct pipeline configs and the selected exact repeat, with
`TABLE_CONFIG_POLICY=distinct|latest|average` and
`TABLE_REPEAT_POLICY=selected|latest|distinct|average`. Explicit pipeline
filters select a pipeline configuration and must match even when only one run
exists. `SELECTED_RUNS.txt` stores only the automatic or pinned exact repeat per
pipeline signature. Every report records the requested filters and the input
manifests actually obtained.

The former `outputs/extractions` and `outputs/adaptation_results` trees could
not be migrated faithfully: synchronized extraction tensor payloads were
missing, and several result folders bundled formulas without enough pipeline or
launch evidence. They are preserved under
`archive/legacy_pre_schema_v1_2026-08-07/`, excluded from all current
readers, and may be consulted only for legacy analysis.

Baseline, gate, and TS-IFA predictions use the sole current disk-backed
`prediction_manifest.json` contract. Each array is a separate `.npy` file below
`predictions/`; the dashboard reads only this contract. `result_manifest.json`
is written last as a family-specific scientific artifact. The overall run
briefly remains `running` with `ready_at_utc` while its finished seed state is
`ready`; immediately after that producer's `srun` returns successfully, the
launcher records overall completion. A later configuration or table failure
therefore preserves completed producers and interrupts only unfinished work.
Ready runs can feed later stages of that same active workflow during the brief
handoff, but are not eligible for external reuse. Obsolete or partial result
folders are not accepted.
Gate runs also index their per-model CatBoost feature-importance CSV/PNG files
from `gate_artifacts.json`. Every TS-IFA variant stores its T3 active-rooter
coefficients under the prediction store's `gate_diagnostics` kind and its
rooter state in `rooter.pt`. Cross-RAG likewise writes its scientific
`result_manifest.json` before the shared run manifest is marked completed.
Seed completion never promotes the overall run. Once completed, the manifest
is authoritative; reuse does not hash or revalidate synchronized files.

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
`MAX_EVAL_FIT_SAMPLES`. All default to unlimited. The single run `SEED`
determines every subsample and model random state; deterministic derived
substreams are not separately configurable. The first three limits affect only T1 fitting, T2 validation, and the
final T1+T2 refit respectively. The T3 maximum applies only to explicitly
optimistic `_eval_fit` methods. Final T3 scoring always uses every evaluation
sample.

For period-aligned retrieval, neighbor query dates `r_j` satisfy
`(s-r_j) mod P = 0`. In fixed mode, both the neighbor lookback and future lie
inside T0. In online mode, `r_j+H <= s`, so the complete retrieved future is
already observable at query date `s`. A neighbor future may overlap the
observed query lookback but can never overlap the query target.

The extraction launcher resolves `P=96` for `ETTm1`, `ETTm2`, `ETT_T_15T`,
and `ETT_L_15T`, and `P=24` otherwise. Publication profiles use datastore,
pooled T1+T2 query, and untouched T3 query strides `25/25/127`, respectively,
with period alignment disabled. These strides are coprime with the 24-step
hourly and 96-step quarter-hourly periods, so sampled origins rotate through
every seasonal phase. The test profile retains its independent aligned
`168/256/256` smoke strides. T1 and T2 share the pooled adaptation stride
because their chronological boundary is assigned downstream by whole query
dates. Explicit `PERIOD`, `DATASTORE_STRIDE`, and `ALIGN_PERIOD` overrides
remain available; an aligned datastore stride must be a multiple of `P`.
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

The standard model folders are `weights/chronos2/`,
`weights/chronos-bolt-base/`, and `weights/tabpfnts/`. Cross-RAG additionally
expects `weights/cross-rag/` to contain exactly one Cross-RAG-trained
`best.pth` recursively. The official repository's linked Drive currently
contains the released TS-RAG `checkpoints/chronos-bolt/best.pth` (about 801.5
MB). Put that file below `weights/ts-rag/` for `tsrag.slurm`; its retrieval head
is strict-loaded by the local TS-RAG source and must never be used as Cross-RAG
weights. Cross-RAG is now represented only by the paper values recorded in
`SOTA_BENCHMARK.json`; no Cross-RAG checkpoint is required by a Slurm front.
`CHRONOS2_WEIGHTS_PATH`,
`CHRONOS_BOLT_WEIGHTS_PATH`,
`CROSSRAG_WEIGHTS_PATH`, `TSRAG_WEIGHTS_PATH`, and `TABPFN_WEIGHTS_PATH`
override the corresponding locations. The model names are `chronos2`,
`chronos-bolt`, and `tabpfnts`.
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
  `K in {1,3,5,10,15,20,100}` while retaining each formula and normalization.
- `h_ablation`: manually named pipelines with `L=504` and
  `H in {24,168,504}`.
- `l_ablation`: manually named pipelines with `H=24` and
  `L in {24,168,504}`.
- `sota_benchmark`: evaluates one selected project method with Chronos-Bolt at
  `L=512`, `H=64` on the seven exact Cross-RAG paper test splits, then combines
  the computed MSE with the published Cross-RAG and TS-RAG rows from
  `SOTA_BENCHMARK.json`. The selected method keeps its own training/retrieval
  protocol; only evaluation is aligned.
- `tsrag`: source-faithful inference with the released TS-RAG ARM checkpoint at
  `L=512`, `H=64`, using its default Chronos-T5 EOS embeddings, Euclidean
  same-channel retrieval, and `K=10`. The selected project method is evaluated
  on the same neighbors with both Chronos-Bolt and Chronos-2; TS-RAG itself is
  Chronos-Bolt-only because that is the checkpoint architecture.

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
| `k_ablation` | `D_primary` | Screen settings | Chronos-2 | Winner's space/metric / `1,3,5,10,15,20,100` | `WINNERS_CSV` only |
| `h_ablation` | `D_primary` | `504:24`, `504:168`, `504:504` | Chronos-2 | Selected pipeline / `1` or `3` | `WINNERS_CSV` only |
| `l_ablation` | `D_primary` | `24:24`, `168:24`, `504:24` | Chronos-2 | Selected pipeline / `1` or `3` | `WINNERS_CSV` only |
| `sota_benchmark` | Seven paper datasets | `512:64` | Chronos-Bolt | Selected method's own retrieval | Computed selected method plus published TS-RAG/Cross-RAG MSE |
| `tsrag` | `D_primary` | `512:64` | Chronos-2, Chronos-Bolt | Chronos-T5 Euclidean same-channel / `10` | Selected method on both backbones; released TS-RAG ARM on Bolt |

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

Modes provide delayed subsets of experiment identities. `test` is the narrow
Electricity smoke subset, `full` is the complete one-backbone subset, and
`ultra` adds the configured backbones. Independently submitted studies such as
`screen`, `crossrag`, and every ablation are families, not modes. Full defaults
to `chronos2`, ultra to `chronos2,tabpfnts`, and the Cross-RAG family fixes the
two-backbone sequence `chronos2,chronos-bolt`.
`MODELS_CSV` explicitly overrides a model list where the launcher enumerates
backbones, for example `MODELS_CSV=chronos2,tabpfnts`. The primary screen is
defined on Chronos-2 and should retain that reference backbone. Every
extraction already writes the matching vanilla backbone forecast and metrics;
there is no separate univariate submission.

`test` is exactly one Electricity setting and one retrieval pipeline:
`L=504`, `H=168`, raw-space Euclidean online retrieval, and `K=3`. `screen`
is the family that performs the complete candidate sweep. `full` is the final selected-candidate
benchmark on `D_full` and the same three default settings; the selected winner
name already fixes formula, raw/instance normalization, metric, K, and online
or fixed retrieval. `ultra`
uses that identical path set and adds the additional default backbones. Matching
full computations are reused because mode does not alter their identities.

The intended submission interface is:

| Slurm front | Accessible mode(s) | What it runs |
|---|---|---|
| `screen.slurm` | `full` | `screen` family: primary extraction, 10 baselines, 8 gate/reference methods, and tables |
| `benchmark.slurm` | `full` or `ultra` | Selected screen winners on `D_full`, followed by tables |
| `extraction.slurm` | `test`, `full`, `ultra` | Standalone extraction, mainly for smoke or TS-IFA inputs |
| `baselines.slurm` | `test` | Primary baseline smoke run; requires test extraction |
| `gates.slurm` | `test` | Primary gate smoke run; requires test extraction |
| `tables.slurm` | `test`, `full`, `ultra` | Standalone tables, including optional TS-IFA tables |
| `ts_ifa.slurm` | `test`, `full`, `ultra` | `ts_ifa` family: complete non-meta configurable TS-IFA grid over the selected scale subset |
| `ts_ifa_h_ablation.slurm` | `full` | `ts_ifa_h_ablation` family at `L=504`, over `H=24,168,504` |
| `ts_ifa_l_ablation.slurm` | `full` | `ts_ifa_l_ablation` family at `H=24`, over `L=24,168,504` |
| `ts_ifa_meta_ridge.slurm` | `full` by default | Meta-learning for textually selected joint-ridge candidates |
| `ts_ifa_meta_neural.slurm` | `full` by default | Meta-learning for textually selected joint-neural candidates |
| `mixed_quantity_ablation.slurm` | `full` | `mixed_quantity_ablation` family on original ETT panels and Weather |
| `horizon_baselines_ablation.slurm` | `full` | separate family: selected shared ridge versus horizon ridge |
| `convex_baselines_ablation.slurm` | `full` | separate family: selected shared ridge versus shared convex |
| `delta_baselines_ablation.slurm` | `full` | separate family: selected shared ridge versus shared delta-ridge |
| `catboost_ablation.slurm` | `full` | separate family: selected shared-regressor gates versus three one-axis variants |
| `k_ablation.slurm` | `full` | `k_ablation` family over the K grid |
| `h_ablation.slurm` | `full` | `h_ablation` family over the H grid |
| `l_ablation.slurm` | `full` | `l_ablation` family over the L grid |
| `sota_benchmark.slurm` | `full` | exact paper test splits: selected project method on Chronos-Bolt plus static published TS-RAG/Cross-RAG MSE rows |
| `tsrag.slurm` | `full` | released TS-RAG defaults at K=10, plus the selected project method on Chronos-Bolt and Chronos-2 |

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
   `outputs/reports/screen/full/chronos2/average/pipeline_ranking.csv`.
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

The SOTA benchmark evaluates one selected project method and then appends the
paper values without executing Cross-RAG or TS-RAG:

```bash
WINNERS_CSV=baselines/instance_euclidean_10_online/full_ridge_shared \
sbatch sota_benchmark.slurm
```

`SOTA_BENCHMARK.json` is the sole static source for the published rows and the
exact dataset protocol. The launcher uses the ETT repositories' fixed
12/4/4-month boundaries, the custom datasets' 70/10/20 boundaries, every
eligible test origin, and per-channel standardization fitted on the official
training segment. Our selected method retains its own fitting and retrieval
protocol. The generated `sota_benchmark_mse.csv/.tex` therefore answers the
intended question: how our trained method scores on the same evaluation data
and metric, without claiming that it used Cross-RAG's training recipe.

The released TS-RAG ARM is re-evaluated separately on the project datasets:

```bash
sbatch tsrag.slurm
```

The front fixes `L=512`, `H=64` because those shapes are embedded in the public
ARM checkpoint. It uses the TS-RAG repository defaults: locally downloaded
`amazon/chronos-t5-base` EOS embeddings, Euclidean distance, same-channel fixed
retrieval, and `K=10`. It evaluates released TS-RAG on Chronos-Bolt and the
selected project method on the identical neighbors with both Chronos-Bolt and
Chronos-2. A Chronos-2 TS-RAG row is intentionally absent: the public ARM
checkpoint contains Chronos-Bolt decoder/head parameters and cannot be loaded
into Chronos-2.

## Fair comparison with the Cross-RAG paper

Current `D_primary` T3 results cannot be placed directly beside the paper's
table. Training may differ, provided it is disclosed; direct numerical
comparison requires the following evaluation contract:

- the seven original multivariate panels `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`,
  `Weather`, `Electricity`, and `Exchange`, retaining every channel;
- the official fixed ETT boundaries (12 months train, 4 validation, 4 test)
  and `7:1:2` for Weather, Electricity, and Exchange;
- every eligible rolling test origin, without evaluation stride subsampling;
- `L=512`, `H=64`, the same released Chronos-Bolt backbone, its median (`0.5`)
  quantile forecast, and the repository's train-fitted standardization;
- pooled element-wise MSE on the standardized repository test windows; project
  nMSE and positive-window metrics may be supplementary only.

The Cross-RAG paper reports an equal-dataset mean MSE of `0.191` for Cross-RAG,
`0.197` for its TS-RAG comparator, and `0.201` for standalone Chronos-Bolt.
Those numbers become a fair external evaluation comparison after the contract
above is reproduced; method-specific training differences remain visible in
the row labels and accompanying text. Its paper and the two source repositories are
the authoritative references:
[Cross-RAG paper](https://arxiv.org/pdf/2603.14709),
[Cross-RAG source](https://github.com/seunghan96/cross-rag), and
[TS-RAG source](https://github.com/UConn-DSIS/TS-RAG).

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
`src/slurm/build_tables.sh`. It checks every selected input rather than silently constructing a sparse table,
loads baseline, gate, TS-IFA, and Cross-RAG metrics only through their current
result manifests, rejects obsolete TS-IFA directories, and validates complete
dataset/setting/model coverage before creating a table directory. Internally,
selected pipelines retain their family-qualified
`<family>/<retrieval>/<method>` names so identically named baseline and gate
rows cannot be confused. It then writes separate Chronos and TabPFN-TS tables.
`full/` reports each
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

Every table-stage run also reads each selected current
`baseline_artifacts.pt` and writes the exact signed fitted coefficients as CSV
and an `imshow` heatmap under `tables/<model>/coefficients/`. Shared baselines
produce a one-row heatmap; horizon baselines produce one row per forecast
horizon. Repeated neighbor signals are expanded as `Y_1,...,Y_K` and
`N_1,...,N_K`. Convex baselines show their simplex weights. Direct baselines
have no fitted parameters and are omitted. `coefficient_index.csv` indexes all
generated coefficient files.

The K-ablation table stage additionally writes
`k_ablation_average_<metric>_improvement.{csv,png,pdf}` under the model's
`average/` table directory. The figure places every selected baseline/gate
candidate on the same logarithmic K axis, reports average held-out improvement
over vanilla, and marks the globally best candidate/K point. It requires the
complete configured K grid before writing the plot.

Completed extraction, baseline, and gate configurations are reused by default
for every profile and ablation front. Resubmitting `screen.slurm`, a baseline
family ablation, or K/H/L ablation therefore skips current signature-matching
computations and reruns its table stage, including coefficient plots. Set
`SKIP_COMPLETE=false` only to force fitted results to be regenerated.

For a TS-IFA-only table, name the completed current variants explicitly, for
example:

```bash
EXPERIMENT_MODE=full FAMILIES_CSV=ts_ifa \
  METHODS_CSV=joint_ridge_shared_softmax_cov \
  DATASETS_CSV=Electricity DISTANCE_SPACES_CSV=raw NEIGHBORS_CSV=3 \
  sbatch tables.slurm
```

## TS-IFA variants and optimization

TS-IFA now has a configurable candidate set. Vanilla is always present and any
non-empty subset of `cov`, `residual`, and `memory` can be enabled. For vanilla
prediction `p in R^H`, the fixed covariate candidate is `c_c`; the learned
branches are

```text
c_r = p + delta_r(theta_r; x, p, Xc, Nc, Ec),
c_m = p + delta_m(theta_m; p, z_m),
theta_b = theta_r union theta_m.
```

Both correction heads are output-zero initialized. With active set `I` and
`d_i=c_i-p`, unconstrained routing has the signed-delta form

```text
y_hat_h = p_h + sum_(i in I) a_(i,h) d_(i,h).
```

Softmax routing instead assigns non-negative weights to vanilla and every
active candidate, constrained to sum to one. Routing can be shared across all
horizon indexes (`a_i`) or horizon-specific (`a_(i,h)`). The ridge form has
free routing values; the neural form has
`a_i=f_omega(z_g,p,d_i,tau_i)`, where `omega` includes the attention encoder,
candidate tokens `tau_i`, normalizers, and shared scorer. Ridge coefficients
and neural outputs are initialized at the vanilla solution: zero deltas for
unconstrained routing and low non-vanilla logits for softmax routing. Neither
rooter consumes handcrafted retrieval
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

For unconstrained ridge routing, gradients pass through the exact standardized
support solves. Softmax ridge uses differentiable inner gradient updates because
the simplex mapping is nonlinear. After outer training, branches freeze; the
unconstrained ridge router is fit exactly on all T2, while neural and softmax
routers are fit by gradient updates.
Meta neural instead takes `K_in` differentiable support updates from the
learned `omega` initialization, then updates `theta_b` and `omega` from the Q1
loss. First-order MAML is the default; `NEURAL_FIRST_ORDER=false` retains the
full higher-order graph. Its branches then freeze and only `omega` is fit on
T2. Both meta variants evaluate T3 exactly once.

`VANILLA_ANCHOR`, `COEFFICIENT_L2`, `HORIZON_SMOOTHNESS`, and
`BRANCH_AUX_WEIGHT` control the common regularizers; `RIDGE_ROOTER_ALPHA`
applies only to the meta-ridge closed-form solves. `META_QUERY_FRACTION`
controls the chronological meta split. Each run writes an isolated
`ts_ifa/<method>/` folder with `branches.pt`, `rooter.pt`, `ts_ifa.pt`, exact
configuration/signature manifests, and disk-backed candidate, active-rooter,
and coefficient predictions.

All TS-IFA training stages use 20,000 steps. The complete non-meta grid is
`ridge/neural x shared/horizon x unconstrained/softmax x raw/instance x`
`{cov, residual, memory, full}`, where the branch labels mean vanilla plus the
named branch(es). Run its 64 Electricity pilot configurations with:

```bash
sbatch ts_ifa.slurm
EXPERIMENT_MODE=full sbatch ts_ifa.slurm
```

After inspecting the grid, copy chosen entries in the format documented by
`TS_IFA_CANDIDATES.txt` into `TS_IFA_CANDIDATES_CSV`. The H/L and meta fronts
accept only that explicit selection, for example:

```bash
TS_IFA_CANDIDATES_CSV='ts_ifa/raw_euclidean_3_online/joint_ridge_shared_softmax_cov' \
  sbatch ts_ifa_h_ablation.slurm
TS_IFA_CANDIDATES_CSV='ts_ifa/raw_euclidean_3_online/joint_ridge_shared_softmax_cov' \
  sbatch ts_ifa_meta_ridge.slurm
```

Every front requires already complete matching extractions. The launcher only
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
- `ts_ifa.slurm`, `ts_ifa_{h,l}_ablation.slurm`, and
  `ts_ifa_meta_{ridge,neural}.slurm` dispatch the configurable TS-IFA studies.

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
- `run_ts_ifa.sh` enumerates the complete non-meta grid or dispatches textually
  selected ablation/meta candidates, reusing the matching extraction and writing
  one isolated method result.
- `build_tables.sh` verifies the selected sweep is complete before producing
  full and equal-configuration-average tables.
- `common.sh` provides resource lookup, setting parsing, manifest checks, and
  timestamped shell logging; it is sourced, not submitted.
- `publish_job.sh` is run manually after Slurm jobs terminate; with a job ID it
  publishes only that job, and without one it publishes all logs and lightweight
  outputs.

The runnable Python modules are:

- `src.experiments.extraction`: frozen-backbone inference, features, neighbors,
  prediction payloads, and the atomic completion manifest.
- `src.experiments.artifacts`: command-line validation of an extraction folder.
- `src.adaptors.baselines.evaluate`: both baseline and gate families, selected
  with `--family baselines` or `--family gates`.
- `src.adaptors.cross_rag.evaluate`: Cross-RAG checkpoint inference on the fixed
  Chronos-Bolt `512:64`, min-max/cosine, `K=15` extraction payload. This source
  remains for architecture inspection but has no current Slurm front.
- `src.adaptors.ts_rag.evaluate`: strict TS-RAG checkpoint inference on a
  declared-`K` Chronos-Bolt `512:64` extraction payload; `tsrag.slurm` fixes
  Chronos-T5/Euclidean/same-channel retrieval to the repository default `K=10`.
- `src.visu.sota_benchmark_table`: combines computed exact-test-split MSE with
  immutable published rows from `SOTA_BENCHMARK.json`.
- `src.visu.tsrag_comparison_table`: tabulates the project-dataset TS-RAG run
  and its Chronos-Bolt/Chronos-2 controls.
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

## Publishing terminal Slurm artifacts

Slurm jobs never submit a publisher or run Git commands. After any job reaches
a terminal state, including failure, cancellation, or timeout, run the manual
publisher from that project's Git root:

```bash
bash publish_job.sh <job-id>
```

The script selects exactly one `logs/*_<job-id>.out`/`.err` pair and every
run/report directory whose manifest records that launch ID. It force-adds only
those paths while excluding `*.pt`, `*.npy`, and `*.cbm`, commits them on
`main`, sources `$HOME/codes/proxy.sh`, and pushes `origin main`. It never pulls
or creates a pull request. Existing unrelated staged paths are excluded from
the commit.

Omit the job ID to force-add, commit, and push the complete `logs/` and
lightweight `outputs/` trees:

```bash
bash publish_job.sh
```

`PROXY_SCRIPT_PATH` overrides the default `$HOME/codes/proxy.sh`. The publisher
simply sources that script in the current interactive shell and then runs
`git push origin main`; it leaves the shell's existing GitHub credential and
askpass context untouched.

## Local checks

Full extraction and model inference run only on the remote cluster.  With the
user-prepared project environment, lightweight checks are:

```bash
python src/tests/smoke/check_extraction_manifest.py
python src/tests/smoke/check_loads.py
python src/tests/test_crossrag_model_contract.py
python src/tests/test_tsrag_model_contract.py
python src/tests/smoke/check_baseline_oracles.py
python src/tests/smoke/check_ts_ifa_training.py
python src/tests/smoke/check_results_table.py
python src/tests/smoke/check_sweep_results_table.py
python src/tests/smoke/check_baseline_coefficients.py
python src/tests/smoke/check_k_ablation_plot.py
python src/tests/smoke/check_retrieval_dashboard.py
```

`latex/experiment_guideline.tex` records the current notation, formulas,
retrieval contract, split semantics, parameter counts, related work, complete
experiment grid, and practical workflow. `latex/executive_summary.tex` records
only synchronized and analyzed results, their limitations, and current
decisions. Their PDFs are kept beside the sources; the guideline uses
`latex/adaptation_references.bib`. Source code, notebooks, tests, and Slurm
helpers remain under `src/`; generated artifacts stay under `outputs/`, and
runtime logs under `logs/`.

## Maintenance workflow

Every project change is recorded in `PENDING_UPDATES.md` with its scope,
affected contracts, focused checks already completed, deferred integration
coverage, documentation impact, and rerun requirements. Routine edits use only
the smallest relevant smoke check. Periodic maintenance verifies pending entries
against the implementation, runs complementary generic lightweight smoke tests,
reconciles this README and the project LaTeX documents, and renders affected
PDFs before resolving the entries.
