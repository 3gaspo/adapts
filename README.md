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
changed, or legacy extraction.

Downstream output contracts are:

```text
<retrieval>/baselines/{baseline_metrics.csv,baseline_metrics.json,baseline_artifacts.pt}
<retrieval>/gates/{gate_metrics.csv,gate_metrics.json,gate_artifacts.pt}
<retrieval>/ts_ifa/TS-IFA/{eval_metrics.json,config.json,ts_ifa.pt,...}
tables/<model>/{full,average}/{baselines_results.tex,gates_results.tex,...}
```

The baseline launcher retains `--fit-baselines-on-eval`.  Methods suffixed
`_eval_fit` are optimistic T3 in-sample oracle diagnostics for the appendix;
they are intentionally excluded from the deployable main comparison.
Ridge fits accumulate float64 sufficient statistics in bounded chunks, so they
use the complete selected fitting split without materializing the full design
matrix. This changes memory use, not the fitted objective.

Baseline and gate fitting may optionally use reproducible subsets of the
already-extracted payloads through `MAX_T1_FIT_SAMPLES`,
`MAX_T2_VALID_SAMPLES`, `MAX_ADAPT_REFIT_SAMPLES`, and
`MAX_EVAL_FIT_SAMPLES`. All default to unlimited; `FIT_SAMPLE_SEED` defaults to
`SEED`. The first three limits affect only T1 fitting, T2 validation, and the
final T1+T2 refit respectively. The T3 maximum applies only to explicitly
optimistic `_eval_fit` methods. Final T3 scoring always uses every evaluation
sample. Legacy `MAX_TRAIN_FIT_SAMPLES` and `MAX_ORACLE_FIT_SAMPLES` are accepted
as aliases for the first two controls.

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

`CHRONOS_WEIGHTS_PATH` and `TABPFN_WEIGHTS_PATH` can override individual model
paths. The full sweep is Chronos-only; ultra contains `chronos` and `tabpfnts`.
TS-ICL is documented as a later extension and is rejected by the launcher until
it is implemented and registered.

Dataset directories may contain a sibling `config.json`. It is discovered by
the Python loader even for direct runs; `--dataset-config` accepts an explicit
JSON file or directory. Portable fields such as `drop_users`, `date_col`, and
aggregation settings live at the top level. Adaptation-only values belong under
`adaptation` (`ts_ifa` remains a supported legacy alias). Project-scoped values
override other settings, while `drop_users` is merged additively with both the
top-level list and `--drop-users`. The loader logs the selected path and applied
keys.

ETTh1 is configured to select only the `OT` target column in every profile.

The repository tracks the curated Electricity `config.json` while leaving its
CSV ignored, so the same exclusions—including source column 245—are carried to
cluster checkouts and shared with RevIN.

## Experiment profiles and required order

Every root launcher accepts the same `EXPERIMENT_MODE=test|small|full|ultra`
switch (`large` remains a compatibility alias for `full`):

- `test` is the existing Electricity 168:24 Chronos smoke profile with raw
  distance, `k=3`, sparse queries, and reduced fitting/training budgets.
- `small` reproduces the RevIN subset: Traffic, Electricity, and Solar with
  `168:24`, `504:24`, `504:168`, and `504:504`; raw and instance distance;
  `k in {1,3,10}`; and Chronos. Extraction has 84 configurations; baselines,
  gates, and TS-IFA have 72 each.
- `full` uses ETTh1 (OT only), Electricity, Traffic, Solar, Weather, and
  Exchange; the four small settings plus the Cross-RAG `512:64` comparison;
  and Chronos. Extraction has 210 configurations; baselines, gates, and TS-IFA
  have 180 each.
- `ultra` adds TabPFN-TS to the full profile. Extraction has 420
  configurations; baselines, gates, and TS-IFA have 360 each.

Start with the test DAG. Each submission is one sequential Slurm job and creates
one `.out` and one `.err` file:

```bash
extract_test=$(EXPERIMENT_MODE=test sbatch --parsable extraction.slurm)

baseline_test=$(EXPERIMENT_MODE=test sbatch --parsable \
  --dependency=afterok:$extract_test \
  baselines.slurm)

gate_test=$(EXPERIMENT_MODE=test sbatch --parsable \
  --dependency=afterok:$extract_test \
  gates.slurm)

EXPERIMENT_MODE=test sbatch --dependency=afterok:$baseline_test:$gate_test \
  tables.slurm
```

Inspect the Slurm logs, extraction manifests, downstream JSON/CSV metrics,
feature-importance plots, and the Chronos full/average test tables. Then run the
Chronos production grid:

```bash
extract_small=$(EXPERIMENT_MODE=small sbatch --parsable extraction.slurm)
baseline_small=$(EXPERIMENT_MODE=small sbatch --parsable \
  --dependency=afterok:$extract_small \
  baselines.slurm)
gate_small=$(EXPERIMENT_MODE=small sbatch --parsable \
  --dependency=afterok:$extract_small \
  gates.slurm)
EXPERIMENT_MODE=small sbatch --dependency=afterok:$baseline_small:$gate_small \
  tables.slurm
```

After checking the reproduction subset, repeat the same DAG with
`EXPERIMENT_MODE=full`. Use `EXPERIMENT_MODE=ultra` only for the multi-backbone
extension.
Extraction defaults to `SKIP_COMPLETE=true` and validates a complete manifest
and exact extraction signature, so ultra skips matching Chronos
payloads and computes the new TabPFN-TS runs. Baselines, gates, and TS-IFA also
default to completion skipping in small/full/ultra mode when all expected files are
newer than the matching extraction manifest. Set `SKIP_COMPLETE=false` after
changing downstream hyperparameters. Tables are always rebuilt for the selected
profile. The `512:64` setting is the Cross-RAG comparison. If a sequential
job exceeds its time limit, resubmit the same mode; split first by model and then
dataset only when needed.

Normal timestamped progress and Python warnings are written to
`logs/<job>_<job-id>.out`. Third-party progress bars are disabled, leaving the
matching `.err` for scheduler, shell, or Python failures.

The screening sweep is intentionally single-seed (`SEED=1`). `SEED` is not part
of the output directory, so never submit different seeds against the same
`OUT_ROOT`: they would replace one another. For exploratory repeats, use one
root per seed (for example `OUT_ROOT=outputs/adaptation_seed_2`) consistently
for extraction and its downstream jobs. The current table builder averages
configurations, not seeds; seed aggregation must be added before presenting a
multi-seed adaptation result.

All sweep dimensions have comma-separated environment overrides:
`DATASETS_CSV`, `MODELS_CSV`, `SETTINGS_CSV`, `DISTANCE_SPACES_CSV`, and
`NEIGHBORS_CSV`. Settings use `L:H`. Extraction loops over
`D*M*S*(1 + spaces*k)` configurations; baselines/gates/TS-IFA loop over
`D*M*S*spaces*k` configurations. For example:

```bash
DATASETS_CSV=Electricity MODELS_CSV=chronos SETTINGS_CSV=168:24 \
DISTANCE_SPACES_CSV=raw NEIGHBORS_CSV=3 \
EXPERIMENT_MODE=small sbatch extraction.slurm
```

Fit-only sample maxima can be applied to one selected configuration without
re-extraction. For example, this caps only the T1 baseline fit while retaining
full T3 evaluation:

```bash
DATASETS_CSV=Traffic SETTINGS_CSV=504:504 \
DISTANCE_SPACES_CSV=raw NEIGHBORS_CSV=10 \
MAX_T1_FIT_SAMPLES=50000 FIT_SAMPLE_SEED=1 \
SKIP_COMPLETE=false EXPERIMENT_MODE=small sbatch baselines.slurm
```

Use `MAX_T2_VALID_SAMPLES` to cap model-local validation and
`MAX_ADAPT_REFIT_SAMPLES` to cap the T1+T2 refit. Use
`MAX_EVAL_FIT_SAMPLES` only to limit the optimistic appendix fits.

Do not submit a downstream job without an `afterok` dependency unless the
corresponding manifests have already been checked.  Downstream launchers fail
before computation when the required extraction is absent, partial, stale, or
legacy, and assert the files expected by table discovery after each run.

## Tables and averages

The only table front end is `tables.slurm`; it delegates to
`src/slurm/build_tables.sh`. It
checks every selected input rather than silently constructing a sparse table,
then writes separate Chronos and TabPFN-TS tables.  `full/` reports each
dataset/setting/retrieval result.  `average/` gives the unweighted mean over the
selected configuration-level metrics and the relative improvement from the
matching vanilla backbone.  This equal-configuration average prevents large
datasets from dominating merely because they contain more windows.  Report it
alongside, not instead of, per-dataset results and user-tail analyses.

Use `FAMILIES_CSV=baselines`, `FAMILIES_CSV=gates`, or include `ts_ifa` after
those outputs exist.  `METRIC=mse` produces the corresponding MSE tables;
`nmse` is the default.

## TS-IFA status and quick overfitting controls

The current architecture forms vanilla, context, residual-attention, and
memory-attention candidates, then learns a separate four-way mixture at every
horizon.  That final horizon-wise gate is expressive enough to overfit, and the
old training loop always evaluated the last step on T3 even when an earlier T2
checkpoint was better.

The trainer now supports `--restore-best-validation`,
`--early-stopping-patience`, `--early-stopping-min-delta`, and separate
`--max-valid-samples`.  The Slurm launcher restores the best T2 checkpoint by
default while leaving early stopping disabled (`EARLY_STOPPING_PATIENCE=0`) so
the historical optimization length remains reproducible.  A first controlled
tuning run should compare:

- best-checkpoint restoration alone;
- patience 5-10 T2 evaluations;
- dropout 0.1 versus 0;
- half-sized attention/MLP dimensions before increasing capacity.

Keep these comparisons on one dataset/setting/retrieval seed until the T2 curve
is stable.  The next architecture experiment should replace the horizon-wise
final mixture with a scalar/shared or low-rank gate, or regularize its departure
from the vanilla weight.  That is a substantive model change and has not been
made without evidence.  Baseline/gate results can proceed independently.

The default `MIXTURE_GATE_INIT=-6` gives each non-vanilla branch only about
0.25% initial mixture weight. Auxiliary branch losses still provide gradients,
but such a small weight can slow coupling between the branches and the final
gate. After establishing best-T2 restoration and early stopping, sweep initial
logits such as `-6`, `-3`, and `-1` on the same pilot configuration.

TS-IFA smoke submission:

```bash
EXPERIMENT_MODE=test sbatch ts_ifa.slurm
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
- `run_ts_ifa.sh` trains TS-IFA with validation checkpoint selection and writes
  T3 metrics.
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
- `src.adaptors.ts_ifa.train`: TS-IFA training, T2 selection, and T3 evaluation.
- `src.visu.sweep_results_table`: full and averaged publication tables.
- `src.visu.results_table` and `src.visu.selected_methods`: focused table
  utilities for individual result folders and selected method subsets.
- `src.visu.dashboard`: interactive retrieval diagnostics. Library modules such
  as `features.py`, `runtime.py`, `models/*.py`, and `data/*.py` support these
  entry points and are not separate jobs.

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
