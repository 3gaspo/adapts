# Forecast adaptation and TS-IFA

This project studies retrieval-based adaptation of frozen time-series
forecasters.  The current paper-first path is to establish strong direct
retrieval baselines and learned gates; TS-IFA remains an architecture-tuning
track until its T2/T3 overfitting is controlled.

## Protocol and outputs

The chronological protocol is fixed throughout the project:

- T0 is the retrieval datastore.
- T1 fits ridge/scalar baselines and trains TS-IFA.
- T2 fits gates and selects TS-IFA checkpoints.
- T3 is untouched final evaluation.  Nothing fitted on T3 belongs in the main
  comparison.

Extraction writes to
`outputs/adaptation/<dataset>/<L>_<H>/<model>/<retrieval>/extracted/`.
A usable extraction contains train/oracle/eval prediction and feature payloads
plus `extraction_manifest.json`.  The manifest is written atomically only after
all payloads exist and records the exact extraction signature and file sizes.
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

## Data and weight locations

Submit from the project root.  Launchers search, in order, the project-local
folder (`datasets/` or `weights/`), the project parent, and the thesis workspace
shared root.  The first folder containing the requested dataset or weight is
used.  When the repository is copied elsewhere, explicitly set the roots:

```bash
DATA_ROOT=/cluster/shared/datasets \
WEIGHTS_ROOT=/cluster/shared/weights \
sbatch src/slurm/extract_multi_backbone_adaptation.slurm
```

`CHRONOS_WEIGHTS_PATH` and `TABPFN_WEIGHTS_PATH` can override individual model
paths.  The active multi-backbone grid contains `chronos` and `tabpfnts` only.
TS-ICL is documented as a later extension and is rejected by the launcher until
it is implemented and registered.

## Required run order

First run the two-task extraction smoke grid (vanilla plus one raw, k=3
retrieval), then its one-task baseline and gate consumers:

```bash
extract_test=$(TEST_MODE=true sbatch --parsable --array=0-1 \
  src/slurm/extract_adaptation.slurm)

baseline_test=$(TEST_MODE=true sbatch --parsable --array=0 \
  --dependency=afterok:$extract_test \
  src/adaptors/baselines/slurm/run.slurm)

gate_test=$(TEST_MODE=true sbatch --parsable --array=0 \
  --dependency=afterok:$extract_test \
  src/adaptors/gates/slurm/run.slurm)

TEST_MODE=true sbatch --dependency=afterok:$baseline_test:$gate_test \
  src/adaptors/baselines/slurm/build_tables.slurm
```

Inspect both Slurm logs, the extraction manifest, downstream JSON/CSV metrics,
feature-importance plots, and the Chronos full/average test tables.  Then submit
the publication arrays:

```bash
extract_job=$(sbatch --parsable src/slurm/extract_multi_backbone_adaptation.slurm)
baseline_job=$(sbatch --parsable --dependency=afterok:$extract_job \
  src/adaptors/baselines/slurm/run.slurm)
gate_job=$(sbatch --parsable --dependency=afterok:$extract_job \
  src/adaptors/gates/slurm/run.slurm)
sbatch --dependency=afterok:$baseline_job:$gate_job \
  src/adaptors/baselines/slurm/build_tables.slurm
```

The full extraction array has 784 tasks: seven datasets, two models, eight
settings, and seven variants (vanilla plus two spaces times three k values).
Baseline and gate arrays have 672 tasks each.  Concurrency is throttled in the
launchers.  The `572:64` setting is intentional for comparison with Cross-RAG.

The screening grid is intentionally single-seed (`SEED=1`). `SEED` is not part
of the output directory, so never submit different seeds against the same
`OUT_ROOT`: they would replace one another. For exploratory repeats, use one
root per seed (for example `OUT_ROOT=outputs/adaptation_seed_2`) consistently
for extraction and its downstream jobs. The current table builder averages
configurations, not seeds; seed aggregation should be added before presenting a
multi-seed adaptation result.

All grid axes have comma-separated environment overrides:
`DATASETS_CSV`, `MODELS_CSV`, `SETTINGS_CSV`, `DISTANCE_SPACES_CSV`, and
`NEIGHBORS_CSV`.  Settings use `L:H`.  When narrowing a grid, override the array
range too.  Extraction needs
`D*M*S*(1 + spaces*k)` tasks; baselines/gates need `D*M*S*spaces*k` tasks.
Out-of-range tasks exit safely, but submitting them wastes scheduler capacity.
For example:

```bash
DATASETS_CSV=Electricity MODELS_CSV=chronos SETTINGS_CSV=168:24 \
DISTANCE_SPACES_CSV=raw NEIGHBORS_CSV=3 \
sbatch --array=0-1 src/slurm/extract_multi_backbone_adaptation.slurm
```

Do not submit a downstream job without an `afterok` dependency unless the
corresponding manifests have already been checked.  Downstream launchers fail
before computation when the required extraction is absent, partial, stale, or
legacy, and assert the files expected by table discovery after each run.

## Tables and averages

The canonical table implementation is `src/slurm/build_tables.sh`; any of the
three `build_tables.slurm` wrappers delegates to it, so submit only one.  It
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
TEST_MODE=true sbatch --array=0 src/adaptors/ts_ifa/slurm/run.slurm
```

Its input extraction must already have a valid completion manifest.

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
`latex/experiment_guides/`.  Source code, notebooks, tests, and Slurm helpers
remain under `src/`; generated artifacts stay under `outputs/`, and runtime
logs under `logs/`.
