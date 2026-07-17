# Forecast adaptation and TS-IFA

This project contains retrieval-based time-series forecast adaptation experiments, including direct baselines, learned gates, and the TS-IFA adapter. The standardized layout keeps every implementation, notebook, test, and launch script under `src/`.

## Layout

```text
src/
  data/                  CSV loading, windows, and neighbor retrieval
  models/                forecasting backbones
  experiments/           extraction and direct experiments
  adaptors/
    baselines/           baseline mixture evaluation
    gates/               learned gate evaluation
    ts_ifa/              TS-IFA model, training, and Slurm jobs
  visu/                  plots, dashboard, notebook, and tables
  slurm/                 extraction and univariate jobs
  tests/smoke/           tiny synthetic checks
datasets/                remote datasets
weights/                 remote pretrained weights
outputs/                 extraction, training, evaluation, plots, and tables
logs/                    runtime and Slurm logs
```

## Protocol

The default chronological protocol uses T0 as the retrieval datastore, T1 for baseline/TS-IFA training, T2 for validation or gate training, and T3 for final evaluation. Extraction builds aligned query and neighbor payloads. Baselines and gates consume those payloads; TS-IFA combines vanilla, context-conditioned, residual, and memory forecasts.

TS-IFA uses random T1 examples for optimization and deterministic T2 validation. Both evaluation frequencies are expressed in optimizer steps:

```bash
python -m src.adaptors.ts_ifa.train \
  --input-dir outputs/adaptation/electricity/168_24/chronos/raw_euclidean_3_online/extracted \
  --output-dir outputs/adaptation/electricity/168_24/chronos/raw_euclidean_3_online/ts_ifa/TS-IFA \
  --epochs 10000 --batch-size 256 --lr 1e-5 \
  --valid-eval-freq 1000 --logging-eval-freq 1000
```

`--valid-eval-freq` controls full T2 validation. `--logging-eval-freq` controls reporting and must be a multiple of the validation frequency. The saved history records the mean training nMSE over each validation interval and the matching validation nMSE. `training_nmse.pdf` plots both curves; `--plot-step-train-loss` optionally adds the raw per-step curve.

## Main commands

```bash
python -m src.experiments.experiment_univariate --help
python -m src.experiments.extraction --help
python -m src.adaptors.baselines.evaluate --help
python -m src.adaptors.gates.evaluate --help
python -m src.adaptors.ts_ifa.train --help
python -m src.visu.sweep_results_table --help
```

Remote launchers are grouped with their implementation. The TS-IFA job is `src/adaptors/ts_ifa/slurm/run.slurm`; baseline and gate jobs are in their sibling adaptor folders; extraction jobs are in `src/slurm/`. Submit from the project root so `python -m src...` resolves correctly.

The dashboard notebook at `src/visu/retrieval_dashboard.ipynb` reads saved artifacts only and includes local and Google Drive/Colab setup branches.

## Lightweight checks

With the prepared project environment:

```bash
python src/tests/smoke/check_loads.py
python src/tests/smoke/check_baseline_oracles.py
python src/tests/smoke/check_ts_ifa_training.py
python src/tests/smoke/check_results_table.py
python src/tests/smoke/check_sweep_results_table.py
python src/tests/smoke/check_retrieval_dashboard.py
```

Do not run full extraction, model inference, or training locally.
