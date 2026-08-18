# Replaced bugged runs

Archived: 2026-08-18

This batch contains 19 interrupted schema-1 runs whose computations were
bugged and were subsequently rerun successfully. Their original relative paths
below `outputs/` are preserved under `runs/`.

## Corrected screen lineage

Eighteen job-42887 screen results at Electricity `504:168`, Chronos-2, raw
Euclidean online retrieval, and `K=3` selected a smoke-profile extraction
instead of the required publication extraction. The dependency audit marked
the ten baseline and eight gate manifests interrupted. Job 44150 reran exactly
those configurations with the correct publication dependency, completed all
18 replacements, skipped the other 846 valid screen results, and rebuilt the
864-input screen report.

For `avgy` and `cov_forecast`, archived `run_0` and `run_1` map respectively to
corrected `run_2` and `run_3`. For every other archived screen identity,
`run_0` maps to corrected `run_1`. The archived and replacement manifests
retain their exact manifest IDs and pipeline signatures.

## Corrected TS-IFA lineage

Job 43980 interrupted the first joint-neural configuration because the neural
rooter referenced an undefined candidate-count variable. Job 44137 completed
the corrected configuration as `run_1` and finished the full 32-configuration
pilot. The archived `run_0` and its exact manifest ID remain here for audit.

## Verification

Before archival, none of these 19 runs was selected, referenced by an active
downstream manifest, or listed as an input to a current report. All replacements
were completed and selected where applicable.

The job-42887 and job-43980 log pairs remain in active `logs/` because both
jobs also produced valid configurations that were reused by their corrective
reruns; the logs cannot be split cleanly by manifest.
