# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-11: Split Chronos-Bolt and Cross-RAG into source-faithful local model
  modules adapted from the official Chronos commit `7dc4435` and Cross-RAG
  commit `b9a5428`. Chronos-Bolt extraction now uses the local released
  architecture, and Cross-RAG evaluation imports its local dual-attention model
  instead of a separately cloned repository. The Cross-RAG front resolves
  `weights/chronos-bolt-base/` and one recursive `best.pth` below
  `weights/cross-rag/`, with explicit path overrides retained. Affected files
  and contracts: `src/models/{chronos_model,chronos_bolt,cross_rag,models}.py`,
  `src/models/__init__.py`, the Cross-RAG evaluator/Slurm weight interface,
  source-contract tests, and README execution instructions. Focused checks
  passed: Python compilation of every touched Python file; three static model
  contract tests; the sweep/profile smoke check; and Git Bash syntax for
  `crossrag.slurm` plus `src/slurm/run_crossrag.sh`. The generic load smoke
  remains deferred because `uv` is unavailable and the shared notebook runtime
  lacks `einops`; a real local weight load is also impossible without the
  downloaded checkpoints. Deferred integration: on-cluster strict loading of
  the downloaded Chronos-Bolt and Cross-RAG state dicts and one fixed
  `512:64`, min-max/cosine, `K=15` inference smoke. The README and experiment
  guideline now document the local implementation and weight locations; the
  guideline compiled with its bibliography and pages 2 and 13--15 passed visual
  inspection. Required rerun: run `crossrag.slurm`; existing Chronos-2 winner
  artifacts may be reused, but all Chronos-Bolt/Cross-RAG extraction and
  comparison identities must be produced or rerun with the local
  implementations. Maintenance 2026-08-12: direct inspection reconfirmed the
  local model/evaluator/weight-path contract. The static model, manifest
  lifecycle, publisher, and extraction-manifest checks passed, as did Bash
  syntax for the Cross-RAG and publisher fronts/helpers; the focused static
  boundaries were repeated because no executable local weight-load boundary is
  available, while the extraction-manifest check supplied complementary
  integration coverage. README content is current. The guideline bibliography
  path was corrected for compilation from `latex/`, long weight variables were
  made breakable, BibTeX plus three pdfLaTeX passes completed without unresolved
  references or overfull boxes, and all 15 rendered pages passed visual
  inspection. Blocker/next action: `uv` is unavailable, the shared runtime lacks
  `einops` and Transformers, and the checkpoints are absent locally; perform the
  strict state-dict load and fixed `512:64`, min-max/cosine, `K=15` smoke on the
  cluster, then run `crossrag.slurm`.

- 2026-08-11: Recover screen job 42837's multi-stage lifecycle. Same-launch
  downstream stages may consume ready manifests, while independent selection
  remains completed-only and a failed launch remains interrupted. Completion
  is applied only by the final successful Slurm exit. Affected contracts:
  manifest selection, root Slurm lifecycle, README, experiment guideline, and
  cluster handoff. Focused lifecycle, result-table, sweep-table, timing-table,
  and baseline-coefficient checks passed. Deferred integration: one successful
  and one failed/cancelled live producer. Required rerun: the interrupted screen
  workflow after deploying the current table fix.

- 2026-08-12: Remove all automatic/Slurm publication and provide the standalone
  root `publish_job.sh` command beside the Slurm fronts. With a job ID it selects
  that numeric job's exact log pair and manifest-tagged run/report roots; without
  an ID it publishes all logs and lightweight outputs. Both modes exclude
  PT/NumPy/CatBoost payloads, commit only the selected paths, source the shared
  proxy credentials, clear stale VS Code askpass variables, and push without
  pulling. Correct table discovery
  so metric rows retain their saved canonical method name when baseline and gate
  workflows share a direct formula but have different pipeline configurations.
  Affected files/contracts: publisher Slurm removal, `src/slurm/common.sh`, the
  manual publisher and contract test, table discovery/regression tests, README,
  experiment guideline, and cluster handoff. Focused checks passed: Git Bash
  syntax for every root Slurm front plus `src/slurm/common.sh` and the manual
  publisher in both selection modes; `src/tests/test_publisher_contract.py`; and the result-table and
  sweep-table smoke checks, including different baseline/gate pipeline configs
  for one direct formula. The experiment guideline completed BibTeX plus three
  pdfLaTeX passes without undefined references or overfull boxes, and all 15
  rendered pages passed visual inspection. Deferred integration: run the
  publisher once from the cluster and rebuild the screen tables. Required
  rerun: screen job 42887 is interrupted; rerun `screen.slurm` after deploying
  this fix. No scientific formula or artifact schema changed.
