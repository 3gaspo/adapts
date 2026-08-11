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
  implementations.

- 2026-08-11: Recover screen job 42837's multi-stage lifecycle and terminal
  publishing. Same-launch downstream stages may consume ready manifests, while
  independent selection remains completed-only and a failed launch remains
  interrupted. Queue one publisher at workflow startup with `afterany`; it
  refreshes the exact producer log and launch-tagged output paths after any
  terminal state, still excluding PT/NumPy/CatBoost payloads and serializing
  Git/proxy/push. Affected contracts: manifest selection, root Slurm lifecycle,
  exact-path publication, README, experiment guideline, and cluster handoff.
  Focused checks passed: Bash syntax,
  `src/tests/test_experiment_runs.py src/tests/test_publisher_contract.py` (9
  tests), plus the result-table, sweep-table, timing-table, and
  baseline-coefficient smoke checks. The experiment guideline compiled
  successfully with its bibliography and pages 13--15 passed visual inspection.
  Deferred integration: a live failed/cancelled Slurm producer and its automatic
  publisher. Required rerun: the interrupted job 42837 screen in full after
  deploying the fix.

- 2026-08-11: Serialize automatic publishers with a repository `flock` across
  add, commit, proxy authentication, and push; make the Slurm publisher fail
  immediately after any command error. Update the publisher contract test,
  README, and experiment guideline. Focused checks passed: Bash syntax and
  `src/tests/test_publisher_contract.py`. Deferred coverage: a live pair of
  concurrent cluster publishers. No experiment rerun is required.
