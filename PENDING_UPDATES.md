# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-14: Increased the `tsrag.slurm` host-memory request from 40 GB to
  80 GB after job 43308 was cgroup OOM-killed while constructing the Traffic
  TS-RAG retrieval datastore. Updated `CLUSTER_STATUS.txt` with the failed job
  and resume action. Affected contract: TS-RAG cluster resource allocation and
  cluster handoff only; the experiment grid, retrieval protocol, signatures,
  and artifacts are unchanged. Check passed: the front declares exactly
  `#SBATCH --mem=80000`. Deferred integration: resubmit `tsrag.slurm` and
  confirm Traffic extraction and the remaining workflow complete within the
  new allocation. Required rerun: resume the interrupted TS-RAG workflow;
  completed exact configurations remain reusable.

- 2026-08-14: Fixed `pipeline_ranking` so each adapted result is compared with
  the vanilla result from the same retrieval run rather than whichever
  retrieval was discovered last. Result records now carry their run identity,
  the screen average tables and CSV/JSON ranking were regenerated, and
  generated reports use stable LF line endings. Standardized manual candidate
  selection in the sole root `SWEEP_CANDIDATES.txt` contract, currently holding
  the requested top five baselines, best shared CatBoost cov/avgY gates, and
  matching shared Bayes cov/avgY gates; the obsolete TS-IFA-only candidate file
  was removed. A shared shell reader validates entries and filters them by
  family/method for all 14 adaptation and TS-IFA ablation/follow-up fronts,
  while their existing CSV variables and `SELECTED_CANDIDATES_FILE` retain
  explicit override support. TS-IFA fronts now fail clearly when the shared
  file has no relevant TS-IFA entry and no CSV override. The new
  `fourier_retrieval_ablation.slurm` applies the adaptation selection and
  compares each original retrieval pipeline with a Fourier-amplitude retrieval
  counterpart while preserving method, distance metric, K, and retrieval mode.
  The new `offline_datastore_ablation.slurm` similarly compares each selected
  pipeline with its fixed-mode counterpart, where the datastore and retrieved
  futures are restricted to T0; explicit pipeline tables now support mixed
  online/fixed retrieval selections.
  Affected contracts:
  result discovery/ranking, generated screen reports, root candidate selection,
  the shared Slurm helper, 14 root fronts, shared profile dispatch, and focused
  smoke checks. Checks
  passed: the sweep result-table regression (including retrieval-specific
  references and LF outputs), exact candidate/filter/front audit, sweep profile
  audit, Python compilation of touched modules/tests, Git Bash syntax for the
  helper and the initial 12 fronts, current screen average-report regeneration,
  plus candidate/profile audits and Bash syntax for the Fourier front and its
  touched dispatch helpers. The offline front additionally passed the
  mixed-mode sweep-table regression, Python compilation, candidate/profile
  audits, and Bash syntax. Deferred integration: exercise the Fourier and
  offline-datastore fronts, one other default selected adaptation front, and
  one TS-IFA override on the
  cluster. README and `latex/experiment_guideline.tex` now document the sole
  shared candidate/override contract and the Fourier and offline-datastore
  fronts. Handoff maintenance reran the selected-candidate, sweep-profile,
  result-table, and sweep-table smoke checks with `PYTHONPATH=src`; compiled the
  six touched Python modules/tests; and passed Git Bash syntax for all 19
  touched fronts/helpers. BibTeX plus four pdfLaTeX passes rebuilt the guideline
  without unresolved references or overfull boxes, and all 15 rendered pages
  passed visual inspection.
  Required reruns: none for the ranking correction; Fourier evidence requires
  submitting `fourier_retrieval_ablation.slurm`; offline-datastore evidence
  requires submitting `offline_datastore_ablation.slurm`; future ablation
  submissions use the new manual list, while any desired
  comparison against older candidate selections must be resubmitted explicitly.

- 2026-08-13: Replaced the obsolete Cross-RAG execution front with
  `sota_benchmark.slurm`. `SOTA_BENCHMARK.json` now records the immutable
  published TS-RAG/Cross-RAG Table-4 MSE values and exact evaluation protocol.
  The new benchmark evaluates one selected project baseline with Chronos-Bolt
  on every official test origin: fixed 12/4/4-month ETT boundaries, 70/10/20
  custom-dataset boundaries, `L=512`, `H=64`, and per-channel standardization
  fitted on the official train segment. Our training/retrieval remains our own
  and is disclosed as such. The table builder combines computed MSE with static
  published rows; Cross-RAG and TS-RAG are not executed by this front.
  Separately, `tsrag.slurm` now uses TS-RAG's actual repository defaults on the
  four project datasets: Chronos-T5-base EOS embeddings, Euclidean same-channel
  fixed retrieval, all available T0 windows, and `K=10`. It strict-loads the
  released ARM only into Chronos-Bolt and evaluates the selected project
  baseline on identical neighbors with both Chronos-Bolt and Chronos-2; a
  Chronos-2 TS-RAG row is impossible with the released state dict. Extraction
  gained explicit split bounds, train-segment standardization, separately
  batched retrieval representations, same-user search, and fixed-store reuse.
  Affected contracts: the two root fronts and runners, extraction signatures,
  TS-RAG model/retriever/evaluator, both focused table builders, shared profiles,
  README, tests, and cluster handoff. Focused checks passed so far: Python
  compilation of all new/changed Python modules; six TS-RAG, four SOTA, and
  three retained Cross-RAG static/protocol tests; the sweep/profile smoke; Git
  Bash syntax for both fronts and all touched shell helpers; and targeted
  `git diff --check`. The generic import smoke is deferred because the available
  local runtime lacks `einops`; no environment changes were made. Deferred
  integration: strict-load
  both public checkpoints and execute one Chronos-T5 retrieval plus TS-RAG
  inference batch on cluster; then verify both generated tables. Documentation
  reconciliation in the experiment guideline remains for daily maintenance.
  Required runs: place `amazon/chronos-t5-base` below
  `weights/chronos-t5-base/`, TS-RAG `best.pth` below `weights/ts-rag/`, then
  submit `sota_benchmark.slurm` and `tsrag.slurm` independently.

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

- 2026-08-12: Tighten the shared manifest lifecycle so `ready` is seed-only
  and can never become an overall run status. Preserve Cross-RAG's two
  `scale == 1` normalization sentinels after verifying them in the pinned
  upstream commit `b9a5428`, and guard that source-faithful behavior in the
  model contract test. Affected files/contracts: `src/experiment_runs.py`,
  lifecycle tests, and the Cross-RAG source contract. Checks passed: 9 manifest
  tests, 1 publisher contract test, 3 Cross-RAG model-contract tests, and Bash
  syntax for `publish_job.sh`. No scientific rerun, artifact migration,
  README, or LaTeX change is required. Deferred integration: exercise one
  failed/cancelled and one successful live launch as already queued.
  Maintenance 2026-08-13: direct inspection reconfirmed the shared four-state
  run lifecycle and same-launch ready selection. The extraction-manifest and
  sweep-table integration checks passed in the shared thesis runtime. The
  README and guideline now state precisely that the overall run stays
  `running` with `ready_at_utc` while seed state is `ready`. BibTeX plus three
  pdfLaTeX passes completed without unresolved references or overfull boxes;
  all 15 rendered pages passed visual inspection. The focused manifest,
  publisher, Cross-RAG source, and shell-syntax checks were not repeated
  because their recorded successful coverage is unchanged. Remaining blockers:
  strict on-cluster Chronos-Bolt/Cross-RAG weight loading and the fixed
  `512:64` smoke, rerun `screen.slurm`, observe successful and failed/cancelled
  lifecycle paths, and run the manual publisher once.

- 2026-08-13: Analyze synchronized Cross-RAG job 43060, screen rerun 43124,
  and obsolete publisher job 43063. Cross-RAG strict loading failed before its
  first inference because the supplied `best.pth` retrieval-head keys do not
  match the pinned local dual-attention architecture; only eight full-ridge
  controls completed, with equal-dataset mean T3 nMSE gains of +2.22% on
  Chronos-2 and +0.04% on Chronos-Bolt. The screen reruns because the previous
  table failure marked every ready producer interrupted, while allocation skips
  only completed runs and clears interrupted artifacts on resume. The same
  ready-run handling duplicated four Chronos-Bolt vanilla extractions within
  job 43060. Publisher 43063 came from the removed automatic publisher and
  failed through a stale VS Code askpass socket, so it does not validate the
  current manual publisher. Affected handoff: `CLUSTER_STATUS.txt` and this
  queue entry; no code or scientific artifact contract was changed. Checks:
  inspected all three exact log pairs, all nine Cross-RAG-family manifests,
  the eight control metric files, allocation/exit code, and the current versus
  historical publisher code. No executable test was needed for this read-only
  diagnosis. Documentation implication: defer the executive summary until an
  actual Cross-RAG metric exists. Required work: align the checkpoint and model
  architecture, correct or explicitly recover producer lifecycle state before
  resubmission, then rerun Cross-RAG inference/tables and exercise the current
  `publish_job.sh` once in its intended interactive shell.

- 2026-08-13: Complete each successful producer configuration immediately so a
  later configuration or table failure preserves reusable work, interrupt only
  unfinished runs, and retain each seed's own artifact list. Preserve distinct
  pipeline-qualified labels for fitted baseline/gate rows while keeping the
  pipeline-independent direct formulas canonical. A one-time repair promoted
  918 still-current job-42887 ready manifests to completed. A post-pull audit
  also promoted 28 ready prerequisite/control manifests from failed Cross-RAG
  job 43060; its genuinely failed inference remains interrupted. Launch 43124
  has completed four recomputed extractions; a fifth artifact-complete `ready`
  extraction was promoted directly, leaving one genuinely running extraction
  identity untouched. Affected contracts: shared
  manifest helper, Adaptation Slurm completion, table discovery, schema-1
  manifests, README, guideline, and `CLUSTER_STATUS.txt`. Checks passed: 11
  manifest tests, publisher and Slurm contracts, the table smoke with two
  pipeline variants, Python AST parsing, Bash syntax, and all 15 rendered PDF
  pages; the LaTeX log has no unresolved references or overfull boxes. No
  scientific recomputation or schema bump is required. Remaining cluster work:
  finish the remaining launch-43124 identity, rebuild tables, exercise one
  successful and one failed/cancelled launch, and run `publish_job.sh` once.

- 2026-08-13: Simplify manual publication and correct the Cross-RAG weight
  handoff. `publish_job.sh` retains its optional single-job/all-artifact
  selection and intentional commit, then only sources `PROXY_SCRIPT_PATH`
  (default `$HOME/codes/proxy.sh`) and pushes `origin/main`; it no longer reads
  a separate proxy credential file or rewrites Git askpass variables. Removed
  all six obsolete `.publish/*.paths` files and twelve `publish_*.out/.err`
  logs. Inspection of the official Cross-RAG README, shell scripts, linked
  Drive, and open artifact-release issue found that installation downloads no
  adapter checkpoint; the Drive's 801.5 MB
  `checkpoints/chronos-bolt/best.pth` is the older TS-RAG model and is
  incompatible with Cross-RAG. The official zero-shot script expects a
  `best.pth` produced by its separate 20,000-step pretraining workflow. Affected
  files/contracts: `publish_job.sh`, its static contract test, README, local
  project guidance, cluster handoff, and publisher logs. Focused checks passed:
  11 manifest lifecycle tests, the publisher contract test, Git Bash syntax for
  `publish_job.sh`, `src/slurm/common.sh`, and `screen.slurm`, `git diff
  --check`, and an exact audit of all 947 repaired manifests (each changes only
  the two status fields and completion timestamp). Deferred integration: execute
  the publisher once in the intended cluster shell. Required experiment work:
  obtain an author-supplied Cross-RAG checkpoint or run the official pretraining
  workflow before rerunning Cross-RAG inference and tables.
