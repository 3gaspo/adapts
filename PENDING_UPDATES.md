# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-15: Corrected sweep-table provenance so each
  `report_manifest.json` records only manifests whose result rows pass the
  report's family, variant, exact-pipeline, retrieval-axis, dataset, setting,
  model, metric, and split filters. The requested section now records all of
  those controls. Added a regression with unrelated completed pipelines that
  verifies a one-pipeline report obtains only the two requested dataset runs.
  Affected contract: sweep table report manifests and their focused smoke
  coverage; table values and existing experiment artifacts are unchanged.
  Checks passed: Python AST parsing for both touched Python files, the focused
  sweep-table smoke including exact report provenance, the general result-table
  smoke, and `git diff --check`. Deferred integration: rebuild one real
  selected-pipeline report on the cluster. README/LaTeX implications: none,
  because they already promise exact obtained inputs. Required reruns: no
  scientific runs; rebuild reports when corrected provenance is required.

- 2026-08-15: Made `SWEEP_CANDIDATES.txt` the sole selected-pipeline source in
  the project guidance and removed duplicated winner names, counts, and
  retrieval settings from `AGENTS.md`. Affected contract: local maintenance
  guidance only; executable selectors, experiment profiles, public
  documentation, and artifacts are unchanged. Check passed: the guidance now
  contains one authoritative selector rule and no hard-coded selected-control
  roster. Deferred integration: none. README/LaTeX implications: none, because
  both already document the current selector contract. Required reruns: none.

- 2026-08-15: Corrected TS-RAG table discovery to read the current schema's
  nested `launch.launch_id`, so a table stage scoped by
  `EXPERIMENT_LAUNCH_ID` retains the four completed job-43579 results instead
  of filtering every manifest out. Raised `sota_benchmark.slurm` host memory
  from 40,000 MB to 80,000 MB after the Electricity `K=10`, `L=512` retrieval
  payload required about 43.4 GiB for persistent evaluation tensors alone.
  Affected files/contracts: `src/visu/tsrag_comparison_table.py`,
  `sota_benchmark.slurm`, and their focused contract tests. Checks passed: six
  standalone TS-RAG contract tests, including same-launch table discovery;
  four standalone SOTA contract tests, including the 80 GB directive; Python
  compilation of the touched Python files; Git Bash syntax for both fronts and
  their runners; `git diff --check`; and an exact local table rebuild scoped to
  launch 43579, which recovered four rows each for TS-RAG and both matched
  Chronos-Bolt controls plus the two Chronos-2 controls. The package-style
  unittest invocation remains unavailable in the shared runtime because
  optional `einops` is not installed, while the same static tests pass when run
  directly. Documentation and LaTeX implications: no scientific protocol or
  analyzed conclusion changed. Required cluster work: run only the TS-RAG table
  stage because its four inference artifacts are already valid; resubmit the
  complete SOTA front, which will reuse its five completed datasets and resume
  Electricity before Exchange and tables.

- 2026-08-15: Added the completed baseline lookback ablation to the experiment
  guideline and executive summary as an exact result table and paired plot of
  mean nMSE improvement and positive-window rate. Across the four datasets,
  $L=168$ is the most balanced choice: the three instance-retrieval designs
  gain +0.90% to +1.14% on average, whereas $L=24$ includes an unstable Traffic
  vanilla reference and the full designs at $L=504$ fall below a 50% mean
  window-win rate. Affected files/artifacts:
  `latex/l_ablation_baseline_results.pdf`,
  `latex/experiment_guideline.tex`/`.pdf`, and
  `latex/executive_summary.tex`/`.pdf`. Checks passed: recomputed the plotted
  values from all 108 completed result records; compiled the guideline through
  BibTeX and the required pdfLaTeX passes; compiled the executive summary
  twice; found no LaTeX errors, unresolved references/citations, or overfull
  boxes; and rendered and visually inspected both integration pages and the
  standalone plot. Required reruns are unchanged: `k_ablation.slurm` must
  complete the 84 missing current K<=20 extraction identities before its
  adaptation/table stages, and `h_ablation.slurm` must resume its incomplete
  workflow while reusing the now-complete extraction that collided with L.
  Submit the two fronts sequentially to avoid another shared-extraction race.

- 2026-08-15: Removed K=100 from the active K-ablation grid, leaving
  K={1,3,5,10,15,20}, and synchronized the root-front comment, profile smoke
  expectation, README grid, experiment-guideline source, and cluster handoff.
  The current synchronized schema-1 tree contains only 60/144 required K<=20
  retrieval extraction manifests, so a rerun will reuse those 60 but must
  compute the remaining 84 before adaptation and tables; forcing an
  adaptation-only launch would fail current manifest validation. Diagnosed the
  H-ablation failure as a concurrent-ownership collision rather than an H-grid
  error: L job 43593 allocated Electricity 504:24 instance-K=3 at 22:47:36,
  and H job 43592 reached the same shared identity at 22:53:32 while it was
  still running. The allocator correctly refused a second writer; the later
  Bash `pop_var_context` messages were secondary failure-unwinding noise from
  nested sourced scripts. Aggregated all 108 completed L-ablation results (nine
  pipelines x three lookbacks x four datasets) into paired nMSE-improvement and
  positive-window plots. Affected files/contracts: `k_ablation.slurm`,
  `src/slurm/profiles.sh`,
  `src/tests/smoke/check_sweep_method_profiles.py`, `README.md`,
  `latex/experiment_guideline.tex`, and `CLUSTER_STATUS.txt`. Checks passed: Git
  Bash syntax for the K front/profile/orchestrator, the focused profile smoke
  test, an exact current-manifest inventory, exact L metric aggregation, and
  responsive/light/dark/tooltip/series-toggle plot checks. Deferred integration:
  consider adding wait-and-reuse behavior for identical concurrently running
  extraction identities if overlapping fronts will routinely be submitted
  together.
  Required reruns: resubmit `k_ablation.slurm` to finish the missing K<=20
  extractions plus adaptation/tables, and resubmit `h_ablation.slurm`; its
  formerly conflicting extraction is now complete and reusable.

- 2026-08-15: Integrated the completed corrected screen, Fourier-retrieval,
  horizon-sharing, convex, delta-ridge, and mixed-quantity ablations into the
  experiment guideline and ICLR 2027 submission draft. The documented evidence
  selects full shared ridge with instance-Euclidean retrieval and $K=3$ for the
  largest mean gain (+1.95%), while the compact cov/avgY shared ridge has the
  highest window win rate (54.17%). Every matched Fourier, per-horizon, convex,
  and delta formulation is worse than its shared-ridge control. Mixed panels
  are non-robust in aggregate, so cross-variable retrieval and fusion should be
  restricted to channels representing the same physical quantity. Affected
  files/artifacts: `latex/experiment_guideline.tex`,
  `latex/experiment_guideline.pdf`, and the workspace ICLR submission
  `latex/submissions/adaptation_ICLR2027/main.tex`/`main.pdf`. Checks passed:
  reconciled values against the exact generated reports; compiled both
  documents through the full bibliography/reference sequence; found no LaTeX
  errors, undefined references, or overfull boxes; rendered and visually
  inspected every changed result-table page; and confirmed the result text is
  extractable with pypdf. Deferred integration: fill the submission's remaining
  direct-conditioning, decomposition, figure, and AI-use placeholders after
  their evidence is available. Required reruns: none for these documentation
  updates; the incomplete $K$ and horizon-length workflows remain required for
  claims about their optima.

- 2026-08-15: Pulled and analyzed jobs 43579--43603. Eight workflows completed
  through their report stages; TS-RAG produced four valid inference metrics but
  failed table discovery, SOTA completed five of seven datasets before an
  Electricity OOM, K stopped at extraction 24/96 on the Electricity 504:168
  K=100 OOM, and H stopped after the raw-K=1 gates on a concurrently running
  instance-extraction manifest. Current ablations retain shared ridge over
  Fourier, horizon, convex, and delta variants; fixed-T0 retrieval is at least
  as good as every paired online ridge and is the preferred stricter store.
  L=24 and the mixed-quantity panels expose severe nMSE/outlier instability and
  must not drive primary conclusions. Updated the cluster handoff and executive
  summary with the exact partial/completed scope and decisions. Affected files:
  `CLUSTER_STATUS.txt`, `latex/executive_summary.tex`, and
  `latex/executive_summary.pdf`. Checks completed: inspected all 12 exact log
  pairs, terminal stages, synchronized manifests, per-run metric JSON, and all
  eight full/average report sets; independently recomputed TS-RAG comparisons
  and partial SOTA MSE means; compiled the executive summary with MiKTeX and
  inspected its rendered pages. Deferred integration: fix TS-RAG table result
  discovery and exercise a table-only rebuild; complete the SOTA, K, and H
  workflows, then reconcile the four-page adaptation result summary. Required
  reruns: resume SOTA after increasing/sharding host memory, resume K from the
  interrupted K=100 identity, and resume H; completed exact configurations are
  reusable. No rerun is required for the eight terminal report workflows or
  the four valid TS-RAG inference artifacts.

- 2026-08-15: Rebuilt the four-page adaptation results summary from the latest
  synchronized schema-1 evidence. The current screen table now uses the
  retrieval-matched ranking from job 43304, all result cells are percentages,
  and the CatBoost feature-importance figure was regenerated from all 96
  current screen CSVs. The summary preserves the exact distance-weight and
  shared CatBoost feature definitions, uses the $V+aV$ baseline notation,
  isolates archived formulation/$K\leq20$ and pre-v4 TS-IFA percentages from
  current comparisons, records that no v4 T1/T2 history exists, refreshes the
  test/full/ultra mode table, and inventories all 23 root Slurm fronts with
  current colored status. Affected files/artifacts:
  `latex/adaptation_results_summary.tex`,
  `latex/catboost_feature_importance_current.pdf`, and
  `outputs/pdf/adaptation_results_summary.pdf`. Checks passed: inspected the
  current report manifest/ranking, all 96 feature CSVs, terminal synchronized
  logs, root-front list, and cluster handoff; compiled twice with MiKTeX after
  the final edit; found no overfull boxes; rendered and visually inspected all
  four A4 pages with Poppler; verified four readable, unencrypted pages with
  pypdf and confirmed all 23 front names are present. The PDF skill's optional
  Node marker could not run because Node is not installed, and no dependency
  was installed. Deferred maintenance: reconcile `latex/executive_summary.tex`
  with the current schema-1 screen ranking; README and the experiment guideline
  require no protocol change. Required reruns: schema-1 formulation and
  $K$ ablations (including an evaluated $K=100$ result), the v4 TS-IFA pilot,
  the incomplete SOTA benchmark after restoring ETTh2, TS-RAG with the revised
  memory request, and the first Fourier/fixed-datastore executions.

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
