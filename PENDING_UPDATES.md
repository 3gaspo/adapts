# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-18: Split retrieval visualization into extraction and adaptation
  dashboards and make multi-result loading fail closed. The extraction notebook
  builds one extraction run from editable dataset, `L_H`, model, retrieval, and
  appended `RUN` components, and owns retrieved-example and frozen-forecast
  widgets. The adaptation notebook accepts one explicit extraction path plus a
  list of complete baseline, gate, or TS-IFA result paths with arbitrary path
  layouts, and owns adaptation comparison widgets. TS-IFA predictions are
  exposed under their manifest method IDs, per-model rooter artifacts remain
  separate, and conflicting duplicate prediction, diagnostic, coefficient, or
  importance names no longer silently overwrite another selected run. Both
  notebooks discover the project root from the kernel directory or its parents,
  resolve project-relative paths from that root, and confine all optional
  installation to the Colab branch. The workspace command codebook separately
  documents submission of the Jupyter Slurm front from the concerned project
  root. Affected files/contracts: both visualization notebooks, dashboard
  loader/helpers, focused dashboard smoke, README, cluster notebook launch
  instructions, and activity log; experiment artifacts and metrics are
  unchanged. Checks passed: Python compilation of the helper and smoke test;
  JSON parsing of both notebooks; setup-cell execution from both the adaptation
  root and `src/visu`, with package installation converted into a test failure;
  and the focused synthetic dashboard smoke covering extraction-only loading,
  baseline/gate loading, two simultaneous TS-IFA methods, coefficients, and
  plots. The smoke used the shared notebook runtime's headless backend and
  bypassed only the eager top-level `src` re-export because that runtime lacks
  `einops`; its normal command remains deferred to the prepared project/cluster
  environment. README now documents both path contracts. No LaTeX update or
  experiment rerun is required.

- 2026-08-18: Expand the retrieval-scope ablation to separate user-pool
  diversity from datastore cardinality. The front now evaluates the existing
  all-user, same-user, and other-users scopes at global materialized-store caps
  of 10k, 20k, 30k, and 50k windows. At 10k it additionally evaluates a
  deterministic `other_users_matched` pool with exactly one different-user
  candidate per datastore date, giving every query the same candidate count
  and temporal grid as same-user retrieval without duplicate windows. The
  current 30k all/same/other identities and report names are unchanged for
  exact reuse. Under the current stride-25 T0 protocol, Electricity and Traffic
  are cap-bound at 30k (96 and 34 dates respectively), while Solar and Exchange
  are bounded by their eligible dates (up to 98 and 84, depending on setting);
  consequently the proposed equal 10k candidate pools cannot be constructed
  from unique same-user windows, and matching requires downsampling the
  cross-user pool. Affected files/contracts: retrieval-scope front and profile
  orchestration, extraction scope choices and neighbor search, result/coefficient
  logical naming, focused smoke tests, cluster handoff, and activity log.
  Checks passed: Python compilation for all touched modules/tests; exact
  Euclidean, cosine, and Pearson matched-pool neighbor checks; sweep-profile,
  result-table, and coefficient-report smokes with `PYTHONPATH=src`; and Bash
  syntax for the front and affected launchers. The first two report-smoke
  invocations omitted `PYTHONPATH=src` and failed during import before tests
  ran; their corrected invocations passed. Required cluster work: resubmit
  `retrieval_scope_ablation.slurm`; exact completed 30k configurations skip,
  while the new cap identities and 10k matched controls run. During maintenance,
  reconcile README and the experiment guideline with the expanded design;
  update the executive summary only after the new results are analyzed.

- 2026-08-18: Reconcile Cross-RAG/TS-RAG metric protocols and add the fitting-
  loss ablation. Cross-RAG Table 4 is now represented as globally
  train-standardized pooled MSE with TS-RAG K=5 and Cross-RAG K=15; the SOTA
  reader fails closed unless local producer manifests prove the official train
  boundary, stride-1 evaluation, and exact test splits. The independent
  released-TS-RAG K=10 benchmark now saves MSE, nMSE, MAE, nMAE, model-only
  inference latency, and total/trainable ARM parameter counts, and its report
  exposes the four metrics plus latency and total parameters. Added
  `training_loss_ablation.slurm`, which holds selected screen pipelines fixed
  while fitting trainable ridge/convex baselines and regression/Bayes gate
  advantages with either MSE or per-query-window nMSE; direct/oracle methods
  remain objective-free. Affected files/contracts: SOTA static protocol and
  table, TS-RAG evaluator/runner/table, shared baseline/gate evaluator and
  launchers, profile orchestration, the new ablation/report module and front,
  focused tests, README, cluster handoff, and activity log. Checks passed:
  Python compilation for all changed modules/tests; Bash syntax for all changed
  fronts/launchers; 6 SOTA, 7 TS-RAG, and 2 training-loss contract tests; and a
  dependency-light numerical check that nMSE gate targets downweight a 10x-
  scale window by 100x. The broader baseline-oracle smoke could not import in
  the shared notebook runtime because `einops` is absent; rerun it in the
  project `uv` environment during maintenance. Required cluster work: rebuild
  SOTA tables only, rerun `tsrag.slurm` for the expanded artifact contract, and
  execute the new training-loss ablation; extraction artifacts remain reusable.
  Reconcile the experiment guideline after maintenance and update the executive
  summary only after the new results are analyzed.

- 2026-08-18: Move stale and deliberately retired schema-1 runs out of the
  active output tree while retaining their complete provenance. Added the
  documented `archive/stale_runs/` contract and preserved original paths below
  three dated batches: 19 bugged screen/TS-IFA runs with completed replacements,
  all 9 retired Cross-RAG workflow runs, and all 3 retired K=100 extraction
  runs. The complete Cross-RAG job-43060 and K=100 job-43589 log pairs moved
  with their retired batches; mixed job-42887 and job-43980 logs remain active
  because they also document valid reused configurations. The Cross-RAG and
  completed K=100 selection indexes moved with their identity roots. The
  existing pre-schema archive reason now records that its
  legacy extraction pipeline used stride 24; direct inspection found no active
  schema-1 stride-24 manifest to move. Affected files/contracts: archived run
  trees and their READMEs, the active `outputs/adaptation/{screen,ts_ifa,crossrag}`
  and `outputs/extraction` inventories, the narrow `.gitignore` exception that
  makes only `archive/stale_runs/` fetchable while older bulk archives remain
  ignored, legacy archive documentation, cluster handoff, and the durable
  thesis activity log. Checks passed: the pre-move
  closure audit found zero active downstream or report references to all 31
  targets; the post-move audit found 3,789 active schema-1 manifests (3,788
  completed and the one pre-existing job-43124 running manifest), zero missing
  or inactive input paths, zero dangling/invalid selections, and zero active
  Cross-RAG, K=100, or stride-24 configurations. No experiment rerun or
  artifact migration is required. Deferred maintenance: reconcile the README
  and experiment guideline only if their historical-artifact descriptions need
  the new schema-1 stale-run archive alongside the existing pre-schema archive.

- 2026-08-18: Retire the completed dependency migration utility. Removed the
  standalone audit script together with its manifest rewriting, provenance
  repair, selection-index repair, invalidation, and line-ending normalization
  functions. No live Slurm front, Python module, test, or public command called
  the utility; normal execution already resolves one exact selectable upstream
  run and embeds its scientific dependency in downstream signatures. Affected
  files/contracts: `src/scripts/audit_run_dependencies.py` only; the current
  manifest and artifact contracts are unchanged. Checks passed: workspace
  reference scan found no caller; all 13 focused manifest lifecycle/selection
  tests passed with `PYTHONPATH=src`; in-memory compilation passed for all 62
  Python sources; and a final source scan found no remaining audit, repair, or
  migration function in Adaptation. The initial test command without the
  required `PYTHONPATH` failed during import before running tests. Historical
  migration records remain in the handoff documents because they describe
  changes already applied. No experiment rerun or artifact migration is
  required. Deferred maintenance: none beyond resolving this documentation-only
  queue entry during the next maintenance pass.

- 2026-08-18: Correct the candidate-selection workflow after the repaired
  screen. Screen reports are evidence for a manually curated
  `SWEEP_CANDIDATES.txt`, not an immutable manifest; selection may retain gates,
  controls, or other scientifically important pipelines. The corrected ranking
  replaced instance K=3 cov-Y and cov-avgY shared ridge with the better raw K=3
  versions while retaining the other three baseline entries and all four gate
  controls. README guidance, the selector regression fixture, and the cluster
  handoff now reflect manual post-screen selection, selective exact reuse, and
  manual post-K promotion into `SECOND_GENERATION_CANDIDATES.txt`. Affected
  files/contracts: README, sweep candidate manifest, selected-candidate smoke,
  local project guidance, and cluster status. Checks passed: selected-candidate
  contract smoke, sweep-method-profile smoke, and Git whitespace validation.
  Required cluster reruns, submitted sequentially: horizon/convex/delta baseline
  formulation, Fourier retrieval, fixed-T0 datastore, retrieval scope, mixed
  quantity, H, L, and K. Exact completed configurations should skip and only
  missing new-candidate work should compute. The unchanged CatBoost/Bayes gates,
  corrected screen, and independent TS-IFA pilot need no rerun. After K analysis,
  refresh the adaptation records in `SECOND_GENERATION_CANDIDATES.txt`, run the
  final benchmark, and rerun TS-RAG/SOTA only if the first shared-ridge record
  changes. Reconcile result documents after the refreshed reports are obtained.

- 2026-08-17: Make upstream run fetching fail closed and bind downstream reuse
  to the upstream scientific computation. The shared manifest helper now has a
  single-run resolver with exact pipeline/seed filtering and can embed an
  upstream schema, identity/model configuration, pipeline/experiment
  configuration, and seeds in the downstream pipeline signature without using
  its path or manifest ID. Adaptation resolves every extraction field, uses
  that resolver, and automatically records the extraction dependency when
  allocating a downstream run. The current-artifact audit found 18
  scientifically invalid job-42887 screen runs (10 baselines and 8 gates) at
  Electricity 504:168/raw/Euclidean/K=3 and changed only those overall statuses
  to `interrupted`. All 72 job-42887 screen manifests at that dataset/setting
  also recorded the smoke vanilla manifest; 54 were provenance/timing-only
  errors because evaluation used the retrieval payload's vanilla arrays, and
  their references were migrated without rerunning. A one-time exact migration
  enriched 82 equivalent historical extraction manifests with reconstructible
  default fields, embedded extraction dependencies into 3,156 valid downstream
  manifests, recomputed their parameter-only signatures, and rebuilt their
  repeat indexes. No result payload was deleted. Affected contracts/files:
  shared manifest helper and tests, extraction resolver, dependency audit/
  migration utility, README, current manifests and selection indexes, and the
  cluster handoff. Checks passed: all 13 manifest lifecycle/selection tests in
  each of the five standardized projects; Adaptation's method-profile contract;
  Bash syntax; exact full-profile resolution to the selected publication run;
  an idempotent zero-change post-migration audit; and a full 3,748-manifest,
  3,667-selection-entry validation with zero dependency, signature, or index
  errors. Required rerun: after synchronizing these changes to the cluster,
  submit `sbatch screen.slurm` normally. Its exact extraction stage and 846
  valid screen results should skip, the 18 interrupted scientific rows should
  be recreated under new dependency-aware signatures, and job 43582's stale
  full/average/coefficient reports should rebuild. No other synchronized
  project needs a rerun. Recheck the screen ranking before treating later
  selection-driven conclusions as final; update analyzed-result documents only
  after the corrected report is synchronized. Deferred maintenance: reconcile
  and render `latex/experiment_guideline.tex` with the dependency contract.

- 2026-08-17: Synchronize and analyze retrieval-scope job 43978. The run
  completed with empty stderr, terminal table and coefficient stages, 96/96
  newly launched restricted-scope extraction manifests completed, and all
  324 report-selected adaptation manifests completed. Across the five selected
  ridge pipelines, equal-configuration T3 nMSE gains are +1.832% with all-user
  retrieval, +1.785% with other-users-only retrieval, and +1.480% with
  same-user-only retrieval; cross-user candidates therefore retain nearly all
  of the benefit, while all-user retrieval remains the preferred default.
  Squared-error gains do not extend to MAE, which worsens by 0.727--0.910%.
  The all-user control also exposed 18 old screen rows at Electricity 504:168,
  raw Euclidean K=3, whose job-42887 results consumed job-42822's smoke
  extraction instead of the full publication extraction. The new scope result
  uses the correct full-profile extraction and revises selected full ridge from
  +1.780% to +1.875%; job 43978 itself requires no rerun. Affected evidence:
  cluster handoff, executive summary/PDF, and the durable thesis activity log.
  Checks completed before documentation: log/stderr scan, exact report coverage
  and duplicate audit, manifest/seed/artifact inventory, paired scope analysis,
  and screen-upstream provenance comparison. Deferred cluster work: rerun the
  18 contaminated screen result rows for that one dataset/setting/retrieval
  pipeline and rebuild the screen report. No other screen row was numerically
  implicated; a later complete audit found and migrated 54 additional
  provenance/timing-only vanilla references from the same job.
  Two final pdfLaTeX passes completed without errors, undefined references, or
  overfull boxes; all three executive-summary pages were rendered at 150 DPI
  and visually inspected without clipping, overlap, or legibility defects.

- 2026-08-17: Simplify artifact publication and align the specialized SOTA and
  TS-RAG tables with shared run selection. A job-scoped `publish_job.sh` call
  now publishes only its exact stdout/stderr pair; an unscoped call stages the
  `logs/` and `outputs/` parent trees directly, with the existing heavy-file
  exclusions. SOTA and TS-RAG tables now use the shared schema-1 manifest
  selector, pipeline/configuration/repeat/purpose policies, and standard report
  provenance. TS-RAG remains a normal run with its existing metric payload;
  published SOTA JSON values remain static rows appended after computed-run
  selection. Affected contracts: canonical publisher, specialized readers and
  orchestrators, focused tests, README, and workspace/experiment guidance.
  Direct execution passed 5 SOTA, 7 TS-RAG, and all 6 publisher contract tests;
  Git Bash syntax passed for both orchestrators and all nine byte-identical
  publishers. The initial package-style unittest command could not import the
  Adaptation package because the shared runtime lacks `einops`; the same
  self-contained tests passed when executed directly. No scientific rerun or
  artifact migration is required; rebuild the specialized reports when their
  table stages next run. Deferred maintenance: reconcile and render the
  experiment guideline with the simplified publisher and selector provenance.

- 2026-08-17: Fix the neural TS-IFA crash from job 43980. The neural rooter now
  constructs candidate-token IDs from the runtime branch count, matching its
  token embedding and scorer inputs for every branch set, routing scope, and
  constraint. The artifact contract and run signatures are unchanged. The 16
  completed joint-ridge configurations remain reusable; resubmitting
  `ts_ifa.slurm` reruns the interrupted first neural configuration and executes
  the remaining 15 neural configurations. A dependency-isolated forward check
  passed all 16 neural combinations of branch set, routing scope, and routing
  constraint; Python compilation, TS-IFA Bash syntax, and Git whitespace checks
  also passed. The complete project smoke remains deferred to the project
  environment because the shared notebook runtime does not provide `einops`.

- 2026-08-17: Analyze completed K/H/CatBoost jobs 43975--43977, promote
  generation-2 candidates, and integrate the L/H evidence into the experiment
  guideline and active ICLR 2027 submission. The eight unique adaptation
  candidates are ordered by each pipeline's best-K held-out improvement;
  instance full shared ridge K=5 is first and is therefore the exact TS-RAG and
  SOTA control, while the final benchmark consumes all eight. The combined L/H
  table records +2.43/+1.14/+0.41% over L=24/168/504 at H=24 and
  +0.41/+2.78/+2.75% over H=24/168/504 at L=504, with the L=24 instability
  stated explicitly. Cluster handoff and README were refreshed. No completed
  scientific artifact is invalidated. Required runs: benchmark, TS-RAG, and
  SOTA can now execute; retrieval scope remains pending terminal publication;
  TS-IFA's undefined neural-router variable is fixed and the pilot is ready for
  a resumable rerun. Checks passed in the shared thesis runtime: selected-
  candidate and sweep-profile smoke contracts, 5 SOTA and 7 TS-RAG contract
  tests, Bash syntax for benchmark/TS-RAG/SOTA, and Git whitespace validation.
  Both LaTeX sources compiled without undefined references, citations, or
  overfull boxes; the affected PDF pages were rendered and visually inspected.

- 2026-08-16: Split selected candidates into immutable experiment generations
  so active cluster work keeps its launch contract while post-K work can adopt
  the winning K. `SWEEP_CANDIDATES.txt` is frozen as generation 1 for the
  already-running K, H, CatBoost, and retrieval-scope workflows;
  `SECOND_GENERATION_CANDIDATES.txt` was introduced as an empty tracked manifest
  (and was populated after K analysis by the 2026-08-17 entry above); only
  `benchmark.slurm`, `tsrag.slurm`, `sota_benchmark.slurm`, and the four TS-IFA
  H/L/meta follow-up fronts consume. The v4 TS-IFA pilot remains a candidate
  producer. The user reports all five producer/ablation workflows as active,
  but their new job IDs and artifacts are not yet synchronized. Required
  cluster sequence: publish and analyze the active runs, write best-K adaptation
  records to generation 2, run benchmark/TS-RAG/SOTA, then promote any selected
  TS-IFA records before its follow-ups. Existing K=10 SOTA evidence is retained;
  its final-current replacement is determined only after generation 2 is
  populated. Affected contracts: candidate manifests, seven Slurm fronts,
  selection/profile smoke tests, README, local guidance, and cluster handoff.
  No active producer manifest or existing scientific artifact was changed, and
  this wiring change itself requires no scientific rerun. Checks passed in the
  shared thesis runtime: selected-candidate and profile smoke contracts, 5 SOTA
  and 7 TS-RAG contract tests, Bash syntax for all seven reassigned fronts, and
  Git whitespace validation. Deferred documentation: reconcile the experiment
  guideline during maintenance after generation 2 is populated.

- 2026-08-16: Centralize candidate parsing and exact selected-control report
  filtering for post-screen Slurm fronts; the default manifest assignment was
  subsequently split into immutable generations by the entry above.
  `benchmark.slurm` consumes all adaptation entries from its assigned manifest,
  while `sota_benchmark.slurm` and `tsrag.slurm` consume its first shared-ridge
  entry; explicit CSV overrides remain available. The specialized table readers
  filter and record the exact selected control so completed outputs from older
  selections cannot leak into a report. Existing instance-K=10 SOTA evidence is
  preserved but is not a final-current selection; its replacement and rerun
  scope will be determined by `SECOND_GENERATION_CANDIDATES.txt` after K
  analysis. Affected contracts: selected candidate reader, three Slurm fronts,
  SOTA/TS-RAG orchestrators and reports, focused tests, README, and cluster
  handoff. Checks passed in the shared thesis runtime: selected-candidate and
  profile smoke contracts, 5 SOTA and 7 TS-RAG contract tests including
  obsolete-control exclusion, Python compilation of both readers, Bash syntax
  for all six changed fronts/helpers, and Git whitespace validation. The
  package-style unittest invocation encountered the runtime's unrelated missing
  `einops`; direct execution of both self-contained test files passed. Deferred
  documentation: reconcile the experiment guideline during maintenance and
  replace the executive summary's former K=10 SOTA paragraph only after the
  generation-2 evidence is obtained and analyzed.

- 2026-08-16: Standardize `publish_job.sh` as the thesis-wide canonical
  publisher. It now sources the external proxy and fast-forward pulls
  `origin/main` before selecting, staging, or committing lightweight terminal
  artifacts; all nine active project copies are byte-identical. Affected
  contracts: publisher, focused contract test, README, and workspace/experiment
  guidance. Checks passed: Bash syntax for all nine copies, matching SHA-256
  hashes, and the Adaptation publisher contract test. No scientific rerun or
  artifact migration is required. Deferred maintenance: reconcile the publisher
  paragraph in `latex/experiment_guideline.tex` and exercise one real cluster
  publish with a remote update present.

- 2026-08-16: Add the selected-winner retrieval-user-scope ablation and finish
  the resumed TS-RAG/SOTA table repairs for cluster handoff. The new
  `retrieval_scope_ablation.slurm` runs every pipeline in
  `SWEEP_CANDIDATES.txt` with the complete all-user pool, the query user's
  windows alone, and every other user's windows while holding the selected
  space, metric, K, retrieval mode, model, datasets, settings, and seed fixed.
  Extraction manifests and downstream result signatures now record
  `retrieval.scope`; vanilla extraction remains scope-independent, and table
  and coefficient readers expose the three variants as distinct logical runs
  without changing their physical identity paths. The exact other-user search
  masks each query user's contiguous user-major datastore slice before top-K.
  README, the experiment guideline/PDF, cluster handoff, profile/fronts,
  extraction, result readers, and focused tests were updated. Checks passed in
  the shared thesis runtime: exact Euclidean/cosine/Pearson same/other-user
  retrieval, selected-candidate and profile contracts, three-scope table
  collision regression, sweep-table smoke, coefficient-export smoke, 11
  manifest-run tests, 5 SOTA and 7 TS-RAG contract tests, Python compilation,
  shell syntax, and Git whitespace validation. The guideline completed the
  pdfLaTeX/BibTeX/pdfLaTeX/pdfLaTeX sequence without undefined references or
  overfull boxes; the changed scope equations and workflow table were visually
  inspected. No existing scientific artifact requires recomputation. Deferred
  cluster checks: rerun the TS-RAG table stage; the former SOTA table-only action
  is superseded by the selected-candidate contract above. Then submit the new
  retrieval-scope front; all 27 selected pipeline/scope combinations over
  its documented primary grid require first execution and analysis.

- 2026-08-16: Repair cross-launch discovery for the TS-RAG and exact-split
  SOTA table stages after jobs 43756 and 43757 scoped completed producers to
  the new table launch and found no inputs. Both readers now consume only
  completed manifests across launches and record the exact obtained manifest
  IDs in their report manifests. Newly synchronized evidence was inspected
  directly: TS-RAG has four released-model plus eight matched-control manifests,
  and SOTA has all seven selected-method manifests across jobs 43305, 43580,
  and 43757. Local report rebuilds succeeded with 12 and 7 obtained manifests.
  The complete SOTA result is mean MSE 0.204286 for full shared ridge versus
  0.207888 for locally evaluated Chronos-Bolt, 0.197 published TS-RAG, and
  0.191 published Cross-RAG; the adapted method improves six of seven local
  rows and loses only on ETTm2. Checks passed in the shared thesis runtime:
  7 TS-RAG contract tests, 5 SOTA contract tests, the complementary sweep-table
  smoke, Python compilation of both readers, both exact local report rebuilds,
  direct status/artifact inventories, and Git whitespace validation. README and
  experiment-guideline inspection found their completed-only independent-reuse
  contract already current. The executive summary and cluster handoff were
  updated; two pdfLaTeX passes completed without errors, undefined references,
  or overfull boxes, and all three rendered pages passed visual inspection.
  The one-off four-page result-summary source and its private figure were moved
  together to `outputs/pdf/`, restoring the project `latex/` directory to its
  two maintained TeX sources. No scientific recomputation is required.
  Remaining cluster action: deploy the reader fix and rebuild the TS-RAG table.
  The former SOTA table-only action is superseded by the selected-candidate
  contract above, which requires a complete selected K=3 evaluation.

- 2026-08-16: Process the synchronized in-progress K-ablation job-43754
  snapshot and the still-pending H ablation. Final direct manifest inspection
  after job 43754 timed out found 118/144 active K<=20 extraction identities
  completed, Traffic 504:168 raw K=15 stale as running, and 25 identities not
  yet present. The two
  instance-retrieval gate methods have 72 completed configurations each; raw
  gates, all baselines, and final K tables have not started. The log snapshot
  has only the Slurm time-limit cancellation in stderr and ends inside that
  extraction, so no K optimum was analyzed or documented. The H workflow still needs a clean rerun after its
  earlier concurrent extraction collision; its formerly conflicting extraction
  is now complete and reusable. Remaining cluster action: mark launch 43754
  interrupted, resume the same K front, then resubmit `h_ablation.slurm`
  separately.

- 2026-08-15: Correct sweep-table provenance so each report records only
  manifests whose rows pass its family, variant, exact-pipeline,
  retrieval-axis, dataset, setting, model, metric, and split filters. Python
  compilation, the focused exact-provenance regression, the general
  result-table smoke, and Git whitespace validation passed. No scientific run
  is required and table values are unchanged. Remaining cluster action:
  rebuild one real selected-pipeline sweep report with the corrected reader.

Maintenance 2026-08-17: direct inspection confirmed the two immutable
candidate generations, the seven generation-2 consumer fronts, the intentionally
empty second-generation manifest, and the unchanged generation-1 contracts of
the five reported active workflows. No newer job IDs or artifacts were present,
so no result or executive-summary conclusion changed. Bash syntax passed for
all nine byte-identical publishers. The complementary sweep-table smoke passed
with the documented `PYTHONPATH=src` after an initial misconfigured
`PYTHONPATH=.` invocation failed during import; Git whitespace validation was
clean. The README and cluster handoff were current. The experiment guideline
was reconciled with both candidate generations and the canonical proxy-first,
fast-forward-pull publisher. Its final compile completed without warnings,
undefined references, or overfull boxes, and all 18 rendered pages passed
visual inspection. Focused selector/profile/TS-RAG/SOTA tests were not repeated
because they were already successful and the sweep-table smoke supplied the
complementary report boundary. Remaining work is external: publish and analyze
the five active workflows, populate generation 2 from K and TS-IFA evidence,
then run the deferred final consumers and one real publisher integration. No
scientific recomputation was performed locally.

Maintenance 2026-08-18: direct inspection covered jobs 43975--43980, the
populated generation-2 manifest, the current reports, the TS-IFA failure, and
the migrated manifest trees. The read-only dependency audit inspected 3,748
manifests with zero missing inputs, pending migrations, provenance repairs, or
new scientific invalidations. Complementary extraction-manifest and generic
result-table smokes passed, as did Git Bash syntax for all nine byte-identical
publishers. The README, executive summary, and cluster handoff were current;
the experiment guideline was reconciled with exact fail-closed extraction
dependencies and exact-log publication. The full TS-IFA training smoke was not
repeated because the prepared project environment is unavailable and the shared
runtime lacks `einops`; its focused 16-combination forward check had already
passed. BibTeX followed by three pdfLaTeX passes completed with a clean log,
and all 18 rendered guideline pages passed visual inspection. Remaining work
is external: rerun the 18 interrupted screen rows and reports, execute the
generation-2 benchmark/TS-RAG/SOTA fronts, resume the
corrected TS-IFA pilot, and exercise the publisher on the cluster.
