# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

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
