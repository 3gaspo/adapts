# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-16: Make `SWEEP_CANDIDATES.txt` the default source for every
  selection-driven post-screen Slurm front. `benchmark.slurm` now consumes all
  adaptation entries, while `sota_benchmark.slurm` and `tsrag.slurm` consume the
  first shared-ridge entry in manifest order; explicit CSV overrides remain
  available. The specialized table readers filter and record the exact selected
  control so completed outputs from older selections cannot leak into a report.
  The SOTA default consequently changes from the non-selected instance-K=10
  control to the selected instance-K=3 full ridge, invalidating the old computed
  SOTA row for current use without deleting it. Affected contracts: selected
  candidate reader, three Slurm fronts, SOTA/TS-RAG orchestrators and reports,
  focused tests, README, and cluster handoff. Required cluster reruns: TS-RAG
  may reuse all producers and rebuild its table; SOTA must recompute all seven
  selected K=3 rows and its table; the final benchmark still needs first
  execution. No other completed scientific artifact is invalidated. Checks
  passed in the shared thesis runtime: selected-candidate and profile smoke
  contracts, 5 SOTA and 7 TS-RAG contract tests including obsolete-control
  exclusion, Python compilation of both readers, Bash syntax for all six
  changed fronts/helpers, and Git whitespace validation. The package-style
  unittest invocation encountered the runtime's unrelated missing `einops`;
  direct execution of both self-contained test files passed. Deferred
  documentation: reconcile the experiment guideline during maintenance and
  replace the executive summary's former K=10 SOTA paragraph only after the
  selected K=3 evidence is obtained and analyzed.

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
