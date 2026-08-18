# Archived schema-1 runs

This directory contains schema-1 runs removed from the active `outputs/` tree
after they became scientifically invalid or their experiment track was
explicitly retired. Each dated batch preserves the run directories under
`runs/` using their original path relative to `outputs/` and documents the
reason for archival and any replacement lineage.

Archived runs are historical evidence only. Current launchers, selectors,
tables, and dependency audits must not read them. To inspect a run, read it in
place. To restore one for execution, copy its relative subtree back below
`outputs/`, reconcile the corresponding `SELECTED_RUNS.txt`, and rerun the
focused manifest/dependency audit before reuse.

Valid current-schema configurations are not archived merely because they are
not presently selected. Archival is reserved for invalidated computations,
superseded artifact contracts, or experiment tracks deliberately retired from
the active repository.
