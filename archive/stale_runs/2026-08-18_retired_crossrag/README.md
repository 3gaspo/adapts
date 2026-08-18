# Retired Cross-RAG runs

Archived: 2026-08-18

This batch contains all nine schema-1 runs owned by the retired Cross-RAG
execution workflow: eight completed full-ridge prerequisite/control runs and
the interrupted Cross-RAG inference run from job 43060. Their original paths
relative to `outputs/`, manifests, metrics, artifacts, and selection indexes
are preserved below `runs/`.

Cross-RAG inference failed before producing a metric because the available
checkpoint used the older TS-RAG retrieval-head architecture and could not be
loaded strictly by the pinned Cross-RAG implementation. No genuine released
Cross-RAG adapter checkpoint was available. The project therefore removed the
Cross-RAG execution front; current SOTA comparison uses immutable published
Cross-RAG values instead.

The eight completed controls were scientifically valid, but the complete
execution track was deliberately retired and is no longer eligible for active
selection. Before archival, none of the nine manifests was referenced by an
active downstream run or current report. The complete job-43060 stdout/stderr
pair is preserved under `logs/` in this batch.
