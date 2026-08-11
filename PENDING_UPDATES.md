# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-11: Serialize automatic publishers with a repository `flock` across
  add, commit, proxy authentication, and push; make the Slurm publisher fail
  immediately after any command error. Update the publisher contract test,
  README, and experiment guideline. Focused checks passed: Bash syntax and
  `src/tests/test_publisher_contract.py`. Deferred coverage: a live pair of
  concurrent cluster publishers. No experiment rerun is required.
