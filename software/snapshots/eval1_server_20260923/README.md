# Recorded eval1 FPGA simulator — 2026-09-23

These two scripts are the exact source files received from the server. All 40
run manifests record the simulator hash in `source_hashes.json`. The other
recorded Python dependencies match the main `software/` files.

The main simulator has subsequently gained optional dt-input diagnostics; this
snapshot does not have that option. `tools/prepare_observed_eval1.py` verifies
the dependencies and assembles a standalone runnable copy in a new directory.
Do not run these two isolated files without those dependencies.

See `docs/SERVER_UPDATE_20260923.md` for commands and result scope. A synthetic
full-tile regression checks the current and recorded integer outputs with a
nonzero D bypass; it is not a fresh GPU reproduction of the 40 datasets/seeds.
