# Frozen historical D1 N0–N3 source

The ten top-level Python source hashes match the completed UP run. The two
single-file fixes (`both_FPGA_nonlinear_D1.py`, `d1_methods.py`) are included;
they compare MEM integer words instead of newline-sensitive raw bytes.
Dependencies are kept adjacent. Use `experiments/20_nonlinear_D1.sh`, not the
current main simulator, to replay this recorded experiment.

`source_hashes.json` verifies the frozen Python/config/reference package.
This README and that manifest are packaging additions. No recorded numerical
source was edited. Hardware-reference files are coefficient vectors and
software numerical references; no RTL source is bundled.

The original `n2_config.json` contains old v5 checkpoint/core metadata. The D1
code imports only its `fit` field, which matches `n2_fit` in the D1 result.
D1 coefficients are constructed from the D1 checkpoint and checked against the
D1 reference vectors. See `docs/COMPLETION_20260923.txt` for scope and commands.
