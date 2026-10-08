# Hardware checks and simulation

## Source and host checks

Run these commands from the repository root:

| Command | Checks |
|---|---|
| `python hardware/run.py check` | Manifest files and hashes |
| `python -m unittest discover -s hardware/tests -v` | Source profiles, includes/modules, initialized IP paths, vectors, Tcl delimiters, staging and HDMI selection |
| `python hardware/power/test_saif_v2.py` | SAIF object partitioning, batches, errors, probe/capture scheduling and final PASS handling with Tcl mocks |
| `python hardware/power/test_saif_sanitize.py` | SAIF identifier conversion, source preservation and fingerprints |
| `python hardware/host/send_scene_v2.py --help` | Sender CLI and Python imports |

Use a Python installation with Tcl/Tk for the Tcl delimiter and mock checks.
These checks run on the host. RTL behavior, IP generation and timing use the
Vivado flow below.

## Full-network simulation and implementation

For each of N0, N1, N2 and N3:

1. `prepare` stages a package snapshot, checks its fingerprint, generates IP and
   creates the Vivado project.
2. `sim` checks 16 different input tiles, 2304 logit bytes, stream boundaries,
   context alignment and the 4096-cycle tile interval.
3. A successful simulation writes `sim_ok.txt` with the package fingerprint.
4. `build` checks that fingerprint, runs synthesis and writes utilization/DCP
   outputs. Builds within resource capacity continue to routing and timing reports.

Each method uses its own independent software reference. N3 is the complete
routed inference-fabric configuration. N0/N1/N2 synthesis results exceed the
KU060 DSP capacity at the configured parallelism; their resource reports are in
[evidence/fullnet](../evidence/fullnet).

## Board simulation and operation

The standard board simulation sets `REAL_FIVE=0` and repeats the first D1 tile
five times. It checks the network, integer interpolation/argmax, behavioral DDR
storage, HDMI path and C2H page buffer. The testbench
[`tb_d1_network_ddr_hdmi.sv`](../board/sim/tb_d1_network_ddr_hdmi.sv) also supports
`REAL_FIVE=1`, using `real5_input_128b.mem` and `real5_expected_logits_chw.mem`
from the same directory to check five different scene tiles.
PCIe link training and physical DDR/HDMI behavior are checked on the board.

The board build generates utilization, timing, clock-domain-crossing, DRC and
route reports. The bitstream is generated when the observed setup/hold checks
pass. After programming and reset, the host sender compares all 207400 labels
from the 858-tile scene with the supplied integer reference and reads the FPGA
tile-interval and display-underflow counters.

## Power checks

Power capture uses a successful N3 RTL simulation and routed DCP. The probe
checks object registration and SAIF closing. Capture checks all 16 tiles through
final PASS; report verifies the capture receipt and input hashes. Activity
matching and operating conditions are listed with the power estimate.

Commands are in the [hardware guide](../README.md) and [power guide](POWER.md).
Implementation measurements and their configurations are in [RESULTS.md](RESULTS.md).
