# N3 post-route activity and power

The power workflow uses a completed N3 100-MHz build, its routed DCP and a
post-route functional simulation. It adds a simulation fileset to the project.
Run one project operation at a time. The recorded capture used tens of GiB of
memory and produced roughly 18.8 GB of SAIF activity. The default free-space
threshold is 40 GiB for probe/capture and 5 GiB for report.

## Commands

Run from the repository root in Windows CMD with `VIVADO_BAT` set to the
installed Vivado executable. `--vivado PATH` selects it for an individual call;
the fallback order is `VIVADO_BAT`, then `PATH`:

```bat
python hardware/power/n3_power.py check --build D:/fpga/n3_v1
python hardware/power/n3_power.py probe --build D:/fpga/n3_v1 --out D:/fpga/probe_v1
python hardware/power/n3_power.py capture --build D:/fpga/n3_v1 --probe-dir D:/fpga/probe_v1 --out D:/fpga/capture_v1
python hardware/power/n3_power.py report --build D:/fpga/n3_v1 --capture-dir D:/fpga/capture_v1 --out D:/fpga/power_v1
```

Output paths must be new ASCII directories with at most 48 characters.
`check` verifies the build files, source fingerprint, RTL PASS and routed status.
`all` combines capture and report after a successful probe.

The Windows wrappers accept the same named arguments.
`run_saif_probe.bat` selects `probe`; `run_capture_after_probe.bat` selects `all`.
`run_n3_power.bat` accepts the mode as its first argument. Set `NF_PYTHON` to the
external Python executable for these wrappers; its default is `python`.
See the [batch examples](../power/README_V2.md#1-检查构建).

## Probe and capture

Probe starts SAIF object registration at simulation time zero, runs 100 ns
during reset, and closes the SAIF. `PROBE_OK.tcl` records completion of this
registration/close sequence. `probe_NOT_FOR_POWER.saif` contains reset activity.

Capture uses the first 16 different tiles and checks all 2304 output logit bytes.
The source, DCP and registration scripts must match the probe. The activity
protocol is:

| Setting | Value |
|---|---|
| Clock | 100 MHz |
| Warmup | 9216 cycles |
| Activity window | 32768 cycles |
| Window start/end | 92160 ns / 419840 ns |
| Input/output tile interval | 4096 cycles |

After closing the activity window, simulation continues through the 16-tile
reference check. Final PASS produces `CAPTURE_OK.tcl`. Report mode reads this
receipt and verifies the SAIF and DCP fingerprints.

Registration is split by hierarchy and submitted in batches of at most 1024
objects. `registration.log` and the mode-specific console log show enumeration,
registration, simulation and SAIF closing progress. A paused simulation continues
within its live XSim process; a terminated process requires a new probe/capture.

## SAIF identifiers

[saif_sanitize.py](../power/saif_sanitize.py) creates a separate SAIF when XSim
emits malformed `(null)` NET identifiers. It replaces those identifiers with
stable `XsimNull_<hash>` names and leaves activity counts, hierarchy and duration
unchanged. The companion JSON records the source/output hashes and renamed count.

```bat
python hardware/power/saif_sanitize.py D:/fpga/capture_v1/activity.saif --capture-receipt D:/fpga/capture_v1/CAPTURE_OK.tcl --output D:/fpga/activity_clean_v1.saif
python hardware/power/n3_power.py report --build D:/fpga/n3_v1 --capture-dir D:/fpga/capture_v1 --saif-manifest D:/fpga/activity_clean_v1.saif.json --out D:/fpga/power_clean_v1
```

## Report contents

The output includes `power_saif.rpt`, `power_advisory.rpt`,
`operating_conditions.rpt`, per-resource `activity_*.rpt`, `protocol.txt` and
the console annotation statistics. `REPORT_COMPLETE.txt` marks report completion.

The estimate covers the D1 inference fabric at the report's process, voltage and
temperature settings. Activity comes from a functional netlist, without SDF
timing back-annotation. Unmatched nets use Vivado propagation/default activity.
PCIe, DDR and HDMI are outside this design scope.

The supplied N3 result is 6.386 W total, with 71% design-net matching, at typical
process and 50 °C. See [results](RESULTS.md) and the
[detailed power report](../evidence/fullnet/N3_POWER_AUDIT_20260925.md).
