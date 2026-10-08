# Hardware dependencies

## Tools and target

- Windows with external Python 3.8+.
- Vivado 2025.2 with support for `xcku060-ffva1156-2-i`.
- The ALIENTEK KU060 board and the DDR4, PCIe and HDMI connections described by
  [board constraints](../board/constraints).
- NumPy for host label references stored as NPY; compact UINT8 BIN references
  use the Python standard library.

Build and power output paths use ASCII characters, contain at most 48 characters,
and name a new directory. `VIVADO_BAT` selects the Vivado executable.
The launchers call the external Python interpreter with `-I`.

## AMD IP and block design

The repository contains 85 XCI configurations and one PCIe block design.
Vivado generates the IP HDL, simulation products and implementation netlists
during project creation. The DDR4 part/pin configuration and XDMA AXI-MM
configuration correspond to the target board. Fabric builds use the network
IP subset; board builds also include DDR4 and HDMI clock IP.

[config/ip_catalog.json](../config/ip_catalog.json) maps each initialized IP
to its COE/MEM file. Project preparation binds these paths to the build's
`package/` snapshot and places generated outputs in the build directory.
`ip_status.rpt` lists the resulting IP status.

## HDMI controller

The board build uses the three files in [vendor/hdmi](../vendor/hdmi).
Project preparation selects this directory by default; `--vendor-dir PATH`
selects an alternative directory with the same controller files.

| File | SHA256 |
|---|---|
| `hdmiout_i2c_dri.v` | `ec53b87de9cbf00ca438d3e443d89101f37377c8ff13663f86c41be00905b917` |
| `hdmiout_i2c_ms7210_cfg.v` | `613db85e571decdb0f1d619070acec6b200e01ba9faf7e66921a4b949af46ce8` |
| `hdmiout_ms72xx_ctl.v` | `1a96dd711f046aa0cbc50bc143ebf97750c85a025fad551f98592d033058ee78` |

The launcher checks these hashes before staging the board project. These sources
contain the ALIENTEK copyright notices and the adapted module names/register
configuration. Full-network fabric and local nonlinear simulations use their own
source profiles.

## XDMA host driver

The scene sender uses the Windows XDMA driver for the board's AMD XDMA
**AXI-MM** configuration. Install the driver using the AMD/board support package.
The Python adapter is [host/host/dual_bank_test.py](../host/host/dual_bank_test.py).
It opens Windows device handles and provides H2C, C2H and user-register access.
Use `--device-index` when selecting among multiple enumerated devices.

The scene sender operates on a programmed and reset E2/D1 design. It writes the
two input banks, starts inference and reads the DDR labels. Bitstream programming
and board reset are performed through the board/Vivado controls.

## Files generated or installed locally

| Item | Source |
|---|---|
| IP HDL, simulation libraries and netlists | Vivado IP generation |
| `.bit` and `.dcp` | Hardware build |
| SAIF activity and current power reports | N3 power workflow |
| XDMA driver and driver certificates | AMD/board support package |
| Raw datasets and training checkpoints | Software data/training workflow |

Source files retain their existing author and vendor notices. AMD tools/IP,
board support, datasets and upstream Mamba software use their respective licenses.
