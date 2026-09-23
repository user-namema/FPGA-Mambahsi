# Sources and third-party notices

This document records provenance; it does not assign a new license to another
project's code. Existing attribution and copyright comments in copied source
files are retained. `source_manifest.json` identifies the local source snapshot
used to assemble this package.

| Component | Use in this repository | Upstream |
|---|---|---|
| MambaHSI | Spatial/spectral model lineage and data-processing utilities; this package adds controlled architecture settings and the QAT/deployment experiments | [li-yapeng/MambaHSI](https://github.com/li-yapeng/MambaHSI) |
| Mamba / selective scan | Installed CUDA selective-scan dependency; the supplied models call its public scan interface | [state-spaces/mamba](https://github.com/state-spaces/mamba) (observed 1.1.2 Python snapshot; alternative recipe 1.2.0) |
| PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, h5py, spectral | Installed scientific-computing and visualization dependencies | Their respective upstream distributions |
| nvidia-ml-py | Installed Python NVML interface for GPU power measurement | NVIDIA package distribution |

Mamba v1.2.0 includes an
[Apache-2.0 license file](https://github.com/state-spaces/mamba/blob/v1.2.0/LICENSE).
This package does not vendor the compiled Mamba extension. The older local
release's claim that a modified external `mamba_ssm.Mamba` class is needed does
not describe this package's main execution path: its FP32 and QAT model classes
are included and use `selective_scan_fn` directly.

The author has not yet selected a project-wide license. A root `LICENSE` has
therefore not been invented. The upstream MambaHSI license/permission and exact
source revision for inherited utility code should be recorded alongside the
author's final license before the public release. The available upstream README
is attribution evidence, not a substitute for an upstream license grant.

Please cite the model and scan work when using the corresponding methods:

- Y. Li, Y. Luo, L. Zhang, Z. Wang, and B. Du, “MambaHSI: Spatial-Spectral Mamba
  for Hyperspectral Image Classification,” IEEE Transactions on Geoscience and
  Remote Sensing, 2024, [doi:10.1109/TGRS.2024.3430985](https://doi.org/10.1109/TGRS.2024.3430985).
- A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces,” [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).

Raw hyperspectral data retain their dataset-specific terms. RTL, Vivado project
files, vendor IP and implementation reports are outside this software package.

## Received Mamba 1.1.2 server snapshot

`third_party/mamba_ssm_server_1_1_2/mamba_ssm/` contains the actual Python
files supplied from the experiment server. The accompanying Apache-2.0 license
and AUTHORS are retained under `distribution_metadata/`. Existing file-level
copyright notices are preserved. `mamba_simple.py` was modified on the server
relative to its installed distribution RECORD; its exact observed bytes are
retained, with this modification notice. No further changes to those Python
files were made during packaging. RECORD describes the original installed
wheel and is not the verification manifest for the modified snapshot; use
`evidence/server_update_20260923/provenance.json` for current hashes.

The CUDA extensions are not bundled. Keeping this package for provenance does
not automatically make it the active Python import: see `ENVIRONMENT.md` for
explicit snapshot selection and the required causal-conv1d dependency.


## Received causal-conv1d 1.1.2 Python snapshot

`third_party/causal_conv1d_server_1_1_2/` preserves the Python interface and
package metadata from the verified installed-runtime backup. Its LICENSE and
AUTHORS are retained in `causal_conv1d-1.1.2.dist-info/`; no new license is
assigned to these files. The compiled extension remains outside Git.
