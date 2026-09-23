# Environment setup

Run training and GPU measurements on Linux with an NVIDIA GPU. The clean-install
recipe below follows the Python 3.9 / PyTorch 1.13.1 / CUDA 11.7 combination in
the [official MambaHSI instructions](https://github.com/li-yapeng/MambaHSI/blob/main/README.md).
The measured server environment is now captured in
`environment/observed_20260923/`. The clean-install recipe below is a separate
reconstruction option; it is not the environment used to produce the reports.

## Recorded experiment environment

| Component | Captured value |
|---|---|
| Python | 3.8.20 |
| PyTorch / torchvision / torchaudio | 1.13.1 / 0.14.1 / 0.13.1 |
| PyTorch CUDA / cuDNN | 11.7 / 8500 |
| Mamba / causal-conv1d | 1.1.2 / 1.1.2 |
| NumPy / SciPy / scikit-learn | 1.24.3 / 1.10.1 / 1.3.2 |
| Matplotlib / einops / spectral | 3.7.5 / 0.8.1 / 0.24 |
| NVML Python / driver | 13.610.43 / 580.178.04 |
| System nvcc toolkit | 12.6 (distinct from PyTorch CUDA 11.7) |

`pip_list.json` and `conda_list.json` are complete observed package inventories,
not automatically installable lockfiles. The environment also records local
wheel origins, CUDA extension hashes and runtime defaults. Benchmark JSON
settings take precedence over those new-process defaults (the power runs
explicitly disabled TF32).

The actual 14-file Mamba Python package is preserved under
`third_party/mamba_ssm_server_1_1_2/`, with its Apache-2.0 license and authors.
`mamba_simple.py` differs from its installed wheel RECORD; the received
`selective_scan_interface.py`, `mamba_simple.py` and `__init__.py` match the
collector hashes. This is an observed server snapshot, not a claim that the
package is identical to upstream Mamba 1.1.2.

The recorded scan interface **requires causal-conv1d and its CUDA extension**.
This differs from the scan-only dependency behavior of the alternative Mamba
1.2.0 recipe below. The original environment used cp38/cu118/torch1.13 wheels
with PyTorch reporting CUDA 11.7. Their names, hashes and paths are retained in
`environment.json`; the wheels and compiled extensions have not been supplied.
Do not compile them with the captured system CUDA 12.6 and assume binary parity.

For replay on the existing server, keep its working `mambahsi` environment and
first verify the optimized scan:

```bash
conda activate mambahsi
python tools/check_environment.py --device cuda:0 --check-scan --json environment_check.json
```

To select the archived Python files without overwriting site-packages, prepend
the snapshot directory for that command only (the existing compatible compiled
extensions and dependencies must still be installed):

```bash
PYTHONPATH="$PWD/third_party/mamba_ssm_server_1_1_2${PYTHONPATH:+:$PYTHONPATH}" \
  python tools/check_environment.py --device cuda:0 --check-scan --json environment_check_snapshot.json
```

A fresh exact recreation starts with `conda create -n mambahsi python=3.8.20`
and the recorded Torch versions, but requires the matching binary wheels in
addition to the Python snapshot. The alternative below is supplied for a new
installation; run CUDA checks and saved-prediction validation before using its
outputs as a reproduction of the captured experiments.

## 1. Create the environment

From the `FPGA-MambaHSI` repository root:

```bash
conda create -n mambahsi python=3.9
conda activate mambahsi
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install -c environment/constraints-torch113.txt -r environment/requirements.txt
python -m pip install -c environment/constraints-torch113.txt --no-build-isolation mamba-ssm==1.2.0
python -m pip check
```

Alternatively, create the base environment with
`conda env create -f environment/conda.yml`, activate `mambahsi`, and execute the
three pip commands above. Use a different conda environment name if `mambahsi`
already exists; do not replace the original experiment environment.

The constraints keep pip from upgrading the installed Torch ABI or installing
NumPy 2.x. Transformers is an upstream Mamba dependency even though the experiment
code calls the selective-scan interface directly; the recipe pins it to an older
Torch-compatible release. `calflops` is optional and affects FLOP reporting:

```bash
python -m pip install -c environment/constraints-torch113.txt -r environment/requirements-optional.txt
```

`causal-conv1d` is not needed by this repository's scan-only integration: Conv1D
is implemented in PyTorch in the supplied model. No replacement of an installed
`mamba_ssm.Mamba` class is required. Mamba 1.2.0 declares Linux/NVIDIA,
PyTorch 1.12+ and CUDA 11.6+ support; see its
[installation instructions](https://github.com/state-spaces/mamba/blob/v1.2.0/README.md)
and [build configuration](https://github.com/state-spaces/mamba/blob/v1.2.0/setup.py).

## 2. Check CUDA and the compiled scan

```bash
nvidia-smi
python tools/check_environment.py --device cuda:0 --check-scan --json environment_check.json
```

The checker imports the runtime dependencies, allocates a CUDA tensor and runs
a small FP32 selective-scan forward/backward operation against the reference
implementation. A successful import alone does not prove that the CUDA extension
matches the installed Torch binary. The JSON report is created only if its path
does not exist. Use `--device cuda:1` for logical GPU 1, or set
`CUDA_VISIBLE_DEVICES=1` and use `--device cuda:0`.

If Mamba must build from source, a matching CUDA toolkit with `nvcc` and a working
C++ compiler must be available. The conda runtime alone does not supply a CUDA
compiler. Record `nvcc --version` along with the Torch CUDA version; they describe
different components. If `nvidia-smi` reports a driver/library mismatch, repair
the server driver installation before collecting GPU results.

For a dependency check without CUDA:

```bash
python tools/check_environment.py --device cpu
```

Missing required packages produce exit code 1. Missing optional packages are
reported separately. The CPU check does not certify the CUDA timing path.
The model's slow Torch scan fallback can support debugging, but its predictions
can differ near quantization thresholds and its runtime cannot stand in for a
CUDA benchmark.

## 3. GPU power measurement

`nvidia-ml-py` (import name `pynvml`) is included in the recommended requirements.
The power script queries the GPU selected by Torch through NVML and saves its
device identity. Use an otherwise idle GPU for reportable power/energy results.
The explicit shared-GPU option marks contaminated runs; it does not subtract
other jobs' power. NVML measures device-level power, not wall-plug system power.

For an otherwise idle GPU hosting Xorg/Xwayland, explicitly pass
`--allow-display-processes` to the power benchmark or numbered stage 16. Only
identified graphics-only display servers are allowed; compute contexts, unknown
graphics tasks and unsupported process queries still fail. The reports retain
the raw process lists and distinguish `display_background_only` from a fully
exclusive GPU. The policy checks run boundaries, not every moment of the trial.

## Reproducibility status

The original Python sources are tracked by `source_manifest.json`. Package pins
in the reconstruction recipe were selected for the legacy Torch stack. The
captured server versions are stored separately under `environment/observed_20260923/`. CUDA installation, complete training and
timing must be verified on the target Linux GPU. Keep both the collector report
and `environment_check.json` with each published experiment release.
