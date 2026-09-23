# FPGA-MambaHSI

Software experiments for finite-code nonlinear compilation and streaming FPGA co-design of a spatial–spectral Mamba network for hyperspectral image classification.

This package contains FP32 training, architecture ablations, shared-A analysis, quantization-aware training (QAT), integer FPGA simulation, and GPU timing/power tools. It contains no RTL or Vivado project. Dataset files and trained checkpoints are supplied separately.

[中文运行说明](README_zh-CN.md) · [Experiment inventory](docs/EXPERIMENTS.md) · [Environment](docs/ENVIRONMENT.md) · [Data](docs/DATA.md) · [Server collection](docs/SERVER_FILES.md)

## 1. Create the environment

Use Linux and an NVIDIA GPU for the CUDA experiments. Start with the installation steps in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), beginning with:

```bash
conda create -n mambahsi python=3.9 -y
conda activate mambahsi
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install the remaining requirements and the selective-scan extension as documented there, then run:

```bash
python tools/check_environment.py --help
```

The installation baseline follows the [official MambaHSI README](https://github.com/li-yapeng/MambaHSI/blob/main/README.md). The original experiment server used Python 3.8 and PyTorch 1.13.1/CUDA 11.7. The captured server environment is now included under `environment/observed_20260923/`: Python 3.8.20, PyTorch 1.13.1/CUDA 11.7 and Mamba 1.1.2. Its actual Mamba Python package is archived with license and hashes under `third_party/`. The Python 3.9/Mamba 1.2.0 recipe above remains a reconstruction option, not the measured environment. See the environment guide before replaying results.

## 2. Prepare data and artifacts

Run the following commands from the extracted `FPGA-MambaHSI` directory. Put the four datasets under `data/`, following the filenames and MAT keys in [docs/DATA.md](docs/DATA.md). PCA and spatial splits are fitted/saved by FP32 training and reused by QAT and simulation.

```bash
export DATA_ROOT="$PWD/data"
export DEVICE=cuda:0
export SEEDS=0,1,2,3,4,5,6,7,8,9
```

The current model is `D1`, `z0`, shared A, two spatial/spectral branches, sum fusion, residual scale 2, hidden width 32, head width 64, four spectral tokens, and 16 states. Its configuration directory is:

```text
current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
```

Train the four FP32 models and save their preprocessing, splits and seed checkpoints:

```bash
bash experiments/01_fp32_current.sh --dry-run
bash experiments/01_fp32_current.sh
```

Existing FP32 artifacts can be used directly. Set `FP32_ROOT` to the parent of the four `<dataset>_all_samples_sqrt_inverse_clip3_2000_nobias` directories. For a nonstandard layout, pass `--fp32-template '/absolute/path/{dataset}/configuration'`. Keep `train_only_preprocess.npz`, the split files, each seed's sample indices, metadata and checkpoint together. Do not refit preprocessing for an existing checkpoint.

## 3. Current evaluation-batch-1 pipeline

```bash
# Fresh QAT: training batch 32; validation, selection and test batch 1.
bash experiments/13_qat_eval1.sh --dry-run
bash experiments/13_qat_eval1.sh

# Integer simulation of those same checkpoints.
bash experiments/14_fpga_eval1.sh --dry-run
bash experiments/14_fpga_eval1.sh
```

Defaults connect `results/qat_eval1_4datasets` to the simulator. Use `QAT_ROOT=/absolute/path/to/existing/qat_eval1_4datasets` for previously trained models. Use `--datasets UP --seeds 0 --output-root /new/path` for a first complete-scene run. For supported simulation recovery, rerun stage 14 with `--resume` and the same input/output paths.

`freeze20` means freezing BN statistics and LSQ scales at **epoch 20**. It does not mean 20 batches. The selected configuration uses patch-embedding max initialization and D mean initialization. Simulation inherits the saved numeric configuration: dt-projection input 9 bits, output 8 bits, K 19 bits with a preferred fractional cap of 24. The compiler determines each core's actual K fractional precision from its range. Do not override these settings in the main accuracy comparison.

## 4. Reproduce the reported experiments

The numbered launchers are independent entries, not one long mandatory training sequence. Their dependencies and completion evidence are listed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

```bash
# Network design and shared-A representation/accuracy cost
bash experiments/02_architecture.sh --dry-run
bash experiments/03_shared_a.sh --dry-run

# Original manuscript: saved evaluation/selection batch 8
bash experiments/08_qat_eval8.sh --dry-run
bash experiments/09_fpga_eval8.sh --dry-run

# RTX 4090 multi-batch timing
bash experiments/11_gpu_fp32.sh --dry-run
bash experiments/12_gpu_fixed.sh --dry-run

# Recompute the eight paired tests from bundled numeric evidence
python tools/paired_statistics.py
```

Remove `--dry-run` to execute. The top-level dry run only prints commands; it does not prove that datasets, checkpoints or CUDA are available. Underlying launchers perform their own checks when executed. GPU jobs run sequentially within each launcher. Do not start concurrent launchers against the same output directory.

The historical manuscript uses 40 QAT models selected with evaluation batch 8 and their 40 integer simulations. The newer 40 evaluation-batch-1 QAT runs and all 40 matching FPGA simulation reports are now included as a separate result set. Each simulator report reproduces the saved QAT test predictions at 100%. Stage 12 measures a wide-integer/FP64 implementation of the hardware arithmetic contract on GPU, not native Tensor Core INT8 throughput. Stage 16 now has 42 completed UP FP32 power trials (seven batches, two scopes, three repeats). They measure the GPU device with Xorg display background; no other compute process was found at the checked boundaries. They are not per-process or wall-plug measurements.

## 5. Files and evidence

```text
software/           Current training, analysis and simulation source snapshots
experiments/        Numbered launchers and portable command planner
environment/        Installation requirements, constraints and observed server records
tools/              Environment/data helpers, evidence checks and replay preparation
evidence/           Numeric summaries, raw lightweight reports and provenance hashes
third_party/        Actual Mamba Python snapshot, license and modification notice
tests/              Release orchestration tests
docs/               Experiment inventory, protocols and historical notes
source_manifest.json  SHA256 of the copied software snapshot
```

The archive audit found 760 result files for the 19 network configurations, 80 paired shared/per-channel-A training runs and 80 dynamics analyses, 40 historical QAT/integer pairs, 40 new eval1 QAT runs with 40 matching integer simulations, 42 UP GPU power trials, and 28 fixed-arithmetic GPU timing reports. The FP32 GPU table covers another 28 settings; its raw reports still need collection. Counts overlap where an experiment reuses the same models; they must not be added as independent repetitions.

`evidence/reference_run_index.csv` records the original relative path and SHA256 of each indexed result. It is an index, not a checkpoint archive. `source_manifest.json` records the current source snapshot; it does not imply that every historical experiment used this exact revision. The eval1 launch metadata does match the bundled QAT trainer hash. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for release checks and their limits.

## Attribution

The network originates from [MambaHSI](https://github.com/li-yapeng/MambaHSI), with selective-scan operations from [Mamba](https://github.com/state-spaces/mamba). See [third-party notices](docs/THIRD_PARTY_NOTICES.md). Repository: [user-namema/FPGA-Mambahsi](https://github.com/user-namema/FPGA-Mambahsi). The manuscript citation will be added when assigned; no publication status is implied.

## Recorded September 23 results

See [the records and exact-source replay guide](docs/SERVER_UPDATE_20260923.md).
All accuracy values below are ten-seed means; the loss is QAT minus integer simulation in percentage points.

| Dataset | Eval1 QAT OA (%) | Integer OA (%) | Loss (pp) |
|---|---:|---:|---:|
| UP | 96.4512 | 96.3853 | 0.0659 |
| HanChuan | 91.5552 | 91.4164 | 0.1388 |
| HongHu | 92.8104 | 92.7586 | 0.0518 |
| Houston | 92.1006 | 91.5403 | 0.5603 |

```bash
# Standard-library checks; no GPU, dataset or checkpoint required.
python tools/verify_release_records.py
# Create the simulator revision recorded by these 40 runs.
python tools/prepare_observed_eval1.py --output-dir ./results/observed_eval1_software
```

The main simulator retains the newer optional dt-input diagnostics. The exact
recorded simulator is preserved separately, with its launch script and source
hashes; neither revision is silently substituted for the other.
