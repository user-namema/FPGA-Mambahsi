# FPGA-MambaHSI reproduction steps

[中文](README_zh-CN.md)

## 1. Obtain the source

```bash
export PROJECT_ROOT="$HOME/FPGA-MambaHSI"
git clone https://github.com/user-namema/FPGA-Mambahsi.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
```

## 2. Configure paths and parameters

Set `DATA_ROOT` to your dataset directory and `CUDA_HOME` to your CUDA Toolkit installation.

```bash
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export ENV_NAME="fpga-mambahsi"
export DEVICE="cuda:0"
export SEEDS="0,1,2,3,4,5,6,7,8,9"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export QAT_TEMPLATE="$QAT_ROOT/{dataset}/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/{dataset}_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed{seed}"
export CUDA_HOME="/absolute/path/to/cuda-11.7"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$RESULTS_ROOT"
```

## 3. Install and check the environment

Prerequisites: Linux x86_64, NVIDIA GPU, CUDA Toolkit 11.7, Conda and a C++ compiler.

```bash
nvidia-smi
nvcc --version
c++ --version
conda create -n "$ENV_NAME" python=3.9 pip -y
conda activate "$ENV_NAME"
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
  pytorch-cuda=11.7 'numpy<2' -c pytorch -c nvidia -y
python -m pip install -c environment/constraints-torch113.txt -r environment/requirements.txt
python -m pip install -c environment/constraints-torch113.txt --no-build-isolation mamba-ssm==1.2.0
python -m pip check
python tools/check_environment.py --device "$DEVICE" --check-scan \
  --json "$RESULTS_ROOT/environment_check.json"
```

## 4. Prepare the datasets

Place UP, HanChuan, HongHu and Houston files using the [directory layout, filenames and variable names](docs/DATA.md).

```bash
mkdir -p "$DATA_ROOT/UP" "$DATA_ROOT/HanChuan" "$DATA_ROOT/HongHu" "$DATA_ROOT/Houston"
```

For HongHu NPY files, convert the pair to MAT first:

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --datasets HongHu --convert-honghu
```

After preparing all four MAT pairs, validate the datasets:

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --json "$RESULTS_ROOT/data_manifest.json"
```

## 5. Train FP32 models

```bash
python experiments/run.py 01_fp32_current \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT"
```

Output: `$FP32_ROOT/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG/`.

## 6. Train QAT models

```bash
python experiments/run.py 13_qat_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"
```

Training batch=32; validation, selection and test batch=1; freeze BN and LSQ at epoch 20.

## 7. Run fixed-point simulation

```bash
python experiments/run.py 14_fpga_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_4datasets"
```

## 8. Compare N0–N3 backends

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" --device "$DEVICE" \
  --trace-tiles 5 --output-dir "$RESULTS_ROOT/backend_error_D1_all40"
```

`--trace-tiles 5`: collect detailed traces for the first five tiles; inference covers the full scene.

## 9. Measure FP32 GPU throughput

```bash
python experiments/run.py 11_gpu_fp32 --datasets UP HanChuan HongHu Houston --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes 1,2,4,8,16,32,64 --output-root "$RESULTS_ROOT/gpu_fp32"
```

## 10. Measure fixed-point GPU throughput

```bash
python experiments/run.py 12_gpu_fixed --datasets UP HanChuan HongHu Houston --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --qat-template "$QAT_TEMPLATE" --batch-sizes 1,2,4,8,16,32,64 \
  --output-root "$RESULTS_ROOT/gpu_fixed_eval1"
```

## 11. Measure GPU power

```bash
python experiments/run.py 16_gpu_power --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes 1,2,4,8,16,32,64 --output-root "$RESULTS_ROOT/gpu_power_UP"
```

Add `--allow-display-processes` when the GPU also runs Xorg/Xwayland graphics.

## 12. Check files and run tests

```bash
python tools/audit_reproducibility.py
python tools/verify_release_records.py
python tools/verify_completion_records.py --check-arrays
python tools/paired_statistics.py --output-dir "$RESULTS_ROOT/paired_statistics"
python -m unittest discover -s tests -v
(cd software && python -m unittest discover -s tests -v)
```

For one dataset and seed, set `--datasets UP --seeds 0` in the training and simulation commands.
Add `--dry-run` to `experiments/run.py` to print the complete child commands.

## 13. Run the FPGA hardware

The hardware flow uses Windows, Vivado 2025.2 and the KU060 board. It includes
N0–N3 full-network experiments and the UP D1 PCIe–DDR–HDMI implementation.

```bat
python hardware/run.py check
python -m unittest discover -s hardware/tests -v
```

Follow the [hardware guide](hardware/README.md) for simulation, implementation,
scene transfer and label comparison. [Hardware results](hardware/docs/RESULTS.md)
list resource usage, timing, power conditions and the board test.

[All experiment commands](docs/EXPERIMENTS.md) · [Environment setup](docs/ENVIRONMENT.md) ·
[Hardware files and entry points](docs/REPRODUCIBILITY_AUDIT_20261008.md)
