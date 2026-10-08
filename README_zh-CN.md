# FPGA-MambaHSI 复现步骤

[English](README.md)

## 1. 获取源码

```bash
export PROJECT_ROOT="$HOME/FPGA-MambaHSI"
git clone https://github.com/user-namema/FPGA-Mambahsi.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
```

## 2. 配置路径和运行参数

将 `DATA_ROOT` 和 `CUDA_HOME` 替换为自己的数据目录和 CUDA Toolkit 安装目录。

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

## 3. 安装并检查环境

运行环境：Linux x86_64、NVIDIA GPU、CUDA Toolkit 11.7、Conda、C++ 编译器。

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

## 4. 准备数据

按 [数据目录、文件名和变量名](docs/DATA.md) 放置 UP、HanChuan、HongHu、Houston 数据。

```bash
mkdir -p "$DATA_ROOT/UP" "$DATA_ROOT/HanChuan" "$DATA_ROOT/HongHu" "$DATA_ROOT/Houston"
```

HongHu 使用 NPY 文件时，先转换为 MAT：

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --datasets HongHu --convert-honghu
```

四个数据集的 MAT 文件准备好后，执行校验：

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --json "$RESULTS_ROOT/data_manifest.json"
```

## 5. 训练 FP32 模型

```bash
python experiments/run.py 01_fp32_current \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT"
```

输出：`$FP32_ROOT/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG/`。

## 6. 训练 QAT 模型

```bash
python experiments/run.py 13_qat_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"
```

训练 batch=32；验证、选模和测试 batch=1；第 20 epoch 冻结 BN 与 LSQ。

## 7. 运行定点模拟

```bash
python experiments/run.py 14_fpga_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_4datasets"
```

## 8. 运行 N0–N3 后端误差实验

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" --device "$DEVICE" \
  --trace-tiles 5 --output-dir "$RESULTS_ROOT/backend_error_D1_all40"
```

`--trace-tiles 5`：采集前 5 个 tile 的详细轨迹，整图执行推理。

## 9. 测量 FP32 GPU 吞吐率

```bash
python experiments/run.py 11_gpu_fp32 --datasets UP HanChuan HongHu Houston --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes 1,2,4,8,16,32,64 --output-root "$RESULTS_ROOT/gpu_fp32"
```

## 10. 测量定点 GPU 吞吐率

```bash
python experiments/run.py 12_gpu_fixed --datasets UP HanChuan HongHu Houston --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --qat-template "$QAT_TEMPLATE" --batch-sizes 1,2,4,8,16,32,64 \
  --output-root "$RESULTS_ROOT/gpu_fixed_eval1"
```

## 11. 测量 GPU 功耗

```bash
python experiments/run.py 16_gpu_power --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes 1,2,4,8,16,32,64 --output-root "$RESULTS_ROOT/gpu_power_UP"
```

GPU 同时用于 Xorg/Xwayland 显示时，增加 `--allow-display-processes`。

## 12. 检查文件和运行测试

```bash
python tools/audit_reproducibility.py
python tools/verify_release_records.py
python tools/verify_completion_records.py --check-arrays
python tools/paired_statistics.py --output-dir "$RESULTS_ROOT/paired_statistics"
python -m unittest discover -s tests -v
(cd software && python -m unittest discover -s tests -v)
```

单数据集运行：将训练和模拟命令的 `--datasets` 改为 `UP`，`--seeds` 改为 `0`。
查看完整子命令：给 `experiments/run.py` 增加 `--dry-run`。

## 13. 运行 FPGA 硬件

硬件流程使用 Windows、Vivado 2025.2 和 KU060 板卡，包含 N0–N3 全网实验
以及 UP D1 的 PCIe–DDR–HDMI 实现。

```bat
python hardware/run.py check
python -m unittest discover -s hardware/tests -v
```

仿真、实现、场景发送与标签比较见 [硬件说明](hardware/README_zh-CN.md)。
[硬件结果](hardware/docs/RESULTS.md)列出资源、时序、功耗条件和上板测试。

[全部实验命令](docs/EXPERIMENTS.md) · [环境安装](docs/ENVIRONMENT.md) ·
[硬件文件与运行入口](docs/REPRODUCIBILITY_AUDIT_20261008.md)
