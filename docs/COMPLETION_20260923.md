# D1 N0–N3 输入与命令

## 1. 检查已有结果

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
cd "$PROJECT_ROOT"
python tools/verify_completion_records.py
python tools/verify_completion_records.py --check-arrays
```

| 内容 | 位置 |
|---|---|
| FP32 GPU 报告 | `../evidence/completion_20260923/gpu_fp32/` |
| dt 输入诊断 | `../evidence/completion_20260923/dt_input_UP_seed0/` |
| D1 预测、logits、系数 | `../evidence/completion_20260923/nonlinear_D1/` |
| D1 源码 | `../software/snapshots/nonlinear_D1_20260916/` |
| 其他结果 | `../evidence/completion_20260923/historical/` |

## 2. 准备指定输入

| 输入 | 要求 |
|---|---|
| `best_qat_foldaware.pth` | SHA256 `d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33`；文件未提供 |
| QAT seed 目录 | `result.json`、`sample_indices.npz`、`qat_deploy_test_prediction.npy` |
| FP32 配置目录 | `train_only_preprocess.npz`、`spatial_split_masks.npz`、`spatial_split.json` 和对应 seed 配置 |
| 原始 UP 数据 | `PaviaU.mat`、`PaviaU_gt.mat`；[下载与检查](DATA.md) |

## 3. 配置输入和输出

```bash
export DATA_ROOT="/absolute/path/to/data"
export D1_QAT_RUN_DIR="/absolute/path/to/matching/D1/run_seed0"
export D1_FP32_DIR="/absolute/path/to/matching/FP32/configuration"
export D1_OUTPUT="$PROJECT_ROOT/results/nonlinear_D1"
export DEVICE="cuda:0"
test -f "$D1_QAT_RUN_DIR/best_qat_foldaware.pth"
sha256sum "$D1_QAT_RUN_DIR/best_qat_foldaware.pth"
```

## 4. 执行 N0–N3

```bash
python tools/run_nonlinear_d1.py \
  --qat-run-dir "$D1_QAT_RUN_DIR" --fp32-dir "$D1_FP32_DIR" \
  --data-path "$DATA_ROOT" --device "$DEVICE" \
  --output-dir "$D1_OUTPUT"
```

从数据训练自己的模型并运行非线性对照：[EXPERIMENTS.md](EXPERIMENTS.md) 中的 01 → 13 → 21。
