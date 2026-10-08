# 复现输入检查

## 1. 配置目录

环境安装：[ENVIRONMENT.md](ENVIRONMENT.md)。数据下载：[DATA.md](DATA.md)。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export FP32_DIR="$FP32_ROOT/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG"
export QAT_RUN_DIR="$QAT_ROOT/UP/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed0"
export DEVICE="cuda:0"
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_ROOT"
```

## 2. 检查环境和数据

```bash
python tools/check_environment.py --device "$DEVICE" --check-scan
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" --datasets UP
```

## 3. 生成模型

```bash
python experiments/run.py 01_fp32_current --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT"

python experiments/run.py 13_qat_eval1 --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"
```

## 4. 检查模型输入

```bash
for name in train_only_preprocess.npz spatial_split_masks.npz spatial_split.json; do
  test -f "$FP32_DIR/$name" || { echo "Missing: $FP32_DIR/$name"; exit 1; }
done
for name in best_model.pth sample_indices.npz; do
  test -f "$FP32_DIR/run_seed0/$name" || { echo "Missing: $FP32_DIR/run_seed0/$name"; exit 1; }
done
for name in result.json best_qat_foldaware.pth sample_indices.npz qat_deploy_test_prediction.npy; do
  test -f "$QAT_RUN_DIR/$name" || { echo "Missing: $QAT_RUN_DIR/$name"; exit 1; }
done
```

## 5. 执行整数模拟

```bash
python experiments/run.py 14_fpga_eval1 --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_4datasets"
```

四数据集十种子命令：[EXPERIMENTS.md](EXPERIMENTS.md)。
