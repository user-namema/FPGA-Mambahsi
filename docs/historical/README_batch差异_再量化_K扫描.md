# Batch、再量化与 K 精度命令

## 1. 配置输入

环境：[../ENVIRONMENT.md](../ENVIRONMENT.md)。先完成 [../EXPERIMENTS.md](../EXPERIMENTS.md) 的 01。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval8_4datasets"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export FP32_DIR="$FP32_ROOT/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG"
export QAT_RUN_DIR="$QAT_ROOT/UP/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20/$CONFIG/run_seed0"
export CUDA_VISIBLE_DEVICES="0"
export DEVICE="cuda:0"
cd "$PROJECT_ROOT"
test -f "$FP32_DIR/train_only_preprocess.npz"
python experiments/run.py 08_qat_eval8 --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"
test -f "$QAT_RUN_DIR/result.json"
test -f "$QAT_RUN_DIR/best_qat_foldaware.pth"
```

## 2. 模拟并保存 SSM 输入

```bash
export CAPTURE_ROOT="$RESULTS_ROOT/ssm_capture_UP0_batch8"
python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --dataset UP --seed 0 --data-path "$DATA_ROOT" --device cuda \
  --report-ssm-errors --ssm-analysis-split test \
  --capture-ssm-inputs --ssm-capture-tiles 8 \
  --output-dir "$CAPTURE_ROOT"
```

## 3. 诊断指定 batch

```bash
python software/diagnose_qat_batch.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --device cuda \
  --group-index 0 --target-index 0 \
  --save-traces --output-dir "$RESULTS_ROOT/diagnose_batch_group0"
```

## 4. 对比 SSM 输出再量化

```bash
python software/run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --device cuda \
  --suite requant --output-dir "$RESULTS_ROOT/ssm_requant"

python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --dataset UP --seed 0 --data-path "$DATA_ROOT" --device cuda \
  --ssm-error-source all --ssm-readout-requantization ideal \
  --report-ssm-errors --output-dir "$RESULTS_ROOT/ssm_ideal_requant"
```

## 5. 局部 K 精度重放

```bash
python software/run_ssm_error_ablation.py \
  --inputs-glob "$CAPTURE_ROOT/replay_tiles/*/ssm_replay_inputs/*.npz" \
  --suite k-precision \
  --k-bits-grid 19 21 23 --k-fraction-grid 24 26 28 \
  --output-dir "$RESULTS_ROOT/local_K_precision"
```

## 6. 全场景 K 精度扫描

```bash
python software/run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --device cuda \
  --suite k-precision \
  --k-bits-grid 19 21 23 --k-fraction-grid 24 26 28 \
  --output-dir "$RESULTS_ROOT/scene_K_precision"
```
