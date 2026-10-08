# QAT 稳定性对照命令

## 1. 配置路径

环境：[../ENVIRONMENT.md](../ENVIRONMENT.md)。FP32 训练输入：[../EXPERIMENTS.md](../EXPERIMENTS.md) 的 01；需要 QAT 输入时先完成 13。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export CONFIG_D1="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export FP32_D1="$FP32_ROOT/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"
export CUDA_VISIBLE_DEVICES="0"
export DEVICE="cuda:0"
cd "$PROJECT_ROOT"
test -f "$FP32_D1/run_seed0/best_model.pth"
test -f "$FP32_D1/run_seed6/best_model.pth"
```

## 2. 对比 BN 冻结轮次

```bash
python software/run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --variants control_bn20 bn1 \
  --max-epoch 100 --eval-interval 1 \
  --output-dir "$RESULTS_ROOT/stability_bn"
```

## 3. 对比尺度更新

```bash
python software/run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --variants bn1_fixed_scales bn1_slow_scales \
  --max-epoch 100 --eval-interval 1 \
  --output-dir "$RESULTS_ROOT/stability_scales"
```

## 4. 运行学习率对照

```bash
python software/run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --variants bn1_low_lr_cosine \
  --max-epoch 100 --eval-interval 1 \
  --output-dir "$RESULTS_ROOT/stability_lr"
```

## 5. 读取验证记录

```bash
find "$RESULTS_ROOT/stability_bn" "$RESULTS_ROOT/stability_scales" "$RESULTS_ROOT/stability_lr" \
  -type f \( -name validation_summary.csv -o -name qat_training_history.jsonl -o -name qat_training_diagnostics.jsonl -o -name best_validation_recheck.json \)
```

初始诊断与跳变捕获：[README_QAT三阶段定位.md](README_QAT三阶段定位.md)。
