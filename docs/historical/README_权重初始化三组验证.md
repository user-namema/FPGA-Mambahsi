# 权重初始化三组验证命令

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

## 2. 执行验证

```bash
python software/train_mambahsi_spatial_split_dense_qat.py \
  --dataset UP --data_set_path "$DATA_ROOT" --fp32_dir "$FP32_D1" \
  --work_dir "$RESULTS_ROOT/weight_init_validation" --run_tag weight_init_compare \
  --seeds 0,6 --device "$DEVICE" \
  --use_D true --use_z false --ssm-u-quantization shared --d-weight-bits 8 \
  --dt-input-bits 9 --dt-output-bits 8 \
  --batch_size 32 --eval_batch_size 8 --calibration_steps 100 \
  --weight_init_validation_only
```

## 3. 读取输出

| 分组 | 权重初始化 |
|---|---|
| `mean` | `2 * mean(abs(W)) / sqrt(127)` |
| `patch_max` | patch 权重使用 `max(abs(W_fold)) / 127` |
| `all_weight_max` | 卷积与投影权重使用 `max(abs(W_effective)) / 127` |

```bash
find "$RESULTS_ROOT/weight_init_validation" \
  -type f \( -name weight_init_validation_summary.json -o -name weight_init_validation.json -o -name weight_init_summary.csv \)
```
