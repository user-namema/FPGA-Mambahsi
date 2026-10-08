# Max 初始化、QAT 与整数模拟命令

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

## 2. 顺序运行校准、QAT、整数模拟

```bash
python software/run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --stage all --max-epoch 20 \
  --output-dir "$RESULTS_ROOT/max_init_all"
```

## 3. 单独运行校准或 QAT

```bash
python software/run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --stage calibrated --output-dir "$RESULTS_ROOT/max_init_calibrated"

python software/run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --stage qat --max-epoch 20 --output-dir "$RESULTS_ROOT/max_init_qat"
```

## 4. 读取输出

```bash
find "$RESULTS_ROOT/max_init_all" \
  -type f \( -name pipeline_manifest.json -o -name pipeline_results.json -o -name result.json -o -name fpga_simulation_result.json \)
```
