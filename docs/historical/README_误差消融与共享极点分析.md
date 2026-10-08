# 误差消融与极点分析命令

## 1. 配置输入

环境：[../ENVIRONMENT.md](../ENVIRONMENT.md)。先完成 [../EXPERIMENTS.md](../EXPERIMENTS.md) 的 01 → 13。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export FP32_DIR="$FP32_ROOT/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG"
export QAT_RUN_DIR="$QAT_ROOT/UP/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed0"
export CUDA_VISIBLE_DEVICES="0"
export DEVICE="cuda:0"
cd "$PROJECT_ROOT"
test -f "$FP32_DIR/train_only_preprocess.npz"
test -f "$QAT_RUN_DIR/result.json"
test -f "$QAT_RUN_DIR/best_qat_foldaware.pth"
```

## 2. 训练 dt 输入位宽对照

```bash
for bits in 8 9 10; do
  python software/train_mambahsi_spatial_split_dense_qat.py \
    --dataset UP --data_set_path "$DATA_ROOT" --fp32_dir "$FP32_DIR" \
    --work_dir "$RESULTS_ROOT/dt_input_width" --run_tag "dtin${bits}_dtout8" \
    --seeds 0 --device "$DEVICE" \
    --use_D true --use_z false --ssm-u-quantization shared \
    --dt-input-bits "$bits" --dt-output-bits 8
done
```

## 3. 训练 dt 输出位宽对照

```bash
for bits in 6 7 8 9 10; do
  python software/train_mambahsi_spatial_split_dense_qat.py \
    --dataset UP --data_set_path "$DATA_ROOT" --fp32_dir "$FP32_DIR" \
    --work_dir "$RESULTS_ROOT/dt_output_width" --run_tag "dtin9_dtout${bits}" \
    --seeds 0 --device "$DEVICE" \
    --use_D true --use_z false --ssm-u-quantization shared \
    --dt-input-bits 9 --dt-output-bits "$bits"
done
```

## 4. 全场景模拟并保存局部输入

```bash
export CAPTURE_ROOT="$RESULTS_ROOT/ssm_capture_UP0"
python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --dataset UP --seed 0 \
  --data-path "$DATA_ROOT" --fp32-dir "$FP32_DIR" \
  --device cuda --output-dir "$CAPTURE_ROOT" \
  --report-ssm-errors --ssm-analysis-split test \
  --capture-ssm-inputs --ssm-capture-tiles 8
```

## 5. 局部误差与位宽重放

```bash
python software/run_ssm_error_ablation.py \
  --inputs-glob "$CAPTURE_ROOT/replay_tiles/*/ssm_replay_inputs/*.npz" \
  --suite sources --output-dir "$RESULTS_ROOT/replay_sources"

python software/run_ssm_error_ablation.py \
  --inputs-glob "$CAPTURE_ROOT/replay_tiles/*/ssm_replay_inputs/*.npz" \
  --suite widths --output-dir "$RESULTS_ROOT/replay_widths"

python software/run_ssm_error_ablation.py \
  --inputs-glob "$CAPTURE_ROOT/replay_tiles/*/ssm_replay_inputs/*.npz" \
  --suite nonlinear --output-dir "$RESULTS_ROOT/replay_nonlinear"
```

## 6. 全场景误差来源与位宽扫描

```bash
python software/run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_RUN_DIR" --data-path "$DATA_ROOT" \
  --fp32-dir "$FP32_DIR" --device cuda \
  --suite both --output-dir "$RESULTS_ROOT/scene_error_sweep"
```

## 7. 指定状态格式

```bash
python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --dataset UP --seed 0 \
  --data-path "$DATA_ROOT" --fp32-dir "$FP32_DIR" --device cuda \
  --ssm-state-bits 28 --ssm-state-fraction-bits 20 \
  --report-ssm-errors --output-dir "$RESULTS_ROOT/state28_fraction20"
```

## 8. 指定 PWL 系数

```bash
python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --dataset UP --seed 0 \
  --data-path "$DATA_ROOT" --fp32-dir "$FP32_DIR" --device cuda \
  --ssm-coefficient-backend pwl --ssm-pwl-segments 32 \
  --report-ssm-errors --output-dir "$RESULTS_ROOT/pwl32"
```

## 9. FP32 极点分析

```bash
python software/analyze_alog_dynamics.py \
  --run-dir "$FP32_DIR/run_seed0" --data-path "$DATA_ROOT" \
  --split test --device cuda --output-dir "$RESULTS_ROOT/poles_fp32_UP0"

python software/analyze_alog_dynamics.py \
  --run-glob "$FP32_ROOT/**/run_seed[0-9]" \
  --data-path "$DATA_ROOT" --device cuda --no-plots \
  --output-dir "$RESULTS_ROOT/poles_fp32_all"
```

## 10. QAT 极点分析

```bash
python software/analyze_alog_dynamics.py \
  --run-dir "$QAT_RUN_DIR" --artifact-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --split test --device cuda \
  --max-tiles 32 --no-plots --output-dir "$RESULTS_ROOT/poles_qat_UP0"
```

## 11. 运行软件测试

```bash
PYTHONPATH="$PROJECT_ROOT/software${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s software/tests -v
```

再量化与 K 扫描：[README_batch差异_再量化_K扫描.md](README_batch差异_再量化_K扫描.md)。
