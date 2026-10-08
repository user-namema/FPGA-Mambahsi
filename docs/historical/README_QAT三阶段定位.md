# QAT 初始诊断、跳变捕获与重放命令

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

## 2. 执行初始量化诊断

```bash
python software/train_mambahsi_spatial_split_dense_qat.py \
  --dataset UP --data_set_path "$DATA_ROOT" --fp32_dir "$FP32_D1" \
  --work_dir "$RESULTS_ROOT/initial_diagnosis" --run_tag initial_diagnosis \
  --seeds 0,6 --device "$DEVICE" \
  --use_D true --use_z false --ssm-u-quantization shared --d-weight-bits 8 \
  --dt-input-bits 9 --dt-output-bits 8 \
  --batch_size 32 --eval_batch_size 8 --calibration_steps 100 \
  --initial_diagnostics_only
```

## 3. 训练并捕获跳变

```bash
export CAPTURE_ROOT="$RESULTS_ROOT/jump_capture"
python software/run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --variants bn1_fixed_scales \
  --max-epoch 100 --eval-interval 1 \
  --capture-oa-drop-pp 5 --capture-start-epoch 25 --capture-max-events 3 \
  --output-dir "$CAPTURE_ROOT"
```

## 4. 选择已捕获事件

事件输入：`event.json`、`before.pth`、`after.pth`、`validation_batch.pt`。

```bash
export CAPTURE_RUN="$CAPTURE_ROOT/bn1_fixed_scales/training/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_bn1_fixed_scales/$CONFIG_D1/run_seed6"
export EVENT_DIR="$(find "$CAPTURE_RUN/jump_events" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n 1)"
test -n "$EVENT_DIR" || { echo "No event reached the capture threshold"; exit 1; }
for name in event.json before.pth after.pth validation_batch.pt; do
  test -f "$EVENT_DIR/$name" || { echo "Missing: $EVENT_DIR/$name"; exit 1; }
done
```

## 5. 分别执行三个后端重放

```bash
python software/replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device "$DEVICE" --mode native \
  --output-dir "$RESULTS_ROOT/jump_replay_native"

python software/replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device "$DEVICE" --mode tf32-off \
  --output-dir "$RESULTS_ROOT/jump_replay_tf32off"

python software/replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device "$DEVICE" --mode reference-tf32-off \
  --output-dir "$RESULTS_ROOT/jump_replay_reference"
```

## 6. 运行参数对照

```bash
python software/run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path "$DATA_ROOT" \
  --dataset UP --device "$DEVICE" --seeds 0,6 \
  --variants bn1_fixed_scales fixed_low_weight_lr fixed_freeze_bn_affine \
  --max-epoch 100 --eval-interval 1 \
  --capture-oa-drop-pp 5 --capture-start-epoch 25 --capture-max-events 3 \
  --output-dir "$RESULTS_ROOT/qat_parameter_comparison"
```

## 7. 读取输出

| 步骤 | 输出文件 |
|---|---|
| 初始诊断 | `initial_quantization.json`、`initial_quantization_summary.csv` |
| 捕获 | `jump_events/manifest.json`、事件目录 |
| 重放 | `jump_replay.json`、`layer_comparison.csv`、`batch_logits.npz` |
| 参数对照 | `validation_summary.csv`、`qat_training_history.jsonl` |
