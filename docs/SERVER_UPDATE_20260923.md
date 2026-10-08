# eval1 结果检查与运行

## 1. 配置目录

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export DEVICE="cuda:0"
export SEEDS="0,1,2,3,4,5,6,7,8,9"
cd "$PROJECT_ROOT"
```

## 2. 检查已有结果

```bash
python tools/verify_release_records.py
python tools/verify_completion_records.py
python tools/verify_completion_records.py --check-arrays
```

| 结果 | 位置 |
|---|---|
| eval1 整数模拟 | `../evidence/server_update_20260923/fpga_eval1/` |
| UP GPU 功耗 | `../evidence/server_update_20260923/gpu_power_UP/` |
| FP32 GPU | `../evidence/completion_20260923/gpu_fp32/` |
| UP dt 输入诊断 | `../evidence/completion_20260923/dt_input_UP_seed0/` |
| D1 N0–N3 | `../evidence/completion_20260923/nonlinear_D1/` |

## 3. 依次训练和模拟

环境：[ENVIRONMENT.md](ENVIRONMENT.md)。数据：[DATA.md](DATA.md)。

```bash
python experiments/run.py 01_fp32_current \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT"

python experiments/run.py 13_qat_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"

python experiments/run.py 14_fpga_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_4datasets"
```

其他实验：[EXPERIMENTS.md](EXPERIMENTS.md)。D1 指定 checkpoint 命令：[COMPLETION_20260923.md](COMPLETION_20260923.md)。
