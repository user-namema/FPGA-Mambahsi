# 实验复现命令

## 1. 配置路径、GPU 和种子

完成 [环境安装](ENVIRONMENT.md) 和 [数据准备](DATA.md)，将以下绝对路径改为本机目录。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export QAT_EVAL8_ROOT="$RESULTS_ROOT/qat_eval8_4datasets"
export CUDA_VISIBLE_DEVICES="0"
export DEVICE="cuda:0"
export SEEDS="0,1,2,3,4,5,6,7,8,9"
export DIAGNOSTIC_SEEDS="0,6"
export ANALYSIS_SEED="0"
export BATCH_SIZES="1,2,4,8,16,32,64"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
export FP32_DIR="$FP32_ROOT/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG"
export QAT_RUN_DIR="$QAT_ROOT/UP/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed$ANALYSIS_SEED"
export QAT_EVAL8_TEMPLATE="$QAT_EVAL8_ROOT/{dataset}/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/{dataset}_patch_max_D_mean_freeze20/$CONFIG/run_seed{seed}"
export QAT_EVAL8_RUN_DIR="$QAT_EVAL8_ROOT/UP/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20/$CONFIG/run_seed$ANALYSIS_SEED"
cd "$PROJECT_ROOT"
python tools/check_environment.py --device "$DEVICE" --check-scan
```

`CUDA_VISIBLE_DEVICES` 选择 GPU，`DEVICE` 使用选定 GPU 的逻辑编号。
`DIAGNOSTIC_SEEDS` 和 `ANALYSIS_SEED` 从 `SEEDS` 中选择。
每项实验使用独立输出目录。以下主流程按 01 → 13 → 14 执行。

## 2. 训练 FP32：01

```bash
python experiments/run.py 01_fp32_current \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT"
```

配置：D1、shared A、16×16 tile、train-only PCA 16、训练 batch 32、评估 batch 8、
最多 400 epoch、验证 mAcc 选模。
输出：`$FP32_ROOT/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG/`。

## 3. 训练 eval1 QAT：13

输入：步骤 2 的 FP32 配置、空间划分、预处理和逐种子采样索引。

```bash
python experiments/run.py 13_qat_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT"
```

配置：训练 batch 32，验证、选模、测试及部署参考 batch 1；patch embedding max、
D mean 初始化；100 epoch；第 20 epoch 冻结 BN/LSQ；校准 epoch 0 参与选模。
输出：`$QAT_ROOT/{dataset}/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/{dataset}_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed{seed}/`。

## 4. 整数全场景模拟：14

输入：步骤 2、3 的输出和原始数据。

```bash
python experiments/run.py 14_fpga_eval1 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_4datasets"
```

配置：DT9 输入、DT8 地址、A25/K19、32-bit Q24 状态、单次状态舍入。
输出：`$RESULTS_ROOT/fpga_eval1_4datasets/per_seed_summary.csv`、`dataset_summary.json`。

## 5. 网络结构实验：02

```bash
python experiments/run.py 02_architecture \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" \
  --output-root "$RESULTS_ROOT/architecture"
```

默认执行 19 项配置。指定配置时增加 `--cases 10_restore_D`，或传入多个配置名称。

## 6. shared/per-channel A 成对训练与分析：03

输入：步骤 2 的 FP32 划分、预处理和逐种子索引。

```bash
python experiments/run.py 03_shared_a \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$RESULTS_ROOT/shared_a_pair"
```

输出：`shared_a_pair/` 保存成对模型与精度表；`shared_a_pair_dynamics/` 保存极点分析。
单独训练或分析时使用 `--phase train` 或 `--phase analyze`；分析另设 `--pair-root`。

## 7. 初始化诊断：04

输入：步骤 2 的 FP32 输出。

```bash
python experiments/run.py 04_initialization --datasets UP --seeds "$DIAGNOSTIC_SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --init-kind weights --output-root "$RESULTS_ROOT/init_weights"

python experiments/run.py 04_initialization --datasets UP --seeds "$DIAGNOSTIC_SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --init-kind groups --output-root "$RESULTS_ROOT/init_groups"
```

`weights` 比较权重尺度初始化；`groups` 比较分组量化配置。两者执行校准和验证诊断。

## 8. QAT 稳定性：05

```bash
python experiments/run.py 05_stability --datasets UP --seeds "$DIAGNOSTIC_SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$RESULTS_ROOT/qat_stability"
```

默认配置：`control_bn20`、`bn1`、`bn1_fixed_scales`、`bn1_slow_scales`、
`fixed_low_weight_lr`、`fixed_freeze_bn_affine`。选择部分配置时使用 `--cases`。

## 9. max 初始化、QAT 和模拟：06

```bash
python experiments/run.py 06_max_init --datasets UP --seeds "$DIAGNOSTIC_SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$RESULTS_ROOT/max_init"
```

配置：patch_max/all_weight_max、D max 初始化、20 epoch QAT。

## 10. freeze1/freeze20 对照：07

```bash
python experiments/run.py 07_freeze --datasets UP --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$RESULTS_ROOT/freeze_comparison"
```

配置：patch_max、D mean、评估 batch 8；分别在第 1、20 epoch 冻结 BN/LSQ。

## 11. eval8 QAT 与全场景模拟：08、09

```bash
python experiments/run.py 08_qat_eval8 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_EVAL8_ROOT"

python experiments/run.py 09_fpga_eval8 \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --qat-template "$QAT_EVAL8_TEMPLATE" --output-root "$RESULTS_ROOT/fpga_eval8"
```

08 配置：训练 batch 32、评估 batch 8、patch_max、D mean、freeze20、100 epoch。
09 输入：08 的 QAT 输出、步骤 2 的 FP32 输出和原始数据。

## 12. 整网误差来源：10

```bash
python experiments/run.py 10_error_sources --datasets UP --seeds "$ANALYSIS_SEED" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --qat-template "$QAT_EVAL8_TEMPLATE" --suite sources \
  --output-root "$RESULTS_ROOT/error_sources"
```

`--suite` 可选 `sources`、`widths`、`nonlinear`、`requant`、`k-precision`。
每个 suite 使用独立输出目录。

## 13. GPU 吞吐：11、12

输入：步骤 2 的 FP32 输出；12 另需 08 的 QAT 输出。

```bash
python experiments/run.py 11_gpu_fp32 \
  --datasets UP HanChuan HongHu Houston --seeds "$ANALYSIS_SEED" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes "$BATCH_SIZES" --output-root "$RESULTS_ROOT/gpu_fp32"

python experiments/run.py 12_gpu_fixed \
  --datasets UP HanChuan HongHu Houston --seeds "$ANALYSIS_SEED" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --qat-template "$QAT_EVAL8_TEMPLATE" --batch-sizes "$BATCH_SIZES" \
  --output-root "$RESULTS_ROOT/gpu_fixed_eval8"
```

11 执行 FP32 CUDA 推理；12 执行 FP64/INT64 定点算术模拟。
输出：各输出根目录的 GPU batch 汇总 CSV。

## 14. dt 输入诊断：15

```bash
python experiments/run.py 15_dt_inputs --datasets UP --seeds "$ANALYSIS_SEED" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/dt_inputs_UP"
```

输入：步骤 3 的 eval1 QAT 输出。输出包含完整模拟结果及 dt 输入探针。

## 15. GPU 功耗与能量：16

输入：步骤 2 的 UP FP32 模型、原始数据和支持 NVML 的空闲 NVIDIA GPU。

```bash
python experiments/run.py 16_gpu_power --datasets UP --seeds "$ANALYSIS_SEED" \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --batch-sizes "$BATCH_SIZES" --output-root "$RESULTS_ROOT/gpu_power_UP"
```

配置：测量 30 秒、预热 10 秒、空闲 5 秒、重复 3 次、NVML 采样间隔 100 ms。
显示服务占用同一 GPU 时增加 `--allow-display-processes`。

## 16. QAT 跳变重放：17

输入：QAT 训练生成的事件目录，包含 `event.json`、before/after 权重和验证 batch。

```bash
export EVENT_DIR="/absolute/path/to/run_seed0/jump_events/epoch_before_to_after"
python experiments/run.py 17_replay_jump --datasets UP --seeds "$ANALYSIS_SEED" \
  --event-dir "$EVENT_DIR" --device "$DEVICE" --mode native \
  --output-root "$RESULTS_ROOT/jump_replay_native"
```

`--mode` 可选 `native`、`tf32-off`、`reference-tf32-off`。

## 17. 捕获输入并重放局部 SSM：18

```bash
export CAPTURE_ROOT="$RESULTS_ROOT/ssm_capture_UP"
python software/both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --dataset UP --seed "$ANALYSIS_SEED" --data-path "$DATA_ROOT" --device cuda \
  --qat-reference-batch-size 1 --capture-ssm-inputs --ssm-capture-tiles 1 \
  --ssm-analysis-split test --output-dir "$CAPTURE_ROOT"

python experiments/run.py 18_local_ssm --datasets UP --seeds "$ANALYSIS_SEED" \
  --inputs-glob "$CAPTURE_ROOT/replay_tiles/*/ssm_replay_inputs/*.npz" \
  --suite sources --output-root "$RESULTS_ROOT/local_ssm_UP"
```

输入：步骤 3 的 UP QAT 模型。局部重放的 `--suite` 可选
`sources`、`widths`、`nonlinear`、`k-precision`。

## 18. batch 数值诊断：19

输入：08 的 UP QAT 模型，保存的评估 batch 为 8。

```bash
python software/diagnose_qat_batch.py \
  --qat-run-dir "$QAT_EVAL8_RUN_DIR" --fp32-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --device cuda --group-index 0 --target-index 0 \
  --save-traces --output-dir "$RESULTS_ROOT/batch_diagnosis_UP"
```

`--group-index` 选择测试 batch，`--target-index` 选择该 batch 中的 tile。

## 19. 四数据集非线性后端：21

输入：步骤 2、3 的 FP32 与 eval1 D1 QAT 输出。

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" --device "$DEVICE" \
  --trace-tiles 5 --output-dir "$RESULTS_ROOT/backend_error_D1"
```

输出：`per_seed_backend_summary.csv`、`dataset_backend_summary.json`。
参数、预检和续跑步骤见 [后端实验命令](BACKEND_ERROR_FOUR_DATASETS_20260927.md)。

## 20. UP D1 指定参数模型：20

输入：`best_qat_foldaware.pth` 的 SHA256 为
`d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33`，
配套文件见 [D1 输入清单](COMPLETION_20260923.md)。

```bash
export D1_QAT_RUN_DIR="/absolute/path/to/D1/run_seed0"
export D1_FP32_DIR="/absolute/path/to/D1/FP32/configuration"
python tools/run_nonlinear_d1.py \
  --qat-run-dir "$D1_QAT_RUN_DIR" --fp32-dir "$D1_FP32_DIR" \
  --data-path "$DATA_ROOT" --device "$DEVICE" \
  --output-dir "$RESULTS_ROOT/nonlinear_D1"
```

## 21. 查看参数和续跑

```bash
python experiments/run.py --help
python software/run_backend_error_four_datasets.py --help
```

给 `experiments/run.py` 命令增加 `--dry-run` 可查看完整 Python 子命令。
03、06、11、12、14、15、21 支持续跑：保持输入、参数和输出目录，追加 `--resume`。
