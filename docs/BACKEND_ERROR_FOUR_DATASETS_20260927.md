# 四数据集 D1 非线性后端实验命令

## 1. 配置目录和 GPU

先完成 [环境安装](ENVIRONMENT.md)、[数据准备](DATA.md) 及
[FP32 → eval1 QAT 训练](EXPERIMENTS.md)。将以下绝对路径改为本机目录。

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export CUDA_VISIBLE_DEVICES="0"
export DEVICE="cuda:0"
export SEEDS="0,1,2,3,4,5,6,7,8,9"
cd "$PROJECT_ROOT"
```

输入要求：

| 目录 | 必需内容 |
|---|---|
| `$DATA_ROOT/{dataset}/` | 原始图像 MAT、标签 MAT |
| FP32 配置目录 | `train_only_preprocess.npz`、`spatial_split_masks.npz`、`spatial_split.json`、`run_seed{seed}/model_config.json`、`run_seed{seed}/sample_indices.npz` |
| QAT 种子目录 | `result.json`、`best_qat_foldaware.pth`、`sample_indices.npz`、`qat_deploy_test_prediction.npy` |

QAT 配置：D1、评估 batch 1、训练 batch 32、patch_max、D mean、freeze20；
DT9 输入、DT8 输出、A25/K19、32-bit Q24 状态、单次状态舍入。
`QAT_ROOT/{dataset}/` 下每个种子对应一个 `result.json`。
N2 所有核的完整 dt 码域须落在 `[-16,16]`。
每次实验选择独立输出目录。

## 2. 检查一个检查点的输入

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" --datasets UP --seeds 0 \
  --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_UP0" --dry-run
```

`--dry-run` 读取输入、检查元数据与哈希，并打印运行命令。

## 3. 运行一个检查点

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" --datasets UP --seeds 0 \
  --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_UP0"
```

每个检查点执行四种完整网络后端：

| 后端 | A/K 系数计算 |
|---|---|
| N3 | Abar/K ROM |
| N0 | 在线定点多项式 |
| N1 | FastMamba-inspired 定点算术 |
| N2 | S2Mamba-inspired Softplus/Exp 计算链 |

各后端共用该检查点的量化尺度、A、D、状态格式及舍入配置。
`--trace-tiles 5` 设置详细 SSM 跟踪 tile 数；全场景统计覆盖全部 tile。

## 4. 运行四数据集和所选种子

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_D1" --continue-on-error
```

改变 `SEEDS` 可选择其他种子集合。`--continue-on-error` 在单项失败后继续后续检查点；
存在失败时退出码为非零。控制台列出各项日志路径。

## 5. 中断后续跑

保留步骤 4 的输入、参数和输出目录，增加 `--resume`。

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds "$SEEDS" \
  --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_D1" --continue-on-error --resume
```

续跑按检查点进行；已完成项经校验后跳过，其余项写入新的 `attemptXXX` 目录。
改变种子、输入或数值参数时选择新的输出目录。

## 6. 使用自定义 FP32/QAT 目录

按需替换两个配置目录，创建显式 UP seed0 任务清单。

```bash
export QAT_RUN_DIR="/absolute/path/to/UP/QAT/run_seed0"
export FP32_DIR="/absolute/path/to/UP/FP32/configuration"
export JOBS_JSON="$RESULTS_ROOT/backend_jobs_UP0.json"
mkdir -p "$RESULTS_ROOT"
python - <<'PYCODE'
import json
import os
from pathlib import Path
job = {
    "dataset": "UP",
    "seed": 0,
    "qat_run_dir": str(Path(os.environ["QAT_RUN_DIR"]).expanduser().resolve()),
    "fp32_dir": str(Path(os.environ["FP32_DIR"]).expanduser().resolve()),
}
with Path(os.environ["JOBS_JSON"]).open("x", encoding="utf-8") as handle:
    json.dump([job], handle, indent=2)
    handle.write("\n")
PYCODE
python software/run_backend_error_four_datasets.py \
  --jobs-json "$JOBS_JSON" --datasets UP --seeds 0 \
  --data-path "$DATA_ROOT" --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_custom_UP0" --dry-run
```

移除 `--dry-run` 执行。多数据集/种子清单为 JSON 数组，每项包含
`dataset`、`seed`、`qat_run_dir`、`fp32_dir`，覆盖命令选定的全部组合。

## 7. 读取输出

| 输出根目录文件 | 内容 |
|---|---|
| `manifest.json` | 输入、命令、哈希、状态和日志路径 |
| `per_seed_backend_summary.csv` | 逐种子、后端和统计范围的标签差异、OA、logit 误差 |
| `dataset_backend_summary.json` | 各数据集的种子均值、样本标准差和完成数量 |

每项结果位于 `{dataset}/seed{seed}/attemptXXX/nonlinear_experiment/`：

| 文件 | 内容 |
|---|---|
| `accuracy_propagation_summary.json` | 完整场景及测试区域统计 |
| `n*_tile_logits_int8.npy` | INT8 logits，形状 `[tiles, classes, 4, 4]` |
| `n*_prediction_integer.npy` | 整数插值后的全场景标签，类别编号从 0 开始 |
| `n*_prediction_float_interpolation.npy` | 浮点插值后的全场景标签 |
| `n*_disagreement_mask_vs_n3.npy` | 相对 N3 的逐像素标签差异 |
| `ground_truth.npy` | 真值类别从 1 开始，0 为无标注 |
| `sample_indices.npz`、`tile_manifest.json` | 测试索引和 tile 顺序 |
| `layer_propagation.json` | 各层相对 N3 的误差 |
| `layer_fixed_vs_float.json` | 各后端相对固定浮点参考的逐层误差 |
| `end_to_end_ssm_vs_n3.json` | 跟踪 tile 的 SSM 传播误差 |
| `isolated_ssm_vs_n3.json` | 相同 U/dt/B/C 输入的局部 SSM 重放误差 |
| `coefficients/` | 六个核的 N0–N3 常量和 256 地址系数表 |

分类精度读取 `test` 范围；全场景数值一致性读取 `scene` 范围。
`integer_vs_n3` 比较后端标签，`integer_vs_qat` 比较部署与 QAT 模型标签。
logit 误差同时提供码字单位与乘 head scale 后的物理单位。

```bash
python software/run_backend_error_four_datasets.py --help
```
