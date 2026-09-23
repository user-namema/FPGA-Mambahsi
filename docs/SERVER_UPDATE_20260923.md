# 2026-09-23 补件与精确版本回放

## 已接收并核对

| 内容 | 本次状态 | 公开位置 |
|---|---|---|
| eval1 FPGA 模拟 | 四数据集各十 seed，40 个 checkpoint 和 QAT result 哈希与本地归档一致 | `evidence/server_update_20260923/fpga_eval1/` |
| 保存 QAT 预测复现 | 40/40 均为 100%，reference batch=1 | 同上每 seed 的原始报告 |
| FPGA 汇总 | 从 40 个报告复算均值、样本标准差，与 CSV/JSON 一致 | 同上 `dataset_summary.json`、`per_seed_summary.csv` |
| UP GPU 功耗 | 42 trial，84 份 power/idle 采样 CSV，14 组汇总 | `evidence/server_update_20260923/gpu_power_UP/` |
| 实测环境 | Python/PyTorch/CUDA/包版本、安装来源、扩展哈希 | `environment/observed_20260923/` |
| 实际 Mamba Python 包 | 14 个 .py、许可证、作者信息及安装元数据；三个环境采样源码哈希一致 | `third_party/mamba_ssm_server_1_1_2/` |
| 实测模拟器版本 | 保存模拟器与驱动的精确快照 | `software/snapshots/eval1_server_20260923/` |

所有源码按原字节保留。结果文本中的服务器工程路径、环境路径、用户目录分别替换为 `$PROJECT_ROOT`、`$CONDA_PREFIX`、`$SERVER_HOME`。`provenance.json` 同时记录输入文件和公开文件的 SHA256。`audit.json` 保存打包检查结果。未纳入 checkpoint、数据集、预测数组和大量 ROM/activation 导出；它们可由运行代码重新生成。

## 精确复用本次 FPGA 模拟器

本次实测模拟器没有后来加入的可选 dt 输入诊断。仓库主版本保留该功能；下列命令生成一个使用实测源码的独立软件目录，先校验所有记录的依赖哈希，再复制/覆盖两个快照文件。目录已存在时会停止。

```bash
python tools/prepare_observed_eval1.py --output-dir ./results/observed_eval1_software
export DATA_ROOT=/absolute/path/MambaHSI/data
export FP32_ROOT=/absolute/path/MambaHSI/results/SPATIAL_SPLIT_3WAY_DENSE
export QAT_ROOT=/absolute/path/MambaHSI/QAT_eval1_GPU_power_20260921/qat_eval1_4datasets

python results/observed_eval1_software/run_fpga_qat_eval1_four_datasets.py \
  --project-root "$PWD/results/observed_eval1_software" \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" --device cuda:1 \
  --datasets UP HanChuan HongHu Houston --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-dir ./results/fpga_eval1_recorded_replay
```

不跨源码版本复用 `--resume` 输出目录。新 dt 诊断使用主版本 `experiments/15_dt_inputs.sh`，另存结果。环境重建和 CUDA scan 检查见 `ENVIRONMENT.md`；本地 macOS 上没有重跑服务器的完整 CUDA 实验。

## 最新精度摘要

| 数据集 | QAT OA 均值 (%) | FPGA 整数 OA 均值 ± 样本SD (%) | QAT−FPGA (pp) |
|---|---:|---:|---:|
| UP | 96.4512 | 96.3853 ± 0.8770 | 0.0659 |
| HanChuan | 91.5552 | 91.4164 ± 0.9699 | 0.1388 |
| HongHu | 92.8104 | 92.7586 ± 1.2150 | 0.0518 |
| Houston | 92.1006 | 91.5403 ± 2.4392 | 0.5603 |

每行十 seed。这是 eval1 QAT 的新闭环，和旧 eval8 的结果分开保存；不覆盖旧数据表。

## 功耗设置与复现

本次是 UP FP32 seed0，GPU0 RTX 4090（PCI `0000:17:00.0`），TF32 关闭、optimized selective scan。输入驻留 GPU；每次完整场景循环后同步，不计 PCA、H2D 或磁盘操作。`model_only` 与 `full_gpu_pipeline` 必须分别比较。

```bash
DEVICE=cuda:0 BATCH_SIZES=1,2,4,8,16,32,64 \
  bash experiments/16_gpu_power.sh --allow-display-processes \
  --output-root ./results/gpu_power_UP_display
```

默认测量至少30秒、预热10秒、idle5秒、三次重复，与本次设置一致。测量以完整场景为单位，实际时长可能略超30秒。能量主指标来自 NVML energy counter，平均功耗=能量/实测时长，能量/tile=能量/tile数；汇总取每次 trial 的指标均值。

42 次均为 `display_background_only`，首尾检查无其他计算进程，保留两个 Xorg 显示进程；`exclusive_at_boundaries=false`。功耗是整卡含显示后台的观测，没有扣除显示进程或 idle，也不是主机插座功耗。与旧含其他 Python 作业的共享 GPU smoke 分开。model-only batch1 为 70.259 W、505.087 tile/s、139156.410 µJ/tile；batch64 为 336.200 W、27046.906 tile/s、12432.551 µJ/tile（三次均值）。旧 FP32/定点测速虽已确认使用同一张 RTX 4090，本次功耗需以这里的 GPU UUID 为准，不仅凭型号认定为同一物理卡。

## 仍需补齐的材料

已收到的 `gpu_current_4datasets` 收集目录只有早期失败日志。用户确认新结果在服务器 `QAT_eval1_GPU_power_20260921` 下；该位置与本次已收到的 UP 功耗结果不等同于四数据集旧 FP32 测速的 28 份原始报告。应从正确子目录再收集，命令见 `SERVER_FILES.md`。dt 诊断、N0–N3 Python 对照仍未收到。

Mamba Python 源码已收到；两个 cu118/cp38 wheel 和对应 CUDA 扩展二进制未收到。环境记录有它们的来源与哈希，不能把 Python 源码快照称为完整可安装 wheel。
