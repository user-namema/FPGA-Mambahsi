# FPGA-MambaHSI：软件实验复现

本目录整理论文已有的训练、QAT、共享 A 分析、FPGA 数值模拟和 GPU 测试代码。RTL、Vivado 工程、数据集和 checkpoint 不放入本次源码包。

## 环境与数据

从新建 `mambahsi` 环境开始，按 [环境说明](docs/ENVIRONMENT.md) 安装；数据位置、MAT 变量名及上游 HongHu NPY 格式的转换见 [数据说明](docs/DATA.md)。官方 MambaHSI 推荐 Python 3.9；你服务器的历史环境是 Python 3.8，本次已收到完整依赖版本记录及实际 Mamba Python 包。实测环境为 Python 3.8.20、PyTorch 1.13.1/CUDA 11.7、Mamba 1.1.2；下面 Python 3.9 的方案是重建环境选项，不能当作实测环境。精确版本见环境说明。

```bash
conda create -n mambahsi python=3.9 -y
conda activate mambahsi
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# 继续执行 docs/ENVIRONMENT.md 中的依赖安装与 CUDA/scan 检查。
```

## 使用服务器现有模型

进入本仓库后设置绝对路径，就可以复用服务器现有数据与 FP32 模型，不必复制到仓库，更不必重训 FP32。

```bash
export DATA_ROOT=/home/music/mzz/MambaHSI/data
export FP32_ROOT=/home/music/mzz/MambaHSI/results/SPATIAL_SPLIT_3WAY_DENSE
export QAT_ROOT=/home/music/mzz/MambaHSI/QAT_eval1_GPU_power_20260921/qat_eval1_4datasets
export DEVICE=cuda:1
export SEEDS=0,1,2,3,4,5,6,7,8,9

# QAT_ROOT 请改为当前实际目录；先预演最新模型的 FPGA 模拟。
bash experiments/14_fpga_eval1.sh --dry-run
bash experiments/14_fpga_eval1.sh
```

需要重新训练最新 QAT 时运行 `13_qat_eval1.sh`，默认结果在本仓库 `results/qat_eval1_4datasets`。训练 batch 为 32，验证、选模、测试 batch 为 1。`freeze20` 是第 20 个 **epoch** 冻结 BN 统计和 LSQ 尺度。

先跑 UP seed0 可用：

```bash
bash experiments/14_fpga_eval1.sh --datasets UP --seeds 0 --output-root ./results/fpga_eval1_UP_smoke
```

这会模拟完整 UP 场景。正式四数据集任务请另用输出目录。已经支持恢复的任务可加 `--resume`；未支持的训练入口请使用新输出目录，不能靠复用路径拼接两次训练结果。

## 编号与结果版本

[实验清单与逐项命令](docs/EXPERIMENTS.md) 包含 20 个入口：01–03 是 FP32/结构/共享 A，04–07 是量化选型，08–12 是论文旧版 eval8 与误差/GPU实验，13–16 是新版 eval1、dt 诊断与功耗，17–19 是回放与 batch 定位，20 是历史 D1 N0–N3 完整对照。

目前核对到：结构消融 19×4×10=760 次；共享 A/逐通道 A 配对训练 80 次及对应 80 次分析；旧 QAT 40 次及其 FPGA 模拟 40 次；新 eval1 QAT 40 次及其 FPGA 模拟 40 次；UP GPU 功耗 42 次；FP32 GPU 与定点语义 GPU 各 28 个设置。辅助实验也已在清单中列出。冻结对照与主结果存在复用，不能把所有行简单相加。

新 eval1 模型的 40 份 FPGA 报告已独立归档，保存 QAT 预测全部 100% 复现。UP 功耗包含 7 种 batch × 2 种范围 × 3 次重复，测量条件为保留 Xorg 显示后台、检查时点无其他计算进程。四数据集 FP32 测速原始 JSON、UP seed0 dt 诊断、D1 N0–N3 软件及完整明细均已补齐；见 [最终补件说明](docs/COMPLETION_20260923.md)。

源码放在 `software/`，保留依赖文件相邻关系；统一运行入口放在 `experiments/`，从任意目录调用时均定位到自身仓库。数值核心采用现有代码快照，新增整理层负责路径、命令和日志，不重新定义量化公式。历史说明放在 `docs/historical/`，其中旧路径、D 初始化和旧模型配置只供追溯，实际运行以当前编号入口为准。

## 本次补件的校验与精确回放

```bash
python tools/verify_release_records.py
python tools/prepare_observed_eval1.py --output-dir ./results/observed_eval1_software
```

上述检查仅依赖 Python 标准库，核验已公开文件哈希，复算 40 份 FPGA 报告的均值/样本标准差，以及 42 次功耗的能量、吞吐与汇总。checkpoint 本体未公开，但打包时已逐一与本地 QAT 归档核对哈希。精确回放命令见 `docs/SERVER_UPDATE_20260923.md`。

GitHub 地址：https://github.com/user-namema/FPGA-Mambahsi 。本次更新的是待上传文件，未执行远程推送。

## 最终补件的核验与 D1 对照

```bash
python tools/verify_completion_records.py --check-arrays
bash experiments/20_nonlinear_D1.sh --help
```

20 使用历史 D1 checkpoint（哈希 `d113c612…`），不能使用最新 eval1 的模型代替。
源码快照已包含与实测哈希匹配的两个 MEM 码字比较修订文件。原始报告、系数与
八个预测/logit数组在 `evidence/completion_20260923/nonlinear_D1/`。
原环境运行库已验证并保留本地，Git 中仅放元数据、校验和与轻量依赖源码。
