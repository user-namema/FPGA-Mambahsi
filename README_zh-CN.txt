FPGA-MambaHSI：软件实验复现说明
================================

本仓库包含论文实验所需的软件：FP32 训练、网络结构消融、共享 A 分析、
QAT、FPGA 定点数值模拟、GPU 批量测速和 GPU 功耗测量。RTL、Vivado 工程、
bitstream、数据集和训练 checkpoint 单独保存，不在本仓库中。

一、安装环境
------------

训练、CUDA 测速和功耗测试需要 Linux 与 NVIDIA GPU。在仓库根目录执行：

    conda create -n mambahsi python=3.8.20 -y
    conda activate mambahsi
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    python -m pip install -c environment/constraints-torch113.txt -r environment/requirements.txt
    python -m pip install -c environment/constraints-torch113.txt --no-build-isolation mamba-ssm==1.2.0
    python -m pip check

仓库中的模型支持 PyTorch 参考扫描后端。它可用于检查路径和输出，但不能代替
CUDA 测速。先执行环境检查：

    python tools/check_environment.py --device cpu
    python tools/check_environment.py --device cuda:0 --check-scan --json environment_check.json

如果使用逻辑 GPU 1，把命令中的 cuda:0 改成 cuda:1。可选 FLOP 统计依赖安装：

    python -m pip install -c environment/constraints-torch113.txt -r environment/requirements-optional.txt

二、准备数据
------------

按照 docs/DATA.txt 中的文件名和目录结构放置四个完整场景，然后执行：

    python tools/prepare_data_formats.py --data-root ./data

如果 HongHu 下载的是 NPY 文件，可以转换为训练代码使用的 MAT 文件：

    python tools/prepare_data_formats.py --data-root ./data --datasets HongHu --convert-honghu

不要用裁剪文件或已经 PCA 降维的文件代替原始场景。FP32 阶段生成的 PCA、缩放、
空间划分和 tile 列表会被 QAT 与 FPGA 模拟复用。

三、设置路径
------------

将下面路径替换成实际绝对路径：

    export DATA_ROOT=/absolute/path/to/data
    export FP32_ROOT=/absolute/path/to/results/SPATIAL_SPLIT_3WAY_DENSE
    export DEVICE=cuda:0
    export SEEDS=0,1,2,3,4,5,6,7,8,9

当前部署配置目录名为：

    current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16

四、按顺序运行 FP32、QAT 和 FPGA 模拟
--------------------------------------

所有命令从仓库根目录执行。加上 --dry-run 只打印命令，不启动任务。每个入口会
先检查输入路径，并将日志写入结果目录。

第一步，生成 FP32 预处理、空间划分、配置和 checkpoint：

    bash experiments/01_fp32_current.sh --dry-run
    bash experiments/01_fp32_current.sh

第二步，运行最新的 batch=1 评估协议 QAT。训练 batch 为 32，验证、选模、测试和
整数参考均为 batch 1。freeze20 表示第 20 个 epoch 冻结 BN 统计量和 LSQ 尺度：

    bash experiments/13_qat_eval1.sh --dry-run
    bash experiments/13_qat_eval1.sh
    bash experiments/14_fpga_eval1.sh --dry-run
    bash experiments/14_fpga_eval1.sh

首次运行建议先测试 UP 的一个 seed：

    bash experiments/14_fpga_eval1.sh --datasets UP --seeds 0 --output-root ./results/fpga_eval1_UP_smoke

完整实验清单和每个入口的依赖写在 docs/EXPERIMENTS.txt。不要让两个任务同时写入
同一个输出目录。

五、GPU 测速和功耗
------------------

四个数据集 FP32 GPU 测速：

    bash experiments/11_gpu_fp32.sh --dry-run
    bash experiments/11_gpu_fp32.sh

硬件定点语义 GPU 测速：

    bash experiments/12_gpu_fixed.sh --dry-run
    bash experiments/12_gpu_fixed.sh

UP GPU 功耗测试需要确保 GPU 没有其他计算任务：

    python -m pip install nvidia-ml-py
    DEVICE=cuda:0 BATCH_SIZES=1,2,4,8,16,32,64 \
      bash experiments/16_gpu_power.sh --allow-display-processes \
      --output-root ./results/gpu_power_UP

NVML 测量的是整张 GPU 的设备功耗和设备能量，不是墙上插座功耗。输出报告会保存
GPU 身份、进程快照、预热时间、测量窗口、吞吐率和每 tile 能耗。

六、检查实验记录
----------------

下面的命令不需要数据集或 checkpoint：

    python tools/verify_release_records.py
    python tools/verify_completion_records.py
    python tools/verify_completion_records.py --check-arrays
    python tools/paired_statistics.py

数值记录位于 evidence/，包括四数据集 QAT/整数精度、共享 A 配对统计、GPU 测速、
功耗试验和 D1 非线性对照。不同文件可能重复使用同一个 checkpoint，不能把所有行
简单相加为独立实验次数。

七、代码范围
------------

软件源码在 software/，编号运行入口在 experiments/，安装、数据和证据说明在 docs/
与 evidence/。本仓库不包含 RTL 或 Vivado 工程。

上游网络：
https://github.com/li-yapeng/MambaHSI

仓库地址：
https://github.com/user-namema/FPGA-Mambahsi
