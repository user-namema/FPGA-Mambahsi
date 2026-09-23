已归档结果的精确复现
======================

本页给出仓库中已归档结果的文件位置和运行命令。数据集、checkpoint 和 CUDA 二进制
不随源码仓库发布；运行前请准备相同配置的外部文件。

一、结果位置
------------

四数据集 eval1 FPGA 模拟：

    evidence/server_update_20260923/fpga_eval1/

UP GPU 功耗试验：

    evidence/server_update_20260923/gpu_power_UP/

四数据集 FP32 GPU 原始报告：

    evidence/completion_20260923/gpu_fp32/

UP seed0 dt 输入诊断：

    evidence/completion_20260923/dt_input_UP_seed0/

D1 N0–N3 非线性对照：

    evidence/completion_20260923/nonlinear_D1/

对应源码快照：

    software/snapshots/eval1_server_20260923/
    software/snapshots/nonlinear_D1_20260916/

二、检查公开记录
----------------

不需要 GPU、数据集或 checkpoint：

    python tools/verify_release_records.py
    python tools/verify_completion_records.py
    python tools/verify_completion_records.py --check-arrays

第一条检查发布记录和统计汇总；第二条检查补充结果；最后一条还会读取 NumPy 数组并
复算 logit、标签和哈希。检查失败时先阅读终端输出，不要覆盖原始证据目录。

三、复用 eval1 FPGA 模拟器
--------------------------

先生成与记录匹配的源码目录：

    python tools/prepare_observed_eval1.py --output-dir ./results/observed_eval1_software

设置外部数据、FP32 和 QAT 目录：

    export DATA_ROOT=/absolute/path/to/data
    export FP32_ROOT=/absolute/path/to/results/SPATIAL_SPLIT_3WAY_DENSE
    export QAT_ROOT=/absolute/path/to/qat_eval1_4datasets

运行一个数据集、一个 seed 验证路径：

    python results/observed_eval1_software/run_fpga_qat_eval1_four_datasets.py \
      --project-root "$PWD/results/observed_eval1_software" \
      --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
      --data-path "$DATA_ROOT" --device cuda:0 \
      --datasets UP --seeds 0 \
      --output-dir ./results/fpga_eval1_recorded_UP_seed0

确认输出正确后，再把 `--datasets` 改为 `UP HanChuan HongHu Houston`，把 `--seeds`
改为 `0,1,2,3,4,5,6,7,8,9`，并使用新的输出目录。不要将不同源码版本的输出目录
混合恢复。

四、复用 D1 N0–N3 对照
-----------------------

D1 对照需要匹配的历史 checkpoint、FP32 配置和数据路径：

    export DATA_ROOT=/absolute/path/to/data
    export N0N3_QAT_RUN_DIR=/absolute/path/to/D1/run_seed0
    export N0N3_FP32_DIR=/absolute/path/to/matching/FP32/configuration
    DEVICE=cuda:0 bash experiments/20_nonlinear_D1.sh --dry-run

确认命令后去掉 `--dry-run`，并指定新的 `--output-dir`。该入口用于比较 N0–N3 的
系数生成方式；它不能替代四数据集 eval1 模型的精度实验。

五、结果解释
------------

QAT 和整数结果按四个数据集、十个 seed 汇总。GPU 功耗是 NVML 整卡设备读数；GPU
吞吐是批量执行测量；FPGA 功耗是 Vivado 布线后估算，两者不要合并成同一级别的实测
能效。dt 输入诊断是局部探针，不等于整网 dt 输出位宽消融。RTL、Vivado、bitstream
和板级工程需使用独立硬件工程完成。
