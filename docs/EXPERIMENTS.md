实验清单与运行顺序
====================

所有命令从仓库根目录执行。先完成环境安装和数据检查，再按需要运行下面的阶段。
每个入口都支持 `--help`；`--dry-run` 只打印命令，不启动训练或模拟。不同任务必须
使用不同的输出目录。

一、通用路径
------------

    export DATA_ROOT=/absolute/path/to/data
    export FP32_ROOT=/absolute/path/to/results/SPATIAL_SPLIT_3WAY_DENSE
    export DEVICE=cuda:0
    export SEEDS=0,1,2,3,4,5,6,7,8,9

当前部署配置目录：

    current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16

默认数据集为 UP、HanChuan、HongHu、Houston。训练阶段通常使用十个 seed；诊断阶段
默认使用 seed 0 或 0,6。FP32_ROOT 必须指向包含四个数据集配置目录的父目录，不能
直接指向 run_seed0。

二、完整复现主流程
------------------

1. 训练 FP32 模型并生成预处理、空间划分和 checkpoint：

    bash experiments/01_fp32_current.sh --dry-run
    bash experiments/01_fp32_current.sh

2. 运行网络结构配置实验（19 个配置）：

    bash experiments/02_architecture.sh --dry-run
    bash experiments/02_architecture.sh

   只运行一个配置时，例如：

    bash experiments/02_architecture.sh --cases 10_restore_D --datasets UP --seeds 0

3. 训练 shared-A 与 per-channel-A 配对模型，并分析极点和表容量：

    bash experiments/03_shared_a.sh --dry-run
    bash experiments/03_shared_a.sh

4. 运行评估 batch=1 的 QAT。训练 batch 为 32，验证、选模、测试和整数参考为
   batch 1；freeze20 表示第 20 个 epoch 冻结 BN 统计量和 LSQ 尺度：

    bash experiments/13_qat_eval1.sh --dry-run
    bash experiments/13_qat_eval1.sh

5. 对相同 QAT checkpoint 做 FPGA 定点数值模拟：

    bash experiments/14_fpga_eval1.sh --dry-run
    bash experiments/14_fpga_eval1.sh

   首次运行先做一个小测试：

    bash experiments/14_fpga_eval1.sh --datasets UP --seeds 0 --output-root ./results/fpga_eval1_UP_smoke

   模拟器默认使用保存的数值配置：dt 输入 9 bit、dt 输出地址 8 bit、K 总位宽
   19 bit、K 小数位上限 24、状态 Q24。主结果不要随意覆盖这些选项。

三、结构和量化选型实验
----------------------

以下实验用于复现网络选择和 QAT 初始化/冻结策略：

    bash experiments/04_initialization.sh --init-kind weights
    bash experiments/04_initialization.sh --init-kind groups --output-root ./results/04_quant_groups
    bash experiments/05_stability.sh
    bash experiments/06_max_init.sh
    bash experiments/07_freeze.sh

04 是不训练的初始化对照；06 比较 patch_max 和 all_weight_max；07 比较 freeze1 和
freeze20。正式部署配置使用 patch embedding max 初始化和 D mean 初始化。不同阶段
的模型不能仅按目录名称合并，应同时读取 result.json 中的配置。

四、旧版 eval8、误差来源和回放
------------------------------

需要复现旧版 batch=8 结果时：

    bash experiments/08_qat_eval8.sh --dry-run
    bash experiments/08_qat_eval8.sh
    bash experiments/09_fpga_eval8.sh --dry-run
    bash experiments/09_fpga_eval8.sh

误差来源和读出再量化扫描：

    bash experiments/10_error_sources.sh --suite sources --dry-run
    bash experiments/10_error_sources.sh --suite sources
    bash experiments/10_error_sources.sh --suite requant --output-root ./results/readout_requant
    bash experiments/10_error_sources.sh --suite k-precision --output-root ./results/K_precision

`sources`、`requant` 和 `k-precision` 是不同实验。误差来源的 OA 差异不能简单相加；
每次扫描都要保存完整命令、配置和输出目录。

五、GPU 批量测速
----------------

FP32 和硬件定点语义测速：

    BATCH_SIZES=1,2,4,8,16,32,64 bash experiments/11_gpu_fp32.sh
    BATCH_SIZES=1,2,4,8,16,32,64 bash experiments/12_gpu_fixed.sh

定点测速入口默认使用已归档的 `qat_eval8_4datasets` checkpoint，和
`evidence/gpu_fixed_batch_summary.csv` 中的记录一致。如果要测速另一套 QAT checkpoint，
请传入包含 `{dataset}` 和 `{seed}` 的模板：

    QAT_TEMPLATE=/absolute/path/to/qat_eval1_4datasets/{dataset}/run_seed{seed} \
      bash experiments/12_gpu_fixed.sh --output-root ./results/12_gpu_fixed_eval1

定点测速采用宽整数/FP64 功能模拟，验证硬件舍入、饱和和状态反馈规则，不等于原生
INT8 Tensor Core 性能。GPU 报告包含 model-only 和完整流程两个范围，不能把不同范围
的行当成同一指标。

六、UP GPU 功耗
---------------

    python -m pip install nvidia-ml-py
    DEVICE=cuda:0 BATCH_SIZES=1,2,4,8,16,32,64 \
      bash experiments/16_gpu_power.sh --allow-display-processes \
      --output-root ./results/gpu_power_UP

测量前停止其他计算任务。NVML 给出整卡设备功耗和能量，不是墙上插座功耗，也不是
单个进程的功耗。若允许 Xorg/Xwayland 显示进程，报告会保留该条件标记。

七、dt 输入、batch 差异和局部 SSM 分析
---------------------------------------

dt 输入局部诊断：

    bash experiments/15_dt_inputs.sh --datasets UP --seeds 0 --output-root ./results/dt_inputs_UP

训练事件回放：

    bash experiments/17_replay_jump.sh --event-dir /absolute/path/captured_event --mode native
    bash experiments/17_replay_jump.sh --event-dir /absolute/path/captured_event --mode tf32-off

局部 SSM 误差：

    bash experiments/18_local_ssm.sh --inputs-glob '/absolute/path/ssm_inputs/*.npz' --suite sources

batch 差异定位：

    bash experiments/19_batch_diagnosis.sh --prediction-dir /absolute/path/simulation

这些入口必须使用同一 checkpoint 的预测、输入轨迹和配置。dt 局部探针不能直接当作
整网 dt 输出位宽消融；要做整网消融，应重新训练并保存独立结果目录。

八、D1 N0–N3 非线性对照
------------------------

    bash experiments/20_nonlinear_D1.sh \
      --qat-run-dir /absolute/path/historical_D1/run_seed0 \
      --fp32-dir /absolute/path/matching_FP32/configuration \
      --data-path /absolute/path/data --device cuda:0 \
      --output-dir ./results/20_nonlinear_D1 --dry-run

去掉最后的 `--dry-run` 执行完整 UP 场景。该入口使用固定的历史 D1 checkpoint 和
N0–N3 软件快照，不能替换成最新 eval1 checkpoint。完整文件位置和哈希见
docs/COMPLETION_20260923.md。

九、检查结果
------------

    python tools/verify_release_records.py
    python tools/verify_completion_records.py
    python tools/verify_completion_records.py --check-arrays
    python tools/paired_statistics.py

这些命令检查公开文件、报告哈希、统计汇总、数组差异和功耗汇总，不会自动下载数据或
checkpoint。完整数字证据在 evidence/；不同实验可能复用同一模型，统计时不要重复计数。
