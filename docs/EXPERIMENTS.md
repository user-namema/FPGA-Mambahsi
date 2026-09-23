# 实验清单与运行顺序

核对日期：2026-09-23。依据 20260921 审稿修订稿及本机结果文件。下表的“完成”表示存在对应运行结果；不表示本次打包重新训练过。原始文件索引及 SHA256 见 `evidence/reference_run_index.csv`，计数见 `reference_run_counts.json`。

## 已完成清单

| 入口 | 实验 | 已完成规模/证据 | 依赖 |
|---|---|---|---|
| 01/02 | FP32 网络结构选型 | 19 配置×4 数据集×10 seed=760 份 result；当前 D1 是其中一个配置 | 原始数据 |
| 03 | 当前 D1/z0 下共享 A 与逐通道 A 配对重训 | 4×2×10=80 份 result | 当前 FP32 保存的划分/预处理 |
| 03 | 极点投影、SVD、实际 Δ 轨迹、A 表 16/20/24 小数位误差 | 对应 80 个模型、每模型六核 | 03 配对模型 |
| 04 | mean、patch_max、all_weight_max，不训练完整验证 | UP seed0/6，共六组比较（两份多组汇总） | D1 FP32 |
| 04 | 初始量化分层/分组诊断 | UP 两个 seed | D1 FP32 |
| 05 | BN 冻结时机、固定/慢尺度、低权重 LR、BN affine 单因素 | 原 BN 四次、尺度四次、单因素六次；有重用对照 | D1 FP32 |
| 06 | patch_max/all_weight_max 校准及 20 epoch 微调并模拟 | 两阶段×两策略×两 seed=8 模型及 8 份模拟 | D1 FP32 |
| 07 | patch_max+D_mean，freeze1 与 freeze20 | UP 每种十 seed，共20次；freeze20 与08的UP重合 | D1 FP32 |
| 08 | 论文旧 QAT：eval8 选模 | 四数据集×十 seed=40 | D1 FP32 |
| 09 | 上述 40 个 QAT 的 FPGA 整数模拟 | 40 份模拟结果及导出 | 08 对应 checkpoint |
| 10/18 | D1 单误差源整网与固定输入回放 | UP seed0，none/a/k/state/d/all/separate 七组；局部回放8个test tile×六核 | 历史 shared-U D1 checkpoint；捕获的 U/dt/B/C |
| 11 | FP32 GPU 多 batch 测速 | 4×7=28 设置；本地只有112行汇总，原始报告待补 | D1 FP32，RTX 4090 |
| 12 | 定点数值语义 GPU 测速 | 4×7=28 原始报告、112行汇总 | 旧 eval8 QAT，RTX 4090 |
| 13 | 最新 QAT：eval1 选模、测试 | 四数据集×十 seed=40 | D1 FP32 |
| 14 | eval1 FPGA 整数模拟 | 40 份完整报告，checkpoint 哈希与 QAT 归档一致；保存预测均 100% 复现 | 13 对应 checkpoint |
| 16 | UP FP32 GPU 功耗 | 7 batch×2 scope×3 repeat=42 次，原始采样/报告/汇总齐全；Xorg 显示后台 | UP FP32 seed0，GPU0 RTX 4090 |
| 17 | 训练跳变捕获、同事件后端回放 | 两 seed 捕获；seed6 native/reference/TF32-off 三种回放 | 捕获目录中的 checkpoint 与事件记录 |
| tools | 共享 A 代价、旧 QAT 相对 FP32 的配对统计 | 两比较×四数据集，每组十 seed | 已附数值 evidence |

19 个结构配置：原始容量参考 R2、容量匹配参考 R1、D1z1、D1z0、D0z1；D0z0 基准及其 mean/softmax 融合、残差系数1/0、GN-ReLU、BN-SiLU、GN-SiLU、head32/128、token2/8、state8/32。历史 D0 单因素不能改称当前 D1 单因素。单谱分支不纳入本清单及默认结构入口。

不同表格可重复使用同一模型或结果，不能把所有数量相加当作独立样本数。原来的 `current` 文件名代表当时 D0 基准；当前正式网络由完整配置中的 **D1/z0** 识别。

## 尚未有完整本地结果

| 入口/项目 | 状态 |
|---|---|
| 15 dt 输入诊断 | 代码已提供；待补输出，INT9应检查量化前分布/超界比例/误差，不凭最终范围判定溢出 |
| 19 B1/B8 首差异定位 | 代码已提供；未找到完整定位报告；17的训练跳变回放不是同一实验 |
| 10/18 requant、K/state/accumulator位宽扫描 | 扫描入口已提供；未找到完整结果，不能列为已完成 |
| dt输出6–10位×十seed | 用户报告已完成；当前归档缺少配置与结果，待服务器补件 |
| N0–N3整网非线性比较 | 论文已有数值；本机缺产生该结果的完整Python实现/驱动及原始记录，需另一台电脑补软件部分 |

## 通用参数与目录

所有命令从仓库根目录运行。`--dry-run` 只打印完整命令，不启动训练，也不创建结果目录。参数优先于同名环境变量。默认四数据集为 UP、HanChuan、HongHu、Houston；诊断入口默认 UP。正式训练默认十 seed，选型诊断默认0/6，GPU和误差诊断默认seed0。每个入口可用 `--help` 查看通用参数。

```bash
export DATA_ROOT=/absolute/path/data
export FP32_ROOT=/absolute/path/results/SPATIAL_SPLIT_3WAY_DENSE
export DEVICE=cuda:1
# 所有路径模板必须加引号，保留占位符。
export FP32_TEMPLATE="$FP32_ROOT/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
```

已有旧 QAT 的四数据集如果不在同一模板下，分别传 `--datasets` 和 `--qat-template`。例如只复用 UP freeze20：

```bash
QAT_TEMPLATE='/absolute/path/freeze20/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_patch_max_D_mean_freeze20/current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16/run_seed{seed}' \
  bash experiments/09_fpga_eval8.sh --datasets UP --output-root ./results/UP_old_sim
```

请以实际目录为准；不要把 `run_seed0` 当成 FP32 数据集配置根目录。默认各阶段结果放 `results/编号名称`，01放`results`，08放`results/qat_eval8_4datasets`，13放`results/qat_eval1_4datasets`。完整子命令与失败末尾日志保存在 `results/launch_logs/`。

## 01–03：网络结构与 A

```bash
# 如果只需最终网络，运行01；若已跑完整02，其10_restore_D就是01。
bash experiments/01_fp32_current.sh
bash experiments/02_architecture.sh

# 配对训练完成后自动执行80个模型的A分析。
bash experiments/03_shared_a.sh
# 已有配对结果时，只运行分析：
bash experiments/03_shared_a.sh --phase analyze --pair-root /absolute/path/a_shared_current_4datasets --output-root ./results/A_reanalysis
```

02 默认执行19配置。限定一个选型：`--cases 10_restore_D --datasets UP --seeds 0`。03比较的是当前 D1/z0 下重新训练的 shared/per-channel 配对，不用早期 D0 的 per-channel 代替。A 表小数位误差分析与网络重训的表示能力代价是两类证据。

## 04–07：量化选型

```bash
bash experiments/04_initialization.sh --init-kind weights
bash experiments/04_initialization.sh --init-kind groups --output-root ./results/04_quant_groups
bash experiments/05_stability.sh
bash experiments/06_max_init.sh
bash experiments/07_freeze.sh
```

04只校准并完整验证，不做梯度更新；weights比较三种普通权重初始化，D初始化保持该历史诊断中的 max 设置。06比较 patch_max 与 all_weight_max，校准基线与20轮训练均跑模拟。07及后续正式QAT改为选定的 D_mean。对照中需同时记录 D 初始化，不能只按普通权重策略名称拼表。

05默认合并已有的六种控制设置，部分原始目录重复保存过同一控制。各设置保持脚本的 LR、BN/尺度开关；不要因为统一存放就解释成一个全因子实验。

## 08–10：旧 QAT 与整数模拟

```bash
bash experiments/08_qat_eval8.sh
bash experiments/09_fpga_eval8.sh
bash experiments/10_error_sources.sh --suite sources
```

10默认使用08生成的 UP seed0。若要复现论文早期七组误差源结果，须显式提供历史 `UP_D1_sharedU_D8_dtin9_dtout8` 模型；其 checkpoint SHA256 为 `d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33`。换成新模型是新实验，不能当作同一模型复算。

数值误差表必须区分：保存QAT与复算QAT协议、QAT-direct与staged-FP参考分解、staged-FP与整数硬件语义。none只关闭指定SSM误差源，网络其他量化仍存在；各来源OA差不能简单相加。sources的auto读出设置和独立requant扫描的解释见 `docs/historical/README_batch差异_再量化_K扫描.md`。

## 11–12：GPU速度

```bash
BATCH_SIZES=1,2,4,8,16,32,64 bash experiments/11_gpu_fp32.sh
BATCH_SIZES=1,2,4,8,16,32,64 bash experiments/12_gpu_fixed.sh
```

FP32默认20次repeat、100次warmup、200个单tile trial；定点默认5次repeat、5次warmup、20个单tile trial。原始报告保留model-only/end-to-end、CUDA-event/wall口径，112行汇总不等于112次独立配置。两种现有结果均为同一张 RTX 4090。定点GPU路径使用宽整数/FP64模拟定点算术，不是优化INT8推理库。

## 13–16：最新eval1与补实验

```bash
bash experiments/13_qat_eval1.sh
bash experiments/14_fpga_eval1.sh

# 仍执行完整模拟，同时输出dt输入分析，另存目录。
bash experiments/15_dt_inputs.sh --datasets UP --seeds 0

# 默认必须独占GPU；NVML功耗是整卡观测，不是单进程计费。
bash experiments/16_gpu_power.sh
```

14/15按`QAT_ROOT`中的result元数据选择eval1模型，严格复现保存预测，继承模型的位宽与定点配置。D旁路在C归约后加D×U再统一量化。K默认19位及小数位上限24不是“越大越好”的最终结论；扩位只应单独运行消融，不覆盖基准。

16默认每batch测量30秒、预热10秒、idle5秒、重复3次、采样100毫秒。检测到其他GPU任务会停止。若研究共享场景，可直接使用底层`software/benchmark_up_gpu_power.py --allow-other-processes`，结果必须标为共享整卡功耗，不能填入独占模型能耗对比。

如果只有 Xorg/Xwayland 显示进程，可用 `bash experiments/16_gpu_power.sh --allow-display-processes`。此选项仅放行已识别的纯图形显示进程，仍拒绝其他计算任务及未知图形任务。输出会保留 `exclusive_at_boundaries=false`，另记录 `no_other_compute_at_boundaries=true` 与 `process_condition=display_background_only`（以实际查询为准）。这表示检查时点未见其他计算进程，不能证明整段时间无短暂任务，也不能分离显示进程的功耗。

## 17–19：事件与输入回放

```bash
bash experiments/17_replay_jump.sh --event-dir /absolute/path/captured_event --mode native
bash experiments/17_replay_jump.sh --event-dir /absolute/path/captured_event --mode tf32-off --output-root ./results/jump_tf32off
bash experiments/17_replay_jump.sh --event-dir /absolute/path/captured_event --mode reference-tf32-off --output-root ./results/jump_reference

bash experiments/18_local_ssm.sh --inputs-glob '/absolute/path/sim/ssm_inputs/*.npz' --suite sources
bash experiments/19_batch_diagnosis.sh --prediction-dir /absolute/path/sim
```

18的glob应指向模拟器实际导出的NPZ位置，可先列出文件再填写。只有replay汇总CSV不能恢复U/dt/B/C输入。要重新捕获，调用底层模拟器并传`--capture-ssm-inputs --ssm-capture-tiles 8 --ssm-analysis-split test`，同时提供同一QAT/FP32/数据路径。

19需同一旧checkpoint的`qat_direct_prediction.npy`、`qat_saved_batch_reproduced_prediction.npy`、`fpga_simulation_result.json`。其QAT路径由`QAT_TEMPLATE`指定。不能拿eval1模型与旧eval8模型的预测比较来定位同一模型的batch差异。

以下扫描已有代码但未列为完成，运行时另建目录：

```bash
bash experiments/10_error_sources.sh --suite requant --output-root ./results/readout_requant
bash experiments/10_error_sources.sh --suite k-precision --output-root ./results/K_precision
bash experiments/18_local_ssm.sh --inputs-glob '/absolute/path/inputs/*.npz' --suite k-precision --output-root ./results/K_local
```

## 数值汇总与追溯

```bash
python tools/paired_statistics.py
# 对作者原始归档重新核对文件数和校验和；不加载checkpoint。
python tools/inventory_results.py --workspace /absolute/path/archive --output-dir /tmp/result_index
```

配对统计计算十seed的均值差、95% Student-t区间、精确双侧符号翻转检验，并对八个主比较做Holm校正。`QAT_loss_FP32_minus_QAT8`的符号为FP32减旧QAT，不能直接套在新eval1上。eval1已另附40seed摘要与一致性核验记录。

2026-09-23 新增结果的数值表、精确源码版本和功耗范围见 [补件说明](SERVER_UPDATE_20260923.md)。保留 Xorg 的完整功耗结果与早期含其他 Python 进程的 shared smoke 不是同一组实验。
