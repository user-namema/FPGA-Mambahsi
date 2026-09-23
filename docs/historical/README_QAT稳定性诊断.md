# D1 QAT 精度跳变：验证、对照与运行方法

本页说明 fixed 对照的稳定性诊断流程。需要捕获训练跳变时，请按
[QAT 三阶段定位](README_QAT三阶段定位.md) 的步骤执行。
fixed 保住最终测试精度但仍有训练跳变；新流程先定位 epoch0 损失，再捕获跳变，最后做独立参数对照。下文保留上一轮实验说明。

更新：2026-09-16。保留 D 的 `max(abs(D))/qmax` 初始化。本次修订增加可观测性与配对训练控制，不宣称已经找到服务器 CUDA 根因或恢复真实数据精度；未修改 RTL。

## 1. 目前日志可以确认的事实

- 最大值初始化已生效：D8 校准尺度约 0.008～0.010。
- 新旧十 seed 平均测试 OA 分别为 94.1161% 与 95.3503%，下降约 1.2342 个百分点。更小的 D 参数量化误差没有保证更高分类精度。
- 新 seed6：FP32 OA=97.2087%，QAT OA=86.9000%，下降 10.3088 个百分点。所选 epoch60 的验证 OA=95.7399%，验证 mAcc=95.7619%。验证与测试是不同空间区域，二者差值不能直接证明 checkpoint 加载错误。
- 新 seed2 从旧 QAT 的 89.4195% 升到 97.2828%；seed6 从旧 QAT 的 95.9078% 降到 86.9000%。这种相反变化要求检查优化过程与泛化，不能只看某一个 seed 给初始化定论。
- 多个 seed 在 epoch20 冻结 BN 时损失骤升，且冻结后仍跳变。BN 统计量切换、持续更新的尺度/权重/BN 仿射参数均是待区分因素。
- 原日志没有 epoch0 验证、每轮尺度变化、逐类验证结果或重新加载后的验证复算，无法确认 10.31 个百分点损失的具体机制。

## 2. 新增内容

`train_mambahsi_spatial_split_dense_qat.py`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--freeze_lsq_epoch` | -1 | 从指定轮开始固定所有激活、权重及 D 的尺度；1 表示校准后第一轮起固定；0 与 1 等效；-1 不固定 |
| `--weight_scale_lr_multiplier` | 1 | 普通权重尺度的学习率倍率 |
| `--d_scale_lr_multiplier` | 1 | D 尺度的独立学习率倍率 |
| `--lr_schedule` | constant | constant 或 cosine |
| `--min_lr_ratio` | 0.1 | cosine 最后一轮与初始学习率之比 |
| `--include_calibrated_baseline` | 关闭 | 将 epoch0 校准后 QAT 纳入验证集最佳模型候选 |
| `--training_diagnostics` | 关闭 | 在每次验证时保存尺度/参数漂移及固定验证 batch 的裁剪比例 |

原 `--activation_scale_lr_multiplier`、`--dt_scale_lr_multiplier` 和 `--freeze_bn_epoch` 保留。冻结尺度不冻结模型权重、D 或 BN γ/β；冻结 BN 统计量也不冻结 γ/β。固定尺度对照用于诊断，不预设其精度一定优于 LSQ 学习。

默认训练超参数不变；所有新训练都会记录 epoch0 验证及验证历史，但只有显式启用 `--include_calibrated_baseline` 才允许 epoch0 被选中。不要将启用/禁用该选项的结果混为同一选择协议。

每个 run 新增：

- `calibrated_baseline_validation.json`：训练前 QAT 验证指标，分离初始量化损失与微调损失。
- `qat_training_history.jsonl`：每个被评估轮次的完整验证指标、混淆矩阵、逐类准确率、学习率与冻结状态。
- `qat_training_diagnostics.jsonl`：启用诊断时记录全部量化尺度、相对上一记录点的变化、参数组最大变化、BN running statistics 漂移，以及固定验证 batch 的量化器输入裁剪比例、输出零值比例、D 占用码数。这里的裁剪比例是超出量化端点的输入比例，不是 FPGA 状态饱和率；固定 batch 也不代表全数据分布。
- `best_validation_recheck.json`：加载所选 checkpoint 后复算验证集，要求混淆矩阵完全一致；不一致时停止并保留报告，先排查复现问题。混淆矩阵一致不等于每个预测像素完全相同。

选择仍只用验证集 mAcc，不按测试集挑模型。不使用测试数据校准尺度。epoch0 候选只能保证最佳验证选择不差于初始验证候选，不能保证测试精度不下降。

## 3. 先运行两组对照，seed0 和 seed6

更新软件包后，在服务器项目目录执行：

```bash
CONFIG_D1=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
FP32_D1="results/SPATIAL_SPLIT_3WAY_DENSE/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"

python run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --variants control_bn20 bn1 \
  --output-dir ./qat_stability_bn_seed0_6
```

共四次训练；保持 batch32/eval batch8、dt9/out8、D8/shared-U、lr=1e-5，唯一对照变量是 BN 统计量从第20轮还是第1轮冻结。所有组统一启用 epoch0 候选和逐轮诊断、逐轮验证。由于验证间隔改为1且有epoch0候选，控制组不是旧日志选择协议的逐字复现，应与同批实验配对比较。

随后比较尺度更新：

```bash
python run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --variants bn1_fixed_scales bn1_slow_scales \
  --output-dir ./qat_stability_scales_seed0_6
```

将第二批与第一批的 bn1 比较。fixed_scales 从第一轮固定校准尺度；slow_scales 保持模型参数 lr=1e-5，将所有 LSQ 尺度（包括 D）lr 降到1e-6。

如果上述实验仍不稳定，可运行综合恢复候选：

```bash
python run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --variants bn1_low_lr_cosine \
  --output-dir ./qat_stability_low_lr_seed0_6
```

该候选同时使用全局 lr=1e-6 和 cosine→1e-7，是组合策略，不应把其效果归于单一变量。

各运行目录必须为空。默认运行100轮；`--max-epoch` 可覆盖。`--dry-run` 仅生成计划，正式运行需要另一个新目录。未指定 `--variants` 时运行前四组，共2 seeds×4组=8次训练。

每组的 `training.log` 保存日志，`training/` 下保存独立 QAT 产物。顶层 `validation_summary.csv` 汇总验证基线、最佳轮次、验证 OA/mAcc，以及记录点的全程/第25轮后 OA 范围。**它不按测试精度排名或选择配置。** 详细混淆矩阵与诊断在各 run 子目录中。

## 4. 如何判读

1. epoch0 已很差：重点检查初始量化、激活分布、共享U及 eval 数值路径，不能只降低微调学习率。
2. epoch0 好，control_bn20 前期恶化而 bn1 稳定：支持 BN 统计量更新/切换参与不稳定。
3. bn1 波动，而 fixed_scales 或 slow_scales 显著减小波动：支持尺度更新参与不稳定；查看同一轮各尺度相对变化与各类准确率变化。
4. 固定尺度仍波动：继续检查模型权重、D/BN仿射参数更新、优化步长和已知 batch 数值差异，不能继续把责任全部归给 LSQ。
5. 重载验证不一致：先读取 recheck 文件并跑 batch 诊断，不将结果作为正常泛化损失。
6. 重载验证一致但测试明显下降：属于同模型在不同空间区域的表现差异，需要逐类/分布分析。测试结果只能描述，不能用来事后挑 epoch。

已有 batch 逐层工具 `diagnose_qat_batch.py` 仍可使用，见 `README_batch差异_再量化_K扫描.md`。现有 batch 差异尚未由这份日志定位，不能直接认定是 TF32 或 scan 导致。

## 5. 范围

保留旧 checkpoint 推理兼容、D 最大值初始化、原 FPGA 数值配置。新增的是诊断和可选训练控制；默认主模型 forward、FPGA 模拟和 RTL 未改变。正式 UP 新实验及服务器 CUDA 验证需执行后确认，不以本地合成流程测试替代。
