# QAT 初始损失、相邻轮次跳变与单变量对照

更新：2026-09-16。对应上传结果：fixed 的最终测试 OA 接近 FP32，但训练验证 OA 仍能下降 18.68 个百分点；seed 6 在校准后、优化前的验证 OA 已降到 35.69%。因此按下面顺序定位，不再把原因预先归于尺度学习。

## 0. 文件和共同设置

这是基于上一版 QAT/FPGA 工程的增量包，解压到服务器项目目录。依赖工程中已有的 `ssm_error_ablation.py`（保留原文件）。此次新增/修改的运行文件：

- `train_mambahsi_spatial_split_dense_qat.py`：诊断入口、跳变捕获、独立 BN affine 控制及普通参数学习率。
- `qat_forensics.py`：诊断实现，必须与训练脚本放在同一目录。
- `replay_qat_jump.py`：离线复查跳变，不依赖原始数据文件。
- `run_qat_stability_sweep.py`：两组新的单变量对照和捕获参数透传。

FPGA 模拟器和 RTL 本次不修改。D 仍用 `max(abs(D))/127` 初始化，shared-U、dt 输入 9 位 / 输出 8 位不变。新诊断默认关闭，量化开关只在诊断上下文里生效，不保存为部署模型，也不会进入训练 forward。

在服务器执行（FP32_D1 必须是包含 `run_seed0`、`run_seed6` 和预处理文件的配置目录）：

```bash
cd ~/mzz/MambaHSI
conda activate mambahsi
CONFIG_D1=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
FP32_D1="results/SPATIAL_SPLIT_3WAY_DENSE/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"
test -f "$FP32_D1/run_seed6/best_model.pth" || { echo 'FP32_D1 路径不正确'; exit 1; }
```

所有示例都用新输出目录。不要覆盖已跑完的训练或诊断。保持同一服务器、GPU 和软件环境，batch32、eval batch8、calibration100；先用 seed 0、6，不必先重跑全部十个 seed。

## 1. 只定位初始损失：不训练、不评估测试集

```bash
python train_mambahsi_spatial_split_dense_qat.py \
  --dataset UP --data_set_path ./data --fp32_dir "$FP32_D1" \
  --work_dir ./qat_initial_diagnosis_seed0_6 --run_tag initial_diagnosis \
  --seeds 0,6 --device cuda:0 \
  --use_D true --use_z false --ssm-u-quantization shared --d-weight-bits 8 \
  --dt-input-bits 9 --dt-output-bits 8 \
  --batch_size 32 --eval_batch_size 8 --calibration_steps 100 \
  --initial_diagnostics_only
```

`--initial_diagnostics_only` 自动启用初始诊断。它复用 FP32 的预处理、空间划分及每 seed 标签；只用训练集校准，再在完整保存的验证集运行以下 15 种策略，并额外计算原 FP32 参考。

| 策略 | 意义 |
|---|---|
| `none` | 关闭全部 LSQ 和 INT32 bias 量化，保留转换后的模型结构，检查转换等价性 |
| `all` | 正常 QAT 校准后的完整量化路径，epoch 0 |
| `only_weight/activation/dt_input/dt_output/D/bias` | 分别只打开一组 |
| `without_weight/activation/dt_input/dt_output/D/bias` | 从完整量化中分别关闭一组 |
| `all_repeat` | 同一模型、同一验证 batch 划分再次推理，检查重复性 |

所有策略共用同一份完整量化校准得到的尺度，不分别重校准，不做优化步。`none` 在校准之后执行，但完全旁路量化，不用伪造的 32 位尺度；因此检查的是转换后浮点计算与原 FP32 的等价性。

分组边界：

- `weight`：普通投影/卷积权重，包括 dt_proj 权重；BN 层使用实际折叠后的权重。
- `activation`：其余激活边界，包括 patch、共享 U、B/C、各分支/融合/残差/输出；不含下面的 dt 两组。
- `dt_input`：dt_proj 输入的 9 位量化；`dt_output`：dt_proj 加 bias 后的 8 位输出量化。两者独立，不混为一个 dt 消融。
- `D`：D 参数的 LSQ 量化；这里不是 FPGA 的 D 系数折叠舍入。
- `bias`：所有投影、卷积及 BN 折叠 bias 的 INT32 累加器量化。

每个 seed 的目录：

```text
qat_initial_diagnosis_seed0_6/SPATIAL_SPLIT_3WAY_DENSE_QAT/
  UP_initial_diagnosis/<CONFIG_D1>/run_seed6/
    initial_quantization_diagnosis/
      initial_quantization.json
      initial_quantization_summary.csv
```

CSV 的 OA/mAcc 是 0–1；JSON 包含逐类指标、混淆矩阵、FP32 预测一致率、输出误差、分组清单、源码哈希及运行环境。`output_vs_fp32` 比较完整验证 tiles 的模型输出（包括未标注位置，插值前），准确率只统计验证标签。

判读顺序：

1. 先看 `conversion_within_tolerance`、`conversion_predictions_match`。若 `none` 都不匹配，先分析转换或浮点后端；容差默认 atol=1e-6、rtol=1e-5，少量容差失败本身不是代码错误证明。
2. 看 `all_repeat.output_vs_all` 和 `prediction_match_all`。同 checkpoint 重复推理不一致时，先查计算重复性。
3. 对比 `only_*` 与 `without_*`，找初始损失主要关联的组；两者交叉看。它们是条件消融，存在交互，损失不能直接相加；关闭某组后恢复精度不等于该组有实现 bug。
4. `model_state_unchanged` 必须为 true。此步骤不会产生 `best_qat_foldaware.pth` 或测试结果 `result.json`，不应送入 FPGA 模拟器。

也可用 `--diagnose_initial_quantization` 在正式训练前诊断并继续。该模式会在转换容差/预测检查失败时停止，保留报告，避免混入后续训练解释。推荐先按本节独立跑完诊断，再进行下一阶段。

## 2. 捕获 fixed 训练的跳变前后模型

```bash
python run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --variants bn1_fixed_scales \
  --max-epoch 100 --eval-interval 1 \
  --capture-oa-drop-pp 5 --capture-start-epoch 25 --capture-max-events 3 \
  --output-dir ./qat_jump_capture_seed0_6
```

捕获条件：当前验证 OA 比上一轮下降至少 5 个**百分点**，当前 epoch≥25；每个 seed 最多保留最先触发的 3 对事件。必须逐轮验证，代码会拒绝捕获模式搭配其他 eval interval。

每次验证保留上一份模型 state_dict 和预测在内存；触发后落盘：

```text
run_seed6/jump_events/
  manifest.json
  epoch0057_to_0058/              # 实際轮次由本次运行决定
    before.pth
    after.pth
    validation_batch.pt
    event.json
```

这些是供推理复查的 fold-aware state_dict，不是含优化器状态的续训 checkpoint。完整验证指标在 event.json；输入选择遵守原 eval batch8 的顺序和分组，选“原来正确、下一轮错误”的验证像素最多的整批，保存输入、标签、tile 位置和两次预期预测。不使用测试集选样。

下面自动找 seed 6 的第一对事件；如果没有达到阈值的事件，会明确退出，不伪造事件：

```bash
CAPTURE_RUN="qat_jump_capture_seed0_6/bn1_fixed_scales/training/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_bn1_fixed_scales/$CONFIG_D1/run_seed6"
EVENT_DIR="$(find "$CAPTURE_RUN/jump_events" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n 1)"
test -n "$EVENT_DIR" || { echo '没有捕获到跳变，先检查验证历史和阈值'; exit 1; }

python replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device cuda:0 --mode native \
  --output-dir ./qat_jump_replay_seed6_native
```

先看 `jump_replay.json`：

- `captured_prediction_match`：重放与捕获时该 batch 已保存验证预测的吻合率，前后两模型应各为1；否则先核查环境/代码，不能直接归因于训练更新。
- `repeat_logits.changed`、`repeat_code_changes`：每个 checkpoint 各自重复推理两次，理想为0。
- `prediction_exchange`：该 batch 正确→错误、错误→正确及预测变化的像素数。
- `first_code_change`、`first_activation_code_change`：执行顺序中的首个整数码差异，不自动认定为根因。

`layer_comparison.csv` 记录实际折叠权重码、激活码、D 码、INT32 bias 码、层/SSM 输出及最终 logits 的前后变化。LSQ 行还记录尺度、裁剪比例及到最近有效半整数阈值的距离，距离单位为该量化器的码步长。比较的是同一批的全部元素，权重元素与激活元素不能直接按原始误差大小排名。报告同时给出变化比例、MAE、MaxAE。

`batch_logits.npz` 保存前后稠密 logits 和标签。可加 `--save-traces` 保存全部逐层张量，文件会明显变大。该 batch 的 OA 不等于全验证集 OA。

当 native 自重复或捕获预测复现有差异时，再对同一事件分别运行：

```bash
python replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device cuda:0 --mode tf32-off \
  --output-dir ./qat_jump_replay_seed6_tf32off

python replay_qat_jump.py \
  --event-dir "$EVENT_DIR" --device cuda:0 --mode reference-tf32-off \
  --output-dir ./qat_jump_replay_seed6_reference
```

native 恢复记录的 TF32/cuDNN 标志；替代模式是后端对照，不要求与 CUDA 训练预测完全一致。CUDA 不可用时本脚本直接报错，不静默改成 CPU；若需要 CPU 数值检查，显式用 `--device cpu`，并承认后端变化。

## 3. 两个独立训练对照

前两阶段排查后，运行以下三组配对对照：

```bash
python run_qat_stability_sweep.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --variants bn1_fixed_scales fixed_low_weight_lr fixed_freeze_bn_affine \
  --max-epoch 100 --eval-interval 1 \
  --capture-oa-drop-pp 5 --capture-start-epoch 25 --capture-max-events 3 \
  --output-dir ./qat_single_factor_seed0_6
```

| 组 | 普通模型参数 LR | D 参数 LR | BN γ/β | 所有 LSQ 尺度 | BN running statistics |
|---|---:|---:|---|---|---|
| `bn1_fixed_scales` | 1e-5 | 1e-5 | LR=1e-5 | 第1轮起固定 | 第1轮起固定 |
| `fixed_low_weight_lr` | **1e-6** | 1e-5 | LR=1e-5 | 第1轮起固定 | 第1轮起固定 |
| `fixed_freeze_bn_affine` | 1e-5 | 1e-5 | **冻结** | 第1轮起固定 | 第1轮起固定 |

普通模型参数包括卷积/投影权重、bias、A_log 等，排除独立 D、BN affine 和 LSQ 尺度。没有同时引入 cosine。日志中的历史组名 `weights_and_bn` 为兼容保留，但它现在不包含 BN affine；后者使用 `bn_affine` 独立组。默认各组 LR/weight decay 语义保持不变。

第三阶段包含重跑 fixed 控制组，用于同版本配对。若已经用**同一软件版本及完全相同设置**完成第二阶段，可只指定两个新 variants，将它们与第二阶段控制组一起分析。

比较第25–100轮的验证 OA 范围、相邻下降幅度、逐类指标和捕获事件，而不只看最佳 checkpoint 的最终 OA。所有组按验证 mAcc 选模型，并统一允许 epoch0 候选；测试指标只在正常训练结束后描述，不用于选 epoch 或挑诊断样本。

`validation_summary.csv` 现增加后期 OA 标准差、相邻记录最大下降/变化和事件数；此流程 `eval_interval=1`，所以相邻记录就是相邻 epoch。少于25轮时后期范围/标准差留空，不能据此评价后期稳定性。

## 4. 新参数与结果回传

训练脚本参数用下划线；sweep 参数用连字符，勿混用。

| 训练参数 | 默认值 |
|---|---|
| `--diagnose_initial_quantization` | 关闭 |
| `--initial_diagnostics_only` | 关闭 |
| `--capture_oa_drop_pp` | 0，关闭捕获 |
| `--capture_start_epoch` | 25 |
| `--capture_max_events` | 3 / seed |
| `--weight_lr_multiplier` | 1，仅普通模型参数 |
| `--freeze_bn_affine` | 关闭 |

第一阶段先回传两个 seed 的 `initial_quantization_diagnosis` 文件夹。第二阶段回传 `jump_events` 和 `qat_jump_replay*`，以及原来的 history/diagnostics JSONL。第三阶段回传各组 `result.json`、验证历史和诊断。旧运行只有最佳 checkpoint，无法补出当时丢失的第57/58轮模型，必须重新捕获。

本地验证覆盖小型合成数据流程、诊断前后模型与随机状态、分组开关、独立学习率、BN affine 冻结、事件数量上限和离线重放；这些不替代服务器 UP/CUDA 结果，也不预先保证某个对照提高精度。
