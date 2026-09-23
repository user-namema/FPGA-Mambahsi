# Batch 差异定位、输出再量化消融与 K 精度扫描

本次新增工具读取现有 D0/D1 QAT checkpoint，无需重新训练。默认 FPGA 模拟的数值行为保持原样；RTL 未修改。请整体更新软件包，至少同时更新 `both_FPGA_single_qat_source.py`、`ssm_error_ablation.py`、`run_ssm_error_ablation.py`、`run_fpga_error_sweep.py`，并添加 `diagnose_qat_batch.py`。训练脚本继续使用已支持 D/shared-U 的版本。

在服务器项目目录设置实际路径：

```bash
CONFIG_D1=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
FP32_D1="results/SPATIAL_SPLIT_3WAY_DENSE/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"
QAT_D1="results/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_D1_sharedU_D8_dtin9_dtout8/$CONFIG_D1/run_seed0"    
```

所有输出目录必须为空或不存在。脚本不会覆盖已完成结果。以下命令均使用同一 seed0 checkpoint 和原始数据根 `./data`。

## 1. 定位 batch 首个分歧

已有 `sim_D1_sharedU_seed0` 中的 batch=1/保存 batch 预测，可自动选择第一个出现有标签测试预测差异的 tile：

```bash
python diagnose_qat_batch.py \
  --qat-run-dir "$QAT_D1" --fp32-dir "$FP32_D1" \
  --data-path ./data --device cuda \
  --prediction-dir ./sim_D1_sharedU_seed0 \
  --save-traces --output-dir ./diagnose_D1_batch_seed0
```

`--prediction-dir` 也可以指向 sources 结果中的 `all` 子目录；需要该目录的 `fpga_simulation_result.json`、`qat_direct_prediction.npy` 和 `qat_saved_batch_reproduced_prediction.npy`。程序校验 checkpoint SHA256、保存 batch 大小、场景形状和保存测试预测，避免使用其他训练轮次/模型的预测定位。

如果没有这些预测文件，或想换一个位置检查，手动指定原始测试 batch 中的 tile：

```bash
python diagnose_qat_batch.py \
  --qat-run-dir "$QAT_D1" --fp32-dir "$FP32_D1" \
  --data-path ./data --device cuda \
  --group-index 0 --target-index 0 \
  --save-traces --output-dir ./diagnose_D1_batch_group0
```

group 从 0 开始，按原保存测试 tile 顺序和保存 eval_batch_size 分组；target 是该组内位置，从 0 开始。手动索引不能与 `--prediction-dir` 同时使用。工具比较同一个目标 tile 单独运行与放回其原始 batch 的输出，不会把不同 tile 相互比较。最后不足两个 tile 的 batch 不能用于该诊断。

默认比较三种环境：

| 模式 | TF32 | scan |
|---|---|---|
| native | 保持当前环境 | 当前可用后端 |
| tf32-off | matmul/cuDNN 均关闭 | 当前可用后端 |
| reference-tf32-off | 关闭 | PyTorch reference |

每种环境重复单 tile 和 batch forward，用于区分稳定 batch 差异与同形状重复执行差异。还会比较 native 与其他环境的同形状输出。reference 会较慢；初查可用 `--modes native tf32-off`，之后再跑默认全部环境。native 总会运行以提供基线。退出或异常时恢复全局后端和 hook。

输出：

- `batch_diagnosis.json`：环境信息、目标 tile、各对比的第一个精确差异、第一个超容差差异、第一个量化码变化、目标 tile 的预测变化。默认浮点容差 `atol=1e-6, rtol=1e-5`，量化码始终按精确相等判断。
- `batch_trace.csv`：按实际执行顺序列出量化输入、量化码、反量化输出、Conv/Linear/激活、各核 scan 的 U/dt/B/C/输出及最终 logits 的误差。
- `native_changed_traces.npz` 和 `trace_keys.json`：仅在 `--save-traces` 下保存 native 中有差异阶段的目标 tile 向量及名称对应表。

先看 `single_repeat`/`batch_repeat` 是否出现差异，再看 `single_vs_batch.first_code_change`。若第一个量化码差异之前的投影已出现浮点差异，应优先追查该投影数值路径；若 U/dt/B/C 都相同而 scan 输出首先变化，应优先比较 scan 后端。关闭 TF32 后差异消失可作为定位证据，但不能单凭最终 OA 给算子定责。

工具不校准 LSQ，不更新 BN，不训练；结束时检查 state_dict 和 LSQ 初始化状态没有变化。这里只验证选中 tile，不能用该输出代替完整测试集 OA。CPU 模式可验证工具流程，不能定位服务器 CUDA 的实际数值差异。

## 2. 补 SSM 输出再量化消融

```bash
python run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_D1" --fp32-dir "$FP32_D1" \
  --data-path ./data --device cuda \
  --suite requant --output-dir ./sweep_D1_requant_seed0
```

运行三个完整场景实验：

| 变体 | SSM 递推、C+D | SSM→INT8 再量化 |
|---|---|---|
| none_ideal | 同量化输入的浮点反事实 | 理想除尺度并 HA0 |
| all_ideal | 当前完整整数运算 | 理想除尺度并 HA0 |
| all_hardware | 与 all_ideal 相同的整数算法 | 现有整数乘法/移位/HA0 |

理想模式将完整整数 C+D 结果按物理尺度转换后直接量化，不增加一次中间状态网格舍入。它是诊断模式，不是已实现 RTL。

`all_ideal` 与 `all_hardware` 的数值配置只有再量化方式不同。`requant_comparison.json` 自动汇总二者测试预测变化数量、OA/mAcc 差；它包括该边界差异向后续层传播的影响。三组完整指标在 `sweep_metrics.csv`，逐核实际 K 小数位在 `coefficient_summary.csv`。

解释时：先用 none_ideal→all_ideal 衡量统一理想输出边界下的 SSM 定点影响，再用 all_ideal→all_hardware 衡量硬件输出再量化的增量。OA 差可能正负抵消，不能当作各误差的独立可加“占比”。原 sources 仍保留历史七组，以便旧结果复现。

也可在单次模拟中设置：

```bash
python both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_D1" --fp32-dir "$FP32_D1" \
  --dataset UP --seed 0 --data-path ./data --device cuda \
  --ssm-error-source all --ssm-readout-requantization ideal \
  --report-ssm-errors --output-dir ./sim_D1_all_ideal_seed0
```

新增 `--ssm-readout-requantization auto|hardware|ideal`。默认 auto 完全保留原行为：all 使用 hardware，其他来源使用 ideal；hardware 只允许整数 SSM。结果 JSON 显式记录实际模式，all_ideal 标记为 `integer-ssm-ideal-readout`。

## 3. 扫描 K 位宽与小数位上限

先在捕获输入上做局部筛查。必须使用原始 `.npz` 输入目录，`replay_D1_sources` 只有汇总 CSV，不能反推出原输入：

```bash
python run_ssm_error_ablation.py \
  --inputs-glob './sim_D1_sharedU_seed0/replay_tiles/*/ssm_replay_inputs/*.npz' \
  --suite k-precision \
  --k-bits-grid 19 21 23 --k-fraction-grid 24 26 28 \
  --output-dir ./replay_D1_K_precision_seed0
```

然后运行完整网络九组配对扫描：

```bash
python run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_D1" --fp32-dir "$FP32_D1" \
  --data-path ./data --device cuda \
  --suite k-precision \
  --k-bits-grid 19 21 23 --k-fraction-grid 24 26 28 \
  --output-dir ./sweep_D1_K_precision_seed0
```

九组名称为 `K_b19_fmax24` 到 `K_b23_fmax28` 的笛卡尔积。保持已训练的 dt 输入/输出位宽、A/状态格式、D、single 舍入和硬件再量化不变。旧结果的 19/24 可作为复现对照。也可缩小网格，例如 `--k-bits-grid 19 21 --k-fraction-grid 26` 只运行两组。

`k-fraction-grid` 指小数位上限，不保证实际采用该值；编译器根据各核 K 的范围自动选择可容纳的小数位。两种请求配置可能得到相同的实际 K 表，这是有效结果，不应误称精度提高。

输出 `coefficient_summary.csv` 逐核记录：

- 请求 K 总位宽、小数位上限；
- **实际 K_fraction_bits**；
- K/A 系数 MAE、最大绝对误差；
- A/K 逻辑 ROM bit 数。

局部 replay 没有 OA，读取各组 `*_error_by_position.csv`；完整 sweep 在 `sweep_metrics.csv` 给出 OA、mAcc、QAT→模拟损失及 ROM。K 扫描增大位宽会改变 K×B 乘法位宽，例如 K=21 时 KB 需要 signed29 位；软件可分析，不代表原 signed27 位硬件通路可直接替换。RTL、DSP 数、时序和吞吐仍需后续验证。

## 4. 运行顺序与测试范围

建议先完成一组 batch 诊断，再运行三组 requant，最后运行 K replay 和九组完整 K sweep。三类工具都能读取已有 checkpoint；如果诊断发现必须修正模型的 batch 数学语义，需先确认修复及 checkpoint 兼容性，再作最终论文实验。

`--suite all` 现在包含 sources、widths、nonlinear、requant 和 k-precision，会启动较多实验。`--dry-run` 会创建独立计划目录；正式运行应使用另一个新目录。

本地验证覆盖有符号舍入/饱和边界、默认硬件路径兼容、K 实际小数位与误差、batch 目标映射及 hook/后端恢复，并使用合成数据运行 D1 训练→模拟→batch 诊断→再量化与 K 消融。真实服务器 CUDA 根因和正式 UP 的新实验结果，需运行上述命令后读取输出。
