# 误差消融与共享极点分析

2026-09-14：新增 D 通路、共享 U 量化和 d-only 折叠误差消融，详见同目录 `README_D通路_QAT与FPGA模拟.txt`。新 QAT 默认 shared U；旧检查点无合同字段时继续按 legacy U 恢复。D1 尚未接入 RTL。

更新日期为 2026-09-08。主模拟入口是 `both_FPGA_single_qat_source.py`。`both_FPGA_21patch_dual.py` 现在转发到同一实现，避免两份代码漂移。

2026-09-10 修复：数据加载同时接受数据根目录和单数据集目录，例如文件位于 `UP/PaviaU.mat` 时，`--data-path .` 和 `--data-path UP` 均可。旧版只接受前者。批量模拟失败时现在直接显示日志最后 80 行，并在清单记录失败配置与退出码；修正后请使用新的输出目录。

## 文件与适用范围

| 文件 | 作用 |
|---|---|
| `train_mambahsi_spatial_split_dense_qat.py` | QAT 的 dt 输入/输出位宽可配置，保存数值配置并校验加载 |
| `both_FPGA_single_qat_source.py` | 全场景整数仿真、递归误差消融、分区域局部诊断及输入采集 |
| `ssm_error_ablation.py` | 公共数值配置、系数编译、范围证书、整数递归与误差统计 |
| `run_ssm_error_ablation.py` | 同一批 U/dt/B/C 输入上的局部重放 |
| `run_fpga_error_sweep.py` | 多 QAT run、多消融配置的完整场景仿真和指标汇总 |
| `analyze_alog_dynamics.py` | 多数据集、多种子、FP32/QAT 的连续极点与离散保持分析 |

模拟器继续面向当前三 block、双分支、共享极点、无 z/D 的 S0 拓扑。QAT 脚本中其他结构变体可以继续训练，但本次没有为这些结构新增 RTL。极点分析支持 FP32 的共享/逐通道极点，也支持 QAT 检查点。

默认数值配置仍为 dt 输入 INT9、dt 输出 INT8、A UQ1.24、K U19（小数位最大 24）、状态 signed32/Q24、单次 HA0 写回。默认配置已经与修改前实现进行整网单 tile 整数 logits 和六核 ROM 一致性回归。

**代码提供数值仿真与实验入口，并未生成不同位宽的 RTL 或测量 FPGA 资源、频率、功耗。** 非默认格式输出标记 `rtl_baseline_compatible=false`，不能直接送入固定 256×419 ROM 和 Q24 RTL。`a-only` 等反事实模式含浮点运算，只用于误差归因，不是硬件整数实现。

## 环境

建议在原训练环境运行，依赖 Python 3.9+、PyTorch、NumPy、Matplotlib、SciPy；MATLAB v7.3 文件需要 h5py。mamba_ssm 可用时使用优化扫描，未安装时可用 CPU/PyTorch 参考扫描进行小规模验证。

模拟器的可选分类图导出还依赖 spectral。

将上述脚本、`mambahsi_ablation_model.py`、`utils/` 一同放在工作目录。新的 QAT 脚本需要同目录的 `ssm_error_ablation.py`，不能只复制 QAT 单个文件。

以下命令中的路径是需要替换的示例。`QAT_RUN` 为包含 `result.json`、`best_qat_foldaware.pth`、样本索引和保存预测的 run 目录。`FP32_DIR` 为包含训练域 PCA、空间划分以及 `run_seedN/` 的目录。

```bash
QAT_RUN=/path/to/qat/run_seed0
FP32_DIR=/path/to/fp32/configuration
DATA_ROOT=/path/to/data
```

## 1. QAT 的 dt 边界位宽消融

### 1.1 dt_proj 输入位宽消融（验证特殊 9-bit 输入的必要性）

原始代码在 dt_proj 内部对低秩输入使用 9-bit LSQ；dt_proj 输出另有独立的 8-bit 量化。两者不是同一量化节点。验证输入 9 bit 的必要性时，应扫描 `--dt-input-bits`，固定 `--dt-output-bits 8`。

```bash
python train_mambahsi_spatial_split_dense_qat.py \
  --dataset UP --data_set_path "$DATA_ROOT" \
  --fp32_dir "$FP32_DIR" \
  --work_dir ./results --run_tag dtin8_dtout8 \
  --seeds 0 --device cuda:0 \
  --dt-input-bits 8 --dt-output-bits 8
```

输入扫描使用 8/9/10 bit。已完成的 dtin9_dtout8 run 在 FP32 初始化、划分、种子、训练参数和脚本版本一致时可直接作为 9-bit 输入基线。

### 1.2 dt_proj 输出/非线性地址码宽消融（独立实验）

固定 `--dt-input-bits 9`，扫描 `--dt-output-bits 6/7/8/9/10`。此前提供的 `dtin9_dtout${bits}` 循环属于本实验，不能用来证明输入 9 bit 的必要性；已有结果应按输出位宽消融保留，不应改名当作输入位宽结果。

- `--dt-input-bits` 可选 8/9/10，改变低秩 dt_proj 的输入量化。
- `--dt-output-bits` 可选 6/7/8/9/10，改变生成 ROM 地址的时间步输出量化。
- 每个配置从相同 FP32 检查点初始化，复用保存的训练/验证/测试区域、PCA 和标签，保持相同训练预算；每组用独立 `--run_tag`。
- 每个 run 保存 `numeric_config.json`，最终 `result.json` 也保存该配置。历史 run 无此字段时按旧 9/8 边界读取。
- 模拟器自动读取训练位宽。不能通过 CLI 将 8-bit 时间步检查点直接冒充为 7-bit QAT；不匹配会报错。
- QAT 内部继续使用浮点 selective scan，新增的 dt 边界参与 LSQ 学习。TA/TK、状态格式与舍入实验在固定 QAT 检查点上由模拟器完成。本次没有把整数递归嵌入训练，也不将其标为“递归感知 QAT”。
- 已有 QAT 检查点的 run 会被拒绝覆盖，需使用新 run_tag。

数据集可选 UP、HongHu、HanChuan、Houston。`--dataset all` 沿用原脚本的多数据集路径解析规则；不要给 all 指定只属于一个数据集的 `--fp32_dir`。

## 2. 全场景整数仿真与同输入诊断

```bash
python both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN" --dataset UP --seed 0 \
  --data-path "$DATA_ROOT" --fp32-dir "$FP32_DIR" \
  --device cuda --output-dir ./sim_baseline \
  --report-ssm-errors --ssm-analysis-split test \
  --capture-ssm-inputs --ssm-capture-tiles 8
```

整网仍计算完整场景和原测试集 OA/mAcc/Kappa/mIoU。`--ssm-analysis-split` 只选择局部误差统计和输入采集的范围，可选 train/validation/test/all，默认 test。工作点选择应使用 validation；最终测试结果不用于挑配置。

新增输出：

- `numeric_config.json` 与总结果中的执行模式、RTL 兼容标记。
- `ssm_local_error_by_position.csv`，每个核、状态/读出、序列位置的 MAE、MaxAE、相对 L2、计数和状态饱和次数。误差是对同一实际 U/dt/B/C 输入的全精度递推；零参考范数的相对 L2 为 null/空值。
- `replay_inputs_manifest.json`，记录数据集、seed、检查点、所选 tile 位置与范围。
- `replay_tiles/tile_*/ssm_replay_inputs/*.npz`，保存每核完整序列的输入码、极点和尺度。默认最多采集 1 个选定区域 tile，可由参数增加。
- 每核 ROM 清单增加范围证书，覆盖所有地址与完整输入/状态码域。若需要超过配置允许的累加宽度，拒绝导出；合法 65-bit 情形使用 Python 大整数精确承载，不会静默 INT64 回绕。

## 3. 固定检查点的递归格式与舍入

在上一条模拟命令上增加相应参数，例如：

```bash
# 相同整数范围、较小状态字宽
--ssm-state-bits 28 --ssm-state-fraction-bits 20

# 改变系数分辨率
--ssm-a-fraction-bits 20

# 减少注入系数精度上限，实际小数位仍逐核选择
--ssm-k-preferred-fraction-bits 20

# 两个乘积分别 HA0 舍入，再相加并饱和
--ssm-rounding separate
```

每种配置必须使用新的 output-dir。

完整参数还包括 `--ssm-k-bits`、`--ssm-accumulator-bits`。实际 K 小数位同时受位数上限和共同累加域约束，并写入清单。

## 4. 误差来源的真正同输入重放

```bash
python run_ssm_error_ablation.py \
  --inputs-glob './sim_baseline/replay_tiles/*/ssm_replay_inputs/*.npz' \
  --suite sources --output-dir ./replay_sources

python run_ssm_error_ablation.py \
  --inputs-glob './sim_baseline/replay_tiles/*/ssm_replay_inputs/*.npz' \
  --suite widths --output-dir ./replay_widths
```

`sources` 包括以下模式。所有模式都使用完全相同的输入码及其物理尺度。

| 模式 | 递归内部改变 |
|---|---|
| none | 全精度 A/K 与浮点状态，局部参考 |
| a-only | 仅使用定点 A，K 和状态保持浮点 |
| k-only | 仅使用定点 K，A 和状态保持浮点 |
| state-only | A/K 全精度，仅在状态写回舍入与饱和 |
| all | 定点 A、K、状态和完整整数更新 |
| separate | 同 all，但保持/注入乘积分开舍入 |

输出每个模式的逐位置误差 CSV、完整配置和编译范围证书。不同来源的 MAE 不能简单相加解释总 MAE，交互与误差抵消由 all 对照体现。

`widths` 包括 A 小数位 16/20/24，状态 32/16、32/20、32/24、24/24、28/24、28/20、24/16，以及 K 小数位上限 16/20/24 的代表工作点。没有在重放中改变 dt 输入/输出位宽，因为这些位宽需要匹配的 QAT 和新输入码。

重放结果属于局部算子误差，不产生整网 OA。全网传播需运行下一节。不同数据集、检查点或采集来源应分别重放，以免汇总口径混淆。

## 5. 整网批量误差消融

```bash
python run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_RUN" --data-path "$DATA_ROOT" \
  --fp32-dir "$FP32_DIR" --device cuda \
  --suite both --output-dir ./sweep_check --dry-run
```

`--dry-run` 只输出 `commands.sh` 与 `sweep_manifest.json`。确认路径和配置后，使用新的目录执行：

```bash
python run_fpga_error_sweep.py \
  --qat-run-dir "$QAT_RUN" --data-path "$DATA_ROOT" \
  --fp32-dir "$FP32_DIR" --device cuda \
  --suite both --output-dir ./sweep_run
```

`sources` 为 D0 六组或 D1 七组误差来源/舍入模式，`widths` 为位宽实验，`both` 为两者，`nonlinear` 为下一节的非线性近似。新增 `requant` 输出再量化三组对照及 `k-precision` K 精度网格，`all` 现在包含上述五类实验。详见 [batch 诊断与新消融说明](README_batch差异_再量化_K扫描.txt)。每个配置独立目录、日志和结果，输出 `sweep_metrics.csv` 汇总 OA、mAcc、Kappa、mIoU、相对 QAT 的 OA 损失和逻辑 ROM 位数。执行失败立即停止，清单只标记已成功完成的作业。

重复 `--qat-run-dir` 可处理多个数据集/种子。数据集和 seed 从每个 result.json 读取；多个 run 使用独立路径哈希标识。`--fp32-dir` 重定位覆盖只用于单 run，多 run 应各自具有可解析的源制品路径。

## 6. 非线性同误差候选点

```bash
python run_ssm_error_ablation.py \
  --inputs-glob './sim_baseline/replay_tiles/*/ssm_replay_inputs/*.npz' \
  --suite nonlinear --output-dir ./replay_nonlinear
```

比较精确函数编译与 8/16/32/64 段均匀 PWL 候选。模拟器也可指定：

```bash
--ssm-coefficient-backend pwl --ssm-pwl-segments 32
```

PWL 分别近似 softplus 和 exp，区间由当前冻结码域与极点确定；本模拟器枚举其输出进行数值评估。清单报告全地址 A/K MAE 和 MaxAE，局部/全网运行报告传播影响。

**这是用于选择同误差对照点的软件数值基线，不是 Mamba-X 的复现，也不是已实现的在线 PWL 电路。** 清单中的逻辑 ROM 位数描述模拟器枚举表示，不代表 PWL 硬件的资源。误差匹配后仍需分别实现电路、综合与布局布线，才能报告同误差硬件资源/时序比较。

## 7. 共享极点分析其他数据集与 QAT

单个 FP32 run：

```bash
python analyze_alog_dynamics.py \
  --run-dir '/path/to/HongHu/config/run_seed0' \
  --data-path "$DATA_ROOT" --split test --device cuda \
  --output-dir ./dynamics_HongHu_seed0
```

多个数据集或种子：

```bash
python analyze_alog_dynamics.py \
  --run-dir '/path/to/UP/config/run_seed0' \
  --run-dir '/path/to/HongHu/config/run_seed0' \
  --run-dir '/path/to/HanChuan/config/run_seed0' \
  --run-dir '/path/to/Houston/config/run_seed0' \
  --data-path "$DATA_ROOT" --device cuda --no-plots \
  --output-dir ./dynamics_four_datasets
```

也可用带引号的 glob：

```bash
python analyze_alog_dynamics.py \
  --run-glob '/path/to/results/**/run_seed[0-9]' \
  --dataset HongHu --data-path "$DATA_ROOT" \
  --device cuda --no-plots --output-dir ./dynamics_HongHu_all_seeds
```

glob 的候选必须是具有保存配置和制品的 run 目录，不要把无关目录传入。`--dataset` 会筛选 glob 结果；显式 run 的数据集不匹配会报错。

QAT run：

```bash
python analyze_alog_dynamics.py \
  --run-dir "$QAT_RUN" --artifact-dir "$FP32_DIR" \
  --data-path "$DATA_ROOT" --split test --device cuda \
  --max-tiles 32 --no-plots --output-dir ./dynamics_qat_pilot
```

脚本自动识别 `best_qat_foldaware.pth`，按 result.json 的模型/位宽配置恢复，并在 BN 融合后分析。QAT 动态统计挂在 **dt_output_quant 之后**，不会把量化前的 dt_proj 输出误当成实际时间步。`--checkpoint` 可覆盖单 run 权重路径；请使用 foldaware 检查点。

新增结果：

- `discrete_retention_stats.csv`，实际访问的 Abar 统计、不同 FA 的舍入为 1 比例及系数误差。
- 真实时间窗口的 `exp(-lambda * sum(delta))`，默认长度 1/4/16/64/256。Spe 的 4-token 序列只计算合法窗口，不跨像素或 tile 拼接。
- 固定同一 delta，将逐通道 lambda 投影到行均值共享基后，对 Abar 与窗口保持因子的误差。此项是数值干预，不表示已重训的共享模型与逐通道模型状态坐标对齐。
- 多 run 输出 `multi_run_summary.json`、`multi_run_pole_summary.csv`，动态分析时另有 `multi_run_retention_summary.csv`。
- 记录数据集、seed、实际检查点路径、SHA256、FP32/QAT 类型、分析区域和 tile 数。

`--static-only --device cpu --no-plots` 可只读取极点，无需原始图像文件。当前本地四数据集 seed 0 的真实静态验证结果在 `output/validation/20260908_error_ablation/four_dataset_static/`。

## 测试与结果边界

```bash
python -m unittest discover -s tests -v
```

测试包括默认整网整数 logits/ROM 对旧代码一致、Python 大整数独立参考、半整数符号舍入、饱和、65-bit 安全回退/危险导出拒绝、状态隔离、非默认 dt 位宽的梯度与恢复、变体导出、PWL 数值、保持窗口边界，以及合成数据上的 QAT→全场景模拟→采集→重放→QAT 动态分析→批量来源消融。

合成数据测试只验证软件功能。没有使用这些数值声称论文精度，也没有开展正式四数据集 QAT 重训或 Vivado/板级测量。修改前四个主脚本备份在 `output/code_backups/20260908_error_ablation/`。
