# 最终补件合并与复现说明

本次合并的是已经运行并验收的材料，没有重新训练模型或测量GPU，也不包括RTL源码。主要软件补件已完成。

## 文件位置与版本

| 内容 | 仓库位置 | 范围 |
|---|---|---|
| FP32 GPU原始报告 | `evidence/completion_20260923/gpu_fp32/` | 四数据集，seed0，batch=1/2/4/8/16/32/64，28份报告；与已有112行表一致 |
| dt输入诊断 | `evidence/completion_20260923/dt_input_UP_seed0/` | 最新eval1 UP seed0，六核×三策略18行 |
| D1 N0–N3 | `evidence/completion_20260923/nonlinear_D1/` | 858 tiles全场景；前5 tiles状态轨迹；四方法预测/logit数组、系数及逐层误差 |
| D1冻结源码 | `software/snapshots/nonlinear_D1_20260916/` | 10个顶层Python文件与运行记录哈希匹配，另附utils、N2拟合参数、参考系数 |
| 旧v5与早期GPU | `evidence/completion_20260923/historical/` | 单独保留，不能当作当前D1/eval1结果 |
| 运行库身份 | `environment/observed_20260923/runtime_manifest.json`、`runtime_archive.sha256` | 可用运行库备份的22文件清单及哈希 |
| 导入溯源 | `evidence/completion_20260923/provenance.json` | 原始/公开SHA256、路径替换、排除项目与重复件说明 |

D1快照使用两个单独修订文件 `both_FPGA_nonlinear_D1.py`、`d1_methods.py`，其余来自原软件包。修订以整数码字比较MEM，避免CRLF/LF引发误判。不要将这套冻结代码覆盖到当前主软件目录。

`n2_config.json` 保留历史字节：顶层checkpoint和cores来自旧v5；D1入口仅读取其中的 `fit` 字段，系数尺度与极点由D1 checkpoint生成，并与D1参考文件核对。已确认 `fit` 与D1运行报告的 `n2_fit` 完全一致。D1有效模型身份由 `d1_methods.CHECKPOINT` 和主报告定义，不取旧配置的顶层checkpoint。

## 第20个运行入口

本对照锁定历史 UP D1 模型：

```text
d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33
```

它不是最新四数据集eval1模型。已有历史模型可直接复用，不必重训。先进入本仓库根目录：

```bash
conda activate mambahsi
export DATA_ROOT=/absolute/path/MambaHSI/data
export N0N3_QAT_RUN_DIR=/absolute/path/MambaHSI/results/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_D1_sharedU_D8_dtin9_dtout8/current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16/run_seed0
export N0N3_FP32_DIR=/absolute/path/matching/FP32/configuration

DEVICE=cuda:1 bash experiments/20_nonlinear_D1.sh --dry-run
DEVICE=cuda:1 bash experiments/20_nonlinear_D1.sh \
  --output-dir ./results/20_nonlinear_D1
```

将绝对路径换为服务器实际路径。若QAT元数据中的FP32路径仍有效，可不设 `N0N3_FP32_DIR`；显式指定更便于迁移。入口校验快照文件与checkpoint哈希，映射逻辑GPU编号，并拒绝复用已存在的输出目录。`--dry-run`不读checkpoint、不创建输出，只验证快照并打印命令。真实运行还需要对应数据、划分、预处理及CUDA环境。20是独立入口，01–19仍通过原通用命令规划器。

## 不启动GPU的核验

```bash
# 第一轮40次FPGA及42次功耗：
python tools/verify_release_records.py
# 最终补件；第一条只依赖标准库，第二条还使用NumPy：
python tools/verify_completion_records.py
python tools/verify_completion_records.py --check-arrays
```

核验包括公开文件哈希、FP32报告统计及旧汇总一致性、D1源码哈希、24份MEM码字、N2 fit、dt统计范围与比例；数组检查复算D1 logit差异、全场景标签差异及N3标签SHA。checkpoint本体未进Git，导入时已经核对的模型身份保留在原报告中。

## 已完成结果的解释

D1四方法测试OA：N0=95.1585%、N1=95.5290%、N2=95.2491%、N3=95.1585%。它们共用同一历史D1网络，改变系数生成方法；OA改善不等于数值更接近N3。N0与N3仍有17个logit码字不同，但最终全场景标签相同。N1/N2是本项目的方法适配，不是对应论文整机实现。软件执行时间不充当RTL周期或资源测量。

dt的 `int8_same_scale` 与 `int8_wider_scale` 是只读局部探针，未传播到整网输出。统计覆盖选定test tiles的内部token，包含背景/边缘填充，不是仅有标签测试像素的比例。仅UP seed0已完成，不能称为四数据集INT8输入精度消融。

FP32多batch报告来自RTX4090、cuda:1、optimized scan、TF32关闭，每个范围/时钟20个样本。功耗是另一次GPU0测量，口径见原功耗说明；不能仅因型号相同就断言为同一物理GPU。

## 原始大文件与历史资料

- 有效运行库 `cuda_runtime_5o7mz0dk/installed_cuda_runtime.tar.gz` 保留在作者本地，不放普通Git；22文件及归档校验均通过。损坏的 `cuda_runtime_mscish0i`（gzip CRC错误）排除，原文件未删除。
- 运行库是Linux/cp38/Torch1.13已安装文件备份，不是通用安装wheel。未在Mac上执行这些Linux扩展。原始wheel可另行保管，但不再是本次补件阻塞项。
- 新复制的28份fixed GPU报告与已有归档逐字节相同，未重复导入为新实验。
- 两份约26MB的早期profiler时间线留在本地；对应summary和算子表已收录，排除文件名/大小/SHA在provenance中。早期单tile模型和新D1多batch模型不能混合汇总。
- 旧v5对照使用 `1ee46f22…` checkpoint，D1使用 `d113c612…`。历史v5脚本按收到字节保存，仅供追溯，不声称旧依赖已完整重建；其结果单列不计入D1重复次数。
- 数据集、checkpoint、常规FPGA激活/ROM大批量导出和RTL工程不进此源码包。允许进Git的NPY/NPZ仅限本次明确列出的数值证据和冻结参考系数。
