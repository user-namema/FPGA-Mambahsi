# 服务器补件与只读收集

本地已经有四数据集 D1 FP32 checkpoint、PCA/归一化预处理、划分与 seed 索引，以及本次 eval1 QAT 的源码和训练结果。**不需要重复传这些文件。** 本次收集用于确认服务器实际依赖及补齐下列运行记录；仓库名为 `FPGA-MambaHSI`。

## 已接收并核验，主要补件已完成

- 四数据集 eval1 FPGA 模拟40次，以及对应 checkpoint/QAT 元数据核对记录。
- UP FP32 GPU0功耗42次、原始采样和汇总，保留Xorg显示后台。
- 四数据集 FP32 GPU 原始报告28份，与已有112行汇总一致。
- UP seed0 dt 输入诊断：六核×三策略18行，非四数据集整网INT8输入消融。
- D1 N0–N3 完整软件与原始结果：858 tiles、前5 tiles详细轨迹；10个源码哈希匹配，24份MEM码字匹配。
- 原环境版本与Mamba Python包；已安装CUDA运行库备份通过22文件校验，两个扩展与原采样哈希一致。

运行和归档位置见 [最终补件说明](COMPLETION_20260923.md)。无须重复收集。

`cuda_runtime_5o7mz0dk` 是有效运行库备份，保留在作者本地，不进普通Git。
`cuda_runtime_mscish0i` 的压缩包CRC失败，未纳入发布材料。原始wheel未收到，
但已有对应已安装运行库备份，因此不作为本次材料闭环的阻塞项。

## 只在论文采用时再补

历史dt输出6–10位×十seed、完整B1/B8首差异定位、requant/K/state位宽扫描：
只有计划公开或论文采用时才要求对应原始结果。不能把已有脚本计为已完成实验。
RTL/Vivado仍在本仓库范围之外。

以下保留收集器说明，供将来新增实验使用。

当前本地 QAT 源码哈希与四数据集 eval1 的 `launch.json` 一致。收集器仍会带上少量服务器源码，用于逐文件核对新旧版本，不会自动覆盖本地整理版本。

## 1. 最小收集：环境与关键源码

把仓库里的 `tools/collect_server_repro.py` 复制到服务器任意位置。例如复制为原工程目录中的 `collect_server_repro.py`，然后执行：

```bash
conda activate mambahsi
cd ~/mzz/MambaHSI
python collect_server_repro.py \
  --project-root "$PWD" \
  --output-dir "$PWD/server_repro_export"
```

收集过程只读取工程、环境及 GPU 信息；不会训练模型，不会安装依赖，不会改变 GPU 参数。它仅在 `--output-dir` 新建带时间和随机后缀的目录、ZIP 及 SHA256 文件。必须用原 `mambahsi` 环境的 `python` 执行，不能改用 base 环境。

## 2. 加入原始测速和已完成的新实验

下面路径是此前使用的目录名；先确认实际存在，再添加相应 `--artifact-root`。**不必为了收集启动新的实验。** 相对路径均相对于 `--project-root`。

```bash
python collect_server_repro.py \
  --project-root "$PWD" \
  --output-dir "$PWD/server_repro_export" \
  --artifact-root gpu_current_4datasets
```

如果 GPU 功耗、新 eval1 FPGA 或 dt 诊断已运行，可在同一命令继续添加它们的实际结果目录，例如：

```bash
# 只保留确实存在的目录；下面两个名字是需要替换的占位符。
python collect_server_repro.py \
  --project-root "$PWD" \
  --output-dir "$PWD/server_repro_export" \
  --artifact-root gpu_current_4datasets \
  --artifact-root /absolute/path/to/completed_gpu_power_results \
  --artifact-root /absolute/path/to/completed_eval1_fpga_results
```

额外软件脚本可显式加入，路径必须在工程目录内、为 `.py` 或 `.sh` 文件：

```bash
python collect_server_repro.py \
  --project-root "$PWD" \
  --output-dir "$PWD/server_repro_export" \
  --source-file QAT_eval1_GPU_power_20260921/benchmark_up_gpu_power.py \
  --source-file QAT_eval1_GPU_power_20260921/gpu_power_monitor.py \
  --source-file QAT_eval1_GPU_power_20260921/run_up_gpu_power.sh
```

## 收集范围、大小与交付

- 默认复制明确列出的 `.py`/`.sh`/`utils` 源码，不递归打包整个工程。
- 仅递归读取显式指定的结果目录，收集其中 `.json`、`.csv`、`.log`。不收集原始 `.mat` 数据集、预测图、RTL、浏览器信息、shell 历史或环境变量全集。
- 默认不包含模型权重、预处理/索引 NPZ 或 CUDA 扩展二进制。扩展只记录位置与 SHA256，便于核对本地修改。
- 默认单文件上限 16 MiB，总复制数据上限 256 MiB。过大或缺失文件在 `manifest.json` 的 `skipped` 中说明。需要完整大日志时可增加 `--max-file-mib 64 --max-total-mib 1024`。
- 日志/JSON/CSV 中 URL 的认证部分和 query 会被移除，常见凭据字段也会遮蔽；清单保留原始 SHA256 与打包后 SHA256，并记录是否改变。该清理不是对任意自由文本的秘密识别保证；请在对外发布前查看收集内容。
- 只有明确需要补传模型时才使用 `--include-checkpoints`；只有需要补传划分/预处理时才使用 `--include-repro-artifacts`。这次已知文件均在本地，通常不需要这两个开关。
- `environment.json` 中命令失败会保存原因，其他项目继续收集；例如没有 `nvcc` 不会阻止导出。新 Python 进程的 TF32/cuDNN 开关表示环境默认值，具体测速设置仍以原始运行 JSON 和源码为准。

运行结束后，把打印的 `FPGA-MambaHSI-server-*.zip` 和对应 `.zip.sha256` 复制到本地“毕设”文件夹即可。保留服务器原始文件，无须上传到公开 GitHub。后续将先核对版本与报告，再把适合公开的环境记录及软件参考合入仓库。

若 N0–N3 软件对照位于另一台电脑，可以单独打包对应 Python、配置及结果 CSV/JSON，并附一条当时的运行命令。无需提供 `.v`、`.sv`、`.xpr`、`.bit` 或综合/实现工程。
