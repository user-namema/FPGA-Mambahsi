# 服务器补件与只读收集

本地已经有四数据集 D1 FP32 checkpoint、PCA/归一化预处理、划分与 seed 索引，以及本次 eval1 QAT 的源码和训练结果。**不需要重复传这些文件。** 本次收集用于确认服务器实际依赖及补齐下列运行记录；仓库名为 `FPGA-MambaHSI`。

## 本次已经收到

- 四数据集 eval1 FPGA 模拟 40 次，结果、源码身份、checkpoint 与 QAT 元数据已核对。
- UP FP32 GPU0 功耗 42 次，包含两种范围、七种 batch、三次重复和原始采样；测量保留 Xorg 显示后台。
- 原环境完整版本清单、wheel 来源、CUDA 扩展哈希，以及实际安装的 Mamba Python 包。Python 源码与环境采样哈希一致，`mamba_simple.py` 存在本地修改。

公开文件和精确回放见 [本次更新说明](SERVER_UPDATE_20260923.md)。不需要重复传上述内容。

## 仍需补充什么

| 优先级 | 文件/记录 | 用途 |
|---|---|---|
| 1 | 四数据集 FP32 GPU 测速 28 组原始 JSON、汇总 CSV | 已有数值汇总；这次收集的旧目录只有早期失败日志。用户确认新结果在 `QAT_eval1_GPU_power_20260921` 下，需选对实际结果子目录 |
| 2 | Mamba 1.1.2 与 causal-conv1d 1.1.2 的原始 cp38/cu118/torch1.13 wheel | Python 包已收到，wheel 和编译扩展未收到；用于精确二进制环境复现，不必放普通 Git 仓库 |
| 2 | 新 eval1 dt 输入诊断输出（如果已经运行） | 本次模拟器版本没有 dt 诊断，不能用已有 INT9 输出范围代替 |
| 3 | N0–N3 Python 软件参考、系数生成器、驱动、JSON/CSV | 论文数值的完整软件复现；不需要 RTL/Vivado |
| 可选 | 历史 dt output=6–10 × 十 seed、完整 B1/B8 定位、requant/K 位宽扫描结果 | 仅在确已完成且计划公开时补齐 |

从服务器项目根目录收集新子目录的记录，可用：

```bash
conda activate mambahsi
cd ~/mzz/MambaHSI
python collect_server_repro.py \
  --project-root "$PWD" --output-dir "$PWD/server_repro_export" \
  --artifact-root QAT_eval1_GPU_power_20260921 \
  --source-file QAT_eval1_GPU_power_20260921/benchmark_up_gpu_power.py \
  --source-file QAT_eval1_GPU_power_20260921/gpu_power_monitor.py \
  --source-file QAT_eval1_GPU_power_20260921/run_up_gpu_power.sh
```

若新目录包含许多训练日志，可将 `--artifact-root` 缩小为实际测速子目录；检查收集清单中的 skipped。wheel 需单独复制，收集器默认不收二进制文件。本地归档仅纳入必要源码、轻量记录和元数据。

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
