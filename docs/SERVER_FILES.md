从另一台 Linux 机器收集实验记录
================================

本页用于在实验机器上收集环境信息、源码快照和轻量结果文件。收集器只读文件，
不会训练模型、安装依赖或修改 GPU。已经复制到仓库的证据不需要再次收集。

1. 准备收集器
--------------

将 tools/collect_server_repro.py 复制到实验机器工程目录，然后执行：

    conda activate mambahsi
    cd /path/to/MambaHSI
    python collect_server_repro.py \
      --project-root "$PWD" \
      --output-dir "$PWD/server_repro_export"

必须使用包含项目依赖的 Python。输出目录必须是新的目录；收集器会生成 ZIP 和 SHA256
文件。收集结束后，将 ZIP 及对应校验文件复制回本地，再运行：

    python tools/verify_release_records.py

2. 添加结果目录
----------------

只添加已经存在的结果目录。相对路径相对于 --project-root：

    python collect_server_repro.py \
      --project-root "$PWD" \
      --output-dir "$PWD/server_repro_export" \
      --artifact-root /absolute/path/to/result_directory

需要多个结果目录时重复 `--artifact-root`。建议优先收集以下类型：QAT result.json、
FPGA 模拟报告、GPU JSON/CSV、日志和运行配置。不要把整个 results 目录递归打包。

3. 添加指定脚本
----------------

只添加工程目录内的 Python 或 shell 文件：

    python collect_server_repro.py \
      --project-root "$PWD" \
      --output-dir "$PWD/server_repro_export" \
      --source-file tools/run_fpga_qat_eval1_four_datasets.py \
      --source-file tools/benchmark_up_gpu_power.py \
      --source-file tools/run_up_gpu_power.sh

4. 收集范围
------------

默认复制明确指定的 Python、shell、utils 文件，以及结果目录中的 JSON、CSV 和 LOG。
默认排除 MAT 数据集、checkpoint、预测图、RTL/Vivado 文件、CUDA 二进制和完整 shell
历史。单文件默认上限 16 MiB，总复制量默认上限 256 MiB；需要更大日志时显式增加：

    --max-file-mib 64 --max-total-mib 1024

`--include-checkpoints` 和 `--include-repro-artifacts` 会复制较大的模型或预处理文件，
只在确实需要传输时使用。公开前请检查 manifest.json；它记录跳过文件、原始 SHA256、
打包 SHA256 和路径替换结果。

5. 复制和验证
--------------

将 `FPGA-MambaHSI-server-*.zip` 和 `.zip.sha256` 复制到本地后解压到临时目录。先
检查清单和路径，再把需要公开的源码、轻量结果及许可证合入仓库。不要把数据集、
checkpoint、凭据、CUDA 运行库或 RTL 工程直接上传到公开仓库。
