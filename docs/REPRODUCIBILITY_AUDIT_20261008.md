# 硬件文件与运行入口

## 源码和输入文件

| 文件 | 用途 | 大小 | SHA256 |
|---|---|---:|---|
| [hdmiout_i2c_dri.v](../hardware/vendor/hdmi/hdmiout_i2c_dri.v) | HDMI I²C 驱动 | 23570 bytes | `ec53b87de9cbf00ca438d3e443d89101f37377c8ff13663f86c41be00905b917` |
| [hdmiout_i2c_ms7210_cfg.v](../hardware/vendor/hdmi/hdmiout_i2c_ms7210_cfg.v) | MS7210 寄存器配置 | 6759 bytes | `613db85e571decdb0f1d619070acec6b200e01ba9faf7e66921a4b949af46ce8` |
| [hdmiout_ms72xx_ctl.v](../hardware/vendor/hdmi/hdmiout_ms72xx_ctl.v) | HDMI 控制模块 | 3337 bytes | `1a96dd711f046aa0cbc50bc143ebf97750c85a025fad551f98592d033058ee78` |
| [scene_tiles_uint8.bin](../hardware/models/up_d1/scene/scene_tiles_uint8.bin) | UP 场景，858 个 UINT8 tile | 3514368 bytes | `5e830f01222de08bbffd59d391f73d8130972deee529c75f7506071edf7c752f` |

HDMI 源码来自 ALIENTEK 板卡 BSP，文件内保留原版权标识。板级构建默认读取 `hardware/vendor/hdmi/`；`--vendor-dir` 可指定另一份具有相同 SHA256 的源码。

## 板级仿真输入

`hardware/board/sim/` 包含单 tile 输入及期望 logits，以及 `REAL_FIVE=1` 分支使用的以下文件：

| 文件 | 内容 | 大小 | SHA256 |
|---|---|---:|---|
| [real5_input_128b.mem](../hardware/board/sim/real5_input_128b.mem) | 整图输入前 5 个 tile，128 bit/行 | 43520 bytes | `7732bf1296d02d7d06dfa1dce022a70039952ad07561402c7e71b969374220a8` |
| [real5_expected_logits_chw.mem](../hardware/board/sim/real5_expected_logits_chw.mem) | 前 5 个 tile 的 INT8 logits，tile/channel/y/x 顺序 | 2160 bytes | `3093572b0694e1d827e385f86a4401f39a87b8268c369f97bd3e655bdefd9863` |

板级入口默认使用 `REAL_FIVE=0`，重复第一个 tile 五次；全网络仿真使用 16 个不同 tile。

## 工具、驱动和设备

- Windows，Python 3.8+；读取 `.npy` 标签时使用 NumPy。
- Vivado 2025.2，安装 KU060 器件支持，通过 `VIVADO_BAT` 指定入口。
- ALIENTEK KU060 板卡，FPGA 型号 `xcku060-ffva1156-2-i`。
- Windows XDMA AXI-MM 驱动及 HDMI 显示链路。
- 构建目录使用新的短 ASCII 路径。

## 运行入口

以下命令从仓库根目录执行：

```bat
python hardware/run.py check
python -m unittest discover -s hardware/tests -v
python hardware/run.py prepare N3 C:/fpga/n3_v1
python hardware/run.py sim C:/fpga/n3_v1
python hardware/run.py build C:/fpga/n3_v1
```

板级工程将 `N3` 替换为 `board`，并使用独立的输出目录。`prepare --stage-only` 生成工程输入文件，`create` 调用 Vivado 创建工程。IP 配置及初始化文件在工程准备时写入构建目录。

| 功能 | 入口 | 说明 |
|---|---|---|
| 全网络和板级工程 | [hardware/run.py](../hardware/run.py) | `N0`、`N1`、`N2`、`N3`、`board` 五个配置 |
| 局部非线性核 | [hardware/nonlinear/run.tcl](../hardware/nonlinear/run.tcl) | 参数为动作、方法、核、输出目录、D 路径选项 |
| 功耗采集与报告 | [hardware/power/n3_power.py](../hardware/power/n3_power.py) | `check`、`probe`、`capture`、`report`、`all`；通过 `--build` 指定已完成布线的 N3 工程 |
| 功耗批处理 | [功耗操作说明](../hardware/power/README_V2.md) | `run_saif_probe.bat`、`run_capture_after_probe.bat`、`run_n3_power.bat` |
| PCIe 场景发送与标签回读 | [send_scene_v2.py](../hardware/host/send_scene_v2.py) | E2 协议、双输入 bank、DDR 标签回读 |

## 整图标签与模型参数

[n3_prediction_rtl_integer.npy](../evidence/completion_20260923/nonlinear_D1/n3_prediction_rtl_integer.npy) 的形状为 610 × 340，类型为 `uint8`，类别 ID 为 0–8。标签字节 SHA256 为 `28f8b8f34b34f2f58ba67a2fe1fa0de20fd3fae80fb8209a5ebb6e44e5481354`。

```bat
python hardware/host/send_scene_v2.py --scene hardware/models/up_d1/scene/scene_tiles_uint8.bin --expected-labels evidence/completion_20260923/nonlinear_D1/n3_prediction_rtl_integer.npy --require-ii4096 --output results/board_test_01
```

[models/up_d1/rom](../hardware/models/up_d1/rom) 提供构建使用的参数初始化文件。[数据约定](../hardware/docs/DATA_CONTRACT.md) 说明输入布局、量化、标签插值和地址映射。这些已导出的参数用于工程构建；重新导出模型参数时使用训练 checkpoint。当前目录保存的是参数导出文件，checkpoint 标识为 `d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33`。
