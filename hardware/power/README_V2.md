# N3 功耗采集与报告

本流程使用已经完成 RTL 仿真与布线的 N3 100 MHz D1 全网工程。
入口为 `hardware/power/n3_power.py`，构建目录中包含
`nf_fullnet.xpr`、`network_routed.dcp`、`sim_ok.txt` 和 `build_status.txt`。

## 1. 检查构建

在仓库根目录的 Windows CMD 中运行，按安装位置设置 `VIVADO_BAT`：

```bat
set VIVADO_BAT=C:\Xilinx\2025.2\Vivado\bin\vivado.bat
python hardware/power/n3_power.py check --build D:/fpga/n3_v1
```

`check` 读取构建文件、源文件指纹、RTL PASS、布线状态和磁盘空间。
采集流程使用外部 Python，并以 `-I` 调用子解释器。
`--vivado PATH` 可为单次调用指定 Vivado；否则依次读取 `VIVADO_BAT`
和系统 `PATH`。

也可使用同目录批处理入口，参数与 Python 入口一致：

```bat
set "NF_PYTHON=C:\Python311\python.exe"
hardware\power\run_saif_probe.bat --build D:/fpga/n3_v1 --out D:/fpga/probe_bat_v1
hardware\power\run_capture_after_probe.bat --build D:/fpga/n3_v1 --probe-dir D:/fpga/probe_bat_v1 --out D:/fpga/power_bat_v1
```

`NF_PYTHON` 按本机 Python 安装路径设置，省略时使用 `python`。
`run_saif_probe.bat` 执行 `probe`，`run_capture_after_probe.bat` 执行 `all`
（capture 后接 report）。通用入口 `run_n3_power.bat` 的首参数为运行模式，
例如 `run_n3_power.bat check --build D:/fpga/n3_v1`。

输出目录使用短 ASCII 路径，长度不超过 48 字符，且目录尚不存在。
probe/capture 默认要求至少 40 GiB 空闲空间，report 要求 5 GiB。
一次采集的 SAIF 约为 18.8 GB，运行时内存需求为数十 GiB。
同一工程的构建和功耗操作顺序运行。

## 2. 登记检查

```bat
python hardware/power/n3_power.py probe --build D:/fpga/n3_v1 --out D:/fpga/probe_v1
```

probe 新建 post-route functional 仿真 fileset，使用 `debug all` 和
`-nosignalhandlers`。仿真在 0 时刻登记对象：前三层逐层处理，
其余子树递归枚举，每批最多 1024 个对象调用 `log_saif`。
登记后运行 100 ns（10 周期）的 reset 活动并关闭 SAIF。

| 输出 | 内容 |
|---|---|
| `probe_console.log` | 编译、展开、枚举、登记、运行和关闭日志 |
| `registration.log` | 对象登记批次、计数和时间 |
| `PROBE_OK.tcl` | 登记与 SAIF 关闭成功的指纹记录 |
| `probe_NOT_FOR_POWER.saif` | reset 阶段活动 |

## 3. 正式采集

```bat
python hardware/power/n3_power.py capture --build D:/fpga/n3_v1 --probe-dir D:/fpga/probe_v1 --out D:/fpga/capture_v1
```

capture 核对 probe 的源文件、DCP 和脚本指纹，以及对象数和分区数。
仿真输入为首 16 个不同 tile，输出为 2304 个 logit 字节，tile 间隔为
4096 周期。100 MHz 下先运行 9216 周期，再连续记录 32768 周期，
对应仿真时间 92160–419840 ns。活动窗口分成 8 段 4096 周期打印进度，
SAIF 在整个窗口连续记录。

关闭 SAIF 后继续推理至全部 16 个 tile 与参考逐位一致，生成
`CAPTURE_OK.tcl`。每次运行的日志、SAIF 和完成记录位于其输出目录。

## 4. 生成功耗报告

```bat
python hardware/power/n3_power.py report --build D:/fpga/n3_v1 --capture-dir D:/fpga/capture_v1 --out D:/fpga/power_v1
```

report 读取 `CAPTURE_OK.tcl` 并验证采集、DCP 和 SAIF 的指纹。
也可使用 `all` 在一次入口调用中顺序完成 capture 和 report：

```bat
python hardware/power/n3_power.py all --build D:/fpga/n3_v1 --probe-dir D:/fpga/probe_v1 --out D:/fpga/power_all_v1
```

输出包括 `power_saif.rpt`、`power_advisory.rpt`、
`operating_conditions.rpt`、`activity_*.rpt`、`protocol.txt`、
控制台注入统计和 `REPORT_COMPLETE.txt`。

## 5. XSim NET 名称处理

对于含非法 `(null)` NET 标识符的 XSim SAIF，
`saif_sanitize.py` 输出名称规范化后的文件及 JSON 清单。
转换使用稳定的 `XsimNull_<hash>` 标识符，保持活动计数、层级和时长。

```bat
python hardware/power/saif_sanitize.py D:/fpga/capture_v1/activity.saif --capture-receipt D:/fpga/capture_v1/CAPTURE_OK.tcl --output D:/fpga/activity_clean_v1.saif
python hardware/power/n3_power.py report --build D:/fpga/n3_v1 --capture-dir D:/fpga/capture_v1 --saif-manifest D:/fpga/activity_clean_v1.saif.json --out D:/fpga/power_clean_v1
```

`--scan-only` 只扫描名称。转换输出使用独立文件名，JSON 清单记录输入、
输出指纹和名称数量；report 的 `--saif-manifest` 参数读取该清单。

## 6. 日志与运行状态

| 最后出现的阶段 | 当前操作 |
|---|---|
| `ENUM_BEGIN` | 当前层或子树对象枚举 |
| `SCOPES_BEGIN` | 子层级枚举 |
| `LOG_BEGIN` | 当前批次 SAIF 登记 |
| `PROBE_RUN_BEGIN` | XSim 仿真推进 |
| `CLOSE_SAIF_BEGIN` | SAIF 汇总和写出 |

暂停的仿真依赖当前 XSim 进程继续运行。进程终止后，从新的 probe/capture
重新启动；已完成的 capture 可以单独调用 report。

## 7. 功耗口径与脚本检查

功耗范围为 D1 推理计算部分，使用布线后功能网表与 SAIF。
时钟为 100 MHz，工艺为 typical，结温为 50 °C。
活动模型包括已匹配 SAIF 活动与未匹配网络的传播/默认活动。
SDF 毛刺活动和板级 PCIe/DDR/HDMI 位于该估算范围之外。

主机侧检查命令：

```bat
python hardware/power/test_saif_v2.py
python hardware/power/test_saif_sanitize.py
```

这些检查使用 Tcl mock 验证对象分区、批次上限、异常传播、采集窗口与
最终 PASS 分支，并验证 SAIF 名称转换。实际功耗数据见
[N3 功耗说明](../evidence/fullnet/N3_POWER_AUDIT_20260925.md)。
