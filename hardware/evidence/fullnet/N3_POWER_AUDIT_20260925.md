# N3 全网 D1 功耗说明

## 测量条件与结果

Vivado 2025.2 对 `xcku060-ffva1156-2-i` 上已布线的完整 D1 推理计算部分进行
SAIF 驱动功耗估算。在 **100 MHz、typical process、50 °C 结温**条件下，
总片上功耗为 **6.386 W**，其中动态 **5.447 W**、静态 **0.939 W**。
设计范围为推理计算部分，PCIe、DDR 和 HDMI 位于板级工程中。

主报告的设计网匹配数为 **453380/638590（71%）**；其余网络使用 Vivado
传播或默认活动模型。总体 `Confidence Level` 为 `High`，
`power_advisory.rpt` 中同时列出部分网络的 Low/Medium 活动置信度。

## 仿真与活动输入

| 项目 | 数值或配置 |
|---|---|
| 网表 | Post-route functional |
| 时钟 | 100 MHz，由 XDC 约束提供 |
| 预热 | 9216 周期 |
| 活动窗口 | 第 9216–41984 周期 |
| 活动时长 | 32768 周期 / 327680 ns |
| 功能检查 | 首 16 个不同 tile，2304 logit 字节，与独立参考逐位一致 |
| 稳态 tile 间隔 | 4096 周期 |
| DCP SHA-256 | `a864ca8268b77a67be6e9bea143ac970f77b226e06818d3093603425d390d457` |
| 采集 SAIF SHA-256 | `03fcc0376762252b45afa419961cd43b8ce214cbb9133040436177c127969bcc` |
| 导入 SAIF SHA-256 | `32256cde10be648233018490f93bb715ab9c6944d73e129e7215cf1127616bf2` |

活动来自功能仿真，未使用 SDF 时序反标。导入 SAIF 的 1594 个 XSim
`(null)` NET 标识符采用稳定名称；活动计数和窗口时长与采集数据一致。
Vivado 使用约束中的时钟频率计算时钟活动。

## 功耗分项

| 分项 | 功耗（W） |
|---|---:|
| 时钟 | 0.471 |
| CLB 逻辑 | 0.824 |
| 信号 | 1.265 |
| Block RAM | 2.011 |
| DSP | 0.863 |
| I/O | 0.011 |
| 静态 | 0.939 |
| 总计 | 6.386 |

100 MHz 下的 4096 周期/tile 对应 40.96 µs/tile。按稳态吞吐率折算，
总片上功耗对应 **261.6 µJ/tile**，动态功耗对应 **223.1 µJ/tile**。
该数值由功耗估计值与 tile 间隔相乘得到。

## 报告文件

- [功耗主报告](N3_power/power_saif.rpt)：总功耗、资源分项和置信度。
- [活动提示](N3_power/power_advisory.rpt)：网络活动与置信度信息。
- [运行条件](N3_power/operating_conditions.rpt)：器件电压、工艺和温度。
- [采集协议](N3_power/protocol.txt)：网表、时钟和活动窗口。
- [BRAM 使能活动](N3_power/activity_bram_enable.rpt)、
  [BRAM 写使能活动](N3_power/activity_bram_wr_enable.rpt)、
  [LUT RAM 活动](N3_power/activity_lut_ram.rpt)：资源活动明细。

运行方式见 [功耗流程](../../docs/POWER.md)。N0/N1/N2 在当前并行度下提供
综合资源结果，N3 提供布线与功耗结果。
