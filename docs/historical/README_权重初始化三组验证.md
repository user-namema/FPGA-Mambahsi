# 普通权重初始化：三组不训练的完整验证对照

2026-09-17。增量更新 `train_mambahsi_spatial_split_dense_qat.py` 与 `qat_forensics.py`，放在服务器工程同一目录；保留已有 `ssm_error_ablation.py`。本入口不创建优化器、不执行训练或测试集模型评估，不生成部署 checkpoint。

## 运行

```bash
cd ~/mzz/MambaHSI
conda activate mambahsi

CONFIG_D1=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
FP32_D1="results/SPATIAL_SPLIT_3WAY_DENSE/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"

python train_mambahsi_spatial_split_dense_qat.py \
  --dataset UP --data_set_path ./data --fp32_dir "$FP32_D1" \
  --work_dir ./qat_weight_init_validation_seed0_6 --run_tag weight_init_compare \
  --seeds 0,6 --device cuda:0 \
  --use_D true --use_z false --ssm-u-quantization shared --d-weight-bits 8 \
  --dt-input-bits 9 --dt-output-bits 8 \
  --batch_size 32 --eval_batch_size 8 --calibration_steps 100 \
  --weight_init_validation_only
```

一个命令运行2个 seed，每个 seed 对同一个完整保存验证集评估3组。另计算原 FP32 验证指标作为参考。不需要设置 epoch，也不要叠加 `--initial_diagnostics_only` 或 `--diagnose_initial_quantization`；它们是前一版的另一套消融，代码会拒绝混用。

输出目录需要是新目录。默认仍按原预处理、空间划分和样本索引运行，校准仅用训练集。

## 三组如何控制变量

每个 seed 先正常执行一次 mean 权重初始化下的校准，再冻结初始化更新。三个验证分支复用这份状态，不分别校准激活。

| case | 普通权重尺度 |
|---|---|
| `mean` | 保留原 mean 校准尺度：`2*mean(abs(W))/sqrt(127)` |
| `patch_max` | 仅 `patch_embedding.0` 改为 `max(abs(W_fold))/127` |
| `all_weight_max` | 全部普通卷积/投影权重改为 `max(abs(W_effective))/127` |

BN 层用 `W_fold = W * gamma/sqrt(running_var+eps)`；其他层用原权重。全零权重的尺度下限为1e-8，避免除零。当前 D1 结构预计三组分别改变0、1、33个权重尺度。

- 浮点权重、BN 参数和运行统计不变。
- D 继续使用原来的 max 初始化，D 权重和尺度均不变。
- 激活、B/C、dt 输入/输出、残差及输出边界尺度均不变；它们仍正常量化。
- 普通权重改变尺度后，相应 INT32 bias 的尺度乘积 `s_x*s_w` 按原规则更新。不是固定 bias 整数码的纯权重替换实验。
- 每组执行后都恢复原尺度，检查整个 state_dict 与对照前一致，避免后一个分支继承前一个分支的更改。

该入口保持现有计算后端，不同时更改 TF32。它不绕过或修改旧初始诊断的转换检查；它是独立的三组初始化对照。

## 结果位置

```text
qat_weight_init_validation_seed0_6/
  SPATIAL_SPLIT_3WAY_DENSE_QAT/
    UP_weight_init_compare/<CONFIG_D1>/
      weight_init_validation_summary.json
      run_seed0/weight_init_validation/
      run_seed6/weight_init_validation/
        weight_init_validation.json
        weight_init_summary.csv
        mean_validation_prediction.npz
        patch_max_validation_prediction.npz
        all_weight_max_validation_prediction.npz
```

JSON 包含原 FP32 验证指标、三组 OA/mAcc/Kappa/逐类准确率/混淆矩阵、相对 mean 的 OA 变化、相对 mean 的预测吻合率、逐层旧/新尺度、运行环境和 `model_state_unchanged`。

CSV 的 OA/mAcc/Kappa 为0–1；`OA_minus_mean_pp` 为百分点。预测 NPZ 只保存验证标签位置的一维索引、标签和预测，不保存测试或背景预测。

先比较 patch_max 能否恢复大部分初始损失，再看 all_weight_max 是否提供额外收益。不要只看 OA，还看 mAcc 和各类指标。完整验证结果出来后再选择需要重新训练的对照；这里没有新训练 checkpoint，不能直接拿本目录跑 FPGA 模拟。

请回传两个 seed 的 `weight_init_validation` 子文件夹；无需回传原模型或数据。
