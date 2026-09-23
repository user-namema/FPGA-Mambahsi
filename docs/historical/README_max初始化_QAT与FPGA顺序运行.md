# patch_max / all_weight_max：校准模型、QAT 与 FPGA 顺序运行

## 1. 更新文件

将增量包中的 `train_mambahsi_spatial_split_dense_qat.py`、`qat_forensics.py`、`run_max_init_qat_fpga.py` 放到服务器工程目录。依赖原工程中已有的 `both_FPGA_single_qat_source.py`、`ssm_error_ablation.py`、`ssm_d_path.py` 及模拟器原有依赖。本包不覆盖 FPGA 模拟器或 RTL。

本流程会生成真正可供模拟器读取的 checkpoint、量化配置和保存预测，不能用上一轮“只验证”的结果文件夹代替。

## 2. 一条命令按顺序跑完两种初始化

```bash
cd ~/mzz/MambaHSI
conda activate mambahsi

CONFIG_D1=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
FP32_D1="results/SPATIAL_SPLIT_3WAY_DENSE/UP_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG_D1"

python run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --stage all --max-epoch 20 \
  --output-dir ./max_init_qat_fpga_seed0_6
```

执行顺序：

1. `patch_max` 校准模型导出（0训练步），对 seed0、6 各做 FPGA 模拟。
2. `all_weight_max` 校准模型导出（0训练步），对 seed0、6 各做 FPGA 模拟。
3. `patch_max` 训练20轮，按验证 mAcc 选出的模型，对 seed0、6 各做 FPGA 模拟。
4. `all_weight_max` 同上。

总共4次训练运行（2初始化×2seed），4份未训练校准模型，以及8次完整模拟。`--max-epoch` 只决定 QAT 轮数，不影响校准模型的零训练步。默认20轮；确认结果后可在新目录改成100轮或增加 seed。

也可分开执行，第一批完成并分析后再跑第二批：

```bash
python run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --stage calibrated --output-dir ./max_init_calibrated_sim_seed0_6

python run_max_init_qat_fpga.py \
  --fp32-dir "$FP32_D1" --data-path ./data \
  --dataset UP --device cuda:0 --seeds 0,6 \
  --stage qat --max-epoch 20 --output-dir ./max_init_qat20_sim_seed0_6
```

两种 stage 都自动包含 patch_max 和 all_weight_max；无需写两次初始化循环。`qat` 从同一 FP32 重新执行相同校准、初始化，再训练，不读取上一 stage 的测试结果或据此选择模型。

## 3. 固定实验协议

- 先进行原 mean 权重下的100步训练集校准，然后替换指定普通权重尺度；不重新校准激活，保持与已完成的三组验证一致。
- max 使用实际 BN 折叠后权重除以127。patch_max 只改第一层；all_weight_max 改全部33个普通权重尺度。
- D8、shared-U、dt输入9位/输出8位不变；D继续max初始化。
- batch32、eval batch8，可分别通过 `--batch-size` / `--eval-batch-size` 修改；配对实验应保持一致。
- QAT 默认普通参数/D/BN affine 学习率1e-5，constant；第1轮冻结全部LSQ尺度和BN运行统计，BN γ/β仍训练。不同时加入上一轮低学习率或冻结affine策略。
- 每轮验证，epoch0也纳入候选。最佳模型可能仍为epoch0，这表示训练未改善验证选择指标，不是未执行训练；`optimization_steps`记录实际训练步数。
- QAT 从第1轮捕获验证 OA 下降≥5个百分点的事件，每seed最多3对，继续可用 `replay_qat_jump.py` 分析。
- 校准模型导出会构建优化器供公共报告代码使用，但不会调用优化器step；`optimization_steps=0`、`best_epoch=0`，不画训练loss曲线。
- 为适配模拟器的保存预测复核，校准导出也会计算测试预测。测试/FPGA指标仅报告，不选择epoch或训练参数。

模拟器自动读取各自新导出的真实尺度和数值配置，重新生成相关系数及输出文件，不复用旧mean模型的ROM/bias/再量化参数。每次启用SSM误差报告，分析split为test，调用的仍是 `both_FPGA_single_qat_source.py`。

## 4. 目录与判读

```text
max_init_qat_fpga_seed0_6/
  pipeline_manifest.json
  pipeline_results.json
  calibrated/
    patch_max/
      model.log
      models/.../run_seed0/
      models/.../run_seed6/
      sim_seed0.log
      sim_seed6.log
      simulations/seed0/fpga_simulation_result.json
      simulations/seed6/fpga_simulation_result.json
    all_weight_max/...
  qat/
    patch_max/...
    all_weight_max/...
```

`pipeline_results.json` 汇总每个stage/初始化/seed的初始验证、最佳轮次、实际训练步数、模型测试OA及完整FPGA结果。`pipeline_manifest.json`记录每条实际命令和状态。保存模型目录还包含 `weight_initialization.json`（逐层旧/新尺度），`result.json`记录初始化策略。

判读时分开两类变化：

1. 校准模型→QAT：同初始化下训练是否提高验证表现，是否引入跳变。
2. 每个模型→其对应FPGA：优先读模拟器的 deployment-mode loss（batch1参考与INT8），同时保留saved-batch差异；不要将batch差异全算成定点误差。

本程序模拟的是软件整数路径；并不代表D通路RTL已实现或已上板验证。

## 5. 单独导出/训练与模拟

若要手动控制某一组，训练脚本新增参数：

```text
--weight_init_policy mean|patch_max|all_weight_max
--export_calibrated_only
```

前者默认mean，保持旧行为；后者强制0轮并导出校准模型。正常QAT只指定初始化策略，并设置 `--include_calibrated_baseline --freeze_lsq_epoch 1 --freeze_bn_epoch 1`。不能与 `--weight_init_validation_only` 或旧的初始量化开关诊断同时使用。

拿到真实run目录后，两种初始化的模拟命令形式相同：

```bash
python both_FPGA_single_qat_source.py \
  --qat-run-dir "$QAT_RUN" --fp32-dir "$FP32_D1" \
  --dataset UP --seed 0 --data-path ./data --device cuda:0 \
  --report-ssm-errors --ssm-analysis-split test \
  --output-dir ./manual_sim_new
```

`QAT_RUN` 指向本次导出的 `run_seed0`；其他seed必须同步修改 `--seed`。不能指向父目录或上一轮只验证目录。

## 6. 失败与计划

输出目录必须为空；失败会保留日志、产物和manifest，显示最后80行并停止。程序不自动跳过失败结果或继续混合统计；修复后可用手动模拟命令复用已成功导出的模型，并使用新的模拟输出目录。

`--dry-run`只保存模型任务计划并报告预计模拟次数，不执行GPU工作；模拟命令在模型真实输出目录解析后写入manifest。正式运行使用另一个新目录。CUDA不可用时本流程报错，不静默改CPU；仅功能验证可显式使用 `--device cpu`。
