# Data preparation

## 1. Configure directories

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export CONFIG="current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16"
cd "$PROJECT_ROOT"
mkdir -p "$DATA_ROOT/UP" "$DATA_ROOT/HanChuan" "$DATA_ROOT/HongHu" "$DATA_ROOT/Houston" "$RESULTS_ROOT"
```

## 2. Obtain the full scenes

Download links: [MambaHSI datasets](https://github.com/li-yapeng/MambaHSI#data-preparation), [WHU-Hi MATLAB datasets](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm), [Houston 2013 provider](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/).

Place MATLAB v5 files under `DATA_ROOT`:

```text
data/
├── UP/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── HanChuan/
│   ├── WHU_Hi_HanChuan.mat
│   └── WHU_Hi_HanChuan_gt.mat
├── HongHu/
│   ├── WHU_Hi_HongHu.mat
│   └── WHU_Hi_HongHu_gt.mat
└── Houston/
    ├── Houston.mat
    └── Houston_GT.mat
```

| Dataset | Image variable | Label variable | Image H × W × bands | Classes |
|---|---|---|---|---:|
| UP | `paviaU` | `paviaU_gt` | 610 × 340 × 103 | 9 |
| HanChuan | `WHU_Hi_HanChuan` | `WHU_Hi_HanChuan_gt` | 1217 × 303 × 274 | 16 |
| HongHu | `WHU_Hi_HongHu` | `WHU_Hi_HongHu_gt` | 940 × 475 × 270 | 22 |
| Houston | `Houston` | `Houston_GT` | 349 × 1905 × 144 | 15 |

Labels: H × W, background `0`, foreground `1..C`. Scene sizes: [MambaHSI, Sections V-A1–V-A4](https://arxiv.org/html/2501.04944v1#S5.SS1).

## 3. Convert HongHu NPY files when needed

Input filenames: `HongHu/WHU_Hi_HongHu.npy`, `HongHu/WHU_Hi_HongHu_gt.npy`.

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --datasets HongHu --convert-honghu
```

For an existing MAT pair, proceed to validation.

## 4. Validate all scenes and save file hashes

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" \
  --json "$RESULTS_ROOT/data_manifest.json"
```

`--json`: choose a new report filename for each run. `--datasets` accepts comma-separated names, for example `--datasets UP,HanChuan`.

Validate one dataset without writing a report:

```bash
python tools/prepare_data_formats.py --data-root "$DATA_ROOT" --datasets UP
```

## 5. Set generated-model paths

```bash
export FP32_TEMPLATE="$FP32_ROOT/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/$CONFIG"
export QAT_TEMPLATE="$QAT_ROOT/{dataset}/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/{dataset}_patch_max_D_mean_freeze20_eval1/$CONFIG/run_seed{seed}"
```

`FP32_TEMPLATE`: full configuration directory. `QAT_TEMPLATE`: seed directory. Training commands: [EXPERIMENTS.md](EXPERIMENTS.md).

FP32 configuration inputs for QAT/simulation:

```text
$FP32_TEMPLATE/
  train_only_preprocess.npz
  spatial_split_masks.npz
  spatial_split.json
  run_seed{seed}/best_model.pth
  run_seed{seed}/model_config.json
  run_seed{seed}/sample_indices.npz
```

QAT seed inputs for simulation:

```text
$QAT_TEMPLATE/
  result.json
  best_qat_foldaware.pth
  sample_indices.npz
  qat_deploy_test_prediction.npy
```
