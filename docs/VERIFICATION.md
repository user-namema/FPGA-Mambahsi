# Verification commands

## 1. Configure paths

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export DATA_ROOT="/absolute/path/to/data"
export RESULTS_ROOT="$PROJECT_ROOT/results"
export FP32_ROOT="$RESULTS_ROOT/SPATIAL_SPLIT_3WAY_DENSE"
export QAT_ROOT="$RESULTS_ROOT/qat_eval1_4datasets"
export DEVICE="cuda:0"
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_ROOT"
```

## 2. Verify source files and numeric records

```bash
python tools/audit_reproducibility.py
python tools/verify_release_records.py
python tools/verify_completion_records.py
python tools/verify_completion_records.py --check-arrays
python tools/paired_statistics.py --output-dir "$RESULTS_ROOT/paired_statistics"
```

`--check-arrays`: requires NumPy. `--output-dir`: destination for computed paired summaries.

## 3. Run software tests

Install [ENVIRONMENT.md](ENVIRONMENT.md) dependencies first.

```bash
python -m unittest discover -s tests -v
cd "$PROJECT_ROOT/software"
python -m unittest discover -s tests -v
cd "$PROJECT_ROOT"
```

## 4. Print configured training and simulation commands

```bash
python experiments/run.py 01_fp32_current --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$RESULTS_ROOT" --dry-run
python experiments/run.py 13_qat_eval1 --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --device "$DEVICE" \
  --output-root "$QAT_ROOT" --dry-run
python experiments/run.py 14_fpga_eval1 --datasets UP --seeds 0 \
  --data-root "$DATA_ROOT" --fp32-root "$FP32_ROOT" --qat-root "$QAT_ROOT" \
  --device "$DEVICE" --output-root "$RESULTS_ROOT/fpga_eval1_UP0" --dry-run
```

`--dry-run` on `experiments/run.py`: print child commands. Remove it to execute stages 01 → 13 → 14.

## 5. Check eval1 simulation inputs

Required inputs: prepared datasets, FP32 configuration files and completed eval1 QAT seed outputs from [DATA.md](DATA.md).

```bash
python software/run_fpga_qat_eval1_four_datasets.py \
  --project-root "$PROJECT_ROOT/software" --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" --datasets UP --seeds 0 --device "$DEVICE" \
  --output-dir "$RESULTS_ROOT/fpga_eval1_preflight_UP0" --dry-run
```

This preflight checks QAT metadata and files. Remove `--dry-run` to execute the full UP seed0 scene.

## 6. Check N0–N3 experiment inputs

```bash
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" --data-path "$DATA_ROOT" \
  --datasets UP --seeds 0 --device "$DEVICE" --trace-tiles 5 \
  --output-dir "$RESULTS_ROOT/backend_error_UP0" --dry-run
```

This preflight checks model, dataset, preprocessing and split hashes. Remove `--dry-run` to execute; `--trace-tiles 5` selects detailed traces for five tiles while inference covers the full scene.

## 7. Run all datasets and seeds

```bash
python software/run_fpga_qat_eval1_four_datasets.py \
  --project-root "$PROJECT_ROOT/software" --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" \
  --data-path "$DATA_ROOT" --datasets UP HanChuan HongHu Houston \
  --seeds 0,1,2,3,4,5,6,7,8,9 --device "$DEVICE" \
  --output-dir "$RESULTS_ROOT/fpga_eval1_4datasets"
python software/run_backend_error_four_datasets.py \
  --qat-root "$QAT_ROOT" --fp32-root "$FP32_ROOT" --data-path "$DATA_ROOT" \
  --datasets UP HanChuan HongHu Houston --seeds 0,1,2,3,4,5,6,7,8,9 \
  --device "$DEVICE" --trace-tiles 5 --output-dir "$RESULTS_ROOT/backend_error_4datasets"
```

`--resume`: continue with the same inputs, parameters and output directory. For a new experiment, choose a new output directory.
