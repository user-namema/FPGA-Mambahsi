# Result files and checks

## 1. Configure directories

```bash
export PROJECT_ROOT="/absolute/path/to/FPGA-MambaHSI"
export RESULTS_ROOT="$PROJECT_ROOT/results"
cd "$PROJECT_ROOT"
```

## 2. Verify result files

```bash
python tools/verify_release_records.py
python tools/verify_completion_records.py
python tools/verify_completion_records.py --check-arrays
```

`--check-arrays` requires NumPy.

## 3. Run paired statistics

```bash
python tools/paired_statistics.py --output-dir "$RESULTS_ROOT/paired_statistics"
```

## 4. Select result files

| Analysis | Files |
|---|---|
| Result index | `reference_run_index.csv`, `reference_run_counts.json` |
| QAT metrics | `qat_metrics.json`, `eval1_*` |
| GPU benchmarks | `gpu_*_batch_summary.csv` |
| Shared/per-channel A | `paired_accuracy_*`, `pole_metrics_*` |
| Paired statistics | `paired_inference.*` |
| Error sources | `d1_error_sources.json`, `d1_local_replay/` |
| eval1 simulation and GPU power | `server_update_20260923/` |
| FP32 GPU, dt diagnostics and D1 | `completion_20260923/` |

Training and simulation commands: [EXPERIMENTS.md](../docs/EXPERIMENTS.md).
