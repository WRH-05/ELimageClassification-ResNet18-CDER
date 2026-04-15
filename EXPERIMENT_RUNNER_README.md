# Experiment Runner Quick Reference

This document summarizes the new scripts added to support the multi-seed variance study and SAHL weight ablation.

## New Scripts

### 1. `experiment_runner.py`
Orchestrates the train → export → report pipeline for multiple configurations.

**Usage: Variance Study (Multi-Seed)**
```powershell
& ".\.venv\Scripts\python.exe" experiment_runner.py --mode variance
```
Trains V1 (MSE) and V3.1 SAHL (weighted_l1, 2.5x) across seeds 42, 123, 2026.
- Checkpoints saved as: `models/pth/best_model_seed{seed}_{loss_type}_w{multiplier}.pth`
- ONNX exports: `models/onnx/best_model_seed{seed}_{loss_type}_w{multiplier}.onnx`
- Test reports: `testCsv/test_split_report_seed{seed}_{loss_type}_w{multiplier}.csv`

**Usage: Ablation Study (Weight Sweep)**
```powershell
& ".\.venv\Scripts\python.exe" experiment_runner.py --mode ablation
```
Trains SAHL (weighted_l1) across weights 1.0x, 1.5x, 2.5x, 5.0x at seed 42.
- Same output structure as variance study (seed fixed at 42)

### 2. `aggregate_results.py`
Reads test report CSVs and computes summary metrics in Markdown/CSV format.

**Usage: Aggregate Variance Results**
```powershell
& ".\.venv\Scripts\python.exe" aggregate_results.py --mode variance
```
Outputs: `results/variance_metrics.md` (Mean ± StdDev for F1, Precision, Recall, MAE, Critical Recall)

**Usage: Aggregate Ablation Results**
```powershell
& ".\.venv\Scripts\python.exe" aggregate_results.py --mode ablation
```
Outputs:
- `results/ablation_summary.csv` (per-weight metrics)
- `results/ablation_summary.md` (formatted table)

### 3. `plot_ablation_study.py`
Generates a publication-ready PDF plot of ablation results.

**Usage**
```powershell
& ".\.venv\Scripts\python.exe" plot_ablation_study.py \
  --csv_path results/ablation_summary.csv \
  --output results/ablation_study.pdf
```
Outputs: `results/ablation_study.pdf` (F1, Critical Recall, Precision vs Weight)

### 4. `export_best_sahl_model.py`
Selects the best 2.5x SAHL model from the variance study and exports to ONNX.

**Usage**
```powershell
& ".\.venv\Scripts\python.exe" export_best_sahl_model.py
```
- Evaluates all variance-study runs at seed 42, 123, 2026 with weighted_l1 @ 2.5x
- Selects best by F1 @ threshold 0.65, MAE as tie-breaker
- Outputs: `models/onnx/best_sahl_2.5x_final.onnx`

## End-to-End Workflow

```powershell
# 1. Run variance study (MSE vs SAHL @ 2.5x across 3 seeds)
& ".\.venv\Scripts\python.exe" experiment_runner.py --mode variance

# 2. Run ablation study (SAHL @ 1.0x, 1.5x, 2.5x, 5.0x at seed 42)
& ".\.venv\Scripts\python.exe" experiment_runner.py --mode ablation

# 3. Aggregate variance results
& ".\.venv\Scripts\python.exe" aggregate_results.py --mode variance

# 4. Aggregate ablation results
& ".\.venv\Scripts\python.exe" aggregate_results.py --mode ablation

# 5. Plot ablation study
& ".\.venv\Scripts\python.exe" plot_ablation_study.py

# 6. Export best 2.5x SAHL model
& ".\.venv\Scripts\python.exe" export_best_sahl_model.py
```

## Output Files

### Checkpoints and ONNX
- `models/pth/best_model_seed*.pth` — Training checkpoints for each configuration
- `models/onnx/best_model_seed*.onnx` — ONNX exports for each checkpoint
- `models/onnx/best_sahl_2.5x_final.onnx` — Final selected best SAHL model

### Reports
- `testCsv/test_split_report_seed*.csv` — Per-image predictions (image_path, target, prediction, abs_error, squared_error)

### Summary Tables
- `results/variance_metrics.md` — Variance study results (Mean ± StdDev)
- `results/ablation_summary.csv` — Ablation metrics (weight, f1, precision, recall, critical_recall, mae)
- `results/ablation_summary.md` — Ablation table (human-readable)

### Plots
- `results/ablation_study.pdf` — Ablation visualization (F1, Critical Recall, Precision vs Weight)

## Metrics Definitions

### F1, Precision, Recall
- Computed at prediction threshold 0.65, target threshold 0.65
- Binarized: y_true = (target >= 0.65), y_pred = (prediction >= 0.65)

### Critical Recall
- Recall on high-severity (critical) cases
- Computed: prediction_threshold = 0.6, target_threshold = 0.8
- Binarized: y_true = (target >= 0.8), y_pred = (prediction >= 0.6)

### MAE
- Mean absolute error on continuous predictions
- MAE = mean(|prediction - target|)

## Notes

- All experiments use CPU (--device cpu) by default for reproducibility
- Random seeds are fixed for train/val/test splits via stratified_split(seed=...)
- Each run creates isolated checkpoints and reports to avoid overwrites
- The final SAHL export selects by F1 with MAE tie-break from the variance study
