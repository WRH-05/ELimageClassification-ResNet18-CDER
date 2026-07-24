# AGENTS.md

## Purpose
This file is the operational guide for coding agents working in this repository.
Use it before making code changes.

## Project Summary
Edge-AI pipeline for nighttime EL solar defect assessment:
- Training: PyTorch ResNet18 transfer learning (regression)
- Export: PyTorch checkpoint to ONNX
- Inference: ONNX Runtime, optional MQTT payload publish

Core files:
- `dataset.py`: CSV loading, transforms, and DataLoaders
- `train.py`: model build + training/eval loop + checkpoint save
- `export_model.py`: ONNX export + model check
- `inference_mqtt_mock.py`: ONNX inference + JSON/MQTT payload
- `ELimageClassification_all_in_one.ipynb`: notebook copy of full pipeline

## Environment
- OS: Windows (PowerShell commonly used)
- Python env path usually: `.venv`
- Dependencies: `requirements.txt`

Recommended setup commands:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Runbook
Train:
```powershell
python train.py --csv_path labels.csv --data_root . --epochs 20 --batch_size 32 --learning_rate 1e-4
```

Export ONNX:
```powershell
python export_model.py --checkpoint best_model.pth --onnx_output best_model.onnx
```

Inference:
```powershell
python inference_mqtt_mock.py --onnx_model best_model.onnx --image_path path/to/el_image.png --pad_id simulated_pad_01 --critical_threshold 0.8
```

## Current Data Reality (Important)
The project currently uses CSV-based regression data:
- labels file at `labels.csv`
- images under `images/`

The loader expects CSV columns:
- `image_path`
- `defect_probability`
- `cell_type`

## Dataset Labeling Guidance
Regression target expected by code:
- `defect_probability` in [0, 1]

Split stratification is done by defect-probability buckets:
- [0.0, 0.1667)
- [0.1667, 0.5)
- [0.5, 0.8334)
- [0.8334, 1.0]

If onboarding a new dataset, provide equivalent CSV rows with valid paths and probabilities.

## Coding Rules for Agents
- Preserve existing behavior unless task explicitly asks for changes.
- Keep changes minimal and localized.
- Do not rename public CLI arguments unless required.
- Do not edit unrelated files.
- Prefer deterministic behavior when adding splits/seeded logic.
- Update docs if behavior or command usage changes.

## Notebook Notes
When editing `ELimageClassification_all_in_one.ipynb`:
- Keep pipeline parity with script files unless intentionally diverging.
- Avoid introducing notebook-only magic that blocks script portability.
- Keep training/export/inference toggles explicit (`RUN_TRAINING`, etc.).

## Validation Checklist
After changes, verify as applicable:
1. Dependency install still succeeds.
2. Training command starts and reaches first epoch.
3. `best_model.pth` is created when validation improves.
4. ONNX export creates a valid `best_model.onnx`.
5. Inference prints payload JSON with `severity_score` and `status`.

## If You Are Unsure
- Read `README.md` first.
- Inspect actual dataset directory layout before changing loaders.
- Prefer asking for clarification over silently changing label semantics.
