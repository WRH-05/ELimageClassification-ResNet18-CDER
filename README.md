# ELimageClassification

Edge-AI pipeline for nighttime EL solar defect severity regression using transfer learning (ResNet18), with ONNX Runtime inference for Raspberry Pi-class devices.

## Project Files
- `dataset.py` - CSV data loading, denoising, transforms, and DataLoaders
- `train.py` - Transfer-learning regression training loop with weighted sampling, Huber loss, scheduler, and early stopping
- `export_model.py` - Export `best_model.pth` to ONNX
- `inference_mqtt_mock.py` - ONNX Runtime inference + JSON/MQTT mock payload
- `evaluate_test_split_report.py` - Per-image held-out test split report (target/prediction/error CSV)

## Final Deployment Recommendation (V3.1)
- Production artifact: `best_model_v3_1_goldilocks.onnx`
- Recommended operational threshold: `0.65` (F1-optimal balance on held-out test set)
- Tested behavior at threshold `0.65` on target `>= 0.8`:
	- Precision: `0.8051`
	- Recall: `0.8796`
	- F1: `0.8407`
- At threshold `0.60` (more sensitive mode):
	- Precision: `0.7742`
	- Recall: `0.8889`

## Deployment Constraints
- The exported production model is ONNX Opset `18`.
- Raspberry Pi target must run an ONNX Runtime build compatible with Opset 18.
- Recommended minimum runtime version: `onnxruntime >= 1.14`.
- Validate runtime compatibility on target before rollout with a one-image smoke inference.

## 1) Install dependencies (PowerShell)
```powershell
& ".\.venv\Scripts\Activate.ps1"
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## 2) Optional: verify GPU visibility
```powershell
& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 3) Train V2 model (weighted sampler + Huber + scheduler + early stopping)
```powershell
& ".\.venv\Scripts\python.exe" train.py --csv_path labels.csv --data_root . --epochs 20 --batch_size 32 --learning_rate 1e-4 --weight_decay 1e-5 --num_workers 2 --checkpoint_path best_model_v2_full.pth --device cuda --loss_type smoothl1 --huber_beta 0.5 --scheduler_patience 3 --scheduler_factor 0.5 --scheduler_min_lr 1e-6 --early_stopping_patience 7 --early_stopping_delta 1e-4
```

If you need CPU fallback, replace `--device cuda` with `--device cpu`.

## 4) Export ONNX model
```powershell
& ".\.venv\Scripts\python.exe" export_model.py --checkpoint best_model_v2_full.pth --onnx_output best_model_v2_full.onnx
```

## 5) Run edge inference
```powershell
& ".\.venv\Scripts\python.exe" inference_mqtt_mock.py --onnx_model best_model_v3_1_goldilocks.onnx --image_path images/cell0001.png --pad_id simulated_pad_01 --critical_threshold 0.65
```

## 6) Generate held-out test split report
ONNX path evaluation:
```powershell
& ".\.venv\Scripts\python.exe" evaluate_test_split_report.py --csv_path labels.csv --data_root . --seed 42 --image_size 224 --onnx_model best_model_v2_full.onnx --output_csv test_split_report_v2_onnx.csv
```

Checkpoint path evaluation:
```powershell
& ".\.venv\Scripts\python.exe" evaluate_test_split_report.py --csv_path labels.csv --data_root . --seed 42 --image_size 224 --checkpoint best_model_v2_full.pth --device cuda --output_csv test_split_report_v2_ckpt.csv
```

## PowerShell troubleshooting
- Use `&` before a quoted executable path.
- Use `.\.venv\...` from repository root (not `..venv\...`).
- If inference cannot find an image, use a path under `images/`, e.g. `images/cell0001.png`.

## Notes
- CSV format must include columns: `image_path`, `defect_probability`, `cell_type`.
- `defect_probability` is the regression target in [0, 1], and split stratification is done by defect-probability buckets.
- EL inputs are denoised with median blur and converted to 3-channel tensors for pretrained ResNet18 compatibility.
- Severity score is the model's direct continuous output in [0, 1].
- V2 training logs MSE, MAE, and R2 for train/val/test.
