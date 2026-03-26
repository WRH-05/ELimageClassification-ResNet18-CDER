# ELimageClassification

Edge-AI pipeline for nighttime EL solar defect severity regression using transfer learning (ResNet18), with ONNX Runtime inference for Raspberry Pi-class devices.

## Project Files
- `dataset.py` - CSV data loading, denoising, transforms, and DataLoaders
- `train.py` - Transfer-learning regression training loop and checkpointing
- `export_model.py` - Export `best_model.pth` to ONNX
- `inference_mqtt_mock.py` - ONNX Runtime inference + JSON/MQTT mock payload

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Train model
```bash
python train.py --csv_path labels.csv --data_root . --epochs 20 --batch_size 32 --learning_rate 1e-4
```

## 3) Export ONNX model
```bash
python export_model.py --checkpoint best_model.pth --onnx_output best_model.onnx
```

## 4) Run edge inference
```bash
python inference_mqtt_mock.py --onnx_model best_model.onnx --image_path path/to/el_image.png --pad_id simulated_pad_01 --critical_threshold 0.8
```

## Notes
- CSV format must include columns: `image_path`, `defect_probability`, `cell_type`.
- `defect_probability` is the regression target in [0, 1], and split stratification is done by defect-probability buckets.
- EL inputs are denoised with median blur and converted to 3-channel tensors for pretrained ResNet18 compatibility.
- Severity score is the model's direct continuous output in [0, 1].
