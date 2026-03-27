What to do next for best final edge model accuracy
1. Train a real model run (not smoke)
- Use 30-80 epochs with early stopping on validation MSE.
- Keep seed fixed for comparability, then run 3 seeds and keep best checkpoint by validation MSE/MAE.

2. Tune the highest-impact hyperparameters
- Learning rate sweep: 1e-3, 3e-4, 1e-4, 3e-5
- Weight decay sweep: 1e-6 to 1e-4
- Batch size: 16, 32, 64 (depending on memory)

3. Improve augmentation carefully for EL images
- Keep current flips, add light intensity/contrast jitter only if it improves validation MAE.
- Avoid aggressive transforms that alter defect morphology.

4. Validate robustness before export
- Report MAE by cell_type (mono vs poly) from labels.csv.
- Check calibration of severity output against true defect_probability bins.

5. Lock export settings for edge reliability
- Use opset 18 explicitly for deterministic export behavior on your stack.
- Re-verify onnxruntime output parity with PyTorch on 50-100 random samples.

6. Optimize for edge deployment
- Quantize ONNX model (dynamic first; static if calibration data available).
- Benchmark latency + memory on target edge hardware with realistic batch size 1.
- Choose deployment threshold using validation curve for your operational false-alarm tolerance.

7. Production checklist
- Freeze final artifacts: best_model.pth, best_model.onnx, exact requirements versions.
- Save training config and seed with the model for reproducibility.

If you want, I can do the next pass now:
1. Run a short LR sweep automatically.
2. Add a validation-by-cell_type report script.
3. Update export to default opset 18 and re-run ONNX parity check.
