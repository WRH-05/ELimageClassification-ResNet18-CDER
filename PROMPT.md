We are preparing our ML training and analysis codebase for a public, academic repository submission IEEE standards. 

Inspect the current codebase, then refactor, clean, and organize the code into the exact structure defined below. Do not just print the code—create and write the refactored files directly to the filesystem.

---

### Directory & File Requirements

#### 1. Directory: `src/training/` (Core ML Pipeline)
- `dataset.py`: Extract all dataset loading logic here. Include:
  - ZAE Bayern CSV reading
  - 5x5 Median Filtering
  - Pseudo-RGB conversion
  - ImageNet normalization
  - Bucketed stratified splitting logic
- `losses.py`: Extract the `SafetyAwareAsymmetricHuberLoss` (SAHL) class into this standalone, well-documented file so researchers can easily import it independently.
- `train.py`: Main orchestration script. Must import from `dataset.py` and `losses.py`, implement the warmup-to-unfreeze training schedule, and save `.pth` checkpoints.
- `export_onnx.py`: Convert trained `.pth` models to `.onnx` (Opset 18 format).

#### 2. Directory: `src/analysis/` (Rigor & Evaluation)
- `multi_seed_ablation.py`: Run the 3-seed, 4-weight ablation study and output Markdown summary tables to standard output or a designated results directory.
- `generate_plots.py`: Use `matplotlib` and `scikit-learn` to compute and save the Precision-Recall (PR) Curve, F1-Threshold Curve, Confusion Matrices, and Error Residuals plots.

#### 3. Mini-READMEs
- Create `src/training/README.md` and `src/analysis/README.md`.
- Each README must include a clean, 3-step command-line guide explaining how to run the scripts.

---

### Mandatory Engineering Constraints

1. **NO HARDCODED PATHS**: Replace all absolute paths (e.g., `C:/...` or `/home/...`) with `argparse` CLI arguments. Supply sensible relative defaults (e.g., `--data_path ../../data/labels.csv`, `--model_dir ../../models/`).
2. **Clean Imports & Modular Design**: Ensure all local imports work modularly (e.g., `from dataset import ...` or relative package imports). Include Python type hints and Google/NumPy-style docstrings for all functions and classes.
3. **PEP 8 Compliance**: Code must be clean, readable, and properly formatted.

---

### Execution & Verification Workflow

1. **Scan**: Search existing repository files to locate all dataset, training, SAHL loss, export, and analysis code.
2. **Refactor**: Create the `src/training/` and `src/analysis/` directories and write the clean scripts and READMEs.
3. **Verify**: Run `python -m py_compile` on all newly created `.py` files and test their CLI interface (`python script.py --help`) to guarantee there are no broken imports or syntax errors.
4. make sure the code and math matches the claims in `main.pdf` meaning the latest tests and code is used.