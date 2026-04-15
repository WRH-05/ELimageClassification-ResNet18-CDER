**We are preparing our final data for an IEEE journal submission and need to add statistical rigor to our methodology to pass peer review. Please write and execute scripts to generate the following two experimental data sets without breaking our existing architecture:**

**1. Statistical Variance Testing (Multi-Seed Run):**
- Please write a wrapper script (or modify `train.py`) to train the V1 model (MSE Baseline) and the V3.1 model (SAHL with 2.5x weight) across **3 different random seeds** (e.g., 42, 123, 2026).
- Evaluate each run on the held-out test set.
- Aggregate the results for Precision, Recall, F1-Score (at threshold 0.65), and MAE.
- Output a Markdown table (e.g., `variance_metrics.md`) reporting the results in `Mean ± Standard Deviation` format (e.g., $0.808 \pm 0.012$). 

**2. SAHL Weight Ablation Study:**
- To mathematically justify our hyperparameter choice, please run the SAHL training pipeline using a single fixed seed for the following asymmetric weights: **1.0x, 1.5x, 2.5x, and 5.0x**.
- Generate a summary CSV/Markdown table comparing F1-score, Critical Recall, and Precision across these four weights.
- Please write a script to plot this data (`ablation_study.pdf`), showing the Asymmetric Weight on the X-axis and the resulting metrics (F1/Recall/Precision) on the Y-axis. 

**3. Export Best Model:**
- After the multi-seed runs, please export the best performing 2.5x SAHL model to ONNX (e.g., `best_sahl_2.5x_final.onnx`) so I can transition it to the edge deployment codebase.

Please provide the code to execute these loops and generate the final tables/plots.