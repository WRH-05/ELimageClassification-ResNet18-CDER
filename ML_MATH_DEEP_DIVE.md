# EL Solar Defect Regression: Full ML + Math Deep Dive (Beginner-Friendly, Math-Rigorous)

This document explains the full project end-to-end so you can confidently answer technical questions about:
- what the model is doing,
- why each engineering choice exists,
- what the core math means,
- how deployment works,
- and where the most interesting technical tradeoffs are.

The tone here assumes:
- you are new to ML/AI tooling and workflow,
- but you are comfortable with advanced math concepts.

---

## 1) The Problem in One Sentence

Given an electroluminescence (EL) image of a solar cell, predict a **continuous severity score** in $[0,1]$ called `defect_probability`.

This is a **regression** task, not a binary classifier, because the label is a real number.

---

## 2) Big Picture Pipeline

The project has 4 phases:

1. **Data loading and split** (`dataset.py`)
2. **Training** (`train.py`)
3. **Export to ONNX** (`export_model.py`)
4. **Edge inference + payload generation** (`inference_mqtt_mock.py`)

Plus evaluation tooling:
- **Per-image held-out report** (`evaluate_test_split_report.py`)

### Data flow

1. Read `labels.csv` with required columns:
   - `image_path`
   - `defect_probability`
   - `cell_type`
2. Validate target range: `defect_probability in [0,1]`
3. Stratified split by severity buckets (not random raw split)
4. Apply image preprocessing + augmentations
5. Train ResNet18 transfer-learning regressor
6. Select best checkpoint using safety-focused criteria (critical recall + precision floor)
7. Export to ONNX
8. Run CPU ONNX inference on device and produce JSON (optional MQTT publish)

---

## 3) Data and Split Strategy (Why It Matters)

### 3.1 CSV schema and constraints

The code enforces:
- files must exist,
- labels must be numeric and in $[0,1]$,
- no empty dataset allowed.

This is crucial because most model failures in practice come from data integrity issues, not architecture.

### 3.2 Stratified regression split via buckets

Because this is regression, there is no direct class label for stratification. The code creates 4 buckets:
- bucket 0: $[0.0, 0.1667)$
- bucket 1: $[0.1667, 0.5)$
- bucket 2: $[0.5, 0.8334)$
- bucket 3: $[0.8334, 1.0]$

Then each bucket is split 70/15/15 into train/val/test.

#### Why this is interesting

A naive random split can produce different severity distributions across train/val/test. If the test set happens to contain more high-severity examples than train, metrics can look artificially bad (or vice versa).

Bucketed stratification makes empirical risk comparisons more stable by approximately preserving the target distribution.

---

## 4) Preprocessing and Feature Space Alignment

### 4.1 Input modality adaptation

EL images are read as grayscale, denoised with median blur, then converted to pseudo-RGB by channel stacking.

Reason: pretrained ResNet18 expects 3-channel input and learned low-level kernels on ImageNet statistics.

### 4.2 Normalization

After scaling pixels to $[0,1]$, each channel is normalized by ImageNet mean/std:

$$
\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c}
$$

with
- $\mu = [0.485, 0.456, 0.406]$
- $\sigma = [0.229, 0.224, 0.225]$

This keeps feature magnitudes in the regime expected by pretrained backbone filters.

### 4.3 Augmentation (train only)

Train transform includes random horizontal + vertical flips. This injects invariance and reduces overfitting.

Evaluation (val/test) removes randomness to preserve metric determinism.

---

## 5) Model Architecture and Transfer Learning

### 5.1 Backbone

`train.py` uses `torchvision.models.resnet18(weights=DEFAULT)`.

### 5.2 Regression head

Final fully connected layer is replaced with:
- `Linear(in_features, 1)`
- `Sigmoid()`

So prediction is

$$
\hat{y} = \sigma(w^T h + b) \in (0,1)
$$

where $h$ is the final backbone feature vector.

#### Why sigmoid is a strong choice here

Since labels are in $[0,1]$, sigmoid enforces codomain consistency. You avoid impossible outputs (e.g. 1.3 or -0.2) at inference.

Potential downside: saturation near 0 or 1 can reduce gradient magnitude.

### 5.3 Warmup and staged unfreezing

Training starts in head-only mode for `warmup_epochs` (default 2), then unfreezes the full network.

Interpretation:
- Early epochs optimize low-dimensional linear head first,
- then full fine-tuning adapts deeper representation once head is calibrated.

This often improves stability vs immediate full-end-to-end updates.

---

## 6) Objective Functions (Losses) and Their Math

The code supports four losses:
- `mse`
- `smoothl1` (Huber-style)
- `weighted_l1`
- `weighted_mse`

### 6.1 MSE

$$
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

Pros: smooth, strongly penalizes large errors.
Cons: sensitive to outliers.

### 6.2 Smooth L1 (Huber-like)

For error $e = \hat{y}-y$ and threshold $\beta$:

$$
\ell(e)=
\begin{cases}
\frac{1}{2\beta}e^2 & |e|<\beta \\
|e| - \frac{\beta}{2} & |e|\ge \beta
\end{cases}
$$

Behavior:
- Quadratic near zero (fine-grained fitting)
- Linear for large residuals (robustness)

This is why it is often a practical default.

### 6.3 Weighted losses for risk-sensitive learning

The weighted variants assign higher penalty when target severity is above threshold (default 0.66).

Generic form:

$$
\mathcal{L} = \frac{1}{N}\sum_i w_i \cdot \phi(\hat{y}_i-y_i)
$$

with

$$
w_i =
\begin{cases}
\text{critical\_weight} & y_i \ge \tau \\
1 & y_i < \tau
\end{cases}
$$

where $\tau$ is `loss_weight_threshold`.

This biases training toward high-severity accuracy, which is useful for safety-critical screening.

---

## 7) Sampling Strategy and Distribution Rebalancing

Train loader uses `WeightedRandomSampler` by bucket inverse frequency:

$$
P(\text{sample } i) \propto \frac{1}{n_{bucket(i)}}
$$

where $n_{bucket(i)}$ is count of that sample's bucket.

This approximates balanced exposure across severity regions and combats regression target imbalance.

### Why this is subtle and important

In imbalanced regression, model can minimize global MSE by over-focusing on common mild defects. Weighted sampling pushes gradient updates to underrepresented severe ranges.

---

## 8) Optimization, Scheduling, and Early Stopping

### 8.1 Optimizer

Adam with weight decay:
- learning rate default $10^{-4}$
- weight decay default $10^{-5}$

Adam update (conceptually):

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

where $\hat{m}_t, \hat{v}_t$ are bias-corrected first/second moments.

### 8.2 ReduceLROnPlateau

If validation loss stalls, learning rate is multiplied by factor (default 0.5) after patience epochs.

This helps transition from coarse to fine optimization.

### 8.3 Early stopping

Stops training if validation MAE does not improve by at least `min_delta` for `patience` epochs.

Interpretation: a regularization mechanism against overfitting and wasted epochs.

---

## 9) Metrics: Regression + Operational Safety Metrics

The code tracks:
- MSE
- MAE
- $R^2$

with

$$
R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}
$$

### 9.1 Critical recall

Define critical ground truth by `target_threshold` (default 0.8). Define positive prediction by `prediction_threshold` (default 0.6).

Then:

$$
\text{Critical Recall} = \frac{TP}{TP+FN}
$$

on the subset $y_i \ge 0.8$.

### 9.2 Precision at threshold

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

with positives defined by predicted score above threshold.

### 9.3 Checkpoint save rule (very important)

A checkpoint is only saved when:
1. precision >= precision_floor (default 0.70), and
2. critical recall improves, or ties but MAE improves.

This is a multi-objective operational policy, not just raw loss minimization.

#### Why this is interesting

It hard-codes deployment priorities into model selection:
- avoid too many false alarms (precision floor),
- maximize capture of critical cases (critical recall),
- prefer lower overall absolute error as tie-break.

This is closer to real-world decision policy than textbook "pick lowest val loss".

---

## 10) Export to ONNX: What Changes and What Stays

`export_model.py`:
- rebuilds architecture,
- loads `model_state_dict`,
- exports with dynamic batch axis,
- validates graph with `onnx.checker.check_model`.

Output node is explicitly named `severity_score`.

### Math invariance point

ONNX export should preserve function $f_\theta(x)$ up to numerical tolerance:

$$
f^{PyTorch}_\theta(x) \approx f^{ONNX}_\theta(x)
$$

Differences can appear from backend kernels, precision, and resize semantics, so smoke validation is critical.

---

## 11) Inference and Decision Layer

### 11.1 ONNX Runtime inference

`inference_mqtt_mock.py` preprocesses image then runs one forward pass:

$$
\hat{y} = f_{ONNX}(x)
$$

### 11.2 Operational decision threshold

Payload status:
- `CRITICAL` if `severity_score > critical_threshold`
- else `OK`

This converts continuous regression into actionable binary status.

### 11.3 Why strict "greater than" matters

Code uses `>` not `>=` for critical status. Borderline equal-to-threshold values are treated as `OK`.

This tiny choice can matter for policy audits.

---

## 12) Held-Out Test Report Script: Why It Is Valuable

`evaluate_test_split_report.py` reproduces the same stratified test split, then writes per-image rows:
- target
- prediction
- absolute error
- squared error

This enables:
- failure analysis by sample,
- calibration checks in high-severity regime,
- ranking worst examples for data curation.

A single aggregate MAE can hide safety-relevant tail failures; per-image reports expose them.

---

## 13) Most Interesting Technical Bits in This Repository

1. **Regression + thresholded ops metrics duality**
   - Core model is continuous regression.
   - Deployment and model selection use thresholded precision/recall constraints.
   - This bridges statistical estimation and operational decision theory.

2. **Risk-aware checkpointing policy**
   - Precision floor + critical recall target is a policy-level objective embedded in training loop.

3. **Imbalanced regression handling**
   - Uses both bucketed stratification and weighted sampling.
   - Optional weighted losses provide another axis of imbalance control.

4. **Transfer learning adaptation from natural images to EL grayscale domain**
   - Pseudo-RGB conversion + ImageNet normalization + staged fine-tune is a practical domain-shift strategy.

5. **Deployment realism**
   - ONNX export, CPU inference, and MQTT payload mirror edge deployment constraints.

---

## 14) Common Questions You May Be Asked (with strong answers)

### Q1. Why not do binary classification directly?
A: Regression retains richer information and supports threshold tuning post hoc. A single trained regressor can support different operating points (e.g., sensitivity-focused vs precision-focused) without retraining.

### Q2. Why sigmoid on output?
A: Labels are bounded in $[0,1]$, so sigmoid enforces physically meaningful prediction range and stabilizes downstream threshold logic.

### Q3. Why stratify by buckets for regression?
A: Random split can distort severity distribution across sets. Bucket stratification approximates distribution matching and yields fairer validation/test signals.

### Q4. Why use weighted sampler if loss already handles errors?
A: Sampling changes which examples are seen how often; loss weighting changes penalty per seen sample. They affect optimization differently and can be complementary.

### Q5. What does MAE vs MSE mean practically?
A: MAE is linear penalty and robust to outliers; MSE amplifies large errors and is more sensitive to severe misses.

### Q6. Why use smooth L1 (Huber-like)?
A: It combines local quadratic behavior (good near optimum) with linear tails (outlier robustness), often improving stability on noisy labels.

### Q7. Why use a precision floor when saving checkpoints?
A: High recall with very low precision can flood operations with false alarms. Precision floor enforces a minimum usability constraint.

### Q8. Could $R^2$ be negative?
A: Yes. Negative $R^2$ means model performs worse than predicting the mean target.

### Q9. Why export ONNX?
A: ONNX provides framework-agnostic graph execution and efficient CPU inference via ONNX Runtime, suitable for edge devices.

### Q10. What are the major error sources in this pipeline?
A: Label noise, domain shift, preprocessing mismatch between train and inference, class/severity imbalance, and threshold miscalibration.

---

## 15) Mathematical Framing (for advanced discussion)

You can present the project as minimizing expected risk under shifted operational cost:

$$
\min_\theta \; \mathbb{E}_{(x,y)\sim \mathcal{D}_{train}}[\ell(f_\theta(x), y)]
$$

but deployment objective is closer to constrained optimization:

$$
\max_\theta \; \text{Recall}_{critical}(\theta)
\quad \text{s.t.} \quad \text{Precision}(\theta) \ge p_{min}
$$

where $p_{min}$ is `precision_floor`.

The code approximates this by model-selection heuristics rather than direct constrained optimization.

---

## 16) Practical Pitfalls and How To Explain Them

1. **Target leakage risk**
   - If split done after augmentation or with duplicate near-identical samples across splits, metrics inflate.

2. **Transform mismatch risk**
   - Training/eval and inference must match preprocessing semantics (resize interpolation, normalization constants, channel order).

3. **Threshold transport risk**
   - A threshold tuned on one distribution may drift under domain shift.

4. **Sigmoid saturation**
   - If logits become extreme, gradients shrink; warmup/unfreeze and LR scheduling help mitigate unstable dynamics.

---

## 17) How To Defend Design Choices in an Interview

If asked "Why this design?", use this structure:

1. **Constraint**: edge deployment + bounded severity target + imbalance.
2. **Modeling decision**: transfer-learned ResNet18 regressor with sigmoid output.
3. **Optimization decision**: weighted sampler, robust loss, staged unfreezing, scheduler, early stopping.
4. **Operational decision**: checkpoint policy tied to critical recall and precision floor.
5. **Deployment decision**: ONNX Runtime CPU path + payload API.

This answer format shows engineering reasoning, not just implementation detail.

---

## 18) Short Technical Summary You Can Memorize

"This project is a risk-aware regression pipeline for EL solar defect severity. It uses a transfer-learned ResNet18 with sigmoid-bounded output to predict defect probability in [0,1]. Data is stratified by severity buckets and rebalanced with weighted sampling. Training tracks regression quality (MSE/MAE/R2) plus operational metrics (critical recall, precision at threshold), and checkpointing enforces a precision floor to keep alerts actionable. The model is exported to ONNX for CPU edge inference, where the continuous score is thresholded into CRITICAL/OK status and optionally published over MQTT." 

---

## 19) Suggested Next Learning Steps (Optional)

1. Add calibration analysis (reliability plots / isotonic or Platt-style post-calibration for bounded regression outputs).
2. Compare uncertainty-aware alternatives (e.g., quantile regression or evidential regression).
3. Run ablations:
   - weighted sampler on/off,
   - smoothL1 vs weighted_mse,
   - warmup epochs 0 vs 2.
4. Add confidence intervals on key metrics via bootstrap on held-out test split.

These make your discussion even stronger in technical reviews.
