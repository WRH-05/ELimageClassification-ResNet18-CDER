# Variance Study Results

Multi-seed study across 3 seeds (42, 123, 2026).
Metrics shown as Mean ± Standard Deviation.

## F1-Score @ threshold 0.65

| Model | F1-Score |
|-------|----------|
| V1 (MSE) | $0.756 \pm 0.013$ |
| V3.1 (SAHL 2.5x) | $0.758 \pm 0.020$ |

## Precision @ threshold 0.65

| Model | Precision |
|-------|-----------|
| V1 (MSE) | $0.856 \pm 0.021$ |
| V3.1 (SAHL 2.5x) | $0.757 \pm 0.018$ |

## Recall @ threshold 0.65

| Model | Recall |
|-------|--------|
| V1 (MSE) | $0.677 \pm 0.032$ |
| V3.1 (SAHL 2.5x) | $0.760 \pm 0.049$ |

## Mean Absolute Error (MAE)

| Model | MAE |
|-------|-----|
| V1 (MSE) | $0.1959 \pm 0.0037$ |
| V3.1 (SAHL 2.5x) | $0.1898 \pm 0.0122$ |

## Critical Recall @ (pred threshold 0.6, target threshold 0.8)

| Model | Critical Recall |
|-------|-----------------|
| V1 (MSE) | $0.784 \pm 0.048$ |
| V3.1 (SAHL 2.5x) | $0.846 \pm 0.028$ |
