I've decided on the final refinements for Version 2 of the model. Please implement the previous plan (Weighted Sampler + Huber Loss) but add a 'Professional Training Routine' to `train.py`.

**1. Data Loading (`dataset.py`):**
Implement the `WeightedRandomSampler` for the training set to force the model to see the underrepresented 0.33 and 0.66 classes as often as the 0.0/1.0 classes.

**2. Robust Loss (`train.py`):**
Switch to `SmoothL1Loss` (Huber Loss) to reduce the impact of extreme outliers on the regression gradient.

**3. Training Optimization (`train.py`):**
Please add the following standard research-grade components:
- **Learning Rate Scheduler:** Use `ReduceLROnPlateau`. It should monitor validation loss and drop the learning rate if the loss stalls for 3-5 epochs.
- **Early Stopping:** Implement a simple Early Stopping class. If the validation MAE (Mean Absolute Error) does not improve for 7-10 consecutive epochs, stop the training and save the best weights. 

**4. Metrics:**
In addition to MSE, please ensure the training script logs **Mean Absolute Error (MAE)** and **R-squared (R²)** for both train and validation sets. These will be the primary metrics for the research paper.

Please provide the updated `dataset.py` and `train.py` so I can begin the final training run.
And update `ELimageClassification_all_in_one.ipynb` accordingly.
