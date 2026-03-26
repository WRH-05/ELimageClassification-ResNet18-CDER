MASSIVE UPDATE: I have found a vastly superior dataset for our Edge-AI pipeline. We are discarding the PVEL-AD XML bounding box approach. 

I am now using the standard `elpv-dataset`. I have a CSV file (e.g., `labels.csv`) with three columns:
1. `image_path` (e.g., `images/cell0001.png`)
2. `defect_probability` (Float: `0.0`, `0.3333`, `0.6666`, or `1.0`)
3. `cell_type` (`mono` or `poly`)

Please update the project plan and generate the code with this simplified architecture:

**1. Data Pipeline (`dataset.py`):**
- Use `pandas` to read the CSV. 
- The target variable (`y`) is the `defect_probability` column (as a float).
- Keep the previously agreed-upon preprocessing: 1-channel to 3-channel broadcast, Median Filter denoising for 15-second exposure thermal noise, and resize to 224x224. 
- The `cell_type` column can be ignored for training purposes, but ensure the `train/val/test` split is stratified based on the `defect_probability` so we get an even distribution of 0.0, 0.33, 0.66, and 1.0 in our validation set.

**2. Regression Model (`train.py`):**
- Use the pre-trained `ResNet18`.
- Modify the final fully-connected layer to output a single continuous value: `nn.Linear(num_ftrs, 1)`.
- Use `MSELoss` (Mean Squared Error).
- Apply a `Sigmoid` activation at the end of the forward pass to guarantee the model's prediction stays strictly between 0.0 and 1.0.

**3. Edge Inference (`inference_mqtt_mock.py`):**
- Keep the previous ONNX runtime requirement for the Raspberry Pi.
- The script should output the continuous score directly from the model as our 'Severity Score' for the MQTT JSON payload.

Please provide the updated `dataset.py` and `train.py` scripts based on this CSV-driven regression approach.
