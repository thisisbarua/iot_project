# IoT Environment Classification Project

This repository contains the full pipeline for an IoT-based environment classification system. It leverages nRF52840 nodes running RIOT OS to collect wireless signal data (RSSI, LQI), Python scripts to log this data over serial, and Deep Learning models (CNN, ResNet) to classify the surrounding environment based on the radio wave characteristics.

## 📁 Repository Structure

### 1. Embedded Firmware (RIOT OS)
The firmware runs on hardware nodes (e.g., nRF52840 dongles or dev boards) flashed with RIOT OS.
- **`iot_sender/`**: Contains the C firmware (`main.c`) for the Sender node. It broadcasts UDP packets every 100ms and forces the radio to maximum TX power (+8 dBm) for better range.
- **`iot_receiver/`**: Contains the C firmware (`main.c`) for the Receiver node (the Sniffer). It configures the radio into promiscuous mode to capture raw packets over the air.

### 2. Data Collection & Logging (Python)
These scripts run on the laptop connected to the "Receiver" node (Sniffer) via USB. They parse the serial output of the Sniffer.
- **`logger.py`**: The main data collection script. Prompts for the receiver identity (A, B, or C), listens on `/dev/ttyACM0` for 15 minutes, calculates differential RSSI, and saves the packets to a CSV file and a raw text log.
- **`second_laptop_logger.py`**: Identical to `logger.py`, but tailored for a specific laptop, listening on `/dev/ttyACM1`.
- **`calibration_logger.py`**: A specialized tool to help deploy the nodes. It verifies range and connectivity by providing real-time console feedback (e.g., "Out of Range", "Healthy Signal") as you walk the Sender away from the Receiver.

### 3. Machine Learning & Processing (Python)
Once data is collected from different environments, it is processed and fed into neural networks.
- **`scripts/1_prepare_time_series.py`**: Takes the `ULTIMATE_MASTER_DATASET.csv`, computes differential RSSI, normalizes all features via MinMax scaling, and slices the data into overlapping 100-sample time windows (approx. 10 seconds of data). Outputs `.npy` array files into a `processed_data/` folder.
- **`scripts/2_scenario1_seen_data.py`**: **Scenario I, Strategy 1 (Seen Data)** — Loads the preprocessed windows and trains two deep learning models (1D CNN and Deep Multi-Block ResNet) to classify **environments**. Uses a random 75/25 train/test split, so the models see data from all nodes and environments during training.
- **`scripts/3_scenario1_unseen_data.py`**: **Scenario I, Strategy 2 (Unseen Data)** — Same environment classification task, but splits by **sensor node**: trains on Node_A + Node_B data, tests on Node_C data (completely unseen). Tests whether environment signatures generalize across different hardware.

### 4. Datasets
- **`five_env_data/`**: Contains the master CSVs recorded for five distinct environments: *Bridge*, *Forest*, *Lake*, *Open Field*, and *River*.
- **`full_merged_data/`**: Contains `ULTIMATE_MASTER_DATASET.csv`, which merges the recordings from the environments to serve as the master input for the ML pipeline.

---

## 🚀 How to Run the Project

### Phase 1: Flashing the Nodes
You must have the [RIOT OS toolchain](https://github.com/RIOT-OS/RIOT) set up.
1. Navigate to `iot_sender/` and run `make all flash term` to flash the Sender node.
2. Navigate to `iot_receiver/` and flash the Receiver node (Sniffer).

### Phase 2: Calibration & Placement
1. Connect the Receiver node to your laptop via USB.
2. Run the calibration script:
   ```bash
   python3 calibration_logger.py
   ```
3. Follow the console prompts. Take the Sender node and walk away until the script warns you that the signal is dropping. Place the Sender node at a safe optimal distance.

### Phase 3: Data Collection
1. Once nodes are placed in the target environment (e.g., Forest), run the main logger:
   ```bash
   python3 logger.py
   ```
   *(If using the second laptop or if the port is ACM1, use `second_laptop_logger.py`)*
2. The script will record data for exactly 15 minutes and save the output CSVs in the root folder.
3. Repeat this process for all environments and merge the CSVs into an `ULTIMATE_MASTER_DATASET.csv` inside `full_merged_data/`.

### Phase 4: Machine Learning Pipeline

#### Step 1: Prepare the Time-Series Data
The `1_prepare_time_series.py` script takes the merged dataset and processes it into ML-ready format.

**What it does:**
- Reads `ULTIMATE_MASTER_DATASET.csv`
- Computes differential RSSI (`Diff_RSSI = RSSI[t] - RSSI[t-1]`)
- Normalizes all features (RSSI, LQI, Diff_RSSI, Diff_LQI) via MinMax scaling to [0, 1]
- Segments the continuous data into sliding windows of **100 samples** with **50% overlap** (STEP_SIZE=50)
- Each window captures approximately **10 seconds** of radio signal data

**How to run:**
```bash
cd scripts
python3 1_prepare_time_series.py
```

**Output:** Creates a `processed_data/` directory with:
- `X_windows.npy` — Shape: `(4279, 100, 4)` — 4279 windows, each 100 timesteps × 4 features
- `y_env_labels.npy` — Environment label per window (bridge, forest, lake, open_field, river)
- `y_node_labels.npy` — Sensor node label per window (Node_A, Node_B, Node_C)

---

#### Step 2: Scenario I — Environment Classification (Seen Data)
The `2_scenario1_seen_data.py` script classifies **which environment** the signal was recorded in.

**Data Split:** Random 75/25 train/test split across all nodes and environments (Strategy 1: "Seen Data"). The model sees examples from every environment during training.

**Models:**
- **1D CNN** — 3-layer convolutional network (128→256→512 filters) with kernel sizes 11, 7, 3. Uses BatchNormalization, ReLU, MaxPooling, GlobalAveragePooling, and a Dense(512) + Dropout(0.5) head.
- **Deep Multi-Block ResNet** — 2 residual blocks with skip connections. Block 1: 128 filters (kernels 5, 3). Block 2: 256 filters with stride-2 downsampling. GlobalAveragePooling + Dense(256) + Dropout(0.5) head.

**Hyperparameters:** 40 epochs, batch size 64, Adam optimizer, ReduceLROnPlateau (patience=3, factor=0.5).

**How to run:**
```bash
python3 2_scenario1_seen_data.py
```

**Expected Output:** ~83-84% accuracy for both CNN and ResNet.

---

#### Step 3: Scenario I — Environment Classification (Unseen Data)
The `3_scenario1_unseen_data.py` script tests whether environment patterns generalize **across different sensor nodes**.

**Data Split:** Split by sensor node (Strategy 2: "Unseen Data"). Trains on Node_A + Node_B, tests on Node_C. The training data is shuffled before Keras `validation_split` to ensure the validation set contains a mix of all environments.

**Models:** Identical architectures to the seen-data script.

**Hyperparameters:** 60 epochs, batch size 64, Adam optimizer, ReduceLROnPlateau (patience=7, factor=0.5). Higher patience and more epochs are used because the unseen-node task requires more training time to converge.

**How to run:**
```bash
python3 3_scenario1_unseen_data.py
```

---

## 🛠️ Requirements (Python)
Ensure you have the required libraries installed:
```bash
pip install pyserial pandas numpy scikit-learn tensorflow
```
