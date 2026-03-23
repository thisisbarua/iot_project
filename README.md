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
- **`scripts/1_prepare_time_series.py`**: Takes the `ULTIMATE_MASTER_DATASET.csv`, computes differential RSSI, normalizes all features via MinMax scaling, and slices the data into overlapping time windows with 50% overlap. Outputs `.npy` array files into a `processed_data/` folder.
- **`scripts/2_scenario1_seen_data.py`**: **Scenario I, Strategy 1 (Seen Data)** — Loads the preprocessed windows and trains two deep learning models (1D CNN and Deep Multi-Block ResNet) to classify **environments**. Uses a random 75/25 train/test split, so the models see data from all nodes and environments during training.
- **`scripts/3_scenario1_unseen_data.py`**: **Scenario I, Strategy 2 (Unseen Data)** — Same environment classification task, but splits by **sensor node**: trains on Node_A + Node_B data, tests on Node_C data (completely unseen). Tests whether environment signatures generalize across different hardware.
- **`scripts/4_prepare_scenario2.py`**: Alternative data preparation for **Scenario II**. Computes both `Diff_RSSI` and `Diff_LQI`, then uses **Zero-Mean Centering** per 1000-step window instead of global MinMaxScaler. Uses all 4 features (RSSI, LQI, Diff_RSSI, Diff_LQI) with a step size of 485 to capture longer-duration RF hardware fingerprints.
- **`scripts/5_scenario2_seen_data.py`**: **Scenario II (Node Classification)** — Uses the Zero-Mean centered data to classify which specific hardware node generated the signal. Implements a Heavy-Duty 1D CNN (128→256→512 filters) and a Deep Multi-Block ResNet with high dropout (0.7) for robust node identification.

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
- Segments the continuous data into sliding windows of **100 samples**
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

> **Why MinMax Scaling here?** For environment classification (Scenario I), absolute signal strength is a powerful discriminator — a Bridge and a Forest produce very different RSSI levels. MinMax scaling preserves these absolute levels while normalizing them to [0, 1]. This is the **opposite** of what Scenario II needs (see Step 4), which is why two separate preprocessing scripts exist.

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

#### Step 4: Prepare Data for Scenario II (Node Classification)
The `4_prepare_scenario2.py` script radically changes the preprocessing strategy since identifying identical hardware nodes is much harder than identifying environments.

**What it does:**
- Computes both differential features: `Diff_RSSI` and `Diff_LQI` (change between consecutive samples).
- Sliding window segmentation with **1000 samples** per window and a step size of 485.
- Uses **4 features**: RSSI, LQI, Diff_RSSI, Diff_LQI — the differential features capture rapid hardware-specific jitter patterns.
- **Core Difference:** Replaces Global MinMax Scaling with **Per-Window Zero-Mean Centering**. It subtracts the mean of *each individual 1000-step window* from itself. 
- *Why?* Global MinMax scaling (used in Step 1) preserves absolute signal levels — great for telling environments apart, but **harmful** for node classification. The same node transmitting in a Bridge (RSSI ≈ -60) vs a Forest (RSSI ≈ -85) would look like two completely different nodes. Zero-Mean Centering destroys the absolute strength, forcing the network to look purely at the localized, high-frequency "shape" and variance of the signal — which contains the unique RF hardware fingerprint of each transmitter chip.

> **Why two preprocessing scripts?** Scenario I needs the absolute signal level (environment = different RSSI). Scenario II needs it removed (same node = same fingerprint regardless of environment). These are fundamentally opposite requirements, hence `1_prepare_time_series.py` (MinMax) and `4_prepare_scenario2.py` (Zero-Mean Centering).

**How to run:**
```bash
python3 4_prepare_scenario2.py
```

---

#### Step 5: Scenario II — Sensor Node Classification (Seen Environment)
The `5_scenario2_seen_data.py` script attempts to identify whether the signal came from Node_A, Node_B, or Node_C, regardless of the environment.

**Challenge:** The nodes are the exact same hardware model (nRF52840), making their RF fingerprints incredibly similar.

**Models:**
- **Heavy-Duty 1D CNN:** A 3-layer convolutional network (128→256→512 filters) with kernel sizes 11, 7, 3. Uses BatchNormalization, ReLU, MaxPooling, GlobalAveragePooling, and a Dense(512) + Dropout(0.7) head.
- **Deep Multi-Block ResNet:** 2 residual blocks with skip connections. Block 1: 128 filters (kernels 5, 3). Block 2: 256 filters with stride-2 downsampling. GlobalAveragePooling + Dense(256) + Dropout(0.7) head.

**Hyperparameters:** 100 epochs, batch size 32, Adam optimizer, ReduceLROnPlateau (patience=7, factor=0.5).

**How to run:**
```bash
python3 5_scenario2_seen_data.py
```

---

#### Step 6: Scenario II — Sensor Node Classification (Unseen Environment)
The `6_scenario2_unseen_data.py` script is the **ultimate test** of whether the RF hardware fingerprint is truly environment-invariant.

**Data Split:** Instead of a random split, the data is split by **environment**. The models train on data from 4 environments (bridge, forest, open_field, river) and are tested on the unseen `lake` environment.

**Environment-Invariant Transforms (applied at runtime, file 4 untouched):**
- **Per-Window Z-Score Standardization:** Divides by per-window standard deviation on top of the already zero-mean centered data. This removes environment-dependent signal variance (e.g., forest = high multipath fading, open_field = low variance).
- **Differential-Only Features:** Slices to keep only `Diff_RSSI` and `Diff_LQI` (columns 2 & 3). Raw RSSI/LQI still carry environmental bias even after centering — differentials capture rapid hardware-specific micro-jitter instead.

**Models:** Same architectures as Step 5 (Heavy-Duty CNN + Deep Multi-Block ResNet).

**Hyperparameters:** 80 epochs, batch size 32, Adam optimizer, ReduceLROnPlateau (patience=7, factor=0.5).

> **Important:** You must re-run `4_prepare_scenario2.py` first, as it saves `y_env_labels_scen2.npy` needed for the environment-based split.

**How to run:**
```bash
python3 4_prepare_scenario2.py   # Re-run to generate env labels
python3 6_scenario2_unseen_data.py
```

---

## 🛠️ Requirements (Python)
Ensure you have the required libraries installed:
```bash
pip install pyserial pandas numpy scikit-learn tensorflow
```
