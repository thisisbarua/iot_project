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
- **`anqur_laptop_logger.py`**: Identical to `logger.py`, but tailored for a specific laptop, listening on `/dev/ttyACM1`.
- **`calibration_logger.py`**: A specialized tool to help deploy the nodes. It verifies range and connectivity by providing real-time console feedback (e.g., "Out of Range", "Healthy Signal") as you walk the Sender away from the Receiver.

### 3. Machine Learning & Processing (Python)
Once data is collected from different environments, it is processed and fed into neural networks.
- **`scripts/1_prepare_time_series.py`**: Takes the `ULTIMATE_MASTER_DATASET.csv`, computes differential RSSI, normalizes all features via MinMax scaling, and slices the data into overlapping 100-sample time windows (approx. 10 seconds of data). Outputs `.npy` array files into a `processed_data/` folder.
- **`scripts/2_scenario1_seen_data.py`**: Loads the preprocessed windows and trains two deep learning models—a 1D CNN and a Deep Multi-Block ResNet—to classify the environment based on the radio signatures.

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
   *(If using Anqur's laptop or if the port is ACM1, use `anqur_laptop_logger.py`)*
2. The script will record data for exactly 15 minutes and save the output CSVs in the root folder.
3. Repeat this process for all environments and merge the CSVs into an `ULTIMATE_MASTER_DATASET.csv` inside `full_merged_data/`.

### Phase 4: Machine Learning Pipeline

1. **Prepare the Time-Series Data**  
   The `1_prepare_time_series.py` script takes the merged dataset, calculates differential RSSI, normalizes features via MinMax scaling, and segments the data into 100-sample sliding windows (with 50% overlap).
   * Ensure `ULTIMATE_MASTER_DATASET.csv` is placed inside the `scripts/` directory, or update the `INPUT_FILE` path in the script directly.
   * Run the data preparation script:
     ```bash
     cd scripts
     python3 1_prepare_time_series.py
     ```
   * **Output**: This creates a `processed_data/` directory containing three numpy array files: `X_windows.npy`, `y_env_labels.npy`, and `y_node_labels.npy`.

2. **Train and Evaluate the ML Models**  
   The `2_scenario1_seen_data.py` script loads the prepared time windows, encodes the labels, and splits the data (75% training, 25% test). It builds and trains two Deep Learning models (a 1D CNN and a Deep Multi-Block ResNet) for 40 epochs.
   * Run the training script:
     ```bash
     python3 2_scenario1_seen_data.py
     ```
   * **Output**: The script prints the training progress and learning curves, and finally outputs the classification accuracy percentages for both models on the test split.

---

## 🛠️ Requirements (Python)
Ensure you have the required libraries installed:
```bash
pip install pyserial pandas numpy scikit-learn tensorflow
```
