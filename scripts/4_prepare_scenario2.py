import pandas as pd
import numpy as np
import os

# ==========================================
# SCENARIO II CONFIGURATION
# ==========================================
INPUT_FILE = "ULTIMATE_MASTER_DATASET.csv"
WINDOW_SIZE = 1000
STEP_SIZE = 485
FEATURES = ['RSSI', 'LQI', 'Diff_RSSI', 'Diff_LQI']

def main():
    print(f"📥 Loading '{INPUT_FILE}' for Scenario II (Node Classification)...")
    df = pd.read_csv(INPUT_FILE)
    
# 1. Differentiation for BOTH RSSI and LQI
    print("⚙️  Computing Differentiation for RSSI and LQI...")
    if 'Diff_RSSI' not in df.columns:
        df['Diff_RSSI'] = df.groupby(['Environment', 'Sender_Node'])['RSSI'].diff().fillna(0)
        
    # THE FIX: Calculate Diff_LQI before trying to slice the features!
    if 'Diff_LQI' not in df.columns:
        df['Diff_LQI'] = df.groupby(['Environment', 'Sender_Node'])['LQI'].diff().fillna(0)
    
    print("⏳ Sorting data by time...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Environment', 'Sender_Node', 'Timestamp'])

    X_list, y_node_list, y_env_list = [], [], []

    # 2. Segmentation & ZERO-MEAN CENTERING
    print(f"🪟 Slicing into {STEP_SIZE}-step windows and applying Zero-Mean Centering...")
    grouped = df.groupby(['Environment', 'Sender_Node'])
    
    for name, group in grouped:
        feats = group[FEATURES].values
        node_label = group['Sender_Node'].iloc[0]
        env_label = group['Environment'].iloc[0]
        
        for i in range(0, len(feats) - WINDOW_SIZE + 1, STEP_SIZE):
            window = feats[i : i + WINDOW_SIZE]
            
            # CRITICAL RF FINGERPRINTING STEP: Zero-Mean Centering
            # We subtract the mean to remove the absolute power level (the environment)
            # We DO NOT divide by variance, because the variance IS the hardware fingerprint
            window_mean = np.mean(window, axis=0)
            window_centered = window - window_mean
            
            X_list.append(window_centered)
            y_node_list.append(node_label)
            y_env_list.append(env_label)

    X = np.array(X_list)
    y_node = np.array(y_node_list)
    y_env = np.array(y_env_list)

    print(f"✅ Created {len(X)} Zero-Mean Centered windows for RF Fingerprinting.")
    
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/X_windows_scen2.npy", X)
    np.save("processed_data/y_node_labels_scen2.npy", y_node)
    np.save("processed_data/y_env_labels_scen2.npy", y_env)
    print(f"💾 Saved to 'processed_data/X_windows_scen2.npy', 'y_node_labels_scen2.npy', and 'y_env_labels_scen2.npy'")

if __name__ == "__main__":
    main()