import pandas as pd
import numpy as np
import os

# ==========================================
# SCENARIO II CONFIGURATION
# ==========================================
INPUT_FILE = "ULTIMATE_MASTER_DATASET.csv"
WINDOW_SIZE = 100  # 10s of data
STEP_SIZE = 50     # 50% overlap 
FEATURES = ['RSSI', 'LQI', 'Diff_RSSI'] 

def main():
    print(f"📥 Loading '{INPUT_FILE}' for Scenario II (Node Classification)...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Differentiation
    if 'Diff_RSSI' not in df.columns:
        print("⚙️  Computing Differentiation...")
        df['Diff_RSSI'] = df.groupby(['Environment', 'Sender_Node'])['RSSI'].diff().fillna(0)
    
    print("⏳ Sorting data by time...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Environment', 'Sender_Node', 'Timestamp'])

    X_list, y_node_list = [], []

    # 2. Segmentation & ZERO-MEAN CENTERING
    print(f"🪟 Slicing into {STEP_SIZE}-step windows and applying Zero-Mean Centering...")
    grouped = df.groupby(['Environment', 'Sender_Node'])
    
    for name, group in grouped:
        feats = group[FEATURES].values
        node_label = group['Sender_Node'].iloc[0]
        
        for i in range(0, len(feats) - WINDOW_SIZE + 1, STEP_SIZE):
            window = feats[i : i + WINDOW_SIZE]
            
            # CRITICAL RF FINGERPRINTING STEP: Zero-Mean Centering
            # We subtract the mean to remove the absolute power level (the environment)
            # We DO NOT divide by variance, because the variance IS the hardware fingerprint
            window_mean = np.mean(window, axis=0)
            window_centered = window - window_mean
            
            X_list.append(window_centered)
            y_node_list.append(node_label)

    X = np.array(X_list)
    y_node = np.array(y_node_list)

    print(f"✅ Created {len(X)} Zero-Mean Centered windows for RF Fingerprinting.")
    
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/X_windows_scen2.npy", X)
    np.save("processed_data/y_node_labels_scen2.npy", y_node)
    print(f"💾 Saved to 'processed_data/X_windows_scen2.npy' and 'y_node_labels_scen2.npy'")

if __name__ == "__main__":
    main()
