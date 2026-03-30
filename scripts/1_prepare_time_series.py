import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "ULTIMATE_MASTER_DATASET.csv"
WINDOW_SIZE = 220
STEP_SIZE = 110     # 50% overlap 
FEATURES = ['RSSI', 'LQI', 'Diff_RSSI']

def main():
    print(f"Loading '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    
    # Differentiation: compute sample-to-sample RSSI change
    if 'Diff_RSSI' not in df.columns:
        print("Computing Differentiation...")
        df['Diff_RSSI'] = df.groupby(['Environment', 'Sender_Node'])['RSSI'].diff().fillna(0)
    
    # 2. Normalization: z_i = (y_i - y_min) / (y_max - y_min)
    # We apply this to the raw and differentiated columns globally
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    print("Sorting data by time...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Environment', 'Sender_Node', 'Timestamp'])

    X_list, y_env_list, y_node_list = [], [], []

    # 3. Segmentation: Overlapping frames
    print(f"Slicing into {STEP_SIZE}-step windows (50% overlap)...")
    grouped = df.groupby(['Environment', 'Sender_Node'])
    
    for name, group in grouped:
        feats = group[FEATURES].values
        env_label = group['Environment'].iloc[0]
        node_label = group['Sender_Node'].iloc[0]
        
        for i in range(0, len(feats) - WINDOW_SIZE + 1, STEP_SIZE):
            window = feats[i : i + WINDOW_SIZE]
            X_list.append(window)
            y_env_list.append(env_label)
            y_node_list.append(node_label)

    X = np.array(X_list)
    y_env = np.array(y_env_list)
    y_node = np.array(y_node_list)

    print(f"Created {len(X)} windows.")
    
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/X_windows.npy", X)
    np.save("processed_data/y_env_labels.npy", y_env)
    np.save("processed_data/y_node_labels.npy", y_node)

if __name__ == "__main__":
    main()