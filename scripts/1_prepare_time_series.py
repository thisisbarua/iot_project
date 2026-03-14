import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Config per PDF
WINDOW_SIZE = 100 # 10s of data [cite: 209]
STEP_SIZE = 10    # 90% overlap for MASSIVE training data augmentation

df = pd.read_csv("ULTIMATE_MASTER_DATASET.csv")

# 1. Differentiation (Requirement: Page 3) [cite: 202]
df['Diff_RSSI'] = df.groupby(['Environment', 'Sender_Node'])['RSSI'].diff().fillna(0)
df['Diff_LQI'] = df.groupby(['Environment', 'Sender_Node'])['LQI'].diff().fillna(0)

# 2. Normalization (Formula: z_i = (y_i - y_min) / (y_max - y_min)) 
scaler = MinMaxScaler()
cols_to_scale = ['RSSI', 'LQI', 'Diff_RSSI', 'Diff_LQI']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 3. Windowing [cite: 207]
X_list, y_env_list, y_node_list = [], [], []
grouped = df.groupby(['Environment', 'Sender_Node'])

for name, group in grouped:
    feats = group[cols_to_scale].values
    env = group['Environment'].iloc[0]
    node = group['Sender_Node'].iloc[0]
    for i in range(0, len(feats) - WINDOW_SIZE, STEP_SIZE):
        X_list.append(feats[i:i+WINDOW_SIZE])
        y_env_list.append(env)
        y_node_list.append(node)

np.save("processed_data/X_windows.npy", np.array(X_list))
np.save("processed_data/y_env_labels.npy", np.array(y_env_list))
np.save("processed_data/y_node_labels.npy", np.array(y_node_list))
print(f"🚀 Prepared {len(X_list)} windows with 4 channels of information!")
