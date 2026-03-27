import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Add, Activation, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==========================================
# 1. LOAD THE SCENARIO 2 CUSTOM DATA
# ==========================================
print("📥 Loading zero-mean centered data for SCENARIO II...")
X = np.load("processed_data/X_windows_scen2.npy")
y_node = np.load("processed_data/y_node_labels_scen2.npy")
y_env = np.load("processed_data/y_env_labels_scen2.npy") 

# ==========================================
# ENVIRONMENT-INVARIANT TRANSFORMS (applied at runtime, file 4 untouched)
# ==========================================
# Step A: Z-Score Standardization — normalizes variance per window.
# The data is already zero-mean centered (from file 4), so dividing by std
# removes the environment-dependent signal spread (e.g., forest=high variance,
# open_field=low variance), leaving only the hardware jitter pattern.
print("⚙️  Applying Per-Window Z-Score Standardization...")
window_std = X.std(axis=1, keepdims=True) + 1e-8
X = X / window_std

# Step B: Keep ONLY differential features (columns 2=Diff_RSSI, 3=Diff_LQI).
# Raw RSSI and LQI still carry environment-specific absolute patterns even after
# centering. Differentials (sample-to-sample change) are inherently more
# hardware-specific since they capture rapid micro-jitter from the transmitter chip.
print("⚙️  Slicing to differential-only features: [Diff_RSSI, Diff_LQI]")
X = X[:, :, [2, 3]]
print(f"📐 Final Data Shape: {X.shape}")

# 🎯 Target is the Node!
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_node)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
num_classes = len(encoder.classes_)

# ==========================================
# 2. THE "UNSEEN ENVIRONMENT" SPLIT 
# ==========================================
unique_envs = np.unique(y_env)
print(f"🌲 Found environments: {unique_envs}")

# THE FIX: Explicitly target the lowercase 'lake' to match your dataset perfectly
unseen_test_env = 'lake' 

print(f"🏋️ Training hardware signatures on everything EXCEPT the {unseen_test_env}...")
print(f"🧪 Testing hardware identification strictly in the UNSEEN {unseen_test_env}...")

# Create masks to isolate the lake
test_mask = y_env == unseen_test_env
train_mask = y_env != unseen_test_env

X_train_raw = X[train_mask]
y_train_raw = y_categorical[train_mask]
X_test = X[test_mask]
y_test = y_categorical[test_mask]

# ==========================================
# THE SHUFFLE FIX
# ==========================================
# Mix the nodes and environments so validation_split gets a healthy random sample!
indices = np.arange(len(X_train_raw))
np.random.shuffle(indices)
X_train = X_train_raw[indices]
y_train = y_train_raw[indices]

print(f"📐 Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

# ==========================================
# 3. HEAVY-DUTY CNN
# ==========================================
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=128, kernel_size=11, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=256, kernel_size=7, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling1D(),
        
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. DEEP MULTI-BLOCK RESNET 
# ==========================================
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(256, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x) 
    
    # Block 1
    shortcut = Conv1D(256, kernel_size=1, padding='same')(x) 
    x = Conv1D(256, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    # Block 2
    shortcut = Conv1D(512, kernel_size=1, padding='same', strides=2)(x) 
    x = Conv1D(512, kernel_size=5, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(512, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.7)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==========================================
# 5. EXECUTION
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])
reduce_lr = lambda: ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)

print("\n🧠 TRAINING CNN (UNSEEN ENV - NODE ID)...")
cnn = build_cnn(input_shape, num_classes)
cnn.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\n🚀 TRAINING RESNET (UNSEEN ENV - NODE ID)...")
resnet = build_resnet(input_shape, num_classes)
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\n📊 FINAL SCORES (SCENARIO II - STRATEGY 2: UNSEEN ENVIRONMENT)")
_, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
_, res_acc = resnet.evaluate(X_test, y_test, verbose=0)
print(f"CNN Node Accuracy: {cnn_acc*100:.2f}% | ResNet Node Accuracy: {res_acc*100:.2f}%")


# ==========================================
# 5. GENERATE METRICS FOR LATEX REPORT
# ==========================================
print("\n" + "="*50)
print(" 📊 GENERATING PERFORMANCE METRICS FOR REPORT")
print("="*50)

# Get raw predictions
print("Evaluating CNN...")
y_pred_cnn_probs = cnn.predict(X_test, verbose=0)
y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

print("Evaluating ResNet...")
y_pred_resnet_probs = resnet.predict(X_test, verbose=0)
y_pred_resnet = np.argmax(y_pred_resnet_probs, axis=1)

# Convert actual test labels back from One-Hot Encoding
y_true = np.argmax(y_test, axis=1)

# Print the final tables for your LaTeX report
print("\n--- 1D CNN PERFORMANCE ---")
print(classification_report(y_true, y_pred_cnn, target_names=encoder.classes_, digits=4))

print("\n--- RESNET PERFORMANCE ---")
print(classification_report(y_true, y_pred_resnet, target_names=encoder.classes_, digits=4))