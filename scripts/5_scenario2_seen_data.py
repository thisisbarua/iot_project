import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, SpatialDropout1D, BatchNormalization, Add, Activation, GlobalAveragePooling1D, GaussianNoise
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# ==========================================
# 1. LOAD THE SCENARIO II DATA
# ==========================================
print("📥 Loading Zero-Mean Centered data for Scenario II...")
X = np.load("processed_data/X_windows_scen2.npy")
y_node = np.load("processed_data/y_node_labels_scen2.npy")

print(f"📐 Data Shape: {X.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_node)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
num_classes = len(encoder.classes_)
print(f"📡 Classifying {num_classes} nodes: {list(encoder.classes_)}")

# STRATIFIED SPLIT - Critical for Node Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.25, random_state=42, stratify=y_categorical
)

print(f"📐 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ==========================================
# 2. SEQUENCE-REGULARIZED RF-FINGERPRINTING CNN
# ==========================================
def build_regularized_cnn(input_shape, num_classes):
    # L2 constraint increased to 5e-4 to combat 91% train vs 66% val overfitting
    reg = l2(5e-4) 
    
    model = Sequential([
        Input(shape=input_shape),
        # Increased noise injection to force ignoring environmental artifacts
        GaussianNoise(0.05), 
        
        Conv1D(filters=64, kernel_size=7, padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2), # Drops entire 1D feature maps to prevent sequence co-adaptation
        
        Conv1D(filters=128, kernel_size=5, padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2),
        
        Conv1D(filters=256, kernel_size=3, padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling1D(),
        
        Dense(128, activation='relu', kernel_regularizer=reg),
        Dropout(0.5), # Standard dropout for dense layers
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. SEQUENCE-REGULARIZED RESNET
# ==========================================
def build_regularized_resnet(input_shape, num_classes):
    reg = l2(5e-4)
    inputs = Input(shape=input_shape)
    
    x = GaussianNoise(0.05)(inputs)
    
    x = Conv1D(64, kernel_size=7, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x) 
    x = SpatialDropout1D(0.2)(x)
    
    # Block 1
    shortcut = Conv1D(64, kernel_size=1, padding='same')(x) 
    x = Conv1D(64, kernel_size=5, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    # Block 2
    shortcut = Conv1D(128, kernel_size=1, padding='same', strides=2)(x) 
    x = Conv1D(128, kernel_size=5, padding='same', strides=2, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==========================================
# 4. EXECUTION
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])

# Mechanics: Reduce LR to find fine minima, and EarlyStopping to prevent memorization
def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    return [reduce_lr, early_stop]

print("\n🧠 TRAINING CNN (SCENARIO II - SEEN DATA - SEQUENCE REGULARIZED)...")
cnn = build_regularized_cnn(input_shape, num_classes)
cnn.fit(X_train, y_train, epochs=80, batch_size=64, validation_split=0.1, 
        callbacks=get_callbacks(), verbose=1)

print("\n🚀 TRAINING RESNET (SCENARIO II - SEEN DATA - SEQUENCE REGULARIZED)...")
resnet = build_regularized_resnet(input_shape, num_classes)
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet.fit(X_train, y_train, epochs=80, batch_size=64, validation_split=0.1, 
           callbacks=get_callbacks(), verbose=1)

print("\n📊 FINAL SCORES (SCENARIO II - STRATEGY 1: SEEN DATA - SEQUENCE REGULARIZED)")
_, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
_, res_acc = resnet.evaluate(X_test, y_test, verbose=0)
print(f"CNN Accuracy: {cnn_acc*100:.2f}% | ResNet Accuracy: {res_acc*100:.2f}%")
