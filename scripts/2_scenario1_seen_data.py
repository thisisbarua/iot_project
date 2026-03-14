import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Add, Activation, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==========================================
# 1. LOAD THE 4-CHANNEL DATA
# ==========================================
print("📥 Loading 4-channel rubric-compliant data...")
X = np.load("processed_data/X_windows.npy")
y_env = np.load("processed_data/y_env_labels.npy")

print(f"📐 Data Shape: {X.shape}") # Expecting (Samples, 100, 4)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_env)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
num_classes = len(encoder.classes_)

# Split 75/25 as per requirement [cite: 56]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.25, random_state=42
)

# ==========================================
# 2. HEAVY-DUTY CNN
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
# 3. DEEP MULTI-BLOCK RESNET
# ==========================================
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x) 
    
    # Block 1
    shortcut = Conv1D(128, kernel_size=1, padding='same')(x) 
    x = Conv1D(128, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    # Block 2
    shortcut = Conv1D(256, kernel_size=1, padding='same', strides=2)(x) 
    x = Conv1D(256, kernel_size=5, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x]) 
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==========================================
# 4. EXECUTION
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])
reduce_lr = lambda: ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

print("\n🧠 TRAINING CNN...")
cnn = build_cnn(input_shape, num_classes)
cnn.fit(X_train, y_train, epochs=40, batch_size=64, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\n🚀 TRAINING RESNET...")
resnet = build_resnet(input_shape, num_classes)
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet.fit(X_train, y_train, epochs=40, batch_size=64, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\n📊 FINAL SCORES")
_, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
_, res_acc = resnet.evaluate(X_test, y_test, verbose=0)
print(f"CNN: {cnn_acc*100:.2f}% | ResNet: {res_acc*100:.2f}%")
