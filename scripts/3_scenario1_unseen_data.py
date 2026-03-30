import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Add, Activation, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers

# ==========================================
# 1. LOAD THE DATA
# ==========================================
print("Loading data...")
X = np.load("processed_data/X_windows.npy")
y_env = np.load("processed_data/y_env_labels.npy")
y_node = np.load("processed_data/y_node_labels.npy")

# Encode the Environment targets
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_env)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
num_classes = len(encoder.classes_)

# ==========================================
# 2. THE "UNSEEN DATA" SPLIT
# ==========================================
unique_nodes = np.unique(y_node)
print(f"Found sensor nodes: {unique_nodes}")

# Reserve the very last node in the dataset as the "Unseen" test node
unseen_test_node = unique_nodes[-1]
train_nodes = unique_nodes[:-1]

print(f"Training on nodes: {train_nodes}")
print(f"Testing strictly on UNSEEN node: {unseen_test_node}")

# Create masks to separate the data
train_mask = np.isin(y_node, train_nodes)
test_mask = y_node == unseen_test_node

X_train_raw = X[train_mask]
y_train_raw = y_categorical[train_mask]
X_test = X[test_mask]
y_test = y_categorical[test_mask]

# ==========================================
# SHUFFLE BEFORE KERAS SPLITS IT
# ==========================================
# Shuffle so validation_split gets a random sample
indices = np.arange(len(X_train_raw))
np.random.shuffle(indices)
X_train = X_train_raw[indices]
y_train = y_train_raw[indices]

print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

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
        
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
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
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==========================================
# 5. EXECUTION
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])
reduce_lr = lambda: ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

print("\nTRAINING CNN (UNSEEN DATA)...")
cnn = build_cnn(input_shape, num_classes)
cnn.fit(X_train, y_train, epochs=60, batch_size=16, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\nTRAINING RESNET (UNSEEN DATA)...")
resnet = build_resnet(input_shape, num_classes)
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet.fit(X_train, y_train, epochs=60, batch_size=16, validation_split=0.1, callbacks=[reduce_lr()], verbose=1)

print("\nFINAL SCORES (SCENARIO I - STRATEGY 2: UNSEEN DATA)")
_, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
_, res_acc = resnet.evaluate(X_test, y_test, verbose=0)
print(f"CNN Accuracy: {cnn_acc*100:.2f}% | ResNet Accuracy: {res_acc*100:.2f}%")

# ==========================================
# 6. GENERATE METRICS & MATRICES FOR LATEX REPORT
# ==========================================
print("\n" + "="*50)
print("GENERATING PERFORMANCE METRICS & MATRICES")
print("="*50)

def evaluate_and_plot(model, model_name, filename):
    print(f"Evaluating {model_name}...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Print Classification Report
    print(f"\n--- {model_name} PERFORMANCE ---")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, digits=4))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(f'Confusion Matrix - Scenario I Unseen Node ({model_name})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved matrix to {filename}")
    plt.close()

# Generate for both
evaluate_and_plot(cnn, "1D CNN", "scen1_unseen_cnn_cm.png")
evaluate_and_plot(resnet, "ResNet", "scen1_unseen_resnet_cm.png")