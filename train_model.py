"""
train_model.py
──────────────
Trains a TensorFlow/Keras MLP classifier on the 63-dimensional
MediaPipe hand-landmark feature vectors produced by generate_training_data.py.

Run AFTER generate_training_data.py:
    python train_model.py
"""

import numpy as np
import os
import json
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("\n── ISLBridge Model Trainer ──────────────────────────────────────────────")
print("Loading dataset…")

X = np.load("model/X_train.npy")
y = np.load("model/y_train.npy")

with open("model/label_map.json") as f:
    label_map = json.load(f)

NUM_CLASSES = len(label_map)
print(f"  Samples : {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : {NUM_CLASSES}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

def build_model(input_dim, num_classes):
    inp = keras.Input(shape=(input_dim,), name="landmarks")
    x = layers.Dense(512)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="ISL_Classifier")

model = build_model(X.shape[1], NUM_CLASSES)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=8, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="model/best_weights.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    ),
]

print("\nTraining…")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.1f}s")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"  Val accuracy : {val_acc*100:.2f}%")
print(f"  Val loss     : {val_loss:.4f}")

# Save .keras only — TFLite conversion is broken in TF2.16+Keras3
model.save("model/isl_classifier.keras")

hist_data = {k: [float(v) for v in vs] for k, vs in history.history.items()}
with open("model/training_history.json", "w") as f:
    json.dump(hist_data, f, indent=2)

print("\nSaved files:")
print("  model/isl_classifier.keras   ← main model (used by app.py)")
print("  model/best_weights.weights.h5← best checkpoint weights")
print("  model/label_map.json         ← class index → sign name")
print("  model/training_history.json  ← loss/accuracy curves")
print("\n✓ Ready. Start the app with:  python app.py")
