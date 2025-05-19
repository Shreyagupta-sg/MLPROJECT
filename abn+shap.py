import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Multiply # type: ignore

print("Starting ABN with SHAP analysis...")

# 1) Load selected-feature splits
print("Loading data...")
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

X_train = train_sel.drop(columns=["Label"]).values
y_train = train_sel["Label"].values
X_val   = val_sel.drop(columns=["Label"]).values

feature_names = train_sel.drop(columns=["Label"]).columns.tolist()
print(f"Training data shape: {X_train.shape}, feature names: {feature_names}")

# 2) Build & train Attention‐NN exactly as before
print("Building and training ABN model...")
input_dim = X_train.shape[1]
inputs   = Input(shape=(input_dim,))
attn_w   = Dense(input_dim, activation="softmax")(inputs)
attn_mul = Multiply()([inputs, attn_w])
x        = Dense(64, activation="relu")(attn_mul)
x        = Dense(32, activation="relu")(x)
output   = Dense(1, activation="sigmoid")(x)
model    = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Training model...")
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, val_sel["Label"].values), verbose=1)

# 3) SHAP DeepExplainer
print("Computing SHAP values...")
# Use a small background of healthy samples
background_idx = np.where(y_train == 0)[0]
background     = X_train[np.random.choice(background_idx, size=min(50, len(background_idx)), replace=False)]

explainer   = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_val)

# 4) Unpack and plot
print("Creating SHAP plot...")
shap_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
shap.summary_plot(
    shap_to_plot,
    X_val,
    feature_names=feature_names
)
print("Analysis complete!")
