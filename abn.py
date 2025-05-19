import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply
from sklearn.metrics import classification_report, confusion_matrix

# — 1) Load selected-feature splits —
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

X_train = train_sel.drop(columns=["Label"]).values
y_train = train_sel["Label"].values
X_val   = val_sel.drop(columns=["Label"]).values
y_val   = val_sel["Label"].values

feature_names = train_sel.drop(columns=["Label"]).columns.tolist()
input_dim     = X_train.shape[1]

# — 2) Build the Attention-NN on 15 features —
inputs = Input(shape=(input_dim,), name="Input")
attn_w = Dense(input_dim, activation="softmax", name="Attention_Weights")(inputs)
attn_mul = Multiply(name="Attention_Multiply")([inputs, attn_w])
x = Dense(64, activation="relu")(attn_mul)
x = Dense(32, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# — 3) Train —
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30, batch_size=16,
    verbose=1
)

# — 4) Evaluate —
print("\n🔹 Confusion Matrix (ABN on Selected Features):")
y_pred = (model.predict(X_val).flatten() >= 0.5).astype(int)
print(confusion_matrix(y_val, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred))

# — 5) SHAP DeepExplainer for the Attention-NN —
# Select background samples (e.g. 50 normals for speed)
norm_idx = np.where(y_train == 0)[0]
back_idx = np.random.choice(norm_idx, size=min(50, len(norm_idx)), replace=False)
background = X_train[back_idx]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_val)

# unpack for binary sigmoid
shap_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

# — 6) Summary plot —
shap.summary_plot(
    shap_to_plot,
    X_val,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Summary — Attention-NN (Selected Features)")
plt.tight_layout()
plt.show()
