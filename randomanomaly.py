import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# 1) Load the RF-selected feature splits (15 features + Label)
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

# 2) Extract feature matrices and labels
X_train_all = train_sel.drop(columns=["Label"]).values
y_train_all = train_sel["Label"].values
X_val       = val_sel.drop(columns=["Label"]).values
y_val       = val_sel["Label"].values

# 3) Filter for healthy (label=0) samples in training
X_norm = X_train_all[y_train_all == 0]

# 4) Scale both training-norm and validation sets
scaler = StandardScaler().fit(X_norm)
X_norm_s = scaler.transform(X_norm)
X_val_s  = scaler.transform(X_val)

# 5) Build a simple autoencoder
input_dim = X_norm_s.shape[1]  # should be 15
inp  = Input(shape=(input_dim,))
e1   = Dense(16, activation="relu")(inp)
e2   = Dense(8,  activation="relu")(e1)
d1   = Dense(16, activation="relu")(e2)
out  = Dense(input_dim, activation="linear")(d1)
ae   = Model(inputs=inp, outputs=out)
ae.compile(optimizer=Adam(1e-3), loss="mse")

# 6) Train on healthy subset
ae.fit(
    X_norm_s, X_norm_s,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 7) Compute reconstruction error on validation
recon_val = ae.predict(X_val_s)
mse_val   = np.mean(np.square(X_val_s - recon_val), axis=1)

# 8) Determine threshold from train errors (mean + 2*std)
recon_train = ae.predict(X_norm_s)
mse_train   = np.mean(np.square(X_norm_s - recon_train), axis=1)
threshold   = mse_train.mean() + 2 * mse_train.std()

# 9) Flag anomalies: 1 if error > threshold, else 0
y_pred_ae = (mse_val > threshold).astype(int)

# 10) Evaluate
print(f"\nAutoencoder on RF-Selected Features — Threshold = {threshold:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_ae))
print("\nClassification Report:")
print(classification_report(y_val, y_pred_ae))
