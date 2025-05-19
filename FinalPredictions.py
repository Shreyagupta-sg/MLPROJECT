# final_inference_selected_clean.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.optimizers import Adam

# 1) Load and prepare the full selected-feature training set
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
X_full    = train_sel.drop(columns=["Label"])
y_full    = train_sel["Label"].values

# 2) Fit scaler on training features
scaler = StandardScaler().fit(X_full)
X_full_s = scaler.transform(X_full)

# 3) Retrain Random Forest on all selected features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_full_s, y_full)

# 4) Retrain Attention-NN on all selected features
input_dim = X_full_s.shape[1]
inp = Input(shape=(input_dim,), name="Input")
attn_w = Dense(input_dim, activation="softmax", name="Attention_Weights")(inp)
attn_mul = Multiply(name="Attention_Multiply")([inp, attn_w])
x = Dense(64, activation="relu", name="Dense_1")(attn_mul)
x = Dense(32, activation="relu", name="Dense_2")(x)
out = Dense(1, activation="sigmoid", name="Output")(x)
abn = Model(inputs=inp, outputs=out, name="AttentionNN")
abn.compile(optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"])
abn.fit(X_full_s, y_full, epochs=30, batch_size=16, verbose=0)

# 5) Prepare ExcelWriter for final predictions
writer = pd.ExcelWriter("Final_Selected_Predictions.xlsx", engine="xlsxwriter")

# 6) Loop through each sheet in the test workbook
test_book = pd.ExcelFile("Test_Data_588_Project_Spring2025.xlsx")
for sheet in test_book.sheet_names:
    # a) Load and clean this sheet’s DataFrame
    df_test = test_book.parse(sheet)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.fillna(df_test.mean())

    # b) Align features and scale
    X_test = df_test[X_full.columns]
    X_test_s = scaler.transform(X_test)

    # c) Random Forest predictions
    rf_preds = rf.predict(X_test_s)

    # d) Attention-NN predictions
    abn_probs = abn.predict(X_test_s).flatten()
    abn_preds = (abn_probs >= 0.5).astype(int)

    # e) Compile results and write to sheet
    out_df = pd.DataFrame({
        "RF_Pred":  rf_preds,
        "ABN_Pred": abn_preds
    })
    out_df.to_excel(writer, sheet_name=sheet, index=False)

# 7) Save and close the Excel file
writer.close()
print("✅ Final_Selected_Predictions.xlsx created successfully.")
