import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1) Load the selected-feature splits
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

X_train = train_sel.drop(columns=["Label"])
y_train = train_sel["Label"]
X_val   = val_sel.drop(columns=["Label"])

# 2) Train Random Forest
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sel.fit(X_train, y_train)

# 3) Compute SHAP values
explainer = shap.TreeExplainer(rf_sel)
shap_vals = explainer.shap_values(X_val)

# 4) Pick class-1 SHAP values (compromised)
shap_to_plot = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

# 5) Plot summary
shap.summary_plot(
    shap_to_plot,
    X_val,
    feature_names=X_val.columns
)
