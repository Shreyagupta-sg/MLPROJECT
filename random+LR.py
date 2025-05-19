import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load the selected-feature splits
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

X_train = train_sel.drop("Label", axis=1)
y_train = train_sel["Label"]
X_val   = val_sel.drop("Label", axis=1)
y_val   = val_sel["Label"]

# 2) Train Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# 3) Predict on validation set
y_pred_lr = lr.predict(X_val)

# 4) Evaluate
print("\n🔹 Confusion Matrix (Selected Features):")
print(confusion_matrix(y_val, y_pred_lr))
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred_lr))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load the selected-feature splits
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

X_train = train_sel.drop("Label", axis=1)
y_train = train_sel["Label"]
X_val   = val_sel.drop("Label", axis=1)
y_val   = val_sel["Label"]

# 2) Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 3) Predict on validation set
y_pred_rf = rf.predict(X_val)

# 4) Evaluate
print("\n🔹 Confusion Matrix (Selected Features, RF):")
print(confusion_matrix(y_val, y_pred_rf))
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred_rf))
