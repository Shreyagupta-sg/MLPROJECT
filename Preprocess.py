import pandas as pd

# Step 1: Load the selected-feature splits
train_sel = pd.read_excel("Training_Scaled_Split.xlsx")
val_sel   = pd.read_excel("Validation_Scaled_Split.xlsx")

# Display basic info
print("Training set shape:", train_sel.shape)
print(train_sel["Label"].value_counts(), "in training")
print("Validation set shape:", val_sel.shape)
print(val_sel["Label"].value_counts(), "in validation")

# Optional: preview the first few rows
print("\nTraining sample:\n", train_sel.head())
