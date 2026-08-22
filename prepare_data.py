import pandas as pd

df = pd.read_csv("attrition.csv")

# Drop columns with no useful information
df = df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])

# Convert the target variable to 1/0
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Identify which remaining columns are categorical (text) vs numeric
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns to encode:", categorical_cols)

# One-hot encode all categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"\nShape before encoding: {df.shape}")
print(f"Shape after encoding: {df_encoded.shape}")

print("\nSample of encoded columns:")
print(df_encoded.columns.tolist())

# Save this prepared version for the next session
df_encoded.to_csv("attrition_prepared.csv", index=False)
print("\nSaved prepared data to attrition_prepared.csv")