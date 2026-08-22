import pandas as pd 

#load the datatset
df = pd.read_csv("attrition.csv")

#Basic shape and structure
print(f"shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Column data type:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum().sum(), "total missing values")

print("\nTarget variables distribution (Attrition):")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True))