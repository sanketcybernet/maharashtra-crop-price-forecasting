import pandas as pd

file = "Dataset/Maharashtra/final_model_data.csv"

df = pd.read_csv(file)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCrops available:")
print(df["Commodity"].unique())

print("\nNumber of crops:", df["Commodity"].nunique())

print("\nMarkets available:", df["Market"].nunique())

print("\nDate range:")
print(df["Arrival_Date"].min(), "to", df["Arrival_Date"].max())