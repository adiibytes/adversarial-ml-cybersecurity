import os
import pandas as pd

data_dir = "../data"
files = [
    "phishing.csv",
    "phishing_attack_substitution.csv",
    "phishing_attack_padding.csv",
    "phishing_attack_noise.csv"
]

for file in files:
    path = os.path.join(data_dir, file)
    print(f"\n {file}")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    label_col = next((col for col in df.columns if col.lower() == 'result'), None)
    print(f"Detected label column: {label_col}")
    print(df[label_col].value_counts(dropna=False))

    # Check for NaNs
    label_nans = df[label_col].isna().sum()
    feature_nans = df.drop(columns=[label_col]).isna().sum().sum()
    print(f"Label NaNs: {label_nans}")
    print(f"Feature NaNs: {feature_nans}")

    # Preview suspicious rows
    if label_nans > 0:
        print("\n Rows with NaN in label column:")
        print(df[df[label_col].isna()].head())

    if feature_nans > 0:
        print("\n Rows with NaN in features:")
        print(df[df.drop(columns=[label_col]).isna().any(axis=1)].head())

    print("-" * 60)
