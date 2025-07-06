import pandas as pd
import numpy as np
import os

# ===============================
# Step 1: Load Original Dataset
# ===============================
data_path = '../data/phishing.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(" phishing.csv not found in ../data. Please run train_models.py first.")

print(" Loading original dataset...")
df = pd.read_csv(data_path)

X = df.drop(columns='result')
y = df['result']

# ===============================
# Step 2: Character Substitution Attack
# ===============================
def substitute_chars(df, columns, substitutions):
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str)
            for old, new in substitutions.items():
                df_copy[col] = df_copy[col].str.replace(old, new, regex=False)
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    return df_copy

# Features likely to contain obfuscated patterns (customize as needed)
char_subs = {'0': '1', '1': '0'}
url_features = ['having_IP_Address', 'URL_Length', 'Prefix_Suffix']  # Example binary/numeric cols
X_attack1 = substitute_chars(X, url_features, char_subs)

# ===============================
# Step 3: Feature Padding Attack
# ===============================
def add_padding(df, cols, noise_level=1):
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] + noise_level
    return df_copy

# Simulate "adding benign signals"
padding_features = ['SSLfinal_State', 'HTTPS_token']
X_attack2 = add_padding(X, padding_features)

# ===============================
# Step 4: Random Noise Injection Attack
# ===============================
def flip_binary_features(df, flip_fraction=0.1):
    df_copy = df.copy()
    binary_cols = [col for col in df_copy.columns if set(df_copy[col].unique()) <= {0, 1}]
    for col in binary_cols:
        n = int(flip_fraction * len(df_copy))
        idx = np.random.choice(df_copy.index, size=n, replace=False)
        df_copy.loc[idx, col] = 1 - df_copy.loc[idx, col]  # flip 0 ↔ 1
    return df_copy

X_attack3 = flip_binary_features(X)

# ===============================
# Step 5: Save Attacked Datasets
# ===============================
os.makedirs('../data', exist_ok=True)

X_attack1['Result'] = y
X_attack2['Result'] = y
X_attack3['Result'] = y

X_attack1.to_csv('../data/phishing_attack_substitution.csv', index=False)
X_attack2.to_csv('../data/phishing_attack_padding.csv', index=False)
X_attack3.to_csv('../data/phishing_attack_noise.csv', index=False)

print(" All adversarial datasets generated and saved in '../data/'")
