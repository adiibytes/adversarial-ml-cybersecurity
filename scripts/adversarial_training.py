import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# =============== 1. Load and Clean Datasets ===============
print(" Loading original dataset and adversarial datasets...")

data_dir = "../data"
files = [
    "phishing.csv",
    "phishing_attack_substitution.csv",
    "phishing_attack_padding.csv",
    "phishing_attack_noise.csv"
]

df_list = []

for file in files:
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path)

    # Standardize label column name
    label_col = next((col for col in df.columns if col.lower() == 'result'), None)
    if label_col is None:
        raise ValueError(f"'Result' column not found in {file}")
    
    print(f"\n📄 File: {file}")
    print(f"Detected label column: {label_col}")
    print(df[label_col].value_counts())

    # Rename label column to consistent name
    df = df.rename(columns={label_col: "Result"})

    # Drop rows with NaNs (just in case)
    df = df.dropna()

    df_list.append(df)

# Combine all datasets
combined_df = pd.concat(df_list, ignore_index=True)
print(f"\n Combined dataset shape: {combined_df.shape}")

# =============== 2. Extract Features + Labels ===============
# Drop NaNs again just in case
combined_df = combined_df.dropna()

X = combined_df.drop(columns="Result")
y = combined_df["Result"]

print(f" Final cleaned dataset shape: {X.shape}")

# =============== 3. Train/Test Split ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =============== 4. Define Models ===============
models = {
    "rf_model_hardened": RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg_model_hardened": LogisticRegression(max_iter=1000),
    "xgb_model_hardened": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# =============== 5. Train, Evaluate, Save ===============
os.makedirs("../models", exist_ok=True)
os.makedirs("../report", exist_ok=True)

with open("../report/robust_model_results.txt", "w") as f:
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        print(f"✅ {name} Accuracy: {acc:.4f}")
        f.write(f"\n--- {name} ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report + "\n")

        model_path = f"../models/{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

print("\nAll hardened models trained and saved successfully.")
