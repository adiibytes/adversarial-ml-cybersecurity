# train_robust_models.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========================
# 1. Load original dataset
# ========================
print(" Loading original dataset...")
df = pd.read_csv("../data/phishing.csv")

label_col = next((col for col in df.columns if col.lower() == 'result'), None)
X_orig = df.drop(columns=label_col)
y_orig = df[label_col]

# ================================
# 2. Load and concatenate attacks
# ================================
adv_files = [
    "phishing_attack_substitution.csv",
    "phishing_attack_padding.csv",
    "phishing_attack_noise.csv"
]

for file in adv_files:
    path = os.path.join("../data", file)
    df_adv = pd.read_csv(path)
    label_adv = next((col for col in df_adv.columns if col.lower() == 'result'), None)
    X_adv = df_adv.drop(columns=label_adv)
    y_adv = df_adv[label_adv]

    X_orig = pd.concat([X_orig, X_adv], ignore_index=True)
    y_orig = pd.concat([y_orig, y_adv], ignore_index=True)

print(f" Total training samples after augmentation: {X_orig.shape[0]}")

# ========================
# 3. Train-test split
# ========================
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.3, random_state=42)

# ========================
# 4. Define models
# ========================
models = {
    "rf_model": RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg_model": LogisticRegression(max_iter=1000),
    "xgb_model": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ========================
# 5. Train and Save Models
# ========================
os.makedirs("../models_robust", exist_ok=True)
os.makedirs("../report", exist_ok=True)

log_file = "../report/robust_model_results.txt"
with open(log_file, "w") as f:
    for name, model in models.items():
        print(f"\n Training {name} with adversarial data...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        print(f" {name} Accuracy: {acc:.4f}")
        f.write(f"\n--- {name} ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report + "\n")

        joblib.dump(model, f"../models_robust/{name}.pkl")
        print(f" Saved to '../models_robust/{name}.pkl'")

print("\n Robust models trained. Results saved to '../report/robust_model_results.txt'")
