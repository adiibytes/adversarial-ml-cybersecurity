# train_models.py

import os
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================
# 1. Load dataset from UCI
# ==========================
print(" Downloading dataset from UCI...")
phishing = fetch_ucirepo(id=327)  # Phishing Website dataset
X = phishing.data.features
y = phishing.data.targets

y = y.replace(-1, 0)

# Save the dataset locally
df = pd.concat([X, y], axis=1)
os.makedirs('../data', exist_ok=True)
df.to_csv('../data/phishing.csv', index=False)
print("Dataset saved to '../data/phishing.csv'")

# ==========================
# 2. Split data
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================
# 3. Define models
# ==========================
models = {
    "rf_model": RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg_model": LogisticRegression(max_iter=1000),
    "xgb_model": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ==========================
# 4. Train & Save models
# ==========================
os.makedirs("../models", exist_ok=True)
os.makedirs("../report", exist_ok=True)

with open("../report/model_results.txt", "w") as log_file:
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        print(f" {name} Accuracy: {acc:.4f}")
        log_file.write(f"\n--- {name} ---\n")
        log_file.write(f"Accuracy: {acc:.4f}\n")
        log_file.write(report + "\n")

        joblib.dump(model, f"../models/{name}.pkl")
        print(f" Saved {name} to '../models/{name}.pkl'")

print("\n All models trained and saved. Logs written to '../report/model_results.txt'")
