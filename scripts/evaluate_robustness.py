import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. Load Models ==========
model_paths = {
    'Baseline': {
        'Random Forest': '../models/rf_model.pkl',
        'Logistic Regression': '../models/logreg_model.pkl',
        'XGBoost': '../models/xgb_model.pkl'
    },
    'Robust': {
        'Random Forest': '../models_robust/rf_model.pkl',
        'Logistic Regression': '../models_robust/logreg_model.pkl',
        'XGBoost': '../models_robust/xgb_model.pkl'
    }
}

models = {'Baseline': {}, 'Robust': {}}

for version in model_paths:
    for name, path in model_paths[version].items():
        if os.path.exists(path):
            models[version][name] = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model not found: {path}")

# ========== 2. Load Adversarial Datasets ==========
data_dir = '../data'
attack_files = {
    'Character Substitution': 'phishing_attack_substitution.csv',
    'Feature Padding': 'phishing_attack_padding.csv',
    'Noise Injection': 'phishing_attack_noise.csv'
}

datasets = {}
for attack_type, file in attack_files.items():
    path = os.path.join(data_dir, file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset missing: {path}")

    df = pd.read_csv(path)
    label_col = next((col for col in df.columns if col.lower() == 'result'), None)
    if not label_col:
        raise ValueError(f"'result' column not found in {file}")

    X = df.drop(columns=label_col)
    y = df[label_col]
    datasets[attack_type] = (X, y)

# ========== 3. Evaluate Models ==========
os.makedirs('../report', exist_ok=True)
log_file = '../report/robust_vs_baseline.txt'

with open(log_file, 'w') as f:
    for model_type in ['Baseline', 'Robust']:
        f.write(f"\n================= {model_type} Models =================\n")
        print(f"\n{model_type} Models\n")
        for model_name, model in models[model_type].items():
            f.write(f"\n--- {model_name} ---\n")
            print(f"Evaluating {model_type} - {model_name}")
            for attack_name, (X_adv, y_adv) in datasets.items():
                preds = model.predict(X_adv)
                acc = accuracy_score(y_adv, preds)
                report = classification_report(y_adv, preds)

                f.write(f"\n{attack_name}:\nAccuracy: {acc:.4f}\n{report}\n")
                print(f"{attack_name}: {acc:.4f}")

print("\nEvaluation complete. Results saved to '../report/robust_vs_baseline.txt'")
