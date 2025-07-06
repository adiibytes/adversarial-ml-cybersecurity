import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ===========================
# 1. Load Pretrained Models
# ===========================
model_dir = '../models'
model_files = {
    'Random Forest': os.path.join(model_dir, 'rf_model.pkl'),
    'Logistic Regression': os.path.join(model_dir, 'logreg_model.pkl'),
    'XGBoost': os.path.join(model_dir, 'xgb_model.pkl')
}

models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        raise FileNotFoundError(f" Model file not found: {path}")

# ===========================
# 2. Load Attacked Datasets
# ===========================
data_dir = '../data'
attack_files = {
    'Character Substitution': 'phishing_attack_substitution.csv',
    'Feature Padding': 'phishing_attack_padding.csv',
    'Noise Injection': 'phishing_attack_noise.csv'
}

attacked_datasets = {}
for attack_type, file_name in attack_files.items():
    path = os.path.join(data_dir, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)

        if 'Result' not in df.columns:
            raise ValueError(f" 'Result' column not found in {file_name}")

        X = df.drop(columns='Result')
        y = df['Result']

        attacked_datasets[attack_type] = (X, y)
    else:
        raise FileNotFoundError(f" Attack file missing: {path}")



# ===========================
# 3. Evaluate Models
# ===========================
os.makedirs('../report', exist_ok=True)
log_path = '../report/adversarial_results.txt'

with open(log_path, 'w') as f:
    for model_name, model in models.items():
        f.write(f"\n===== {model_name} =====\n")
        print(f"\n Evaluating: {model_name}")
        for attack_name, (X_adv, y_adv) in attacked_datasets.items():
            preds = model.predict(X_adv)
            acc = accuracy_score(y_adv, preds)
            report = classification_report(y_adv, preds)

            print(f"  {attack_name}: Accuracy = {acc:.4f}")
            f.write(f"\n--- {attack_name} ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n")

print("\n Adversarial evaluation complete. Results saved to '../report/adversarial_results.txt'")
