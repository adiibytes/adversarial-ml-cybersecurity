import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load models
model_dir = '../models'
model_files = {
    'Random Forest': os.path.join(model_dir, 'rf_model.pkl'),
    'Logistic Regression': os.path.join(model_dir, 'logreg_model.pkl'),
    'XGBoost': os.path.join(model_dir, 'xgb_model.pkl')
}
models = {name: joblib.load(path) for name, path in model_files.items()}

# Load adversarial datasets
data_dir = '../data'
attack_files = {
    'Character Substitution': 'phishing_attack_substitution.csv',
    'Feature Padding': 'phishing_attack_padding.csv',
    'Noise Injection': 'phishing_attack_noise.csv'
}

attacked_datasets = {}
for attack_type, file_name in attack_files.items():
    path = os.path.join(data_dir, file_name)
    df = pd.read_csv(path)

    label_col = next((col for col in df.columns if col.lower() == 'result'), None)
    if not label_col:
        raise ValueError(f"'result' column not found in {file_name}")

    X = df.drop(columns=label_col)
    y = df[label_col]
    attacked_datasets[attack_type] = (X, y)

# Plotting confusion matrices
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
for row_idx, (model_name, model) in enumerate(models.items()):
    for col_idx, (attack_name, (X_adv, y_adv)) in enumerate(attacked_datasets.items()):
        preds = model.predict(X_adv)
        cm = confusion_matrix(y_adv, preds)

        ax = axes[row_idx, col_idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

        ax.set_title(f"{model_name}\n{attack_name}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Set tick labels only on outer edges
        if row_idx < 2:
            ax.set_xlabel('')
        if col_idx > 0:
            ax.set_ylabel('')

plt.tight_layout()
os.makedirs("../report", exist_ok=True)
out_path = "../report/adversarial_confusion_matrices.png"
plt.savefig(out_path)
plt.show()
print(f" Confusion matrices saved to: {out_path}")
