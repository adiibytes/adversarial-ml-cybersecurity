import shap
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ========== 1. Load Dataset ==========
print("Loading dataset...")
data_path = '../data/phishing.csv'
df = pd.read_csv(data_path)

# Detect label column
label_col = next((col for col in df.columns if col.lower() == 'result'), None)
if not label_col:
    raise ValueError(" 'Result' column not found in dataset.")

# Clean
df = df.dropna(subset=[label_col])
df = df.dropna()
X = df.drop(columns=[label_col])
y = df[label_col]

# ========== 2. Load Models ==========
print(" Loading hardened models...")
model_dir = '../models'
model_files = {
    "Random Forest": "rf_model_hardened.pkl",
    "XGBoost": "xgb_model_hardened.pkl",
    "Logistic Regression": "logreg_model_hardened.pkl"
}

models = {}
for name, filename in model_files.items():
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f" Loaded {name}")
    else:
        print(f" Model not found: {path}")

# ========== 3. SHAP Explainability ==========
print(" Generating SHAP explanations...")
os.makedirs("../report", exist_ok=True)

for model_name, model in models.items():
    print(f"\n Explaining {model_name}...")

    try:
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_summary = shap_values[1] if isinstance(shap_values, list) else shap_values
        elif isinstance(model, LogisticRegression):
            explainer = shap.Explainer(model, X)
            shap_summary = explainer(X)
        else:
            print(f" Unsupported model: {model_name}")
            continue

        # Plot and save
        plt.title(f"SHAP Summary - {model_name}")
        shap.summary_plot(shap_summary, X, show=False)
        plot_path = f"../report/shap_summary_{model_name.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.clf()
        print(f"Saved SHAP summary for {model_name} at: {plot_path}")

    except Exception as e:
        print(f" SHAP failed for {model_name}: {e}")

print("\n All SHAP explainability visualizations saved.")
