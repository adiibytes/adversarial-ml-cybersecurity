# plot_adversarial_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# Step 1: Parse results file
# ============================
log_path = '../report/adversarial_results.txt'

results = []
current_model = None

with open(log_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("====="):
            current_model = line.strip("= ").strip()
        elif line.startswith("---"):
            attack = line.strip("- ").strip()
        elif line.startswith("Accuracy:"):
            accuracy = float(line.split(":")[1].strip())
            results.append({
                "Model": current_model,
                "Attack": attack,
                "Accuracy": accuracy
            })

# Create DataFrame
df = pd.DataFrame(results)

# ============================
# Step 2: Plot bar chart
# ============================
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x="Attack", y="Accuracy", hue="Model", data=df)

plt.title("Model Accuracy under Adversarial Attacks")
plt.ylim(0.85, 1.0)
plt.ylabel("Accuracy")
plt.xlabel("Attack Type")
plt.legend(title="Model")
plt.tight_layout()

# Save plot
os.makedirs("../report", exist_ok=True)
plot_path = "../report/adversarial_accuracy_plot.png"
plt.savefig(plot_path)
plt.show()
print(f" Bar plot saved to: {plot_path}")

# ============================
# Step 3: Export LaTeX Table
# ============================
latex_path = "../report/adversarial_results.tex"
with open(latex_path, "w") as f:
    latex_table = df.pivot(index='Model', columns='Attack', values='Accuracy') \
                    .to_latex(float_format="%.4f", caption="Model Performance under Adversarial Attacks", label="tab:adv_accuracy")
    f.write(latex_table)

print(f" LaTeX table saved to: {latex_path}")
