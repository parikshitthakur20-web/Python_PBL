import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

df = pd.read_csv("smartphone_sensor_data.csv")

print("=" * 60)
print("SMARTPHONE SENSOR DATA - ML CLASSIFICATION SYSTEM")
print("=" * 60)
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values before cleaning:\n", df.isnull().sum())

numeric_cols = ["screen_time", "battery_usage", "touch_frequency", "motion_activity", "device_activity"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:\n", df.isnull().sum())

le = LabelEncoder()
df["usage_encoded"] = le.fit_transform(df["usage"])

print("\nLabel Encoding: Normal =", le.transform(["Normal"])[0], 
      "| High =", le.transform(["High"])[0])

X = df[["screen_time", "battery_usage", "touch_frequency", 
        "motion_activity", "device_activity"]]
y = df["usage_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, average="weighted"), 4)
    }
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, metrics in results.items():
    print(f"\n{name}")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value}")
    print(f"  Confusion Matrix:\n{conf_matrices[name]}")

results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string())

best_model = results_df["Accuracy"].idxmax()
print(f"\nBest Model by Accuracy: {best_model} ({results_df.loc[best_model, 'Accuracy']})")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
class_labels = le.classes_

for idx, (name, cm) in enumerate(conf_matrices.items()):
    ax = axes[idx]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name}\nAcc: {results[name]['Accuracy']}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)

axes[5].axis("off")
metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(models))
width = 0.18

fig2, ax2 = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics_list):
    values = [results[m][metric] for m in models]
    ax2.bar(x + i * width, values, width, label=metric)

ax2.set_xlabel("Models")
ax2.set_ylabel("Score")
ax2.set_title("Model Performance Comparison")
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(list(models.keys()), rotation=15, ha="right")
ax2.set_ylim(0, 1.1)
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig("confusion_matrices.png", dpi=120, bbox_inches="tight")
fig2.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

print("\nPlots saved: confusion_matrices.png | model_comparison.png")
print("Done.")
