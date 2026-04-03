"""
Task 03: Decision Tree Classifier — Bank Marketing Dataset
Predict whether a customer will subscribe to a term deposit (y: yes/no).
Dataset: UCI Bank Marketing (simulated here; replace with real CSV as needed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load / Simulate Dataset ───────────────────────────────────────────────
print("=" * 60)
print("TASK 03 — Decision Tree Classifier (Bank Marketing)")
print("=" * 60)

try:
    df = pd.read_csv('bank.csv', sep=';')
    print("Loaded bank.csv")
except FileNotFoundError:
    np.random.seed(42)
    n = 4521
    df = pd.DataFrame({
        'age':       np.random.randint(18, 80, n),
        'job':       np.random.choice(['admin.','technician','services','management',
                                       'retired','blue-collar','unemployed','entrepreneur',
                                       'housemaid','self-employed','student','unknown'], n),
        'marital':   np.random.choice(['married','single','divorced'], n, p=[0.6, 0.28, 0.12]),
        'education': np.random.choice(['primary','secondary','tertiary','unknown'], n, p=[0.15, 0.51, 0.3, 0.04]),
        'default':   np.random.choice(['yes','no'], n, p=[0.015, 0.985]),
        'balance':   np.random.randint(-500, 10000, n),
        'housing':   np.random.choice(['yes','no'], n, p=[0.56, 0.44]),
        'loan':      np.random.choice(['yes','no'], n, p=[0.16, 0.84]),
        'contact':   np.random.choice(['cellular','telephone','unknown'], n, p=[0.65, 0.15, 0.2]),
        'day':       np.random.randint(1, 31, n),
        'month':     np.random.choice(['jan','feb','mar','apr','may','jun',
                                       'jul','aug','sep','oct','nov','dec'], n),
        'duration':  np.random.randint(0, 3600, n),
        'campaign':  np.random.randint(1, 50, n),
        'pdays':     np.where(np.random.rand(n) < 0.82, -1, np.random.randint(0, 400, n)),
        'previous':  np.random.randint(0, 10, n),
        'poutcome':  np.random.choice(['unknown','failure','success','other'], n, p=[0.81, 0.1, 0.05, 0.04]),
        'y':         np.random.choice(['yes','no'], n, p=[0.117, 0.883]),
    })
    print("Using simulated Bank Marketing dataset (n=4521).")

print(f"\n── Shape: {df.shape}")
print("── Target distribution:\n", df['y'].value_counts())

# ── 2. Data Preparation ───────────────────────────────────────────────────────
df_enc = df.copy()
le = LabelEncoder()
cat_cols = df_enc.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

X = df_enc.drop('y', axis=1)
y = df_enc['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n── Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 3. Train Decision Tree ────────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                             min_samples_leaf=5, random_state=42)
dt.fit(X_train, y_train)

y_pred  = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)

print(f"\n── Test Accuracy: {acc * 100:.2f}%")
print("\n── Classification Report:\n", classification_report(y_test, y_pred, target_names=['No','Yes']))

cv_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"── 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── 4. Visualisations ─────────────────────────────────────────────────────────
BG = '#1e1e2e'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'text.color': 'white', 'axes.labelcolor': 'white',
                     'xtick.color': 'white', 'ytick.color': 'white',
                     'axes.edgecolor': '#444'})

fig = plt.figure(figsize=(20, 18))

# 4a. Decision Tree (top 3 levels)
ax1 = fig.add_subplot(2, 2, (1, 2))
ax1.set_facecolor('white')
plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, max_depth=3,
          fontsize=7, ax=ax1,
          impurity=True, proportion=False)
ax1.set_title('Decision Tree (max_depth=3 shown)',
              color='white', fontsize=13, fontweight='bold', pad=10,
              bbox=dict(facecolor=BG, edgecolor='none'))

# 4b. Confusion Matrix
ax2 = fig.add_subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('Confusion Matrix', fontweight='bold')
ax2.set_facecolor(BG)

# 4c. Feature Importances
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_facecolor(BG)
importances = pd.Series(dt.feature_importances_, index=X.columns)
top10 = importances.nlargest(10).sort_values()
colors_fi = ['#ffd54f' if v == top10.max() else '#4fc3f7' for v in top10.values]
ax3.barh(top10.index, top10.values, color=colors_fi, edgecolor='white', linewidth=0.3)
ax3.set_title('Top 10 Feature Importances', fontweight='bold')
ax3.set_xlabel('Importance Score')
ax3.xaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax3.set_axisbelow(True)

fig.suptitle(f'Task 03 — Decision Tree Classifier  |  Test Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('task03_output.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("\nVisualisation saved → task03_output.png")

# Save predictions
results = X_test.copy()
results['actual']    = y_test.values
results['predicted'] = y_pred
results['prob_yes']  = y_proba.round(4)
results.to_csv('task03_predictions.csv', index=False)
print("Predictions saved → task03_predictions.csv")
