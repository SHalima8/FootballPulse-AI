import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv('../data/processed/model_features.csv')
print(f"Data loaded: {df.shape}")

# ── 2. Split Features and Target ─────────────────────────
X = df.drop('result', axis=1)
y = df['result']

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# ── 3. Train Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 4. Train Random Forest ────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_train_acc = rf_model.score(X_train, y_train)
rf_test_acc  = rf_model.score(X_test,  y_test)
rf_cv        = cross_val_score(rf_model, X, y, cv=5)

# ── 5. Train Logistic Regression ──────────────────────────
print("Training Logistic Regression...")
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, random_state=42))
])
lr_model.fit(X_train, y_train)
lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc  = lr_model.score(X_test,  y_test)
lr_cv        = cross_val_score(lr_model, X, y, cv=5)

# ── 6. Train XGBoost ──────────────────────────────────────
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_train_acc = xgb_model.score(X_train, y_train)
xgb_test_acc  = xgb_model.score(X_test,  y_test)
xgb_cv        = cross_val_score(xgb_model, X, y, cv=5)

# ── 7. Full Diagnosis ─────────────────────────────────────
print("\n" + "="*55)
print("FULL MODEL DIAGNOSIS")
print("="*55)

all_models = [
    ("Random Forest",       rf_train_acc,  rf_test_acc,  rf_cv,  rf_model),
    ("Logistic Regression", lr_train_acc,  lr_test_acc,  lr_cv,  lr_model),
    ("XGBoost",             xgb_train_acc, xgb_test_acc, xgb_cv, xgb_model),
]

for name, train, test, cv, _ in all_models:
    gap = train - test
    if train < 0.55 and test < 0.55:
        verdict = "UNDERFIT"
    elif gap > 0.15:
        verdict = "OVERFIT"
    elif gap < 0:
        verdict = "HEALTHY — test above train"
    else:
        verdict = "HEALTHY"

    print(f"\n{name}:")
    print(f"  Train    : {train:.2%}")
    print(f"  Test     : {test:.2%}")
    print(f"  Gap      : {gap:.2%}  → {verdict}")
    print(f"  CV Mean  : {cv.mean():.2%}")
    print(f"  CV Std   : {cv.std():.2%}")

# ── 8. Pick Best Model by CV Mean ────────────────────────
print("\n" + "="*55)
print("MODEL COMPARISON — CV MEAN IS THE JUDGE")
print("="*55)

best_name, best_score, best_model = max(
    [(name, cv.mean(), m) for name, _, _, cv, m in all_models],
    key=lambda x: x[1]
)
best_preds = best_model.predict(X_test)

for name, _, _, cv, _ in all_models:
    marker = " ← WINNER" if name == best_name else ""
    print(f"  {name:<25} CV: {cv.mean():.2%}{marker}")

print(f"\nBest model: {best_name} ({best_score:.2%})")

# ── 9. Classification Report ──────────────────────────────
print("\n" + "="*55)
print(f"CLASSIFICATION REPORT — {best_name}")
print("="*55)
print(classification_report(
    y_test, best_preds,
    target_names=['Draw', 'Team A Win', 'Team B Win']
))

# ── 10. Feature Importance ────────────────────────────────
print("="*55)
print("FEATURE IMPORTANCE")
print("="*55)

if best_name == "Random Forest":
    importances = rf_model.feature_importances_
elif best_name == "XGBoost":
    importances = xgb_model.feature_importances_
else:
    importances = None

if importances is not None:
    feat_imp = sorted(
        zip(X.columns, importances),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<25} {imp:.2%}  {bar}")
else:
    # LR coefficients
    coefs = np.abs(lr_model.named_steps['lr'].coef_).mean(axis=0)
    feat_imp = sorted(zip(X.columns, coefs), key=lambda x: x[1], reverse=True)
    for feat, coef in feat_imp:
        bar = "█" * int(coef * 10)
        print(f"  {feat:<25} {coef:.3f}  {bar}")

# ── 11. Save Everything ───────────────────────────────────
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model,         '../models/model.pkl')
joblib.dump(best_name,          '../models/model_name.pkl')
joblib.dump(X.columns.tolist(), '../models/feature_columns.pkl')

# Save label encoder
from sklearn.preprocessing import LabelEncoder
clean_df  = pd.read_csv('../data/processed/clean_matches.csv')
le        = LabelEncoder()
all_teams = pd.concat([clean_df['Home Team Name'], clean_df['Away Team Name']])
le.fit(all_teams)
joblib.dump(le, '../models/label_encoder.pkl')

print(f"\nModel saved       : models/model.pkl ({best_name}) ✅")
print(f"Feature list saved: models/feature_columns.pkl ✅")
print(f"Label encoder saved: models/label_encoder.pkl ✅")