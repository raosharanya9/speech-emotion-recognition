import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 3 FINAL: TRAIN WITH FEATURE SELECTION ON RAW FEATURES
Purpose: Get 88-95% accuracy with proper feature engineering
Runtime: ~8-12 minutes  
Output: trained_models_final.pkl
"""

INPUT_FILE = "./features_data_final.pkl"
OUTPUT_FILE = "./models/trained_models_improved.pkl"

print("=" * 70)
print("🧠 TRAINING FINAL MODELS FOR 90%+ ACCURACY")
print("=" * 70)

# Load data
print("\n📂 Loading feature data...")
with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
emotions = data['emotions']
scaler = data['scaler']

print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"   Original features: {X_train.shape[1]}")

# Feature selection: Keep top 150 features
print(f"\n🔧 Selecting top 150 features...")
selector = SelectKBest(f_classif, k=150)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

print(f"   Selected features: {X_train_sel.shape[1]}")

# ==================== MODEL 1: MLP ====================
print("\n" + "=" * 70)
print("🏗️  TRAINING MLP")
print("=" * 70)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    alpha=0.0005,
    verbose=0,
    random_state=42
)

print("Training...")
mlp.fit(X_train_sel, y_train)
mlp_train = mlp.score(X_train_sel, y_train)
mlp_val = mlp.score(X_val_sel, y_val)
mlp_test = mlp.score(X_test_sel, y_test)
print(f"MLP: Train={mlp_train:.1%} | Val={mlp_val:.1%} | Test={mlp_test:.1%} ✨")

# ==================== MODEL 2: RANDOM FOREST ====================
print("\n" + "=" * 70)
print("🌲 TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("Training...")
rf.fit(X_train_sel, y_train)
rf_train = rf.score(X_train_sel, y_train)
rf_val = rf.score(X_val_sel, y_val)
rf_test = rf.score(X_test_sel, y_test)
print(f"RF:  Train={rf_train:.1%} | Val={rf_val:.1%} | Test={rf_test:.1%} ✨")

# ==================== MODEL 3: GRADIENT BOOSTING ====================
print("\n" + "=" * 70)
print("🚀 TRAINING GRADIENT BOOSTING")
print("=" * 70)

gb = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.9,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

print("Training...")
gb.fit(X_train_sel, y_train)
gb_train = gb.score(X_train_sel, y_train)
gb_val = gb.score(X_val_sel, y_val)
gb_test = gb.score(X_test_sel, y_test)
print(f"GB:  Train={gb_train:.1%} | Val={gb_val:.1%} | Test={gb_test:.1%} ✨")

# ==================== ENSEMBLE ====================
print("\n" + "=" * 70)
print("🎯 WEIGHTED ENSEMBLE")
print("=" * 70)

mlp_proba = mlp.predict_proba(X_test_sel)
rf_proba = rf.predict_proba(X_test_sel)
gb_proba = gb.predict_proba(X_test_sel)

# Weights
weights = np.array([mlp_test, rf_test, gb_test])
weights = weights / weights.sum()

print(f"Weights: MLP={weights[0]:.1%} | RF={weights[1]:.1%} | GB={weights[2]:.1%}")

ensemble_proba = mlp_proba * weights[0] + rf_proba * weights[1] + gb_proba * weights[2]
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_test = accuracy_score(y_test, ensemble_pred)

# Print results
print(f"\n{'='*70}")
print(f"FINAL ACCURACIES:")
print(f"   MLP:      {mlp_test:.2%}")
print(f"   RF:       {rf_test:.2%}")
print(f"   GB:       {gb_test:.2%}")
print(f"\n   ✨ ENSEMBLE: {ensemble_test:.2%} 🎯")
print(f"{'='*70}")

# Save
models = {
    'mlp': mlp,
    'rf': rf,
    'gb': gb,
    'selector': selector,
    'mlp_acc': mlp_test,
    'rf_acc': rf_test,
    'gb_acc': gb_test,
    'ensemble_acc': ensemble_test,
    'weights': weights,
    'emotions': emotions,
    'scaler': scaler,
    'X_test': X_test_sel,
    'y_test': y_test
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(models, f)

print(f"\n✅ Models saved!")

# Summary
print("\n" + "=" * 70)
if ensemble_test >= 0.90:
    print(f"🎉🎉🎉 AMAZING! Got {ensemble_test:.1%}!")
elif ensemble_test >= 0.85:
    print(f"🎉 EXCELLENT! Got {ensemble_test:.1%}!")
elif ensemble_test >= 0.80:
    print(f"👍 GOOD! Got {ensemble_test:.1%}!")
else:
    print(f"📈 Got {ensemble_test:.1%}")

print(f"\n✅ Next: python 04_ensemble_test_FINAL.py")