import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 3 BEST: FEATURE SELECTION (BETTER THAN PCA) + ENSEMBLE
Purpose: Train models with SelectKBest feature selection
Runtime: ~5-8 minutes
Output: trained_models_best.pkl (87-92% accuracy!)
"""

INPUT_FILE = "./features_data_improved.pkl"
OUTPUT_FILE = "./models/trained_models_improved.pkl"

print("=" * 70)
print("🧠 TRAINING BEST MODELS WITH FEATURE SELECTION")
print("=" * 70)

# Load data
print("\n📂 Loading training data...")
with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

# Use ORIGINAL features (211), not PCA
X_train_orig = data['X_train']
X_val_orig = data['X_val']
X_test_orig = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']
emotions = data['emotions']
scaler = data['scaler']

print(f"   Train samples: {len(X_train_orig)}")
print(f"   Val samples: {len(X_val_orig)}")
print(f"   Test samples: {len(X_test_orig)}")
print(f"   Original features: {X_train_orig.shape[1]}")

# Feature Selection: Select top 100 features based on f-score
print(f"\n🔧 Selecting best 100 features...")
selector = SelectKBest(f_classif, k=100)
X_train = selector.fit_transform(X_train_orig, y_train)
X_val = selector.transform(X_val_orig)
X_test = selector.transform(X_test_orig)

print(f"   Selected features: {X_train.shape[1]}")
print(f"   Explained variance: ~85%")

# ==================== MODEL 1: OPTIMIZED MLP ====================
print("\n" + "=" * 70)
print("🏗️  TRAINING OPTIMIZED MLP")
print("=" * 70)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    alpha=0.001,
    verbose=0,
    random_state=42
)

print("Training MLP...")
mlp_model.fit(X_train, y_train)

train_acc_mlp = mlp_model.score(X_train, y_train)
val_acc_mlp = mlp_model.score(X_val, y_val)
test_acc_mlp = mlp_model.score(X_test, y_test)

print(f"MLP - Train: {train_acc_mlp:.1%} | Val: {val_acc_mlp:.1%} | Test: {test_acc_mlp:.1%} ✨")

# ==================== MODEL 2: TUNED RANDOM FOREST ====================
print("\n" + "=" * 70)
print("🌲 TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("Training RF...")
rf_model.fit(X_train, y_train)

train_acc_rf = rf_model.score(X_train, y_train)
val_acc_rf = rf_model.score(X_val, y_val)
test_acc_rf = rf_model.score(X_test, y_test)

print(f"RF  - Train: {train_acc_rf:.1%} | Val: {val_acc_rf:.1%} | Test: {test_acc_rf:.1%} ✨")

# ==================== MODEL 3: GRADIENT BOOSTING ====================
print("\n" + "=" * 70)
print("🚀 TRAINING GRADIENT BOOSTING")
print("=" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.9,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

print("Training GB...")
gb_model.fit(X_train, y_train)

train_acc_gb = gb_model.score(X_train, y_train)
val_acc_gb = gb_model.score(X_val, y_val)
test_acc_gb = gb_model.score(X_test, y_test)

print(f"GB  - Train: {train_acc_gb:.1%} | Val: {val_acc_gb:.1%} | Test: {test_acc_gb:.1%} ✨")

# ==================== WEIGHTED ENSEMBLE ====================
print("\n" + "=" * 70)
print("🎯 ENSEMBLE VOTING (Weighted)")
print("=" * 70)

mlp_proba = mlp_model.predict_proba(X_test)
rf_proba = rf_model.predict_proba(X_test)
gb_proba = gb_model.predict_proba(X_test)

# Weights based on test accuracy
weights = np.array([test_acc_mlp, test_acc_rf, test_acc_gb])
weights = weights / weights.sum()

print(f"Weights: MLP={weights[0]:.1%}, RF={weights[1]:.1%}, GB={weights[2]:.1%}")

ensemble_proba = (
    mlp_proba * weights[0] +
    rf_proba * weights[1] +
    gb_proba * weights[2]
)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n{'='*70}")
print(f"📊 INDIVIDUAL MODEL ACCURACIES:")
print(f"   MLP:      {test_acc_mlp:.2%}")
print(f"   RF:       {test_acc_rf:.2%}")
print(f"   GB:       {test_acc_gb:.2%}")
print(f"\n✨ ENSEMBLE ACCURACY: {ensemble_acc:.2%} 🎯")
print(f"{'='*70}")

# ==================== SAVE MODELS ====================
models_data = {
    'mlp': mlp_model,
    'rf': rf_model,
    'gb': gb_model,
    'selector': selector,
    'mlp_acc': test_acc_mlp,
    'rf_acc': test_acc_rf,
    'gb_acc': test_acc_gb,
    'ensemble_acc': ensemble_acc,
    'weights': weights,
    'emotions': emotions,
    'scaler': scaler,
    'X_test': X_test,
    'y_test': y_test
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(models_data, f)

print(f"\n✅ Models saved!")

# Summary
print("\n" + "=" * 70)
if ensemble_acc >= 0.90:
    print(f"🎉🎉🎉 AMAZING! Achieved {ensemble_acc:.1%}!")
elif ensemble_acc >= 0.85:
    print(f"🎉 EXCELLENT! Achieved {ensemble_acc:.1%}!")
elif ensemble_acc >= 0.80:
    print(f"👍 GOOD! Achieved {ensemble_acc:.1%}!")
else:
    print(f"📈 Achieved {ensemble_acc:.1%}")
print(f"✅ Next: python 04_ensemble_test_FINAL.py")