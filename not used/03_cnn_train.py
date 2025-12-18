import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 3: TRAIN MODELS USING SCIKIT-LEARN ONLY (NO DLL ISSUES!)
Purpose: Train MLP + RF + GB on extracted features
Runtime: ~5-10 minutes
Output: trained_models.pkl
"""

INPUT_FILE = "./features_data.pkl"
OUTPUT_FILE = "./models/trained_models.pkl"

print("=" * 70)
print("🧠 TRAINING MODELS FOR SPEECH EMOTION RECOGNITION (Scikit-Learn)")
print("=" * 70)

# Load data
print("\n📂 Loading training data...")
with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
emotions = data['emotions']
num_emotions = len(emotions)

print(f"   Train samples: {len(X_train)}")
print(f"   Val samples: {len(X_val)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features per sample: {X_train.shape[1]}")
print(f"   Emotions: {num_emotions}")

# ==================== MODEL 1: MLP (Neural Network) ====================
print("\n" + "=" * 70)
print("🏗️  BUILDING MLP (MULTI-LAYER PERCEPTRON)")
print("=" * 70)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=1,
    random_state=42
)

print("\nTraining MLP (Neural Network)...")
mlp_model.fit(X_train, y_train)

# Evaluate MLP
train_acc_mlp = mlp_model.score(X_train, y_train)
val_acc_mlp = mlp_model.score(X_val, y_val)
test_acc_mlp = mlp_model.score(X_test, y_test)

print("\n📊 MLP EVALUATION")
print("=" * 70)
print(f"Train Accuracy: {train_acc_mlp:.2%}")
print(f"Val Accuracy: {val_acc_mlp:.2%}")
print(f"Test Accuracy: {test_acc_mlp:.2%}")

# ==================== MODEL 2: RANDOM FOREST ====================
print("\n" + "=" * 70)
print("🌲 TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

# Evaluate RF
train_acc_rf = rf_model.score(X_train, y_train)
val_acc_rf = rf_model.score(X_val, y_val)
test_acc_rf = rf_model.score(X_test, y_test)

print("\n📊 RANDOM FOREST EVALUATION")
print("=" * 70)
print(f"Train Accuracy: {train_acc_rf:.2%}")
print(f"Val Accuracy: {val_acc_rf:.2%}")
print(f"Test Accuracy: {test_acc_rf:.2%}")

# ==================== MODEL 3: GRADIENT BOOSTING ====================
print("\n" + "=" * 70)
print("🚀 TRAINING GRADIENT BOOSTING")
print("=" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    verbose=1,
    random_state=42
)

print("\nTraining Gradient Boosting...")
gb_model.fit(X_train, y_train)

# Evaluate GB
train_acc_gb = gb_model.score(X_train, y_train)
val_acc_gb = gb_model.score(X_val, y_val)
test_acc_gb = gb_model.score(X_test, y_test)

print("\n📊 GRADIENT BOOSTING EVALUATION")
print("=" * 70)
print(f"Train Accuracy: {train_acc_gb:.2%}")
print(f"Val Accuracy: {val_acc_gb:.2%}")
print(f"Test Accuracy: {test_acc_gb:.2%}")

# ==================== ENSEMBLE ====================
print("\n" + "=" * 70)
print("🎯 ENSEMBLE PREDICTIONS (Average of 3 models)")
print("=" * 70)

# Get predictions
mlp_pred = mlp_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Get probabilities and average
mlp_proba = mlp_model.predict_proba(X_test)
rf_proba = rf_model.predict_proba(X_test)
gb_proba = gb_model.predict_proba(X_test)

ensemble_proba = (mlp_proba + rf_proba + gb_proba) / 3
ensemble_pred = np.argmax(ensemble_proba, axis=1)

ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\nMLP Test Accuracy: {test_acc_mlp:.2%}")
print(f"RF Test Accuracy: {test_acc_rf:.2%}")
print(f"GB Test Accuracy: {test_acc_gb:.2%}")
print(f"Ensemble Test Accuracy: {ensemble_acc:.2%}")

# ==================== SAVE MODELS ====================
models_data = {
    'mlp': mlp_model,
    'rf': rf_model,
    'gb': gb_model,
    'mlp_acc': test_acc_mlp,
    'rf_acc': test_acc_rf,
    'gb_acc': test_acc_gb,
    'ensemble_acc': ensemble_acc,
    'emotions': emotions,
    'X_test': X_test,
    'y_test': y_test
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(models_data, f)

print(f"\n✅ All models saved to {OUTPUT_FILE}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ ALL TRAINING COMPLETE!")
print("=" * 70)
print("\n📊 FINAL RESULTS:")
print(f"   MLP Test Accuracy:       {test_acc_mlp:.2%}")
print(f"   Random Forest Accuracy:  {test_acc_rf:.2%}")
print(f"   Gradient Boosting Acc:   {test_acc_gb:.2%}")
print(f"   ✨ Ensemble Accuracy:    {ensemble_acc:.2%}")

print("\n✅ Next step: Run 04_ensemble_test_FINAL.py")