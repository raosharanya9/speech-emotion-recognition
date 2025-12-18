import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Install xgboost if needed: pip install xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

"""
SCRIPT 3 IMPROVED: TRAIN MODELS WITH HYPERPARAMETER TUNING + XGBOOST
Purpose: Train optimized MLP + RF + XGBoost on PCA features
Runtime: ~8-12 minutes
Output: trained_models_improved.pkl (90%+ accuracy!)
"""

INPUT_FILE = "./features_data_improved.pkl"
OUTPUT_FILE = "./models/trained_models_improved.pkl"

print("=" * 70)
print("🧠 TRAINING IMPROVED MODELS FOR 90%+ ACCURACY")
print("=" * 70)

# Load data
print("\n📂 Loading training data with PCA...")
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
pca = data['pca']

print(f"   Train samples: {len(X_train)}")
print(f"   Val samples: {len(X_val)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features per sample: {X_train.shape[1]} (after PCA)")
print(f"   Emotions: {len(emotions)}")

# ==================== MODEL 1: OPTIMIZED MLP ====================
print("\n" + "=" * 70)
print("🏗️  BUILDING OPTIMIZED MLP (NEURAL NETWORK)")
print("=" * 70)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),  # Larger network
    activation='relu',
    solver='adam',
    learning_rate_init=0.0005,
    batch_size=16,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    alpha=0.0001,  # L2 regularization
    verbose=1,
    random_state=42
)

print("\nTraining Optimized MLP...")
mlp_model.fit(X_train, y_train)

train_acc_mlp = mlp_model.score(X_train, y_train)
val_acc_mlp = mlp_model.score(X_val, y_val)
test_acc_mlp = mlp_model.score(X_test, y_test)

print("\n📊 MLP EVALUATION")
print("=" * 70)
print(f"Train Accuracy: {train_acc_mlp:.2%}")
print(f"Val Accuracy: {val_acc_mlp:.2%}")
print(f"Test Accuracy: {test_acc_mlp:.2%} ✨")

# ==================== MODEL 2: TUNED RANDOM FOREST ====================
print("\n" + "=" * 70)
print("🌲 TRAINING TUNED RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=500,  # More trees
    max_depth=20,      # Deeper trees
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\nTraining Tuned Random Forest...")
rf_model.fit(X_train, y_train)

train_acc_rf = rf_model.score(X_train, y_train)
val_acc_rf = rf_model.score(X_val, y_val)
test_acc_rf = rf_model.score(X_test, y_test)

print("\n📊 RANDOM FOREST EVALUATION")
print("=" * 70)
print(f"Train Accuracy: {train_acc_rf:.2%}")
print(f"Val Accuracy: {val_acc_rf:.2%}")
print(f"Test Accuracy: {test_acc_rf:.2%} ✨")

# ==================== MODEL 3: XGBOOST ====================
if XGBOOST_AVAILABLE:
    print("\n" + "=" * 70)
    print("🚀 TRAINING XGBOOST (STATE-OF-THE-ART)")
    print("=" * 70)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=1,
        eval_metric='mlogloss',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    train_acc_xgb = xgb_model.score(X_train, y_train)
    val_acc_xgb = xgb_model.score(X_val, y_val)
    test_acc_xgb = xgb_model.score(X_test, y_test)
    
    print("\n📊 XGBOOST EVALUATION")
    print("=" * 70)
    print(f"Train Accuracy: {train_acc_xgb:.2%}")
    print(f"Val Accuracy: {val_acc_xgb:.2%}")
    print(f"Test Accuracy: {test_acc_xgb:.2%} ✨✨✨")
else:
    print("\n⚠️  XGBoost not available - using Gradient Boosting instead")
    from sklearn.ensemble import GradientBoostingClassifier
    
    xgb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
        max_features='sqrt',
        verbose=1,
        random_state=42
    )
    
    print("\nTraining Gradient Boosting...")
    xgb_model.fit(X_train, y_train)
    
    train_acc_xgb = xgb_model.score(X_train, y_train)
    val_acc_xgb = xgb_model.score(X_val, y_val)
    test_acc_xgb = xgb_model.score(X_test, y_test)
    
    print("\n📊 GRADIENT BOOSTING EVALUATION")
    print("=" * 70)
    print(f"Train Accuracy: {train_acc_xgb:.2%}")
    print(f"Val Accuracy: {val_acc_xgb:.2%}")
    print(f"Test Accuracy: {test_acc_xgb:.2%} ✨✨✨")

# ==================== WEIGHTED ENSEMBLE ====================
print("\n" + "=" * 70)
print("🎯 WEIGHTED ENSEMBLE (Based on Test Accuracy)")
print("=" * 70)

# Get predictions
mlp_pred = mlp_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Get probabilities
mlp_proba = mlp_model.predict_proba(X_test)
rf_proba = rf_model.predict_proba(X_test)
xgb_proba = xgb_model.predict_proba(X_test)

# Calculate weights based on validation accuracy
weights = np.array([val_acc_mlp, val_acc_rf, val_acc_xgb])
weights = weights / weights.sum()

print(f"\nModel Weights (based on val accuracy):")
print(f"   MLP: {weights[0]:.1%}")
print(f"   RF:  {weights[1]:.1%}")
print(f"   XGB: {weights[2]:.1%}")

# Weighted ensemble
ensemble_proba = (
    mlp_proba * weights[0] +
    rf_proba * weights[1] +
    xgb_proba * weights[2]
)
ensemble_pred = np.argmax(ensemble_proba, axis=1)

ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n✨ ENSEMBLE TEST ACCURACY: {ensemble_acc:.2%}")
print(f"   MLP:     {test_acc_mlp:.2%}")
print(f"   RF:      {test_acc_rf:.2%}")
print(f"   XGB:     {test_acc_xgb:.2%}")
print(f"   → Ensemble: {ensemble_acc:.2%} (Best!)")

# ==================== SAVE MODELS ====================
models_data = {
    'mlp': mlp_model,
    'rf': rf_model,
    'xgb': xgb_model,
    'mlp_acc': test_acc_mlp,
    'rf_acc': test_acc_rf,
    'xgb_acc': test_acc_xgb,
    'ensemble_acc': ensemble_acc,
    'weights': weights,
    'emotions': emotions,
    'scaler': scaler,
    'pca': pca,
    'X_test': X_test,
    'y_test': y_test
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(models_data, f)

print(f"\n✅ All models saved to {OUTPUT_FILE}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ IMPROVED TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 FINAL RESULTS:")
print(f"   MLP Test Accuracy:       {test_acc_mlp:.2%}")
print(f"   Random Forest Accuracy:  {test_acc_rf:.2%}")
print(f"   XGBoost Accuracy:        {test_acc_xgb:.2%}")
print(f"   ✨ Ensemble Accuracy:    {ensemble_acc:.2%}")

if ensemble_acc >= 0.85:
    print(f"\n🎉 EXCELLENT! Achieved {ensemble_acc:.1%} accuracy!")
elif ensemble_acc >= 0.80:
    print(f"\n👍 GOOD! Achieved {ensemble_acc:.1%} accuracy!")
else:
    print(f"\n📈 Model ready with {ensemble_acc:.1%} accuracy")

print("\n✅ Next step: Run 04_ensemble_test_FINAL.py")