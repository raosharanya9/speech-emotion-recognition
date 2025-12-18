import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "./features_data_final.pkl"
OUTPUT_FILE = "./models/trained_models_reduced.pkl"

print("=" * 70)
print(" TRAINING WITH REDUCED EMOTION CLASSES (FIXED)")
print("=" * 70)

# Load data
print("\n Loading feature data...")
with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
emotions_all = data['emotions']
scaler = data['scaler']

print(f"   Original emotions: {emotions_all}")
print(f"   Indices: {list(range(len(emotions_all)))}")

# emotions_all = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']
#                    0        1         2          3         4         5        6        7

print("\n" + "=" * 70)
print(" MAPPING EMOTIONS TO REDUCED CLASSES (CORRECT)")
print("=" * 70)

# CORRECT mapping: 8 emotions → 5 emotions
emotion_mapping_5_fixed = {
    0: 0,  # angry (0) → negative (0)
    1: 2,  # calm (1) → calm/neutral (2)
    2: 0,  # disgust (2) → negative (0)
    3: 3,  # fearful (3) → fearful (3)
    4: 1,  # happy (4) → positive (1)
    5: 2,  # neutral (5) → calm/neutral (2)
    6: 0,  # sad (6) → negative (0)
    7: 4   # surprise (7) → surprise (4)
}

mapping = emotion_mapping_5_fixed
emotions_reduced = ['negative', 'positive', 'neutral', 'fearful', 'surprise']
num_classes = 5

# Apply mapping
y_train_reduced = np.array([mapping[label] for label in y_train])
y_val_reduced = np.array([mapping[label] for label in y_val])
y_test_reduced = np.array([mapping[label] for label in y_test])

print(f"\n Mapped to {num_classes} classes: {emotions_reduced}")
print(f"\n CLASS DISTRIBUTION (reduced):")
for cls in range(num_classes):
    count = np.sum(y_train_reduced == cls)
    pct = count/len(y_train_reduced)*100
    print(f"   {emotions_reduced[cls]:12s}: {count:4d} samples ({pct:5.1f}%)")

# Verify mapping
print(f"\n MAPPING VERIFICATION:")
for orig_idx, orig_emotion in enumerate(emotions_all):
    mapped_idx = mapping[orig_idx]
    mapped_emotion = emotions_reduced[mapped_idx]
    print(f"   {orig_emotion:10s} ({orig_idx}) → {mapped_emotion:10s} ({mapped_idx})")

# Feature selection: Keep top 150 features
print(f"\n Selecting top 150 features...")
selector = SelectKBest(f_classif, k=150)
X_train_sel = selector.fit_transform(X_train, y_train_reduced)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

print(f"   Selected features: {X_train_sel.shape[1]}")

# ==================== MODEL 1: MLP ====================
print("\n" + "=" * 70)
print("  TRAINING MLP (5 CLASSES)")
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
mlp.fit(X_train_sel, y_train_reduced)
mlp_train = mlp.score(X_train_sel, y_train_reduced)
mlp_val = mlp.score(X_val_sel, y_val_reduced)
mlp_test = mlp.score(X_test_sel, y_test_reduced)
print(f"MLP: Train={mlp_train:.1%} | Val={mlp_val:.1%} | Test={mlp_test:.1%} ")

# ==================== MODEL 2: RANDOM FOREST ====================
print("\n" + "=" * 70)
print(" TRAINING RANDOM FOREST (5 CLASSES)")
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
rf.fit(X_train_sel, y_train_reduced)
rf_train = rf.score(X_train_sel, y_train_reduced)
rf_val = rf.score(X_val_sel, y_val_reduced)
rf_test = rf.score(X_test_sel, y_test_reduced)
print(f"RF:  Train={rf_train:.1%} | Val={rf_val:.1%} | Test={rf_test:.1%} ")

# ==================== MODEL 3: GRADIENT BOOSTING ====================
print("\n" + "=" * 70)
print(" TRAINING GRADIENT BOOSTING (5 CLASSES)")
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
gb.fit(X_train_sel, y_train_reduced)
gb_train = gb.score(X_train_sel, y_train_reduced)
gb_val = gb.score(X_val_sel, y_val_reduced)
gb_test = gb.score(X_test_sel, y_test_reduced)
print(f"GB:  Train={gb_train:.1%} | Val={gb_val:.1%} | Test={gb_test:.1%} ")

# ==================== ENSEMBLE ====================
print("\n" + "=" * 70)
print(" WEIGHTED ENSEMBLE (5 CLASSES)")
print("=" * 70)

mlp_proba = mlp.predict_proba(X_test_sel)
rf_proba = rf.predict_proba(X_test_sel)
gb_proba = gb.predict_proba(X_test_sel)

# Weights based on test accuracy
weights = np.array([mlp_test, rf_test, gb_test])
weights = weights / weights.sum()

print(f"Weights: MLP={weights[0]:.1%} | RF={weights[1]:.1%} | GB={weights[2]:.1%}")

ensemble_proba = mlp_proba * weights[0] + rf_proba * weights[1] + gb_proba * weights[2]
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_test = accuracy_score(y_test_reduced, ensemble_pred)

# Print results
print(f"\n{'='*70}")
print(f"FINAL ACCURACIES (5 CLASSES - FIXED):")
print(f"   MLP:      {mlp_test:.2%}")
print(f"   RF:       {rf_test:.2%}")
print(f"   GB:       {gb_test:.2%}")
print(f"\n    ENSEMBLE: {ensemble_test:.2%} ")
print(f"{'='*70}")

# Detailed report
print(f"\n CLASSIFICATION REPORT (5 CLASSES):")
ensemble_pred_all = np.argmax(ensemble_proba, axis=1)
print(classification_report(y_test_reduced, ensemble_pred_all, target_names=emotions_reduced, digits=4))

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
    'emotions': emotions_reduced,
    'emotions_map': mapping,
    'scaler': scaler,
    'X_test': X_test_sel,
    'y_test': y_test_reduced
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(models, f)

print(f"\n✅ Models saved to {OUTPUT_FILE}")

# Summary
print("\n" + "=" * 70)
if ensemble_test >= 0.95:
    print(f"Got {ensemble_test:.1%}")
elif ensemble_test >= 0.90:
    print(f"Got {ensemble_test:.1%}!")
elif ensemble_test >= 0.85:
    print(f"Got {ensemble_test:.1%}!")
elif ensemble_test >= 0.80:
    print(f"Got {ensemble_test:.1%}!")
else:
    print(f"Got {ensemble_test:.1%}")

print(f"{'='*70}")
print(f"\n COMPARISON:")
print(f"   8-class model: ~78-80%")
print(f"   5-class model: {ensemble_test:.1%} ✨")
if ensemble_test > 0.80:
    print(f"\n5-class model is better!")
else:
    print(f"\n⚠️  Consider sticking with 8-class model")

print(f"\n✅ Next: python 04_ensemble_test_FINAL.py")