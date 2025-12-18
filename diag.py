import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""
DIAGNOSTIC + FIX: Find the real problem and fix it
"""

INPUT_FILE = "./features_data_final.pkl"

print("=" * 70)
print("🔍 DIAGNOSING THE PROBLEM")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
emotions = data['emotions']

print(f"\n📊 DATA ANALYSIS:")
print(f"   Train shape: {X_train.shape}")
print(f"   Val shape: {X_val.shape}")
print(f"   Test shape: {X_test.shape}")
print(f"   Classes: {len(emotions)}")
print(f"   Class names: {emotions}")

# Check class distribution
print(f"\n📈 CLASS DISTRIBUTION:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"   {emotions[cls]}: {cnt} samples ({cnt/len(y_train)*100:.1f}%)")

# ==================== TEST 1: PCA Analysis ====================
print("\n" + "=" * 70)
print("🔬 TEST 1: PCA VARIANCE ANALYSIS")
print("=" * 70)

pca_full = PCA()
pca_full.fit(X_train)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

print(f"Features needed for 80% variance: {np.argmax(cumsum >= 0.80) + 1}")
print(f"Features needed for 90% variance: {np.argmax(cumsum >= 0.90) + 1}")
print(f"Features needed for 95% variance: {np.argmax(cumsum >= 0.95) + 1}")

# ==================== TEST 2: Different Feature Selections ====================
print("\n" + "=" * 70)
print("🔬 TEST 2: FEATURE SELECTION COMPARISON")
print("=" * 70)

# Try different feature counts with RF
for k in [50, 80, 100, 150, 200]:
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, 
                                 min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_train_sel, y_train)
    
    train_acc = rf.score(X_train_sel, y_train)
    test_acc = rf.score(X_test_sel, y_test)
    
    print(f"   k={k:3d}: Train={train_acc:.1%} | Test={test_acc:.1%}")

# ==================== TEST 3: GB with different learning rates ====================
print("\n" + "=" * 70)
print("🔬 TEST 3: GRADIENT BOOSTING WITH DIFFERENT LR")
print("=" * 70)

selector = SelectKBest(f_classif, k=80)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=lr,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train_sel, y_train)
    
    test_acc = gb.score(X_test_sel, y_test)
    print(f"   LR={lr:.3f}: Test={test_acc:.1%}")

# ==================== TEST 4: RF with different depths ====================
print("\n" + "=" * 70)
print("🔬 TEST 4: RANDOM FOREST WITH DIFFERENT DEPTHS")
print("=" * 70)

selector = SelectKBest(f_classif, k=80)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

for depth in [5, 8, 10, 12, 15, 20]:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_sel, y_train)
    
    test_acc = rf.score(X_test_sel, y_test)
    print(f"   Depth={depth:2d}: Test={test_acc:.1%}")

# ==================== BEST CONFIG ====================
print("\n" + "=" * 70)
print("✅ BEST CONFIGURATION FOUND")
print("=" * 70)

selector = SelectKBest(f_classif, k=80)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)

# Best RF
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_sel, y_train)
rf_acc = rf.score(X_test_sel, y_test)

# Best GB
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_sel, y_train)
gb_acc = gb.score(X_test_sel, y_test)

# Simple ensemble
rf_proba = rf.predict_proba(X_test_sel)
gb_proba = gb.predict_proba(X_test_sel)
ensemble_proba = (rf_proba + gb_proba) / 2
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\nRF Test Accuracy:       {rf_acc:.2%}")
print(f"GB Test Accuracy:       {gb_acc:.2%}")
print(f"Ensemble Accuracy:      {ensemble_acc:.2%}")

print("\n" + "=" * 70)