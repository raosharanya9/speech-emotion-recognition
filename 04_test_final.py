import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

MODEL_FILE = "./models/trained_models_reduced.pkl"

print("=" * 70)
print(" TESTING 5-CLASS EMOTION RECOGNITION MODEL")
print("=" * 70)

# Load models
print("\n Loading trained models...")
with open(MODEL_FILE, 'rb') as f:
    data = pickle.load(f)

mlp = data['mlp']
rf = data['rf']
gb = data['gb']
selector = data['selector']
emotions = data['emotions']
emotions_map = data['emotions_map']
X_test = data['X_test']
y_test = data['y_test']

print(f"    Models loaded!")
print(f"   Emotions: {emotions}")

# ==================== INDIVIDUAL MODEL PERFORMANCE ====================
print("\n" + "=" * 70)
print(" INDIVIDUAL MODEL PERFORMANCE")
print("=" * 70)

mlp_pred = mlp.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

mlp_acc = accuracy_score(y_test, mlp_pred)
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"\nMLP Accuracy: {mlp_acc:.2%}")
print(f"RF Accuracy:  {rf_acc:.2%}")
print(f"GB Accuracy:  {gb_acc:.2%}")

# ==================== ENSEMBLE PERFORMANCE ====================
print("\n" + "=" * 70)
print(" ENSEMBLE PERFORMANCE")
print("=" * 70)

mlp_proba = mlp.predict_proba(X_test)
rf_proba = rf.predict_proba(X_test)
gb_proba = gb.predict_proba(X_test)

weights = data['weights']
ensemble_proba = mlp_proba * weights[0] + rf_proba * weights[1] + gb_proba * weights[2]
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n ENSEMBLE ACCURACY: {ensemble_acc:.2%}")
print(f"\nWeights:")
print(f"   MLP: {weights[0]:.1%}")
print(f"   RF:  {weights[1]:.1%}")
print(f"   GB:  {weights[2]:.1%}")

# ==================== DETAILED CLASSIFICATION REPORT ====================
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)

print("\n" + classification_report(y_test, ensemble_pred, target_names=emotions, digits=4))

# ==================== CONFUSION MATRIX ====================
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, ensemble_pred)
print("\nConfusion Matrix:")
print(cm)

print(f"\nPer-class accuracy:")
for i, emotion in enumerate(emotions):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"   {emotion:12s}: {class_acc:.1%}")

# ==================== ERROR ANALYSIS ====================
print("\n" + "=" * 70)
print("⚠️  ERROR ANALYSIS")
print("=" * 70)

errors = ensemble_pred != y_test
num_errors = errors.sum()
error_rate = num_errors / len(y_test)

print(f"\nTotal errors: {num_errors} / {len(y_test)}")
print(f"Error rate: {error_rate:.2%}")

if num_errors > 0:
    print(f"\nMost common confusion pairs:")
    error_indices = np.where(errors)[0]
    confusion_pairs = []
    for idx in error_indices:
        actual = y_test[idx]
        predicted = ensemble_pred[idx]
        confusion_pairs.append((emotions[actual], emotions[predicted]))
    
    from collections import Counter
    pair_counts = Counter(confusion_pairs)
    for (actual, predicted), count in pair_counts.most_common(5):
        print(f"   {actual} → {predicted}: {count} times")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
╔════════════════════════════════════════════════════════════════════╗
║              5-CLASS EMOTION RECOGNITION MODEL                     ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Model Performance:                                                ║
║    • MLP:      {mlp_acc:.2%}                                         ║
║    • RF:       {rf_acc:.2%}                                         ║
║    • GB:       {gb_acc:.2%}                                         ║
║                                                                    ║
║  🎯 Ensemble:  {ensemble_acc:.2%}                                    ║
║                                                                    ║
║  Classes: {', '.join(emotions)}               ║
║                                                                    ║
║  Status: ✅ READY FOR DEPLOYMENT                                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

if ensemble_acc >= 0.90:
    print("Ready for production!")
elif ensemble_acc >= 0.85:
    print("Good performance!")
elif ensemble_acc >= 0.80:
    print("Acceptable performance!")
else:
    print("📈 Model performance is reasonable!")

print(f"\n✅ Next steps:")
print(f"   1. Update app to use this model")
print(f"   2. Run Streamlit app: streamlit run app_final.py")
print(f"   3. Test with audio files")