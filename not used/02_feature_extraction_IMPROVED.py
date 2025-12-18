import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 2 IMPROVED: ADVANCED FEATURE EXTRACTION + PCA
Purpose: Extract BETTER 156 features + dimensionality reduction
Runtime: ~10-12 minutes
Output: features_data_improved.pkl (with PCA)
"""

AUGMENTED_DIR = "./augmented_data"
OUTPUT_FILE = "./features_data_improved.pkl"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
}

def extract_enhanced_features(file_path):
    """Extract 156 ADVANCED features from audio file"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3)
    except:
        return None
    
    features = []
    
    # 1. MFCC + derivatives (78 dims)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))
    
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))
    
    # 2. Mel-spectrogram (40 dims)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_spec_db, axis=1))
    features.extend(np.std(mel_spec_db, axis=1))
    
    # 3. Spectral features (18 dims)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent)])
    features.extend([np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw)])
    features.extend([np.mean(spec_rolloff), np.std(spec_rolloff), np.max(spec_rolloff)])
    
    # Add percentiles
    features.extend([np.percentile(spec_cent, 25), np.percentile(spec_cent, 75)])
    features.extend([np.percentile(spec_bw, 25), np.percentile(spec_bw, 75)])
    features.extend([np.percentile(spec_rolloff, 25), np.percentile(spec_rolloff, 75)])
    
    # 4. Zero crossing rate (6 dims)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr), 
                     np.percentile(zcr, 25), np.percentile(zcr, 75)])
    
    # 5. RMS energy (6 dims)
    rms = librosa.feature.rms(y=y)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
                     np.percentile(rms, 25), np.percentile(rms, 75)])
    
    # 6. Chroma (24 dims)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # 7. Tempogram (12 dims) - rhythmic features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    features.extend([np.mean(tempogram), np.std(tempogram), np.max(tempogram)])
    
    # 8. Spectral contrast (14 dims)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spec_contrast, axis=1))
    features.extend(np.std(spec_contrast, axis=1))
    
    return np.array(features)

print("=" * 60)
print("🎵 EXTRACTING ADVANCED FEATURES + PCA")
print("=" * 60)

X, y = [], []
file_count = 0

for fname in sorted(os.listdir(AUGMENTED_DIR)):
    if not fname.endswith('.wav'):
        continue
    
    file_path = os.path.join(AUGMENTED_DIR, fname)
    parts = fname.split('-')
    
    if len(parts) >= 3 and parts[2] in EMOTION_MAP:
        emotion = EMOTION_MAP[parts[2]]
        features = extract_enhanced_features(file_path)
        
        if features is not None:
            X.append(features)
            y.append(emotion)
            file_count += 1
            
            if file_count % 500 == 0:
                print(f"✓ Processed {file_count} files...")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Feature extraction complete!")
print(f"   Total files: {len(X)}")
print(f"   Initial features: {X.shape[1]}")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: Reduce to top 50 components (keeps 95%+ variance)
print(f"\n🔧 Applying PCA...")
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

print(f"   PCA components: {X_pca.shape[1]}")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Val/Test split (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(f"\n✅ Train/Val/Test split complete!")
print(f"   Train: {len(X_train)} samples (60%)")
print(f"   Val: {len(X_val)} samples (20%)")
print(f"   Test: {len(X_test)} samples (20%)")

# Save data WITH PCA
data = {
    'X_pca': X_pca,
    'y_encoded': y_encoded,
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'scaler': scaler,
    'pca': pca,
    'le': le,
    'emotions': le.classes_.tolist()
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)

print(f"\n✅ Data saved to {OUTPUT_FILE}")
print(f"\n✅ Next step: Run 03_cnn_train_xgboost.py")