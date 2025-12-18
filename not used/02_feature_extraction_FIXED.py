import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 2: ENHANCED FEATURE EXTRACTION
Purpose: Extract 156 features from augmented audio files
Runtime: ~8-12 minutes
Output: features_data.pkl
"""

AUGMENTED_DIR = "./augmented_data"
OUTPUT_FILE = "./features_data.pkl"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
}

def extract_extended_features(file_path):
    """Extract 156 features from audio file"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3)
    except:
        return None
    
    features = []
    
    # 1. MFCC (26 dims)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    # 2. Delta MFCC (26 dims)
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))
    
    # 3. Delta-Delta MFCC (26 dims)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))
    
    # 4. Mel-spectrogram (80 dims) - COMPLETE
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_spec_db, axis=1))
    features.extend(np.std(mel_spec_db, axis=1))
    
    # 5. Spectral contrast (14 dims)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048)
    features.extend(np.mean(spec_contrast, axis=1))
    features.extend(np.std(spec_contrast, axis=1))
    
    # 6. Chroma (24 dims)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # 7. Zero crossing rate (3 dims)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
    
    # 8. RMS energy (3 dims)
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.std(rms), np.max(rms)])
    
    # 9. Spectral centroid (3 dims)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent)])
    
    # 10. Spectral bandwidth (3 dims)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw)])
    
    # 11. Spectral rolloff (3 dims)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(spec_rolloff), np.std(spec_rolloff), np.max(spec_rolloff)])
    
    return np.array(features)

print("=" * 60)
print("🎵 EXTRACTING ENHANCED AUDIO FEATURES")
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
        features = extract_extended_features(file_path)
        
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
print(f"   Features per file: {X.shape}")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Val/Test split (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(f"\n✅ Train/Val/Test split complete!")
print(f"   Train: {len(X_train)} samples (60%)")
print(f"   Val: {len(X_val)} samples (20%)")
print(f"   Test: {len(X_test)} samples (20%)")

# Save data
data = {
    'X_scaled': X_scaled,
    'y_encoded': y_encoded,
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'scaler': scaler,
    'le': le,
    'emotions': le.classes_.tolist()
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)

print(f"\n✅ Data saved to {OUTPUT_FILE}")
print("\n✅ Next step: Run 03_cnn_train.py")
