import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
Purpose: Extract 200+ HIGH-QUALITY features from audio-->reduced 150 via SelectKBest.
Runtime: ~12-15 minutes
Output: features_data_final.pkl
"""

AUGMENTED_DIR = "./augmented_data"
OUTPUT_FILE = "./features_data_final.pkl"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
}

def extract_best_features(file_path):
    """Extract 200+ high-quality features"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3)
    except:
        return None
    
    features = []
    
    # 1. MFCC - detailed (52 dims) --for vocal tract shape, timbre
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))      # 13
    features.extend(np.std(mfcc, axis=1))       # 13
    features.extend(np.max(mfcc, axis=1))       # 13
    features.extend(np.min(mfcc, axis=1))       # 13
    
    # 2. Delta & Delta-Delta MFCC (52 dims) --how speech change
    dmfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(dmfcc, axis=1))     # 13
    features.extend(np.std(dmfcc, axis=1))      # 13
    features.extend(np.max(dmfcc, axis=1))      # 13
    
    ddmfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(ddmfcc, axis=1))    # 13
    
    # 3. Mel-Spectrogram (60 dims) --for freq energy
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_spec_db, axis=1))   # 30
    features.extend(np.std(mel_spec_db, axis=1))    # 30
    
    # 4. Chroma Features (24 dims) --for pitch class
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)
    features.extend(np.mean(chroma, axis=1))    # 12
    features.extend(np.std(chroma, axis=1))     # 12
    
    # 5. Spectral Contrast (28 dims) --contrasts
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048)
    features.extend(np.mean(spec_contrast, axis=1))  # 7
    features.extend(np.std(spec_contrast, axis=1))   # 7
    features.extend(np.max(spec_contrast, axis=1))   # 7
    features.extend(np.min(spec_contrast, axis=1))   # 7
    
    # 6. Spectral Centroids (12 dims) -- brightness.
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([
        np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent), np.min(spec_cent),
        np.median(spec_cent), np.percentile(spec_cent, 25), np.percentile(spec_cent, 75)
    ])  # 7
    features.extend([np.percentile(spec_cent, 10), np.percentile(spec_cent, 90)])  # 2
    features.append(np.ptp(spec_cent))  # 1 (peak-to-peak)
    features.append(np.var(spec_cent))  # 1
    
    # 7. Spectral Rolloff (12 dims) --how much energy is in high frequencies
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend([
        np.mean(spec_rolloff), np.std(spec_rolloff), np.max(spec_rolloff), np.min(spec_rolloff),
        np.median(spec_rolloff), np.percentile(spec_rolloff, 25), np.percentile(spec_rolloff, 75)
    ])  # 7
    features.extend([np.percentile(spec_rolloff, 10), np.percentile(spec_rolloff, 90)])  # 2
    features.append(np.ptp(spec_rolloff))
    features.append(np.var(spec_rolloff))
    
    # 8. Zero Crossing Rate (12 dims) --for noisiness
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([
        np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr),
        np.median(zcr), np.percentile(zcr, 25), np.percentile(zcr, 75)
    ])  # 7
    features.extend([np.percentile(zcr, 10), np.percentile(zcr, 90)])  # 2
    features.append(np.ptp(zcr))
    features.append(np.var(zcr))
    
    # 9. RMS Energy (12 dims) --for loudness
    rms = librosa.feature.rms(y=y)[0]
    features.extend([
        np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
        np.median(rms), np.percentile(rms, 25), np.percentile(rms, 75)
    ])  # 7
    features.extend([np.percentile(rms, 10), np.percentile(rms, 90)])  # 2
    features.append(np.ptp(rms))
    features.append(np.var(rms))
    
    # 10. Tempogram (8 dims) --rhythmic structure and tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    features.extend([
        np.mean(tempogram), np.std(tempogram), np.max(tempogram), np.min(tempogram),
        np.median(tempogram), np.percentile(tempogram, 25), np.percentile(tempogram, 75),
        np.var(tempogram)
    ])
    
    # 11. Spectral Bandwidth (12 dims) --for brightness
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend([
        np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw), np.min(spec_bw),
        np.median(spec_bw), np.percentile(spec_bw, 25), np.percentile(spec_bw, 75)
    ])  # 7
    features.extend([np.percentile(spec_bw, 10), np.percentile(spec_bw, 90)])  # 2
    features.append(np.ptp(spec_bw))
    features.append(np.var(spec_bw))
    
    return np.array(features)

print("=" * 70)
print("🎵 EXTRACTING HIGH-QUALITY FEATURES (NO REDUCTION)")
print("=" * 70)

X, y = [], []
file_count = 0

for fname in sorted(os.listdir(AUGMENTED_DIR)):
    if not fname.endswith('.wav'):
        continue
    
    file_path = os.path.join(AUGMENTED_DIR, fname)
    parts = fname.split('-')
    
    if len(parts) >= 3 and parts[2] in EMOTION_MAP:
        emotion = EMOTION_MAP[parts[2]]
        features = extract_best_features(file_path)
        
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
print(f"   Features per file: {X.shape[1]}")

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

print(f"\n✅ Train/Val/Test split!")
print(f"   Train: {len(X_train)} samples")
print(f"   Val: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

# Save
data = {
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
print(f"\n✅ Next: python 03_train_final.py")