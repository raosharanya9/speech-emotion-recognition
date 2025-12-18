import os
import librosa
import numpy as np
from librosa import effects
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

"""
SCRIPT 1: DATA AUGMENTATION
Purpose: Expand 1440 → 5760 training samples
Runtime: 15-20 minutes
Output: augmented_data/ folder with 5760 WAV files
"""

ARCHIVE_DIR = "./archive"
OUTPUT_DIR = "./augmented_data"

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
}

def augment_audio(y, sr):
    """Create 3 augmented versions of audio"""
    augmented = []
    
    # Original
    augmented.append(y)
    
    # 1. Pitch shift (±2 semitones)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented.append(y_pitch)
    
    # 2. Time stretch (1.1x speed)
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    augmented.append(y_stretch)
    
    # 3. Add noise
    noise = np.random.normal(0, 0.005, len(y))
    y_noise = y + noise
    augmented.append(y_noise)
    
    return augmented

print("=" * 60)
print("🎵 AUGMENTING AUDIO FILES FOR TRAINING")
print("=" * 60)
print(f"Input: {ARCHIVE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Expected samples: 1440 → 5760 (4x expansion)")
print("=" * 60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_count = 0
total_files = 0

for actor_folder in sorted(os.listdir(ARCHIVE_DIR)):
    actor_path = os.path.join(ARCHIVE_DIR, actor_folder)
    
    if not os.path.isdir(actor_path):
        continue
    
    print(f"\n✓ Processing {actor_folder}...")
    
    for wav_file in sorted(os.listdir(actor_path)):
        if not wav_file.endswith('.wav'):
            continue
        
        file_path = os.path.join(actor_path, wav_file)
        
        try:
            y, sr = librosa.load(file_path, sr=22050)
            augmented_versions = augment_audio(y, sr)
            
            # Save all versions
            for idx, aug_audio in enumerate(augmented_versions):
                output_file = os.path.join(OUTPUT_DIR, f"{wav_file[:-4]}_aug{idx}.wav")
                sf.write(output_file, aug_audio, sr)
            
            file_count += 1
            total_files += 4
            
            if file_count % 100 == 0:
                print(f"  Processed: {file_count} files → {total_files} augmented")
        
        except Exception as e:
            print(f"  Error processing {wav_file}: {e}")

print("\n" + "=" * 60)
print("✅ AUGMENTATION COMPLETE!")
print("=" * 60)
print(f"Original files: {file_count}")
print(f"Total files created: {total_files}")
print(f"Expansion factor: {total_files/file_count if file_count > 0 else 0:.1f}x")
print(f"\n✅ Output saved to: {OUTPUT_DIR}")
print("\n✅ Next step: Run 02_feature_extraction_FIXED.py")
