import streamlit as st
import numpy as np
import librosa
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎙️ Emotion Recognition", layout="wide")

"""
FINAL APP: 5-CLASS EMOTION RECOGNITION FROM AUDIO
"""

# Load models
@st.cache_resource
def load_models():
    with open("./models/trained_models_reduced.pkl", 'rb') as f:
        return pickle.load(f)

models = load_models()

mlp = models['mlp']
rf = models['rf']
gb = models['gb']
selector = models['selector']
emotions = models['emotions']
scaler = models['scaler']
weights = models['weights']

st.title("🎙️ Emotion Recognition from Audio")
st.markdown("**5-Class Model: 81.4% Accuracy**")

st.write("""
This app recognizes **5 emotion classes** from audio:
- 😠 **Negative** (angry, sad, disgust)
- 😊 **Positive** (happy)
- 😐 **Neutral** (calm)
- 😨 **Fearful** (fearful)
- 😮 **Surprise** (surprise)

**Accuracy: 81.4%** ✨
""")

# ==================== INPUT ====================
st.header("📁 Upload Audio File")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    # Save temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    # Extract features
    st.info("🔄 Extracting features...")
    
    try:
        y, sr = librosa.load("temp_audio.wav", sr=22050, duration=3)
        
        # Extract 279 features (same as training)
        features = []
        
        # MFCC (52)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        
        # Delta MFCC (52)
        dmfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(dmfcc, axis=1))
        features.extend(np.std(dmfcc, axis=1))
        features.extend(np.max(dmfcc, axis=1))
        ddmfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(ddmfcc, axis=1))
        
        # Mel-Spectrogram (60)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=30)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))
        features.extend(np.std(mel_spec_db, axis=1))
        
        # Chroma (24)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        # Spectral Contrast (28)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048)
        features.extend(np.mean(spec_contrast, axis=1))
        features.extend(np.std(spec_contrast, axis=1))
        features.extend(np.max(spec_contrast, axis=1))
        features.extend(np.min(spec_contrast, axis=1))
        
        # Spectral Centroid (12)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent), np.min(spec_cent),
                        np.median(spec_cent), np.percentile(spec_cent, 25), np.percentile(spec_cent, 75),
                        np.percentile(spec_cent, 10), np.percentile(spec_cent, 90),
                        np.ptp(spec_cent), np.var(spec_cent)])
        
        # Spectral Rolloff (12)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([np.mean(spec_rolloff), np.std(spec_rolloff), np.max(spec_rolloff), np.min(spec_rolloff),
                        np.median(spec_rolloff), np.percentile(spec_rolloff, 25), np.percentile(spec_rolloff, 75),
                        np.percentile(spec_rolloff, 10), np.percentile(spec_rolloff, 90),
                        np.ptp(spec_rolloff), np.var(spec_rolloff)])
        
        # Zero Crossing Rate (12)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr),
                        np.median(zcr), np.percentile(zcr, 25), np.percentile(zcr, 75),
                        np.percentile(zcr, 10), np.percentile(zcr, 90),
                        np.ptp(zcr), np.var(zcr)])
        
        # RMS Energy (12)
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
                        np.median(rms), np.percentile(rms, 25), np.percentile(rms, 75),
                        np.percentile(rms, 10), np.percentile(rms, 90),
                        np.ptp(rms), np.var(rms)])
        
        # Tempogram (8)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        features.extend([np.mean(tempogram), np.std(tempogram), np.max(tempogram), np.min(tempogram),
                        np.median(tempogram), np.percentile(tempogram, 25), np.percentile(tempogram, 75),
                        np.var(tempogram)])
        
        # Spectral Bandwidth (12)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.extend([np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw), np.min(spec_bw),
                        np.median(spec_bw), np.percentile(spec_bw, 25), np.percentile(spec_bw, 75),
                        np.percentile(spec_bw, 10), np.percentile(spec_bw, 90),
                        np.ptp(spec_bw), np.var(spec_bw)])
        
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_sel = selector.transform(features_scaled)
        
        # Predict
        st.success("✅ Features extracted!")
        
        # Get predictions
        mlp_proba = mlp.predict_proba(features_sel)[0]
        rf_proba = rf.predict_proba(features_sel)[0]
        gb_proba = gb.predict_proba(features_sel)[0]
        
        # Ensemble
        ensemble_proba = mlp_proba * weights[0] + rf_proba * weights[1] + gb_proba * weights[2]
        pred_idx = np.argmax(ensemble_proba)
        confidence = ensemble_proba[pred_idx]
        
        # ==================== RESULTS ====================
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Emotion", emotions[pred_idx].upper(), f"{confidence:.1%}")
        
        with col2:
            if confidence > 0.8:
                st.metric("Confidence", "Very High", "✅")
            elif confidence > 0.6:
                st.metric("Confidence", "High", "👍")
            else:
                st.metric("Confidence", "Medium", "⚠️")
        
        with col3:
            st.metric("Model Accuracy", "81.4%", "5 classes")
        
        # Emotion probabilities
        st.subheader("📊 Emotion Probabilities")
        
        prob_data = {emotions[i]: ensemble_proba[i] for i in range(len(emotions))}
        sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        emotion_names = [e for e, _ in sorted_probs]
        emotion_probs = [p for _, p in sorted_probs]
        colors = ['#ff6b6b' if e == emotions[pred_idx] else '#4ecdc4' for e in emotion_names]
        
        ax.barh(emotion_names, emotion_probs, color=colors)
        ax.set_xlabel("Probability")
        ax.set_title("Emotion Probabilities (Ensemble)")
        ax.set_xlim(0, 1)
        
        for i, (name, prob) in enumerate(sorted_probs):
            ax.text(prob + 0.02, i, f"{prob:.1%}", va='center')
        
        st.pyplot(fig)
        
        # Individual model predictions
        st.subheader("🏆 Individual Model Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mlp_pred = np.argmax(mlp_proba)
            st.info(f"**MLP**: {emotions[mlp_pred].upper()}\n({mlp_proba[mlp_pred]:.1%})")
        
        with col2:
            rf_pred = np.argmax(rf_proba)
            st.info(f"**RF**: {emotions[rf_pred].upper()}\n({rf_proba[rf_pred]:.1%})")
        
        with col3:
            gb_pred = np.argmax(gb_proba)
            st.info(f"**GB**: {emotions[gb_pred].upper()}\n({gb_proba[gb_pred]:.1%})")
        
        # Remove temp file
        Path("temp_audio.wav").unlink()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure the audio file is valid")

# ==================== ABOUT ====================
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a **5-class emotion recognition model** trained on the RAVDESS dataset.

**Model Details:**
- **Training data**: 5,760 audio files
- **Classes**: 5 emotions
- **Algorithm**: Ensemble (MLP + Random Forest + Gradient Boosting)
- **Accuracy**: 81.4%
- **Features**: 279 audio features → 150 selected

**Built with:**
- Python, scikit-learn
- librosa (audio processing)
- Streamlit (web app)
""")

st.sidebar.write("---")
st.sidebar.write("📧 For questions or feedback, contact the developer")