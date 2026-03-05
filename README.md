# Speech Emotion Recognition (5-Class Model)

A Digital Signal Processing + Machine Learning based system that classifies short speech clips into five emotion categories: Negative, Positive, Neutral, Fearful, and Surprise.

##  Project Overview

This project was developed to design a complete end-to-end Speech Emotion Recognition (SER) system.

The system:
- Accepts short audio clips
- Performs data augmentation (4× expansion)
- Extracts 279 spectral and temporal features
- Reduces features to top 150 using SelectKBest
- Uses a weighted ensemble classifier
- Displays emotion probabilities through a Streamlit web interface

The final ensemble model achieved **81.4% accuracy** on the 5-class classification task.

##  System Architecture

→ Audio Input
<br>→ Data Augmentation (Pitch Shift, Time Stretch, Noise Addition)
<br>→ Feature Extraction (279 DSP-based features)
<br>→ Feature Selection (Top 150 features)
<br>→ Ensemble Classification (MLP + Random Forest + Gradient Boosting)
<br>→ Probability Visualization via Streamlit

##  Dataset
- Dataset: RAVDESS Emotional Speech Database
- Short English speech clips (2–4 seconds)
- Original: 1,440 samples
- After augmentation: 5,760 samples
- Grouped into 5 broad emotion classes

##  Feature Extraction

Extracted features include:
- MFCCs (mean, std, max, min)
- Delta & Delta-Delta MFCC
- Mel-Spectrogram features
- Chroma features
- Spectral centroid, bandwidth, roll-off
- Zero Crossing Rate
- RMS Energy
- Tempogram

Features are standardized and reduced to 150 most informative features before training.

##  Classification Models
- Multi-Layer Perceptron (MLP)
- Random Forest
- Gradient Boosting
- Weighted Ensemble (Final model)


##  Results

Overall Accuracy: 81.4%
<br>Train/Validation/Test split: 60/20/20 (Stratified)
<br>Strongest performance: Surprise & Positive classes
<br>Major confusion: Negative vs Fearful (acoustic similarity)
<br>Detailed evaluation includes confusion matrix, classification report, and probability visualization.


##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/raosharanya9/speech-emotion-recognition.git
```
<br>

2. Move into the directory:
```bash
cd speech-emotion-recognition
```
<br>

3. Install the program dependencies:
```bash
pip install -r requirements.txt
```
<br>

4. Run the program using streamlit:
```bash
streamlit run 05_app_final.py
```
