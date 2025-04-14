# app.py
import os
import numpy as np
import librosa
import json
from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import tempfile
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
UPLOAD_FOLDER = tempfile.gettempdir()
MODEL_PATH = 'quran_classifier.pkl'

# Load the trained model and components
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Surah mapping (must match your training labels)
        surah_mapping = {
            "Surah Al-asr": 1,
            "Surah Al-falaq": 2,
            "Surah Al-fatiha": 3,
            "Surah Al-ikhlas": 4,
            "Surah Al-kausar": 5,
            "Surah An-nas": 6
        }
        
        # Create reverse mappings
        surah_name_to_id = surah_mapping
        surah_id_to_name = {v: k for k, v in surah_mapping.items()}
        
        # Create a mapping between model's class indices and surah names
        # Assuming the model was trained with classes in the same order as surah_mapping
        class_index_to_surah = {
            i: name for i, name in enumerate(sorted(surah_mapping.keys()))
        }
        
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Extract features from audio file matching training setup"""
    try:
        # Load audio with same parameters as training
        y, sr = librosa.load(
            audio_path,
            sr=22050,
            offset=8.0,  # Same 8-second offset as training
            duration=None
        )
        
        # Process the same way as training
        y, _ = librosa.effects.trim(y, top_db=20)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate number of segments (same as training)
        segment_duration = 5
        stride = 2
        features_list = []
        
        for start in np.arange(0, total_duration - segment_duration + 0.1, stride):
            y_seg = y[int(start * sr) : int((start + segment_duration) * sr)]
            
            # Extract features (must match training exactly)
            mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_seg, sr=sr)
            
            # Additional features used in training
            spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_seg)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)
            
            # Combine features exactly as done during training
            segment_features = np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.std(mfccs.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(contrast.T, axis=0),
                np.mean(spectral_centroid),
                np.mean(zero_crossing_rate),
                np.mean(spectral_bandwidth),
                librosa.feature.rms(y=y_seg).flatten()
            ])
            features_list.append(segment_features)
        
        if not features_list:
            return None
            
        # Calculate mean and std as done in training
        features_mean = np.mean(features_list, axis=0)
        features_std = np.std(features_list, axis=0)
        return np.concatenate([features_mean, features_std])
        
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get surah details using our mapping
        surah_name = class_index_to_surah[prediction_idx]
        surah_id = surah_name_to_id[surah_name]
        confidence = float(probabilities.max() * 100)
        
        # Prepare probabilities dictionary
        prob_dict = {
            class_index_to_surah[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'recognized': confidence > 60,
            'surahId': surah_id,
            'surahName': surah_name,
            'confidence': confidence,
            'probabilities': prob_dict
        })
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/surahs', methods=['GET'])
def list_surahs():
    """Return the complete surah mapping"""
    return jsonify({
        'surahs': surah_mapping,
        'count': len(surah_mapping)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)