import os
import numpy as np
import librosa
import json
from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile
import pickle
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
UPLOAD_FOLDER = tempfile.gettempdir()
MODEL_PATH = 'quran_classifier.pkl'

# Load the trained model and components
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Surah mapping (must match your training labels)
        surah_mapping = {
            "Surah Al-asr": 6,
            "Surah Al-falaq": 3,
            "Surah Al-fatiha": 1,
            "Surah Al-ikhlas": 4,
            "Surah Al-kausar": 5,
            "Surah An-nas": 2
        }

        surah_name_to_id = surah_mapping
        surah_id_to_name = {v: k for k, v in surah_mapping.items()}

        class_index_to_surah = {
            i: name for i, name in enumerate(sorted(model_data['classes']))
        }

    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None
    scaler = None
    surah_mapping = {
        "Surah Al-asr": 6,
        "Surah Al-falaq": 3,
        "Surah Al-fatiha": 1,
        "Surah Al-ikhlas": 4,
        "Surah Al-kausar": 5,
        "Surah An-nas": 2
    }
    print("⚠️ Running in fallback mode - will return random predictions")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Extract features from audio file matching training setup"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        total_duration = librosa.get_duration(y=y, sr=sr)

        segment_duration = 5
        stride = 2
        features_list = []

        for start in np.arange(0, total_duration - segment_duration + 0.1, stride):
            y_seg = y[int(start * sr): int((start + segment_duration) * sr)]

            mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_seg, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_seg)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)

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

        features_mean = np.mean(features_list, axis=0)
        features_std = np.std(features_list, axis=0)
        return np.concatenate([features_mean, features_std])

    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

def fallback_prediction():
    """Generate a fallback prediction when model is unavailable"""
    surah_names = list(surah_mapping.keys())
    selected_surah = np.random.choice(surah_names)
    surah_id = surah_mapping[selected_surah]
    confidence = 60 + np.random.random() * 35

    probabilities = {}
    for surah in surah_names:
        if surah == selected_surah:
            probabilities[surah] = confidence / 100
        else:
            probabilities[surah] = (100 - confidence) / (len(surah_names) - 1) / 100

    return {
        'recognized': confidence > 60,
        'surahId': surah_id,
        'surahName': selected_surah,
        'confidence': confidence,
        'probabilities': probabilities,
        'fallback': True
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    if model is None or scaler is None:
        return jsonify(fallback_prediction())

    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400

        features_scaled = scaler.transform([features])
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        surah_name = class_index_to_surah[prediction_idx]
        surah_id = surah_name_to_id[surah_name]
        confidence = float(probabilities.max() * 100)

        prob_dict = {
            class_index_to_surah[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
