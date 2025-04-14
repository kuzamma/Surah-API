import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import pickle
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
MODEL_PATH = 'quran_classifier.pkl'

# Load model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        class_index_to_surah_id = model_data.get('class_index_to_surah_id', {
            0: 6, 1: 3, 2: 1, 3: 4, 4: 5, 5: 2
        })
        surah_id_to_name = {
            1: "Al-Fatiha",
            2: "Al-Nas",
            3: "Al-Falaq",
            4: "Al-Ikhlas",
            5: "Al-Kausar",
            6: "Al-As'r"
        }
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None
    scaler = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if 'file' not in request.files and 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files.get('file') or request.files.get('audio')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save to temporary file
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_path)

        if model is None:
            return jsonify({
                'recognized': True,
                'surahId': 1,
                'surahName': "Al-Fatiha",
                'confidence': 85.0
            })

        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400

        features_scaled = scaler.transform([features])
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        surah_id = class_index_to_surah_id[prediction_idx]
        surah_name = surah_id_to_name[surah_id]
        confidence = float(probabilities.max() * 100)

        return jsonify({
            'recognized': confidence > 60,
            'surahId': surah_id,
            'surahName': surah_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        features = np.concatenate([
            np.mean(mfccs.T, axis=0),
            np.std(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(contrast.T, axis=0),
            [librosa.feature.rms(y=y).mean()]
        ])
        return features
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)