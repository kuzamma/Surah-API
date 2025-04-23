import os
import time
import numpy as np
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tempfile
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Log incoming requests
@app.before_request
def log_request_info():
    logger.info(f"📥 {request.method} {request.path} - from {request.remote_addr}")

# Log outgoing responses
@app.after_request
def log_response_info(response):
    logger.info(f"🔄 Response Status: {response.status}")
    return response

# Health check endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "✅ Surah API is running"}), 200

# Allowed file types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
MAX_FILE_SIZE_MB = 10
PROCESSING_TIMEOUT = 20  # seconds

# Load model and components
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        class_names = model_data['classes']
    logger.info("✅ Model and components loaded successfully")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")
    model = scaler = label_encoder = class_names = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Extract MFCC and spectral features from the audio"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=20, mono=True, res_type='kaiser_fast')
        y, _ = librosa.effects.trim(y, top_db=40)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, hop_length=512, n_fft=2048)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(spectral_centroid)
        ])
        return features
    except Exception as e:
        logger.error(f"⚠️ Feature extraction failed: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"rec_{int(time.time())}_{filename}")
        file.save(temp_path)

        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return jsonify({
                "error": f"File too large ({file_size/1024/1024:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)"
            }), 400

        if time.time() - start_time > PROCESSING_TIMEOUT - 2:
            return jsonify({"error": "Processing timeout"}), 500

        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Could not extract audio features"}), 400

        if not model or not scaler or not label_encoder:
            return jsonify({"error": "Model not loaded"}), 500

        features_scaled = scaler.transform([features])
        prediction_index = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        surah_name = class_names[prediction_index]
        surah_id = label_encoder.transform([surah_name])[0]

        return jsonify({
            "surahId": int(surah_id),
            "surahName": surah_name,
            "confidence": float(np.max(probabilities) * 100),
            "processingTime": time.time() - start_time,
            "probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        })

    except Exception as e:
        logger.error(f"🚨 Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"🧹 Failed to delete temp file: {e}")

# Run app
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        threaded=True,
        debug=False
    )
