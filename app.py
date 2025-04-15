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

# Configs - Adjusted for better performance
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
MAX_FILE_SIZE_MB = 8  # Increased from 5MB
PROCESSING_TIMEOUT = 28  # Increased from 25s (Render kills at 30s)

# Load model with error handling
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")
    model = scaler = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Optimized feature extraction with memory efficiency"""
    try:
        # Load audio with more efficient parameters
        y, sr = librosa.load(
            audio_path,
            sr=16000,  # Downsample to reduce processing
            duration=20,  # Reduced from 30s
            mono=True,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        # Trim with less aggressive settings
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # More efficient feature extraction
        features = []
        segment_length = sr * 2  # 2 second segments
        hop_length = sr // 2  # 0.5 second hops
        
        for i in range(0, len(y) - segment_length, hop_length):
            y_segment = y[i:i + segment_length]
            
            # Extract only essential features
            mfcc = librosa.feature.mfcc(
                y=y_segment,
                sr=sr,
                n_mfcc=10,
                hop_length=512,
                n_fft=2048
            )
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y_segment,
                sr=sr
            )
            
            features.append(np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(spectral_centroid)
            ]))
        
        if not features:
            return None
            
        return np.concatenate([
            np.mean(features, axis=0),
            np.std(features, axis=0)
        ])
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Early validation
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
        # Save to temp file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"rec_{int(time.time())}_{filename}")
        file.save(temp_path)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return jsonify({
                "error": f"File too large ({file_size/1024/1024:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)"
            }), 400
        
        # Timeout check
        if time.time() - start_time > PROCESSING_TIMEOUT - 2:  # 2s buffer
            return jsonify({"error": "Processing timeout"}), 500
            
        # Feature extraction
        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Could not extract audio features"}), 400
            
        if not model or not scaler:
            return jsonify({"error": "Model not loaded"}), 500
            
        # Prediction
        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            "surahId": int(prediction),
            "confidence": float(np.max(probabilities) * 100),
            "processingTime": time.time() - start_time,
            "probabilities": probabilities.tolist()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        threaded=True,
        debug=False
    )