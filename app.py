import os
import time  # Added for timeout handling
import numpy as np
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tempfile
import pickle

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configs (Optimized for Render.com free tier)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
MAX_FILE_SIZE_MB = 5
PROCESSING_TIMEOUT = 25  # Seconds (Render kills at 30s)

# Load model (unchanged)
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = scaler = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Optimized feature extraction"""
    try:
        # Downsample to 16kHz and limit duration to 30s
        y, sr = librosa.load(audio_path, sr=16000, duration=30)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Faster processing with smaller segments
        segment_duration = 3  # Reduced from 5s
        stride = 1  # Reduced from 2s
        features = []
        
        for start in np.arange(0, len(y)/sr - segment_duration, stride):
            y_seg = y[int(start*sr):int((start+segment_duration)*sr)]
            
            # Only critical features
            mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=10)  # Reduced from 13
            spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
            
            features.append(np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.mean(spectral_centroid)
            ]))
        
        return np.concatenate([np.mean(features, axis=0), np.std(features, axis=0)])
    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Check timeout early
    if time.time() - start_time > PROCESSING_TIMEOUT:
        return jsonify({"error": "Timeout"}), 500
        
    if 'audio' not in request.files:
        return jsonify({"error": "No file"}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
        
    try:
        # Save to temp file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Check file size
        if os.path.getsize(temp_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
            os.remove(temp_path)
            return jsonify({"error": f"File > {MAX_FILE_SIZE_MB}MB"}), 400
        
        # Process
        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500
            
        if model and scaler:
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            return jsonify({
                "surahId": int(pred),
                "confidence": float(np.max(model.predict_proba(features_scaled)) * 100)
            })
        else:
            return jsonify({"error": "Model not loaded"}), 500
            
    except Exception as e:
        print(f"🔥 Prediction crashed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), threaded=True)