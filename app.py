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
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration (must match train.py)
SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_DURATION = 5
STRIDE = 2
START_OFFSET = 8.0

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}
MAX_FILE_SIZE_MB = 10
PROCESSING_TIMEOUT = 20  # seconds

# Load model and components
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']  # Full pipeline (scaler + pca + classifier)
        label_encoder = model_data['label_encoder']
        surah_mapping = model_data['surah_mapping']
        class_names = model_data['classes']

    logger.info("✅ Model and components loaded successfully")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")
    model = label_encoder = surah_mapping = None
    class_names = []

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Feature extraction matching exactly what was done in train.py"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=START_OFFSET, duration=None)
        y, _ = librosa.effects.trim(y, top_db=40)

        total_duration = librosa.get_duration(y=y, sr=sr)
        if total_duration < SEGMENT_DURATION:
            logger.warning(f"Audio too short after offset: {total_duration:.2f} seconds")
            return None

        segment_features_list = []

        # Additional features (global for whole file)
        try:
            from librosa.feature import rhythm
            tempo = rhythm.tempo(y=y, sr=sr)[0]
        except ImportError:
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        additional_features = np.array([
            tempo,
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=y)),
            np.mean(librosa.effects.harmonic(y=y)),
            np.mean(librosa.effects.percussive(y=y))
        ])

        # Now process segments
        for start in range(0, int(total_duration) - SEGMENT_DURATION + 1, STRIDE):
            y_segment = y[start * sr : (start + SEGMENT_DURATION) * sr]

            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=y_segment)

            if mfcc.size == 0 or chroma.size == 0 or contrast.size == 0 or flatness.size == 0:
                logger.warning("⚠️ Skipping empty segment")
                continue

            # Take mean + std for each
            feature_vector = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.mean(contrast, axis=1),
                np.std(contrast, axis=1),
                np.mean(flatness, axis=1)  # flatness is (1, frames)
            ])

            segment_features_list.append(feature_vector)

        if not segment_features_list:
            return None

        segment_features_array = np.array(segment_features_list)
        # Mean and Std across segments
        features_mean = np.mean(segment_features_array, axis=0)
        features_std = np.std(segment_features_array, axis=0)

        full_features = np.concatenate([features_mean, features_std, additional_features])

        logger.info(f"Extracted feature shape: {full_features.shape}")
        return full_features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None



@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "✅ Surah API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"rec_{int(time.time())}_{filename}")
        file.save(temp_path)

        features = extract_features(temp_path)
        if features is None:
            return jsonify({"error": "Could not extract features"}), 400

        features = features.reshape(1, -1)  # Reshape to 2D
        logger.info(f"Features ready: {features.shape}")

        # Predict directly with the pipeline
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        surah_name = next(k for k, v in surah_mapping.items() if v == prediction)

        return jsonify({
            "surahId": int(prediction),
            "surahName": surah_name,
            "confidence": float(np.max(probabilities)),
            "processingTime": time.time() - start_time,
            "probabilities": {
                name: float(probabilities[i]) for i, name in enumerate(label_encoder.classes_)
            }
        })

    except Exception as e:
        logger.error(f"🚨 Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"🧹 Error cleaning temp file: {e}")

# Start app
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        threaded=True,
        debug=False
    )
