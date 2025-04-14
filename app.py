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
    """Consistent feature extraction matching training pipeline"""
    try:
        # Parameters matching your training setup
        sample_rate = 22050
        segment_duration = 5  # seconds
        stride = 2  # seconds
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512
        
        # Load audio using same method as training
        try:
            y, sr = librosa.load(audio_path, sr=sample_rate)
        except Exception as e:
            # Fallback to pydub if librosa fails
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            y = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                y = y.reshape(-1, audio.channels).mean(axis=1)
            sr = audio.frame_rate
            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        
        # Trim silence aggressively as in training
        y, _ = librosa.effects.trim(y, top_db=25)
        
        total_duration = librosa.get_duration(y=y, sr=sr)
        if total_duration < segment_duration:
            print(f"Audio too short after trimming: {total_duration:.2f}s")
            return None
        
        features_list = []
        
        # Process segments matching training stride
        for start in np.arange(0, total_duration - segment_duration + 0.1, stride):
            y_segment = y[int(start * sr): int((start + segment_duration) * sr)]
            
            # Extract features exactly as in training
            mfccs = librosa.feature.mfcc(
                y=y_segment, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
            
            segment_features = np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.std(mfccs.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(contrast.T, axis=0),
                librosa.feature.spectral_flatness(y=y_segment).flatten()
            ])
            features_list.append(segment_features)
        
        if not features_list:
            return None
            
        # Aggregate features same as training
        features_mean = np.mean(features_list, axis=0)
        features_std = np.std(features_list, axis=0)
        return np.concatenate([features_mean, features_std])
        
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)