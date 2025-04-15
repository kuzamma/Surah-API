import os
import tempfile
import shutil
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import librosa
import numpy as np
import pickle

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model components
try:
    with open('quran_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['quran_classifier.pkl']
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

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_dir = tempfile.mkdtemp()
    try:
        filename = secure_filename(file.filename)
        if '.' not in filename:
            filename += '.m4a'
        
        temp_path = os.path.join(temp_dir, filename)
        print(f"📁 Saving uploaded file to: {temp_path}")
        file.save(temp_path)

        if not os.path.exists(temp_path):
            print("❌ File not found after save.")
            return jsonify({'error': 'File save failed'}), 500
        if os.path.getsize(temp_path) == 0:
            print("❌ File saved but it's empty.")
            return jsonify({'error': 'Empty file received'}), 400

        print("🔍 Extracting features...")
        features = extract_features(temp_path)
        if features is None:
            print("❌ Feature extraction failed.")
            return jsonify({'error': 'Failed to extract features'}), 500

        print("🔬 Scaling features...")
        scaled_features = scaler.transform([features])

        print("🧠 Making prediction...")
        prediction = model.predict(scaled_features)[0]
        confidence = max(model.predict_proba(scaled_features)[0])

        surah_id = class_index_to_surah_id.get(prediction, None)
        surah_name = surah_id_to_name.get(surah_id, "Unknown")

        print(f"✅ Prediction complete: {surah_name} (confidence: {confidence:.2f})")

        return jsonify({
            'surah_id': surah_id,
            'surah_name': surah_name,
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        print(f"🔥 Internal error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️ Cleanup error: {str(e)}")


def extract_features(audio_path):
    """Modified feature extraction without aggressive trimming"""
    try:
        print("📥 Loading audio...")
        sample_rate = 22050
        segment_duration = 5  # seconds
        stride = 2  # seconds
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512

        try:
            y, sr = librosa.load(audio_path, sr=sample_rate)
            print(f"🎧 Loaded with librosa — duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
        except Exception as e:
            print(f"⚠️ Librosa failed: {e}, trying pydub...")
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            y = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                y = y.reshape(-1, audio.channels).mean(axis=1)
            sr = audio.frame_rate
            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            print(f"🎧 Loaded with pydub — duration: {librosa.get_duration(y=y, sr=sample_rate):.2f}s")

        # ✅ Skip trimming
        total_duration = librosa.get_duration(y=y, sr=sr)
        if total_duration < segment_duration:
            print(f"⏳ Audio too short: {total_duration:.2f}s")
            return None

        features_list = []

        for start in np.arange(0, total_duration - segment_duration + 0.1, stride):
            y_segment = y[int(start * sr): int((start + segment_duration) * sr)]
            print(f"🎯 Segment {start:.2f}s → {start + segment_duration:.2f}s")

            mfccs = librosa.feature.mfcc(
                y=y_segment,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=y_segment)

            segment_features = np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.std(mfccs.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(contrast.T, axis=0),
                flatness.flatten()
            ])
            features_list.append(segment_features)

        if not features_list:
            print("❌ No segments processed")
            return None

        features_mean = np.mean(features_list, axis=0)
        features_std = np.std(features_list, axis=0)
        print("✅ Feature extraction complete")

        return np.concatenate([features_mean, features_std])

    except Exception as e:
        print(f"🔥 Feature extraction error: {str(e)}")
        return None

    
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'up'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)