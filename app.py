from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import pickle
import logging
from logging.handlers import RotatingFileHandler
import uuid
import time

# Configuration
MODEL_PATH = "models/quran_classifier.pkl"  # Path to your trained model
UPLOAD_FOLDER = 'uploads'  # Folder to store temporary audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}  # Allowed audio file extensions
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB file size limit

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
handler = RotatingFileHandler('quran_api.log', maxBytes=1000000, backupCount=5)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load the trained model
class QuranClassifierAPI:
    def __init__(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
                self.model = components['model']
                self.scaler = components.get('scaler')
                self.pca = components.get('pca')
                self.label_encoder = components['label_encoder']
                self.surah_mapping = components['surah_mapping']
                self.feature_extractor = components['feature_extractor']
            app.logger.info("Model loaded successfully")
        except Exception as e:
            app.logger.error(f"Failed to load model: {str(e)}")
            raise

    def extract_features(self, audio_path):
        """Extract features using the loaded feature extractor"""
        features, _ = self.feature_extractor.extract_features(audio_path)
        return features

    def predict(self, audio_path):
        """Make prediction on an audio file"""
        try:
            features = self.extract_features(audio_path)
            if features is None:
                return None
                
            features = features.reshape(1, -1)
            
            # Apply preprocessing steps if they exist
            if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
                # Pipeline has its own scaler
                pass
            elif self.scaler:
                features = self.scaler.transform(features)
                
            if hasattr(self.model, 'named_steps') and 'pca' in self.model.named_steps:
                # Pipeline has its own PCA
                pass
            elif self.pca:
                features = self.pca.transform(features)
                
            pred = self.model.predict(features)[0]
            probs = self.model.predict_proba(features)[0]
            
            surah_name = self.label_encoder.inverse_transform([pred])[0]
            surah_label = next(k for k, v in self.surah_mapping.items() if v == surah_name)
            
            return {
                'surah': surah_label,
                'confidence': float(probs.max()),
                'probabilities': {
                    k: float(v) for k, v in zip(
                        self.label_encoder.classes_, 
                        [round(p, 4) for p in probs]
                    )
                }
            }
        except Exception as e:
            app.logger.error(f"Prediction failed: {str(e)}")
            return None

# Initialize classifier
try:
    classifier = QuranClassifierAPI(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Failed to initialize classifier: {str(e)}")
    classifier = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for surah prediction"""
    if not classifier:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser may submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file temporarily
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")
            
            # Process the file
            start_time = time.time()
            result = classifier.predict(filepath)
            processing_time = time.time() - start_time
            
            # Clean up - remove the temporary file
            try:
                os.remove(filepath)
            except Exception as e:
                app.logger.warning(f"Could not remove temporary file: {str(e)}")
            
            if result:
                result['processing_time'] = processing_time
                return jsonify(result)
            else:
                return jsonify({'error': 'Could not process audio file'}), 400
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if classifier:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)