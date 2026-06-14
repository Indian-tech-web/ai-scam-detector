from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import librosa
import joblib
import io
import soundfile as sf

app = Flask(__name__)

SAMPLE_RATE = 22050

def extract_features(audio):
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE))
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma, axis=1)
    
    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=audio))
    
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [spectral_centroid, spectral_rolloff, spectral_bandwidth],
        [pitch_mean, pitch_std],
        [zcr],
        chroma_mean,
        [rms]
    ])
    return features

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if audio file was sent
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided"
            }), 400

        # Read audio file
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Load audio
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Extract features
        features = extract_features(audio.astype(np.float32))
        features = features.reshape(1, -1)
        
        # Load model and predict
        model = joblib.load("scam_detector_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = max(probability) * 100

        # Return result
        return jsonify({
            "prediction": int(prediction),
            "confidence": round(confidence, 2),
            "result": "AI_VOICE" if prediction == 1 else "REAL_HUMAN",
            "message": "⚠️ SCAM DETECTED!" if prediction == 1 else "✅ Genuine call",
            "is_scam": bool(prediction == 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🚀 Starting AI Scam Detector API...")
    print("📡 API running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)