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
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )
    return np.mean(mfccs, axis=1)

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