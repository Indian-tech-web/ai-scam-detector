from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import librosa
import joblib
import io
import tempfile
import soundfile as sf

app = Flask(__name__)

SAMPLE_RATE = 22050

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE))
    pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma, axis=1)
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

def advanced_ai_checks(audio):
    score = 0
    reasons = []
    noise_level = np.std(audio[:1000])
    if noise_level < 0.001:
        score += 1
        reasons.append("Too clean — no background noise")
    silence_threshold = 0.01
    silent_frames = np.sum(np.abs(audio) < silence_threshold)
    silence_ratio = silent_frames / len(audio)
    if silence_ratio < 0.1:
        score += 1
        reasons.append("No natural pauses detected")
    pitches, _ = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) > 0:
        pitch_smoothness = np.std(pitch_values)
        if pitch_smoothness < 500:
            score += 1
            reasons.append("Pitch too smooth — AI pattern")
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    if flatness < 0.01:
        score += 1
        reasons.append("Spectral pattern matches AI voice")
    return score, reasons

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()

        # Save to temp file and load with librosa
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp_path)

        audio = audio.astype(np.float32)

        features = extract_features(audio)
        features = features.reshape(1, -1)

        model = joblib.load("scam_detector_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = max(probability) * 100

        ai_score, reasons = advanced_ai_checks(audio)

        if prediction == 1 or ai_score >= 2:
            is_scam = True
            message = "⚠️ SCAM DETECTED!"
        else:
            is_scam = False
            message = "✅ Genuine call!"

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(confidence, 2),
            "result": "AI_VOICE" if is_scam else "REAL_HUMAN",
            "message": message,
            "is_scam": is_scam,
            "ai_indicators": reasons,
            "ai_score": ai_score
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🚀 Starting AI Scam Detector API...")
    print("📡 API running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)