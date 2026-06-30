from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import librosa
import joblib
import tempfile


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

print("Loading model...")
model = joblib.load("scam_detector_model.pkl")
scaler = joblib.load("scaler.pkl")
print("Model loaded successfully!")

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

        # Detect format from filename
        filename = audio_file.filename or 'recording.webm'
        ext = '.m4a' if filename.endswith('.m4a') else '.webm'
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Convert webm to wav using ffmpeg
        import subprocess
        wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(wav_fd)
        # Works on both Windows and Linux/Render
        ffmpeg_path = r"C:\Users\LOQ\Desktop\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg"  # Linux/Render pe system ffmpeg use karo
        result = subprocess.run([
            ffmpeg_path,
            "-y", "-i", tmp_path, "-ar", "22050", "-ac", "1", wav_path
        ], capture_output=True)
        print(f"FFmpeg result: {result.returncode}")
        print(f"FFmpeg stderr: {result.stderr.decode()[:200]}")
        import soundfile as sf2
        audio, sr = sf2.read(wav_path, dtype='float32', always_2d=False)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        import resampy
        if sr != SAMPLE_RATE:
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        os.unlink(wav_path)
        os.unlink(tmp_path)
        audio = audio.astype(np.float32)
        
        import soundfile as sf3
        sf3.write("debug_flutter_recording.wav", audio, SAMPLE_RATE)

        print(f"DEBUG: audio length={len(audio)}, energy={np.sqrt(np.mean(audio**2)):.5f}")

        energy = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0
        if energy < 0.003:
            return jsonify({
                "confidence": 0,
                "result": "NO_VOICE",
                "message": "🎤 No voice detected. Please speak clearly.",
                "is_scam": False
            })

        features = extract_features(audio).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0]
        confidence = round(max(probability) * 100, 2)

        print(f"DEBUG: prediction={prediction}, confidence={confidence}")

        is_scam = (prediction == 1)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "result": "AI_VOICE" if is_scam else "REAL_HUMAN",
            "message": "⚠️ AI VOICE DETECTED! Possible SCAM!" if is_scam else "✅ Real Human Voice! Genuine Call!",
            "is_scam": is_scam
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🚀 Starting AI Scam Detector API...")
    app.run(debug=True, host='0.0.0.0', port=5000)