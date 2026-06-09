import sounddevice as sd
import numpy as np
import librosa
import joblib

SAMPLE_RATE = 22050
DURATION = 5

def record_audio():
    print("\n🎤 Recording... Please speak for 5 seconds")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✅ Recording done!")
    return audio.flatten()

def extract_features(audio):
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )
    return np.mean(mfccs, axis=1)

def predict_voice(audio):
    # Load trained model
    model = joblib.load("scam_detector_model.pkl")
    
    # Extract features
    features = extract_features(audio)
    features = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    confidence = max(probability) * 100
    
    print("\n📊 Voice Analysis Result:")
    print(f"Confidence: {confidence:.2f}%")
    
    if prediction == 1:
        print("⚠️  WARNING: AI GENERATED VOICE DETECTED!")
        print("🚨 This is likely a SCAM call!")
    else:
        print("✅ This sounds like a REAL human voice!")
        print("😊 This call appears genuine!")

if __name__ == "__main__":
    print("🔍 AI Scam Call Detector v2.0")
    print("================================")
    
    while True:
        input("\nPress ENTER to analyze a voice (Ctrl+C to quit)...")
        audio = record_audio()
        predict_voice(audio)