import numpy as np
import librosa
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

SAMPLE_RATE = 22050

def extract_advanced_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=5)
        
        # 1. MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 2. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        
        # 3. Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 4. Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 6. RMS Energy
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,          # 40 features
            mfcc_std,           # 40 features
            [spectral_centroid, spectral_rolloff, spectral_bandwidth],  # 3
            [pitch_mean, pitch_std],  # 2
            [zcr],              # 1
            chroma_mean,        # 12 features
            [rms]               # 1
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_folder):
    X = []
    y = []
    
    # Real voices
    real_folder = os.path.join(data_folder, "real")
    if os.path.exists(real_folder):
        files = [f for f in os.listdir(real_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
        print(f"📂 Found {len(files)} real voice files")
        for i, file in enumerate(files):
            path = os.path.join(real_folder, file)
            features = extract_advanced_features(path)
            if features is not None:
                X.append(features)
                y.append(0)
            if (i+1) % 50 == 0:
                print(f"   Processed {i+1}/{len(files)} real files...")
    
    # Fake voices
    fake_folder = os.path.join(data_folder, "fake")
    if os.path.exists(fake_folder):
        files = [f for f in os.listdir(fake_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
        print(f"📂 Found {len(files)} fake voice files")
        for i, file in enumerate(files):
            path = os.path.join(fake_folder, file)
            features = extract_advanced_features(path)
            if features is not None:
                X.append(features)
                y.append(1)
            if (i+1) % 50 == 0:
                print(f"   Processed {i+1}/{len(files)} fake files...")
    
    return np.array(X), np.array(y)

def train():
    print("🚀 Starting Advanced Model Training...")
    print("=" * 50)
    
    X, y = load_dataset("data")
    
    if len(X) == 0:
        print("❌ No data found!")
        return
    
    print(f"\n✅ Total samples loaded: {len(X)}")
    print(f"   Real voices: {sum(y==0)}")
    print(f"   AI voices: {sum(y==1)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train advanced model
    print("\n🤖 Training Advanced Model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n🎯 Model Accuracy: {accuracy*100:.2f}%")
    print("\n📊 Detailed Report:")
    print(classification_report(y_test, predictions,
        target_names=['Real Human', 'AI Voice']))
    
    # Save model and scaler
    joblib.dump(model, "scam_detector_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n💾 Model saved successfully!")

if __name__ == "__main__":
    train()