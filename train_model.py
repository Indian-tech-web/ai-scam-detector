import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

SAMPLE_RATE = 22050

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfccs, axis=1)
    return features

def load_dataset():
    X = []
    y = []
    
    # Load real human voices (label = 0)
    real_folder = "data/real"
    if os.path.exists(real_folder):
        for file in os.listdir(real_folder):
            if file.endswith(".wav"):
                path = os.path.join(real_folder, file)
                features = extract_features(path)
                X.append(features)
                y.append(0)  # 0 = real human
    
    # Load AI generated voices (label = 1)
    fake_folder = "data/fake"
    if os.path.exists(fake_folder):
        for file in os.listdir(fake_folder):
            if file.endswith(".wav"):
                path = os.path.join(fake_folder, file)
                features = extract_features(path)
                X.append(features)
                y.append(1)  # 1 = AI voice
    
    return np.array(X), np.array(y)

def train():
    print("📂 Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ No data found! Please add audio files first.")
        return
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"   Real voices: {sum(y==0)}")
    print(f"   AI voices: {sum(y==1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\n🤖 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model accuracy: {accuracy*100:.2f}%")
    
    # Save model
    joblib.dump(model, "scam_detector_model.pkl")
    print("💾 Model saved as scam_detector_model.pkl")

if __name__ == "__main__":
    train()