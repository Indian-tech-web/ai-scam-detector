import librosa
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 22050

def add_light_noise(audio, level=0.008):
    noise = np.random.randn(len(audio)) * level
    return np.clip(audio + noise, -1, 1)

def add_fan_hum(audio, level=0.005):
    t = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    hum = np.sin(2 * np.pi * 120 * t) * level
    hum += np.sin(2 * np.pi * 240 * t) * level * 0.3
    return np.clip(audio + hum, -1, 1)

def augment_mic_samples(folder, prefix, label):
    files = [f for f in os.listdir(folder) if f.startswith("mic_")]
    print(f"📂 Found {len(files)} mic_{label} files to augment")
    
    saved = 0
    for f in files:
        path = os.path.join(folder, f)
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        
        # Version 1 — light background noise
        noisy1 = add_light_noise(audio)
        sf.write(os.path.join(folder, f"smart_{prefix}_{saved}.wav"), noisy1, SAMPLE_RATE)
        saved += 1
        
        # Version 2 — fan/cooler hum
        noisy2 = add_fan_hum(audio)
        sf.write(os.path.join(folder, f"smart_{prefix}_{saved}.wav"), noisy2, SAMPLE_RATE)
        saved += 1

    print(f"✅ Created {saved} augmented files")

if __name__ == "__main__":
    print("🔊 Smart augmentation — mic samples only!")
    augment_mic_samples("data/real", "real", "real")
    augment_mic_samples("data/fake", "fake", "fake")
    print("\n✅ Done! Now run: python train_deep_model.py")