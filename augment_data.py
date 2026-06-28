import librosa
import numpy as np
import soundfile as sf
import os
import random

SAMPLE_RATE = 22050

def add_background_noise(audio, noise_level=0.02):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def add_traffic_noise(audio, noise_level=0.015):
    # Simulate traffic — low frequency rumble
    t = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    traffic = np.sin(2 * np.pi * 80 * t) * noise_level
    traffic += np.random.randn(len(audio)) * noise_level * 0.5
    return audio + traffic

def add_crowd_noise(audio, noise_level=0.018):
    # Simulate crowd — multiple frequencies
    t = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    crowd = np.sin(2 * np.pi * 200 * t) * noise_level * 0.3
    crowd += np.sin(2 * np.pi * 350 * t) * noise_level * 0.2
    crowd += np.random.randn(len(audio)) * noise_level
    return audio + crowd

def add_fan_noise(audio, noise_level=0.012):
    # Simulate fan/cooler — constant hum
    t = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    fan = np.sin(2 * np.pi * 120 * t) * noise_level
    fan += np.sin(2 * np.pi * 240 * t) * noise_level * 0.3
    fan += np.random.randn(len(audio)) * noise_level * 0.2
    return audio + fan

def add_bus_noise(audio, noise_level=0.025):
    # Simulate bus/vehicle
    t = np.linspace(0, len(audio)/SAMPLE_RATE, len(audio))
    bus = np.sin(2 * np.pi * 60 * t) * noise_level
    bus += np.random.randn(len(audio)) * noise_level * 0.8
    return audio + bus

def augment_folder(input_folder, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) 
             if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"📂 Processing {len(files)} files from {input_folder}...")
    
    noise_functions = [
        add_background_noise,
        add_traffic_noise,
        add_crowd_noise,
        add_fan_noise,
        add_bus_noise
    ]
    
    saved = 0
    for i, file in enumerate(files):
        try:
            path = os.path.join(input_folder, file)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            
            # Apply 2 random noise types per file
            selected = random.sample(noise_functions, 2)
            for j, noise_fn in enumerate(selected):
                noisy = noise_fn(audio.copy())
                # Normalize
                noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
                out_name = f"{prefix}_aug_{saved}.wav"
                out_path = os.path.join(output_folder, out_name)
                sf.write(out_path, noisy, SAMPLE_RATE)
                saved += 1
            
            if (i+1) % 100 == 0:
                print(f"   Processed {i+1}/{len(files)} files...")
                
        except Exception as e:
            continue
    
    print(f"✅ Created {saved} augmented files in {output_folder}")
    return saved

if __name__ == "__main__":
    print("🔊 Starting noise augmentation...")
    print("=" * 50)
    
    # Augment real voices
    real_saved = augment_folder(
        "data/real", 
        "data/real", 
        "noisy_real"
    )
    
    # Augment fake voices  
    fake_saved = augment_folder(
        "data/fake",
        "data/fake",
        "noisy_fake"
    )
    
    print("\n" + "=" * 50)
    print(f"✅ Total augmented files created: {real_saved + fake_saved}")
    print("🚀 Now run: python train_deep_model.py")