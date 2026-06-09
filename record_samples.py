import sounddevice as sd
import numpy as np
import wave
import os

SAMPLE_RATE = 22050
DURATION = 3  # 3 seconds each

def record_and_save(filename):
    print(f"\n🎤 Recording: {filename}")
    print("Speak naturally for 3 seconds...")
    
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    # Save as wav file
    audio_int = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int.tobytes())
    
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    print("🎙️ We will record 5 samples of YOUR real voice")
    print("Say anything naturally — count numbers, talk, etc.\n")
    
    for i in range(1, 6):
        input(f"Press ENTER when ready for sample {i}...")
        record_and_save(f"data/real/real_{i}.wav")
    
    print("\n✅ All 5 real voice samples recorded!")
    print("Now run: python train_model.py")