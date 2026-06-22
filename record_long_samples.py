import sounddevice as sd
import numpy as np
import wave
import os

SAMPLE_RATE = 22050

def record_long(filename, duration_seconds):
    print(f"\n🎤 Recording for {duration_seconds} seconds...")
    print("Bolna shuru karo... (ya AI voice play karo speaker se)")
    
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    audio_int = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int.tobytes())
    
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("LONG RECORDING TOOL")
    print("=" * 50)
    
    choice = input("\nRecord REAL (your voice) or FAKE (AI voice)? Type 'real' or 'fake': ").strip().lower()
    
    minutes = float(input("Kitne minutes record karna hai? (e.g. 5): "))
    duration = minutes * 60
    
    input(f"\nPress ENTER when ready to record {minutes} minutes...")
    
    filename = f"long_{choice}_recording.wav"
    record_long(filename, duration)
    
    print(f"\n✅ Done! Ab 'chunk_audio.py' chalao to split karne ke liye.")