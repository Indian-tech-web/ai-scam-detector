import librosa
import soundfile as sf
import os

SAMPLE_RATE = 22050
CHUNK_DURATION = 3  # seconds

def chunk_audio(input_file, output_folder, prefix):
    print(f"📂 Loading {input_file}...")
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    total_chunks = len(audio) // chunk_samples
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find existing files to continue numbering
    existing = [f for f in os.listdir(output_folder) if f.startswith(prefix)]
    start_num = len(existing) + 1
    
    saved = 0
    for i in range(total_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]
        
        # Skip near-silent chunks
        energy = (chunk ** 2).mean() ** 0.5
        if energy < 0.005:
            continue
        
        filename = os.path.join(output_folder, f"{prefix}_{start_num + saved}.wav")
        sf.write(filename, chunk, SAMPLE_RATE)
        saved += 1
    
    print(f"✅ Created {saved} chunks in {output_folder}")

if __name__ == "__main__":
    choice = input("Chunk 'real' or 'fake' recording? ").strip().lower()
    
    input_file = f"long_{choice}_recording.wav"
    output_folder = f"data/{choice}"
    prefix = "mic_real" if choice == "real" else "mic_fake"
    
    chunk_audio(input_file, output_folder, prefix)