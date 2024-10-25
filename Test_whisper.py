import whisper
import os

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully.")

audio_file_path = "C:\\Users\\akhettat\\Documents\\Projets Python\\Blind_Assistant\\test_audio.wav"

if os.path.exists(audio_file_path):
    print("File exists and is accessible.")
    try:
        result = model.transcribe(audio_file_path)
        print("Transcription:", result["text"])
    except Exception as e:
        print("Error during transcription:", e)
else:
    print("Error: File not found.")
