from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

# Charger le modèle et le processeur pour le français
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Charger et pré-traiter l'audio
audio_input, _ = sf.read("test_audio.wav")

inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Décodage de la sortie
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)
