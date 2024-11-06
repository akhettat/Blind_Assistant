"""
===============================================================================
Voice Assistant Script with Silence Detection and Real-Time Audio Processing
===============================================================================
"""

import os
import queue
import sounddevice as sd
import pyttsx3
import threading
import time
import keyboard
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk

# Télécharger les ressources de NLTK si elles ne sont pas encore disponibles
nltk.download('punkt')  # Utilisé pour la tokenisation des phrases

# Configurations audio
sample_rate = 16000
block_size = 32000
amplitude_threshold = 0.001  # Ajusté pour une meilleure sensibilité
silence_limit = 2   # Durée du silence en secondes

# Charger les modèles Wav2Vec2 et GPT-2
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

q = queue.Queue()
speech_queue = queue.Queue()
engine = pyttsx3.init()
speaking = False
assistant_active = False

def toggle_assistant_state():
    global assistant_active
    assistant_active = not assistant_active
    if assistant_active:
        print("L'Assistant est activé")
        speech_queue.put("L'assistant est activé")
    else:
        print("L'Assistant est désactivé")
        speech_queue.put("L'assistant est désactivé")

def stop_speaking():
    global speaking
    if speaking:
        engine.stop()
        speaking = False
        print("Speaking interrupted by user.")
        assistant_active = True

# Définition du raccourci pour arrêter la parole
keyboard.add_hotkey('ctrl+shift+²', stop_speaking)

def speak_worker():
    """
    Fonction de gestion de la synthèse vocale.
    """
    global speaking
    while True:
        response = speech_queue.get()
        if response is None:
            break
        print(f"Synthesizing speech: {response}")
        speaking = True
        engine.say(response)
        engine.runAndWait()
        speaking = False
        print(f"Finished speaking: {response}")

def generate_response(prompt):
    """
    Utilise GPT-2 pour générer une réponse basée sur le prompt donné.
    """
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(**inputs, max_length=100)
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_command(command_text):
    """
    Traite le texte de la commande en utilisant NLTK pour extraire des mots-clés
    et formuler une réponse appropriée.
    """
    print(f"Processing command: {command_text}")
    
    # Tokenisation des mots
    words = nltk.word_tokenize(command_text.lower())
    
    if "heure" in words or "time" in words:
        import datetime
        now = datetime.datetime.now()
        response = f"Il est {now.hour} heure et {now.minute} minutes."
    elif "bonjour" in words or "hello" in words:
        response = "Bonjour, comment puis-je vous aider aujourd'hui ?"
    else:
        # Si aucun mot-clé particulier n'est détecté, passer le texte à GPT-2
        response = generate_response(command_text)
    
    print(f"Queuing response: {response}")
    speech_queue.put(response)

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

def recognize_and_respond():
    silence_duration = 0
    with sd.RawInputStream(samplerate=sample_rate, blocksize=block_size, dtype='int16', channels=1, callback=callback):
        print("Assistant is listening... Speak into the microphone.")
        
        while True:
            if assistant_active and not speaking:
                data = q.get()
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_normalized = audio_data / 32768  # Normalisation dans l'intervalle [-1, 1]
                audio_amplitude = np.abs(audio_normalized).mean()
                
                print(f"Captured audio amplitude: {audio_amplitude:.4f}")
                
                if audio_amplitude > amplitude_threshold:
                    silence_duration = 0
                    print("Audio detected. Transcribing...")

                    # Prétraitement audio et transcription
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).div_(32767.0).unsqueeze(0)
                    input_values = tokenizer(audio_tensor.squeeze().numpy(), return_tensors="pt", padding="longest").input_values
                    logits = wav2vec_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    recognized_text = tokenizer.batch_decode(predicted_ids)[0]
                    
                    print(f"Recognized text: {recognized_text}")

                    if recognized_text:
                        process_command(recognized_text)
                else:
                    silence_duration += 1 / sample_rate * block_size
                    print(f"Silence duration: {silence_duration:.2f} seconds")
                    if silence_duration >= silence_limit:
                        print("Detected prolonged silence. Ending recording session.")
                        break

if __name__ == "__main__":
    threading.Thread(target=speak_worker, daemon=True).start()
    keyboard.add_hotkey('ctrl+shift+a', toggle_assistant_state)

    try:
        recognize_and_respond()
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user.")
        speech_queue.put(None)
