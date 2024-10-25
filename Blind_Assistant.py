"""
===============================================================================
Voice Assistant Script with Silence Detection and Real-Time Audio Processing
===============================================================================

Description:
    This script is a comprehensive implementation of a voice assistant designed
    to operate offline, featuring real-time speech recognition, natural language
    processing, and text-to-speech synthesis. The assistant uses the Wav2Vec2 
    model for speech transcription and can respond to various commands using 
    a local language model. It also includes silence detection to end audio 
    capture after prolonged silence.

Modules Used:
    - sounddevice: Captures real-time audio input from the microphone.
    - wav2vec2: Transcribes audio data using the Facebook Wav2Vec2 model.
    - pyttsx3: Converts text responses to speech.
    - keyboard: Enables activation and deactivation of the assistant using
      keyboard shortcuts.

Features:
    - Offline speech recognition via Wav2Vec2 model.
    - Real-time command processing.
    - Dynamic responses using pre-trained language models.
    - Activation toggle through keyboard shortcuts.
    - Silence detection to end audio capture.

Author: Amine Khettat
Date Created: 22/10/2024
Last Modified: 25/10/2024
Version: 0.1

===============================================================================
"""

import os
import queue
import sounddevice as sd
import pyttsx3
import json
import threading
import time
import keyboard  # For handling keyboard shortcuts
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk  # Import de nltk

# Configurer les paramètres audio
sample_rate = 16000
block_size = 16000
amplitude_threshold = 0.02  # Seuil d'amplitude pour détecter la voix
silence_limit = 2  # Temps en secondes pour détecter un silence prolongé

# Charger les modèles Wav2Vec2 pour la transcription audio et GPT-2 pour les réponsesmodel_name = 
model_name = "facebook/wav2vec2-base-960h"  #Name of the model used to transcript the acquired text
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialiser les files d'attente pour l'audio et la synthèse vocale
q = queue.Queue()
speech_queue = queue.Queue()  # Queue for managing speech synthesis

# Flag pour indiquer si l'assistant est actif
assistant_active = False  # Initialement inactif

def toggle_assistant_state():
    """
    Active ou désactive l'assistant et annonce l'état actuel.
    """
    global assistant_active
    assistant_active = not assistant_active
    if assistant_active:
        print("l'Assistant est activé")
        speech_queue.put("l'assistant est activé")
    else:
        print("l'Assistant est désactivé")
        speech_queue.put("l'assistant est désactivé")

def speak_worker():
    """
    Fonction de travail qui traite la synthèse vocale depuis la file d'attente.
    """
    while True:
        response = speech_queue.get()
        if response is None:
            break
        print(f"Synthesizing speech: {response}")
        
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        engine.stop()
        
        print(f"Finished speaking: {response}")
        time.sleep(1)

def generate_response(prompt):
    """
    Génère une réponse à partir du modèle GPT-2 en utilisant le texte fourni.
    """
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(**inputs, max_length=100)
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_command(command_text):
    """
    Traite le texte de la commande reconnu et met une réponse dans la file d'attente de synthèse vocale.
    """
    print(f"Processing command: {command_text}")
    if "heure" in command_text:
        import datetime
        now = datetime.datetime.now()
        response = f"il est {now.hour} heure et {now.minute} minutes."
    elif "bonjour" in command_text:
        response = "bonjour, comment puis-je vous aider aujourd'hui ?"
    else:
        # Si la commande n'est pas spécifique, transmettre le texte au modèle GPT-2
        response = generate_response(command_text)

    print(f"Queuing response: {response}")
    speech_queue.put(response)

def callback(indata, frames, time, status):
    """
    Cette fonction de rappel est invoquée pour chaque bloc de données audio.
    """
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

# Capture et traitement audio en temps réel avec Wav2Vec2
def recognize_and_respond():
    """
    Capture et traitement de l'audio en temps réel avec Wav2Vec2, avec détection de silence.
    """
    silence_duration = 0  # Durée du silence
    with sd.RawInputStream(samplerate=sample_rate, blocksize=block_size, dtype='int16', channels=1, callback=callback):
        print("Assistant is listening... Speak into the microphone.")
        
        while True:
            if assistant_active:
                data = q.get()
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_amplitude = np.abs(audio_data).mean()
                
                print(f"Captured audio amplitude: {audio_amplitude:.4f}")
                
                # Détecte la voix et réinitialise le silence si le seuil est dépassé
                if audio_amplitude > amplitude_threshold:
                    silence_duration = 0
                    print("Audio detected. Transcribing...")

                    # Transcription directe depuis l'entrée audio
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
                    input_values = tokenizer(audio_tensor.squeeze().numpy(), return_tensors="pt", padding="longest").input_values
                    logits = wav2vec_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    recognized_text = tokenizer.batch_decode(predicted_ids)[0]
                    
                    print(f"Recognized text: {recognized_text}")

                    if recognized_text:
                        process_command(recognized_text)
                else:
                    # Si le silence persiste au-delà de la limite
                    silence_duration += 1 / sample_rate * block_size
                    print(f"Silence duration: {silence_duration:.2f} seconds")
                    if silence_duration >= silence_limit:
                        print("Detected prolonged silence. Ending recording session.")
                        break  # Arrête la capture après un silence prolongé

if __name__ == "__main__":
    """
    Point d'entrée principal pour le script.
    """
    nltk.download('punkt')

    threading.Thread(target=speak_worker, daemon=True).start()

    # Configurer un raccourci clavier pour activer/désactiver l'assistant
    keyboard.add_hotkey('ctrl+shift+a', toggle_assistant_state)

    try:
        recognize_and_respond()
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user.")
        speech_queue.put(None)  # Stop the speech worker thread
