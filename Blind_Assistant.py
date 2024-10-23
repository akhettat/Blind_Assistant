import os
import queue
import sounddevice as sd
import vosk
import pyttsx3
import json
import nltk
import threading
import time
import keyboard  # For handling keyboard shortcuts

# Initialize the Vosk model for speech recognition
model_path = "vosk-model-fr-0.6-linto-2.2.0"  # Path to the French Vosk model
if not os.path.exists(model_path):
    print(f"Model not found in {model_path}")
    exit(1)

model = vosk.Model(model_path)

# Initialize a queue for audio processing
q = queue.Queue()
speech_queue = queue.Queue()  # Queue for managing speech synthesis

# Flag to track if the assistant is active
assistant_active = False  # Initially inactive

def toggle_assistant_state():
    """
    Toggle the assistant's active state and announce it.
    """
    global assistant_active
    assistant_active = not assistant_active
    if assistant_active:
        print("l'Assistant est activé")  # Debug message
        speech_queue.put("l'assistant est activé")
    else:
        print("l'Assistant est désactivé")  # Debug message
        speech_queue.put("l'assistant est désactivé")

def speak_worker():
    """
    Worker function that continuously runs and processes speech from the queue.
    This avoids overlapping of speech synthesis processes.
    """
    while True:
        response = speech_queue.get()
        if response is None:
            break
        print(f"Synthesizing speech: {response}")  # Debugging: print the response to ensure it's being processed
        
        # Re-initialize pyttsx3 engine for each response
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        engine.stop()  # Stop the engine and reset it
        
        print(f"Finished speaking: {response}")  # Debugging message after speaking
        time.sleep(1)  # Add a delay to ensure speech finishes properly

def process_command(command_text):
    """
    Process the recognized speech and queue a response for synthesis.
    """
    # Simple natural language processing using keywords
    print(f"Processing command: {command_text}")  # Debugging message
    if "heure" in command_text:
        import datetime
        now = datetime.datetime.now()
        response = f"il est {now.hour} heure et {now.minute} minutes."
    elif "bonjour" in command_text:
        response = "bonjour, comment puis-je vous aider aujourd'hui ?"
    else:
        response = "désolé, je n'ai pas compris."

    print(f"Queuing response: {response}")  # Debugging message
    # Add the response to the speech queue for synthesis
    speech_queue.put(response)

def callback(indata, frames, time, status):
    """
    This callback function is invoked for each block of audio data.

    It continuously feeds the audio data into the recognizer and prints
    the recognized text when complete sentences are recognized.
    """
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

# Capture live audio and process it with Vosk
def recognize_and_respond():
    """
    Start capturing live audio using the microphone and recognize speech in real-time.
    """
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("Assistant is listening... Speak into the microphone.")
        
        rec = vosk.KaldiRecognizer(model, 16000)
        
        while True:
            if assistant_active:  # Only process commands if the assistant is active
                data = q.get()
                try:
                    if rec.AcceptWaveform(data):
                        result = rec.Result()  # Get the full recognition result in JSON format
                        result_dict = json.loads(result)  # Parse JSON string into a dictionary
                        recognized_text = result_dict.get('text')
                        print(f"Recognized text: {recognized_text}")  # Debugging message
                        if recognized_text:
                            process_command(recognized_text)
                    else:
                        partial_result = rec.PartialResult()  # Get partial results during recognition
                        partial_dict = json.loads(partial_result)
                        print(f"Partial recognition: {partial_dict.get('partial')}")
                except Exception as e:
                    print(f"Error processing waveform: {e}")

if __name__ == "__main__":
    """
    Main entry point for the script.

    Initializes the required resources, starts the audio input stream,
    and processes speech commands in a continuous loop.
    """
    # Download nltk data if needed
    nltk.download('punkt')

    # Start the speech synthesis worker thread
    threading.Thread(target=speak_worker, daemon=True).start()

    # Set up a keyboard shortcut to toggle the assistant's state
    keyboard.add_hotkey('ctrl+shift+a', toggle_assistant_state)  # Ctrl+Shift+A to toggle activation

    try:
        recognize_and_respond()
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user.")
        speech_queue.put(None)  # Stop the speech worker thread
