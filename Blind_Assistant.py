import os
import queue
import sounddevice as sd
import vosk
import pyttsx3
import json
import nltk

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the Vosk model for speech recognition
model_path = "vosk-model-fr-0.6-linto-2.2.0"  # Path to the French Vosk model
if not os.path.exists(model_path):
    print(f"Model not found in {model_path}")
    exit(1)

model = vosk.Model(model_path)

# Initialize a queue for audio processing
q = queue.Queue()

def process_command(command_text):
    """
    Process the recognized speech and return a response.

    This function uses simple keyword-based logic to determine
    the appropriate response based on the recognized command.

    :param command_text: The recognized speech in text format
    :type command_text: str
    :return: The response to be spoken by the assistant
    :rtype: str
    """
    # Simple natural language processing using keywords
    if "heure" in command_text:
        import datetime
        now = datetime.datetime.now()
        response = f"It is {now.hour} hours and {now.minute} minutes."
    elif "bonjour" in command_text:
        response = "Hello, how can I assist you today?"
    else:
        response = "Sorry, I didn't understand."

    return response

def callback(indata, frames, time, status):
    """
    This callback function is invoked for each block of audio data.

    It continuously feeds the audio data into the recognizer and prints
    the recognized text when complete sentences are recognized.

    :param indata: The audio data captured from the microphone
    :param frames: Number of frames per block
    :param time: Timing information for the audio stream
    :param status: The status of the audio input stream
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
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()  # Get the full recognition result in JSON format
                result_dict = json.loads(result)  # Parse JSON string into a dictionary
                print(f"Recognized text: {result_dict.get('text')}")
            else:
                partial_result = rec.PartialResult()  # Get partial results during recognition
                partial_dict = json.loads(partial_result)
                print(f"Partial recognition: {partial_dict.get('partial')}")

if __name__ == "__main__":
    """
    Main entry point for the script.

    Initializes the required resources, starts the audio input stream,
    and processes speech commands in a continuous loop.
    """
    # Download nltk data if needed
    nltk.download('punkt')

    try:
        recognize_and_respond()
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user.")
