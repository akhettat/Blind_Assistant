#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLINDY: Text-Based Assistant for Visually Impaired Users
Uses pyttsx3 for offline text-to-speech with dynamic speed control and language switching
:Author: Amine KHETTAT
:Date: 2024-11-06
:Version: 1.0
"""

import os
import sys
import threading
from queue import Queue
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pyttsx3
import keyboard
import time

class EnhancedVoiceAssistant:
    def __init__(self, model_path):
        # Initialize basic components
        self.setup_logging()
        self.speech_queue = Queue()
        self.stop_event = threading.Event()
        self.is_speaking = threading.Event()
        self.last_response = None
        self.speech_rate = 250  # Faster default speech rate
        self.current_engine = None  # Keep track of current engine
        self.current_text = None  # Keep track of current text being spoken
        
        # Set up language support
        self.setup_languages()
        
        # Load model
        print("Loading the AI model. This may take a few moments. Please wait...")
        self.model_path = model_path
        self.load_model()
        
        # Set up keyboard controls
        self.setup_keyboard_handlers()

    def setup_logging(self):
        """Set up logging to track errors and usage"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"assistant_{datetime.now().strftime('%Y%m%d')}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_languages(self):
        """Initialize available languages and current language"""
        # Get a temporary engine to get available voices
        temp_engine = pyttsx3.init()
        self.available_voices = temp_engine.getProperty('voices')
        temp_engine.stop()
        del temp_engine

        # Store voice information
        self.voices = []
        for voice in self.available_voices:
            # Get language code from voice ID (usually last 2 characters)
            lang_code = voice.id[-2:].lower()
            self.voices.append({
                'id': voice.id,
                'name': voice.name,
                'lang_code': lang_code
            })

        # Set initial voice to first available
        self.current_voice_index = 0
        print("\nAvailable voices:")
        for i, voice in enumerate(self.voices):
            print(f"{i+1}. {voice['name']} ({voice['lang_code']})")

    def cycle_voice(self):
        """Switch to the next available voice"""
        if len(self.voices) > 1:
            self.current_voice_index = (self.current_voice_index + 1) % len(self.voices)
            voice_info = self.voices[self.current_voice_index]
            message = f"Switched to {voice_info['name']}"
            print(f"System: {message}")
            
            # If currently speaking, restart with new voice
            if self.current_text and self.is_speaking.is_set():
                text_to_restart = self.current_text
                self.stop_speaking()
                self.speak_text(message)
                self.speak_text(text_to_restart)
            else:
                self.speak_text(message)

    def get_engine(self):
        """Create a new TTS engine instance with current settings"""
        engine = pyttsx3.init()
        voice_id = self.voices[self.current_voice_index]['id']
        engine.setProperty('voice', voice_id)
        engine.setProperty('rate', self.speech_rate)
        engine.setProperty('volume', 0.9)
        return engine

    def load_model(self):
        """Load the BLOOM model with error handling"""
        try:
            # Initialize tokenizer with explicit truncation settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                truncation=True,
                padding=True
            )
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Create pipeline with proper tokenizer settings
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                truncation=True,
                padding=True
            )
            
            self.speak_text("Model loaded successfully.")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            sys.exit(1)

    def setup_keyboard_handlers(self):
        """Set up keyboard shortcuts for controlling the assistant"""
        keyboard.add_hotkey('ctrl+q', self.quit_assistant)     # Quit application
        keyboard.add_hotkey('esc', self.stop_speaking)         # Stop current speech
        keyboard.add_hotkey('ctrl+r', self.repeat_last_response)  # Repeat last response
        keyboard.add_hotkey('ctrl+s', lambda: self.adjust_speech_rate(25, restart_speech=True))   # Speed up
        keyboard.add_hotkey('ctrl+d', lambda: self.adjust_speech_rate(-25, restart_speech=True))  # Slow down
        keyboard.add_hotkey('ctrl+l', self.cycle_voice)        # Change language/voice

    def adjust_speech_rate(self, change, restart_speech=False):
        """Adjust the speech rate and optionally restart current speech"""
        old_rate = self.speech_rate
        self.speech_rate = max(50, min(400, self.speech_rate + change))  # Allow higher maximum speed
        
        if self.speech_rate != old_rate:
            message = f"Speech rate adjusted to {self.speech_rate}"
            print(f"System: {message}")
            
            if restart_speech and self.current_text and self.is_speaking.is_set():
                # Store the current text
                text_to_restart = self.current_text
                # Stop current speech
                self.stop_speaking()
                # Restart speech with new rate
                self.speak_text(text_to_restart)
            else:
                # Just announce the rate change
                self.speak_text(message)

    def speak_text(self, text):
        """Speak text using offline TTS"""
        if self.stop_event.is_set():
            return

        print(f"Assistant: {text}")
        self.current_text = text  # Store current text being spoken
        
        try:
            # Clean up any existing engine
            if self.current_engine:
                try:
                    self.current_engine.stop()
                    del self.current_engine
                except:
                    pass
            
            # Create new engine instance
            self.current_engine = self.get_engine()
            self.is_speaking.set()
            
            # Set up callback for speech end
            def onEnd(name, completed):
                self.is_speaking.clear()
                if completed:  # Only cleanup if speech completed normally
                    self.current_text = None  # Clear current text
                    try:
                        self.current_engine.stop()
                        del self.current_engine
                        self.current_engine = None
                    except:
                        pass
            
            self.current_engine.connect('finished-utterance', onEnd)
            self.current_engine.say(text)
            self.current_engine.runAndWait()
            
        except Exception as e:
            print(f"Error in speech processing: {str(e)}")
            logging.error(f"Speech processing error: {str(e)}")
            self.is_speaking.clear()
            self.current_text = None
        finally:
            if not self.is_speaking.is_set() and self.current_engine:
                try:
                    self.current_engine.stop()
                    del self.current_engine
                    self.current_engine = None
                except:
                    pass

    def generate_response(self, user_input):
        """Generate response using the BLOOM model"""
        try:
            if not hasattr(self, 'generator'):
                return "Model not properly initialized. Please restart the assistant."

            response = self.generator(
                user_input,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                truncation=True,
                return_full_text=False
            )[0]['generated_text']
            
            response_text = response.strip()
            
            if not response_text or response_text.isspace():
                return "I apologize, but I couldn't generate a meaningful response. Could you please rephrase your question?"
            
            self.last_response = response_text
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(error_msg)
            return "I apologize, but I encountered an error generating a response. Please try again."

    def stop_speaking(self):
        """Stop current speech output"""
        self.stop_event.set()
        
        # Force stop current speech
        if self.current_engine and self.is_speaking.is_set():
            try:
                self.current_engine.stop()
                del self.current_engine
                self.current_engine = None
            except:
                pass
            
        self.is_speaking.clear()
        time.sleep(0.1)
        print("System: Speech stopped.")
        self.stop_event.clear()

    def repeat_last_response(self):
        """Repeat the last response"""
        if self.last_response:
            self.speak_text("Repeating last response:")
            self.speak_text(self.last_response)
        else:
            self.speak_text("No previous response to repeat.")

    def quit_assistant(self):
        """Safely quit the assistant"""
        self.stop_speaking()  # Stop any ongoing speech
        self.speak_text("Shutting down assistant. Goodbye!")
        time.sleep(1)
        logging.info("Assistant shutdown initiated by user")
        
        # Final cleanup
        if self.current_engine:
            try:
                self.current_engine.stop()
                del self.current_engine
            except:
                pass
                
        sys.exit(0)

    def run(self):
        """Main loop of the assistant"""
        welcome_message = """
        Welcome to the Enhanced Voice Assistant!
        You can use the following keyboard shortcuts:
        - Press Escape to stop current speech
        - Press Ctrl+R to repeat the last response
        - Press Ctrl+Q to quit the assistant
        - Press Ctrl+S to speed up speech (works during speech)
        - Press Ctrl+D to slow down speech (works during speech)
        - Press Ctrl+L to cycle through available voices
        
        Please enter your message and press Enter to begin.
        """
        
        self.speak_text(welcome_message)
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "stop"]:
                    self.quit_assistant()
                
                if not user_input:
                    self.speak_text("I didn't catch that. Please try again.")
                    continue
                
                self.speak_text("Processing your request...")
                response = self.generate_response(user_input)
                self.speak_text(response)

            except KeyboardInterrupt:
                self.quit_assistant()
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                logging.error(error_msg)
                self.speak_text(error_msg)

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "bloom_model")
    assistant = EnhancedVoiceAssistant(model_path)
    assistant.run()