import vosk
import numpy as np
import sounddevice as sd
import queue
import json
import threading
import time
from OllamaAPI import OllamaAPI

# Create an instance of OllamaAPI
ollama_api = OllamaAPI(host="http://192.168.1.81:11434",model="llama3.1:latest")  # Replace with your actual remote host if applicable


# Vosk model and recognizer setup
model = vosk.Model("vosk-model-small-en-us-0.15/")
recognizer = vosk.KaldiRecognizer(model, 16000)

# Audio settings
CHANNELS = 2
RATE = 44100
CHUNK = 1024

# Queue to store audio data
audio_queue = queue.Queue()

# Callback function for sounddevice
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())
    #print(f"Max volume: {np.max(np.abs(indata)):.4f}", end='\r')

# Function to process audio data and transcribe
def process_audio():
    audio_buffer = np.array([])
    
    while True:
        audio_data = audio_queue.get()
        # Convert stereo to mono by averaging channels
        mono_data = audio_data.mean(axis=1)
        # Downsample from 44.1kHz to 16kHz
        mono_data = mono_data[::3]
        
        # Append to buffer
        audio_buffer = np.concatenate((audio_buffer, mono_data))
        
        if len(audio_buffer) >= 8192:  # Process in larger chunks
            print(f"\nProcessing {len(audio_buffer)} samples")
            print(f"Audio buffer stats: min={np.min(audio_buffer):.4f}, max={np.max(audio_buffer):.4f}, mean={np.mean(audio_buffer):.4f}")
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_buffer * 32767).astype(np.int16)
            
            if recognizer.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(recognizer.Result())
                print(f"Full result: {result}")
                if result['text']:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] Transcription: {result['text']}")
                    
                    # Add transcription to the buffer
                    ollama_api.add_transcription(result['text'])
                    
                    # Check if we should process the accumulated transcription
                    if ollama_api.should_process():
                        print("SENDING TO OLLAMA")
                        ollama_api.process_transcription()
            else:
                partial = json.loads(recognizer.PartialResult())
                print(f"Partial result: {partial}")
                if partial['partial']:
                    print(f"\nPartial: {partial['partial']}")
            
            audio_buffer = np.array([])  # Clear the buffer

# Function to list audio devices
def list_audio_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")

# Function to get user's device choice
def get_device_choice():
    list_audio_devices()
    while True:
        try:
            choice = int(input("Enter the number of the BlackHole 2ch device: "))
            devices = sd.query_devices()
            if 0 <= choice < len(devices):
                if devices[choice]['max_input_channels'] > 0:
                    return choice
                else:
                    print("This device doesn't support input. Please choose a device with input channels.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Get user's device choice
device_index = get_device_choice()

# Start the audio processing thread
audio_thread = threading.Thread(target=process_audio)
audio_thread.start()

# Start the sounddevice InputStream with the chosen device
try:
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=RATE, blocksize=CHUNK, device=device_index):
        print(f"Listening to audio from device {device_index}. Press Ctrl+C to stop.")
        while True:
            sd.sleep(10)
except KeyboardInterrupt:
    print("\nStopping...")
except sd.PortAudioError as e:
    print(f"Error: {e}")
    print("This might be because the selected device doesn't support the specified settings.")
    print("Try a different device or adjust the audio settings.")

# Clean up
audio_thread.join()