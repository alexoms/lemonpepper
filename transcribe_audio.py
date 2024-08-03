import vosk
import numpy as np
import sounddevice as sd
import queue
import json
import threading
import time
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTranscriber:
    def __init__(self, model_path="../vosk-model-small-en-us-0.15/", sample_rate=44100, channels=2, chunk=1024):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.audio_queue = queue.Queue()
        self.transcription_callback = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.audio_thread = None
        self.input_stream = None
        logging.info("AudioTranscriber initialized")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.pause_event.is_set():
            self.audio_queue.put(indata.copy())
            logging.debug("Audio data added to queue")

    def process_audio(self):
        audio_buffer = np.array([])
        logging.info("Audio processing started")

        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                mono_data = audio_data.mean(axis=1)
                mono_data = mono_data[::3]  # Downsample from 44.1kHz to 16kHz

                audio_buffer = np.concatenate((audio_buffer, mono_data))

                if len(audio_buffer) >= 8192:
                    audio_int16 = (audio_buffer * 32767).astype(np.int16)

                    if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        if result['text']:
                            logging.info(f"Full transcription: {result['text']}")
                            if self.transcription_callback:
                                self.transcription_callback(result['text'], is_partial=False)
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        if partial['partial']:
                            logging.info(f"Partial transcription: {partial['partial']}")
                            if self.transcription_callback:
                                self.transcription_callback(partial['partial'], is_partial=True)

                    audio_buffer = np.array([])

            except queue.Empty:
                continue

    def list_audio_devices(self):
        devices = sd.query_devices()
        return [f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})"
                for i, device in enumerate(devices)]

    def get_device_choice(self):
        self.list_audio_devices()
        while True:
            try:
                choice = int(input("Enter the number of the audio input device: "))
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

    def start_transcribing(self, device_index=None, transcription_callback=None):
        self.transcription_callback = transcription_callback
        self.stop_event.clear()
        self.pause_event.clear()

        self.audio_thread = threading.Thread(target=self.process_audio)
        self.audio_thread.start()
        logging.info(f"Started audio thread with device {device_index}")

        try:
            self.input_stream = sd.InputStream(callback=self.audio_callback, channels=self.channels,
                                               samplerate=self.sample_rate, blocksize=self.chunk,
                                               device=device_index)
            self.input_stream.start()
            logging.info(f"Listening to audio from device {device_index}")
            while not self.stop_event.is_set():
                sd.sleep(1000)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, stopping...")
        except sd.PortAudioError as e:
            logging.error(f"PortAudioError: {e}")
            logging.error("This might be because the selected device doesn't support the specified settings.")
            logging.error("Try a different device or adjust the audio settings.")

        self.stop_event.set()
        self.audio_thread.join()
        logging.info("Audio thread stopped")

    def stop_transcribing(self):
        self.stop_event.set()
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        logging.info("Stop event set for audio transcription")

    def pause_transcribing(self):
        self.pause_event.set()
        logging.info("Audio transcription paused")

    def resume_transcribing(self):
        self.pause_event.clear()
        logging.info("Audio transcription resumed")
