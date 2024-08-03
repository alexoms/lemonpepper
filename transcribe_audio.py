import vosk
import numpy as np
import sounddevice as sd
import queue
import json
import threading
import time
import logging
import webrtcvad
import collections

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTranscriber:
    def __init__(self, model_path="../vosk-model-en-us-0.42-gigaspeech/", sample_rate=16000, channels=1, chunk=480):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.audio_queue = queue.Queue()
        self.transcription_callback = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.audio_thread = None
        self.input_stream = None
        self.vad = webrtcvad.Vad(1)  # VAD aggressiveness is set to 3 (highest)
        self.ring_buffer = collections.deque(maxlen=30)  # 30 * 30ms = 900ms audio buffer
        self.silence_threshold = 2.0  # 2 seconds of silence before considering it a pause
        self.min_transcription_length = 3  # Minimum number of words to consider a transcription valid
        logging.info("AudioTranscriber initialized with VAD")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.pause_event.is_set():
            self.audio_queue.put(indata.copy())

    def process_audio(self):
        audio_buffer = np.array([])
        logging.info("Audio processing started with VAD")
        last_speech_time = time.time()
        continuous_silence = 0
        current_partial = ""

        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                mono_data = audio_data.mean(axis=1) if audio_data.ndim > 1 else audio_data
                
                audio_int16 = (mono_data * 32767).astype(np.int16)
                
                # Process in 30ms chunks (480 samples at 16kHz)
                for i in range(0, len(audio_int16), self.chunk):
                    chunk = audio_int16[i:i+self.chunk]
                    if len(chunk) == self.chunk:
                        is_speech = self.vad.is_speech(chunk.tobytes(), self.sample_rate)
                        self.ring_buffer.append((chunk, is_speech))
                        if is_speech:
                            last_speech_time = time.time()
                            continuous_silence = 0
                        else:
                            continuous_silence += self.chunk / self.sample_rate

                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                current_time = time.time()
                
                if num_voiced > 0.9 * self.ring_buffer.maxlen:
                    voiced_audio = np.concatenate([chunk for chunk, _ in self.ring_buffer])
                    
                    if self.recognizer.AcceptWaveform(voiced_audio.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        if result['text'] and len(result['text'].split()) >= self.min_transcription_length:
                            logging.info(f"Full transcription: {result['text']}")
                            if self.transcription_callback:
                                self.transcription_callback(result['text'], is_partial=False)
                            current_partial = ""  # Reset partial transcription
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        if partial['partial'] and len(partial['partial'].split()) >= self.min_transcription_length:
                            logging.info(f"Partial transcription: {partial['partial']}")
                            if self.transcription_callback:
                                self.transcription_callback(partial['partial'], is_partial=True)
                            current_partial = partial['partial']  # Update current partial transcription
                    
                    self.ring_buffer.clear()
                elif continuous_silence > self.silence_threshold:
                    # If there's been silence for more than the threshold
                    if current_partial and len(current_partial.split()) >= self.min_transcription_length:
                        # Send the current partial transcription as a full transcription
                        logging.info(f"Silence detected, sending partial as full: {current_partial}")
                        if self.transcription_callback:
                            self.transcription_callback(current_partial, is_partial=False)
                    else:
                        # If no valid partial transcription, just log the silence detection
                        logging.info("Silence detected, no valid transcription to send")
                    
                    self.ring_buffer.clear()
                    continuous_silence = 0
                    current_partial = ""  # Reset partial transcription

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
