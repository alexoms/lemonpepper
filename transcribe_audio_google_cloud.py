import queue
import sounddevice as sd
import numpy as np
from google.cloud import speech
import time
import threading

class AudioTranscriberGoogleCloud:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1600):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.client = speech.SpeechClient()
        self.audio_levels = [0] * channels
        self.gain = 1.0

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        # Apply gain
        indata = indata * self.gain
        # Calculate RMS for each channel
        self.audio_levels = [np.sqrt(np.mean(indata[:, i]**2)) for i in range(self.channels)]
        self.audio_queue.put(indata.copy())
    
    def set_gain(self, gain):
        self.gain = gain

    def get_audio_levels(self):
        return self.audio_levels
    
    def create_audio_generator(self):
        while not self.stop_event.is_set():
            chunk = self.audio_queue.get()
            if chunk is None:
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())

    def start_transcribing(self, device_index=None, transcription_callback=None):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-US",
            max_alternatives=1,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        with sd.InputStream(callback=self.audio_callback, channels=self.channels,
                            samplerate=self.sample_rate, blocksize=self.chunk_size,
                            device=device_index):
            audio_generator = self.create_audio_generator()
            requests = (request for request in audio_generator)

            responses = self.client.streaming_recognize(streaming_config, requests)

            try:
                for response in responses:
                    if self.stop_event.is_set():
                        break

                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        if transcription_callback:
                            transcription_callback(transcript, is_partial=False)
                    else:
                        if transcription_callback:
                            transcription_callback(transcript, is_partial=True)

            except Exception as e:
                print(f"An error occurred: {e}")

    def stop_transcribing(self):
        self.stop_event.set()
        self.audio_queue.put(None)  # Signal to stop the audio generator

    def pause_transcribing(self):
        # Implement pause functionality if needed
        pass

    def resume_transcribing(self):
        # Implement resume functionality if needed
        pass

    def list_audio_devices(self):
        devices = sd.query_devices()
        return [f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})"
                for i, device in enumerate(devices)]

    def get_blackhole_16ch_index(self):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "BlackHole 16ch" in device['name'] and device['max_input_channels'] > 0:
                return i
        return None  # If BlackHole 16ch is not found