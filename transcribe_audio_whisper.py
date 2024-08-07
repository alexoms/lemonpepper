import logging
import numpy as np
import queue
import threading
import sounddevice as sd
from pywhispercpp.model import Model
import pywhispercpp.constants as constants

class WhisperStreamTranscriber:
    def __init__(self, model_path, sample_rate=16000, channels=1, n_threads=8, block_size=4096, buffer_size=3):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.n_threads = n_threads
        self.block_size = block_size
        self.buffer_size = buffer_size
        
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        self.whisper_thread = None
        self.transcription_callback = None
        
        self.audio_levels = [float('-inf')] * channels
        self.peak_levels = [float('-inf')] * channels
        self.lock = threading.Lock()
        self.last_words = []
        self.initialize_whisper()

    def initialize_whisper(self):
        try:
            logging.info(f"Initializing Whisper with model path: {self.model_path}")
            self.model = Model(self.model_path,
                               n_threads=self.n_threads,
                               print_realtime=False,
                               print_progress=False,
                               print_timestamps=False,
                               single_segment=False)
            logging.info("Whisper initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Whisper: {e}")
            raise

    def whisper_worker(self):
        audio_data = np.array([])
        overlap = int(0.5 * constants.WHISPER_SAMPLE_RATE)  # 0.5 second overlap
        try:
            while not self.stop_event.is_set():
                try:
                    new_audio = self.audio_queue.get(timeout=0.1)
                    audio_data = np.append(audio_data, new_audio)

                    if len(audio_data) >= 8 * constants.WHISPER_SAMPLE_RATE:
                        segments = self.model.transcribe(audio_data)
                        transcription = " ".join(segment.text for segment in segments)
                        if self.transcription_callback and transcription.strip():
                            self.transcription_callback(transcription, is_partial=False)
                        
                        # Keep the overlap for the next processing
                        audio_data = audio_data[-overlap:]

                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error processing audio in Whisper thread: {e}")

        except Exception as e:
            logging.error(f"Error in Whisper thread: {e}")
        finally:
            logging.info("Whisper thread terminated")

    def process_segments(self, segments):
        processed_texts = []
        for segment in segments:
            words = segment.text.split()
            if words and not all(word.upper() == '[SILENCE]' for word in words):
                # Remove duplicated words from the beginning of the segment
                while words and self.last_words and words[0].lower() == self.last_words[-1].lower():
                    words.pop(0)
                
                # Remove [SILENCE] markers
                words = [word for word in words if word.upper() != '[SILENCE]']
                
                if words:
                    processed_texts.append(" ".join(words))
                    # Store the last few words for next iteration
                    self.last_words = words[-2:]  # Store last 3 words

        return " ".join(processed_texts)
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.pause_event.is_set():
            # Calculate RMS for each channel
            rms_levels = [np.sqrt(np.mean(indata[:, i]**2)) for i in range(self.channels)]
            # Convert to dB, avoiding log(0)
            db_levels = [20 * np.log10(max(level, 1e-7)) for level in rms_levels]
            
            with self.lock:
                self.audio_levels = db_levels
                self.peak_levels = [max(current, peak) for current, peak in zip(db_levels, self.peak_levels)]
            
            audio = np.frombuffer(indata, dtype=np.float32)
            self.audio_queue.put(audio)

    def get_audio_levels(self):
        with self.lock:
            return self.audio_levels, self.peak_levels

    def reset_peak_levels(self):
        with self.lock:
            self.peak_levels = list(self.audio_levels)

    def start_transcribing(self, device_index=None, transcription_callback=None):
        self.transcription_callback = transcription_callback
        self.stop_event.clear()
        self.pause_event.clear()

        self.whisper_thread = threading.Thread(target=self.whisper_worker)
        self.whisper_thread.start()

        try:
            self.input_stream = sd.InputStream(
                callback=self.audio_callback, 
                channels=self.channels,
                samplerate=self.sample_rate, 
                blocksize=self.block_size,
                device=device_index
            )
            self.input_stream.start()
            logging.info(f"Started audio stream from device {device_index}")
        except sd.PortAudioError as e:
            logging.error(f"PortAudioError: {e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.pause_event.is_set():
            audio = np.frombuffer(indata, dtype=np.float32)
            self.audio_queue.put(audio)

    def stop_transcribing(self):
        self.stop_event.set()
        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()
        if self.whisper_thread:
            self.whisper_thread.join()
        logging.info("Transcription stopped")

    def pause_transcribing(self):
        self.pause_event.set()
        logging.info("Transcription paused")

    def resume_transcribing(self):
        self.pause_event.clear()
        logging.info("Transcription resumed")