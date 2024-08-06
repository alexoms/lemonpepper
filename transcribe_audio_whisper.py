import logging
import os
import ctypes
import sys
import numpy as np
import queue
import threading
import sounddevice as sd

class WhisperStreamTranscriber:
    def __init__(self, model_path, sample_rate=16000, channels=1, n_threads=4, step_ms=1000, length_ms=10000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.n_threads = n_threads
        self.step_ms = step_ms
        self.length_ms = length_ms
        
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        self.whisper_thread = None
        self.transcription_callback = None
        
        self.initialize_whisper()

    def initialize_whisper(self):
        try:
            logging.info(f"Initializing Whisper with model path: {self.model_path}")
            
            if os.name == 'posix':
                lib_ext = '.so' if not sys.platform.startswith('darwin') else '.dylib'
            elif os.name == 'nt':
                lib_ext = '.dll'
            else:
                raise OSError("Unsupported operating system")

            lib_path = f"./libwhisper{lib_ext}"
            if not os.path.exists(lib_path):
                raise FileNotFoundError(f"Whisper library not found: {lib_path}")
            
            self.whisper = ctypes.CDLL(lib_path)

            self.ctx = self.whisper.whisper_init_from_file(self.model_path.encode())
            if not self.ctx:
                raise RuntimeError("Failed to initialize Whisper model")

             # Check if WHISPER_SAMPLING_GREEDY is available
            try:
                sampling_strategy = self.whisper.WHISPER_SAMPLING_GREEDY
            except AttributeError:
                logging.warning("WHISPER_SAMPLING_GREEDY not found, using default sampling strategy")
                sampling_strategy = 0  # Use 0 as a default value

            self.stream_params = self.whisper.whisper_full_default_params(sampling_strategy)
            self.stream_params.print_realtime = ctypes.c_bool(True)
            self.stream_params.print_progress = ctypes.c_bool(False)
            self.stream_params.no_context = ctypes.c_bool(True)
            self.stream_params.single_segment = ctypes.c_bool(True)
            self.stream_params.max_len = ctypes.c_int(0)
            self.stream_params.language = ctypes.c_char_p(b"en")


            self.whisper.whisper_set_threads(self.ctx, ctypes.c_int(self.n_threads))

            logging.info("Whisper initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Whisper: {e}")
            raise

    def whisper_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    audio_int16 = (audio_data * 32767).astype(np.int16)

                    result = self.whisper.whisper_full(
                        self.ctx,
                        self.stream_params,
                        audio_int16.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                        len(audio_int16)
                    )

                    if result == 0:
                        transcription = ctypes.create_string_buffer(1024)
                        self.whisper.whisper_full_get_segment_text(self.ctx, 0, transcription, 1024)
                        text = transcription.value.decode('utf-8')
                        if self.transcription_callback and text.strip():
                            self.transcription_callback(text, is_partial=False)

                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error processing audio in Whisper thread: {e}")

        except Exception as e:
            logging.error(f"Error in Whisper thread: {e}")
        finally:
            if self.ctx:
                self.whisper.whisper_free(self.ctx)
            logging.info("Whisper thread terminated")

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
                blocksize=int(self.sample_rate * self.step_ms / 1000),
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
            self.audio_queue.put(indata.copy())

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