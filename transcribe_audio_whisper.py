import ctypes
import numpy as np
import sounddevice as sd
import queue
import threading
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)

class WhisperStreamTranscriber:
    def __init__(self, model_path, sample_rate=16000, channels=1, n_threads=8, step_ms=1000, length_ms=10000):
        logging.info(f"Initializing WhisperStreamTranscriber with model path: {model_path}")
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model file not found: {model_path}")
        
        # Determine the correct library extension
        if os.name == 'posix':
            lib_ext = '.so' if not sys.platform.startswith('darwin') else '.dylib'
        elif os.name == 'nt':
            lib_ext = '.dll'
        else:
            logging.error("Unsupported operating system")
            raise OSError("Unsupported operating system")

        # Check if the Whisper library exists
        lib_path = f"./whisper_cpp_lib/libwhisper.1.6.2{lib_ext}"
        if not os.path.exists(lib_path):
            logging.error(f"Whisper library not found: {lib_path}")
            raise FileNotFoundError(f"Whisper library not found: {lib_path}")
        
        logging.info(f"Loading Whisper library from: {lib_path}")
        try:
            self.whisper = ctypes.CDLL(lib_path)
            logging.info("Whisper library loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Whisper library: {str(e)}")
            raise

        logging.info("Initializing Whisper model")
        try:
            self.ctx = self.whisper.whisper_init_from_file(model_path.encode())
            if not self.ctx:
                logging.error("Failed to initialize Whisper model")
                raise RuntimeError("Failed to initialize Whisper model")
            logging.info("Whisper model initialized successfully")
        except Exception as e:
            logging.error(f"Error during Whisper model initialization: {str(e)}")
            raise
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.transcription_callback = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Determine the correct library extension
        if os.name == 'posix':
            lib_ext = '.so' if not sys.platform.startswith('darwin') else '.dylib'
        elif os.name == 'nt':
            lib_ext = '.dll'
        else:
            raise OSError("Unsupported operating system")

        # Load the whisper.cpp shared library
        try:
            self.whisper = ctypes.CDLL(f"./whisper_cpp_lib/libwhisper.1.6.2{lib_ext}")
        except OSError as e:
            logging.error(f"Failed to load the whisper library: {e}")
            raise

        # Initialize the model
        self.ctx = self.whisper.whisper_init_from_file(model_path.encode())
        if not self.ctx:
            raise RuntimeError("Failed to initialize whisper model")

        # Set up the streaming parameters
        self.stream_params = self.whisper.whisper_full_default_params(
            self.whisper.WHISPER_SAMPLING_GREEDY)
        self.stream_params.print_realtime = ctypes.c_bool(True)
        self.stream_params.print_progress = ctypes.c_bool(False)
        self.stream_params.no_context = ctypes.c_bool(True)
        self.stream_params.single_segment = ctypes.c_bool(True)
        self.stream_params.max_len = ctypes.c_int(0)
        self.stream_params.language = ctypes.c_char_p(b"en")

        # Set thread count
        self.whisper.whisper_set_threads(self.ctx, ctypes.c_int(n_threads))

        # Set step and length
        self.step_samples = int(sample_rate * step_ms / 1000)
        self.length_samples = int(sample_rate * length_ms / 1000)

        self.audio_buffer = np.zeros(self.length_samples, dtype=np.float32)
        logging.info("WhisperStreamTranscriber initialized successfully")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.pause_event.is_set():
            self.audio_queue.put(indata.copy())

    def process_audio(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Roll the buffer and add new data
                self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                self.audio_buffer[-len(audio_data):] = audio_data.flatten()

                # Convert to 16-bit int
                audio_int16 = (self.audio_buffer * 32767).astype(np.int16)

                # Process with whisper
                result = self.whisper.whisper_full(
                    self.ctx,
                    self.stream_params,
                    audio_int16.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                    self.length_samples
                )

                if result == 0:
                    # Get the transcription
                    transcription = ctypes.create_string_buffer(1024)
                    self.whisper.whisper_full_get_segment_text(
                        self.ctx,
                        0,
                        transcription,
                        1024
                    )
                    text = transcription.value.decode('utf-8')

                    if self.transcription_callback and text.strip():
                        self.transcription_callback(text, is_partial=False)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing audio: {e}")

    def start_transcribing(self, device_index=None, transcription_callback=None):
        self.transcription_callback = transcription_callback
        self.stop_event.clear()
        self.pause_event.clear()

        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()

        try:
            self.input_stream = sd.InputStream(
                callback=self.audio_callback, 
                channels=self.channels,
                samplerate=self.sample_rate, 
                blocksize=self.step_samples,
                device=device_index
            )
            self.input_stream.start()
            logging.info("Transcription started. Press Ctrl+C to stop.")
            while not self.stop_event.is_set():
                sd.sleep(1000)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, stopping...")
        except sd.PortAudioError as e:
            logging.error(f"PortAudioError: {e}")
        finally:
            self.stop_transcribing()

    def stop_transcribing(self):
        self.stop_event.set()
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.audio_thread:
            self.audio_thread.join()
        logging.info("Transcription stopped.")

    def pause_transcribing(self):
        self.pause_event.set()
        logging.info("Transcription paused.")

    def resume_transcribing(self):
        self.pause_event.clear()
        logging.info("Transcription resumed.")

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.whisper.whisper_free(self.ctx)

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

# Usage example:
if __name__ == "__main__":
    def print_transcription(text, is_partial):
        print(f"{'Partial: ' if is_partial else 'Final: '}{text}")

    model_path = "./models/ggml-base.en.bin"  # Adjust this path to your model file
    
    transcriber = WhisperStreamTranscriber(
        model_path=model_path,
        n_threads=8,
        step_ms=1000,
        length_ms=10000
    )
    
    print("Available audio devices:")
    devices = transcriber.list_audio_devices()
    for device in devices:
        print(device)
    
    device_index = int(input("Enter the number of the audio input device to use: "))
    
    transcriber.start_transcribing(device_index=device_index, transcription_callback=print_transcription)