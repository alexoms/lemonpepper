import ctypes
import os
import sys
import logging
import numpy as np
import faulthandler

faulthandler.enable()
logging.basicConfig(level=logging.DEBUG)

# Set environment variable (though it may not work for GPU on M1/M2 Macs)
os.environ['WHISPER_NO_GPU'] = '1'

class whisper_full_params(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("max_tokens", ctypes.c_int),
        ("speed_up", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_non_speech_tokens", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
    ]

def test_whisper():
    try:
        if os.name == 'posix':
            lib_ext = '.so' if not sys.platform.startswith('darwin') else '.dylib'
        elif os.name == 'nt':
            lib_ext = '.dll'
        else:
            raise OSError("Unsupported operating system")

        lib_path = f"./libwhisper{lib_ext}"
        logging.info(f"Loading Whisper library from: {lib_path}")
        
        whisper = ctypes.CDLL(lib_path)
        logging.info("Whisper library loaded successfully")

        whisper_init_from_file = whisper.whisper_init_from_file
        whisper_init_from_file.argtypes = [ctypes.c_char_p]
        whisper_init_from_file.restype = ctypes.c_void_p

        model_path = "./whisper_models/ggml-small.en.bin"
        logging.info(f"Initializing Whisper model from: {model_path}")
        
        ctx = whisper_init_from_file(model_path.encode())
        if not ctx:
            raise RuntimeError("Failed to initialize Whisper model")
        logging.info("Whisper model initialized successfully")

        logging.info("Creating audio buffer")
        audio_data = np.zeros(16000, dtype=np.float32)
        audio_data_p = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        logging.info("Audio buffer created")

        whisper_full = whisper.whisper_full
        whisper_full.argtypes = [ctypes.c_void_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        whisper_full.restype = ctypes.c_int

        logging.info("Creating default parameters")
        params = whisper_full_params(
            strategy=0,
            n_threads=1,
            n_max_text_ctx=16384,
            offset_ms=0,
            duration_ms=0,
            translate=False,
            no_context=True,
            single_segment=True,
            print_special=False,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            token_timestamps=False,
            thold_pt=0.01,
            thold_ptsum=0.01,
            max_len=0,
            max_tokens=0,
            speed_up=False,
            audio_ctx=0,
            prompt_tokens=None,
            prompt_n_tokens=0,
            language=b"en",
            suppress_blank=True,
            suppress_non_speech_tokens=True,
            temperature=0.0,
            max_initial_ts=1.0,
            length_penalty=1.0
        )
        logging.info("Default parameters created")

        logging.info("About to call whisper_full")
        result = whisper_full(ctx, params, audio_data_p, ctypes.c_int(16000))
        logging.info(f"whisper_full called successfully with result: {result}")

        if result != 0:
            errno = ctypes.get_errno()
            logging.error(f"whisper_full failed with error code: {errno}")
            raise RuntimeError(f"whisper_full failed with error code: {errno}")

        whisper_full_n_segments = whisper.whisper_full_n_segments
        whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        whisper_full_n_segments.restype = ctypes.c_int

        n_segments = whisper_full_n_segments(ctx)
        logging.info(f"Number of segments: {n_segments}")

        if n_segments > 0:
            whisper_full_get_segment_text = whisper.whisper_full_get_segment_text
            whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            whisper_full_get_segment_text.restype = ctypes.c_int

            for i in range(n_segments):
                try:
                    transcription = ctypes.create_string_buffer(1024)
                    result = whisper_full_get_segment_text(ctx, ctypes.c_int(i), transcription, ctypes.c_int(1024))
                    if result != 0:
                        logging.error(f"whisper_full_get_segment_text failed for segment {i} with result: {result}")
                    else:
                        text = transcription.value.decode('utf-8')
                        logging.info(f"Transcription for segment {i}: {text}")
                except Exception as e:
                    logging.exception(f"Error while getting segment text for segment {i}")
        else:
            logging.info("No segments to transcribe")

        logging.info("Test completed successfully")

        # Commenting out the whisper_free call to avoid segmentation fault
        # logging.info("Freeing Whisper context")
        # whisper.whisper_free(ctx)
        # logging.info("Whisper context freed successfully")

    except Exception as e:
        logging.exception(f"Error in test_whisper: {e}")

if __name__ == "__main__":
    test_whisper()