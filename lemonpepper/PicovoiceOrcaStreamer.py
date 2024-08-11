import threading
import queue
import sounddevice as sd
import numpy as np
import pvorca
import os
import time
from typing import Sequence, Optional
from pvorca import OrcaError

class StreamingAudioDevice:
    def __init__(self, device_index: Optional[int] = None) -> None:
        if device_index is None:
            device_info = sd.query_devices(kind="output")
            device_index = int(device_info["index"])

        self._device_index = device_index
        self._queue = queue.Queue()
        self._buffer = None
        self._stream = None
        self._sample_rate = None
        self._blocksize = None
        self._stop_event = threading.Event()

    def start(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._blocksize = self._sample_rate // 20
        self._stream = sd.OutputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype=np.int16,
            device=self._device_index,
            callback=self._callback,
            blocksize=self._blocksize)
        self._stream.start()

    def _callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        if self._stop_event.is_set() or self._queue.empty():
            outdata[:] = 0
            return

        pcm = self._queue.get()
        outdata[:, 0] = pcm

    def play(self, pcm_chunk: Sequence[int]) -> None:
        if self._stream is None:
            raise ValueError("Stream is not started. Call `start` method first.")

        pcm_chunk = np.array(pcm_chunk, dtype=np.int16)

        if self._buffer is not None:
            if pcm_chunk is not None:
                pcm_chunk = np.concatenate([self._buffer, pcm_chunk])
            else:
                pcm_chunk = self._buffer
            self._buffer = None

        length = pcm_chunk.shape[0]
        for index_block in range(0, length, self._blocksize):
            if self._stop_event.is_set():
                break
            if (length - index_block) < self._blocksize:
                self._buffer = pcm_chunk[index_block: index_block + (length - index_block)]
            else:
                self._queue.put_nowait(pcm_chunk[index_block: index_block + self._blocksize])

    def stop(self):
        self._stop_event.set()
        self._queue.queue.clear()
        if self._buffer is not None:
            self._buffer = None

    def reset(self):
        self._stop_event.clear()
        self._queue.queue.clear()
        if self._buffer is not None:
            self._buffer = None

    def flush_and_terminate(self) -> None:
        self.stop()
        self.flush()
        self.terminate()

    def flush(self) -> None:
        if self._buffer is not None:
            chunk = np.zeros(self._blocksize, dtype=np.int16)
            chunk[:self._buffer.shape[0]] = self._buffer
            self._queue.put_nowait(chunk)

        time_interval = self._blocksize / self._sample_rate
        while not self._queue.empty():
            time.sleep(time_interval)

        time.sleep(time_interval)

    def terminate(self) -> None:
        self._stream.stop()
        self._stream.close()

    @staticmethod
    def list_output_devices():
        return sd.query_devices(kind="output")

class PicovoiceOrcaStreamer:
    def __init__(self, access_key):
        self.access_key = access_key
        self.audio_device = None
        self.orca = None
        self.orca_stream = None
        self.sample_rate = None
        self.device_index = None
        self.playback_thread = None
        self.stop_event = threading.Event()

    def initialize_orca(self):
        try:
            self.audio_device = StreamingAudioDevice(self.device_index)
            self.orca = pvorca.create(access_key=self.access_key)
            self.orca_stream = self.orca.stream_open()
            self.sample_rate = self.orca.sample_rate
            if self.audio_device:
                self.audio_device.start(self.sample_rate)
        except OrcaError as e:
            raise Exception(f"Failed to initialize Orca: {e}")

    def synthesize_and_play(self, text):
        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_playback()
        
        self.stop_event.clear()
        if self.audio_device:
            self.audio_device.reset()
        self.playback_thread = threading.Thread(target=self._synthesize_and_play_thread, args=(text,))
        self.playback_thread.start()

    def _synthesize_and_play_thread(self, text):
        if not self.orca:
            self.initialize_orca()

        try:
            chunks = text.split('.')
            for chunk in chunks:
                if self.stop_event.is_set():
                    break
                if chunk.strip():
                    pcm = self.orca_stream.synthesize(chunk.strip() + '.')
                    if pcm is not None and not self.stop_event.is_set():
                        self.audio_device.play(pcm)

            # Flush any remaining audio
            if not self.stop_event.is_set():
                pcm = self.orca_stream.flush()
                if pcm is not None:
                    self.audio_device.play(pcm)

        except OrcaError as e:
            print(f"Error during speech synthesis: {e}")

    def stop_playback(self):
        self.stop_event.set()
        if self.audio_device:
            self.audio_device.stop()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

    def stop_and_clear(self):
        self.stop_playback()
        if self.orca_stream:
            self.orca_stream.close()
        if self.orca:
            self.orca.delete()
        if self.audio_device:
            self.audio_device.flush_and_terminate()

    @staticmethod
    def list_audio_devices():
        return StreamingAudioDevice.list_output_devices()


# Example usage
if __name__ == "__main__":
    # List available devices
    print(PicovoiceOrcaStreamer.list_audio_devices())
    
    # Create streamer with default device
    streamer = PicovoiceOrcaStreamer(access_key="YOUR_ACCESS_KEY_HERE")
    
    # Synthesize and play audio
    streamer.synthesize_and_play("Hello, this is a test of the Picovoice Orca streaming library. This version includes threaded audio playback for improved performance.")
    
    # Allow some time for audio to finish playing
    time.sleep(10)
    
    # Stop and clear resources
    streamer.stop_and_clear()