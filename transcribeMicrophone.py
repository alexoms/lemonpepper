import vosk
import pyaudio

model = vosk.Model("vosk-model-small-en-us-0.15/")
recognizer = vosk.KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

while True:
    data = stream.read(4000)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        print(result)
    else:
        partial = recognizer.PartialResult()
        print(partial)