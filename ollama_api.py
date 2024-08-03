import time
import ollama
from threading import Lock

class OllamaAPI:
    def __init__(self, host="http://localhost:11434", model="llama2"):
        self.model = model
        self.client = ollama.Client(host=host)
        self.transcription_buffer = []
        self.conversation_history = []
        self.last_transcription_time = time.time()
        self.start_time = time.time()
        self.pause_threshold = 10
        self.max_buffer_time = 60
        self.min_words_to_process = 20
        self.max_transcription_history = 10  # Keep last 10 transcriptions
        self.lock = Lock()  # Initialize the lock
        self.response_history = []

    def add_transcription(self, transcription):
        with self.lock:
            self.transcription_buffer.append(transcription)
            if len(self.transcription_buffer) > self.max_transcription_history:
                self.transcription_buffer.pop(0)
            self.last_transcription_time = time.time()

    def get_transcription(self):
        with self.lock:
            return "\n".join(self.transcription_buffer)

    def generate_response(self, prompt):
        try:
            with self.lock:
                full_prompt = "\n".join(self.conversation_history + [prompt])
            response = self.client.generate(model=self.model, prompt=full_prompt)
            
            with self.lock:
                self.conversation_history.append(f"Human: {prompt}")
                self.conversation_history.append(f"AI: {response['response']}")
                self.response_history.insert(0, f"AI: {response['response']}")
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
            return response['response']
        except Exception as e:
            return f"Error calling Ollama API: {e}"

    def get_conversation(self):
        with self.lock:
            return "\n".join(self.conversation_history)

    def should_process(self):
        current_time = time.time()
        with self.lock:
            time_since_last = current_time - self.last_transcription_time
            time_since_start = current_time - self.start_time
            word_count = sum(len(t.split()) for t in self.transcription_buffer)

        should_process = (len(self.transcription_buffer) > 0 and 
                          (time_since_last > self.pause_threshold or 
                           time_since_start > self.max_buffer_time or
                           word_count >= self.min_words_to_process))
        
        return should_process, {
            "time_since_last": time_since_last,
            "time_since_start": time_since_start,
            "word_count": word_count
        }

    def process_transcription(self):
        with self.lock:
            if not self.transcription_buffer:
                return "No transcription to process."

            full_transcription = "\n".join(self.transcription_buffer)
        
        prompt = (
            f"The following is a transcription history of an interview question or a part of it. "
            f"Each line represents a separate transcription. Analyze it and provide a response or ask for more information if needed.\n\n"
            f"Transcription History:\n{full_transcription}\n\n"
            f"If the question seems incomplete, indicate that you're waiting for more information. "
            f"If it seems complete, provide a detailed answer or ask clarifying questions if necessary."
        )
        response = self.generate_response(prompt)
        # Don't clear the buffer after processing, to maintain history
        return response
    
    def get_responses(self):
        with self.lock:
            return "\n".join(self.response_history)

    def clear_conversation(self):
            with self.lock:
                self.transcription_buffer = []
                self.conversation_history = []
                self.response_history = []