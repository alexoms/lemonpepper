import ollama
import time
import curses
from threading import Thread, Lock

class OllamaAPI:
    def __init__(self, model="llama2", host="http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
        self.transcription_buffer = []
        self.conversation_history = []
        self.last_transcription_time = time.monotonic()
        self.pause_threshold = 10  # 10 seconds pause threshold
        self.max_buffer_time = 60  # 60 seconds maximum buffer time
        self.min_words_to_process = 20  # Minimum number of words before processing

    def generate_response(self, prompt):
        print("Attempting to generate response from Ollama...")
        try:
            # Include conversation history in the prompt
            full_prompt = "\n".join(self.conversation_history + [prompt])
            response = self.client.generate(model=self.model, prompt=full_prompt)
            print("Successfully received response from Ollama.")
            
            # Add the new exchange to the conversation history
            self.conversation_history.append(f"Human: {prompt}")
            self.conversation_history.append(f"AI: {response['response']}")
            
            # Optionally, limit the history to prevent it from growing too large
            if len(self.conversation_history) > 10:  # Keep last 5 exchanges
                self.conversation_history = self.conversation_history[-10:]
            
            return response['response']
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return None

    def process_transcription(self):
        if not self.transcription_buffer:
            return

        full_transcription = " ".join(self.transcription_buffer)
        #prompt = f"Respond to this transcription as if you're assisting in a technical interview: {transcription}\nIf it's a programming question, provide a Python code example in your response."
        prompt = (
            f"You are assisting in a technical interview.  The following is a partial or complete transcription of an interview question. "
            f"It may be incomplete. Please analyze it and provide a response. "
            f"If it seems incomplete, indicate that you're waiting for more information.\n\n"
            f"Transcription: {full_transcription}"
        )
        response = self.generate_response(prompt)
        
        if response:
            print(f"Accumulated Transcription: {full_transcription}")
            print(f"Ollama Response:\n{response}")
        else:
            print("Failed to get a response from Ollama.")

        self.transcription_buffer = []  # Clear the buffer after processing

    def add_transcription(self, transcription):
        if not self.transcription_buffer:
            self.start_time = time.monotonic()
        self.transcription_buffer.append(transcription)
        self.last_transcription_time = time.monotonic()

    def should_process(self):
        current_time = time.monotonic()
        time_since_last = current_time - self.last_transcription_time
        time_since_start = current_time - self.start_time
        word_count = sum(len(t.split()) for t in self.transcription_buffer)

        print(f"Time since last transcription: {time_since_last:.2f} seconds")
        print(f"Time since start of current question: {time_since_start:.2f} seconds")
        print(f"Current word count: {word_count}")
        
        should_process = (len(self.transcription_buffer) > 0 and 
                          (time_since_last > self.pause_threshold or 
                           time_since_start > self.max_buffer_time or
                           word_count >= self.min_words_to_process))
        
        print(f"Should process: {should_process}")
        return should_process

