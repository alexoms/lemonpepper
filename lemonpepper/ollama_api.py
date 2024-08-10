import time
import ollama
from threading import Lock

class OllamaAPI:
    def __init__(self, host="http://localhost:11434", model="llama2", app=None):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.transcription_buffer = []
        self.conversation_history = []
        self.last_transcription_time = time.time()
        self.start_time = time.time()
        self.pause_threshold = 5  # Reduced from 10 to 5 seconds
        self.max_buffer_time = 60
        self.min_words_to_process = 20
        self.max_transcription_history = 10
        self.lock = Lock()
        self.response_history = []
        self.last_processed_time = time.time()
        self.processing_threshold = 3.0
        self.last_processed_transcription = ""
        self.prompts = {
            "default": self.default_prompt,
            "non_coding_interview": self.non_coding_interview_prompt
        }
        self.current_prompt = "default"
        self.app = app

    def update_settings(self, host, model):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        
    # fix mistranscriptions using the context of the transcript and based on potential phonetic issues. I asked it to replace transcribed words that don't make sense with "[unintelligable]"        

    def default_prompt(self, transcription):
        return (
            f"The following is a transcription of an interview question or a part of it, "
            f"potentially including user input and coding language preferences. "
            f"Analyze it and provide a response.\n\n"
            f"Transcription and User Input: {transcription}\n\n"
            f"If the information seems incomplete, indicate that you're waiting for more. "
            f"If it seems complete, provide a detailed answer or ask clarifying questions if necessary. "
            f"If a specific programming language is mentioned, use that language for any code examples."
        )

    def non_coding_interview_prompt(self, transcription):
        return (
            f"The following is a transcription of a non-coding interview question or a part of it, "
            f"potentially including user input. Analyze it and provide a response.\n\n"
            f"Transcription and User Input: {transcription}\n\n"
            f"If the information seems incomplete, indicate that you're waiting for more. "
            f"If it seems complete, provide a detailed answer or ask clarifying questions if necessary. "
            f"Focus on conceptual explanations and avoid using code examples in your response."
        )
    
    def clear_transcription(self):
        with self.lock:
            self.transcription_buffer.clear()
            self.last_transcription_time = time.time()

    def set_prompt(self, prompt_name):
        if prompt_name in self.prompts:
            self.current_prompt = prompt_name
        else:
            raise ValueError(f"Unknown prompt: {prompt_name}")

    def add_transcription(self, transcription):
        with self.lock:
            if transcription and len(transcription.strip()) > 0:
                self.transcription_buffer.append(transcription)
                if len(self.transcription_buffer) > self.max_transcription_history:
                    self.transcription_buffer.pop(0)
                self.last_transcription_time = time.time()
    
    def append_to_transcription(self, text):
        with self.lock:
            if self.transcription_buffer:
                self.transcription_buffer[-1] += text
            else:
                self.transcription_buffer.append(text)
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
                #self.response_history.insert(0, f"AI: {response['response']}")
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
            time_since_last_process = current_time - self.last_processed_time
            word_count = sum(len(t.split()) for t in self.transcription_buffer)
            new_content = " ".join(self.transcription_buffer).strip() != self.last_processed_transcription

        should_process = (len(self.transcription_buffer) > 0 and 
                          word_count >= self.min_words_to_process and
                          (time_since_last > self.pause_threshold or 
                           time_since_start > self.max_buffer_time) and
                          time_since_last_process >= self.processing_threshold and new_content)
        
        return should_process, {
            "time_since_last": time_since_last,
            "time_since_start": time_since_start,
            "word_count": word_count,
            "time_since_last_process": time_since_last_process,
            "new_content": new_content
        }

    def process_transcription(self, force=False):
        with self.lock:
            if not self.transcription_buffer:
                return "No transcription to process."

            full_transcription = " ".join(self.transcription_buffer)
            #if not force and full_transcription.strip() == self.last_processed_transcription:
            #if full_transcription.strip() == self.last_processed_transcription:
            #    return "No new content to process."

            self.last_processed_transcription = full_transcription.strip()
        self.app.update_ai_status("Prompting LLM...")
        prompt = self.prompts[self.current_prompt](full_transcription)
        self.app.update_ai_status("Waiting for LLM response...")
        response = self.generate_response(prompt)
        self.app.update_ai_status("LLM response received")
        self.last_processed_time = time.time()
        self.response_history.insert(0, f"LLM: {response}")
        return response
    
    def get_responses(self):
        with self.lock:
            return "\n".join(self.response_history)

    def clear_conversation(self):
            with self.lock:
                self.transcription_buffer = []
                self.conversation_history = []
                self.response_history = []