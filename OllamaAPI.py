import curses
import time
import ollama
from threading import Thread, Lock

class OllamaAPI:
    def __init__(self, model="llama2", host="http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
        self.transcription_buffer = []
        self.conversation_history = []
        self.lock = Lock()
        self.last_transcription_time = time.time()
        self.start_time = time.time()
        self.pause_threshold = 10  # 10 seconds pause threshold
        self.max_buffer_time = 60  # 60 seconds maximum buffer time
        self.min_words_to_process = 20  # Minimum number of words before processing

    def generate_response(self, prompt):
        try:
            full_prompt = "\n".join(self.conversation_history + [prompt])
            response = self.client.generate(model=self.model, prompt=full_prompt)
            
            with self.lock:
                self.conversation_history.append(f"Human: {prompt}")
                self.conversation_history.append(f"AI: {response['response']}")
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
            return response['response']
        except Exception as e:
            return f"Error calling Ollama API: {e}"

    def add_transcription(self, transcription):
        with self.lock:
            if not self.transcription_buffer:
                self.start_time = time.time()
            self.transcription_buffer.append(transcription)
            self.last_transcription_time = time.time()

    def get_transcription(self):
        with self.lock:
            return "\n".join(self.transcription_buffer[-5:])  # Last 5 transcriptions

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

            full_transcription = " ".join(self.transcription_buffer)
            prompt = (
                f"The following is a transcription of an interview question or a part of it. "
                f"It may be incomplete or ongoing. Analyze it and provide a response or ask for more information if needed.\n\n"
                f"Transcription: {full_transcription}\n\n"
                f"If the question seems incomplete, indicate that you're waiting for more information. "
                f"If it seems complete, provide a detailed answer or ask clarifying questions if necessary."
            )
            response = self.generate_response(prompt)
            self.transcription_buffer = []  # Clear the buffer after processing
            return response

def main(stdscr):
    curses.curs_set(0)
    sh, sw = stdscr.getmaxyx()
    transcription_win = curses.newwin(sh // 2, sw, 0, 0)
    ollama_win = curses.newwin(sh // 2, sw, sh // 2, 0)

    transcription_win.bkgd(' ', curses.color_pair(1))
    ollama_win.bkgd(' ', curses.color_pair(2))

    ollama_api = OllamaAPI()

    def update_windows():
        while True:
            transcription_win.clear()
            transcription_win.addstr(1, 1, "Transcription:")
            transcription_win.addstr(2, 1, ollama_api.get_transcription())
            should_process, stats = ollama_api.should_process()
            transcription_win.addstr(sh // 2 - 3, 1, f"Time since last: {stats['time_since_last']:.2f}s")
            transcription_win.addstr(sh // 2 - 2, 1, f"Time since start: {stats['time_since_start']:.2f}s")
            transcription_win.addstr(sh // 2 - 1, 1, f"Word count: {stats['word_count']}")
            transcription_win.box()
            transcription_win.refresh()

            ollama_win.clear()
            ollama_win.addstr(1, 1, "Ollama Conversation:")
            ollama_win.addstr(2, 1, ollama_api.get_conversation())
            ollama_win.box()
            ollama_win.refresh()

            if should_process:
                response = ollama_api.process_transcription()
                ollama_win.addstr(sh // 2 - 1, 1, "Processing transcription...")
                ollama_win.refresh()

            time.sleep(0.1)

    update_thread = Thread(target=update_windows)
    update_thread.daemon = True
    update_thread.start()

    # Simulate transcription and Ollama interaction
    transcriptions = [
        "Design a class with three operations:",
        "1. Insert a value (no duplicates)",
        "2. Remove a value",
        "3. Get a random inserted value"
    ]

    for t in transcriptions:
        ollama_api.add_transcription(t)
        time.sleep(2)

    time.sleep(10)  # Keep the window open to see the result

if __name__ == "__main__":
    curses.wrapper(main)
    curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)