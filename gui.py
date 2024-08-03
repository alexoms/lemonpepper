import curses
import time
from threading import Thread
from ollama_api import OllamaAPI
from transcribe_audio import AudioTranscriber

def get_audio_device():
    transcriber = AudioTranscriber()
    devices = transcriber.list_audio_devices()
    
    print("Available audio devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    while True:
        try:
            choice = int(input("Enter the number of the audio input device: "))
            if 0 <= choice < len(devices):
                return choice
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

class ScrollableWindow:
    def __init__(self, window):
        self.window = window
        self.scroll_pos = 0
        self.max_y, self.max_x = window.getmaxyx()

    def addstr(self, y, x, string):
        try:
            if 0 <= y < self.max_y and 0 <= x < self.max_x:
                self.window.addstr(y, x, string[:self.max_x - x - 1])
        except curses.error:
            pass

    def scroll(self, direction):
        if direction > 0 and self.scroll_pos > 0:
            self.scroll_pos = max(0, self.scroll_pos - 1)
        elif direction < 0:
            self.scroll_pos += 1

    def refresh(self):
        self.window.refresh()

def safe_addstr(window, y, x, string):
    height, width = window.getmaxyx()
    try:
        if 0 <= y < height and 0 <= x < width:
            window.addstr(y, x, string[:width - x - 1])
    except curses.error:
        pass

def main(stdscr, device_index):
    curses.curs_set(0)  # Hide cursor
    curses.use_default_colors()
    
    # Define specific color pairs
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)

    sh, sw = stdscr.getmaxyx()
    min_height = 5
    if sh < min_height * 3:
        raise ValueError("Terminal window is too small. Please resize and try again.")

    partial_height = 4
    remaining_height = sh - partial_height
    window_height = max(remaining_height // 2, min_height)
    
    transcription_win = curses.newwin(window_height, sw, 0, 0)
    partial_win = curses.newwin(partial_height, sw, window_height, 0)
    ollama_win = curses.newwin(sh - window_height - partial_height, sw, window_height + partial_height, 0)

    transcription_win.bkgd(' ', curses.color_pair(1))
    partial_win.bkgd(' ', curses.color_pair(4))
    ollama_win.bkgd(' ', curses.color_pair(2))

    scrollable_transcription = ScrollableWindow(transcription_win)
    scrollable_ollama = ScrollableWindow(ollama_win)

    ollama_api = OllamaAPI(host="http://192.168.1.81:11434", model="llama3.1:latest")
    transcriber = AudioTranscriber()

    partial_transcription = ""

    def update_windows():
        while True:
            try:
                scrollable_transcription.window.erase()
                safe_addstr(scrollable_transcription.window, 1, 1, "Full Transcription History:")
                transcription_history = ollama_api.get_transcription().split('\n')
                for i, line in enumerate(transcription_history[scrollable_transcription.scroll_pos:]):
                    scrollable_transcription.addstr(i + 2, 1, line)
                should_process, stats = ollama_api.should_process()
                safe_addstr(scrollable_transcription.window, window_height - 3, 1, f"Time since last: {stats['time_since_last']:.2f}s")
                safe_addstr(scrollable_transcription.window, window_height - 2, 1, f"Time since start: {stats['time_since_start']:.2f}s")
                safe_addstr(scrollable_transcription.window, window_height - 1, 1, f"Word count: {stats['word_count']}")
                scrollable_transcription.window.box()
                scrollable_transcription.refresh()

                partial_win.erase()
                safe_addstr(partial_win, 1, 1, "Partial Transcription:")
                safe_addstr(partial_win, 2, 1, partial_transcription)
                partial_win.box()
                partial_win.refresh()

                scrollable_ollama.window.erase()
                safe_addstr(scrollable_ollama.window, 1, 1, "Ollama Conversation:")
                conversation_history = ollama_api.get_conversation().split('\n')
                for i, line in enumerate(conversation_history[scrollable_ollama.scroll_pos:]):
                    scrollable_ollama.addstr(i + 2, 1, line)
                scrollable_ollama.window.box()
                scrollable_ollama.refresh()

                if should_process:
                    response = ollama_api.process_transcription()
                    safe_addstr(scrollable_ollama.window, window_height - 1, 1, "Processing transcription...")
                    scrollable_ollama.refresh()

            except Exception as e:
                safe_addstr(partial_win, 1, 1, f"Error in update: {str(e)}")
                partial_win.refresh()

            time.sleep(0.1)

    update_thread = Thread(target=update_windows)
    update_thread.daemon = True
    update_thread.start()

    def transcription_callback(text, is_partial=False):
        nonlocal partial_transcription
        if is_partial:
            partial_transcription = text
        else:
            ollama_api.add_transcription(text)
            partial_transcription = ""

    # Start transcribing in a separate thread
    transcribe_thread = Thread(target=transcriber.start_transcribing, 
                               args=(device_index, transcription_callback))
    transcribe_thread.daemon = True
    transcribe_thread.start()

    # Main loop for user input and scrolling
    while True:
        try:
            command = stdscr.getch()
            if command == ord('q'):
                break
            elif command == ord('u'):
                scrollable_transcription.scroll(1)
            elif command == ord('d'):
                scrollable_transcription.scroll(-1)
            elif command == ord('j'):
                scrollable_ollama.scroll(1)
            elif command == ord('k'):
                scrollable_ollama.scroll(-1)
        except Exception as e:
            safe_addstr(partial_win, 1, 1, f"Error in main loop: {str(e)}")
            partial_win.refresh()
            time.sleep(1)

if __name__ == "__main__":
    device_index = get_audio_device()
    curses.wrapper(lambda stdscr: main(stdscr, device_index))