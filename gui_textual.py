import io
import logging
from collections import deque
from threading import Thread, Lock
# Create a rotating log buffer
log_buffer = deque(maxlen=100)
log_lock = Lock()

# Custom logging handler to write to the rotating buffer
class RotatingLogHandler(logging.Handler):
    def emit(self, record):
        with log_lock:
            log_buffer.appendleft(self.format(record)) 
            #log_buffer.append(self.format(record))

# Configure logging to use the custom handler
handler = RotatingLogHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
import uuid
from datetime import datetime
import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Button, ListView, ListItem, Label, Markdown, TextArea, RadioSet, RadioButton, Select, RichLog, ProgressBar
from textual.widget import Widget
from textual.reactive import reactive
from textual_slider import Slider
from transcribe_audio import AudioTranscriber
from ollama_api import OllamaAPI
import sounddevice as sd
import signal
import time
from textual.containers import Container, VerticalScroll, Horizontal
from textual.binding import Binding
import re
from textual.message import Message
from rich.markdown import Markdown as RichMarkdown
from transcribe_audio import AudioTranscriber
from transcribe_audio_google_cloud import AudioTranscriberGoogleCloud

class UpdateOllamaDisplay(Message):
    def __init__(self, conversation: str, status: str):
        self.conversation = conversation
        self.status = status
        super().__init__()

class AudioLevelMonitor(Static):
    levels = reactive(([float('-inf')] * 2, [float('-inf')] * 2))

    def on_mount(self):
        self.set_interval(1/10, self.update_levels)  # Update at 10 fps instead of 30
        self.set_interval(3, self.reset_peaks)

    def update_levels(self):
        if self.app.transcriber:
            new_levels = self.app.transcriber.get_audio_levels()
            if new_levels != self.levels:
                self.levels = new_levels
                self.refresh()

    def reset_peaks(self):
        if self.app.transcriber:
            self.app.transcriber.reset_peak_levels()

    def normalize_db(self, db_value):
        if db_value <= -60:
            return 0
        return min(max(int((db_value + 60) * 50 / 60), 0), 50)

    def render(self):
        bars = []
        for i, (level, peak) in enumerate(zip(*self.levels)):
            normalized = self.normalize_db(level)
            peak_normalized = self.normalize_db(peak)
            
            bar = f"Ch{i+1}: [{'#' * normalized}{'-' * (peak_normalized - normalized)}{' ' * (50 - peak_normalized)}] {level:.1f}dB ({peak:.1f}dB)"
            bars.append(bar)
        return "\n".join(bars)
    
class RealtimeTranscribeToAI(App):
    TRANSCRIPTION_OPTIONS = [
        ("Vosk", "vosk"),
        ("Google Cloud", "google_cloud")
    ]
    CSS_PATH = "gui_textual.tcss"
    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]
    CSS = """
    Screen {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    #transcription {
        height: 100%;
        border: solid green;
        overflow-y: auto;
        column-span: 1;
        row-span: 2;
    }
    #partial {
        height: 100%;
        border: solid magenta;
        overflow-y: auto;
    }
    #app-grid {
        column-span: 2;
        row-span: 5;
    }
    #ollama {
        height: auto;
        
        overflow-y: auto;
        scrollbar-size: 3 2;
        scrollbar-color: rgba(0,0,0,0.5) rgba(255,255,255,0.5);
        background: orange 20%;
    }
    #device {
        border: solid yellow;
        height: 100%;
    }
    ScrollableContainer {
        width: 1fr;
        overflow-y: auto;
    }
    #left-pane > Static {
        background: $boost;
        color: auto;
        margin-bottom: 1;
        padding: 1;
    }
    #left-pane {
        border: solid orange;
        height: 100%;
        overflow-y: auto;
        column-span: 2;
        row-span: 9;
    }
    #log-pane {
        border: solid blue;
        height: 100%;
        overflow-y: auto;
        column-span: 2;
        row-span: 4;
    }
    #session-list {
        column-span: 1;
        row-span: 1;
    }
    #coding_language {
        column-span: 1;
        row-span: 3;
    }
    .paused {
        background: yellow;
    }
    .not-paused {
        background: red;
    }
    .capture-paused {
        background: orange;
    }
    .capture-running {
        background: green;
        
    }
    #device_selector {
        width: 100%;
        height: auto;
        margin: 1;
    }
    AudioLevelMonitor {
        height: 3;
        border: solid green;
    }
    #gain_slider {
        width: 50;
    }
    """
    # add new window
    
    transcription = reactive("")   
    partial_transcription = reactive("")
    ollama_conversation = reactive("")
    selected_device = reactive("No device selected")
    processing_status = reactive("")
    is_paused = reactive(False)
    is_capture_paused = reactive(False)
    history = reactive({})
    current_session_id = reactive("")
    gain = reactive(1.0)
    PROMPT_OPTIONS = [
        ("Default (with coding)", "default"),
        ("Non-coding Interview", "non_coding_interview"),
    ]
    devices = sd.query_devices()
    PROMPT_DEVICE_OPTIONS = [
        (f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})", i)
        for i, device in enumerate(devices)
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        #with Container(id="app-grid"):
        with VerticalScroll(id="left-pane"):
            #yield RichLog(id="ollama")
            yield Static(id="ollama")
            yield Markdown()
        #yield Static(id="transcription")
        yield TextArea(id="transcription", language="markdown")
        yield TextArea(id="partial", language="markdown")
        #yield Static(id="partial")
        #yield Static(id="device")
        yield ListView(id="session_list")

        with Horizontal():
            yield Button(label="Pause Processing", id="pause_button", classes="not-paused", disabled=False)
            yield Button(label="Pause Capture", id="pause_capture_button", classes="capture-running", disabled=False)
            yield Button(label="Clear", id="clear_button", disabled=False)
            yield Button(label="Add Detailed Solution Prompt", id="add_solution_prompt", disabled=False)
            yield Button(label="Resubmit Transcription", id="resubmit_transcription", disabled=False)

        yield TextArea(id="user_input")
        yield Button(label="Submit", id="submit_user_input", variant="primary")
        
        with RadioSet(id="coding_language"):
            yield RadioButton("No code involved", id="no_code", value=True)
            yield RadioButton("Python", id="python")
            yield RadioButton("JavaScript", id="javascript")
            yield RadioButton("Java JDK 1.8", id="java_jdk_1_8")
            yield RadioButton("Rust", id="rust")
            yield RadioButton("C++", id="cpp")
        yield Button(label="Reprocess using specified coding language", id="reprocess_with_language")
        yield AudioLevelMonitor()
        yield Slider(value=50, min=0, max=100, step=1, id="gain_slider")
        with VerticalScroll(id="log-pane"):
            yield Static("", id="log", classes="lbl3")
        yield Select(
            options=self.PROMPT_OPTIONS,
            id="prompt_selector",
            allow_blank=False
        )
        yield Select(
            options=self.TRANSCRIPTION_OPTIONS,
            id="transcription_selector",
            allow_blank=False
        )
        yield Select(id="device_selector", prompt="Select an audio input device", options=self.PROMPT_DEVICE_OPTIONS, allow_blank=True)
        yield Footer()

    def on_mount(self):
        self.transcriber = None
        self.transcription_method = "vosk"  # Default to Vosk
        self.audio_monitor = self.query_one(AudioLevelMonitor)
        self.devices = self.list_audio_devices()
        self.transcriber = AudioTranscriber()
        self.ollama_api = OllamaAPI(host="http://192.168.1.81:11434", model="llama3.1:latest")
        self.stop_event = False
        self.update_thread = Thread(target=self.update_content, daemon=True)
        self.update_thread.start()
        self.log_thread = Thread(target=self.update_log, daemon=True)
        self.log_thread.start()
        self.start_new_session()
        self.set_timer(0.1, self.update_device_selector)
        self.query_one("#transcription", TextArea).border_title = "Complete Transcription"
        self.query_one("#left-pane", VerticalScroll).border_title = "AI Response"
        self.query_one("#partial", TextArea).border_title = "Partial Transcription"
        self.query_one("#user_input", TextArea).border_title = "Send Chat Message to AI"
        self.query_one("#log-pane", VerticalScroll).border_title = "Log"
        # Set up signal handling
        signal.signal(signal.SIGINT, self.handle_interrupt)
        #self.update_content_timer = self.set_interval(1/30, self.update_content)
    
    def on_slider_changed(self, event: Slider.Changed) -> None:
        if event.slider.id == "gain_slider":
            # Map 0-100 to a more useful gain range, e.g., 0.1 to 10
            self.gain = 0.1 * (10 ** (event.value / 50))
            if self.transcriber:
                self.transcriber.set_gain(self.gain)

    def sanitize_markdown(self, text):
        # Remove any potential HTML tags
        text = re.sub('<[^<]+?>', '', text)
        # Escape special Markdown characters
        text = re.sub(r'([\\`*_{}[\]()#+\-.!])', r'\\\1', text)
        return text

    #def watch_ollama_conversation(self, new_value: str):
    #    #self.query_one(Markdown).update(new_value)
        #self.query_one("#ollama").update(new_value)
    #    test=1
    def watch_ollama_conversation(self):
        self.update_ollama_display()

    def watch_processing_status(self):
        self.update_ollama_display()

    def update_ollama_display(self):
        try:
            markdown_content = self.ollama_conversation
            #logging.info(markdown_content)
            #self.query_one(Markdown).update(markdown_content)
            md = RichMarkdown(markdown_content)
            self.query_one("#ollama", Static).update(md)
        except Exception as e:
            logging.error(f"Error updating Markdown: {e}")
            # Fallback to using a Static widget
            self.query_one("#ollama", Static).update(markdown_content)
    
    def on_update_ollama_display(self, message: UpdateOllamaDisplay):
        if message.conversation != self.ollama_conversation or message.status != self.processing_status:
            self.ollama_conversation = message.conversation
            self.processing_status = message.status
            self.update_ollama_display()

    #def update_ollama_display(self):
        #logging.info(self.ollama_conversation)
        #self.ollama_conversation = self.ollama_api.get_responses()
        #markdown_content = self.sanitize_markdown(self.ollama_conversation)
        #self.query_one("#ollama").update(self.ollama_conversation)
        #self.query_one("#ollama", RichLog).write(self.ollama_conversation)
        #self.query_one(Markdown).update(markdown_content)
    #def update_ollama_display(self):
    #    markdown_content = self.ollama_conversation + "\n"
        #self.query_one("#ollama").update(Markdown(markdown_content))
        #self.query_one("#ollama", Markdown).text = markdown_content
        #self.query_one(Markdown).update(markdown_content)
    #    self.query_one("#ollama").update(markdown_content)
    def handle_interrupt(self, signum, frame):
        logging.info("Received interrupt signal. Exiting...")
        self.exit()

    def update_device_selector(self):
        device_options = [(device, i) for i, device in enumerate(self.devices)]
        self.PROMPT_DEVICE_OPTIONS = device_options
        selector = self.query_one("#device_selector", Select)
        logging.debug(f"Updating selector with options: {device_options}")
        selector.options = self.PROMPT_DEVICE_OPTIONS
        logging.debug(f"Selector options after update: {selector.options}")
        selector.refresh()
        selector = Select(options=device_options, id="device_selector", prompt="Select an audio input device")
        

    def list_audio_devices(self):
        devices = sd.query_devices()
        return [f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})"
                for i, device in enumerate(devices)]

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "transcription_selector":
            self.transcription_method = event.value
            if self.transcription_method == "vosk":
                self.transcriber = AudioTranscriber()
            elif self.transcription_method == "google_cloud":
                self.transcriber = AudioTranscriberGoogleCloud()
            self.log(f"Changed transcription method to: {self.transcription_method}")
        elif event.select.id == "device_selector":
            selected_device_index = event.value
            self.selected_device = self.devices[selected_device_index]
            logging.info(f"Selected device: {self.selected_device}")

            #blackhole_index = self.transcriber.get_blackhole_16ch_index()
            #device_index = blackhole_index if blackhole_index is not None else selected_device_index

            #self.query_one("#device").update(f"Selected Device: {self.selected_device}")
            self.transcribe_thread = Thread(target=self.start_transcribing, args=(selected_device_index,), daemon=True)
            self.transcribe_thread.start()
        elif event.select.id == "prompt_selector":
            # Existing code for prompt selection
            self.ollama_api.set_prompt(event.value)
            self.log(f"Changed prompt to: {event.value}")


    def transcription_callback(self, text, is_partial=False):
        def update():
            if is_partial:
                self.partial_transcription = text
                self.query_one("#partial", TextArea).text = self.partial_transcription
            else:
                self.transcription += text + "\n"
                self.partial_transcription = ""
                self.ollama_api.add_transcription(text)
            
            if self.current_session_id:
                self.history[self.current_session_id]["transcription"] = self.transcription
            
            # Update the TextArea
            self.query_one("#transcription", TextArea).text = self.transcription
            #self.update_ollama_display()
            #self.query_one(Markdown).update(self.history[self.current_session_id]["ai_responses"])
        self.call_from_thread(update)

    def start_transcribing(self, device_index):
        if self.transcriber:
            self.transcriber.start_transcribing(device_index=device_index, transcription_callback=self.transcription_callback)
        else:
            logging.error("No transcriber selected")

    def update_content(self):
        while not self.stop_event:
            if not self.is_paused and self.transcriber:
                conversation = self.ollama_api.get_responses()
                status = self.processing_status
                self.post_message(UpdateOllamaDisplay(conversation, status))
                #logging.debug(f"Updating content - Transcription: {self.transcription}, Partial: {self.partial_transcription}")
                #self.query_one("#partial").update(self.partial_transcription + 'asdasd')
                self.query_one("#partial", TextArea).text = self.partial_transcription
                #self.query_one("#transcription", TextArea).text = self.transcription
                #self.query_one("#transcription").update(self.transcription)
                
                
                # Update the Markdown widget with the content
                #markdown_content = self.ollama_conversation
                #self.query_one("#ollama").update(Markdown(markdown_content))
                #self.query_one("#ollama").update(markdown_content)
                #self.query_one(Markdown).update("""stuff""")
                
                should_process, stats = self.ollama_api.should_process()
                #logging.info(f"Should process: {should_process}")
                if should_process:
                    self.processing_status = "Processing transcription..."
                    response = self.ollama_api.process_transcription()
                    self.ollama_conversation = self.ollama_api.get_responses()
                    self.history[self.current_session_id]["ai_responses"] = self.ollama_conversation
                    #self.update_ollama_display()
                    #logging.info(f"Updating content - Response: {self.ollama_conversation}")
                    self.processing_status = ""
                    #self.query_one(Markdown).update(self.history[self.current_session_id]["ai_responses"])
            #self.audio_monitor.update_levels()
            #time.sleep(1/30)  # Update at 30 fps
            time.sleep(0.5)  # Reduced frequency of updates
        logging.info("Update content thread stopped.")

    def update_log(self):
        previous_log_content = ""
        while not self.stop_event:
            with log_lock:
                log_content = "\n".join(log_buffer)
            if log_content != previous_log_content:
                previous_log_content = log_content
                self.query_one("#log").update(log_content)
            time.sleep(1)  # Reduced frequency of log updates
        logging.info("Update log thread stopped.")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "device_selector":
            selected_device_index = event.value
            self.selected_device = self.devices[selected_device_index]
            logging.info(f"Selected device: {self.selected_device}")
            #self.query_one("#device").update(f"Selected Device: {self.selected_device}")
            self.transcribe_thread = Thread(target=self.start_transcribing, args=(selected_device_index,), daemon=True)
            self.transcribe_thread.start()
        elif event.select.id == "prompt_selector":
            # Existing code for prompt selection
            self.ollama_api.set_prompt(event.value)
            self.log(f"Changed prompt to: {event.value}")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "pause_button":
            self.is_paused = not self.is_paused
            event.button.label = "Resume Processing" if self.is_paused else "Pause Processing"
            event.button.set_class("paused" if self.is_paused else "not-paused")
            logging.info("AI processing paused" if self.is_paused else "AI processing resumed")
        elif event.button.id == "pause_capture_button":
            if self.is_capture_paused:
                self.transcriber.resume_transcribing()
                self.is_capture_paused = False
                event.button.label = "Pause Capture"
                event.button.set_class("capture-running")
                logging.info("Audio capture resumed")
            else:
                self.transcriber.pause_transcribing()
                self.is_capture_paused = True
                event.button.label = "Resume Capture"
                event.button.set_class("capture-paused")
                logging.info("Audio capture paused")
        elif event.button.id == "clear_button":
            self.clear_all_data()
            logging.info("All data cleared")
        elif event.button.id == "add_solution_prompt":
            self.add_solution_prompt()
        elif event.button.id == "submit_user_input":
            self.submit_user_input()
        elif event.button.id == "reprocess_with_language":
            self.reprocess_with_language()
        elif event.button.id == "resubmit_transcription":
            self.resubmit_transcription()

    def resubmit_transcription(self):
        transcription = self.query_one("#transcription", TextArea).text
        self.log("Resubmitting edited transcription")
        self.ollama_api.clear_transcription()
        self.ollama_api.add_transcription(transcription)
        self.force_process_transcription()

    def submit_user_input(self):
        user_input = self.query_one("#user_input").text
        if user_input.strip():
            self.ollama_api.append_to_transcription(f"\nUser input: {user_input}")
            self.force_process_transcription()
            self.query_one("#user_input").clear()  # Clear the input after submission

    def add_solution_prompt(self):
        prompt_addition = "\nPlease provide a detailed solution with the information given."
        self.ollama_api.append_to_transcription(prompt_addition)
        self.force_process_transcription()

    def reprocess_with_language(self):
        selected_language = self.query_one("#coding_language").pressed_button
        if selected_language:
            language = selected_language.id
            self.log(f"Reprocessing with language: {language}")
            
            if language == "no_code":
                prompt_addition = "\nPlease provide a solution without using any code."
            else:
                prompt_addition = f"\nIf the solution involves writing code, use the programming language {language}."
            
            self.ollama_api.append_to_transcription(prompt_addition)
            self.force_process_transcription()
        else:
            self.log("Please select a language before reprocessing.")

    def force_process_transcription(self):
        self.log("Processing transcription...")
        response = self.ollama_api.process_transcription(force=True)
        self.ollama_conversation = self.ollama_api.get_responses()
        self.history[self.current_session_id]["ai_responses"] = self.ollama_conversation
        self.update_ollama_display()
        self.log("Transcription processed.")
    
    

    def start_new_session(self):
        # Create a parent ID
        parent_id = str(uuid.uuid4())
        self.current_session_id = parent_id

        # Create a session name and timestamp
        session_name = f"Session {len(self.history) + 1}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialize the session in history
        self.history[parent_id] = {
            "session_name": session_name,
            "timestamp": timestamp,
            "transcription": self.transcription,
            "ai_responses": self.ollama_conversation
        }

        # Add the session to the ListView
        self.query_one("#session_list").append(ListItem(Label(session_name)))

    def clear_all_data(self):
        # Clear current data
        self.transcription = ""
        self.partial_transcription = ""
        self.ollama_conversation = ""
        self.processing_status = ""
        self.ollama_api.clear_conversation()

        # Update the history for the current session
        self.history[self.current_session_id]["transcription"] = self.transcription
        self.history[self.current_session_id]["ai_responses"] = self.ollama_conversation

        # Start a new session
        self.start_new_session()

        # Update the UI
        self.query_one("#transcription", TextArea).text = ""
        self.query_one("#partial", TextArea).text = ""
        self.query_one(Markdown).update("")

    def on_list_view_selected(self, event: ListView.Selected):
        selected_session_name = str(event.item.query_one(Label).renderable)
        logging.info(f"Selected session: {selected_session_name}")
        
        selected_session = None
        for session_id, session in self.history.items():
            if session["session_name"] == selected_session_name:
                selected_session = session
                break
        
        if selected_session:
            logging.info(f"Transcription: {selected_session['transcription']}")
            logging.info(f"AI Responses: {selected_session['ai_responses']}")
            self.transcription = selected_session["transcription"]
            self.ollama_conversation = selected_session["ai_responses"]
            self.query_one("#transcription", TextArea).text = self.transcription
            self.query_one(Markdown).update(self.ollama_conversation)
            
        else:
            logging.error("Session not found")

    def cleanup(self):
        logging.info("Performing cleanup operations...")
        self.stop_event = True
        if hasattr(self, 'transcriber'):
            self.transcriber.stop_transcribing()
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=2)
        if hasattr(self, 'log_thread'):
            self.log_thread.join(timeout=2)
        logging.info("Cleanup complete.")

    def exit(self, *args, **kwargs):
        self.cleanup()
        super().exit(*args, **kwargs)

    def action_quit(self):
        self.exit()

if __name__ == "__main__":
    app = RealtimeTranscribeToAI()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
        app.exit()
