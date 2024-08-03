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

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Button, ListView, ListItem, Label, Markdown, TextArea, RadioSet, RadioButton, Select
from textual.reactive import reactive

from transcribe_audio import AudioTranscriber
from ollama_api import OllamaAPI
import sounddevice as sd
import signal
import time
from textual.containers import Container, VerticalScroll, Horizontal
from textual.binding import Binding

class RealtimeTranscribeToAI(App):
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
    }
    #partial {
        height: 100%;
        border: solid magenta;
        overflow-y: auto;
    }
    #app-grid {
        column-span: 2;
        row-span: 2;
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
        row-span: 2;
    }
    #log-pane {
        border: solid blue;
        height: 100%;
        overflow-y: auto;
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
    PROMPT_OPTIONS = [
        ("Default (with coding)", "default"),
        ("Non-coding Interview", "non_coding_interview"),
    ]
    devices = sd.query_devices()
    PROMPT_DEVICE_OPTIONS = [
        (f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})", i)
        for i, device in enumerate(devices)
    ]

    EXAMPLE_MARKDOWN = """\
# Markdown Document

This is an example of Textual's `Markdown` widget.

## Features

Markdown syntax and extensions are supported.

- Typography *emphasis*, **strong**, `inline code` etc.
- Headers
- Lists (bullet and ordered)
- Syntax highlighted code blocks
- Tables!
"""

    def compose(self) -> ComposeResult:
        yield Header()
        #with Container(id="app-grid"):
        with VerticalScroll(id="left-pane"):
            yield Markdown(markdown="Awaiting input...")
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
        yield Button(label="Reprocess using specified coding language", id="reprocess_with_language")
        with RadioSet(id="coding_language"):
            yield RadioButton("No code involved", id="no_code", value=True)
            yield RadioButton("Python", id="python")
            yield RadioButton("JavaScript", id="javascript")
            yield RadioButton("Java", id="java")
            yield RadioButton("C++", id="cpp")
        yield Button(label="Submit", id="submit_user_input", variant="primary")
        yield Select(
            options=self.PROMPT_OPTIONS,
            id="prompt_selector",
            allow_blank=False
        )
        with VerticalScroll(id="log-pane"):
            yield Static("", id="log", classes="lbl3")
        yield Select(id="device_selector", prompt="Select an audio input device", options=self.PROMPT_DEVICE_OPTIONS, allow_blank=True)
        yield Footer()

    def on_mount(self):
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

    def transcription_callback(self, text, is_partial=False):
        if is_partial:
            self.partial_transcription = text
        else:
            self.transcription += text + "\n"
            self.partial_transcription = ""
            self.ollama_api.add_transcription(text)
        
        # Update the current session
        if self.current_session_id:
            self.history[self.current_session_id]["transcription"] = self.transcription

        # Update the TextArea
        self.query_one("#transcription", TextArea).text = self.transcription


    def start_transcribing(self, device_index):
        self.transcriber.start_transcribing(device_index=device_index, transcription_callback=self.transcription_callback)

    def update_content(self):
        while not self.stop_event:
            if not self.is_paused:
                self.update_ollama_display()
                #logging.debug(f"Updating content - Transcription: {self.transcription}, Partial: {self.partial_transcription}")
                #self.query_one("#partial").update(self.partial_transcription + 'asdasd')
                self.query_one("#partial", TextArea).text = self.partial_transcription
                #self.query_one("#transcription", TextArea).text = self.transcription
                #self.query_one("#transcription").update(self.transcription)
                
                
                # Update the Markdown widget with the content
                markdown_content = self.ollama_conversation + "\n" + self.processing_status
                #self.query_one("#ollama").update(Markdown(markdown_content))
                #self.query_one("#ollama").update(markdown_content)
                #self.query_one(Markdown).update(markdown_content)
                
                should_process, stats = self.ollama_api.should_process()
                if should_process:
                    self.processing_status = "Processing transcription..."
                    response = self.ollama_api.process_transcription()
                    self.ollama_conversation = self.ollama_api.get_responses()
                    self.history[self.current_session_id]["ai_responses"] = self.ollama_conversation
                    self.processing_status = ""
            
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
    
    def update_ollama_display(self):
        markdown_content = self.ollama_conversation + "\n"
        #self.query_one("#ollama").update(Markdown(markdown_content))
        #self.query_one("#ollama", Markdown).text = markdown_content
        self.query_one(Markdown).update(markdown_content)

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
        self.query_one("#transcription").update("")
        self.query_one("#partial").update("")
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
            self.query_one("#transcription").update(self.transcription)
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
