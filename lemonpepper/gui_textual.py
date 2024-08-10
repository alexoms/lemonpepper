import io
import os
import sys
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


import traceback

def excepthook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = excepthook

import requests
import uuid
from datetime import datetime
import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Button, ListView, ListItem, Label, Markdown, TextArea, RadioSet, RadioButton, Select, RichLog, ProgressBar, TabbedContent, TabPane
from textual.widget import Widget
from textual.reactive import reactive
from textual_slider import Slider
from textual.worker import Worker, WorkerState
import sounddevice as sd
import signal
import time
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.binding import Binding
import re
from textual.message import Message
from rich.markdown import Markdown as RichMarkdown
from .ollama_api import OllamaAPI
from .transcribe_audio import AudioTranscriber
from .transcribe_audio_google_cloud import AudioTranscriberGoogleCloud
from .transcribe_audio_whisper import WhisperStreamTranscriber
from rich.console import Console
from rich.text import Text
from pyperclip import copy as copy_to_clipboard
from io import StringIO
import json
from rich.text import Text
from rich.style import Style
from textual.timer import Timer
import threading
from .utils import get_model_directory
from .model_manager import ModelDownloadButton
from textual import work
import httpx
import appdirs

class LogRedirector(io.StringIO):
    def write(self, string):
        string = string.strip()
        if string:
            logging.info(string)

    def flush(self):
        pass

class UpdateOllamaDisplay(Message):
    def __init__(self, conversation: str, status: str):
        self.conversation = conversation
        self.status = status
        super().__init__()

# Create a custom footer with AudioLevelMonitor
class CustomFooter(Footer):

    def __init__(self):
        super().__init__()
        self.status_message = Static("Idle", id="ai-status")
        self.left_bar = Static("", id="left_bar")
        self.spinner = Static("", id="spinner")
        self.shortcuts = Static("", id="shortcuts")
        self.spinner_thread = None
        self.is_spinning = False
        self.highlights = []  # Initialize highlights as an empty list
        self.status_timer: threading.Timer | None = None
        self.timer_lock = threading.Lock()

        # Load spinners from JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        spinners_path = os.path.join(script_dir, "spinners.json")
        with open(spinners_path, "r") as f:
            self.spinners = json.load(f)
        self.current_spinner = self.spinners["dots"]

    def compose(self):
        yield self.left_bar
        yield self.spinner
        yield self.status_message
        yield Static("", id="footer-spacer-left")
        yield self.shortcuts
        yield Static("", id="footer-spacer-right")
        yield AudioLevelMonitor(id="audio-monitor")
    
    def on_mount(self):
        self.update_highlights()
        
    
    
    def update_shortcuts(self):
        shortcuts = self.app.get_shortcuts()
        shortcut_text = " ".join(f"[{key}]{action}" for key, action in shortcuts.items())
        logging.info(f"Shortcut array: {shortcuts.items()}")
        self.shortcuts.update(shortcut_text)

    def update_highlights(self) -> None:
        shortcuts = self.app.get_shortcuts()
        highlight_keys = set(self.highlights) & set(shortcuts.keys())
        non_highlight_keys = set(shortcuts.keys()) - set(self.highlights)
        
        text = Text()
        for key in highlight_keys:
            text.append(f" {key} ", "reverse")
            text.append(f" {shortcuts[key]}      ", "bold")
        
        for key in non_highlight_keys:
            text.append(f" {key} ", "dim")
            text.append(f" {shortcuts[key]}", "")
        
        self.shortcuts.update(text)

    def set_highlights(self, keys: list[str]) -> None:
        """Set the keys to be highlighted."""
        self.highlights = keys
        self.update_highlights()
        
    def update_status(self, message: str):
        text2 = Text()
        # Left angle bracket
        text2.append("\uE0B6", Style(color="#162a33", bgcolor="#292828"))
        text2.append(" ", Style(color="#223a45", bgcolor="#223a45"))
        self.left_bar.update(text2)
        #colored_text = Text()
        #colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
        #for i, char in enumerate(message):
        #    colored_text.append(char, Style(color=colors[i % len(colors)]))
        #self.status_message.update(colored_text)
        text = Text()
        # Left angle bracket
        #text.append("\uE0B6", Style(color="#162a33", bgcolor="#292828"))
        # Spinner and Status
        #text.append(f" {self.spinner} ", Style(color="#f199f2", bgcolor="#223a45", bold=True))
        #text.append("", Style(color="#192f52", bgcolor="#223a45"))
        text.append(f" {message}   ", Style(color="#65d67b", bgcolor="#223a45", bold=True))
        # Right angle bracket
        text.append("\uE0B0", Style(color="#162a33", bgcolor="#292828"))
        self.status_message.update(text)

        #self.status_message.update(message)
        
        if message == "Waiting for LLM response...":
            self.start_spinner()
        elif message == "LLM response received":
            self.stop_spinner()
            self.set_reset_timer()
            self.app.log(f"Set status to '{message}' and started timer")  # Debug print
        else:
            self.stop_spinner()
        self.app.log(f"Status updated to: {message}")  # Debug print
    
    def set_reset_timer(self):
        with self.timer_lock:
            if self.status_timer:
                self.status_timer.cancel()
            self.status_timer = threading.Timer(2.0, self.reset_status)
            self.status_timer.start()

    def reset_status(self):
        self.app.log("Actually resetting status to Idle")  # Debug print
        self.update_status("Idle")

    def start_spinner(self):
        if not self.is_spinning:
            self.is_spinning = True
            self.spinner_thread = Thread(target=self._spin, daemon=True)
            self.spinner_thread.start()

    def stop_spinner(self):
        self.is_spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        self.spinner.update("")

    def _spin(self):
        index = 0
        while self.is_spinning:
            frame = self.current_spinner["frames"][index]
            self.spinner.update(frame)  # Remove the Text wrapper
            time.sleep(self.current_spinner["interval"] / 1000)
            index = (index + 1) % len(self.current_spinner["frames"])

class AudioLevelMonitor(Static):
    levels = reactive(([float('-inf')] * 2, [float('-inf')] * 2))

    def on_mount(self):
        self.set_interval(0.05, self.update_levels)  # Update at 20 fps
        self.set_interval(3, self.reset_peaks)

    def update_levels(self):
        if hasattr(self.app, 'transcriber') and self.app.transcriber:
            new_levels = self.app.transcriber.get_audio_levels()
            if new_levels != self.levels:
                self.levels = new_levels
                self.refresh()

    def reset_peaks(self):
        if self.app.transcriber:
            self.app.transcriber.reset_peak_levels()

    def normalize_db(self, db_value):
        min_db = -60
        max_db = 0
        if db_value <= min_db:
            return 0
        return min(max(int((db_value - min_db) * 50 / (max_db - min_db)), 0), 50)

    def render(self):
        result = Text()
        for i, (level, peak) in enumerate(zip(*self.levels)):
            normalized = self.normalize_db(level)
            peak_normalized = self.normalize_db(peak)
            
            bar = Text(f"C{i+1}:")
            bar.append("[")
            bar.append("#" * normalized, "green")
            if peak_normalized > normalized:
                bar.append("-" * (peak_normalized - normalized), "yellow")
            bar.append(" " * (10 - max(normalized, peak_normalized)))
            bar.append("]")
            
            result.append(bar)
            if i < len(self.levels[0]) - 1:
                result.append(" ")

        result.append(f" L:{self.levels[0][0]:.0f}dB P:{self.levels[1][0]:.0f}dB")
        return result

            
class LemonPepper(App):
    ollama_host = reactive("http://localhost:11434")
    class OllamaHostChanged(Message):
        def __init__(self, host: str):
            self.host = host
            super().__init__()
    

    #WHISPER_MODEL_PATH = "./whisper_models/ggml-base.en.bin"
    WHISPER_MODEL_PATH = "base.en"
    TRANSCRIPTION_OPTIONS = [
        ("OpenAI Whisper", "whisper", WHISPER_MODEL_PATH),
        ("Alpha Cephei Vosk", "vosk", None),
        ("Google Cloud", "google_cloud", None)
    ]
    CSS_PATH = "gui_textual.tcss"
    BINDINGS = [Binding("ctrl+q", "quit", "Quit"),
                Binding("ctrl+c", "copy_ai_response", "Copy AI Response")
                ]
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1;
    }
    #home {
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
        background: $primary-background;
        color: $primary
         
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

    #log-tab-pane {
        
    }
    #log-vertical-scroll > Static {
       
    }
    #log-vertical-scroll {
        border: solid orange;
        
        width: 100%;

    }
    #log {
        
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
    #user_input {
        border: solid cyan;
    }
    #audio-device-settings {
        height: auto;
        border: solid pink;
        overflow-y: auto;
        column-span: 1;
        row-span: 1;
    }
    
    #prompt-settings {
        height: auto;
        border: solid cyan;
        overflow-y: auto;
        column-span: 1;
        row-span: 1;
    }
    #device_selector {
        width: 100%;
        height: auto;
        margin: 1;
    }
    
    #gain_slider {
        width: auto;
    }
    Footer {
        layout: horizontal;
        height: auto;
        padding: 1;
    }
    #ai-status {
        width: auto;
        min-width: 30;
        
        content-align: left middle;
        
    }
    #footer-spacer-left, #footer-spacer-right {
        width: 1fr;
    }
    #left_bar {
       width: auto;
       content-align: left middle;
    }
    #spinner {
        width: auto;
        
        content-align: center middle;
        background: #223a45;
        color: #FF69B4;  /* Hot pink color */
    }
    #audio-monitor {
        width: auto;
        min-width: 90;  # Adjust this value as needed
        height: 100%;
        border: none;
        
        color: $text;
        content-align: right middle;
    }
    #shortcuts {
        width: auto;
        color: $text;
        content-align: left middle;
        padding-left: 1;
        padding-right: 1;
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

    def __init__(self):
        logging.info("Initializing LemonPepper")
        super().__init__()
        self.model_dir = get_model_directory()
        logging.info(f"model directory: {self.model_dir}")
        self.TRANSCRIPTION_OPTIONS = [
            ("OpenAI Whisper base.en", "whisper_base_en", os.path.join(self.model_dir, "ggml-base.en.bin")),
            ("OpenAI Whisper small.en", "whisper_small_en", os.path.join(self.model_dir, "ggml-small.en.bin")),
            ("Alpha Cephei Vosk", "vosk", None),
            ("Google Cloud", "google_cloud", None)
        ]
        self.devices = self.list_audio_devices()
        self.selected_device = Select.BLANK
        self.transcriber = None
        self.transcription_method = self.TRANSCRIPTION_OPTIONS[0][1]  # Default to the first option
        self.settings = self.load_settings()
        self.ollama_host = self.settings.get("ollama_host", "http://localhost:11434")
        self.saved_ollama_model = None
        self.saved_device = None

    def compose(self) -> ComposeResult:
        yield Header()
        footer = CustomFooter()
        yield footer
        footer.set_highlights(["ctrl+q", "ctrl+c"])  # Set the keys you want to highlight
        with TabbedContent(initial="home"):
            with TabPane("Home", id="home"):
                #with Container(id="app-grid"):
                with VerticalScroll(id="left-pane"):
                    #yield RichLog(id="ollama")
                    yield Static(id="ollama")
                    yield Markdown()
                    yield Button("Copy to Clipboard", id="copy_ai_response", tooltip="Copy AI response to clipboard")
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
                
                with Horizontal():
                    yield TextArea(id="user_input")
                    yield Button(label="Submit", id="submit_user_input", variant="primary")
                
                with Horizontal():
                    with RadioSet(id="coding_language"):
                        yield RadioButton("No code involved", id="no_code", value=True)
                        yield RadioButton("Python", id="python")
                        yield RadioButton("JavaScript", id="javascript")
                        yield RadioButton("Java JDK 1.8", id="java_jdk_1_8")
                        yield RadioButton("Rust", id="rust")
                        yield RadioButton("C++", id="cpp")
                    yield Button(label="Reprocess using specified coding language", id="reprocess_with_language")
                
                with Vertical():
                    #yield AudioLevelMonitor()
                    yield Slider(value=50, min=0, max=100, step=1, id="gain_slider")
                    
  
            with TabPane("Settings", id="settings"):
                with VerticalScroll(id="audio-device-settings"):
                    yield Static("Ollama API Settings:")
                    yield Input(placeholder="Ollama API Host", id="ollama_host", value=self.settings.get("ollama_host", "http://localhost:11434"))
                    yield Button("Refresh Models", id="refresh_models", variant="primary")
                    # Update the Ollama model Select widget initialization
                    ollama_model = self.settings.get("ollama_model", "")
                    yield Select(
                        options=[],  # We'll populate this later in on_mount
                        id="ollama_model",
                        allow_blank=True,
                        prompt="Select Ollama Model",
                    )
                    yield Button("Update Ollama Settings", id="update_ollama_settings")
                    #yield Input(placeholder="Ollama API Host", id="ollama_host", value=self.settings.get("ollama_host", "http://localhost:11434"))
                    #yield Select(options=[], id="ollama_model", value=Select.BLANK, prompt="Select Ollama Model")
                    #yield Select(id="device_selector", prompt="Select an audio input device", options=self.PROMPT_DEVICE_OPTIONS, value=Select.BLANK)
        
                    yield Static("Audio input device to transcribe: ")
                    yield Select(id="device_selector", prompt="Select an audio input device", options=self.PROMPT_DEVICE_OPTIONS, allow_blank=True)
                    yield Static("Audio transcriber to use: ")
                    yield RadioSet(*(RadioButton(label, id=f"transcription_{value}") for label, value, _ in self.TRANSCRIPTION_OPTIONS), id="transcription_selector")
                        # yield Select(
                        #     options=self.TRANSCRIPTION_OPTIONS,
                        #     id="transcription_selector",
                        #     allow_blank=False
                        # )
                    with VerticalScroll(id="prompt-settings"):
                        yield Static("AI Prompt to use: ")
                        with RadioSet(id="prompt_selector"):
                            for label, value in self.PROMPT_OPTIONS:
                                yield RadioButton(label, id=f"prompt_{value}")
                        # yield Select(
                        #     options=self.PROMPT_OPTIONS,
                        #     id="prompt_selector",
                        #     allow_blank=False
                        # )
                    with VerticalScroll(id="model-management"):
                        yield Static("Whisper Model Management:")
                        yield ModelDownloadButton("ggml-base.en", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin", "Download Base English Model")
                        yield ModelDownloadButton("ggml-small.en", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin", "Download Small English Model")
                        # Add more model download buttons as needed
                    yield Button("Save Settings", id="save_settings", variant="primary")
            with TabPane("Log", id="log-tab-pane"):
                with VerticalScroll(id="log-vertical-scroll"):
                    yield Static("Awaiting logging information", id="log")       
                             
            with TabPane("About", id="about"):
                yield Static("""
888      888      888b     d888       8888888b.                                          888          8888888888                   d8b                                    d8b                         
888      888      8888b   d8888       888   Y88b                                         888          888                          Y8P                                    Y8P                         
888      888      88888b.d88888       888    888                                         888          888                                                                                             
888      888      888Y88888P888       888   d88P 888d888  .d88b.  88888b.d88b.  88888b.  888888       8888888    88888b.   .d88b.  888 88888b.   .d88b.   .d88b.  888d888 888 88888b.   .d88b.        
888      888      888 Y888P 888       8888888P"  888P"   d88""88b 888 "888 "88b 888 "88b 888          888        888 "88b d88P"88b 888 888 "88b d8P  Y8b d8P  Y8b 888P"   888 888 "88b d88P"88b       
888      888      888  Y8P  888       888        888     888  888 888  888  888 888  888 888          888        888  888 888  888 888 888  888 88888888 88888888 888     888 888  888 888  888       
888      888      888   "   888       888        888     Y88..88P 888  888  888 888 d88P Y88b.        888        888  888 Y88b 888 888 888  888 Y8b.     Y8b.     888     888 888  888 Y88b 888       
88888888 88888888 888       888       888        888      "Y88P"  888  888  888 88888P"   "Y888       8888888888 888  888  "Y88888 888 888  888  "Y8888   "Y8888  888     888 888  888  "Y88888       
                                                                                888                                            888                                                          888       
                                                                                888                                       Y8b d88P                                                     Y8b d88P       
                                                                                888                                        "Y88P"                                                       "Y88P"        
 .d888                               d8888               888 d8b                      888         d8b                                         d8888                            888                    
d88P"                               d88888               888 Y8P                      888         Y8P                                        d88888                            888                    
888                                d88P888               888                          888                                                   d88P888                            888                    
888888  .d88b.  888d888           d88P 888 888  888  .d88888 888  .d88b.          .d88888 888d888 888 888  888  .d88b.  88888b.            d88P 888  .d88b.   .d88b.  88888b.  888888 .d8888b         
888    d88""88b 888P"            d88P  888 888  888 d88" 888 888 d88""88b        d88" 888 888P"   888 888  888 d8P  Y8b 888 "88b          d88P  888 d88P"88b d8P  Y8b 888 "88b 888    88K             
888    888  888 888             d88P   888 888  888 888  888 888 888  888 888888 888  888 888     888 Y88  88P 88888888 888  888         d88P   888 888  888 88888888 888  888 888    "Y8888b.        
888    Y88..88P 888            d8888888888 Y88b 888 Y88b 888 888 Y88..88P        Y88b 888 888     888  Y8bd8P  Y8b.     888  888        d8888888888 Y88b 888 Y8b.     888  888 Y88b.       X88        
888     "Y88P"  888           d88P     888  "Y88888  "Y88888 888  "Y88P"          "Y88888 888     888   Y88P    "Y8888  888  888       d88P     888  "Y88888  "Y8888  888  888  "Y888  88888P'        
                                                                                                                                                         888                                          
                                                                                                                                                    Y8b d88P                                          
                                                                                                                                                     "Y88P"                                           
888                                                                                                                                                                                                   
888                                                                                                                                                                                                   
888                                                                                                                                                                                                   
88888b.  888  888                                                                                                                                                                                     
888 "88b 888  888                                                                                                                                                                                     
888  888 888  888                                                                                                                                                                                     
888 d88P Y88b 888                                                                                                                                                                                     
88888P"   "Y88888                                                                                                                                                                                     
              888                                                                                                                                                                                     
         Y8b d88P                                                                                                                                                                                     
          "Y88P"                                                                                                                                                                                                                                                                                  
                             
                           
  _   _       _     _       _                     ___       _                       _           _   ____                _            _         _     _     ____ 
 | | | |_ __ (_) __| | __ _| |_ _   _ _ __ ___   |_ _|_ __ | |_ ___  __ _ _ __ __ _| |_ ___  __| | |  _ \ _ __ ___   __| |_   _  ___| |_ ___  | |   | |   / ___|
 | | | | '_ \| |/ _` |/ _` | __| | | | '_ ` _ \   | || '_ \| __/ _ \/ _` | '__/ _` | __/ _ \/ _` | | |_) | '__/ _ \ / _` | | | |/ __| __/ __| | |   | |  | |    
 | |_| | | | | | (_| | (_| | |_| |_| | | | | | |  | || | | | ||  __/ (_| | | | (_| | ||  __/ (_| | |  __/| | | (_) | (_| | |_| | (__| |_\__ \ | |___| |__| |___ 
  \___/|_| |_|_|\__,_|\__,_|\__|\__,_|_| |_| |_| |___|_| |_|\__\___|\__, |_|  \__,_|\__\___|\__,_| |_|   |_|  \___/ \__,_|\__,_|\___|\__|___/ |_____|_____\____|
                                                                    |___/                                                                                       
All Rights Reserved
Unidatum Integrated Products LLC © 2024
http://www.unidatum.com/

This is a command-line tool for transcribing audio feeds and packaging the transcriptions 
into pre-made large language model (LLM) prompt templates and capturing the responses from the LLMs.    

                             """)
            # # Update the Settings tab to show current values
            # yield Input(placeholder="Ollama API Host", id="ollama_host", value=self.settings.get("ollama_host", self.ollama_host))
            # yield Select(options=[], id="ollama_model", value=self.settings.get("ollama_model", ""), prompt="Select Ollama Model")
            
            # # Update the device selector
            # yield Select(id="device_selector", prompt="Select an audio input device", options=self.PROMPT_DEVICE_OPTIONS, value=self.settings.get("selected_device", ""))
            
            # # Update the transcription selector
            # for label, value, _ in self.TRANSCRIPTION_OPTIONS:
            #     yield RadioButton(label, id=f"transcription_{value}", value=(value == self.settings.get("transcription_method", self.transcription_method)))

            # # Update the prompt selector
            # for label, value in self.PROMPT_OPTIONS:
            #     yield RadioButton(label, id=f"prompt_{value}", value=(value == self.settings.get("current_prompt", "default")))


    def on_mount(self) -> None:
        logging.info("LemonPepper mounted")
        self.update_transcription_options()
        
        # Set the first prompt option as default
        first_prompt_id = f"prompt_{self.PROMPT_OPTIONS[0][1]}"
        prompt_radio_button = self.query_one(f"#{first_prompt_id}", RadioButton)
        if prompt_radio_button:
            prompt_radio_button.value = True
        
        self.audio_monitor = self.query_one(AudioLevelMonitor)
        self.devices = self.list_audio_devices()
        self.update_ai_status("Idle")
        self.ollama_api = OllamaAPI(host="http://192.168.1.81:11434", model="llama3.1:latest", app=self)
        self.stop_event = False
        self.update_thread = Thread(target=self.update_content, daemon=True)
        self.update_thread.start()
        self.log_thread = Thread(target=self.update_log, daemon=True)
        self.log_thread.start()
        self.start_new_session()
        self.set_timer(0.1, self.update_device_selector)
        
        # Set border titles for various components
        self.query_one("#transcription", TextArea).border_title = "Complete Transcription (editable)"
        self.query_one("#left-pane", VerticalScroll).border_title = "LLM Response"
        self.query_one("#partial", TextArea).border_title = "Partial Transcription"
        self.query_one("#user_input", TextArea).border_title = "Include Custom Message with Transcription"
        self.query_one("#log-vertical-scroll", VerticalScroll).border_title = "Log"
        self.query_one("#audio-device-settings", VerticalScroll).border_title = "Audio Device"
        self.query_one("#prompt-settings", VerticalScroll).border_title = "LLM Prompt Engineering"
        # Populate Ollama model options and set value
        self.update_ollama_models()
        self.apply_saved_settings()
        self.update_device_selector()
        # Apply saved settings after populating options
        #self.apply_saved_settings()
        # Set up signal handling
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        #self.update_content_timer = self.set_interval(1/30, self.update_content)
        
        # Find and set BlackHole 2ch as default if it exists
        # blackhole_index = self.find_blackhole_2ch()
        # if blackhole_index is not None:
        #     self.selected_device = self.devices[blackhole_index]
        #     self.log(f"Default audio device set to BlackHole 2ch (index: {blackhole_index})")
            
        #     # Update the device selector in the UI
        #     device_selector = self.query_one("#device_selector", expect_type=Select)
        #     device_selector.value = blackhole_index
            
        #     # Start transcribing with the selected device
        #     self.transcribe_thread = Thread(target=self.start_transcribing, args=(blackhole_index,), daemon=True)
        #     self.transcribe_thread.start()
        # else:
        #     self.log("BlackHole 2ch not found. Please select an audio device manually.")

    def apply_saved_settings(self):
        # Apply Ollama host setting
        ollama_host_input = self.query_one("#ollama_host", Input)
        ollama_host_input.value = self.settings.get("ollama_host", "http://localhost:11434")

        # For Ollama model, we'll set it after the options are populated
        self.saved_ollama_model = self.settings.get("ollama_model")

        # For device selection, we'll set it after the options are populated
        self.saved_device = self.settings.get("selected_device")

        # Apply transcription method
        self.transcription_method = self.settings.get("transcription_method", self.TRANSCRIPTION_OPTIONS[0][1])
        transcription_radio = self.query_one(f"#transcription_{self.transcription_method}", RadioButton)
        if transcription_radio:
            transcription_radio.value = True
        else:
            # If the saved transcription method is not found, set to the first available option
            first_option = self.query_one("#transcription_selector RadioButton")
            if first_option:
                first_option.value = True
                self.transcription_method = first_option.id.split("_")[1]

        # Apply prompt settings
        current_prompt = self.settings.get("current_prompt", "default")
        prompt_radio = self.query_one(f"#prompt_{current_prompt}", RadioButton)
        if prompt_radio:
            prompt_radio.value = True

        # Log the loaded settings
        logging.info(f"Loaded settings: {self.settings}")
        logging.info(f"Current transcription method: {self.transcription_method}")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "ollama_host":
            if self.ollama_host != event.value:
                logging.info(f"Ollama host changed from {self.ollama_host} to: {event.value}")
                self.ollama_host = event.value
                self.settings["ollama_host"] = event.value
                self.unsaved_changes = True
                self.post_message(self.OllamaHostChanged(event.value))
    
    #def on_lemon_pepper_ollama_host_changed(self, event: OllamaHostChanged) -> None:
    #    logging.info(f"Updating Ollama host to: {event.host}")
    #    self.ollama_host = event.host

    @work(exclusive=True)
    async def update_ollama_models(self) -> None:
        logging.info(f"Updating Ollama models from host: {self.ollama_host}")
        model_select = self.query_one("#ollama_model", Select)
        refresh_button = self.query_one("#refresh_models", Button)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()
                logging.info(f"Received JSON response: {json.dumps(data, indent=2)}")
                models = data.get("models", [])
                model_options = [(model["name"], model["name"]) for model in models]

                model_select = self.query_one("#ollama_model", Select)
                model_select.set_options(model_options)
                logging.info(f"Updated Select widget options: {model_options}")
                
                # Apply saved Ollama model setting after options are populated
                if self.saved_ollama_model and self.saved_ollama_model in dict(model_options):
                    model_select.value = self.saved_ollama_model
                    logging.info(f"Applied saved Ollama model: {self.saved_ollama_model}")
                else:
                    model_select.value = Select.BLANK
                    logging.info("No saved Ollama model applied")

                self.notify("Model list updated successfully", severity="information")
                logging.info(f"Model list updated successfully. Found {len(models)} models.")
        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to fetch models: HTTP {e.response.status_code}"
            logging.error(error_msg)
            logging.error(f"Response content: {e.response.text}")
            self.notify(error_msg, severity="error")
        except httpx.RequestError as e:
            error_msg = f"Error fetching models: {str(e)}"
            logging.error(error_msg)
            self.notify(error_msg, severity="error")
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {str(e)}"
            logging.error(error_msg)
            logging.error(f"Raw response content: {response.text}")
            self.notify(error_msg, severity="error")
        finally:
            refresh_button.disabled = False
            refresh_button.label = "Refresh Models"

    def _fetch_models_worker(self):
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_options = [(model["name"], model["name"]) for model in models]
                return model_options, "success"
            else:
                return None, f"Failed to fetch models: HTTP {response.status_code}"
        except requests.RequestException as e:
            return None, f"Error fetching models: {str(e)}"

    def on_model_download_button_download_complete(self, message: ModelDownloadButton.DownloadComplete):
        self.notify(f"Model {message.model_name} downloaded successfully to {message.model_path}")
        # Update the available models in the transcription options
        self.update_transcription_options()

    def on_model_download_button_download_progress(self, message: ModelDownloadButton.DownloadProgress):
        # This method will be called when the download progress updates
        # You can use it to update a global progress indicator if needed
        pass

    def update_transcription_options(self):
        radio_set = self.query_one("#transcription_selector", RadioSet)
        for button in radio_set.children:
            option = next((opt for opt in self.TRANSCRIPTION_OPTIONS if f"transcription_{opt[1]}" == button.id), None)
            if option:
                button.disabled = not os.path.exists(option[2]) if option[2] else False
        
        if not radio_set.pressed_button or radio_set.pressed_button.disabled:
            first_enabled = next((button for button in radio_set.children if not button.disabled), None)
            if first_enabled:
                first_enabled.value = True
        
        self.transcription_method = radio_set.pressed_button.id.split("_")[1] if radio_set.pressed_button else None


    def _add_new_radio_set(self, settings_container):
        # Create new radio buttons
        new_radio_buttons = [
            RadioButton(label, id=f"transcription_{value}")
            for label, value, _ in self.TRANSCRIPTION_OPTIONS
        ]
        
        # Create a new RadioSet and mount it
        new_radio_set = RadioSet(*new_radio_buttons, id="transcription_selector")
        settings_container.mount(new_radio_set)
        
        # Ensure a button is selected
        if new_radio_buttons:
            new_radio_set.press_button(new_radio_buttons[0])
        
        # Update the transcription method
        self.transcription_method = new_radio_set.pressed_button.id.split("_")[1] if new_radio_set.pressed_button else self.TRANSCRIPTION_OPTIONS[0][1]

    def get_shortcuts(self) -> dict[str, str]:
        return {binding.key: binding.description for binding in self.BINDINGS}
    
    def initialize_transcriber(self):
        logging.info(f"Initializing transcriber with method: {self.transcription_method}")
        transcription_info = next((option for option in self.TRANSCRIPTION_OPTIONS if option[1] == self.transcription_method), None)
        if transcription_info:
            _, method, model_name = transcription_info
            logging.info(f"Transcription info: method={method}, model_name={model_name}")
            if method == "vosk":
                logging.info("Initializing Vosk transcriber")
                old_stdout = sys.stdout
                sys.stdout = LogRedirector()
                try:
                    self.transcriber = AudioTranscriber()
                    self.transcriber.transcription_method = method
                    logging.info("Vosk transcriber initialized successfully")
                except Exception as e:
                    logging.error(f"Error initializing Vosk transcriber: {e}")
                finally:
                    sys.stdout = old_stdout
            elif method == "google_cloud":
                logging.info("Initializing Google Cloud transcriber")
                try:
                    self.transcriber = AudioTranscriberGoogleCloud()
                    self.transcriber.transcription_method = method
                    logging.info("Google Cloud transcriber initialized successfully")
                except Exception as e:
                    logging.error(f"Error initializing Google Cloud transcriber: {e}")
            elif method.startswith("whisper"):
                if model_name:
                    model_path = os.path.join(get_model_directory(), model_name)
                    logging.info(f"Initializing Whisper transcriber with model path: {model_path}")
                    try:
                        self.transcriber = WhisperStreamTranscriber(model_path=model_path)
                        self.transcriber.transcription_method = method
                        if self.transcriber.model is None:
                            logging.error("Error: Failed to initialize Whisper model")
                            return None
                        logging.info("Whisper transcriber initialized successfully")
                    except Exception as e:
                        logging.error(f"Error initializing Whisper transcriber: {e}")
                        return None
                else:
                    logging.error("Error: Whisper model not selected")
                    return None
            else:
                logging.error(f"Unknown transcription method: {method}")
                return None

            logging.info(f"Transcriber initialized: {method}")
            return self.transcriber
        else:
            logging.error(f"Error: Unknown transcription method {self.transcription_method}")
            return None

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "prompt_selector":
            selected_value = event.pressed.id.split("_", 1)[1]  # Extract value from radio button id
            self.ollama_api.set_prompt(selected_value)
            self.settings["current_prompt"] = selected_value
            self.unsaved_changes = True
            self.log(f"Changed prompt to: {event.pressed.label}")
        elif event.radio_set.id == "transcription_selector":
            selected_value = event.pressed.id.split("_", 1)[1]
            self.transcription_method = selected_value
            self.settings["transcription_method"] = selected_value
            self.unsaved_changes = True
            if self.transcriber:
                self.transcriber.stop_transcribing()
            self.transcriber = None  # Reset the transcriber
            logging.info(f"Changed transcription method to: {event.pressed.label}")
            logging.info(f"Current transcription method: {self.transcription_method}")

        # if event.radio_set.id == "prompt_selector":
        #     selected_value = event.pressed.id.split("_", 1)[1]  # Extract value from radio button id
        #     self.ollama_api.set_prompt(selected_value)
        #     self.settings["current_prompt"] = selected_value
        #     self.unsaved_changes = True
        #     self.log(f"Changed prompt to: {event.pressed.label}")
        # elif event.radio_set.id == "transcription_selector":
        #     selected_value = event.pressed.id.split("_", 1)[1]
        #     if selected_value != self.transcription_method:
        #         self.transcription_method = selected_value
        #         self.settings["transcription_method"] = selected_value
        #         self.unsaved_changes = True
        #         if self.transcriber:
        #             self.transcriber.stop_transcribing()
        #         self.transcriber = None  # Reset the transcriber
        #         logging.info(f"Changed transcription method to: {event.pressed.label}")
        #     else:
        #         logging.info(f"Transcription method unchanged: {event.pressed.label}")

    def save_settings(self):
        if not self.unsaved_changes:
            logging.info("No changes to save")
            return

        settings_dir = appdirs.user_config_dir("lemonpepper", "UnidatumIntegratedProductsLLC")
        os.makedirs(settings_dir, exist_ok=True)
        settings_file = os.path.join(settings_dir, "settings.json")
        
        # Convert Select.BLANK to None for JSON serialization
        serializable_settings = {
            k: (None if v is Select.BLANK else v) for k, v in self.settings.items()
        }
        
        try:
            with open(settings_file, "w") as f:
                json.dump(serializable_settings, f, indent=2)
            logging.info(f"Settings saved successfully to {settings_file}")
            logging.info(f"Saved settings: {serializable_settings}")
            self.unsaved_changes = False
        except IOError as e:
            logging.error(f"Error saving settings to {settings_file}: {e}")


    def load_settings(self):
        settings_dir = appdirs.user_config_dir("lemonpepper", "UnidatumIntegratedProductsLLC")
        settings_file = os.path.join(settings_dir, "settings.json")
        
        default_settings = {
            "ollama_host": "http://localhost:11434",
            "ollama_model": Select.BLANK,
            "selected_device": Select.BLANK,
            "transcription_method": self.TRANSCRIPTION_OPTIONS[0][1],
            "current_prompt": "default"
        }
        
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    loaded_settings = json.load(f)
                
                # Convert None back to Select.BLANK for relevant fields
                if loaded_settings.get("ollama_model") is None:
                    loaded_settings["ollama_model"] = Select.BLANK
                if loaded_settings.get("selected_device") is None:
                    loaded_settings["selected_device"] = Select.BLANK
                
                # Merge loaded settings with default settings
                default_settings.update(loaded_settings)
                
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding settings file: {e}")
                logging.info("Using default settings")
                # Rename the corrupted file
                os.rename(settings_file, settings_file + ".corrupted")
                logging.info(f"Renamed corrupted settings file to {settings_file}.corrupted")
            except IOError as e:
                logging.error(f"Error reading settings file: {e}")
                logging.info("Using default settings")
        else:
            logging.info("Settings file not found. Using default settings")
        
        return default_settings
    
    def find_blackhole_2ch(self):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "BlackHole 2ch" in device['name'] and device['max_input_channels'] > 0:
                return i
        return None
    
    def action_copy_ai_response(self):
        self.copy_ai_response_to_clipboard()

    def update_ai_status(self, status: str):
        footer = self.query_one(CustomFooter)
        footer.update_status(status)

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
        
        # Apply saved device setting after options are populated
        if isinstance(self.saved_device, int) and 0 <= self.saved_device < len(selector.options):
            selector.value = self.saved_device
            logging.info(f"Applied saved device: {self.devices[self.saved_device]}")
        else:
            selector.value = Select.BLANK
            logging.info("No saved device applied or invalid saved device")
        
        logging.debug(f"Selector options after update: {selector.options}")
        selector.refresh()
        

    def list_audio_devices(self):
        devices = sd.query_devices()
        return [f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})"
                for i, device in enumerate(devices)]


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
            self.query_one("#partial", TextArea).text = self.partial_transcription
            #self.update_ollama_display()
            #self.query_one(Markdown).update(self.history[self.current_session_id]["ai_responses"])
        self.call_from_thread(update)

    def start_transcribing(self, device_index):
        logging.info(f"Starting transcription with device index: {device_index}")
        logging.info(f"Current transcription method: {self.transcription_method}")
        
        if not self.transcriber or self.transcriber.transcription_method != self.transcription_method:
            logging.info("Transcriber not initialized or method changed, initializing...")
            self.transcriber = self.initialize_transcriber()
        
        if self.transcriber:
            logging.info("Transcriber initialized, starting transcription...")
            try:
                self.transcriber.start_transcribing(device_index=device_index, transcription_callback=self.transcription_callback)
                logging.info("Transcription started successfully")
            except Exception as e:
                logging.error(f"Error starting transcription: {e}")
        else:
            logging.error("Failed to initialize transcriber")

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
        if event.select.id == "ollama_model":
            self.settings["ollama_model"] = event.value
            self.unsaved_changes = True
            logging.info(f"Ollama model changed to: {event.value}")
        elif event.select.id == "device_selector":
            if event.value is not Select.BLANK:
                selected_device_index = event.value
                self.selected_device = self.devices[selected_device_index]
                logging.info(f"Selected device: {event.value}")
                self.settings["selected_device"] = event.value
                logging.info(f"Changed device_selector to: {self.selected_device}")
                self.unsaved_changes = True
                self.transcribe_thread = Thread(target=self.start_transcribing, args=(selected_device_index,), daemon=True)
                self.transcribe_thread.start()
            else:
                self.settings["selected_device"] = None
                logging.info("No device selected")
        elif event.select.id == "prompt_selector":
            # Existing code for prompt selection
            self.ollama_api.set_prompt(event.value)
            self.unsaved_changes = True
            self.log(f"Changed prompt to: {event.value}")
        elif event.select.id == "transcription_selector":
            self.transcription_method = event.value
            if self.transcription_method == "vosk":
                self.transcriber = AudioTranscriber()
            elif self.transcription_method == "google_cloud":
                self.transcriber = AudioTranscriberGoogleCloud()
            self.settings["transcription_selector"] = event.value
            self.unsaved_changes = True
            self.log(f"Changed transcription method to: {self.transcription_method}")        

    # def on_select_changed(self, event: Select.Changed) -> None:
    #     if event.select.id == "device_selector":
    #         selected_device_index = event.value
    #         self.selected_device = self.devices[selected_device_index]
    #         logging.info(f"Selected device: {self.selected_device}")
    #         #self.query_one("#device").update(f"Selected Device: {self.selected_device}")
    #         self.transcribe_thread = Thread(target=self.start_transcribing, args=(selected_device_index,), daemon=True)
    #         self.transcribe_thread.start()
    #     elif event.select.id == "prompt_selector":
    #         # Existing code for prompt selection
    #         self.ollama_api.set_prompt(event.value)
    #         self.log(f"Changed prompt to: {event.value}")

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
        elif event.button.id == "copy_ai_response":
            self.copy_ai_response_to_clipboard()
        elif event.button.id == "update_ollama_settings":
            self.update_ollama_settings()
        elif event.button.id == "refresh_models":
            self.refresh_models()
        elif event.button.id == "save_settings":
            self.save_settings()
            self.unsaved_changes = False
            self.notify("Settings saved successfully")

    def refresh_models(self) -> None:
        logging.info(f"Refreshing models from host: {self.ollama_host}")
        refresh_button = self.query_one("#refresh_models", Button)
        refresh_button.disabled = True
        refresh_button.label = "Refreshing..."
        self.update_ollama_models()
        self.save_settings()

    def copy_ai_response_to_clipboard(self):
        # Get the latest transcription
        latest_transcription = self.query_one("#transcription", TextArea).text
    
        # Assuming ollama_conversation contains the Markdown string
        markdown_content = self.ollama_conversation

        # Combine the latest transcription and AI response
        combined_content = f"> **_TRANSCRIPTION:_** {self.transcription}\n\n---\n\n{markdown_content}"

        # If ollama_conversation is not directly accessible, you might need to get it from the widget
        # markdown_content = self.query_one("#ollama", Static).renderable.markup

        # Copy the Markdown content to clipboard
        copy_to_clipboard(combined_content)
        self.notify("Latest transcription and LLM response copied to clipboard", timeout=3)

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
        self.update_ai_status("Waiting for LLM response...")
        response = self.ollama_api.process_transcription(force=True)
        self.ollama_conversation = self.ollama_api.get_responses()
        self.history[self.current_session_id]["ai_responses"] = self.ollama_conversation
        self.update_ollama_display()
        self.log("Transcription processed.")
    
    def update_ollama_settings(self):
        host = self.query_one("#ollama_host", Input).value
        model = self.query_one("#ollama_model", Select).value
        if host and model:
            self.ollama_api.update_settings(host, model)
            self.settings["ollama_host"] = host
            self.settings["ollama_model"] = model
            self.save_settings()
            self.notify("Ollama settings updated successfully")
        else:
            self.notify("Please provide both host and model", severity="warning")


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

def main():
    logging.info("Starting LemonPepper application")
    try:
        app = LemonPepper()
        app.run()
    except Exception as e:
        print(f"Unhandled exception in LemonPepper: {e}")
        print(traceback.format_exc())
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
        app.exit()
    finally:
        logging.info("LemonPepper application exited")


if __name__ == "__main__":
    main()