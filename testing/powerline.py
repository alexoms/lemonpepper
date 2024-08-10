import os
import json
import time
from threading import Thread
from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.style import Style
from rich.console import Console

class PowerlineCustomFooter(Static):
    levels = reactive(([float('-inf')] * 2, [float('-inf')] * 2))

    def __init__(self):
        super().__init__()
        self.status_message = "Idle"
        self.spinner = ""
        self.is_spinning = False
        self.spinner_thread = None
        self.demo_state = 0
        self.demo_timer = None

        # Load spinners from JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        spinners_path = os.path.join(script_dir, "lemonpepper/spinners.json")
        with open(spinners_path, "r") as f:
            self.spinners = json.load(f)
        self.current_spinner = self.spinners["dots"]

    def on_mount(self):
        self.update_footer()
        self.set_interval(0.05, self.update_levels)
        self.set_interval(3, self.reset_peaks)
        self.demo_timer = self.set_interval(5, self.run_demo_loop)

    def update_footer(self):
        text = Text()
        
        # Left angle bracket
        text.append("\uE0B6", Style(color="#162a33", bgcolor="black"))

        # Spinner and Status
        text.append(f" {self.spinner} ", Style(color="#f199f2", bgcolor="#223a45", bold=True))
        text.append("", Style(color="#192f52", bgcolor="#223a45"))
        text.append(f" {self.status_message}   ", Style(color="#65d67b", bgcolor="#223a45", bold=True))
        
        # Right angle bracket
        text.append("\uE0B0", Style(color="#162a33", bgcolor="black"))
        
        # Calculate remaining space and add padding
        console = Console()
        status_width = console.measure(text).maximum
        audio_levels = self.render_audio_levels()
        audio_width = console.measure(audio_levels).maximum
        total_width = self.size.width if self.size else 100  # Fallback to 100 if size is not available
        padding = total_width - status_width - audio_width - 1  # -1 for safety
        
        if padding > 0:
            text.append(" " * padding, Style(bgcolor="black"))
        
        # Audio Levels (right-justified)
        text.append(audio_levels)
        
        self.update(text)

    def update_levels(self):
        if hasattr(self.app, 'transcriber') and self.app.transcriber:
            new_levels = self.app.transcriber.get_audio_levels()
            if new_levels != self.levels:
                self.levels = new_levels
                self.update_footer()

    def reset_peaks(self):
        if hasattr(self.app, 'transcriber') and self.app.transcriber:
            self.app.transcriber.reset_peak_levels()

    def normalize_db(self, db_value):
        min_db, max_db = -60, 0
        if db_value <= min_db or db_value == float('-inf'):
            return 0
        return min(max(int((db_value - min_db) * 50 / (max_db - min_db)), 0), 50)

    def render_audio_levels(self):
        result = Text(style=Style(bgcolor="black"))
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

    def update_status(self, message: str):
        self.status_message = message
        self.update_footer()
        
        if message == "Waiting for LLM response...":
            self.start_spinner()
        elif message == "LLM response received":
            self.stop_spinner()
        else:
            self.stop_spinner()

    def start_spinner(self):
        if not self.is_spinning:
            self.is_spinning = True
            self.spinner_thread = Thread(target=self._spin, daemon=True)
            self.spinner_thread.start()

    def stop_spinner(self):
        self.is_spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        self.spinner = ""
        self.update_footer()

    def _spin(self):
        index = 0
        while self.is_spinning:
            frame = self.current_spinner["frames"][index]
            self.spinner = frame
            self.update_footer()
            time.sleep(self.current_spinner["interval"] / 1000)
            index = (index + 1) % len(self.current_spinner["frames"])

    def run_demo_loop(self):
        states = [
            "Waiting for LLM response...",
            "LLM response received",
            "Idle"
        ]
        self.demo_state = (self.demo_state + 1) % len(states)
        self.update_status(states[self.demo_state])

class PowerlineApp(App):
    def compose(self) -> ComposeResult:
        yield Static("Main content area")
        yield PowerlineCustomFooter()

if __name__ == "__main__":
    app = PowerlineApp()
    app.run()