import os
import aiohttp
import asyncio
from textual.widgets import Button, ProgressBar, Static
from textual.reactive import reactive
from textual.message import Message
from textual.containers import Vertical
from textual.app import ComposeResult

from .utils import get_model_directory

class ModelDownloadButton(Vertical):
    class DownloadComplete(Message):
        def __init__(self, model_name: str, model_path: str):
            self.model_name = model_name
            self.model_path = model_path
            super().__init__()

    class DownloadProgress(Message):
        def __init__(self, progress: float):
            self.progress = progress
            super().__init__()

    download_progress = reactive(0.0)

    def __init__(self, model_name: str, model_url: str, button_label: str):
        super().__init__()
        self.model_name = model_name
        self.model_url = model_url
        self.button_label = button_label
        button_id = f"download_{self.model_name.replace('.', '_')}"
        self.button = Button(self.button_label, id=button_id)
        self.progress_bar = ProgressBar(total=100, show_percentage=True)
        self.status = Static("Ready to download")

    def compose(self) -> ComposeResult:
        yield self.button
        yield self.progress_bar
        yield self.status

    def on_button_pressed(self, event: Button.Pressed):
        self.download_model()

    def download_model(self):
        self.button.disabled = True
        self.progress_bar.visible = True
        self.status.update("Downloading...")
        asyncio.create_task(self._download())

    async def _download(self):
        model_dir = get_model_directory()
        model_path = os.path.join(model_dir, f"{self.model_name}.bin")

        async with aiohttp.ClientSession() as session:
            async with session.get(self.model_url) as response:
                if response.status != 200:
                    self.status.update(f"Download failed: HTTP {response.status}")
                    self.button.disabled = False
                    return

                total_size = int(response.headers.get('content-length', 0))
                if total_size == 0:
                    self.status.update("Unable to determine file size")
                    self.button.disabled = False
                    return

                downloaded_size = 0
                with open(model_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (downloaded_size / total_size) * 100
                        self.download_progress = progress
                        self.progress_bar.update(progress=progress)
                        await asyncio.sleep(0)  # Allow other tasks to run

        self.status.update("Download complete")
        self.button.disabled = False
        self.post_message(self.DownloadComplete(self.model_name, model_path))

    def on_mount(self):
        self.styles.width = "100%"
        self.button.styles.width = "100%"
        self.progress_bar.styles.width = "100%"
        self.progress_bar.visible = False