from mss import mss
from lidia.config.config import global_config
from PIL import Image
from threading import Thread, Lock
from time import sleep
from transformers import pipeline

class ScreenshotCapturer:
    def __init__(self, interval: float = None):
        if interval is None:
            interval = global_config.get("screenshot", {}).get("capture_interval", 3.0)
        self.interval = interval
        self.latest_screenshot = None
        self._running = False
        self._lock = Lock()
        self.monitor_index = global_config.get("screenshot", {}).get("monitor_index", 1)

    def start(self) -> None:
        self._running = True
        thread = Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self) -> None:
        with mss() as sct:
            monitors = sct.monitors
            if self.monitor_index < 1 or self.monitor_index >= len(monitors):
                self.monitor_index = 1
            while self._running:
                screenshot = sct.grab(monitors[self.monitor_index])
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                with self._lock:
                    self.latest_screenshot = img
                sleep(self.interval)

    def get_latest_screenshot(self):
        with self._lock:
            return self.latest_screenshot

class ImageCaptioner:
    def __init__(self, model: str = None):
        if model is None:
            model = global_config.get("image_captioning", {}).get("model", "Salesforce/blip-image-captioning-base")
        self.caption_pipeline = pipeline("image-to-text", model=model, use_fast=False)

    def caption(self, image) -> str:
        if image is None:
            return "No screenshot available."
        result = self.caption_pipeline(image)
        caption = ""
        if isinstance(result, list) and len(result) > 0:
            caption = result[0].get("generated_text", "No caption generated.")
        meta_info = "Resolution: {}x{}.".format(image.width, image.height)
        return "{} {}".format(caption, meta_info)
