"""
API for capturing screenshots with integrated OCR and captioning.
"""
from PIL import Image
from mss import mss
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from torch import no_grad
from logging import getLogger
from lidia.config.config import global_config
from typing import Dict, Optional

logger = getLogger(__name__)

class ScreenshotAPI:
    """API for capturing and analyzing screenshots."""
    
    def __init__(self):
        # Initialize OCR
        ocr_model = global_config["ocr"]["model"]
        self.ocr_processor = TrOCRProcessor.from_pretrained(ocr_model)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model)
        self.ocr_model.eval()

        # Initialize image captioning
        caption_model = global_config["image_captioning"]["model"]
        self.caption_pipeline = pipeline("image-to-text", model=caption_model, use_fast=False)

        # Get default monitor index
        self.default_monitor = global_config["screenshot"].get("monitor_index", 1)

    def _perform_ocr(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            pixel_values = self.ocr_processor(images=image, return_tensors="pt").pixel_values
            with no_grad():
                generated_ids = self.ocr_model.generate(pixel_values)
            text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        except Exception as e:
            logger.error("Error during OCR: %s", e)
            return f"Error during OCR: {e}"

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        try:
            result = self.caption_pipeline(image)
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No caption generated.")
            return "No caption generated."
        except Exception as e:
            logger.error("Error generating caption: %s", e)
            return f"Error generating caption: {e}"

    def execute(self, params: Optional[Dict] = None) -> Dict:
        """
        Capture a screenshot with integrated OCR and captioning.
        
        Args:
            params: Optional dictionary with 'monitor_index'
            
        Returns:
            Dictionary containing screenshot data including caption, OCR text, and resolution
        """
        monitor_index = (params or {}).get('monitor_index', self.default_monitor)
        
        try:
            with mss() as sct:
                monitors = sct.monitors
                if monitor_index < 1 or monitor_index >= len(monitors):
                    monitor_index = 1
                    
                screenshot = sct.grab(monitors[monitor_index])
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                
                # Perform OCR and generate caption
                ocr_text = self._perform_ocr(image)
                caption = self._generate_caption(image)
                
                return {
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "resolution": image.size
                }
                
        except Exception as e:
            logger.error("Error capturing screenshot: %s", e)
            return {
                "caption": f"Error capturing screenshot: {e}",
                "ocr_text": "",
                "resolution": (0, 0)
            }