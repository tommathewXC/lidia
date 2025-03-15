from PIL import Image
from torch import no_grad
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from logging import getLogger

logging = getLogger(__name__)

class OCRProcessor:
    def __init__(self, model: str):
        self.processor = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model)
        self.model.eval()

    def extract_text(self, image: Image.Image) -> str:
        try:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            with no_grad():
                generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        except Exception as e:
            logging.error("Error during OCR: %s", e)
            return f"Error during OCR: {e}"

    def grid_extract(self, image: Image.Image, grid: tuple = (2, 2)) -> str:
        width, height = image.size
        cols, rows = grid
        cell_width = width // cols
        cell_height = height // rows
        grid_results = []
        for row in range(rows):
            for col in range(cols):
                left = col * cell_width
                upper = row * cell_height
                right = left + cell_width
                lower = upper + cell_height
                cell = image.crop((left, upper, right, lower))
                cell_text = self.extract_text(cell)
                grid_results.append("Cell({},{})".format(row+1, col+1) + ": " + cell_text)
        return "\n".join(grid_results)
