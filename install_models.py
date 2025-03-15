#!/usr/bin/env python3
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from lidia.config.config import global_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MODELS = {
    "audio": {
        "whisper-base": "openai/whisper-base",
        "speecht5_tts": "microsoft/speecht5_tts",
        "speecht5_hifigan": "microsoft/speecht5_hifigan"
    },
    "llm": {
        "DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    },
    "image": {
        "blip": global_config["image_captioning"]["model"]
    },
    "ocr": {
        "trocr": global_config["ocr"]["model"]
    },
    "video": {}
}

BASE_DIR = Path(__file__).parent / "models"

def install_models():
    logging.info("Installing models into: %s", BASE_DIR.resolve())
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for model_type, model_dict in MODELS.items():
        type_dir = BASE_DIR / model_type
        type_dir.mkdir(exist_ok=True)
        for local_name, repo_id in model_dict.items():
            target_dir = type_dir / local_name
            if target_dir.exists():
                logging.info("Model '%s' already installed at %s. Skipping.", repo_id, target_dir)
            else:
                logging.info("Installing model '%s' to %s...", repo_id, target_dir)
                snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
                logging.info("Finished installing '%s'.", repo_id)

if __name__ == "__main__":
    install_models()
