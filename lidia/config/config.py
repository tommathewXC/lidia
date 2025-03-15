global_config = {
    "speech_to_text": "models/audio/whisper-base",
    "llm": "models/llm/DeepSeek-R1-Distill-Llama-70B",
    "ollama_model": "deepseek-r1:70b",
    "openai": {
        "model": "gpt-4o", 
        "api_key_path": "~/.accesstokens/openai-api",
        "temperature": 0.7,
        "stream": True
    },
    "text_to_speech": {
        "model": "models/audio/speecht5_tts",
        "vocoder": "models/audio/speecht5_hifigan"
    },
    "speaker_embedding": [0.0] * 512,
    "voice_settings": {
        "meta_instruction": "You are a voice-enabled chatbot that receives voice-to-text instructions and continuously receives visual input from the desktop environment.",
        "name": "Lydia"
    },
    "image_captioning": {
        "model": "Salesforce/blip-image-captioning-base"
    },
    "ocr": {
        "model": "microsoft/trocr-base-printed"
    },
    "screenshot": {
        "monitor_index": 2,
        "capture_interval": 5.0
    },
    "audio_recording_settings": {
        "silence_timeout": 2,
        "audio_chunk_duration": 0.5
    }
}