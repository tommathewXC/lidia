# Lidia

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)

Lidia is a voice-enabled AI assistant framework that integrates audio processing, computer vision, and language models into a cohesive multimodal experience. It can understand spoken commands, visually perceive the desktop environment, and respond with natural-sounding speech.

## 🌟 Features

- **Voice Interaction**: Speech-to-text and text-to-speech capabilities for natural conversations
- **Computer Vision**: Real-time desktop monitoring with OCR and image captioning
- **Multiple LLM Backends**: Support for local models, Ollama, and OpenAI
- **API Extensions**: Modular design with APIs for screenshot capture and datetime
- **Orchestration**: Seamless coordination between all components

## 📋 Requirements

- Python 3.9+
- Required packages listed in `requirements.txt`
- For local LLM mode: 16GB+ RAM recommended
- For speech synthesis: Audio output device
- For speech recognition: Microphone

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lidia.git
cd lidia
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required models:
```bash
python scripts/install_models.py
```

5. Run Lidia with default settings:
```bash
python main.py
```

## 🔧 Configuration

Configuration options can be found in `lidia/config/config.py`. Key settings include:

- **LLM Backend**: Choose between local models, Ollama, or OpenAI
- **Model Paths**: Customize paths to speech recognition, LLM, and TTS models
- **Voice Settings**: Adjust voice parameters and assistant name
- **Screenshot Settings**: Configure monitor index and capture interval
- **Audio Settings**: Adjust silence timeout and chunk duration

## 📊 Architecture

Lidia is organized into several modular components:

- **Audio Module**: Handles speech-to-text and text-to-speech
  - `audiostreamer.py`: Manages audio input and transcription
  - `audiosynthesizer.py`: Converts text to speech

- **Vision Module**: Processes visual information
  - `screenshot.py`: Captures desktop screenshots
  - `ocr_processor.py`: Extracts text from images

- **LLM Module**: Manages language model interactions
  - `llmagent.py`: Interfaces with various LLM backends

- **APIs**: Exposes functionality to the LLM
  - `screenshot_api.py`: Provides screen analysis capabilities
  - `datetime_api.py`: Offers date and time information

- **Orchestrator**: Coordinates all components for a seamless experience

## 💻 Usage

### Basic Usage

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run with default settings
python main.py
```

### LLM Options

```bash
# Use OpenAI's API (requires API key in ~/.accesstokens/openai-api)
python main.py --llm_mode openai

# Use Ollama (must have Ollama installed with models available)
python main.py --llm_mode ollama

# Use a local model (requires downloaded models)
python main.py --llm_mode local
```

### TTS Options

```bash
# Use system TTS (faster, less resource-intensive)
python main.py --tts_mode system

# Use ML-based TTS (higher quality, more resource-intensive)
python main.py --tts_mode ml
```

## 🛠️ Development

The project includes helpful Makefile commands:

```bash
# List all project files (copied to clipboard)
make list

# Collect all project file contents (copied to clipboard)
make collect

# Clean all __pycache__ directories
make clean
```

## 🧠 Model Repository

Lidia currently uses HuggingFace as its primary model repository. Models are downloaded during the installation process and stored locally for optimal performance.

### Current Implementation

The system downloads models from HuggingFace for:
- Speech recognition (Whisper)
- Text-to-speech (SpeechT5)
- LLM functionality (when using local mode)
- OCR processing (TrOCR)
- Image captioning (BLIP)

### Future Extensions

The architecture is designed to be model-repository agnostic. Future versions will:
- Support multiple model repositories beyond HuggingFace
- Allow easy switching between different model sources
- Enable custom model integrations from any open-source repository
- Support model mixing from different sources

## 📚 How It Works

1. **Audio Streaming**: The system continuously listens for user input using the microphone
2. **Speech Recognition**: User speech is transcribed to text via Whisper
3. **LLM Processing**: Text is sent to the configured LLM with context and tools
4. **Tool Integration**: The LLM can use tools like screenshot analysis when needed
5. **Response Generation**: Responses are generated and streamed from the LLM
6. **Speech Synthesis**: Text responses are converted to speech using the configured TTS system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Known Limitations

- Resource usage can be high when using ML-based TTS and local LLMs
- Limited multi-language support in the current version
- OCR may struggle with complex visual content

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- HuggingFace for transformer models
- OpenAI for Whisper and GPT
- Microsoft for Speech T5 and TrOCR
- Ollama for local LLM integration