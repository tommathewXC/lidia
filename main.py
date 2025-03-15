from argparse import ArgumentParser
from lidia.config.config import global_config
from lidia.audio.audiostreamer import AudioStreamer
from lidia.audio.audiosynthesizer import AudioSynthesizer
from lidia.llm.llmagent import LLMAgent
from lidia.orchestrator import Orchestrator
from logging import basicConfig, INFO, getLogger, ERROR
from os import environ
from warnings import filterwarnings

# Set environment variable to ignore Python warnings
environ["PYTHONWARNINGS"] = "ignore"

# Configure logging levels
getLogger("transformers").setLevel(ERROR)
getLogger("pydantic").setLevel(ERROR)
getLogger("mss").setLevel(ERROR)
getLogger("langchain").setLevel(ERROR)
getLogger("langchain.memory").setLevel(ERROR)
getLogger("langchain_community").setLevel(ERROR)

# Filter various warnings
filterwarnings("ignore", category=DeprecationWarning, module="transformers")
filterwarnings("ignore", category=FutureWarning, module="transformers")
filterwarnings("ignore", category=UserWarning, module="pydantic")
filterwarnings("ignore", category=DeprecationWarning, module="langchain")
filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
filterwarnings("ignore", message=".*ChatOllama.*")
filterwarnings("ignore", message=".*Please see the migration guide.*")
filterwarnings("ignore", category=UserWarning, module="langchain.memory")
filterwarnings("ignore", module="langchain.memory")

basicConfig(
    level=INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = getLogger(__name__)

def main():
    parser = ArgumentParser(description="Voice-enabled AI Assistant")
    parser.add_argument("--llm_mode", type=str, default="ollama", choices=["local", "ollama", "openai"],
                      help="Mode for LLM: local, ollama, or openai")
    parser.add_argument("--tts_mode", type=str, default="system", choices=["ml", "system"],
                      help="Mode for TTS: ml (machine learning) or system")
    args = parser.parse_args()
    audio_streamer = AudioStreamer()
    audio_synthesizer = AudioSynthesizer(tts_mode=args.tts_mode)

    llm_path = global_config["llm"]
    ollama_model = global_config["ollama_model"]
    llm_agent = LLMAgent(mode=args.llm_mode, model_path=llm_path, ollama_model=ollama_model)

    orchestrator = Orchestrator(
        audio_streamer=audio_streamer,
        llm_agent=llm_agent,
        audio_synthesizer=audio_synthesizer,
        tts_stretch_factor=1.0,
        stream_timeout=10.0,
        min_sentence_length=10
    )

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main()