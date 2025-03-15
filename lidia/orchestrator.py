from time import sleep, time
from re import sub, search
from logging import getLogger

from lidia.config.config import global_config
from lidia.audio.audiostreamer import AudioStreamer
from lidia.audio.audiosynthesizer import AudioSynthesizer
from lidia.llm.llmagent import LLMAgent

logger = getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        audio_streamer: AudioStreamer,
        llm_agent: LLMAgent,
        audio_synthesizer: AudioSynthesizer,
        tts_stretch_factor: float = 1.0,
        stream_timeout: float = 10.0,
        min_sentence_length: int = 10
    ):
        self.audio_streamer = audio_streamer
        self.llm_agent = llm_agent
        self.audio_synthesizer = audio_synthesizer
        self.conversation_history = []
        self.tts_stretch_factor = tts_stretch_factor
        self.stream_timeout = stream_timeout
        self.min_sentence_length = min_sentence_length
        self.in_thinking_phase = False
        self.thinking_buffer = ""

    def _should_process_sentence(self, sentence: str, word: str) -> bool:
        """Determine if the current sentence should be processed based on punctuation and length."""
        ends_with_punctuation = word.endswith('.') or word.endswith('!') or word.endswith('?') or word.endswith('\n')
        sentence_length = len(sentence.split())
        return ends_with_punctuation and sentence_length >= self.min_sentence_length

    def _process_thinking_tags(self, text: str) -> tuple[bool, str]:
        """
        Process thinking tags in the text.
        Returns a tuple of (is_thinking, processed_text).
        """
        if "<think>" in text:
            return True, ""
        elif "</think>" in text:
            return False, ""
        return None, text

    def run(self) -> None:
        while True:
            audio = self.audio_streamer.record_audio()
            transcription = self.audio_streamer.transcribe(audio)
            if not transcription:
                continue

            self.conversation_history.append({"role": "user", "content": transcription})
            
            logger.info("Streaming response from LLMAgent...")
            meta_context = (
                f"You are {global_config['voice_settings']['name']}. "
                "Do not output raw JSON or code blocks. When asked about time, use the DateTime API. "
                "When asked to look at or analyze the screen, use the take_screenshot tool."
            )
            
            response_stream = iter(self.llm_agent.stream_response(
                self.conversation_history, 
                meta_context=meta_context,
                max_length=150
            ))
            
            current_sentence = ""
            last_stream_time = time()
            full_response = ""
            self.in_thinking_phase = False
            self.thinking_buffer = ""

            try:
                while True:
                    try:
                        word = next(response_stream)
                        last_stream_time = time()

                        # Check for thinking tags
                        if "<think>" in word:
                            self.in_thinking_phase = True
                            self.thinking_buffer = ""
                            continue
                        elif "</think>" in word:
                            self.in_thinking_phase = False
                            logger.info("Thinking phase: %s", self.thinking_buffer)
                            continue

                        # Handle text based on thinking phase
                        if self.in_thinking_phase:
                            self.thinking_buffer += word + " "
                            full_response += word + " "
                        else:
                            current_sentence += word + " "
                            full_response += word + " "

                            if self._should_process_sentence(current_sentence, word):
                                processed = sub(r"```.*?```", "", current_sentence)
                                logger.info("Processing sentence: %s", processed)
                                if processed.strip():  # Only synthesize non-empty sentences
                                    self.audio_synthesizer.synthesise_sentence(
                                        processed, 
                                        stretch_factor=self.tts_stretch_factor
                                    )
                                current_sentence = ""

                    except StopIteration:
                        break

                    if time() - last_stream_time > self.stream_timeout:
                        logger.warning("Stream timeout reached after %s seconds", self.stream_timeout)
                        break

            finally:
                # Process any remaining text
                if current_sentence.strip() and not self.in_thinking_phase:
                    processed = sub(r"```.*?```", "", current_sentence)
                    logger.info("Processing final sentence: %s", processed)
                    if processed.strip():  # Only synthesize non-empty sentences
                        self.audio_synthesizer.synthesise_sentence(
                            processed,
                            stretch_factor=self.tts_stretch_factor
                        )

                # Add the full response to conversation history
                if full_response.strip():
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": sub(r"```.*?```", "", full_response.strip())
                    })

            sleep(1)