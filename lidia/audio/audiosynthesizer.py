from librosa.effects import time_stretch
from lidia.config.config import global_config
from logging import getLogger
from numpy import array
from pyttsx3 import init as pyttsx3_init
from sounddevice import play, wait
from torch import tensor
from transformers import pipeline, SpeechT5HifiGan

logging = getLogger(__name__)

class AudioSynthesizer:
    def __init__(self, tts_mode="ml"):
        self.tts_mode = tts_mode
        self.speaker_embedding = tensor(global_config.get("speaker_embedding", [0.0]*512)).unsqueeze(0)
        self.sample_rate = 22050
        if self.tts_mode == "ml":
            tts_config = global_config["text_to_speech"]
            vocoder = SpeechT5HifiGan.from_pretrained(tts_config["vocoder"])
            self.tts_pipeline = pipeline("text-to-speech", model=tts_config["model"], vocoder=vocoder, device=-1, trust_remote_code=True)
        else:
            self.tts_pipeline = None
            self.engine = pyttsx3_init()
            voices = self.engine.getProperty("voices")
            for voice in voices:
                if "female" in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    break

    def synthesise_sentence(self, sentence, stretch_factor=1.0):
        if self.tts_mode == "ml":
            try:
                tts_output = self.tts_pipeline(sentence, forward_params={"speaker_embeddings": self.speaker_embedding})
                audio_array = tts_output.get("audio")
                sample_rate = tts_output.get("sample_rate", self.sample_rate)
                if audio_array is not None:
                    audio_array = array(audio_array)
                    if audio_array.ndim > 1:
                        if audio_array.shape[1] not in [1, 2]:
                            audio_array = audio_array.mean(axis=1)
                        elif audio_array.shape[1] == 1:
                            audio_array = audio_array.flatten()
                    if stretch_factor < 1.0:
                        audio_array = time_stretch(audio_array.astype(float), rate=stretch_factor)
                    play(audio_array, samplerate=sample_rate)
                    wait()
                else:
                    logging.error("ML TTS pipeline did not return audio.")
            except Exception as e:
                logging.error("Error in ML TTS pipeline: %s", e)
        else:
            try:
                self.engine.say(sentence)
                self.engine.runAndWait()
            except Exception as e:
                logging.error("Error in system TTS: %s", e)
