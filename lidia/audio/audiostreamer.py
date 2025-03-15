from lidia.config.config import global_config
from transformers import pipeline
from logging import getLogger
from numpy import concatenate, mean, sqrt
from sounddevice import InputStream
from queue import Queue
from time import time

logging = getLogger(__name__)

class AudioStreamer:
    def __init__(self, asr_model_path=global_config["speech_to_text"], sample_rate=16000, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        audio_settings = global_config.get("audio_recording_settings", {})
        self.silence_timeout = audio_settings.get("silence_timeout", 4.0)
        self.chunk_duration = audio_settings.get("audio_chunk_duration", 0.5)
        self.asr = pipeline("automatic-speech-recognition", model=asr_model_path, framework="pt")
        logging.info("ASR model loaded.")

    def record_audio(self):
        logging.info("Starting continuous audio recording...")
        q = Queue()
        recording = []

        def callback(indata, frames, time_info, status):
            if status:
                logging.error("Audio stream status: %s", status)
            q.put(indata.copy())

        non_silence_detected = False
        silence_start = None
        blocksize = int(self.chunk_duration * self.sample_rate)
        with InputStream(samplerate=self.sample_rate, channels=1, blocksize=blocksize, callback=callback):
            while True:
                chunk = q.get()
                recording.append(chunk)
                rms = sqrt(mean(chunk ** 2))
                if rms >= self.silence_threshold:
                    if not non_silence_detected:
                        logging.info("Non-silence detected, starting to record utterance...")
                    non_silence_detected = True
                    silence_start = None
                else:
                    if non_silence_detected:
                        if silence_start is None:
                            silence_start = time()
                        elif time() - silence_start >= self.silence_timeout:
                            logging.info("Silence timeout reached (%.2f seconds), stopping recording.", self.silence_timeout)
                            break
        audio_data = concatenate(recording, axis=0).flatten()
        return audio_data

    def transcribe(self, audio):
        rms = sqrt(mean(audio ** 2))
        if rms < self.silence_threshold:
            logging.info("Final audio is silent. Skipping transcription.")
            return None
        result = self.asr(audio)
        transcription = result.get("text", "").strip()
        logging.info("Transcribed text: %s", transcription)
        return transcription
