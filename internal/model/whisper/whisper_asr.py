import logging
from typing import Any

from internal.model.asr import Asr
from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor

logging.basicConfig(level=logging.INFO)

class WhisperAsr(Asr):
    def __init__(self, asr: FasterWhisperASR, asr_processor: OnlineASRProcessor):
        self.asr = FasterWhisperASR(
            "ru",
            "base",
            cache_dir="/app/model-cache",
        )

        if not self.asr.model:
            raise RuntimeError("Model initialization failed")
        
        self.asr_processor = OnlineASRProcessor(
            asr=self.asr,
            buffer_trimming=("segment", 5),
        )

        logging.info("ASR model initialized successfully")

    def insert_audio_chunk(self, audio: Any) -> None:
        self.asr_processor.insert_audio_chunk(audio)

    def process_iter(self) -> tuple[Any | None, Any | None, Any]:
        return self.asr_processor.process_iter()