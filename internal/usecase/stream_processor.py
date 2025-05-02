import librosa
import logging
import numpy as np
from typing import Protocol

from internal.model.asr import Asr

logging.basicConfig(level=logging.INFO)

class StreamProcessorI(Protocol):
    def process_chunk(self, chunk): ...

class StreamProcessor(StreamProcessorI):
    def __init__(self, asr: Asr):
        self.asr = asr
        self.audio_buffer = []
        self.unique_sentences = set()

    def process_chunk(self, chunk):
        audio_data = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_data = audio_data.reshape(-1, 2).mean(axis=1)  
        audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

        self.audio_buffer.extend(audio_data)
        if len(self.audio_buffer) >= 80000:
            self.asr.insert_audio_chunk(np.array(self.audio_buffer[:80000]))
            self.audio_buffer = self.audio_buffer[80000:]
            
            result = self.asr.process_iter()
            if result:
                text = result[2].strip()
                if text and text not in self.unique_sentences:
                    logging.info(f"Partial transcription: {text}")
                    self.unique_sentences.add(text)

                    return text
