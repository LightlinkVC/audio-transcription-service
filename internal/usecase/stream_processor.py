import librosa
import logging
import numpy as np
from typing import Protocol

from internal.model.asr import Asr
from internal.infrastructure.ws import MessagingServerI

logging.basicConfig(level=logging.INFO)

class StreamProcessorI(Protocol):
    def process_chunk(self, chunk): ...

class StreamProcessor(StreamProcessorI):
    def __init__(self, asr: Asr, messaging_server: MessagingServerI):
        self.asr = asr
        self.messaging_server = messaging_server
        self.audio_buffer = []

    def process_chunk(self, chunk):
        audio_data = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_data = audio_data.reshape(-1, 2).mean(axis=1)  
        audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

        self.audio_buffer.extend(audio_data)
        if len(self.audio_buffer) >= 480000:
            self.asr.insert_audio_chunk(np.array(self.audio_buffer[:480000]))
            self.audio_buffer = self.audio_buffer[480000:]
            
            result = self.asr.process_iter()
            if result:
                text = result[2].strip()
                logging.info(f"Partial transcription: {text}")
                self.messaging_server.publish_to_group(
                    chunk.room_id, 
                    {
                        "type": "transcription_update",
                        "payload": {
                            "content": text
                        }
                    }
                )
                
                return text
