import asyncio
import logging
import wave
from datetime import datetime
from pathlib import Path

import grpc
import librosa
import numpy as np

from protogen.audiostream import audiostream_pb2, audiostream_pb2_grpc
from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor

logging.basicConfig(level=logging.INFO)

class AudioStreamerServicer(audiostream_pb2_grpc.AudioStreamerServiceServicer):
    def __init__(self):
        logging.info(f"Starting server init")
        self.asr = FasterWhisperASR(
            "ru",
            "base",
            cache_dir="/app/model-cache",
        )
        
        if not self.asr.model:
            raise RuntimeError("Model initialization failed")
        
        self.online_processor = OnlineASRProcessor(
            asr=self.asr,
            buffer_trimming=("segment", 5),
        )
        
        self.audio_buffer = []
        self.unique_sentences = set()
        logging.info("ASR model initialized successfully")

    async def StreamAudio(self, request_iterator, context):
        room_id = None
        try:
            async for chunk in request_iterator:
                if room_id is None:
                    room_id = chunk.room_id
                    logging.info(f"Starting recording and transcription for room: {room_id}")

                audio_data = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0

                audio_data = audio_data.reshape(-1, 2).mean(axis=1)  

                audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

                self.audio_buffer.extend(audio_data)

                if len(self.audio_buffer) >= 80000:
                    self.online_processor.insert_audio_chunk(np.array(self.audio_buffer[:80000]))
                    self.audio_buffer = self.audio_buffer[80000:]
                    
                    result = self.online_processor.process_iter()
                    if result:
                        text = result[2].strip()
                        if text and text not in self.unique_sentences:
                            logging.info(f"Partial transcription: {text}")
                            self.unique_sentences.add(text)

                yield audiostream_pb2.Acknowledgement(status=True)
        except Exception as e:
            logging.error(f"Recording error: {e.with_traceback()}")
        finally:
            final_result = self.online_processor.finish()
            if final_result:
                logging.info(f"Final transcription: {final_result}")

async def serve():
    server = grpc.aio.server()
    audiostream_pb2_grpc.add_AudioStreamerServiceServicer_to_server(
        AudioStreamerServicer(), server
    )
    server.add_insecure_port("[::]:8087")
    logging.info("gRPC server started on port 8087")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
