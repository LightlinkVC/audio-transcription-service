# import asyncio
# import logging
# import grpc
# import numpy as np
# import wave
# from datetime import datetime
# from pathlib import Path

# from protogen.audiostream import audiostream_pb2_grpc, audiostream_pb2

# logging.basicConfig(level=logging.INFO)

# class AudioRecorder:
#     def __init__(self, room_id: str):
#         self.room_id = room_id
#         self.sample_rate = 48000  # Должно совпадать с параметрами клиента
#         self.channels = 2
#         self.sampwidth = 2  # 16-bit PCM
#         self.file_path = self._get_wav_path()
#         self.wav_file = wave.open(self.file_path, 'wb')
#         self.wav_file.setnchannels(self.channels)
#         self.wav_file.setsampwidth(self.sampwidth)
#         self.wav_file.setframerate(self.sample_rate)
#         logging.info(f"Recording started: {self.file_path}")

#     def _get_wav_path(self) -> str:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         Path("recordings").mkdir(exist_ok=True)
#         return f"recordings/{self.room_id}_{timestamp}.wav"

#     def add_chunk(self, audio_data: np.ndarray):
#         int16_data = (audio_data * 32767).astype(np.int16)
#         self.wav_file.writeframes(int16_data.tobytes())

#     def close(self):
#         self.wav_file.close()
#         logging.info(f"Recording saved: {self.file_path}")

# class AudioStreamerServicer(audiostream_pb2_grpc.AudioStreamerServiceServicer):
#     async def StreamAudio(self, request_iterator, context):
#         recorder = None
#         room_id = None
#         try:
#             async for chunk in request_iterator:
#                 if recorder is None:
#                     room_id = chunk.room_id
#                     recorder = AudioRecorder(room_id)
                
#                 audio_data = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0
#                 recorder.add_chunk(audio_data)
                
#                 yield audiostream_pb2.Acknowledgement(status=True)
#         except Exception as e:
#             logging.error(f"Recording error: {e}")
#         finally:
#             if recorder:
#                 recorder.close()

# async def serve():
#     server = grpc.aio.server()
#     audiostream_pb2_grpc.add_AudioStreamerServiceServicer_to_server(
#         AudioStreamerServicer(), server
#     )
#     server.add_insecure_port("[::]:8087")
#     logging.info("gRPC server started on port 8087")
#     await server.start()
#     await server.wait_for_termination()

# if __name__ == "__main__":
#     try:
#         asyncio.run(serve())
#     except KeyboardInterrupt:
#         logging.info("Server stopped by user.")
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")

import asyncio
import logging
import grpc
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor
import librosa

from protogen.audiostream import audiostream_pb2_grpc, audiostream_pb2

logging.basicConfig(level=logging.INFO)

class AudioRecorder:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.sample_rate = 48000  # Должно совпадать с параметрами клиента
        self.channels = 2
        self.sampwidth = 2  # 16-bit PCM
        self.file_path = self._get_wav_path()
        self.wav_file = wave.open(self.file_path, 'wb')
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(self.sampwidth)
        self.wav_file.setframerate(self.sample_rate)
        logging.info(f"Recording started: {self.file_path}")

    def _get_wav_path(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("recordings").mkdir(exist_ok=True)
        return f"recordings/{self.room_id}_{timestamp}.wav"

    def add_chunk(self, audio_data: np.ndarray):
        int16_data = (audio_data * 32767).astype(np.int16)
        self.wav_file.writeframes(int16_data.tobytes())

    def close(self):
        self.wav_file.close()
        logging.info(f"Recording saved: {self.file_path}")

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
        
        self.audio_buffer = []  # Буфер для накопления аудио
        self.unique_sentences = set()
        logging.info("ASR model initialized successfully")

    async def StreamAudio(self, request_iterator, context):
        recorder = None
        room_id = None
        try:
            async for chunk in request_iterator:
                if recorder is None:
                    room_id = chunk.room_id
                    recorder = AudioRecorder(room_id)
                    logging.info(f"Starting recording and transcription for room: {room_id}")

                # Декодируем PCM
                audio_data = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32768.0
                recorder.add_chunk(audio_data)

                # Конвертируем стерео -> моно
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)  
                # Ресемплим 48k -> 16k
                audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

                # 🔥 Накапливаем данные перед отправкой в ASR
                self.audio_buffer.extend(audio_data)

                # Отправляем только если накопилось ≥ 1 секунда (16000 сэмплов)
                if len(self.audio_buffer) >= 80000:
                    self.online_processor.insert_audio_chunk(np.array(self.audio_buffer[:80000]))
                    self.audio_buffer = self.audio_buffer[80000:]   # Оставляем остаток
                    
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
            if recorder:
                recorder.close()
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
