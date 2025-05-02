import asyncio
import logging

from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor

from internal.model.whisper.whisper_asr import WhisperAsr
from internal.usecase.stream_processor import StreamProcessor
from internal.delivery.grpc.service import serve

logging.basicConfig(level=logging.INFO)

async def main():
    asr = FasterWhisperASR(
        "ru",
        "base",
        cache_dir="/app/model-cache",
    )

    if not asr.model:
        raise RuntimeError("Model initialization failed")
    
    asr_processor = OnlineASRProcessor(
        asr=asr,
        buffer_trimming=("segment", 5),
    )

    logging.info("ASR model initialized successfully")

    whisper_asr = WhisperAsr(asr, asr_processor)
    stream_processor = StreamProcessor(whisper_asr)

    await serve(stream_processor)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")