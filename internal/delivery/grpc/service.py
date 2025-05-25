import grpc
import logging

from protogen.audiostream import audiostream_pb2, audiostream_pb2_grpc
from internal.usecase.stream_processor import StreamProcessorI

logging.basicConfig(level=logging.INFO)

class AudioStreamerService(audiostream_pb2_grpc.AudioStreamerService):
    def __init__(self, stream_processor: StreamProcessorI):
        self.processor = stream_processor

    async def StreamAudio(self, request_iterator, context):
        room_id = None

        try:
            async for chunk in request_iterator:
                if room_id is None:
                    room_id = chunk.room_id
                    logging.info(f"Starting transcription for room: {room_id}")

                partial_transcription = self.processor.process_chunk(chunk)

                yield audiostream_pb2.Acknowledgement(status=True)
        except Exception as e:
            logging.error(f"Exception occured during gRPC stream: {e.with_traceback()}")

async def serve(processor: StreamProcessorI):
    server = grpc.aio.server()
    audiostream_pb2_grpc.add_AudioStreamerServiceServicer_to_server(
        AudioStreamerService(processor), server
    )
    server.add_insecure_port("[::]:8087")
    logging.info("gRPC server started on port 8087")
    
    await server.start()
    await server.wait_for_termination()
