from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AudioChunk(_message.Message):
    __slots__ = ("timestamp", "data", "room_id")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ROOM_ID_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    data: bytes
    room_id: str
    def __init__(self, timestamp: _Optional[int] = ..., data: _Optional[bytes] = ..., room_id: _Optional[str] = ...) -> None: ...

class Acknowledgement(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: bool
    def __init__(self, status: bool = ...) -> None: ...
