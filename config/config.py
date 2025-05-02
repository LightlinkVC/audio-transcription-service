import yaml
from pathlib import Path

from pydantic import BaseModel

class ServerConfig(BaseModel):
    host: str
    port: int

class ModelConfig(BaseModel):
    input_lang: str
    model_size: str
    cache_dir: Path

class StreamConfig(BaseModel):
    input_channels_num: int
    input_frequency: int
    output_channels_num: int
    output_frequency: int


class AppConfig(BaseModel):
    server: ServerConfig
    model: ModelConfig
    stream: StreamConfig

def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

settings = load_config()
