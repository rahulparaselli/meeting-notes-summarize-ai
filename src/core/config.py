from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini / Gemma — set GEMINI_API_KEY in .env
    gemini_api_key: str = "placeholder"
    gemma_model: str = "gemma-4-27b-it"
    embedding_model: str = "models/text-embedding-004"
    gemma_temperature: float = 0.2
    gemma_max_tokens: int = 4096

    # RAG
    chunk_size: int = 300
    chunk_overlap: int = 40
    top_k_retrieval: int = 6
    rerank_top_k: int = 3

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "meetings"

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    similarity_cache_threshold: float = 0.92

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Whisper
    whisper_model: str = "base"


@lru_cache
def get_settings() -> Settings:
    return Settings()
