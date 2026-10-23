from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --------------------------------------------------
    # Application
    # --------------------------------------------------
    APP_NAME: str = "NeuroDev"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # --------------------------------------------------
    # Groq LLM
    # --------------------------------------------------
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.2

    # --------------------------------------------------
    # Replicate - BGE-M3 Embeddings
    # --------------------------------------------------
    REPLICATE_API_TOKEN: str
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # --------------------------------------------------
    # Redis - Short Term Memory
    # --------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 86400

    # --------------------------------------------------
    # PostgreSQL - Structured Memory
    # --------------------------------------------------
    POSTGRES_URL: str

    # --------------------------------------------------
    # pgvector
    # --------------------------------------------------
    VECTOR_COLLECTION: str = "neurodev_memory"
    VECTOR_DIMENSION: int = 1024

    # --------------------------------------------------
    # Application Settings
    # --------------------------------------------------
    MAX_RETRIEVAL_RESULTS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.70

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()