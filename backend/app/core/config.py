from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vector Search Scale & Benchmarker"
    
    # Qdrant Connection Config
    QDRANT_HOST: str = "localhost"
    QDRANT_HTTP_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    PREFER_GRPC: bool = True
    
    # Collection & Embedding Config
    COLLECTION_NAME: str = "benchmark_collection"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_DIMENSION: int = 384  # bge-small-en-v1.5 outputs 384-d vectors

    class Config:
        env_file = ".env"

settings = Settings()