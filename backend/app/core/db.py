from qdrant_client import QdrantClient
from app.core.config import settings

class DatabaseManager:
    _client: QdrantClient | None = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Returns a singleton instance of the QdrantClient.
        Reuses the underlying gRPC channel across requests.
        """
        if cls._client is None:
            cls._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_HTTP_PORT,
                grpc_port=settings.QDRANT_GRPC_PORT,
                prefer_grpc=settings.PREFER_GRPC,
                timeout=60.0  # High timeout for heavy HNSW re-indexing operations
            )
        return cls._client

def get_qdrant() -> QdrantClient:
    """FastAPI dependency wrapper."""
    return DatabaseManager.get_client()