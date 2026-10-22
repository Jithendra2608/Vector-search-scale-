import time
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingService:
    _model: SentenceTransformer | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Loads the embedding model into memory once (Singleton).
        """
        if cls._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._model

    @classmethod
    def encode_batch(cls, texts: list[str], batch_size: int = 64) -> tuple[np.ndarray, float]:
        """
        Encodes a list of strings into normalized 384-d float32 vectors.
        Returns the numpy matrix and the time taken in seconds.
        """
        model = cls.get_model()
        start_time = time.perf_counter()
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        elapsed = time.perf_counter() - start_time
        return embeddings, elapsed