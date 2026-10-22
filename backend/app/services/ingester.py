import uuid
import time
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.services.embedder import EmbeddingService

class IngestionService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.COLLECTION_NAME

    def init_collection(self, recreate: bool = False) -> None:
        """
        Creates the Qdrant collection with baseline HNSW settings if it doesn't exist,
        or wipes and recreates it if recreate=True.
        """
        exists = self.client.collection_exists(self.collection_name)
        
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_DIMENSION,
                    distance=models.Distance.COSINE,
                    on_disk=False  # Keep raw vectors in RAM for fair baseline speed
                ),
                # Baseline HNSW parameters (M=16, ef_construct=100 are industry defaults)
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10  # Force HNSW index usage even on smaller datasets
                )
            )

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
        """
        Splits a long document into overlapping word chunks.
        """
        words = text.split()
        if not words:
            return []
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def ingest_documents(
        self,
        raw_texts: list[str],
        chunk_size: int = 100,
        overlap: int = 20,
        batch_size: int = 128,
        recreate_collection: bool = False
    ) -> dict:
        """
        Full pipeline: Chunk -> Batch Embed -> Stream Upsert into Qdrant.
        """
        self.init_collection(recreate=recreate_collection)

        # 1. Chunk all incoming documents
        all_chunks: list[str] = []
        for doc_id, text in enumerate(raw_texts):
            chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)

        total_chunks = len(all_chunks)
        if total_chunks == 0:
            return {"status": "empty", "total_vectors_ingested": 0}

        total_embed_time = 0.0
        total_upsert_time = 0.0

        # 2. Process in batches so we never blow up RAM on huge datasets
        for start_idx in range(0, total_chunks, batch_size):
            batch_texts = all_chunks[start_idx : start_idx + batch_size]

            # Generate 384-d embeddings
            vectors, embed_time = EmbeddingService.encode_batch(batch_texts)
            total_embed_time += embed_time

            # Construct Qdrant PointStructs (ID + Vector + Payload metadata)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "text": text_chunk,
                        "chunk_index": start_idx + i,
                        "word_count": len(text_chunk.split())
                    }
                )
                for i, (text_chunk, vector) in enumerate(zip(batch_texts, vectors))
            ]

            # Upsert batch over gRPC into Qdrant
            t0 = time.perf_counter()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            total_upsert_time += (time.perf_counter() - t0)

        return {
            "status": "success",
            "documents_received": len(raw_texts),
            "total_vectors_ingested": total_chunks,
            "embedding_time_sec": round(total_embed_time, 4),
            "db_upsert_time_sec": round(total_upsert_time, 4),
            "throughput_vectors_per_sec": round(
                total_chunks / (total_embed_time + total_upsert_time), 2
            )
        }