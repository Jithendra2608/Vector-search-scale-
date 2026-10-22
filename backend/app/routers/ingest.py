import io
import json
import random
import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from app.core.db import get_qdrant
from app.services.ingester import IngestionService

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

class SyntheticIngestRequest(BaseModel):
    num_documents: int = Field(default=1000, ge=50, le=20000)
    chunk_size: int = Field(default=80, ge=20, le=500)
    overlap: int = Field(default=15, ge=0, le=100)
    recreate_collection: bool = True

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    text_column: str = "text",
    chunk_size: int = 100,
    overlap: int = 20,
    recreate_collection: bool = False,
    client: QdrantClient = Depends(get_qdrant)
):
    """
    Accepts a CSV or JSON file, extracts the text column, chunks, embeds,
    and streams vectors into Qdrant.
    """
    content = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            if text_column not in df.columns:
                # Fallback to the first string/object column if 'text' isn't found
                str_cols = df.select_dtypes(include=["object", "string"]).columns
                if len(str_cols) == 0:
                    raise ValueError(f"Column '{text_column}' not found in CSV.")
                text_column = str_cols[0]
            raw_texts = df[text_column].dropna().astype(str).tolist()

        elif filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                raw_texts = [
                    item.get(text_column, str(item)) if isinstance(item, dict) else str(item)
                    for item in data
                ]
            else:
                raise ValueError("JSON root must be an array of objects or strings.")
        else:
            raise ValueError("Unsupported file format. Please upload a .csv or .json file.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    ingester = IngestionService(client)
    return ingester.ingest_documents(
        raw_texts=raw_texts,
        chunk_size=chunk_size,
        overlap=overlap,
        recreate_collection=recreate_collection
    )

@router.post("/synthetic")
def ingest_synthetic_dataset(
    req: SyntheticIngestRequest,
    client: QdrantClient = Depends(get_qdrant)
):
    """
    Generates a diverse synthetic technical corpus so users can immediately
    test HNSW graphs and Quantization at scale without needing a large CSV file.
    """
    topics = [
        "distributed consensus in Raft and Paxos clusters",
        "HNSW graph traversal and greedy routing in high-dimensional spaces",
        "scalar quantization from float32 to int8 representations",
        "binary quantization using Hamming distance and bitwise XOR popcount",
        "transformer self-attention mechanisms and key-value cache optimization",
        "Linux kernel epoll event loops and zero-copy network sockets",
        "B-tree vs LSM-tree storage engines in modern databases",
        "PostgreSQL MVCC transaction isolation and write-ahead logging",
        "CUDA warp scheduling and shared memory bank conflicts on GPUs",
        "gRPC HTTP/2 multiplexing and Protocol Buffer binary serialization"
    ]
    modifiers = [
        "under high tail-latency pressure",
        "for sub-millisecond p99 SLA compliance",
        "across multi-node cloud deployments",
        "with memory-mapped mmap files on NVMe SSDs",
        "during concurrent write-heavy ingestion spikes",
        "when optimizing recall vs throughput Pareto frontiers"
    ]

    random.seed(42)
    synthetic_docs = [
        f"Technical report #{i}: Analysis of {random.choice(topics)} {random.choice(modifiers)}. "
        f"Furthermore, combining {random.choice(topics)} with {random.choice(modifiers)} "
        f"alters memory bandwidth and L3 CPU cache locality significantly (sample {random.randint(1000, 9999)})."
        for i in range(req.num_documents)
    ]

    ingester = IngestionService(client)
    return ingester.ingest_documents(
        raw_texts=synthetic_docs,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
        recreate_collection=req.recreate_collection
    )