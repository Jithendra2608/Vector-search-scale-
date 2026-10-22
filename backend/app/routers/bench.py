import random
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from app.core.db import get_qdrant
from app.core.config import settings
from app.services.benchmarker import BenchmarkService

router = APIRouter(prefix="/api/bench", tags=["Benchmarking"])

DEFAULT_BENCHMARK_QUERIES = [
    "How does HNSW graph traversal improve approximate nearest neighbor search?",
    "Reducing memory footprint with scalar quantization int8 vs float32",
    "Binary quantization Hamming distance and bitwise XOR performance",
    "Distributed consensus algorithms Raft and Paxos under network partitions",
    "Optimizing transformer key-value cache and GPU memory bandwidth",
    "Database storage engines B-tree versus LSM-tree write amplification",
    "Low latency gRPC HTTP/2 binary serialization in microservices",
    "Linux kernel zero-copy networking and epoll asynchronous I/O",
    "PostgreSQL MVCC concurrency control and write-ahead log durability",
    "Balancing recall and p95 latency on the Pareto frontier"
]

class BenchmarkRequest(BaseModel):
    m: int = Field(default=16, ge=4, le=128, description="HNSW M parameter (max edges per node)")
    ef_construct: int = Field(default=100, ge=16, le=512, description="HNSW efConstruction")
    quantization: str = Field(default="none", pattern="^(none|sq8|bq)$")
    ef_search_list: list[int] = Field(default=[8, 16, 32, 64, 128, 256])
    top_k: int = Field(default=10, ge=1, le=100)
    num_queries: int = Field(default=30, ge=5, le=500)
    custom_queries: list[str] | None = None

@router.get("/status")
def get_collection_status(client: QdrantClient = Depends(get_qdrant)):
    """
    Returns live collection vector count and status for the dashboard header.
    """
    exists = client.collection_exists(settings.COLLECTION_NAME)
    if not exists:
        return {"exists": False, "points_count": 0, "status": "uninitialized"}
    
    info = client.get_collection(settings.COLLECTION_NAME)
    return {
        "exists": True,
        "points_count": info.points_count or 0,
        "status": str(info.status.value) if info.status else "unknown",
        "dimension": settings.VECTOR_DIMENSION,
        "model": settings.EMBEDDING_MODEL
    }

@router.post("/run")
def run_benchmark(
    req: BenchmarkRequest,
    client: QdrantClient = Depends(get_qdrant)
):
    """
    Runs a full HNSW + Quantization benchmark sweep across ef_search_list.
    """
    if not client.collection_exists(settings.COLLECTION_NAME):
        raise HTTPException(status_code=400, detail="Collection does not exist. Ingest data first.")

    # Build query pool up to req.num_queries
    if req.custom_queries and len(req.custom_queries) > 0:
        queries = req.custom_queries
    else:
        random.seed(101)
        queries = [
            f"{random.choice(DEFAULT_BENCHMARK_QUERIES)} (variant {i})"
            for i in range(req.num_queries)
        ]

    bench_service = BenchmarkService(client)
    try:
        return bench_service.run_benchmark_suite(
            query_texts=queries,
            top_k=req.top_k,
            ef_search_list=sorted(req.ef_search_list),
            m=req.m,
            ef_construct=req.ef_construct,
            quantization=req.quantization
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare-quantization")
def compare_all_quantizations(
    m: int = 16,
    ef_construct: int = 100,
    top_k: int = 10,
    num_queries: int = 25,
    client: QdrantClient = Depends(get_qdrant)
):
    """
    Runs FP32 ('none'), Scalar INT8 ('sq8'), and Binary 1-bit ('bq') back-to-back
    so the frontend can plot all three Pareto curves on the same chart!
    """
    if not client.collection_exists(settings.COLLECTION_NAME):
        raise HTTPException(status_code=400, detail="Collection does not exist. Ingest data first.")

    random.seed(101)
    queries = [
        f"{random.choice(DEFAULT_BENCHMARK_QUERIES)} (variant {i})"
        for i in range(num_queries)
    ]

    bench_service = BenchmarkService(client)
    ef_list = [8, 16, 32, 64, 128]
    comparison = {}

    for mode in ["none", "sq8", "bq"]:
        comparison[mode] = bench_service.run_benchmark_suite(
            query_texts=queries,
            top_k=top_k,
            ef_search_list=ef_list,
            m=m,
            ef_construct=ef_construct,
            quantization=mode
        )

    return {
        "m": m,
        "ef_construct": ef_construct,
        "top_k": top_k,
        "num_queries": num_queries,
        "profiles": comparison
    }