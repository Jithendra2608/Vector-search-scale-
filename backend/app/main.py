from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import ingest, bench

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Interactive HNSW & Vector Quantization Benchmarking Engine"
)

# Allow Next.js frontend (localhost:3000) to call FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(bench.router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}
