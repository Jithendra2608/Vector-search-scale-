# 🚀 Vector Search at Scale & Performance Benchmarker

A full-stack web application that allows users to ingest custom text datasets, generate embeddings, and store them in a high-performance vector database. The core feature is an interactive benchmarking dashboard that tests and visualizes the trade-offs between **Recall@K**, **p95 Latency**, and **Memory Usage** across different HNSW hyperparameters (`M`, `efConstruction`, `efSearch`) and vector quantization methods (Scalar Quantization - SQ8, Binary Quantization - BQ).

---

## 🏗️ System Architecture & Workflow

1. **Data Ingestion Pipeline**:
   * Accepts raw `.csv`/`.json` uploads or procedurally generates a synthetic technical corpus.
   * Chunks text documents using a sliding window approach.
   * Batch-embeds passages using local HuggingFace sentence-transformers (`BAAI/bge-small-en-v1.5`).
   * Streams vectors over **gRPC** into Qdrant with `wait=True` synchronization.

2. **Indexing & Quantization Engine**:
   * Exposes configuration controls to dynamically tune HNSW graph topology (`M` max edges, `efConstruction` build beam width).
   * Applies vector quantization profiles: **FP32 baseline** vs. **INT8 Scalar Quantization (SQ8)** vs. **1-bit Binary Quantization (BQ)**.

3. **Benchmarking Engine**:
   * Computes exact Ground Truth via brute-force linear search (`exact=True` vector scan).
   * Sweeps across runtime search beam widths (`efSearch`) to evaluate **Recall@K**.
   * Measures tail latencies (`p50`, `p95`, `p99`) using high-resolution hardware performance counters (`time.perf_counter()`) and NumPy percentiles.

---

## 🛠️ Tech Stack

* **Frontend**: Next.js (React), TypeScript, Tailwind CSS, Recharts (for interactive Pareto curves and memory bar charts), Lucide React.
* **Backend**: Python, FastAPI (asynchronous and lightweight high-throughput API endpoints).
* **Vector Database**: Qdrant (Rust-powered vector search engine with native gRPC binary transport).
* **Embeddings**: `sentence-transformers` (`BAAI/bge-small-en-v1.5`, 384-dimensional normalized vectors).
* **Containerization**: Docker & Docker Compose.

---

## 📂 Project Directory Structure

```text
vector-search-scale/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point & CORS
│   │   ├── core/                # Config and singleton Qdrant gRPC client
│   │   │   ├── config.py
│   │   │   └── db.py
│   │   ├── services/
│   │   │   ├── embedder.py      # HuggingFace batch encoder pipeline
│   │   │   ├── ingester.py      # Text chunking & streaming gRPC upsert
│   │   │   └── benchmarker.py   # Ground-truth, recall, p95/p99 latency evaluation
│   │   └── routers/
│   │       ├── ingest.py        # Dataset upload & synthetic corpus endpoints
│   │       └── bench.py         # Benchmark sweep & multi-quantization routes
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js app router dashboard page
│   │   └── services/            # TypeScript API client (Fetch wrapper)
│   └── package.json
│
├── docker-compose.yml           # Orchestrates Qdrant vector database
└── README.md

