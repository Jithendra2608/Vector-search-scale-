import time
import numpy as np
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.services.embedder import EmbeddingService

class BenchmarkService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.COLLECTION_NAME

    def configure_index(
        self,
        m: int = 16,
        ef_construct: int = 100,
        quantization: str = "none"  # Options: "none" (FP32), "sq8" (INT8), "bq" (Binary 1-bit)
    ) -> dict:
        """
        Dynamically updates the HNSW graph parameters and Vector Quantization mode
        on the live Qdrant collection, and waits for the background optimizer to finish.
        """
        t0 = time.perf_counter()

        # 1. Select Quantization Config
        quant_config = models.Disabled.DISABLED
        if quantization.lower() == "sq8":
            quant_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,       # Clips extreme 1% outliers so bucket resolution stays sharp
                    always_ram=True      # Keep compressed INT8 vectors in fast RAM
                )
            )
        elif quantization.lower() == "bq":
            quant_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True      # Keep 1-bit packed vectors in RAM
                )
            )

        # 2. Trigger live re-indexing in Qdrant
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
                full_scan_threshold=10
            ),
            quantization_config=quant_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10    # Force immediate HNSW graph building even on small datasets
            )
        )

        # 3. Poll Qdrant until background HNSW graph rebuild completes (status == GREEN)
        while True:
            info = self.client.get_collection(self.collection_name)
            if info.status == models.CollectionStatus.GREEN:
                break
            time.sleep(0.2)

        rebuild_time = time.perf_counter() - t0
        return {
            "status": "ready",
            "m": m,
            "ef_construct": ef_construct,
            "quantization": quantization,
            "index_rebuild_time_sec": round(rebuild_time, 4)
        }

    def estimate_memory_bytes(self, points_count: int, m: int, quantization: str) -> dict:
        """
        Calculates theoretical memory footprint of raw vectors, quantized vectors,
        and HNSW graph adjacency lists (M * 2 * 4 bytes per node).
        """
        dim = settings.VECTOR_DIMENSION
        
        # Raw FP32 vector size: 384 dims * 4 bytes = 1,536 bytes/vector
        raw_vector_bytes = points_count * dim * 4

        # Quantized vector size
        if quantization.lower() == "sq8":
            active_vector_bytes = points_count * dim * 1       # 1 byte per dim (INT8)
        elif quantization.lower() == "bq":
            active_vector_bytes = points_count * (dim // 8)    # 1 bit per dim (48 bytes)
        else:
            active_vector_bytes = raw_vector_bytes             # Uncompressed FP32

        # HNSW Graph Overhead: each point stores ~ (M * 2) neighbor IDs (4-byte uint32) + layer pointers
        hnsw_graph_bytes = points_count * (m * 2 * 4 + 16)

        total_active_ram_bytes = active_vector_bytes + hnsw_graph_bytes

        return {
            "points_count": points_count,
            "active_vector_kb": round(active_vector_bytes / 1024, 2),
            "hnsw_graph_kb": round(hnsw_graph_bytes / 1024, 2),
            "total_active_ram_kb": round(total_active_ram_bytes / 1024, 2),
            "compression_ratio": round(raw_vector_bytes / max(active_vector_bytes, 1), 2)
        }

    def run_benchmark_suite(
        self,
        query_texts: list[str],
        top_k: int = 10,
        ef_search_list: list[int] = [16, 32, 64, 128],
        m: int = 16,
        ef_construct: int = 100,
        quantization: str = "none"
    ) -> dict:
        """
        1. Reconfigures the collection index (M, efConstruct, Quantization).
        2. Computes exact Brute-Force Ground Truth for all query vectors.
        3. Sweeps across ef_search_list to measure Recall@K, p50, p95, p99 Latency, and QPS.
        """
        # Step 1: Apply HNSW & Quantization configuration
        index_meta = self.configure_index(m=m, ef_construct=ef_construct, quantization=quantization)

        col_info = self.client.get_collection(self.collection_name)
        points_count = col_info.points_count or 0
        if points_count == 0:
            raise ValueError("Collection is empty. Please ingest documents before benchmarking.")

        # Adjust top_k if collection has fewer points than top_k
        effective_k = min(top_k, points_count)

        # Step 2: Embed all benchmark queries once upfront
        query_vectors, _ = EmbeddingService.encode_batch(query_texts)

        # Step 3: Compute Exact Brute-Force Ground Truth (exact=True bypasses HNSW and Quantization)
        ground_truth_ids: list[set[str]] = []
        for q_vec in query_vectors:
            exact_res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec.tolist(),
                limit=effective_k,
                search_params=models.SearchParams(exact=True)
            ).points
            ground_truth_ids.append({str(pt.id) for pt in exact_res})

        # Step 4: Sweep across each efSearch value to generate the Pareto Curve
        sweep_results = []
        use_quantization = quantization.lower() in ("sq8", "bq")

        for ef in ef_search_list:
            latencies_ms = []
            recalls = []

            search_params = models.SearchParams(
                hnsw_ef=ef,
                exact=False,
                quantization=models.QuantizationSearchParams(
                    ignore=not use_quantization,
                    rescore=False  # Pure quantized search without FP32 disk rescoring
                )
            )

            for idx, q_vec in enumerate(query_vectors):
                t_start = time.perf_counter()
                approx_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_vec.tolist(),
                    limit=effective_k,
                    search_params=search_params
                ).points
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                latencies_ms.append(t_elapsed_ms)

                # Calculate Recall@K against exact Ground Truth
                approx_ids = {str(pt.id) for pt in approx_res}
                true_ids = ground_truth_ids[idx]
                recall = len(true_ids.intersection(approx_ids)) / len(true_ids) if true_ids else 1.0
                recalls.append(recall)

            # Calculate Latency Percentiles & Recall
            lat_array = np.array(latencies_ms)
            total_time_sec = np.sum(lat_array) / 1000.0

            sweep_results.append({
                "ef_search": ef,
                "mean_recall": round(float(np.mean(recalls)), 4),
                "mean_latency_ms": round(float(np.mean(lat_array)), 3),
                "p50_latency_ms": round(float(np.percentile(lat_array, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lat_array, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lat_array, 99)), 3),
                "qps": round(len(query_vectors) / max(total_time_sec, 1e-6), 2)
            })

        memory_stats = self.estimate_memory_bytes(points_count, m=m, quantization=quantization)

        return {
            "configuration": index_meta,
            "memory_footprint": memory_stats,
            "num_queries": len(query_texts),
            "top_k": effective_k,
            "pareto_sweep": sweep_results
        }
    

    