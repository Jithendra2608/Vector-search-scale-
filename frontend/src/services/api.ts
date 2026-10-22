const API_BASE = "http://localhost:8000/api";

export interface CollectionStatus {
  exists: boolean;
  points_count: number;
  status: string;
  dimension?: number;
  model?: string;
}

export interface ParetoPoint {
  ef_search: number;
  mean_recall: number;
  mean_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps: number;
}

export interface BenchmarkResponse {
  configuration: {
    status: string;
    m: number;
    ef_construct: number;
    quantization: string;
    index_rebuild_time_sec: number;
  };
  memory_footprint: {
    points_count: number;
    active_vector_kb: number;
    hnsw_graph_kb: number;
    total_active_ram_kb: number;
    compression_ratio: number;
  };
  num_queries: number;
  top_k: number;
  pareto_sweep: ParetoPoint[];
}

export async function fetchStatus(): Promise<CollectionStatus> {
  const res = await fetch(`${API_BASE}/bench/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function ingestSyntheticData(
  num_documents: number,
  chunk_size: number,
  recreate_collection: boolean
) {
  const res = await fetch(`${API_BASE}/ingest/synthetic`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      num_documents,
      chunk_size,
      overlap: 15,
      recreate_collection,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadDatasetFile(
  file: File,
  chunk_size: number,
  recreate_collection: boolean
) {
  const formData = new FormData();
  formData.append("file", file);
  const url = `${API_BASE}/ingest/upload?chunk_size=${chunk_size}&overlap=20&recreate_collection=${recreate_collection}`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runSingleBenchmark(params: {
  m: number;
  ef_construct: number;
  quantization: string;
  top_k: number;
  num_queries: number;
}): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/bench/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...params,
      ef_search_list: [8, 16, 32, 64, 128, 256],
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runQuantizationComparison(params: {
  m: number;
  ef_construct: number;
  top_k: number;
  num_queries: number;
}): Promise<{
  m: number;
  ef_construct: number;
  profiles: Record<"none" | "sq8" | "bq", BenchmarkResponse>;
}> {
  const query = new URLSearchParams({
    m: String(params.m),
    ef_construct: String(params.ef_construct),
    top_k: String(params.top_k),
    num_queries: String(params.num_queries),
  });
  const res = await fetch(`${API_BASE}/bench/compare-quantization?${query}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}