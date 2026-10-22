"use client";

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import {
  Database,
  Cpu,
  Zap,
  HardDrive,
  Upload,
  Play,
  Layers,
  Activity,
} from "lucide-react";
import {
  fetchStatus,
  ingestSyntheticData,
  uploadDatasetFile,
  runSingleBenchmark,
  runQuantizationComparison,
  CollectionStatus,
  BenchmarkResponse,
} from "@/services/api";

export default function BenchmarkDashboard() {
  // Collection Status State
  const [status, setStatus] = useState<CollectionStatus | null>(null);
  const [loadingIngest, setLoadingIngest] = useState(false);
  const [loadingBench, setLoadingBench] = useState(false);
  const [ingestStats, setIngestStats] = useState<any>(null);

  // Ingestion Controls
  const [numSyntheticDocs, setNumSyntheticDocs] = useState(500);
  const [chunkSize, setChunkSize] = useState(80);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // HNSW & Quantization Controls
  const [mParam, setMParam] = useState(16);
  const [efConstruct, setEfConstruct] = useState(100);
  const [quantization, setQuantization] = useState<"none" | "sq8" | "bq">("none");
  const [topK, setTopK] = useState(10);
  const [numQueries, setNumQueries] = useState(30);

  // Benchmark Results State
  const [singleResult, setSingleResult] = useState<BenchmarkResponse | null>(null);
  const [comparisonResult, setComparisonResult] = useState<Record<
    "none" | "sq8" | "bq",
    BenchmarkResponse
  > | null>(null);

  const refreshStatus = async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
    } catch (err) {
      console.error("Backend unreachable:", err);
    }
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  const handleSyntheticIngest = async () => {
    setLoadingIngest(true);
    try {
      const res = await ingestSyntheticData(numSyntheticDocs, chunkSize, true);
      setIngestStats(res);
      await refreshStatus();
    } catch (e: any) {
      alert("Ingestion Error: " + e.message);
    } finally {
      setLoadingIngest(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;
    setLoadingIngest(true);
    try {
      const res = await uploadDatasetFile(selectedFile, chunkSize, true);
      setIngestStats(res);
      await refreshStatus();
    } catch (e: any) {
      alert("Upload Error: " + e.message);
    } finally {
      setLoadingIngest(false);
    }
  };

  const handleRunSingle = async () => {
    setLoadingBench(true);
    try {
      const res = await runSingleBenchmark({
        m: mParam,
        ef_construct: efConstruct,
        quantization,
        top_k: topK,
        num_queries: numQueries,
      });
      setSingleResult(res);
      setComparisonResult(null);
      await refreshStatus();
    } catch (e: any) {
      alert("Benchmark Error: " + e.message);
    } finally {
      setLoadingBench(false);
    }
  };

  const handleCompareAll = async () => {
    setLoadingBench(true);
    try {
      const res = await runQuantizationComparison({
        m: mParam,
        ef_construct: efConstruct,
        top_k: topK,
        num_queries: numQueries,
      });
      setComparisonResult(res.profiles);
      setSingleResult(res.profiles[quantization]);
      await refreshStatus();
    } catch (e: any) {
      alert("Comparison Error: " + e.message);
    } finally {
      setLoadingBench(false);
    }
  };

  // Format data for Multi-Quantization Comparison Chart
  const multiChartData =
    comparisonResult &&
    comparisonResult.none.pareto_sweep.map((pt, idx) => ({
      ef_search: `ef=${pt.ef_search}`,
      FP32_Recall: +(pt.mean_recall * 100).toFixed(1),
      SQ8_Recall: +(comparisonResult.sq8.pareto_sweep[idx].mean_recall * 100).toFixed(1),
      BQ_Recall: +(comparisonResult.bq.pareto_sweep[idx].mean_recall * 100).toFixed(1),
      FP32_p95: pt.p95_latency_ms,
      SQ8_p95: comparisonResult.sq8.pareto_sweep[idx].p95_latency_ms,
      BQ_p95: comparisonResult.bq.pareto_sweep[idx].p95_latency_ms,
    }));

  const memoryBarData = comparisonResult
    ? [
        {
          name: "FP32 (Baseline)",
          Vector_KB: comparisonResult.none.memory_footprint.active_vector_kb,
          HNSW_KB: comparisonResult.none.memory_footprint.hnsw_graph_kb,
        },
        {
          name: "SQ8 (INT8 - 4x)",
          Vector_KB: comparisonResult.sq8.memory_footprint.active_vector_kb,
          HNSW_KB: comparisonResult.sq8.memory_footprint.hnsw_graph_kb,
        },
        {
          name: "BQ (1-Bit - 32x)",
          Vector_KB: comparisonResult.bq.memory_footprint.active_vector_kb,
          HNSW_KB: comparisonResult.bq.memory_footprint.hnsw_graph_kb,
        },
      ]
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      {/* Header Bar */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-5 mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2 text-white">
            <Layers className="text-indigo-400" />
            Vector Search at Scale & HNSW Benchmarker
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time Pareto trade-offs: Recall@K vs. p95 Tail Latency vs. Memory Quantization (FP32 / SQ8 / BQ)
          </p>
        </div>

        <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 px-4 py-2.5 rounded-xl">
          <Database className="text-emerald-400 w-5 h-5" />
          <div className="text-xs">
            <div className="text-slate-400">Qdrant Active Vectors</div>
            <div className="text-base font-bold text-white">
              {status ? status.points_count.toLocaleString() : "0"} pts
              <span className="ml-2 px-2 py-0.5 text-[10px] rounded bg-emerald-500/20 text-emerald-300 uppercase">
                {status?.status || "offline"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT CONTROL PANEL (4 Cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* 1. Data Ingestion Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
            <h2 className="text-base font-semibold flex items-center gap-2 text-indigo-300">
              <Upload className="w-4 h-4" /> 1. Dataset Ingestion Pipeline
            </h2>

            <div>
              <label className="text-xs text-slate-400 block mb-1">
                Synthetic Corpus Size: <span className="text-white font-semibold">{numSyntheticDocs} docs</span>
              </label>
              <input
                type="range"
                min={100}
                max={5000}
                step={100}
                value={numSyntheticDocs}
                onChange={(e) => setNumSyntheticDocs(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>

            <button
              onClick={handleSyntheticIngest}
              disabled={loadingIngest}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2"
            >
              <Zap className="w-4 h-4" />
              {loadingIngest ? "Embedding & Indexing..." : `Generate & Ingest ${numSyntheticDocs} Docs`}
            </button>

            <div className="border-t border-slate-800 pt-3">
              <label className="text-xs text-slate-400 block mb-1.5">Or Upload Custom CSV / JSON</label>
              <div className="flex gap-2">
                <input
                  type="file"
                  accept=".csv,.json"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-slate-800 file:text-slate-200"
                />
                <button
                  onClick={handleFileUpload}
                  disabled={!selectedFile || loadingIngest}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 rounded-lg text-xs font-medium"
                >
                  Upload
                </button>
              </div>
            </div>

            {ingestStats && (
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs space-y-1">
                <div className="text-emerald-400 font-semibold">✓ Ingestion Complete</div>
                <div className="flex justify-between text-slate-400">
                  <span>Throughput:</span>
                  <span className="text-white font-mono">{ingestStats.throughput_vectors_per_sec} vec/s</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Embedding / gRPC Upsert:</span>
                  <span className="text-white font-mono">
                    {ingestStats.embedding_time_sec}s / {ingestStats.db_upsert_time_sec}s
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* 2. HNSW & Quantization Controls Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
            <h2 className="text-base font-semibold flex items-center gap-2 text-indigo-300">
              <Cpu className="w-4 h-4" /> 2. HNSW & Quantization Tuning
            </h2>

            <div>
              <label className="text-xs text-slate-400 flex justify-between mb-1">
                <span>HNSW M (Max Edges/Node)</span>
                <span className="text-white font-mono">{mParam}</span>
              </label>
              <input
                type="range"
                min={4}
                max={64}
                step={4}
                value={mParam}
                onChange={(e) => setMParam(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 flex justify-between mb-1">
                <span>efConstruction (Build Beam Width)</span>
                <span className="text-white font-mono">{efConstruct}</span>
              </label>
              <input
                type="range"
                min={16}
                max={256}
                step={16}
                value={efConstruct}
                onChange={(e) => setEfConstruct(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1.5">Vector Quantization Mode</label>
              <div className="grid grid-cols-3 gap-2">
                {(["none", "sq8", "bq"] as const).map((q) => (
                  <button
                    key={q}
                    onClick={() => setQuantization(q)}
                    className={`py-2 rounded-xl text-xs font-semibold border transition ${
                      quantization === q
                        ? "bg-indigo-600/20 border-indigo-500 text-indigo-300"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800"
                    }`}
                  >
                    {q === "none" ? "FP32 (1x)" : q === "sq8" ? "SQ8 (4x)" : "BQ (32x)"}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Recall@K (Top-K)</label>
                <input
                  type="number"
                  value={topK}
                  min={1}
                  max={50}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Test Queries</label>
                <input
                  type="number"
                  value={numQueries}
                  min={5}
                  max={200}
                  onChange={(e) => setNumQueries(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="space-y-2 pt-2">
              <button
                onClick={handleRunSingle}
                disabled={loadingBench}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                {loadingBench ? "Running Benchmark..." : "Run Single Profile Sweep"}
              </button>

              <button
                onClick={handleCompareAll}
                disabled={loadingBench}
                className="w-full py-2.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2"
              >
                <Activity className="w-4 h-4" />
                {loadingBench ? "Sweeping FP32 vs SQ8 vs BQ..." : "Compare All Quantizations (FP32 / SQ8 / BQ)"}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT VISUALIZATION PANEL (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* KPI Metric Cards */}
          {singleResult && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl">
                <div className="text-xs text-slate-400">Compression Ratio</div>
                <div className="text-2xl font-bold text-indigo-400 mt-1">
                  {singleResult.memory_footprint.compression_ratio}x
                </div>
                <div className="text-[11px] text-slate-500 mt-1">
                  Active RAM: {singleResult.memory_footprint.total_active_ram_kb} KB
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl">
                <div className="text-xs text-slate-400">Peak Recall@{singleResult.top_k}</div>
                <div className="text-2xl font-bold text-emerald-400 mt-1">
                  {(
                    Math.max(...singleResult.pareto_sweep.map((p) => p.mean_recall)) * 100
                  ).toFixed(1)}
                  %
                </div>
                <div className="text-[11px] text-slate-500 mt-1">vs Brute-Force Ground Truth</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl">
                <div className="text-xs text-slate-400">Best p95 Latency</div>
                <div className="text-2xl font-bold text-amber-400 mt-1">
                  {Math.min(...singleResult.pareto_sweep.map((p) => p.p95_latency_ms))} ms
                </div>
                <div className="text-[11px] text-slate-500 mt-1">gRPC Binary Transport</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl">
                <div className="text-xs text-slate-400">Index Rebuild Time</div>
                <div className="text-2xl font-bold text-sky-400 mt-1">
                  {singleResult.configuration.index_rebuild_time_sec}s
                </div>
                <div className="text-[11px] text-slate-500 mt-1">
                  M={singleResult.configuration.m}, efC={singleResult.configuration.ef_construct}
                </div>
              </div>
            </div>
          )}

          {/* Chart 1: Recall & Latency Sweep across efSearch */}
          {singleResult && !comparisonResult && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">
                HNSW Search Sweep ({singleResult.configuration.quantization.toUpperCase()}) — Recall@{singleResult.top_k} vs. p95 Latency across efSearch
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={singleResult.pareto_sweep}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="ef_search"
                      stroke="#94a3b8"
                      tickFormatter={(v) => `ef=${v}`}
                    />
                    <YAxis yAxisId="left" stroke="#34d399" domain={[0, 1]} />
                    <YAxis yAxisId="right" orientation="right" stroke="#fbbf24" />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b" }}
                    />
                    <Legend />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="mean_recall"
                      name="Mean Recall@K (0-1)"
                      stroke="#34d399"
                      strokeWidth={3}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="p95_latency_ms"
                      name="p95 Latency (ms)"
                      stroke="#fbbf24"
                      strokeWidth={3}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Chart 2: Multi-Quantization Comparison (FP32 vs SQ8 vs BQ) */}
          {comparisonResult && multiChartData && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <h3 className="text-sm font-semibold text-slate-200 mb-3">
                  Recall@{topK} Comparison (%) — FP32 vs SQ8 vs BQ
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={multiChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="ef_search" stroke="#94a3b8" />
                      <YAxis domain={[0, 100]} stroke="#94a3b8" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b" }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="FP32_Recall" stroke="#60a5fa" strokeWidth={2.5} />
                      <Line type="monotone" dataKey="SQ8_Recall" stroke="#34d399" strokeWidth={2.5} />
                      <Line type="monotone" dataKey="BQ_Recall" stroke="#f59e0b" strokeWidth={2.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <HardDrive className="w-4 h-4 text-indigo-400" /> Memory Footprint Breakdown (KB)
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={memoryBarData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b" }}
                      />
                      <Legend />
                      <Bar dataKey="Vector_KB" stackId="a" fill="#6366f1" name="Vector Storage (KB)" />
                      <Bar dataKey="HNSW_KB" stackId="a" fill="#10b981" name="HNSW Graph Links (KB)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Detailed Metrics Table */}
          {singleResult && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 overflow-x-auto">
              <h3 className="text-sm font-semibold text-slate-200 mb-3">
                Raw Pareto Sweep Telemetry ({singleResult.configuration.quantization.toUpperCase()})
              </h3>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2">efSearch</th>
                    <th className="py-2">Recall@{singleResult.top_k}</th>
                    <th className="py-2">p50 (ms)</th>
                    <th className="py-2">p95 (ms)</th>
                    <th className="py-2">p99 (ms)</th>
                    <th className="py-2">Throughput (QPS)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {singleResult.pareto_sweep.map((row) => (
                    <tr key={row.ef_search} className="hover:bg-slate-800/40">
                      <td className="py-2.5 font-bold text-indigo-300">{row.ef_search}</td>
                      <td className="py-2.5 text-emerald-400">{(row.mean_recall * 100).toFixed(2)}%</td>
                      <td className="py-2.5">{row.p50_latency_ms} ms</td>
                      <td className="py-2.5 text-amber-300">{row.p95_latency_ms} ms</td>
                      <td className="py-2.5 text-rose-300">{row.p99_latency_ms} ms</td>
                      <td className="py-2.5 text-sky-300">{row.qps} req/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}