import type {
  Benchmark,
  Run,
  RunLogs,
  UploadResponse,
  CreateBenchmarkResponse,
  HarnessType,
  ModelType,
} from "@/types";

const API_BASE = "/api";

// Standalone exports for direct imports
export async function getRunLogs(benchmarkId: string, runId: string): Promise<RunLogs> {
  const res = await fetch(
    `${API_BASE}/benchmarks/${benchmarkId}/runs/${runId}/logs`
  );
  if (!res.ok) throw new Error("Failed to fetch logs");
  return res.json();
}

export const api = {
  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || "Upload failed");
    }

    return res.json();
  },

  async createBenchmark(uploadId: string, harness: HarnessType = "harbor", model?: ModelType): Promise<CreateBenchmarkResponse> {
    const res = await fetch(`${API_BASE}/benchmarks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId, harness, model }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || "Failed to create benchmark");
    }

    return res.json();
  },

  async listBenchmarks(): Promise<Benchmark[]> {
    const res = await fetch(`${API_BASE}/benchmarks`);
    if (!res.ok) throw new Error("Failed to fetch benchmarks");
    return res.json();
  },

  async getBenchmark(id: string): Promise<Benchmark> {
    const res = await fetch(`${API_BASE}/benchmarks/${id}`);
    if (!res.ok) throw new Error("Benchmark not found");
    return res.json();
  },

  async getRuns(benchmarkId: string): Promise<Run[]> {
    const res = await fetch(`${API_BASE}/benchmarks/${benchmarkId}/runs`);
    if (!res.ok) throw new Error("Failed to fetch runs");
    return res.json();
  },

  async getRunLogs(benchmarkId: string, runId: string): Promise<RunLogs> {
    const res = await fetch(
      `${API_BASE}/benchmarks/${benchmarkId}/runs/${runId}/logs`
    );
    if (!res.ok) throw new Error("Failed to fetch logs");
    return res.json();
  },
};
