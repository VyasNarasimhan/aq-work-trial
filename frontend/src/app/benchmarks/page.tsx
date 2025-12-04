"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useBenchmark } from "@/lib/hooks/useBenchmark";
import { useRuns } from "@/lib/hooks/useRuns";
import { ProgressView } from "@/components/benchmark/ProgressView";
import { ResultsDashboard } from "@/components/benchmark/ResultsDashboard";
import { useEffect, useState, Suspense } from "react";
import { api } from "@/lib/api";
import { LogViewer } from "@/components/logs/LogViewer";
import { formatDateCompact } from "@/lib/utils/formatDate";
import type { RunLogs, Run, Benchmark } from "@/types";

export default function BenchmarksPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <BenchmarksRouter />
    </Suspense>
  );
}

function LoadingFallback() {
  return (
    <main className="min-h-screen py-12">
      <div className="max-w-4xl mx-auto px-4">
        <div className="animate-pulse">Loading...</div>
      </div>
    </main>
  );
}

function BenchmarksRouter() {
  const searchParams = useSearchParams();
  const id = searchParams.get("id");
  const runId = searchParams.get("runId");

  // Route: /benchmarks?id=xxx&runId=yyy (run logs)
  if (id && runId) {
    return <RunLogsDetail id={id} runId={runId} />;
  }

  // Route: /benchmarks?id=xxx (benchmark detail)
  if (id) {
    return <BenchmarkDetail id={id} />;
  }

  // Route: /benchmarks (list)
  return <BenchmarksList />;
}

// Benchmarks List Component
function BenchmarksList() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchBenchmarks = async () => {
    try {
      const data = await api.listBenchmarks();
      const sorted = [...data].sort((a, b) => {
        if (!a.started_at) return 1;
        if (!b.started_at) return -1;
        return new Date(b.started_at).getTime() - new Date(a.started_at).getTime();
      });
      setBenchmarks(sorted);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchBenchmarks();
  }, []);

  // Poll when there are running benchmarks
  const hasRunning = benchmarks.some(
    (b) => b.status === "running" || b.status === "pending"
  );

  useEffect(() => {
    if (!hasRunning) return;

    const interval = setInterval(fetchBenchmarks, 5000);
    return () => clearInterval(interval);
  }, [hasRunning]);

  if (loading) {
    return (
      <main className="min-h-screen py-12">
        <div className="max-w-4xl mx-auto px-4">
          <div className="animate-pulse">Loading benchmarks...</div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen py-12">
      <div className="max-w-4xl mx-auto px-4">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Benchmarks</h1>
          <Link
            href="/"
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            New Benchmark
          </Link>
        </div>

        {benchmarks.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <p className="text-gray-500 mb-4">No benchmarks yet</p>
            <Link href="/" className="text-blue-600 hover:underline">
              Upload a task to get started
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {benchmarks.map((benchmark) => (
              <Link
                key={benchmark.id}
                href={`/benchmarks?id=${benchmark.id}`}
                className="block bg-white rounded-lg shadow p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">
                      {benchmark.task_name}
                    </h2>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                        benchmark.harness === "harbor"
                          ? "bg-purple-100 text-purple-800"
                          : "bg-orange-100 text-orange-800"
                      }`}>
                        {benchmark.harness}
                      </span>
                      <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
                        {benchmark.model?.split("/").pop() || "unknown"}
                      </span>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {benchmark.started_at
                        ? formatDateCompact(benchmark.started_at)
                        : "Not started"}
                    </p>
                  </div>
                  {/* <div className="text-right">
                    <div
                      className={`
                      inline-block px-3 py-1 rounded-full text-sm font-medium
                      ${
                        benchmark.status === "completed"
                          ? "bg-green-100 text-green-800"
                          : benchmark.status === "running"
                          ? "bg-blue-100 text-blue-800"
                          : benchmark.status === "failed"
                          ? "bg-red-100 text-red-800"
                          : "bg-gray-100 text-gray-800"
                      }
                    `}
                    >
                      {benchmark.status}
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {benchmark.passed_runs}/{benchmark.total_runs} passed
                    </p>
                  </div> */}
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// Benchmark Detail Component
function BenchmarkDetail({ id }: { id: string }) {
  const { benchmark, error, loading, refetch: refetchBenchmark } = useBenchmark(id);
  const backendSaysRunning = benchmark?.status === "running" || benchmark?.status === "pending";

  // Poll runs while backend says running (will determine actual status from runs)
  const { runs, refetch: refetchRuns } = useRuns(id, backendSaysRunning ? 3000 : 0);

  // Calculate actual running status from runs data
  const completedRunsCount = runs.filter((r) => r.status === "completed").length;
  const totalRuns = benchmark?.total_runs || 10;
  const allRunsCompleted = runs.length > 0 && completedRunsCount >= totalRuns;

  // Consider running if backend says so AND not all runs are completed
  const isRunning = backendSaysRunning && !allRunsCompleted;

  const handleRefresh = () => {
    refetchBenchmark();
    refetchRuns();
  };

  if (loading) {
    return (
      <main className="min-h-screen py-12">
        <div className="max-w-4xl mx-auto px-4">
          <div className="animate-pulse">Loading benchmark...</div>
        </div>
      </main>
    );
  }

  if (error || !benchmark) {
    return (
      <main className="min-h-screen py-12">
        <div className="max-w-4xl mx-auto px-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h2 className="text-red-700 font-medium">Error</h2>
            <p className="text-red-600">{error || "Benchmark not found"}</p>
            <Link href="/benchmarks" className="text-blue-600 hover:underline mt-4 inline-block">
              ← Back to benchmarks
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen py-8 bg-gray-50">
      <div className="max-w-4xl mx-auto px-4">
        <div className="mb-6">
          <Link href="/benchmarks" className="text-blue-600 hover:underline text-sm">
            ← Back to benchmarks
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 mt-2">
            {benchmark.task_name}
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
              benchmark.harness === "harbor"
                ? "bg-purple-100 text-purple-800"
                : "bg-orange-100 text-orange-800"
            }`}>
              {benchmark.harness}
            </span>
            <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
              {benchmark.model?.split("/").pop() || "unknown"}
            </span>
          </div>
          <p className="text-gray-500 text-sm mt-1">
            Started{" "}
            {benchmark.started_at
              ? formatDateCompact(benchmark.started_at)
              : "N/A"}
          </p>
        </div>

        {isRunning && <ProgressView benchmark={benchmark} runs={runs} />}

        {benchmark.status === "failed" && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h2 className="text-red-700 font-medium">Benchmark Failed</h2>
            <p className="text-red-600">{benchmark.error}</p>
          </div>
        )}

        <ResultsDashboard
          runs={runs}
          benchmarkId={id}
          onRefresh={handleRefresh}
        />

        {(benchmark.status === "completed" || allRunsCompleted) && (
          <div className="mt-8 text-center text-gray-500 text-sm">
            Benchmark completed{" "}
            {benchmark.finished_at
              ? formatDateCompact(benchmark.finished_at)
              : ""}
          </div>
        )}
      </div>
    </main>
  );
}

// Run Logs Detail Component
function RunLogsDetail({ id, runId }: { id: string; runId: string }) {
  const [logs, setLogs] = useState<RunLogs | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [logsData, runsData] = await Promise.all([
          api.getRunLogs(id, runId),
          api.getRuns(id),
        ]);
        setLogs(logsData);
        setRun(runsData.find((r) => r.id === runId) || null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load logs");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id, runId]);

  if (loading) {
    return (
      <main className="min-h-screen py-12">
        <div className="max-w-6xl mx-auto px-4">
          <div className="animate-pulse">Loading logs...</div>
        </div>
      </main>
    );
  }

  if (error || !logs) {
    return (
      <main className="min-h-screen py-12">
        <div className="max-w-6xl mx-auto px-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h2 className="text-red-700 font-medium">Error</h2>
            <p className="text-red-600">{error || "Logs not found"}</p>
            <Link
              href={`/benchmarks?id=${id}`}
              className="text-blue-600 hover:underline mt-4 inline-block"
            >
              ← Back to benchmark
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="mb-6">
          <Link
            href={`/benchmarks?id=${id}`}
            className="text-blue-600 hover:underline text-sm"
          >
            ← Back to benchmark
          </Link>
          <div className="flex items-center gap-4 mt-2">
            <h1 className="text-2xl font-bold text-gray-900">
              Run {run?.run_number || "?"}
            </h1>
            {run && (
              <span
                className={`
                  px-3 py-1 rounded-full text-sm font-medium
                  ${
                    run.passed
                      ? "bg-green-100 text-green-800"
                      : "bg-red-100 text-red-800"
                  }
                `}
              >
                {run.passed ? "Passed" : "Failed"}
              </span>
            )}
          </div>
          {run?.error && (
            <p className="text-red-600 text-sm mt-2">Error: {run.error}</p>
          )}
        </div>

        <LogViewer logs={logs} />
      </div>
    </main>
  );
}
