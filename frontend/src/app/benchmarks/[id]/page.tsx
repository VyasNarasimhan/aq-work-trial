"use client";

import Link from "next/link";
import { use } from "react";
import { useBenchmark } from "@/lib/hooks/useBenchmark";
import { useRuns } from "@/lib/hooks/useRuns";
import { ProgressView } from "@/components/benchmark/ProgressView";
import { ResultsDashboard } from "@/components/benchmark/ResultsDashboard";

export default function BenchmarkPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { benchmark, error, loading, refetch: refetchBenchmark } = useBenchmark(id);
  const isRunning = benchmark?.status === "running" || benchmark?.status === "pending";
  const { runs, refetch: refetchRuns } = useRuns(id, isRunning ? 3000 : 0);

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
        {/* Header */}
        <div className="mb-6">
          <Link
            href="/benchmarks"
            className="text-blue-600 hover:underline text-sm"
          >
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
              ? new Date(benchmark.started_at).toLocaleString()
              : "N/A"}
          </p>
        </div>

        {/* Progress View (while running) */}
        {isRunning && <ProgressView benchmark={benchmark} />}

        {/* Error State */}
        {benchmark.status === "failed" && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h2 className="text-red-700 font-medium">Benchmark Failed</h2>
            <p className="text-red-600">{benchmark.error}</p>
          </div>
        )}

        {/* Results Dashboard */}
        <ResultsDashboard
          runs={runs}
          benchmarkId={id}
          onRefresh={handleRefresh}
        />

        {/* Completed message */}
        {benchmark.status === "completed" && (
          <div className="mt-8 text-center text-gray-500 text-sm">
            Benchmark completed{" "}
            {benchmark.finished_at
              ? new Date(benchmark.finished_at).toLocaleString()
              : ""}
          </div>
        )}
      </div>
    </main>
  );
}
