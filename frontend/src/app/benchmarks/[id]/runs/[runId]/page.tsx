"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { LogViewer } from "@/components/logs/LogViewer";
import type { RunLogs, Run } from "@/types";

export default function RunLogsPage({
  params,
}: {
  params: Promise<{ id: string; runId: string }>;
}) {
  const { id, runId } = use(params);
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
              href={`/benchmarks/${id}`}
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
        {/* Header */}
        <div className="mb-6">
          <Link
            href={`/benchmarks/${id}`}
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

        {/* Log Viewer */}
        <LogViewer logs={logs} />
      </div>
    </main>
  );
}
