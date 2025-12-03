"use client";

import type { Run } from "@/types";
import { AttemptCard } from "./AttemptCard";

interface Props {
  runs: Run[];
  benchmarkId: string;
  onRefresh?: () => void;
}

export function ResultsDashboard({ runs, benchmarkId, onRefresh }: Props) {
  // Only show completed runs (not running/pending ones)
  const completedRuns = runs.filter((run) => run.status === "completed" || run.status === "failed");
  const runningCount = runs.filter((run) => run.status === "running" || run.status === "pending").length;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">
          LLM Attempt Results
        </h2>
        <div className="flex items-center gap-4">
          {runningCount > 0 && (
            <span className="text-sm text-blue-600">
              {runningCount} run{runningCount !== 1 ? "s" : ""} in progress...
            </span>
          )}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
            >
              <span>↻</span>
              <span>Refresh Status</span>
            </button>
          )}
        </div>
      </div>

      {/* Attempt Cards - only completed runs */}
      <div className="space-y-4">
        {completedRuns.map((run) => (
          <AttemptCard key={run.id} run={run} benchmarkId={benchmarkId} />
        ))}
      </div>

      {completedRuns.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          {runningCount > 0
            ? `Waiting for ${runningCount} run${runningCount !== 1 ? "s" : ""} to complete...`
            : "No attempts yet. Waiting for runs to complete..."}
        </div>
      )}
    </div>
  );
}
