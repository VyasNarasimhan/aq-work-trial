"use client";

import type { Benchmark, Run } from "@/types";

interface Props {
  benchmark: Benchmark;
  runs?: Run[];
}

export function ProgressView({ benchmark, runs }: Props) {
  // Calculate progress from runs if available, otherwise use benchmark data
  const completedRuns = runs
    ? runs.filter((r) => r.status === "completed").length
    : benchmark.completed_runs;
  const passedRuns = runs
    ? runs.filter((r) => r.status === "completed" && r.passed).length
    : benchmark.passed_runs;
  const totalRuns = benchmark.total_runs;

  const progress = totalRuns > 0 ? (completedRuns / totalRuns) * 100 : 0;

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Running Benchmark
          </h2>
          <p className="text-gray-500 text-sm">
            {completedRuns} of {totalRuns} runs completed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
          <span className="text-blue-600 font-medium">Running</span>
        </div>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className="bg-blue-500 h-3 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="mt-4 flex gap-6 text-sm">
        <div>
          <span className="text-gray-500">Passed:</span>{" "}
          <span className="text-green-600 font-medium">{passedRuns}</span>
        </div>
        <div>
          <span className="text-gray-500">Failed:</span>{" "}
          <span className="text-red-600 font-medium">
            {completedRuns - passedRuns}
          </span>
        </div>
      </div>
    </div>
  );
}
