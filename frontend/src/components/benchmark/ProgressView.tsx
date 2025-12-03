"use client";

import type { Benchmark } from "@/types";

interface Props {
  benchmark: Benchmark;
}

export function ProgressView({ benchmark }: Props) {
  const progress = (benchmark.completed_runs / benchmark.total_runs) * 100;

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Running Benchmark
          </h2>
          <p className="text-gray-500 text-sm">
            {benchmark.completed_runs} of {benchmark.total_runs} runs completed
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
          <span className="text-green-600 font-medium">
            {benchmark.passed_runs}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Failed:</span>{" "}
          <span className="text-red-600 font-medium">
            {benchmark.completed_runs - benchmark.passed_runs}
          </span>
        </div>
      </div>
    </div>
  );
}
