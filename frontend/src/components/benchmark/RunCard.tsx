"use client";

import Link from "next/link";
import type { Run } from "@/types";

interface Props {
  run: Run;
  benchmarkId: string;
}

export function RunCard({ run, benchmarkId }: Props) {
  const isRunning = run.status === "running";
  const passed = run.passed;

  return (
    <Link
      href={`/benchmarks/${benchmarkId}/runs/${run.id}`}
      className={`
        block p-4 rounded-lg border-2 transition-all
        hover:shadow-md hover:scale-105
        ${
          isRunning
            ? "border-blue-300 bg-blue-50"
            : passed
            ? "border-green-300 bg-green-50"
            : "border-red-300 bg-red-50"
        }
      `}
    >
      <div className="text-center">
        <div className="text-2xl mb-2">
          {isRunning ? "⏳" : passed ? "✅" : "❌"}
        </div>
        <div className="font-medium text-gray-900">Run {run.run_number}</div>
        <div
          className={`text-xs mt-1 ${
            isRunning
              ? "text-blue-600"
              : passed
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {isRunning ? "Running" : passed ? "Passed" : "Failed"}
        </div>
      </div>
    </Link>
  );
}
