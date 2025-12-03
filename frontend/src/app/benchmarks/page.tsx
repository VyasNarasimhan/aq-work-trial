"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Benchmark } from "@/types";

export default function BenchmarksPage() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listBenchmarks()
      .then((data) => {
        // Sort by started_at descending (most recent first)
        const sorted = [...data].sort((a, b) => {
          if (!a.started_at) return 1;
          if (!b.started_at) return -1;
          return new Date(b.started_at).getTime() - new Date(a.started_at).getTime();
        });
        setBenchmarks(sorted);
      })
      .finally(() => setLoading(false));
  }, []);

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
                href={`/benchmarks/${benchmark.id}`}
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
                        ? new Date(benchmark.started_at).toLocaleString()
                        : "Not started"}
                    </p>
                  </div>
                  <div className="text-right">
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
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
