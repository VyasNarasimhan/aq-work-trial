"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import type { Run, Episode } from "@/types";
import { getRunLogs } from "@/lib/api";

interface Props {
  run: Run;
  benchmarkId: string;
}

export function AttemptCard({ run, benchmarkId }: Props) {
  const [isTestCasesExpanded, setIsTestCasesExpanded] = useState(false);
  const [isEpisodesExpanded, setIsEpisodesExpanded] = useState(false);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [expandedEpisodeIds, setExpandedEpisodeIds] = useState<Set<number>>(new Set());
  const [loadingEpisodes, setLoadingEpisodes] = useState(false);

  const passed = run.passed;
  const testCases = run.test_cases || [];
  const passedCount = run.passed_count ?? testCases.filter((tc) => tc.status === "passed").length;
  const totalCount = run.total_count ?? testCases.length;

  // Load episodes when expanded
  useEffect(() => {
    if (isEpisodesExpanded && episodes.length === 0 && !loadingEpisodes) {
      setLoadingEpisodes(true);
      getRunLogs(benchmarkId, run.id)
        .then((logs) => {
          setEpisodes(logs.episodes || []);
          // Expand first episode by default
          if (logs.episodes && logs.episodes.length > 0) {
            setExpandedEpisodeIds(new Set([0]));
          }
        })
        .catch((err) => {
          console.error("Failed to load episodes:", err);
        })
        .finally(() => {
          setLoadingEpisodes(false);
        });
    }
  }, [isEpisodesExpanded, episodes.length, loadingEpisodes, benchmarkId, run.id]);

  const toggleEpisode = (idx: number) => {
    const newSet = new Set(expandedEpisodeIds);
    if (newSet.has(idx)) {
      newSet.delete(idx);
    } else {
      newSet.add(idx);
    }
    setExpandedEpisodeIds(newSet);
  };

  return (
    <div className={`border-l-4 ${passed ? "border-l-green-500" : "border-l-red-500"} border border-gray-200 rounded-lg bg-white overflow-hidden shadow-sm`}>
      {/* Header */}
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-xl font-semibold text-gray-900">
              Attempt {run.run_number}
            </span>
            <span
              className={`
                px-2.5 py-1 rounded text-xs font-medium
                ${
                  passed
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }
              `}
            >
              {passed ? "AGENT PASSED" : "AGENT FAILED"}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span
              className={`
                px-3 py-1 rounded-full text-sm font-medium
                ${run.status === "running" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}
              `}
            >
              {run.status === "running" ? "running..." : "completed"}
            </span>
          </div>
        </div>

        {/* Expandable sections */}
        <div className="space-y-3">
          {/* Test Case Pass Rate */}
          <div>
            <button
              className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900 w-full text-left"
              onClick={() => setIsTestCasesExpanded(!isTestCasesExpanded)}
            >
              <span className="text-gray-400">{isTestCasesExpanded ? "▼" : "▶"}</span>
              <span className="font-medium">Attempt Test Case Pass Rate</span>
              <span className="text-gray-400 text-xs">(from parser results)</span>
              <span className="ml-auto px-3 py-0.5 bg-gray-100 rounded-full text-sm font-medium">
                {passedCount}/{totalCount} passed
              </span>
            </button>

            {isTestCasesExpanded && (
              <div className="mt-3 ml-6 space-y-2">
                {testCases.length > 0 ? (
                  testCases.map((testCase, idx) => (
                    <div
                      key={testCase.name || idx}
                      className="flex items-center justify-between py-1.5"
                    >
                      <span className="text-sm text-gray-700">
                        {testCase.name}
                      </span>
                      <span
                        className={`text-sm px-3 py-0.5 rounded-full border ${
                          testCase.status === "passed"
                            ? "border-green-200 bg-white text-green-700"
                            : "border-red-200 bg-white text-red-700"
                        }`}
                      >
                        {testCase.status === "passed" ? "✓ Passed" : "✗ Failed"}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500 italic">
                    No test results available
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Episodes */}
          <div>
            <button
              className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900 w-full text-left"
              onClick={() => setIsEpisodesExpanded(!isEpisodesExpanded)}
            >
              <span className="text-gray-400">{isEpisodesExpanded ? "▼" : "▶"}</span>
              <span className="font-medium">Episodes</span>
              {(episodes.length > 0 || loadingEpisodes) && (
                <span className="text-gray-500">({loadingEpisodes ? "..." : episodes.length})</span>
              )}
            </button>

            {isEpisodesExpanded && (
              <div className="mt-3 ml-6 space-y-4">
                {loadingEpisodes ? (
                  <p className="text-sm text-gray-500">Loading episodes...</p>
                ) : episodes.length > 0 ? (
                  episodes.map((episode, idx) => (
                    <div
                      key={idx}
                      className="border border-gray-200 rounded-lg bg-white overflow-hidden"
                    >
                      <button
                        onClick={() => toggleEpisode(idx)}
                        className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-gray-50 border-b border-gray-100"
                      >
                        <span className="font-semibold text-gray-900">
                          Episode {idx}
                        </span>
                        <span className="text-gray-400">
                          {expandedEpisodeIds.has(idx) ? "▼" : "▶"}
                        </span>
                      </button>

                      {expandedEpisodeIds.has(idx) && (
                        <div className="p-4 space-y-4">
                          {/* State Analysis */}
                          {episode.state_analysis && (
                            <div>
                              <h4 className="text-sm font-semibold text-gray-900 mb-2">
                                State Analysis:
                              </h4>
                              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-700 leading-relaxed">
                                {episode.state_analysis}
                              </div>
                            </div>
                          )}

                          {/* Explanation */}
                          {episode.explanation && (
                            <div>
                              <h4 className="text-sm font-semibold text-gray-900 mb-2">
                                Explanation:
                              </h4>
                              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-gray-700 leading-relaxed">
                                {episode.explanation}
                              </div>
                            </div>
                          )}

                          {/* Commands */}
                          {episode.commands && (
                            <div>
                              <h4 className="text-sm font-semibold text-gray-900 mb-2">
                                Commands:
                              </h4>
                              <div className="bg-gray-900 rounded-lg p-4 overflow-auto max-h-96">
                                <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
                                  {episode.commands}
                                </pre>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500 italic">
                    No episodes recorded for this run
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Error message if failed */}
        {run.error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">
              <strong>Error:</strong> {run.error}
            </p>
          </div>
        )}

        {/* View Container Logs link */}
        <div className="mt-4 pt-4 border-t border-gray-100">
          <Link
            href={`/benchmarks?id=${benchmarkId}&runId=${run.id}`}
            className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
          >
            View Container Logs
          </Link>
        </div>
      </div>
    </div>
  );
}
