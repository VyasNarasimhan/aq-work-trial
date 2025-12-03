"use client";

import type { RunLogs } from "@/types";

interface Props {
  logs: RunLogs;
}

export function LogViewer({ logs }: Props) {
  return (
    <div className="space-y-4">
      {/* Trial Log */}
      {logs.trial_log && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Trial Log
          </h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
              {logs.trial_log}
            </pre>
          </div>
        </div>
      )}

      {/* Agent Logs */}
      {logs.agent_logs.map((log) => (
        <div key={log.name}>
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            {log.name}
          </h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
              {log.content || "No content"}
            </pre>
          </div>
        </div>
      ))}

      {/* Test Output */}
      {logs.test_stdout && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Test Output
          </h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
              {logs.test_stdout}
            </pre>
          </div>
        </div>
      )}

      {/* Test Errors */}
      {logs.test_stderr && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Test Errors
          </h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-auto">
            <pre className="text-red-400 text-sm font-mono whitespace-pre-wrap">
              {logs.test_stderr}
            </pre>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!logs.trial_log && logs.agent_logs.length === 0 && !logs.test_stdout && !logs.test_stderr && (
        <div className="text-center py-8 text-gray-500">
          No logs available for this run
        </div>
      )}
    </div>
  );
}
