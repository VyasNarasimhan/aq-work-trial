import Link from "next/link";
import { DropZone } from "@/components/upload/DropZone";

export default function HomePage() {
  return (
    <main className="min-h-screen py-12">
      <div className="max-w-2xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Benchmark Runner
          </h1>
          <p className="text-gray-600 text-lg">
            Upload a task zip file to run 10 AI agent benchmark trials
          </p>
        </div>

        <DropZone />

        <div className="mt-8 text-center">
          <Link
            href="/benchmarks"
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            View previous benchmarks →
          </Link>
        </div>

        <div className="mt-12 bg-white rounded-lg shadow p-6">
          <h2 className="font-semibold text-gray-900 mb-3">
            How it works
          </h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-600">
            <li>Upload a Harbor (task.toml) or Terminus (task.yaml) task zip file</li>
            <li>Select the harness to use for execution</li>
            <li>The system runs 10 independent agent trials</li>
            <li>View real-time progress as trials complete</li>
            <li>See pass/fail results and detailed logs for each run</li>
          </ol>
        </div>
      </div>
    </main>
  );
}
