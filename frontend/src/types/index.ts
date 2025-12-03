export type HarnessType = "harbor" | "terminus";

export type ModelType =
  | "openrouter/anthropic/claude-sonnet-4"
  | "openrouter/anthropic/claude-sonnet-4-5"
  | "openrouter/openai/gpt-5"
  | "openrouter/google/gemini-2.0-flash";

export interface ModelOption {
  value: ModelType;
  label: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { value: "openrouter/anthropic/claude-sonnet-4", label: "Claude Sonnet 4" },
  { value: "openrouter/anthropic/claude-sonnet-4-5", label: "Claude Sonnet 4.5" },
  { value: "openrouter/openai/gpt-5", label: "GPT-5" },
  { value: "openrouter/google/gemini-2.0-flash", label: "Gemini 2.0 Flash" },
];

export const DEFAULT_MODEL: ModelType = "openrouter/openai/gpt-5";

export interface Benchmark {
  id: string;
  task_name: string;
  harness: HarnessType;
  model?: ModelType;
  status: "pending" | "running" | "completed" | "failed";
  total_runs: number;
  completed_runs: number;
  passed_runs: number;
  started_at: string | null;
  finished_at: string | null;
  error?: string | null;
}

export interface TestCase {
  name: string;
  status: "passed" | "failed" | "skipped" | "pending";
}

export interface Run {
  id: string;
  run_number: number;
  name: string;
  trial_dir?: string;
  status: "pending" | "running" | "completed" | "failed";
  passed: boolean | null;
  started_at: string | null;
  finished_at: string | null;
  error?: string | null;
  test_cases?: TestCase[];
  passed_count?: number;
  total_count?: number;
  aws_batch_job_id?: string;
  aws_batch_status?: string;
}

export interface Episode {
  state_analysis: string;
  explanation: string;
  commands: string;
}

export interface RunLogs {
  trial_log: string;
  agent_logs: Array<{
    name: string;
    content: string;
  }>;
  test_stdout: string;
  test_stderr: string;
  episodes: Episode[];
}

export interface UploadResponse {
  upload_id: string;
  task_name: string;
  task_path: string;
  detected_format: "harbor" | "terminus" | "unknown";
}

export interface CreateBenchmarkResponse {
  id: string;
  status: string;
  task_name: string;
  harness: HarnessType;
  model?: ModelType;
}
