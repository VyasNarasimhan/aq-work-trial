import asyncio
import fcntl
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if AWS Batch is enabled
USE_AWS_BATCH = os.environ.get("USE_AWS_BATCH", "false").lower() == "true"

# Check if Daytona worker mode is enabled (tasks are queued, worker picks them up)
USE_DAYTONA_WORKER = os.environ.get("USE_DAYTONA_WORKER", "false").lower() == "true"


class BenchmarkManager:
    """Manages benchmark execution and state persistence."""

    def __init__(self, jobs_dir: Path, uploads_dir: Path, data_dir: Path):
        self.jobs_dir = jobs_dir
        self.uploads_dir = uploads_dir
        self.data_dir = data_dir
        self.benchmarks_file = data_dir / "benchmarks.json"
        self.benchmarks: dict[str, dict] = {}
        self.num_runs = 10  # Default to 10 runs for scaling
        self._load_benchmarks()

        # Initialize AWS clients if using AWS Batch
        self.use_aws_batch = USE_AWS_BATCH
        self.batch_client = None
        self.s3_client = None

        if self.use_aws_batch or USE_DAYTONA_WORKER:
            try:
                from app.aws import S3Client
                self.s3_client = S3Client()
                if self.use_aws_batch:
                    from app.aws import BatchClient
                    self.batch_client = BatchClient()
            except Exception as e:
                print(f"Warning: Failed to initialize AWS clients: {e}")
                if self.use_aws_batch:
                    self.use_aws_batch = False

    def _load_benchmarks(self):
        """Load benchmarks from JSON file with file locking for concurrent access."""
        if self.benchmarks_file.exists():
            try:
                with open(self.benchmarks_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                    try:
                        data = json.load(f)
                        self.benchmarks = {b["id"]: b for b in data.get("benchmarks", [])}
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (json.JSONDecodeError, KeyError):
                self.benchmarks = {}

    def _save_benchmarks(self):
        """Save benchmarks to JSON file with file locking for concurrent access."""
        data = {"benchmarks": list(self.benchmarks.values())}

        # Ensure file exists before opening in r+ mode
        if not self.benchmarks_file.exists():
            self.benchmarks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.benchmarks_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(self.benchmarks_file, "r+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                try:
                    f.seek(0)
                    json.dump(data, f, indent=2, default=str)
                    f.truncate()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Sync to S3 for worker mode
        if USE_DAYTONA_WORKER and self.s3_client:
            try:
                self.s3_client.upload_json("state/benchmarks.json", data)
            except Exception as e:
                print(f"Warning: Failed to sync benchmarks to S3: {e}")

    def start_benchmark(self, benchmark_id: str, task_path: Path, harness: str = "harbor", model: str = "openrouter/openai/gpt-5") -> dict:
        """Start a new benchmark run with multiple trials.

        Args:
            benchmark_id: Unique identifier for this benchmark.
            task_path: Path to the task directory.
            harness: Harness to use ('harbor' or 'terminus').
            model: Model to use for the benchmark.
        """
        logger.info(f"Starting benchmark {benchmark_id} with harness={harness}, model={model}")

        benchmark = {
            "id": benchmark_id,
            "task_name": task_path.name,
            "task_path": str(task_path),
            "harness": harness,
            "model": model,
            "status": "pending",
            "job_dirs": [],
            "batch_job_ids": [],  # AWS Batch job IDs
            "s3_task_path": None,  # S3 path for task files
            "total_runs": self.num_runs,
            "completed_runs": 0,
            "passed_runs": 0,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "error": None,
            "use_aws_batch": self.use_aws_batch,
        }

        self.benchmarks[benchmark_id] = benchmark
        self._save_benchmarks()

        if USE_DAYTONA_WORKER:
            # Worker mode: upload task to S3 and save as pending
            # Worker will download from S3 and execute
            if self.s3_client:
                try:
                    s3_uri = self.s3_client.upload_task(benchmark_id, task_path)
                    benchmark["s3_task_path"] = s3_uri
                    self._save_benchmarks()
                except Exception as e:
                    print(f"Warning: Failed to upload task to S3: {e}")
            return benchmark

        if self.use_aws_batch:
            # Start benchmark using AWS Batch
            self._start_aws_batch_benchmark(benchmark_id, task_path, harness, model)
        else:
            # Start benchmark in background thread (original behavior)
            thread = threading.Thread(
                target=self._run_in_thread,
                args=(benchmark_id, task_path, harness, model),
                daemon=True,
            )
            thread.start()

        return benchmark

    def _start_aws_batch_benchmark(self, benchmark_id: str, task_path: Path, harness: str = "harbor", model: str = "openrouter/openai/gpt-5"):
        """Start benchmark using AWS Batch.

        Submits a single AWS Batch job that runs all trials concurrently using
        the specified harness's concurrency feature.
        """
        benchmark = self.benchmarks[benchmark_id]

        logger.info(f"Starting AWS Batch benchmark {benchmark_id} with harness={harness}, model={model}")

        try:
            # Upload task to S3
            upload_id = benchmark_id
            s3_task_path = self.s3_client.upload_task(upload_id, task_path)
            benchmark["s3_task_path"] = s3_task_path

            # Submit job to AWS Batch
            # Terminus: all trials go under run-1/, so total_runs=1
            # Harbor: each trial in run-N/, so total_runs=n_attempts (default)
            job_id = self.batch_client.submit_job(
                benchmark_id=benchmark_id,
                s3_task_path=s3_task_path,
                n_attempts=self.num_runs,
                total_runs=1 if harness == "terminus" else None,
                harness=harness,
                model_name=model,
            )

            # Store single job ID (keep batch_job_ids empty for backward compat)
            benchmark["batch_job_id"] = job_id
            benchmark["batch_job_ids"] = []  # Empty for backward compatibility
            benchmark["status"] = "running"
            self._save_benchmarks()

            # Start background thread to poll for completion
            thread = threading.Thread(
                target=self._poll_aws_batch_jobs,
                args=(benchmark_id,),
                daemon=True,
            )
            thread.start()

        except Exception as e:
            benchmark["status"] = "failed"
            benchmark["error"] = str(e)
            benchmark["finished_at"] = datetime.now().isoformat()
            self._save_benchmarks()
            raise

    def _poll_aws_batch_jobs(self, benchmark_id: str):
        """Poll single AWS Batch job and check S3 for individual run statuses.

        Routes to terminus-specific or harbor polling logic based on harness type.
        """
        import time

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return

        harness = benchmark.get("harness", "harbor")

        # Use terminus-specific polling for terminus benchmarks
        if harness == "terminus":
            self._poll_terminus_job(benchmark_id)
        else:
            self._poll_harbor_job(benchmark_id)

    def _poll_terminus_job(self, benchmark_id: str):
        """Poll terminus job - only checks run-1/results.json for all trial statuses."""
        import time

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return

        job_id = benchmark.get("batch_job_id")
        if not job_id:
            return

        # Poll until job completes or fails
        while True:
            job_status = self.batch_client.get_job_status(job_id)
            if not job_status:
                time.sleep(10)
                continue

            aws_status = job_status.get("status", "UNKNOWN")

            # For terminus, check results.json which has all trial results
            terminus_results = self.s3_client.get_terminus_results(benchmark_id)

            if terminus_results and isinstance(terminus_results, list):
                completed_count = len(terminus_results)
                passed_count = sum(1 for r in terminus_results if r.get("is_resolved", False))
                benchmark["completed_runs"] = completed_count
                benchmark["passed_runs"] = passed_count

            if aws_status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
                self._save_benchmarks()
                time.sleep(10)
                continue

            elif aws_status == "SUCCEEDED":
                # Final count from results.json
                terminus_results = self.s3_client.get_terminus_results(benchmark_id)
                if terminus_results and isinstance(terminus_results, list):
                    benchmark["completed_runs"] = len(terminus_results)
                    benchmark["passed_runs"] = sum(1 for r in terminus_results if r.get("is_resolved", False))

                benchmark["status"] = "completed"
                benchmark["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
                break

            elif aws_status == "FAILED":
                benchmark["status"] = "failed"
                benchmark["error"] = job_status.get("statusReason", "Batch job failed")
                benchmark["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
                break

            else:
                self._save_benchmarks()
                time.sleep(10)

    def _poll_harbor_job(self, benchmark_id: str):
        """Poll harbor job - checks run-{N}/status.json for each run."""
        import time

        # Get initial job_id (won't change during polling)
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return

        job_id = benchmark.get("batch_job_id")
        if not job_id:
            return

        total_runs = benchmark.get("total_runs", 10)

        # Poll until job completes or fails
        while True:
            # Get status of the single job
            job_status = self.batch_client.get_job_status(job_id)
            if not job_status:
                time.sleep(10)
                continue

            aws_status = job_status.get("status", "UNKNOWN")

            # Check S3 for individual run statuses regardless of job state
            completed_count = 0
            passed_count = 0

            for run_number in range(1, total_runs + 1):
                run_status = self.s3_client.get_run_status(benchmark_id, run_number)
                if run_status and run_status.get("status") == "completed":
                    completed_count += 1
                    if run_status.get("passed"):
                        passed_count += 1

            # Update benchmark counts - use self.benchmarks[benchmark_id] directly
            # to avoid stale reference after _load_benchmarks() is called elsewhere
            if benchmark_id in self.benchmarks:
                self.benchmarks[benchmark_id]["completed_runs"] = completed_count
                self.benchmarks[benchmark_id]["passed_runs"] = passed_count

            # Check if all runs are completed based on S3 data
            if completed_count >= total_runs:
                # All runs completed - mark benchmark as done
                if benchmark_id in self.benchmarks:
                    self.benchmarks[benchmark_id]["status"] = "completed"
                    self.benchmarks[benchmark_id]["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
                break

            if aws_status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
                # Job still running
                self._save_benchmarks()
                time.sleep(10)
                continue

            elif aws_status == "SUCCEEDED":
                # Job completed - do final count from S3
                completed_count = 0
                passed_count = 0

                for run_number in range(1, total_runs + 1):
                    run_status = self.s3_client.get_run_status(benchmark_id, run_number)
                    if run_status:
                        completed_count += 1
                        if run_status.get("passed"):
                            passed_count += 1

                if benchmark_id in self.benchmarks:
                    self.benchmarks[benchmark_id]["completed_runs"] = completed_count
                    self.benchmarks[benchmark_id]["passed_runs"] = passed_count
                    self.benchmarks[benchmark_id]["status"] = "completed"
                    self.benchmarks[benchmark_id]["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
                break

            elif aws_status == "FAILED":
                if benchmark_id in self.benchmarks:
                    self.benchmarks[benchmark_id]["status"] = "failed"
                    self.benchmarks[benchmark_id]["error"] = job_status.get("statusReason", "Batch job failed")
                    self.benchmarks[benchmark_id]["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
                break

            else:
                # Unknown status, keep polling
                self._save_benchmarks()
                time.sleep(10)

    def _run_in_thread(self, benchmark_id: str, task_path: Path, harness: str = "harbor", model: str = "openrouter/openai/gpt-5"):
        """Run benchmark in a background thread."""
        logger.info(f"Running benchmark {benchmark_id} in thread with harness={harness}, model={model}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_benchmark(benchmark_id, task_path, harness, model))
        except Exception as e:
            benchmark = self.benchmarks.get(benchmark_id)
            if benchmark:
                benchmark["status"] = "failed"
                benchmark["error"] = str(e)
                benchmark["finished_at"] = datetime.now().isoformat()
                self._save_benchmarks()
        finally:
            loop.close()

    async def _run_benchmark(self, benchmark_id: str, task_path: Path, harness: str = "harbor", model: str = "openrouter/openai/gpt-5"):
        """Execute the benchmark - 10 independent runs in parallel.

        Note: Local execution currently only supports Harbor. For Terminus,
        use AWS Batch mode which invokes the `tb run` CLI.
        """
        logger.info(f"Executing benchmark {benchmark_id} with harness={harness}, model={model}")

        if harness == "terminus":
            raise NotImplementedError(
                "Terminus harness is only supported in AWS Batch mode. "
                "Set USE_AWS_BATCH=true to run Terminus benchmarks."
            )

        from harbor.job import Job
        from harbor.models.job.config import (
            JobConfig,
            AgentConfig,
            TaskConfig,
            OrchestratorConfig,
        )
        from harbor.models.orchestrator_type import OrchestratorType

        benchmark = self.benchmarks[benchmark_id]
        benchmark["status"] = "running"
        self._save_benchmarks()

        async def run_single_job(run_num: int):
            """Run a single job and return the result."""
            job_name = f"benchmark-{benchmark_id[:8]}-run-{run_num + 1}"

            logger.info(f"Starting job {job_name} with model={model}")

            config = JobConfig(
                job_name=job_name,
                jobs_dir=self.jobs_dir,
                n_attempts=1,
                agents=[
                    AgentConfig(
                        name="terminus-2",
                        model_name=model,
                    )
                ],
                tasks=[TaskConfig(path=task_path)],
                orchestrator=OrchestratorConfig(
                    type=OrchestratorType.LOCAL,
                    n_concurrent_trials=1,
                ),
            )

            job = Job(config)
            result = await job.run()
            return run_num, job_name, result

        try:
            # Create all 10 job tasks
            tasks = [run_single_job(i) for i in range(self.num_runs)]

            # Pre-populate job_dirs in order
            for i in range(self.num_runs):
                job_name = f"benchmark-{benchmark_id[:8]}-run-{i + 1}"
                benchmark["job_dirs"].append(str(self.jobs_dir / job_name))
            self._save_benchmarks()

            # Run all jobs in parallel, updating progress as each completes
            errors = []
            completed_count = 0
            passed_count = 0

            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    run_num, job_name, job_result = result
                    job_dir = self.jobs_dir / job_name

                    # Check if this run passed
                    passed = self._check_job_passed(job_dir)

                    completed_count += 1
                    if passed:
                        passed_count += 1

                    # Update benchmark progress
                    benchmark["completed_runs"] = completed_count
                    benchmark["passed_runs"] = passed_count
                    self._save_benchmarks()

                except Exception as e:
                    errors.append(str(e))
                    completed_count += 1
                    benchmark["completed_runs"] = completed_count
                    self._save_benchmarks()

            benchmark["status"] = "completed"
            benchmark["finished_at"] = datetime.now().isoformat()

            if errors:
                benchmark["error"] = f"{len(errors)} runs failed: {errors[0]}"

        except Exception as e:
            benchmark["status"] = "failed"
            benchmark["error"] = str(e)
            benchmark["finished_at"] = datetime.now().isoformat()
            raise

        finally:
            self._save_benchmarks()

    def _check_job_passed(self, job_dir: Path) -> bool:
        """Check if a job's trial passed (has positive reward)."""
        if not job_dir.exists():
            return False

        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            result_file = trial_dir / "verifier/ctrf.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        result = json.load(f)

                    verifier_result = result.get("verifier_result")
                    if verifier_result and verifier_result.get("rewards"):
                        rewards = verifier_result["rewards"]
                        if any(v > 0 for v in rewards.values() if isinstance(v, (int, float))):
                            return True
                except (json.JSONDecodeError, KeyError):
                    pass

        return False

    def get_benchmark(self, benchmark_id: str) -> dict | None:
        """Get benchmark status."""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return None

        # Return a copy without internal fields
        return {
            "id": benchmark["id"],
            "task_name": benchmark["task_name"],
            "harness": benchmark.get("harness", "harbor"),
            "model": benchmark.get("model", "openrouter/openai/gpt-5"),
            "status": benchmark["status"],
            "total_runs": benchmark["total_runs"],
            "completed_runs": benchmark["completed_runs"],
            "passed_runs": benchmark["passed_runs"],
            "started_at": benchmark["started_at"],
            "finished_at": benchmark["finished_at"],
            "error": benchmark.get("error"),
        }

    def list_benchmarks(self) -> list[dict]:
        """List all benchmarks."""
        return [self._format_benchmark(b) for b in self.benchmarks.values() if b]

    def _format_benchmark(self, benchmark: dict) -> dict:
        """Format benchmark for API response."""
        return {
            "id": benchmark["id"],
            "task_name": benchmark["task_name"],
            "harness": benchmark.get("harness", "harbor"),
            "model": benchmark.get("model", "openrouter/openai/gpt-5"),
            "status": benchmark["status"],
            "total_runs": benchmark["total_runs"],
            "completed_runs": benchmark["completed_runs"],
            "passed_runs": benchmark["passed_runs"],
            "started_at": benchmark["started_at"],
            "finished_at": benchmark["finished_at"],
            "error": benchmark.get("error"),
        }

    def get_runs(self, benchmark_id: str) -> list[dict] | None:
        """Get all runs for a benchmark."""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return None

        # Check if this benchmark uses AWS Batch
        if benchmark.get("use_aws_batch") and self.s3_client:
            return self._get_aws_batch_runs(benchmark)

        # Original local file-based approach
        runs = []
        for idx, job_dir_str in enumerate(benchmark.get("job_dirs", [])):
            job_dir = Path(job_dir_str)
            run_info = self._get_run_info(job_dir, idx + 1)
            if run_info:
                runs.append(run_info)

        return runs

    def _get_aws_batch_runs(self, benchmark: dict) -> list[dict]:
        """Get run information for AWS Batch-based benchmark.

        Routes to terminus-specific or harbor logic based on harness type.
        """
        harness = benchmark.get("harness", "harbor")

        # Use terminus-specific parsing for terminus benchmarks
        if harness == "terminus":
            return self._get_terminus_runs(benchmark)

        # Harbor: Queries S3 for each run number (1 to total_runs)
        return self._get_harbor_runs(benchmark)

    def _extract_run_number_from_task_id(self, task_id: str) -> int:
        """Extract run number from task_id like 'broken-postgres-backup-restore-7' → 7.

        The task_id ends with -{N} where N is the run number.
        """
        import re

        if not task_id:
            return 999  # Sort to end if no task_id

        # Match the last number after a hyphen: "task-name-7" → 7
        match = re.search(r'-(\d+)$', task_id)
        if match:
            return int(match.group(1))
        return 999

    def _get_terminus_runs(self, benchmark: dict) -> list[dict]:
        """Get run information for terminus benchmark from S3.

        Terminus stores all trials under run-1/task/task.X-of-10.../
        and results in run-1/results.json.
        """
        benchmark_id = benchmark["id"]
        job_id = benchmark.get("batch_job_id")
        total_runs = benchmark.get("total_runs", 10)
        runs = []

        # Get AWS Batch job status
        batch_status = None
        aws_status = "UNKNOWN"
        if job_id and self.batch_client:
            batch_status = self.batch_client.get_job_status(job_id)
            if batch_status:
                aws_status = batch_status.get("status", "UNKNOWN")

        # Try to get terminus results.json (contains all trial results)
        terminus_results = self.s3_client.get_terminus_results(benchmark_id)

        # Handle wrapped format: {"id": ..., "results": [...]}
        # Terminal-Bench outputs this format instead of a direct array
        if terminus_results and isinstance(terminus_results, dict):
            terminus_results = terminus_results.get("results", [])

        if terminus_results and isinstance(terminus_results, list):
            # Parse each trial result from results.json
            for trial_result in terminus_results:
                # Extract run number from task_id (e.g., "broken-postgres-backup-restore-7" → 7)
                # The results array is NOT sorted - trials complete in arbitrary order
                task_id = trial_result.get("task_id", "")
                run_number = self._extract_run_number_from_task_id(task_id)

                is_resolved = trial_result.get("is_resolved", False)
                trial_name = trial_result.get("trial_name", f"task.{run_number}-of-{total_runs}")

                # Parse test cases from parser_results
                test_cases = []
                parser_results = trial_result.get("parser_results", {})
                if isinstance(parser_results, dict):
                    for test_name, result in parser_results.items():
                        status = "passed" if result == "passed" else "failed"
                        test_cases.append({"name": test_name, "status": status})

                passed_count = sum(1 for tc in test_cases if tc.get("status") == "passed")
                total_count = len(test_cases)

                # Handle field name differences between formats:
                # - Terminal-Bench uses "id" instead of "trial_id"
                # - Terminal-Bench uses "trial_started_at"/"trial_ended_at" instead of "started_at"/"finished_at"
                trial_id = trial_result.get("trial_id") or trial_result.get("id", f"run-{run_number}")
                started_at = trial_result.get("started_at") or trial_result.get("trial_started_at")
                finished_at = trial_result.get("finished_at") or trial_result.get("trial_ended_at")

                run_info = {
                    "id": trial_id,
                    "run_number": run_number,
                    "name": trial_name,
                    "task_id": task_id,  # Store for log lookup
                    "status": "completed",
                    "passed": is_resolved,
                    "test_cases": test_cases,
                    "passed_count": passed_count,
                    "total_count": total_count,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": None,
                    "aws_batch_job_id": job_id,
                    "aws_batch_status": aws_status,
                }
                runs.append(run_info)

            # Sort runs by run_number since results.json is not in order
            runs.sort(key=lambda r: r["run_number"])
        else:
            # No results yet - check if job is still running
            # Also try to get trial names from S3 to show progress
            trial_names = self.s3_client.list_terminus_trial_names(benchmark_id)

            for run_number in range(1, total_runs + 1):
                if aws_status in ["SUBMITTED", "PENDING", "RUNNABLE"]:
                    status = "pending"
                elif aws_status in ["STARTING", "RUNNING"]:
                    status = "running"
                elif aws_status == "SUCCEEDED":
                    status = "completed"
                elif aws_status == "FAILED":
                    status = "failed"
                else:
                    status = "pending"

                run_info = {
                    "id": f"run-{run_number}",
                    "run_number": run_number,
                    "name": f"run-{run_number}",
                    "status": status,
                    "passed": None,
                    "test_cases": [],
                    "passed_count": 0,
                    "total_count": 0,
                    "started_at": batch_status.get("startedAt") if batch_status else None,
                    "finished_at": batch_status.get("stoppedAt") if batch_status else None,
                    "error": batch_status.get("statusReason") if batch_status and aws_status == "FAILED" else None,
                    "aws_batch_job_id": job_id,
                    "aws_batch_status": aws_status,
                }
                runs.append(run_info)

        return runs

    def _get_harbor_runs(self, benchmark: dict) -> list[dict]:
        """Get run information for harbor benchmark from S3.

        Queries S3 for each run number (1 to total_runs) to get individual run statuses.
        """
        benchmark_id = benchmark["id"]
        job_id = benchmark.get("batch_job_id")
        total_runs = benchmark.get("total_runs", 10)
        runs = []

        # Get single AWS Batch job status
        batch_status = None
        aws_status = "UNKNOWN"
        if job_id and self.batch_client:
            batch_status = self.batch_client.get_job_status(job_id)
            if batch_status:
                aws_status = batch_status.get("status", "UNKNOWN")

        for run_number in range(1, total_runs + 1):
            # Try to get detailed status from S3
            s3_status = self.s3_client.get_run_status(benchmark_id, run_number)

            if s3_status:
                # Use S3 status if available (more detailed)
                status = s3_status.get("status", "running")

                # Fetch test cases from S3 for completed runs
                test_cases = []
                passed_count = 0
                total_count = 0

                if status == "completed":
                    test_cases = self._get_test_cases_from_s3(benchmark_id, run_number)
                    passed_count = sum(1 for tc in test_cases if tc.get("status") == "passed")
                    total_count = len(test_cases)

                # Calculate passed from test cases if available, otherwise use S3 status
                if total_count > 0:
                    passed = passed_count == total_count
                else:
                    passed = s3_status.get("passed")

                run_info = {
                    "id": s3_status.get("run_id", f"run-{run_number}"),
                    "run_number": run_number,
                    "name": f"run-{run_number}",
                    "status": status,
                    "passed": passed,
                    "test_cases": test_cases,
                    "passed_count": passed_count,
                    "total_count": total_count,
                    "started_at": s3_status.get("started_at"),
                    "finished_at": s3_status.get("finished_at"),
                    "error": s3_status.get("error"),
                    "aws_batch_job_id": job_id,
                    "aws_batch_status": aws_status,
                }
            else:
                # No S3 status yet - derive from batch job status
                if aws_status in ["SUBMITTED", "PENDING", "RUNNABLE"]:
                    status = "pending"
                elif aws_status in ["STARTING", "RUNNING"]:
                    status = "running"
                elif aws_status == "SUCCEEDED":
                    # Job succeeded but no S3 status - might still be uploading
                    status = "completed"
                elif aws_status == "FAILED":
                    status = "failed"
                else:
                    status = "pending"

                run_info = {
                    "id": f"run-{run_number}",
                    "run_number": run_number,
                    "name": f"run-{run_number}",
                    "status": status,
                    "passed": None,
                    "test_cases": [],
                    "passed_count": 0,
                    "total_count": 0,
                    "started_at": batch_status.get("startedAt") if batch_status else None,
                    "finished_at": batch_status.get("stoppedAt") if batch_status else None,
                    "error": batch_status.get("statusReason") if batch_status else None,
                    "aws_batch_job_id": job_id,
                    "aws_batch_status": aws_status,
                }

            runs.append(run_info)

        return runs

    def _get_test_cases_from_s3(self, benchmark_id: str, run_number: int) -> list[dict]:
        """Fetch test case results from S3 ctrf.json, falling back to test-stdout.txt."""
        import re

        files = self.s3_client.list_run_files(benchmark_id, run_number)

        # Try ctrf.json first
        ctrf_path = None
        stdout_path = None
        for f in files:
            if f.endswith("ctrf.json"):
                ctrf_path = f
            elif f.endswith("test-stdout.txt"):
                stdout_path = f

        # Try parsing ctrf.json
        if ctrf_path:
            try:
                content = self.s3_client.get_run_result(benchmark_id, run_number, ctrf_path)
                if content:
                    data = json.loads(content.decode("utf-8"))
                    tests = data.get("results", {}).get("tests", [])
                    if tests:  # Only return if we actually have tests
                        test_results = []
                        for test in tests:
                            raw_name = test.get("name", "")
                            # Handle pytest format (has ::) - extract last part
                            if "::" in raw_name:
                                test_name = raw_name.split("::")[-1]
                            else:
                                # Go/other formats - use name as-is
                                test_name = raw_name
                            test_results.append({
                                "name": test_name,
                                "status": test.get("status", "unknown"),
                            })
                        return test_results
            except (json.JSONDecodeError, KeyError, Exception):
                pass

        # Fall back to parsing test-stdout.txt
        if stdout_path:
            try:
                content = self.s3_client.get_run_result(benchmark_id, run_number, stdout_path)
                if content:
                    stdout_text = content.decode("utf-8")
                    test_results = []

                    for line in stdout_text.splitlines():
                        line = line.strip()

                        # Pytest format: PASSED/FAILED
                        if line.startswith("PASSED ") or line.startswith("FAILED "):
                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                status = parts[0].lower()
                                test_path = parts[1].strip()
                                if "::" in test_path:
                                    test_name = test_path.split("::")[-1]
                                else:
                                    test_name = test_path
                                if " " in test_name:
                                    test_name = test_name.split()[0]
                                test_results.append({
                                    "name": test_name,
                                    "status": status,
                                })

                        # Go test format: --- PASS: TestName (0.00s)
                        elif line.startswith("--- PASS:") or line.startswith("--- FAIL:"):
                            go_match = re.match(r'--- (PASS|FAIL): (\S+)', line)
                            if go_match:
                                status = "passed" if go_match.group(1) == "PASS" else "failed"
                                test_name = go_match.group(2)
                                test_results.append({
                                    "name": test_name,
                                    "status": status,
                                })

                        # Go test JSON format: {"Action":"pass","Test":"TestName"...}
                        elif line.startswith("{"):
                            try:
                                event = json.loads(line)
                                action = event.get("Action")
                                test_name = event.get("Test")
                                if test_name and action in ("pass", "fail"):
                                    status = "passed" if action == "pass" else "failed"
                                    test_results.append({
                                        "name": test_name,
                                        "status": status,
                                    })
                            except json.JSONDecodeError:
                                pass

                        # Formatted Go test output: ✓ PASS: TestName or ✗ FAIL: TestName
                        elif line.startswith("✓ PASS:") or line.startswith("✗ FAIL:"):
                            formatted_match = re.match(r'[✓✗] (PASS|FAIL): (\S+)', line)
                            if formatted_match:
                                status = "passed" if formatted_match.group(1) == "PASS" else "failed"
                                test_name = formatted_match.group(2)
                                test_results.append({
                                    "name": test_name,
                                    "status": status,
                                })

                    return test_results
            except Exception:
                pass

        return []

    def _get_test_case_results(self, job_dir: Path) -> list[dict]:
        if not job_dir.exists():
            return []

        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            result_file = trial_dir / "verifier/ctrf.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        result = json.load(f)

                    tests = result.get("results", {}).get("tests", [])
                    test_results = []
                    for test in tests:
                        raw_name = test.get('name', '')
                        # Handle pytest format (has ::) - extract last part
                        if '::' in raw_name:
                            test_name = raw_name.split('::')[-1]
                        else:
                            # Go/other formats - use name as-is
                            test_name = raw_name
                        test_status = test.get('status', 'unknown')
                        test_results.append({
                            "name": test_name,
                            "status": test_status,
                        })
                    return test_results
                except (json.JSONDecodeError, KeyError):
                    pass

        return []

    def _get_run_info(self, job_dir: Path, run_number: int) -> dict | None:
        """Get info for a single run from its job directory."""
        if not job_dir.exists():
            return None

        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            result_file = trial_dir / "result.json"
            if not result_file.exists():
                continue

            try:
                with open(result_file) as f:
                    result = json.load(f)

                # Check for error
                error = None
                exception_info = result.get("exception_info")
                if exception_info:
                    error = exception_info.get("exception_message") or exception_info.get("exception_type")

                # Get test cases from ctrf.json first, fall back to test-stdout.txt
                test_cases = self._get_test_case_results(job_dir)
                if not test_cases:
                    # Fall back to parsing test-stdout.txt
                    test_cases = self._parse_test_output(trial_dir)

                # Calculate passed/failed counts
                passed_count = sum(1 for tc in test_cases if tc.get("status") == "passed")
                total_count = len(test_cases)

                # Overall pass if all tests passed
                passed = passed_count == total_count and total_count > 0

                return {
                    "id": result.get("id", str(run_number)),
                    "run_number": run_number,
                    "name": result.get("trial_name", f"run-{run_number}"),
                    "trial_dir": str(trial_dir),
                    "status": "completed" if result.get("finished_at") else "running",
                    "passed": passed,
                    "test_cases": test_cases,
                    "passed_count": passed_count,
                    "total_count": total_count,
                    "started_at": result.get("started_at"),
                    "finished_at": result.get("finished_at"),
                    "error": error,
                }
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def _parse_test_output(self, trial_dir: Path) -> list[dict]:
        """Parse test results from test-stdout.txt."""
        import re

        test_cases = []
        stdout_file = trial_dir / "verifier" / "test-stdout.txt"

        if not stdout_file.exists():
            return test_cases

        try:
            content = stdout_file.read_text()

            # Parse test output line by line
            # Supports both pytest and Go test formats
            for line in content.splitlines():
                line = line.strip()

                # Pytest format: PASSED ../tests/test_file.py::test_name
                # or: FAILED ../tests/test_file.py::test_name[param]
                if line.startswith("PASSED ") or line.startswith("FAILED "):
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        status = parts[0].lower()
                        test_path = parts[1].strip()

                        # Extract test case name from path like ../tests/test_file.py::test_name
                        if "::" in test_path:
                            test_name = test_path.split("::")[-1]
                        else:
                            test_name = test_path

                        # If test name has spaces, take just the first part
                        if " " in test_name:
                            test_name = test_name.split()[0]

                        test_cases.append({
                            "name": test_name,
                            "passed": status == "passed",
                            "status": status,
                        })

                # Go test format: --- PASS: TestName (0.00s) or --- FAIL: TestName (0.00s)
                elif line.startswith("--- PASS:") or line.startswith("--- FAIL:"):
                    go_match = re.match(r'--- (PASS|FAIL): (\S+)', line)
                    if go_match:
                        status = "passed" if go_match.group(1) == "PASS" else "failed"
                        test_name = go_match.group(2)
                        test_cases.append({
                            "name": test_name,
                            "passed": status == "passed",
                            "status": status,
                        })

            # Fallback: try parsing summary lines like "9 passed in 45.67s"
            if not test_cases:
                summary_match = re.search(r'(\d+)\s+passed', content)
                failed_match = re.search(r'(\d+)\s+failed', content)

                if summary_match:
                    passed_count = int(summary_match.group(1))
                    for i in range(passed_count):
                        test_cases.append({
                            "name": f"test_{i+1}",
                            "passed": True,
                            "status": "passed",
                        })

                if failed_match:
                    failed_count = int(failed_match.group(1))
                    for i in range(failed_count):
                        test_cases.append({
                            "name": f"test_failed_{i+1}",
                            "passed": False,
                            "status": "failed",
                        })

        except Exception:
            pass

        return test_cases

    def get_run_logs(self, benchmark_id: str, run_id: str) -> dict | None:
        """Get logs for a specific run."""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return None

        # Check if this is an AWS Batch benchmark
        if benchmark.get("use_aws_batch") and self.s3_client:
            return self._get_aws_batch_run_logs(benchmark, run_id)

        # Find the trial directory for this run (local mode)
        for job_dir_str in benchmark.get("job_dirs", []):
            job_dir = Path(job_dir_str)
            if not job_dir.exists():
                continue

            for trial_dir in job_dir.iterdir():
                if not trial_dir.is_dir():
                    continue

                result_file = trial_dir / "result.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        result = json.load(f)

                    if result.get("id") == run_id:
                        return self._read_logs(trial_dir)
                except (json.JSONDecodeError, KeyError):
                    pass

        return None

    def _get_aws_batch_run_logs(self, benchmark: dict, run_id: str) -> dict | None:
        """Get logs for an AWS Batch run from S3.

        Routes to terminus-specific or harbor logic based on harness type.
        """
        harness = benchmark.get("harness", "harbor")

        if harness == "terminus":
            return self._get_terminus_run_logs(benchmark, run_id)
        else:
            return self._get_harbor_run_logs(benchmark, run_id)

    def _get_terminus_run_logs(self, benchmark: dict, run_id: str) -> dict | None:
        """Get logs for a terminus run from S3.

        Terminus stores logs at run-1/{task_id}/{trial_name}/agent-logs/episode-N/.
        """
        import re
        import tempfile

        benchmark_id = benchmark["id"]
        total_runs = benchmark.get("total_runs", 10)

        # Initialize empty logs
        logs = {
            "trial_log": "",
            "agent_logs": [],
            "test_stdout": "",
            "test_stderr": "",
            "episodes": [],
        }

        # Find the task_id for this run
        # The run_id might be "run-N" format or a UUID
        target_task_id = None

        # Method 1: run_id is "run-N" format - derive task_id from run number
        match = re.match(r"run-(\d+)", run_id)
        if match:
            run_number = int(match.group(1))
            # Need to find task_id from results.json that corresponds to this run number
            terminus_results = self.s3_client.get_terminus_results(benchmark_id)
            if terminus_results and isinstance(terminus_results, dict):
                terminus_results = terminus_results.get("results", [])
            if terminus_results and isinstance(terminus_results, list):
                for result in terminus_results:
                    task_id = result.get("task_id", "")
                    if self._extract_run_number_from_task_id(task_id) == run_number:
                        target_task_id = task_id
                        break

        # Method 2: run_id is a UUID - find the matching result and get its task_id
        if target_task_id is None:
            terminus_results = self.s3_client.get_terminus_results(benchmark_id)
            if terminus_results and isinstance(terminus_results, dict):
                terminus_results = terminus_results.get("results", [])
            if terminus_results and isinstance(terminus_results, list):
                for result in terminus_results:
                    if result.get("trial_id") == run_id or result.get("id") == run_id:
                        target_task_id = result.get("task_id")
                        break

        if target_task_id is None:
            return logs

        # Get list of terminus trials and find the one matching our task_id
        # trial_paths are "{task_id}/{trial_name}" format
        trial_paths = self.s3_client.list_terminus_trial_names(benchmark_id)
        trial_path = None
        for tp in trial_paths:
            # tp is like "broken-postgres-backup-restore-7/broken-postgres-backup-restore-7.1-of-1.timestamp"
            if tp.startswith(target_task_id + "/"):
                trial_path = tp
                break

        if trial_path is None:
            return logs

        # Download trial files to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.s3_client.download_terminus_trial(benchmark_id, trial_path, temp_path)

            # Read agent logs from agent-logs/episode-N directories
            agent_logs_dir = temp_path / "agent-logs"
            if agent_logs_dir.exists():
                # Find all episode directories
                episode_dirs = []
                for item in agent_logs_dir.iterdir():
                    if item.is_dir() and item.name.startswith("episode-"):
                        try:
                            episode_num = int(item.name.split("-")[1])
                            episode_dirs.append((episode_num, item))
                        except (IndexError, ValueError):
                            continue

                episode_dirs.sort(key=lambda x: x[0])

                for episode_num, episode_dir in episode_dirs:
                    # Look for response.json or response.txt
                    response_file = episode_dir / "response.json"
                    if not response_file.exists():
                        response_file = episode_dir / "response.txt"

                    if response_file.exists():
                        try:
                            content = response_file.read_text()
                            response_data = json.loads(content)

                            # Terminal-Bench uses "state_analysis" and "explanation"
                            analysis = response_data.get("state_analysis", "") or response_data.get("analysis", "")
                            plan = response_data.get("explanation", "") or response_data.get("plan", "")

                            # Extract commands
                            commands_list = response_data.get("commands", [])
                            commands_text = ""
                            if isinstance(commands_list, list):
                                keystrokes = []
                                for cmd in commands_list:
                                    if isinstance(cmd, dict) and "keystrokes" in cmd:
                                        ks = cmd["keystrokes"]
                                        if ks:
                                            keystrokes.append(ks.rstrip("\n"))
                                commands_text = "\n".join(keystrokes)

                            logs["episodes"].append({
                                "state_analysis": analysis,
                                "explanation": plan,
                                "commands": commands_text,
                            })
                        except (json.JSONDecodeError, KeyError):
                            pass

            # Read results.json for trial-specific info
            results_file = temp_path / "results.json"
            if results_file.exists():
                try:
                    logs["trial_log"] = results_file.read_text()
                except Exception:
                    pass

        return logs

    def _get_harbor_run_logs(self, benchmark: dict, run_id: str) -> dict | None:
        """Get logs for a harbor run from S3."""
        import re
        import tempfile

        benchmark_id = benchmark["id"]
        batch_job_ids = benchmark.get("batch_job_ids", [])
        total_runs = benchmark.get("total_runs", 10)

        # Find the run number for this run_id
        run_number = None

        # Method 1: Try old batch_job_ids list (backward compatibility)
        for job_info in batch_job_ids:
            if job_info.get("run_id") == run_id:
                run_number = job_info["run_number"]
                break

        # Method 2: Try to extract from "run-N" format
        if run_number is None:
            match = re.match(r"run-(\d+)", run_id)
            if match:
                run_number = int(match.group(1))

        # Method 3: Search S3 status files for matching run_id (UUID format)
        if run_number is None and self.s3_client:
            for n in range(1, total_runs + 1):
                s3_status = self.s3_client.get_run_status(benchmark_id, n)
                if s3_status and s3_status.get("run_id") == run_id:
                    run_number = n
                    break

        if run_number is None:
            return None

        # Initialize empty logs
        logs = {
            "trial_log": "",
            "agent_logs": [],
            "test_stdout": "",
            "test_stderr": "",
            "episodes": [],
        }

        # List files in S3 for this run
        files = self.s3_client.list_run_files(benchmark_id, run_number)
        if not files:
            return logs

        # Download to temp directory and read logs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.s3_client.download_run_results(benchmark_id, run_number, temp_path)

            # Find the trial directory
            trial_dir = None

            # Method 1: Look for task_* subdirectory (old structure)
            for item in temp_path.iterdir():
                if item.is_dir() and item.name.startswith("task_"):
                    trial_dir = item
                    break

            # Method 2: Check if files are directly in temp_path (new structure)
            # This happens when batch_runner.py uploads trial contents directly
            if trial_dir is None:
                agent_dir = temp_path / "agent"
                verifier_dir = temp_path / "verifier"
                if agent_dir.exists() or verifier_dir.exists():
                    trial_dir = temp_path

            if trial_dir:
                # Use standard log reading
                return self._read_logs(trial_dir)

            # Fallback: read logs directly from downloaded S3 files
            episode_files = []

            for file_path in files:
                full_path = temp_path / file_path

                if not full_path.exists():
                    continue

                try:
                    if file_path.endswith("trial.log"):
                        logs["trial_log"] = full_path.read_text()
                    elif file_path.endswith("test-stdout.txt"):
                        raw_stdout = full_path.read_text()
                        logs["test_stdout"] = self._format_go_test_output(raw_stdout)
                    elif file_path.endswith("test-stderr.txt"):
                        logs["test_stderr"] = full_path.read_text()
                    elif "/agent/" in file_path and file_path.endswith((".log", ".txt")):
                        if "episode-" not in file_path:
                            logs["agent_logs"].append({
                                "name": Path(file_path).name,
                                "content": full_path.read_text(),
                            })
                    elif "/episode-" in file_path and file_path.endswith("response.txt"):
                        # Extract episode number and store for sorting
                        match = re.search(r'/episode-(\d+)/', file_path)
                        if match:
                            episode_num = int(match.group(1))
                            episode_files.append((episode_num, full_path))
                except Exception:
                    pass

            # Sort and parse episodes
            episode_files.sort(key=lambda x: x[0])
            for _, ep_path in episode_files:
                try:
                    content = ep_path.read_text()
                    response_data = json.loads(content)

                    analysis = response_data.get("analysis", "")
                    plan = response_data.get("plan", "")

                    commands_list = response_data.get("commands", [])
                    commands_text = ""
                    if isinstance(commands_list, list):
                        keystrokes = []
                        for cmd in commands_list:
                            if isinstance(cmd, dict) and "keystrokes" in cmd:
                                ks = cmd["keystrokes"]
                                if ks:
                                    keystrokes.append(ks.rstrip("\n"))
                        commands_text = "\n".join(keystrokes)

                    logs["episodes"].append({
                        "state_analysis": analysis,
                        "explanation": plan,
                        "commands": commands_text,
                    })
                except (json.JSONDecodeError, KeyError):
                    pass

        return logs

    def _read_logs(self, trial_dir: Path) -> dict:
        """Read all logs from a trial directory."""
        logs = {
            "trial_log": "",
            "agent_logs": [],
            "test_stdout": "",
            "test_stderr": "",
            "episodes": [],
        }

        # Read trial.log
        trial_log = trial_dir / "trial.log"
        if trial_log.exists():
            logs["trial_log"] = trial_log.read_text()

        # Read agent logs and episodes from agent directory
        agent_dir = trial_dir / "agent"
        if agent_dir.exists():
            # Read top-level log/txt files
            for log_file in agent_dir.glob("*.log"):
                logs["agent_logs"].append({
                    "name": log_file.name,
                    "content": log_file.read_text(),
                })
            for txt_file in agent_dir.glob("*.txt"):
                logs["agent_logs"].append({
                    "name": txt_file.name,
                    "content": txt_file.read_text(),
                })

            # Read episodes from episode-N directories
            logs["episodes"] = self._read_episodes(agent_dir)

        # Read verifier outputs
        verifier_dir = trial_dir / "verifier"
        if verifier_dir.exists():
            stdout_file = verifier_dir / "test-stdout.txt"
            stderr_file = verifier_dir / "test-stderr.txt"
            if stdout_file.exists():
                raw_stdout = stdout_file.read_text()
                # Format Go test JSON output if detected
                logs["test_stdout"] = self._format_go_test_output(raw_stdout)
            if stderr_file.exists():
                logs["test_stderr"] = stderr_file.read_text()

        return logs

    def _read_episodes(self, agent_dir: Path) -> list[dict]:
        """Read episodes from agent/episode-N directories."""
        episodes = []

        # Find all episode directories and sort them numerically
        episode_dirs = []
        for item in agent_dir.iterdir():
            if item.is_dir() and item.name.startswith("episode-"):
                try:
                    episode_num = int(item.name.split("-")[1])
                    episode_dirs.append((episode_num, item))
                except (IndexError, ValueError):
                    continue

        # Sort by episode number
        episode_dirs.sort(key=lambda x: x[0])

        for episode_num, episode_dir in episode_dirs:
            response_file = episode_dir / "response.txt"
            if response_file.exists():
                try:
                    content = response_file.read_text()
                    response_data = json.loads(content)

                    # Extract analysis and plan
                    analysis = response_data.get("analysis", "")
                    plan = response_data.get("plan", "")

                    # Extract commands from keystrokes array
                    commands_list = response_data.get("commands", [])
                    commands_text = ""
                    if isinstance(commands_list, list):
                        keystrokes = []
                        for cmd in commands_list:
                            if isinstance(cmd, dict) and "keystrokes" in cmd:
                                ks = cmd["keystrokes"]
                                if ks:  # Skip empty keystrokes
                                    keystrokes.append(ks.rstrip("\n"))
                        commands_text = "\n".join(keystrokes)

                    episodes.append({
                        "state_analysis": analysis,
                        "explanation": plan,
                        "commands": commands_text,
                    })

                except (json.JSONDecodeError, KeyError):
                    # If response.txt isn't valid JSON, skip this episode
                    continue

        return episodes

    def _format_go_test_output(self, raw_output: str) -> str:
        """Parse Go test JSON output and format it for display.

        Go test -json outputs one JSON object per line with test events.
        This function extracts test results and formats them nicely.
        """
        lines = raw_output.split("\n")
        tests = []
        has_go_json = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as Go test JSON
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                    action = event.get("Action")
                    test_name = event.get("Test")

                    # Only process test-level pass/fail events
                    if test_name and action in ("pass", "fail"):
                        has_go_json = True
                        elapsed = event.get("Elapsed", 0) or 0
                        tests.append({
                            "name": test_name,
                            "status": "PASS" if action == "pass" else "FAIL",
                            "elapsed": elapsed,
                        })
                except json.JSONDecodeError:
                    pass

        # If no Go JSON detected, return original output
        if not has_go_json:
            return raw_output

        # Format the test results nicely
        output_lines = ["Go Test Results", "=" * 50, ""]

        passed = sum(1 for t in tests if t["status"] == "PASS")
        failed = sum(1 for t in tests if t["status"] == "FAIL")

        for test in tests:
            status_icon = "✓" if test["status"] == "PASS" else "✗"
            output_lines.append(
                f"{status_icon} {test['status']}: {test['name']} ({test['elapsed']:.2f}s)"
            )

        output_lines.append("")
        output_lines.append("-" * 50)
        output_lines.append(f"Total: {len(tests)} tests, {passed} passed, {failed} failed")

        return "\n".join(output_lines)
