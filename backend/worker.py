#!/usr/bin/env python3
"""
Harbor Worker - Polls for pending tasks and executes them with Daytona.

Run with: python worker.py

Environment variables:
    DAYTONA_API_KEY     - Daytona API key
    DAYTONA_API_URL     - Daytona API URL (default: https://app.daytona.io/api)
    S3_BUCKET           - S3 bucket for results
    MAX_PARALLEL_TASKS  - Max concurrent Harbor jobs (default: 10)
    N_CONCURRENT_RUNS   - Concurrent Daytona sandboxes per job (default: 10)
    AGENT_NAME          - Agent to use (default: terminus-2)
    MODEL_NAME          - Model to use (default: openrouter/openai/gpt-4o)
    OPENROUTER_API_KEY  - OpenRouter API key for LLM calls
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harbor.job import Job
from harbor.models.job.config import (
    AgentConfig,
    JobConfig,
    OrchestratorConfig,
    TaskConfig,
)
from harbor.models.orchestrator_type import OrchestratorType
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import EnvironmentConfig

# Configuration from environment
MAX_PARALLEL_TASKS = int(os.getenv("MAX_PARALLEL_TASKS", "10"))
N_CONCURRENT_RUNS = int(os.getenv("N_CONCURRENT_RUNS", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))  # seconds
AGENT_NAME = os.getenv("AGENT_NAME", "terminus-2")
MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/openai/gpt-4o")
S3_BUCKET = os.getenv("S3_BUCKET", "")

DATA_FILE = Path(__file__).parent / "data" / "benchmarks.json"
JOBS_DIR = Path(__file__).parent / "jobs"
S3_BENCHMARKS_KEY = "state/benchmarks.json"  # S3 key for shared state

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("harbor-worker")


class HarborWorker:
    """Worker that polls for pending benchmark tasks and executes them with Harbor + Daytona."""

    def __init__(self):
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.s3_client = None

        # Initialize S3 client if bucket is configured
        if S3_BUCKET:
            try:
                import boto3
                self.s3_client = boto3.client("s3")
                logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")

    def load_benchmarks(self) -> list[dict]:
        """Load benchmarks from S3."""
        if not self.s3_client or not S3_BUCKET:
            logger.warning("S3 not configured, cannot load benchmarks")
            return []

        try:
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=S3_BENCHMARKS_KEY)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return data.get("benchmarks", [])
        except self.s3_client.exceptions.NoSuchKey:
            logger.info("No benchmarks.json found in S3 yet")
            return []
        except Exception as e:
            logger.error(f"Failed to load benchmarks from S3: {e}")
            return []

    def save_benchmarks(self, benchmarks: list[dict]):
        """Save benchmarks to S3."""
        if not self.s3_client or not S3_BUCKET:
            logger.warning("S3 not configured, cannot save benchmarks")
            return

        try:
            data = {"benchmarks": benchmarks}
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=S3_BENCHMARKS_KEY,
                Body=json.dumps(data, indent=2, default=str),
                ContentType="application/json",
            )
            logger.debug(f"Saved benchmarks to S3: {len(benchmarks)} tasks")
        except Exception as e:
            logger.error(f"Failed to save benchmarks to S3: {e}")

    def get_pending_tasks(self, limit: int) -> list[dict]:
        """Get pending tasks up to the specified limit."""
        benchmarks = self.load_benchmarks()
        pending = [b for b in benchmarks if b.get("status") == "pending"]
        return pending[:limit]

    def update_task_status(self, task_id: str, **updates):
        """Update a task's status and other fields."""
        benchmarks = self.load_benchmarks()
        for b in benchmarks:
            if b.get("id") == task_id:
                b.update(updates)
                break
        self.save_benchmarks(benchmarks)

    async def download_task_from_s3(self, task_id: str, s3_uri: str) -> Path:
        """Download task files from S3.

        Args:
            task_id: Benchmark task ID
            s3_uri: S3 URI like s3://bucket/tasks/id/

        Returns:
            Local path to downloaded task directory
        """
        # Parse S3 URI: s3://bucket/tasks/{task_id}/
        # Extract the prefix after bucket name
        if s3_uri.startswith("s3://"):
            parts = s3_uri[5:].split("/", 1)
            if len(parts) == 2:
                s3_prefix = parts[1]
            else:
                s3_prefix = f"tasks/{task_id}/"
        else:
            s3_prefix = f"tasks/{task_id}/"

        # Local directory for downloaded task
        local_task_dir = JOBS_DIR / f"tasks/{task_id}"
        local_task_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading task from s3://{S3_BUCKET}/{s3_prefix}")

        # List and download all objects with the prefix
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_path = key[len(s3_prefix) :]
                if not relative_path:
                    continue

                local_path = local_task_dir / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(S3_BUCKET, key, str(local_path))

        logger.info(f"Downloaded task to {local_task_dir}")
        return local_task_dir

    async def sync_to_s3(self, task_id: str, jobs_dir: Path):
        """Sync job results to S3."""
        if not self.s3_client or not S3_BUCKET:
            logger.info("S3 sync skipped (no S3 client configured)")
            return

        try:
            # Walk the jobs directory and upload all files
            for file_path in jobs_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(jobs_dir)
                    s3_key = f"results/{task_id}/{relative_path}"
                    self.s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)

            logger.info(f"Synced results to s3://{S3_BUCKET}/results/{task_id}/")
        except Exception as e:
            logger.error(f"Failed to sync to S3: {e}")

    async def execute_task(self, task: dict):
        """Execute a single task with Harbor + Daytona."""
        task_id = task["id"]
        task_name = task.get("task_name", "unknown")
        logger.info(f"Starting task {task_id}: {task_name}")

        self.update_task_status(
            task_id,
            status="running",
            started_at=datetime.utcnow().isoformat(),
        )

        try:
            total_runs = task.get("total_runs", 10)

            # Download task from S3 if s3_task_path is set
            s3_task_path = task.get("s3_task_path")
            if s3_task_path and self.s3_client:
                task_path = await self.download_task_from_s3(task_id, s3_task_path)
            else:
                task_path = Path(task["task_path"])

            if not task_path.exists():
                raise FileNotFoundError(f"Task path does not exist: {task_path}")

            # Jobs dir for this task
            task_jobs_dir = JOBS_DIR / f"benchmark-{task_id}"
            task_jobs_dir.mkdir(parents=True, exist_ok=True)

            job_config = JobConfig(
                job_name=f"benchmark-{task_id[:8]}",
                jobs_dir=task_jobs_dir,
                n_attempts=total_runs,  # e.g., 10 runs per task
                agents=[
                    AgentConfig(
                        name=AGENT_NAME,
                        model_name=MODEL_NAME,
                    )
                ],
                tasks=[TaskConfig(path=task_path)],
                orchestrator=OrchestratorConfig(
                    type=OrchestratorType.LOCAL,
                    n_concurrent_trials=N_CONCURRENT_RUNS,  # e.g., 10 parallel Daytona sandboxes
                ),
                # Use Daytona for sandboxed execution
                environment=EnvironmentConfig(
                    type=EnvironmentType.DAYTONA,
                    force_build=False,
                    delete=True,  # Clean up sandboxes after each run
                ),
            )

            logger.info(
                f"Task {task_id}: Running {total_runs} attempts with "
                f"{N_CONCURRENT_RUNS} concurrent, agent={AGENT_NAME}, model={MODEL_NAME}"
            )

            job = Job(job_config)
            result = await job.run()

            # Parse results from JobResult
            completed = result.n_total_trials
            errors = result.stats.n_errors
            passed = completed - errors

            # Sync results to S3
            await self.sync_to_s3(task_id, task_jobs_dir)

            self.update_task_status(
                task_id,
                status="completed",
                completed_runs=completed,
                passed_runs=passed,
                finished_at=datetime.utcnow().isoformat(),
            )
            logger.info(f"Task {task_id} completed: {passed}/{completed} passed")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            self.update_task_status(
                task_id,
                status="failed",
                error=str(e),
                finished_at=datetime.utcnow().isoformat(),
            )
        finally:
            self.running_tasks.pop(task_id, None)

    async def run(self):
        """Main worker loop - polls for pending tasks and executes them."""
        logger.info(
            f"Harbor Worker started "
            f"(MAX_PARALLEL_TASKS={MAX_PARALLEL_TASKS}, "
            f"N_CONCURRENT_RUNS={N_CONCURRENT_RUNS}, "
            f"AGENT={AGENT_NAME}, MODEL={MODEL_NAME})"
        )

        while True:
            try:
                # Calculate available slots
                available = MAX_PARALLEL_TASKS - len(self.running_tasks)

                if available > 0:
                    pending = self.get_pending_tasks(limit=available)
                    for task in pending:
                        task_id = task.get("id")
                        if task_id and task_id not in self.running_tasks:
                            coro = self.execute_task(task)
                            self.running_tasks[task_id] = asyncio.create_task(coro)
                            logger.info(
                                f"Queued task {task_id} "
                                f"({len(self.running_tasks)}/{MAX_PARALLEL_TASKS} slots used)"
                            )

                # Clean up completed tasks
                completed_ids = [
                    tid for tid, t in self.running_tasks.items() if t.done()
                ]
                for tid in completed_ids:
                    task = self.running_tasks.pop(tid)
                    # Check for exceptions
                    try:
                        task.result()
                    except Exception as e:
                        logger.error(f"Task {tid} raised exception: {e}")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)


def main():
    """Entry point for the worker."""
    # Validate required environment variables
    if not os.getenv("DAYTONA_API_KEY"):
        logger.warning("DAYTONA_API_KEY not set - Daytona integration may fail")
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("OPENROUTER_API_KEY not set - LLM calls may fail")

    worker = HarborWorker()
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
