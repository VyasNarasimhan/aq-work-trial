#!/usr/bin/env python3
"""
AWS Batch runner for benchmark jobs.

This script is the entrypoint for AWS Batch jobs. It:
1. Downloads the task from S3
2. Runs the job using the specified harness (Harbor or Terminus)
3. Uploads results back to S3
"""

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import boto3

# Timeout for the entire job (in minutes)
# If trials hang (e.g., on LLM API calls), this ensures the job eventually completes
JOB_TIMEOUT_MINUTES = 120

from botocore.exceptions import ClientError


class JobTimeoutError(Exception):
    """Raised when a job exceeds the allowed timeout."""
    pass


def create_timeout_handler(timeout_minutes: int):
    """Create a signal handler that raises JobTimeoutError."""
    def handler(signum, frame):
        raise JobTimeoutError(f"Job timed out after {timeout_minutes} minutes")
    return handler

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set all loggers to DEBUG
for logger_name in ['harbor', 'urllib3', 'botocore', 'boto3']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def get_env_var(name: str, default: str | None = None) -> str:
    """Get environment variable or raise error if not set."""
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is required")
    return value


def download_task_from_s3(s3_client, bucket: str, s3_prefix: str, local_dir: Path) -> tuple[Path, str]:
    """Download task files from S3 to local directory.

    Returns:
        Tuple of (task_path, detected_format) where detected_format is 'harbor', 'terminus', or 'unknown'.
    """
    logger.info(f"Downloading task from s3://{bucket}/{s3_prefix}")

    # List all objects with the prefix
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Get relative path from prefix
            relative_path = key[len(s3_prefix):].lstrip('/')
            if not relative_path:
                continue

            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Downloading: {key} -> {local_path}")
            s3_client.download_file(bucket, key, str(local_path))

    # Find the task directory - check for task.toml (Harbor) first, then task.yaml (Terminus)
    for path in local_dir.rglob("task.toml"):
        task_dir = path.parent
        logger.info(f"Found Harbor task directory: {task_dir}")
        return task_dir, "harbor"

    for path in local_dir.rglob("task.yaml"):
        task_dir = path.parent
        logger.info(f"Found Terminus task directory: {task_dir}")
        return task_dir, "terminus"

    # If no task config found, use the local_dir itself
    logger.warning(f"No task.toml or task.yaml found, using: {local_dir}")
    return local_dir, "unknown"


def upload_results_to_s3(s3_client, bucket: str, benchmark_id: str, run_number: int, job_dir: Path):
    """Upload job results to S3."""
    s3_prefix = f"results/{benchmark_id}/run-{run_number}/"
    logger.info(f"Uploading results to s3://{bucket}/{s3_prefix}")

    if not job_dir.exists():
        logger.warning(f"Job directory does not exist: {job_dir}")
        return

    file_count = 0
    for file_path in job_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(job_dir)
            s3_key = f"{s3_prefix}{relative_path}"
            logger.debug(f"  Uploading: {file_path} -> s3://{bucket}/{s3_key}")
            s3_client.upload_file(str(file_path), bucket, s3_key)
            file_count += 1

    logger.info(f"Uploaded {file_count} files to S3")


def upload_status_to_s3(s3_client, bucket: str, benchmark_id: str, run_number: int, status: dict):
    """Upload run status to S3."""
    s3_key = f"results/{benchmark_id}/run-{run_number}/status.json"
    logger.info(f"Uploading status to s3://{bucket}/{s3_key}")
    logger.debug(f"Status content: {json.dumps(status, default=str)}")

    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(status, default=str),
        ContentType='application/json'
    )


def is_trial_complete(trial_dir: Path) -> bool:
    """Check if a trial has completed (verifier has run and written results).

    We check that files exist AND have content to avoid race conditions
    where files are created but not yet written.
    """
    verifier_dir = trial_dir / "verifier"
    if not verifier_dir.exists():
        return False

    # Check ctrf.json - must exist and have content
    ctrf_file = verifier_dir / "ctrf.json"
    if ctrf_file.exists():
        try:
            content = ctrf_file.read_text().strip()
            if content and len(content) > 10:  # Valid JSON should be longer than 10 chars
                return True
        except Exception:
            pass

    # Check test-stdout.txt - must exist and have content
    stdout_file = verifier_dir / "test-stdout.txt"
    if stdout_file.exists():
        try:
            content = stdout_file.read_text().strip()
            if content and len(content) > 10:  # Should have some test output
                return True
        except Exception:
            pass

    return False


def upload_single_trial(
    s3_client,
    bucket: str,
    benchmark_id: str,
    trial_dir: Path,
    run_number: int,
) -> dict:
    """Upload a single trial's results to S3.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        benchmark_id: Benchmark ID.
        trial_dir: Path to the trial directory.
        run_number: Run number for this trial.

    Returns:
        Status dictionary for this run.
    """
    logger.info(f"Uploading trial {run_number}: {trial_dir.name}")

    # Verify trial directory contents before upload
    agent_dir = trial_dir / "agent"
    verifier_dir = trial_dir / "verifier"

    logger.info(f"  Trial directory structure check:")
    logger.info(f"    agent/ exists: {agent_dir.exists()}")
    logger.info(f"    verifier/ exists: {verifier_dir.exists()}")

    if agent_dir.exists():
        episode_dirs = sorted([d for d in agent_dir.iterdir() if d.is_dir() and d.name.startswith("episode-")])
        logger.info(f"    episode directories found: {len(episode_dirs)}")
        if episode_dirs:
            logger.info(f"    episodes: {[d.name for d in episode_dirs]}")
        else:
            logger.warning(f"    ⚠️  NO EPISODE DIRECTORIES found in {agent_dir}")
            # List what IS in the agent directory
            agent_contents = list(agent_dir.iterdir())
            logger.info(f"    agent/ contents: {[item.name for item in agent_contents]}")
    else:
        logger.warning(f"    ⚠️  AGENT DIRECTORY MISSING for trial {run_number}")

    if verifier_dir.exists():
        verifier_contents = list(verifier_dir.iterdir())
        logger.info(f"    verifier/ contents: {[item.name for item in verifier_contents]}")

    # Check if trial passed
    passed = check_trial_passed(trial_dir)

    # Upload all files from this trial directory
    s3_prefix = f"results/{benchmark_id}/run-{run_number}/"
    file_count = 0

    for file_path in trial_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(trial_dir)
            s3_key = f"{s3_prefix}{relative_path}"
            logger.debug(f"  Uploading: {file_path} -> s3://{bucket}/{s3_key}")
            try:
                s3_client.upload_file(str(file_path), bucket, s3_key)
                file_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")

    logger.info(f"  Uploaded {file_count} files for run-{run_number}")

    # Generate and upload status.json for this run
    status = {
        "run_id": str(uuid.uuid4()),
        "benchmark_id": benchmark_id,
        "run_number": run_number,
        "status": "completed",
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "passed": passed,
        "error": None,
        "trial_name": trial_dir.name,
    }

    upload_status_to_s3(s3_client, bucket, benchmark_id, run_number, status)
    logger.info(f"  Run-{run_number} passed: {passed}")

    return status


async def watch_and_upload_trials(
    s3_client,
    bucket: str,
    benchmark_id: str,
    job_dir: Path,
    total_runs: int,
    uploaded_trials: set,
    poll_interval: float = 5.0,
):
    """Background task that watches for completed trials and uploads them.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        benchmark_id: Benchmark ID.
        job_dir: Path to the job directory containing trial subdirectories.
        total_runs: Expected number of trials/runs.
        uploaded_trials: Set to track which trials have been uploaded (modified in place).
        poll_interval: How often to check for completed trials (seconds).
    """
    logger.info(f"Starting trial watcher for {job_dir}")

    while len(uploaded_trials) < total_runs:
        await asyncio.sleep(poll_interval)

        if not job_dir.exists():
            continue

        # Get all trial directories and sort them
        trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()])

        for run_number, trial_dir in enumerate(trial_dirs, start=1):
            # Skip if already uploaded
            if trial_dir.name in uploaded_trials:
                continue

            # Check if trial is complete
            if is_trial_complete(trial_dir):
                logger.info(f"Trial {run_number} ({trial_dir.name}) completed, waiting for files to flush...")
                # Wait a moment to ensure all files are fully written
                await asyncio.sleep(2.0)
                try:
                    upload_single_trial(s3_client, bucket, benchmark_id, trial_dir, run_number)
                    uploaded_trials.add(trial_dir.name)
                    logger.info(f"Uploaded {len(uploaded_trials)}/{total_runs} trials")
                except Exception as e:
                    logger.error(f"Failed to upload trial {run_number}: {e}")

    logger.info("All trials uploaded, watcher stopping")


async def run_harbor_job(
    task_path: Path,
    job_name: str,
    jobs_dir: Path,
    n_attempts: int = 1,
    n_concurrent_trials: int = 1,
) -> dict:
    """Run a Harbor job and return results.

    Args:
        task_path: Path to the task directory.
        job_name: Name for the job.
        jobs_dir: Directory to store job outputs.
        n_attempts: Number of trials/attempts to run.
        n_concurrent_trials: Number of trials to run concurrently.
    """
    from harbor.job import Job
    from harbor.models.job.config import (
        JobConfig,
        AgentConfig,
        TaskConfig,
        OrchestratorConfig,
    )
    from harbor.models.orchestrator_type import OrchestratorType

    logger.info(f"=" * 60)
    logger.info(f"Running Harbor job: {job_name}")
    logger.info(f"  Task path: {task_path}")
    logger.info(f"  Jobs dir: {jobs_dir}")
    logger.info(f"  n_attempts: {n_attempts}")
    logger.info(f"  n_concurrent_trials: {n_concurrent_trials}")
    logger.info(f"=" * 60)

    # List task directory contents
    logger.info("Task directory contents:")
    for item in task_path.rglob("*"):
        if item.is_file():
            logger.info(f"  {item.relative_to(task_path)} ({item.stat().st_size} bytes)")

    # Get the model from environment or use default
    model_name = os.environ.get("MODEL_NAME", "openrouter/openai/gpt-5")
    agent_name = os.environ.get("AGENT_NAME", "terminus-2")

    logger.info(f"Agent: {agent_name}")
    logger.info(f"Model: {model_name}")

    config = JobConfig(
        job_name=job_name,
        jobs_dir=jobs_dir,
        n_attempts=n_attempts,
        agents=[
            AgentConfig(
                name=agent_name,
                model_name=model_name,
            )
        ],
        tasks=[TaskConfig(path=task_path)],
        orchestrator=OrchestratorConfig(
            type=OrchestratorType.LOCAL,
            n_concurrent_trials=n_concurrent_trials,
        ),
    )

    logger.info("Creating Harbor Job instance...")
    job = Job(config)

    # Log system resources to help diagnose concurrency issues
    import psutil
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    logger.info("=" * 60)
    logger.info("SYSTEM RESOURCES:")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Total memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {memory.available / (1024**3):.1f} GB")
    logger.info(f"  Memory usage: {memory.percent}%")
    logger.info("CONCURRENCY CONFIG:")
    logger.info(f"  Requested n_attempts (total trials): {n_attempts}")
    logger.info(f"  Requested n_concurrent_trials: {n_concurrent_trials}")
    if n_concurrent_trials > cpu_count:
        logger.warning(f"  ⚠️  n_concurrent_trials ({n_concurrent_trials}) > CPU cores ({cpu_count})")
        logger.warning(f"  ⚠️  Some trials may not run truly in parallel due to CPU constraints")
    else:
        logger.info(f"  ✓ n_concurrent_trials ({n_concurrent_trials}) <= CPU cores ({cpu_count}) - good for parallelism")
    logger.info("=" * 60)

    logger.info("Starting Harbor job execution...")
    start_time = datetime.now()
    try:
        result = await job.run()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Harbor job completed in {elapsed:.1f} seconds. Result: {result}")

        # Analyze if trials ran in parallel
        # If n trials completed in ~same time as 1 trial would take, they ran in parallel
        avg_time_per_trial = elapsed / n_attempts if n_attempts > 0 else 0
        logger.info("=" * 60)
        logger.info("PARALLELISM ANALYSIS:")
        logger.info(f"  Total elapsed time: {elapsed:.1f}s")
        logger.info(f"  Total trials: {n_attempts}")
        logger.info(f"  Avg time per trial (if sequential): {elapsed:.1f}s / {n_attempts} = {avg_time_per_trial:.1f}s")
        if n_concurrent_trials >= n_attempts:
            logger.info(f"  Expected behavior: All {n_attempts} trials should have run in parallel")
            logger.info(f"  If total time ≈ single trial time, parallelism worked correctly")
        else:
            batches = (n_attempts + n_concurrent_trials - 1) // n_concurrent_trials
            logger.info(f"  Expected batches: {batches} (due to n_concurrent_trials={n_concurrent_trials})")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Harbor job failed with exception: {e}")
        logger.exception("Full traceback:")
        raise

    # Log job directory contents after completion
    job_dir = jobs_dir / job_name
    if job_dir.exists():
        logger.info(f"Job output directory contents ({job_dir}):")
        for item in job_dir.rglob("*"):
            if item.is_file():
                logger.info(f"  {item.relative_to(job_dir)} ({item.stat().st_size} bytes)")
                # Print contents of small text files for debugging
                if item.suffix in ['.json', '.txt', '.log'] and item.stat().st_size < 10000:
                    try:
                        content = item.read_text()
                        logger.info(f"  --- Contents of {item.name} ---")
                        logger.info(content[:2000])
                        if len(content) > 2000:
                            logger.info("... (truncated)")
                        logger.info(f"  --- End of {item.name} ---")
                    except Exception as e:
                        logger.warning(f"Could not read {item}: {e}")

        # Specifically check for reward.txt
        logger.info("=" * 40)
        logger.info("Checking for reward.txt in trial directories...")
        for trial_dir in job_dir.iterdir():
            if trial_dir.is_dir():
                reward_file = trial_dir / "verifier" / "reward.txt"
                ctrf_file = trial_dir / "verifier" / "ctrf.json"
                verifier_dir = trial_dir / "verifier"

                logger.info(f"Trial: {trial_dir.name}")
                logger.info(f"  Verifier dir exists: {verifier_dir.exists()}")
                if verifier_dir.exists():
                    logger.info(f"  Verifier dir contents: {list(verifier_dir.iterdir()) if verifier_dir.exists() else 'N/A'}")
                logger.info(f"  reward.txt exists: {reward_file.exists()}")
                logger.info(f"  ctrf.json exists: {ctrf_file.exists()}")

                if reward_file.exists():
                    logger.info(f"  reward.txt content: {reward_file.read_text()}")

                # Check for verifier logs
                verifier_log = trial_dir / "verifier" / "verifier.log"
                if verifier_log.exists():
                    logger.info(f"  --- Verifier log ---")
                    logger.info(verifier_log.read_text()[:3000])

                # Check for test output
                test_stdout = trial_dir / "verifier" / "test-stdout.txt"
                test_stderr = trial_dir / "verifier" / "test-stderr.txt"
                if test_stdout.exists():
                    logger.info(f"  --- Test stdout ---")
                    logger.info(test_stdout.read_text()[:2000])
                if test_stderr.exists():
                    logger.info(f"  --- Test stderr ---")
                    logger.info(test_stderr.read_text()[:2000])
        logger.info("=" * 40)
    else:
        logger.warning(f"Job directory does not exist: {job_dir}")

    return {
        "job_name": job_name,
        "job_dir": str(jobs_dir / job_name),
        "completed": True,
    }


async def run_terminus_job(
    task_path: Path,
    job_name: str,
    jobs_dir: Path,
    n_attempts: int = 1,
    n_concurrent_trials: int = 1,
) -> dict:
    """Run a Terminus (Terminal-Bench) job.

    Uses the terminal-bench Python API (Harness class) to execute benchmarks.

    Args:
        task_path: Path to the task directory.
        job_name: Name for the job.
        jobs_dir: Directory to store job outputs.
        n_attempts: Number of trials/attempts to run.
        n_concurrent_trials: Number of trials to run concurrently.
    """
    from terminal_bench import Harness
    from terminal_bench.agents.agent_name import AgentName

    logger.info("=" * 60)
    logger.info(f"Running Terminus job: {job_name}")
    logger.info(f"  Task path: {task_path}")
    logger.info(f"  Jobs dir: {jobs_dir}")
    logger.info(f"  n_attempts: {n_attempts}")
    logger.info(f"  n_concurrent_trials: {n_concurrent_trials}")
    logger.info("=" * 60)

    # List task directory contents
    logger.info("Task directory contents:")
    for item in task_path.rglob("*"):
        if item.is_file():
            logger.info(f"  {item.relative_to(task_path)} ({item.stat().st_size} bytes)")

    # Get configuration from environment
    model_name = os.environ.get("MODEL_NAME", "openrouter/openai/gpt-5")

    output_dir = jobs_dir / job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")

    # Log system resources
    import psutil
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    logger.info("=" * 60)
    logger.info("SYSTEM RESOURCES:")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Total memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {memory.available / (1024**3):.1f} GB")
    logger.info(f"  Memory usage: {memory.percent}%")
    logger.info("CONCURRENCY CONFIG:")
    logger.info(f"  Requested n_attempts (total trials): {n_attempts}")
    logger.info(f"  Requested n_concurrent_trials: {n_concurrent_trials}")
    logger.info("=" * 60)

    start_time = datetime.now()
    try:
        # Terminal-bench expects dataset_path to be a directory containing task subdirectories.
        # Each task subdirectory should have a task.yaml file.
        # Structure: dataset_path/<task_id>/task.yaml
        # Our task_path IS the task directory (contains task.yaml), so we need:
        #   - dataset_path = task_path.parent (parent directory containing the task folder)
        #   - task_ids = [task_path.name] (the task folder name)
        dataset_path = task_path.parent
        task_id = task_path.name

        logger.info(f"Terminus dataset_path: {dataset_path}")
        logger.info(f"Terminus task_id: {task_id}")

        # Create Harness instance with configuration
        # Note: model_name must be passed both to Harness AND via agent_kwargs
        # because the Terminus agent requires it in its constructor
        harness = Harness(
            output_path=output_dir,
            run_id=job_name,
            agent_name=AgentName.TERMINUS,
            dataset_path=dataset_path,
            task_ids=[task_id],
            model_name=model_name,
            agent_kwargs={"model_name": model_name},
            n_concurrent_trials=n_concurrent_trials,
            n_attempts=n_attempts,
            livestream=True,  # Enable real-time output
            cleanup=True,  # Clean up Docker containers after completion
        )

        logger.info("Starting Terminus harness execution...")
        results = harness.run()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Terminus job completed in {elapsed:.1f} seconds")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Terminus job failed with exception: {e}")
        logger.exception("Full traceback:")
        raise

    # Log output directory contents
    if output_dir.exists():
        logger.info(f"Output directory contents ({output_dir}):")
        for item in output_dir.rglob("*"):
            if item.is_file():
                logger.info(f"  {item.relative_to(output_dir)} ({item.stat().st_size} bytes)")
    else:
        logger.warning(f"Output directory does not exist: {output_dir}")

    return {
        "job_name": job_name,
        "job_dir": str(output_dir),
        "completed": True,
    }


async def watch_and_upload_terminus_trials(
    s3_client,
    bucket: str,
    benchmark_id: str,
    output_dir: Path,
    n_attempts: int,
    uploaded_trials: set,
    poll_interval: float = 5.0,
) -> None:
    """Watch for completed Terminus trials and upload them incrementally to S3.

    Terminus structure: runs/{timestamp}/{task_id}/{trial_name}/
    A trial is complete when its results.json exists in the trial directory.
    """
    logger.info(f"Starting Terminus trial watcher for {output_dir}")

    while True:
        await asyncio.sleep(poll_interval)

        # Find the timestamped output directory
        result_dir = None
        if output_dir.exists():
            for item in sorted(output_dir.iterdir(), reverse=True):
                if item.is_dir() and "__" in item.name:
                    result_dir = item
                    break

        if not result_dir or not result_dir.exists():
            continue

        # Check for completed trials (each task_id folder)
        for task_dir in result_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_id = task_dir.name

            # Each task_id has one trial folder inside
            for trial_dir in task_dir.iterdir():
                if not trial_dir.is_dir():
                    continue

                trial_name = trial_dir.name
                trial_key = f"{task_id}/{trial_name}"

                if trial_key in uploaded_trials:
                    continue

                # Check if trial is complete (has results.json)
                trial_results = trial_dir / "results.json"
                if not trial_results.exists():
                    continue

                logger.info(f"Found completed Terminus trial: {trial_key}")

                # Upload all files for this trial
                s3_prefix = f"results/{benchmark_id}/run-1/{trial_key}/"
                file_count = 0

                for file_path in trial_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(trial_dir)
                        s3_key = f"{s3_prefix}{relative_path}"
                        try:
                            s3_client.upload_file(str(file_path), bucket, s3_key)
                            file_count += 1
                        except Exception as e:
                            logger.error(f"Failed to upload {file_path}: {e}")

                logger.info(f"Uploaded {file_count} files for trial {trial_key}")
                uploaded_trials.add(trial_key)

        # Also upload the main results.json if it exists and has been updated
        main_results = result_dir / "results.json"
        if main_results.exists():
            try:
                s3_key = f"results/{benchmark_id}/run-1/results.json"
                s3_client.upload_file(str(main_results), bucket, s3_key)
            except Exception as e:
                logger.debug(f"Failed to upload main results.json: {e}")

        # Check if all trials are done
        if len(uploaded_trials) >= n_attempts:
            logger.info(f"All {n_attempts} Terminus trials uploaded")
            break


async def run_terminus_cli_job(
    task_path: Path,
    job_name: str,
    jobs_dir: Path,
    n_attempts: int = 10,
    n_concurrent_trials: int = 10,
) -> dict:
    """Run a Terminus (Terminal-Bench) job using the tb CLI.

    This function:
    1. Creates a dataset directory with n_attempts copies of the task
    2. Runs `tb run` with --n-attempts 1 --n-concurrent n_concurrent_trials
    3. Returns the results directory

    Args:
        task_path: Path to the single task directory (contains task.yaml).
        job_name: Name for the job (used as run_id).
        jobs_dir: Directory to store job outputs.
        n_attempts: Number of task copies to create (default 10 for 10 trials).
        n_concurrent_trials: Number of concurrent trials to run.
    """
    import shutil

    logger.info("=" * 60)
    logger.info(f"Running Terminus CLI job: {job_name}")
    logger.info(f"  Task path: {task_path}")
    logger.info(f"  Jobs dir: {jobs_dir}")
    logger.info(f"  n_attempts (task copies): {n_attempts}")
    logger.info(f"  n_concurrent_trials: {n_concurrent_trials}")
    logger.info("=" * 60)

    # Get configuration from environment
    model_name = os.environ.get("MODEL_NAME", "openrouter/openai/gpt-5")

    # Create dataset directory with n_attempts copies of the task
    dataset_dir = jobs_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    task_base_name = task_path.name
    logger.info(f"Creating {n_attempts} copies of task '{task_base_name}' in {dataset_dir}")

    for i in range(1, n_attempts + 1):
        dest_name = f"{task_base_name}-{i}"
        dest_path = dataset_dir / dest_name
        logger.info(f"  Copying task to: {dest_path}")
        shutil.copytree(task_path, dest_path)

    # List dataset directory contents
    logger.info(f"Dataset directory contents:")
    for item in dataset_dir.iterdir():
        if item.is_dir():
            logger.info(f"  {item.name}/")

    # Create output directory
    output_dir = jobs_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log system resources
    import psutil
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    logger.info("=" * 60)
    logger.info("SYSTEM RESOURCES:")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Total memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {memory.available / (1024**3):.1f} GB")
    logger.info(f"  Memory usage: {memory.percent}%")
    logger.info("=" * 60)

    # Build tb run command
    # tb run --agent terminus --model $MODEL --dataset-path ./dataset --n-attempts 1 --n-concurrent 10
    cmd = [
        "tb", "run",
        "--agent", "terminus",
        "--model", model_name,
        "--dataset-path", str(dataset_dir),
        "--output-path", str(output_dir),
        "--n-attempts", "1",  # 1 attempt per task (we have n_attempts copies)
        "--n-concurrent", str(n_concurrent_trials),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    start_time = datetime.now()

    try:
        # Run tb command using async subprocess to allow watcher to run concurrently
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Stream output in real-time (async)
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            logger.info(f"[tb] {line.decode().rstrip()}")

        # Wait for completion
        return_code = await process.wait()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"tb run completed in {elapsed:.1f} seconds with return code {return_code}")

        if return_code != 0:
            raise RuntimeError(f"tb run failed with return code {return_code}")

    except Exception as e:
        logger.error(f"Terminus CLI job failed with exception: {e}")
        logger.exception("Full traceback:")
        raise

    # Find the output directory (runs/{timestamp}/)
    # tb creates a timestamped directory like 2025-12-04__09-54-08
    result_dir = None
    if output_dir.exists():
        for item in sorted(output_dir.iterdir(), reverse=True):
            if item.is_dir() and "__" in item.name:
                result_dir = item
                break

    if result_dir:
        logger.info(f"Found result directory: {result_dir}")
        logger.info(f"Result directory contents:")
        for item in result_dir.rglob("*"):
            if item.is_file():
                logger.info(f"  {item.relative_to(result_dir)} ({item.stat().st_size} bytes)")
    else:
        logger.warning(f"No result directory found in {output_dir}")
        result_dir = output_dir

    return {
        "job_name": job_name,
        "job_dir": str(result_dir),
        "output_dir": str(output_dir),  # Return output_dir for watcher
        "completed": True,
    }


def parse_test_output(stdout_content: str) -> tuple[bool, int, int, list[dict]]:
    """Parse test output (pytest or Go) to extract test results.

    Returns:
        Tuple of (all_passed, passed_count, failed_count, tests_list)
    """
    import re

    tests = []
    passed_count = 0
    failed_count = 0

    # Parse test output line by line
    # Supports both pytest and Go test formats
    for line in stdout_content.splitlines():
        line = line.strip()

        # Pytest format: PASSED ../tests/test_file.py::test_name
        # or: FAILED ../tests/test_file.py::test_name[param]
        if line.startswith("PASSED ") or line.startswith("FAILED "):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                raw_status = parts[0]
                test_path = parts[1].strip()

                # Extract test case name from path like ../tests/test_file.py::test_name
                if "::" in test_path:
                    test_name = test_path.split("::")[-1]
                else:
                    test_name = test_path

                # If test name has spaces, take just the first part
                if " " in test_name:
                    test_name = test_name.split()[0]

                status = "passed" if raw_status == "PASSED" else "failed"
                tests.append({
                    "name": test_name,
                    "status": status,
                    "duration": 0
                })

                if status == "passed":
                    passed_count += 1
                else:
                    failed_count += 1

        # Go test format: --- PASS: TestName (0.00s) or --- FAIL: TestName (0.00s)
        elif line.startswith("--- PASS:") or line.startswith("--- FAIL:"):
            go_match = re.match(r'--- (PASS|FAIL): (\S+)', line)
            if go_match:
                status = "passed" if go_match.group(1) == "PASS" else "failed"
                test_name = go_match.group(2)
                tests.append({
                    "name": test_name,
                    "status": status,
                    "duration": 0
                })

                if status == "passed":
                    passed_count += 1
                else:
                    failed_count += 1

    # If no line-by-line matches, try summary line as fallback
    if not tests:
        summary_match = re.search(r'(\d+) passed', stdout_content)
        if summary_match:
            passed_count = int(summary_match.group(1))

        failed_match = re.search(r'(\d+) failed', stdout_content)
        if failed_match:
            failed_count = int(failed_match.group(1))

    all_passed = failed_count == 0 and passed_count > 0
    return all_passed, passed_count, failed_count, tests


# Keep old name for backward compatibility
parse_pytest_output = parse_test_output


def generate_reward_files(verifier_dir: Path, passed: bool, passed_count: int, failed_count: int, tests: list[dict]):
    """Generate reward.txt and ctrf.json files from parsed test results."""
    logger.info(f"Generating reward files in {verifier_dir}")

    # Generate reward.txt (1.0 for all passed, 0.0 otherwise)
    reward = 1.0 if passed else 0.0
    reward_file = verifier_dir / "reward.txt"
    reward_file.write_text(str(reward))
    logger.info(f"Generated reward.txt with value: {reward}")

    # Generate ctrf.json
    ctrf_data = {
        "results": {
            "tool": {
                "name": "pytest"
            },
            "summary": {
                "tests": passed_count + failed_count,
                "passed": passed_count,
                "failed": failed_count,
                "pending": 0,
                "skipped": 0,
                "other": 0
            },
            "tests": tests
        }
    }

    ctrf_file = verifier_dir / "ctrf.json"
    ctrf_file.write_text(json.dumps(ctrf_data, indent=2))
    logger.info(f"Generated ctrf.json with {len(tests)} tests")


def check_job_passed(job_dir: Path) -> bool:
    """Check if a job passed by examining the verifier results."""
    if not job_dir.exists():
        return False

    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        verifier_dir = trial_dir / "verifier"
        result_file = verifier_dir / "ctrf.json"

        # First try to read existing ctrf.json
        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)

                # Check if all tests passed
                tests = result.get("results", {}).get("tests", [])
                if tests:
                    all_passed = all(t.get("status") == "passed" for t in tests)
                    return all_passed
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: Parse test-stdout.txt if ctrf.json doesn't exist
        test_stdout = verifier_dir / "test-stdout.txt"
        if test_stdout.exists() and not result_file.exists():
            logger.info(f"ctrf.json not found, parsing test-stdout.txt as fallback")
            try:
                stdout_content = test_stdout.read_text()
                all_passed, passed_count, failed_count, tests = parse_pytest_output(stdout_content)

                # Generate the missing files
                generate_reward_files(verifier_dir, all_passed, passed_count, failed_count, tests)

                return all_passed
            except Exception as e:
                logger.error(f"Failed to parse test-stdout.txt: {e}")
                return False

    return False


def check_trial_passed(trial_dir: Path) -> bool:
    """Check if a single trial passed by examining its verifier results."""
    verifier_dir = trial_dir / "verifier"
    result_file = verifier_dir / "ctrf.json"

    # First try to read existing ctrf.json
    if result_file.exists():
        try:
            with open(result_file) as f:
                result = json.load(f)

            # Check if all tests passed
            tests = result.get("results", {}).get("tests", [])
            if tests:
                return all(t.get("status") == "passed" for t in tests)
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: Parse test-stdout.txt if ctrf.json doesn't exist
    test_stdout = verifier_dir / "test-stdout.txt"
    if test_stdout.exists() and not result_file.exists():
        logger.info(f"ctrf.json not found in {trial_dir.name}, parsing test-stdout.txt")
        try:
            stdout_content = test_stdout.read_text()
            all_passed, passed_count, failed_count, tests = parse_pytest_output(stdout_content)

            # Generate the missing files
            verifier_dir.mkdir(parents=True, exist_ok=True)
            generate_reward_files(verifier_dir, all_passed, passed_count, failed_count, tests)

            return all_passed
        except Exception as e:
            logger.error(f"Failed to parse test-stdout.txt: {e}")
            return False

    return False


def upload_all_trials_to_s3(
    s3_client,
    bucket: str,
    benchmark_id: str,
    job_dir: Path,
    total_runs: int,
) -> list[dict]:
    """Upload each trial's results to its designated S3 run folder.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        benchmark_id: Benchmark ID.
        job_dir: Path to the job directory containing trial subdirectories.
        total_runs: Expected number of trials/runs.

    Returns:
        List of status dictionaries for each run.
    """
    logger.info(f"Uploading all trials from {job_dir} to S3")

    if not job_dir.exists():
        logger.warning(f"Job directory does not exist: {job_dir}")
        return []

    # Get all trial directories and sort them to assign run numbers
    trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(trial_dirs)} trial directories")

    statuses = []

    for run_number, trial_dir in enumerate(trial_dirs, start=1):
        logger.info(f"Processing trial {run_number}: {trial_dir.name}")

        # Check if trial passed
        passed = check_trial_passed(trial_dir)

        # Upload all files from this trial directory
        s3_prefix = f"results/{benchmark_id}/run-{run_number}/"
        file_count = 0

        for file_path in trial_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(trial_dir)
                s3_key = f"{s3_prefix}{relative_path}"
                logger.debug(f"  Uploading: {file_path} -> s3://{bucket}/{s3_key}")
                s3_client.upload_file(str(file_path), bucket, s3_key)
                file_count += 1

        logger.info(f"  Uploaded {file_count} files for run-{run_number}")

        # Generate and upload status.json for this run
        status = {
            "run_id": str(uuid.uuid4()),
            "benchmark_id": benchmark_id,
            "run_number": run_number,
            "status": "completed",
            "started_at": datetime.now().isoformat(),  # Approximate
            "finished_at": datetime.now().isoformat(),
            "passed": passed,
            "error": None,
            "trial_name": trial_dir.name,
        }

        upload_status_to_s3(s3_client, bucket, benchmark_id, run_number, status)
        statuses.append(status)
        logger.info(f"  Run-{run_number} passed: {passed}")

    # If we have fewer trials than expected, create placeholder statuses for missing ones
    for run_number in range(len(trial_dirs) + 1, total_runs + 1):
        logger.warning(f"Missing trial for run-{run_number}, creating failed status")
        status = {
            "run_id": str(uuid.uuid4()),
            "benchmark_id": benchmark_id,
            "run_number": run_number,
            "status": "failed",
            "started_at": None,
            "finished_at": datetime.now().isoformat(),
            "passed": False,
            "error": "Trial did not complete",
        }
        upload_status_to_s3(s3_client, bucket, benchmark_id, run_number, status)
        statuses.append(status)

    return statuses


def main():
    """Main entry point for the batch runner.

    This runner executes multiple trials concurrently using Harbor's built-in
    concurrency feature. All trials run on the same EC2 instance.
    """
    logger.info("=" * 60)
    logger.info("AWS Batch Benchmark Runner (Concurrent Trials)")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # Log all environment variables (excluding secrets)
    logger.info("Environment variables:")
    for key, value in sorted(os.environ.items()):
        if any(secret in key.lower() for secret in ['key', 'secret', 'password', 'token']):
            logger.info(f"  {key}=***REDACTED***")
        else:
            logger.info(f"  {key}={value}")

    # Get environment variables
    try:
        benchmark_id = get_env_var("BENCHMARK_ID")
        s3_task_path = get_env_var("S3_TASK_PATH")
        bucket = get_env_var("S3_BUCKET")
        aws_region = get_env_var("AWS_REGION", "us-west-2")
        harness = get_env_var("HARNESS", "harbor")
        # N_ATTEMPTS: actual number of trials to run (10)
        # TOTAL_RUNS: number of run-N/ folders for S3 (10 for harbor, 1 for terminus)
        # N_CONCURRENT_TRIALS: parallelism level (same as n_attempts)
        n_attempts = int(get_env_var("N_ATTEMPTS", "10"))
        total_runs = int(get_env_var("TOTAL_RUNS", "10"))
        n_concurrent_trials = int(get_env_var("N_CONCURRENT_TRIALS", "10"))
    except ValueError as e:
        logger.error(f"Missing environment variable: {e}")
        sys.exit(1)

    logger.info(f"Benchmark ID: {benchmark_id}")
    logger.info(f"S3 Task Path: {s3_task_path}")
    logger.info(f"S3 Bucket: {bucket}")
    logger.info(f"Harness: {harness}")
    logger.info(f"N Attempts (trials): {n_attempts}")
    logger.info(f"Total Runs (S3 folders): {total_runs}")
    logger.info(f"Concurrent Trials: {n_concurrent_trials}")

    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=aws_region)

    # Create temporary directories
    work_dir = Path(tempfile.mkdtemp(prefix="benchmark-"))
    task_dir = work_dir / "task"
    jobs_dir = work_dir / "jobs"

    task_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Working directory: {work_dir}")

    job_name = f"benchmark-{benchmark_id[:8]}"
    job_dir = jobs_dir / job_name
    started_at = datetime.now().isoformat()
    overall_status = "running"
    overall_error = None

    # Track which trials have been uploaded (for incremental uploads)
    uploaded_trials: set[str] = set()

    try:
        # Parse S3 path
        # Expected format: s3://bucket/prefix/ or just prefix/
        if s3_task_path.startswith("s3://"):
            # Parse s3://bucket/prefix format
            parts = s3_task_path[5:].split("/", 1)
            s3_bucket = parts[0]
            s3_prefix = parts[1] if len(parts) > 1 else ""
        else:
            # Assume it's just the prefix
            s3_bucket = bucket
            s3_prefix = s3_task_path

        # Download task from S3
        task_path, detected_format = download_task_from_s3(s3_client, s3_bucket, s3_prefix, task_dir)

        # Validate harness matches detected format (warn but continue)
        if harness == "harbor" and detected_format == "terminus":
            logger.warning("Harbor harness requested but task.yaml found. Switching to Terminus.")
            harness = "terminus"
        elif harness == "terminus" and detected_format == "harbor":
            logger.warning("Terminus harness requested but task.toml found. Switching to Harbor.")
            harness = "harbor"

        # Set up job timeout to prevent trials from hanging indefinitely
        logger.info(f"Setting job timeout to {JOB_TIMEOUT_MINUTES} minutes")
        signal.signal(signal.SIGALRM, create_timeout_handler(JOB_TIMEOUT_MINUTES))
        signal.alarm(JOB_TIMEOUT_MINUTES * 60)

        if harness == "terminus":
            # Terminus: Run job with concurrent watcher for incremental uploads
            logger.info(f"Starting Terminus CLI job with {n_attempts} trials, {n_concurrent_trials} concurrent")

            # Create output directory for watcher to monitor
            output_dir = jobs_dir / "runs"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(output_dir)
            # output_dir = Path(f"/tmp/{job_name}/jobs/runs")

            async def run_terminus_with_watcher():
                """Run Terminus job with concurrent trial watcher for incremental uploads."""
                watcher_task = asyncio.create_task(
                    watch_and_upload_terminus_trials(
                        s3_client,
                        bucket,
                        benchmark_id,
                        output_dir,
                        n_attempts,
                        uploaded_trials,
                        poll_interval=5.0,
                    )
                )

                try:
                    result = await run_terminus_cli_job(
                        task_path,
                        job_name,
                        jobs_dir,
                        n_attempts=n_attempts,
                        n_concurrent_trials=n_concurrent_trials,
                    )
                    return result
                finally:
                    watcher_task.cancel()
                    try:
                        await watcher_task
                    except asyncio.CancelledError:
                        pass

            result = asyncio.run(run_terminus_with_watcher())
            terminus_result_dir = Path(result["job_dir"])
            logger.info(f"Terminus job completed. Result dir: {terminus_result_dir}")

            # Final upload of any remaining files (including main results.json)
            logger.info(f"Final upload of remaining files to run-1/...")
            s3_prefix = f"results/{benchmark_id}/run-1/"
            file_count = 0

            if terminus_result_dir.exists():
                for file_path in terminus_result_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(terminus_result_dir)
                        s3_key = f"{s3_prefix}{relative_path}"
                        try:
                            s3_client.upload_file(str(file_path), bucket, s3_key)
                            file_count += 1
                        except Exception as e:
                            logger.error(f"Failed to upload {file_path}: {e}")
            else:
                logger.warning(f"Terminus result directory does not exist: {terminus_result_dir}")

            logger.info(f"Final upload: {file_count} files to S3")
            logger.info(f"Trials uploaded incrementally: {len(uploaded_trials)}")

            # Create status.json for run-1
            status = {
                "run_id": str(uuid.uuid4()),
                "benchmark_id": benchmark_id,
                "run_number": 1,
                "status": "completed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "passed": True,  # Will be determined by results.json parsing
                "error": None,
            }
            upload_status_to_s3(s3_client, bucket, benchmark_id, 1, status)
            overall_status = "completed"
            logger.info(f"Terminus results uploaded to run-1/")

        else:
            # Harbor: Run with trial watcher for incremental uploads to run-N/
            async def run_with_watcher():
                """Run job with concurrent trial watcher for incremental uploads."""
                watcher_task = asyncio.create_task(
                    watch_and_upload_trials(
                        s3_client,
                        bucket,
                        benchmark_id,
                        job_dir,
                        n_attempts,
                        uploaded_trials,
                        poll_interval=5.0,
                    )
                )

                try:
                    result = await run_harbor_job(
                        task_path,
                        job_name,
                        jobs_dir,
                        n_attempts=n_attempts,
                        n_concurrent_trials=n_concurrent_trials,
                    )
                    return result
                finally:
                    watcher_task.cancel()
                    try:
                        await watcher_task
                    except asyncio.CancelledError:
                        pass

            logger.info(f"Starting Harbor job with {n_attempts} trials, {n_concurrent_trials} concurrent")
            logger.info("Trial watcher will upload results incrementally as they complete")
            result = asyncio.run(run_with_watcher())

            logger.info(f"Harbor job completed. Checking for any remaining uploads...")

            # Upload any trials that weren't caught by the watcher
            if job_dir.exists():
                trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()])
                for run_number, trial_dir in enumerate(trial_dirs, start=1):
                    if trial_dir.name not in uploaded_trials:
                        logger.info(f"Uploading remaining trial {run_number}: {trial_dir.name}")
                        try:
                            upload_single_trial(s3_client, bucket, benchmark_id, trial_dir, run_number)
                            uploaded_trials.add(trial_dir.name)
                        except Exception as e:
                            logger.error(f"Failed to upload trial {run_number}: {e}")

            # Create failed status for any missing trials
            trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()]) if job_dir.exists() else []
            for run_number in range(len(trial_dirs) + 1, n_attempts + 1):
                logger.warning(f"Missing trial for run-{run_number}, creating failed status")
                status = {
                    "run_id": str(uuid.uuid4()),
                    "benchmark_id": benchmark_id,
                    "run_number": run_number,
                    "status": "failed",
                    "started_at": None,
                    "finished_at": datetime.now().isoformat(),
                    "passed": False,
                    "error": "Trial did not complete",
                }
                upload_status_to_s3(s3_client, bucket, benchmark_id, run_number, status)

            overall_status = "completed"
            logger.info(f"All trials processed. Uploaded: {len(uploaded_trials)}/{n_attempts}")

    except JobTimeoutError as e:
        logger.warning(f"Job timeout: {e}")

        # Cancel the alarm since we're handling the timeout
        signal.alarm(0)

        # Upload any completed trials that weren't caught by the watcher
        if job_dir.exists():
            try:
                trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()])
                for run_number, trial_dir in enumerate(trial_dirs, start=1):
                    if trial_dir.name not in uploaded_trials and is_trial_complete(trial_dir):
                        logger.info(f"Uploading completed trial {run_number} after timeout")
                        upload_single_trial(s3_client, bucket, benchmark_id, trial_dir, run_number)
                        uploaded_trials.add(trial_dir.name)
            except Exception as upload_error:
                logger.error(f"Failed to upload completed trials after timeout: {upload_error}")

        # Create failed status for any incomplete runs with timeout message
        trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()]) if job_dir.exists() else []
        completed_run_numbers = set()
        for run_number, trial_dir in enumerate(trial_dirs, start=1):
            if trial_dir.name in uploaded_trials:
                completed_run_numbers.add(run_number)

        for run_number in range(1, n_attempts + 1):
            if run_number not in completed_run_numbers:
                logger.warning(f"Run-{run_number} timed out, creating failed status")
                status = {
                    "run_id": str(uuid.uuid4()),
                    "benchmark_id": benchmark_id,
                    "run_number": run_number,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(),
                    "passed": False,
                    "error": f"Trial timed out after {JOB_TIMEOUT_MINUTES} minutes",
                }
                upload_status_to_s3(s3_client, bucket, benchmark_id, run_number, status)

        # Mark overall job as completed (not failed) since we have partial results
        overall_status = "completed"
        overall_error = str(e)
        logger.info(f"Timeout handled. Uploaded: {len(uploaded_trials)}/{n_attempts} trials completed")

    except Exception as e:
        logger.error(f"Error running job: {e}")
        logger.exception("Full traceback:")

        overall_status = "failed"
        overall_error = str(e)

        # Try to upload any partial results that weren't caught by the watcher
        if job_dir.exists():
            try:
                trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir()])
                for run_number, trial_dir in enumerate(trial_dirs, start=1):
                    if trial_dir.name not in uploaded_trials and is_trial_complete(trial_dir):
                        logger.info(f"Uploading partial result for trial {run_number}")
                        upload_single_trial(s3_client, bucket, benchmark_id, trial_dir, run_number)
                        uploaded_trials.add(trial_dir.name)
            except Exception as upload_error:
                logger.error(f"Failed to upload partial results: {upload_error}")

    finally:
        # Cancel the timeout alarm
        signal.alarm(0)
        # Results are uploaded incrementally by the watcher

    logger.info("=" * 60)
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    logger.info(f"Overall status: {overall_status}")
    if overall_error:
        logger.info(f"Error: {overall_error}")
    logger.info("=" * 60)

    # Exit with appropriate code
    if overall_status == "failed":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
