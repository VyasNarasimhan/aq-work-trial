"""AWS Batch client for submitting and monitoring benchmark jobs."""

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError


class BatchClient:
    """Client for interacting with AWS Batch."""

    def __init__(
        self,
        job_queue: str | None = None,
        job_definition: str | None = None,
        region: str | None = None,
    ):
        """Initialize the Batch client.

        Args:
            job_queue: Name of the AWS Batch job queue.
            job_definition: Name of the AWS Batch job definition.
            region: AWS region. Defaults to AWS_REGION env var or us-west-2.
        """
        self.region = region or os.environ.get("AWS_REGION", "us-west-2")
        self.job_queue = job_queue or os.environ.get(
            "AWS_BATCH_JOB_QUEUE", "aq-benchmark-queue"
        )
        self.job_definition = job_definition or os.environ.get(
            "AWS_BATCH_JOB_DEF", "aq-benchmark-runner"
        )
        self.s3_bucket = os.environ.get("S3_BUCKET", "")

        self.client = boto3.client("batch", region_name=self.region)

    def submit_job(
        self,
        benchmark_id: str,
        s3_task_path: str,
        total_runs: int = 10,
        n_concurrent_trials: int = 10,
        model_name: str | None = None,
        agent_name: str | None = None,
        harness: str = "harbor",
    ) -> str:
        """Submit a benchmark job to AWS Batch.

        This submits a single job that runs all trials concurrently using
        the specified harness's concurrency feature.

        Args:
            benchmark_id: ID of the parent benchmark.
            s3_task_path: S3 path to the task files.
            total_runs: Total number of trials to run (default: 10).
            n_concurrent_trials: Number of trials to run concurrently (default: 10).
            model_name: Optional model name override.
            agent_name: Optional agent name override.
            harness: Harness to use ('harbor' or 'terminus').

        Returns:
            AWS Batch job ID.
        """
        job_name = f"benchmark-{benchmark_id[:8]}"

        environment = [
            {"name": "BENCHMARK_ID", "value": benchmark_id},
            {"name": "S3_TASK_PATH", "value": s3_task_path},
            {"name": "TOTAL_RUNS", "value": str(total_runs)},
            {"name": "N_CONCURRENT_TRIALS", "value": str(n_concurrent_trials)},
            {"name": "S3_BUCKET", "value": self.s3_bucket},
            {"name": "AWS_REGION", "value": self.region},
            {"name": "HARNESS", "value": harness},
        ]

        if model_name:
            environment.append({"name": "MODEL_NAME", "value": model_name})
        if agent_name:
            environment.append({"name": "AGENT_NAME", "value": agent_name})

        # Get OpenRouter API key from environment
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            environment.append({"name": "OPENROUTER_API_KEY", "value": openrouter_key})

        try:
            response = self.client.submit_job(
                jobName=job_name,
                jobQueue=self.job_queue,
                jobDefinition=self.job_definition,
                containerOverrides={
                    "environment": environment,
                },
            )
            return response["jobId"]
        except ClientError as e:
            raise RuntimeError(f"Failed to submit batch job: {e}") from e

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a batch job.

        Args:
            job_id: AWS Batch job ID.

        Returns:
            Job details dictionary or None if not found.
        """
        try:
            response = self.client.describe_jobs(jobs=[job_id])
            if response.get("jobs"):
                return response["jobs"][0]
            return None
        except ClientError:
            return None

    def get_jobs_status(self, job_ids: list[str]) -> list[dict[str, Any]]:
        """Get status of multiple batch jobs.

        Args:
            job_ids: List of AWS Batch job IDs.

        Returns:
            List of job details dictionaries.
        """
        if not job_ids:
            return []

        try:
            # AWS Batch allows up to 100 jobs per describe call
            all_jobs = []
            for i in range(0, len(job_ids), 100):
                batch = job_ids[i : i + 100]
                response = self.client.describe_jobs(jobs=batch)
                all_jobs.extend(response.get("jobs", []))
            return all_jobs
        except ClientError:
            return []

    def cancel_job(self, job_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a batch job.

        Args:
            job_id: AWS Batch job ID.
            reason: Reason for cancellation.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.cancel_job(jobId=job_id, reason=reason)
            return True
        except ClientError:
            return False

    def terminate_job(self, job_id: str, reason: str = "Terminated by user") -> bool:
        """Terminate a batch job.

        Args:
            job_id: AWS Batch job ID.
            reason: Reason for termination.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.terminate_job(jobId=job_id, reason=reason)
            return True
        except ClientError:
            return False
