"""AWS S3 client for uploading tasks and downloading results."""

import json
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


class S3Client:
    """Client for interacting with AWS S3."""

    def __init__(self, bucket: str | None = None, region: str | None = None):
        """Initialize the S3 client.

        Args:
            bucket: S3 bucket name. Defaults to S3_BUCKET env var.
            region: AWS region. Defaults to AWS_REGION env var or us-west-2.
        """
        self.region = region or os.environ.get("AWS_REGION", "us-west-2")
        self.bucket = bucket or os.environ.get("S3_BUCKET", "")

        if not self.bucket:
            raise ValueError("S3 bucket name is required")

        self.client = boto3.client("s3", region_name=self.region)

    def upload_task(self, upload_id: str, local_path: Path) -> str:
        """Upload task directory to S3.

        Args:
            upload_id: Unique identifier for the upload.
            local_path: Local path to the task directory.

        Returns:
            S3 URI for the uploaded task.
        """
        s3_prefix = f"tasks/{upload_id}/"

        if not local_path.exists():
            raise FileNotFoundError(f"Task directory not found: {local_path}")

        # Upload all files in the directory
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}{relative_path}"

                try:
                    self.client.upload_file(str(file_path), self.bucket, s3_key)
                except ClientError as e:
                    raise RuntimeError(f"Failed to upload {file_path}: {e}") from e

        return f"s3://{self.bucket}/{s3_prefix}"

    def get_run_status(self, benchmark_id: str, run_number: int) -> dict[str, Any] | None:
        """Get the status of a run from S3.

        Args:
            benchmark_id: ID of the benchmark.
            run_number: Run number.

        Returns:
            Status dictionary or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-{run_number}/status.json"

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def get_run_result(self, benchmark_id: str, run_number: int, file_path: str) -> bytes | None:
        """Get a specific file from a run's results.

        Args:
            benchmark_id: ID of the benchmark.
            run_number: Run number.
            file_path: Relative path to the file within the run results.

        Returns:
            File contents as bytes or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-{run_number}/{file_path}"

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def list_run_files(self, benchmark_id: str, run_number: int) -> list[str]:
        """List all files in a run's results.

        Args:
            benchmark_id: ID of the benchmark.
            run_number: Run number.

        Returns:
            List of S3 keys for the run's files.
        """
        s3_prefix = f"results/{benchmark_id}/run-{run_number}/"

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            files = []

            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    # Get relative path from prefix
                    relative_path = obj["Key"][len(s3_prefix) :]
                    if relative_path:
                        files.append(relative_path)

            return files
        except ClientError:
            return []

    def download_run_results(self, benchmark_id: str, run_number: int, local_dir: Path) -> bool:
        """Download all results for a run to a local directory.

        Args:
            benchmark_id: ID of the benchmark.
            run_number: Run number.
            local_dir: Local directory to download to.

        Returns:
            True if successful, False otherwise.
        """
        s3_prefix = f"results/{benchmark_id}/run-{run_number}/"

        try:
            paginator = self.client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix) :]
                    if not relative_path:
                        continue

                    local_path = local_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    self.client.download_file(self.bucket, key, str(local_path))

            return True
        except ClientError:
            return False

    def check_results_exist(self, benchmark_id: str, run_number: int) -> bool:
        """Check if results exist for a run.

        Args:
            benchmark_id: ID of the benchmark.
            run_number: Run number.

        Returns:
            True if results exist, False otherwise.
        """
        s3_key = f"results/{benchmark_id}/run-{run_number}/status.json"

        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def upload_json(self, s3_key: str, data: dict[str, Any]) -> None:
        """Upload JSON data to S3.

        Args:
            s3_key: S3 key for the JSON file.
            data: Dictionary to serialize as JSON.
        """
        self.client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json.dumps(data, indent=2, default=str),
            ContentType="application/json",
        )

    # =========================================================================
    # Terminus-specific methods
    # Terminus stores all trials under run-1/task/task.X-of-10.../
    # rather than separate run-{N}/ folders like harbor
    # =========================================================================

    def is_terminus_structure(self, benchmark_id: str) -> bool:
        """Check if the benchmark results use terminus structure.

        Terminus stores all trials under run-1/task/task.X-of-10.../
        and has a results.json with all trial results.

        Args:
            benchmark_id: ID of the benchmark.

        Returns:
            True if terminus structure detected, False otherwise.
        """
        s3_key = f"results/{benchmark_id}/run-1/results.json"
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_terminus_results(self, benchmark_id: str) -> list[dict[str, Any]] | None:
        """Get the main results.json from terminus benchmark.

        This file contains an array of all trial results with is_resolved,
        parser_results, trial_name, timestamps, etc.

        Args:
            benchmark_id: ID of the benchmark.

        Returns:
            List of trial result dictionaries or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-1/results.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def get_terminus_metadata(self, benchmark_id: str) -> dict[str, Any] | None:
        """Get the run_metadata.json from terminus benchmark.

        Contains accuracy, model_name, pass_at_k, n_attempts, etc.

        Args:
            benchmark_id: ID of the benchmark.

        Returns:
            Metadata dictionary or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-1/run_metadata.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def get_terminus_run1_status(self, benchmark_id: str) -> dict[str, Any] | None:
        """Get the status.json from terminus run-1.

        Args:
            benchmark_id: ID of the benchmark.

        Returns:
            Status dictionary or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-1/status.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def list_terminus_trial_names(self, benchmark_id: str) -> list[str]:
        """List all trial directory names for a terminus benchmark.

        tb CLI structure: results/{benchmark_id}/run-1/{task_id}/{trial_name}/
        Each task_id folder contains one trial folder.

        Args:
            benchmark_id: ID of the benchmark.

        Returns:
            List of (task_id, trial_name) tuples as "{task_id}/{trial_name}" strings,
            sorted by trial number extracted from the trial_name.
        """
        s3_prefix = f"results/{benchmark_id}/run-1/"
        trials = set()

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(s3_prefix):]
                    parts = relative.split("/")
                    # Structure: {task_id}/{trial_name}/...
                    # Skip files directly under run-1/ (like results.json)
                    if len(parts) >= 3:
                        task_id = parts[0]
                        trial_name = parts[1]
                        # Skip if task_id looks like a file (contains '.')
                        if "." not in task_id or task_id.endswith("-1") or task_id.endswith("-10"):
                            trials.add(f"{task_id}/{trial_name}")

            # Sort by trial number extracted from trial_name
            # trial_name format: {task_id}.1-of-1.{timestamp}
            def get_trial_num(path: str) -> int:
                try:
                    trial_name = path.split("/")[1]
                    # Extract number after first dot: task-name-1.1-of-1.timestamp
                    parts = trial_name.split(".")
                    if len(parts) >= 2:
                        return int(parts[1].split("-")[0])
                    return 999
                except (IndexError, ValueError):
                    return 999
            return sorted(trials, key=get_trial_num)
        except ClientError:
            return []

    def list_terminus_trial_files(self, benchmark_id: str, trial_path: str) -> list[str]:
        """List all files in a terminus trial directory.

        Args:
            benchmark_id: ID of the benchmark.
            trial_path: Trial path as "{task_id}/{trial_name}".

        Returns:
            List of relative file paths within the trial directory.
        """
        s3_prefix = f"results/{benchmark_id}/run-1/{trial_path}/"
        files = []

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    relative_path = obj["Key"][len(s3_prefix):]
                    if relative_path:
                        files.append(relative_path)
            return files
        except ClientError:
            return []

    def get_terminus_trial_file(self, benchmark_id: str, trial_path: str, file_path: str) -> bytes | None:
        """Get a specific file from a terminus trial.

        Args:
            benchmark_id: ID of the benchmark.
            trial_path: Trial path as "{task_id}/{trial_name}".
            file_path: Relative path within the trial directory.

        Returns:
            File contents as bytes or None if not found.
        """
        s3_key = f"results/{benchmark_id}/run-1/{trial_path}/{file_path}"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def download_terminus_trial(self, benchmark_id: str, trial_path: str, local_dir: Path) -> bool:
        """Download all files for a terminus trial to local directory.

        Args:
            benchmark_id: ID of the benchmark.
            trial_path: Trial path as "{task_id}/{trial_name}".
            local_dir: Local directory to download to.

        Returns:
            True if successful, False otherwise.
        """
        s3_prefix = f"results/{benchmark_id}/run-1/{trial_path}/"

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix):]
                    if not relative_path:
                        continue

                    local_path = local_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.client.download_file(self.bucket, key, str(local_path))
            return True
        except ClientError:
            return False
