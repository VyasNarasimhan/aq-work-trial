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
