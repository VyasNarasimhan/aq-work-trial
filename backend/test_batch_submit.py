#!/usr/bin/env python3
"""Test script to submit a single job to AWS Batch."""

import os
import sys
import uuid

# Set up environment
os.environ.setdefault("S3_BUCKET", "aq-benchmark-jobs-956152913627")
os.environ.setdefault("AWS_BATCH_JOB_QUEUE", "aq-benchmark-queue")
os.environ.setdefault("AWS_BATCH_JOB_DEF", "aq-benchmark-runner")
os.environ.setdefault("AWS_REGION", "us-west-2")

from pathlib import Path

# Add the app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.aws import BatchClient, S3Client


def test_batch_submit():
    """Test submitting a single job to AWS Batch."""
    print("=" * 60)
    print("Testing AWS Batch Job Submission")
    print("=" * 60)

    # Initialize clients
    print("\n1. Initializing AWS clients...")
    batch_client = BatchClient()
    s3_client = S3Client()
    print(f"   S3 Bucket: {s3_client.bucket}")
    print(f"   Batch Queue: {batch_client.job_queue}")
    print(f"   Batch Job Def: {batch_client.job_definition}")

    # Use an existing task for testing
    task_path = Path("/Users/vyas/code/aq/backend/uploads/00523222-5975-48f8-92ef-2cef0b149685/adaptive-rejection-sampler")

    if not task_path.exists():
        print(f"\nError: Task path not found: {task_path}")
        print("Looking for alternative task...")
        # Find any existing task
        uploads_dir = Path("/Users/vyas/code/aq/backend/uploads")
        for upload_dir in uploads_dir.iterdir():
            if upload_dir.is_dir():
                for task_dir in upload_dir.iterdir():
                    if (task_dir / "task.toml").exists():
                        task_path = task_dir
                        break
                if task_path.exists():
                    break

    print(f"\n2. Using task: {task_path}")

    # Generate test IDs
    benchmark_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    run_number = 1

    print(f"   Benchmark ID: {benchmark_id}")
    print(f"   Run ID: {run_id}")
    print(f"   Run Number: {run_number}")

    # Upload task to S3
    print("\n3. Uploading task to S3...")
    s3_task_path = s3_client.upload_task(benchmark_id, task_path)
    print(f"   Uploaded to: {s3_task_path}")

    # Submit job to AWS Batch
    print("\n4. Submitting job to AWS Batch...")
    job_id = batch_client.submit_job(
        run_id=run_id,
        benchmark_id=benchmark_id,
        s3_task_path=s3_task_path,
        run_number=run_number,
    )
    print(f"   Job ID: {job_id}")

    # Check job status
    print("\n5. Checking job status...")
    status = batch_client.get_job_status(job_id)
    if status:
        print(f"   Status: {status.get('status')}")
        print(f"   Status Reason: {status.get('statusReason', 'N/A')}")
    else:
        print("   Could not retrieve status")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Monitor job in AWS Console:")
    print(f"  https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2#jobs")
    print("=" * 60)

    return job_id


if __name__ == "__main__":
    job_id = test_batch_submit()
