import uuid
import json
from flask import Blueprint, request, jsonify, current_app

from app.utils.task_format import find_task_dir, validate_harness_format
import logging

logger = logging.getLogger(__name__)

bp = Blueprint("benchmarks", __name__, url_prefix="/api/benchmarks")

# Terminus and model selection disabled for now
VALID_HARNESSES = {"harbor"}
DEFAULT_HARNESS = "harbor"
DEFAULT_MODEL = "openrouter/openai/gpt-5"


@bp.route("", methods=["POST"])
def create_benchmark():
    """Start a new benchmark from an uploaded zip."""
    data = request.json or {}
    upload_id = data.get("upload_id")
    # Harness and model selection disabled - use defaults
    harness = DEFAULT_HARNESS
    model = DEFAULT_MODEL

    if not upload_id:
        return jsonify({"error": "upload_id required"}), 400

    logger.info(f"Creating benchmark with harness={harness}, model={model}")

    manager = current_app.benchmark_manager
    upload_folder = current_app.config["UPLOAD_FOLDER"]

    # Find the extracted task directory
    extract_dir = upload_folder / upload_id
    if not extract_dir.exists():
        return jsonify({"error": "Upload not found"}), 404

    # Find task path and validate format matches harness
    task_path, detected_format = find_task_dir(extract_dir)
    if not task_path:
        return jsonify({"error": "Invalid task format - no task.toml or task.yaml found"}), 400

    # Validate harness matches task format
    validation_error = validate_harness_format(harness, detected_format)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    # Create benchmark with harness and model
    benchmark_id = str(uuid.uuid4())
    benchmark = manager.start_benchmark(benchmark_id, task_path, harness=harness, model=model)

    logger.info(f"Benchmark {benchmark_id} started with model={model}")

    return jsonify({
        "id": benchmark["id"],
        "status": benchmark["status"],
        "task_name": benchmark["task_name"],
        "harness": benchmark["harness"],
        "model": benchmark.get("model", DEFAULT_MODEL),
    }), 201


@bp.route("", methods=["GET"])
def list_benchmarks():
    """List all benchmarks."""
    manager = current_app.benchmark_manager
    benchmarks = manager.list_benchmarks()
    return jsonify(benchmarks)


@bp.route("/<benchmark_id>", methods=["GET"])
def get_benchmark(benchmark_id):
    """Get benchmark status and summary."""
    manager = current_app.benchmark_manager
    benchmark = manager.get_benchmark(benchmark_id)

    if not benchmark:
        return jsonify({"error": "Benchmark not found"}), 404

    return jsonify(benchmark)


@bp.route("/<benchmark_id>/runs", methods=["GET"])
def get_runs(benchmark_id):
    """Get all runs for a benchmark."""
    manager = current_app.benchmark_manager
    runs = manager.get_runs(benchmark_id)

    if runs is None:
        return jsonify({"error": "Benchmark not found"}), 404

    return jsonify(runs)


@bp.route("/<benchmark_id>/runs/<run_id>/logs", methods=["GET"])
def get_run_logs(benchmark_id, run_id):
    """Get logs for a specific run."""
    manager = current_app.benchmark_manager
    logs = manager.get_run_logs(benchmark_id, run_id)

    if logs is None:
        return jsonify({"error": "Run not found"}), 404

    return jsonify(logs)


