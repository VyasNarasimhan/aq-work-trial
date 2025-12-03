import uuid
import zipfile
from flask import Blueprint, request, jsonify, current_app
from pathlib import Path

from app.utils.task_format import find_task_dir

bp = Blueprint("uploads", __name__, url_prefix="/api")


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload a task zip file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "File must be a .zip file"}), 400

    # Generate unique upload ID
    upload_id = str(uuid.uuid4())

    # Save the zip file
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    zip_path = upload_folder / f"{upload_id}.zip"
    file.save(zip_path)

    # Extract the zip to get task name
    extract_dir = upload_folder / upload_id
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the task directory and detect format (might be nested)
        task_path, detected_format = find_task_dir(extract_dir)
        if not task_path:
            return jsonify({"error": "Invalid task format - no task.toml or task.yaml found"}), 400

        task_name = task_path.name

        return jsonify({
            "upload_id": upload_id,
            "task_name": task_name,
            "task_path": str(task_path),
            "detected_format": detected_format,
        })

    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        return jsonify({"error": "Invalid zip file"}), 400


def _find_task_dir(extract_dir: Path) -> Path | None:
    """Find the task directory containing task.toml."""
    # Check if task.toml is in the root
    if (extract_dir / "task.toml").exists():
        return extract_dir

    # Check one level deep (common case: zip contains a folder)
    for subdir in extract_dir.iterdir():
        if subdir.is_dir() and (subdir / "task.toml").exists():
            return subdir

    # Check two levels deep
    for subdir in extract_dir.iterdir():
        if subdir.is_dir():
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir() and (subsubdir / "task.toml").exists():
                    return subsubdir

    return None
