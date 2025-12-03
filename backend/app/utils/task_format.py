"""Utilities for detecting and validating task formats."""

from pathlib import Path
from typing import Literal, Tuple

TaskFormat = Literal["harbor", "terminus", "unknown"]


def detect_task_format(task_dir: Path) -> TaskFormat:
    """Detect the format of a task directory.

    Args:
        task_dir: Path to the task directory.

    Returns:
        'harbor' if task.toml exists, 'terminus' if task.yaml exists,
        'unknown' otherwise.
    """
    if (task_dir / "task.toml").exists():
        return "harbor"
    if (task_dir / "task.yaml").exists():
        return "terminus"
    return "unknown"


def find_task_dir(extract_dir: Path) -> Tuple[Path | None, TaskFormat]:
    """Find task directory within an extracted zip and detect its format.

    Searches up to 2 levels deep for task.toml or task.yaml files.

    Args:
        extract_dir: Root directory of extracted zip file.

    Returns:
        Tuple of (task_path, format). task_path is None if no task found.
    """

    def check_dir(d: Path) -> Tuple[Path | None, TaskFormat]:
        fmt = detect_task_format(d)
        if fmt != "unknown":
            return d, fmt
        return None, "unknown"

    # Check root
    result, fmt = check_dir(extract_dir)
    if result:
        return result, fmt

    # Check one level deep
    for subdir in extract_dir.iterdir():
        if subdir.is_dir():
            result, fmt = check_dir(subdir)
            if result:
                return result, fmt

    # Check two levels deep
    for subdir in extract_dir.iterdir():
        if subdir.is_dir():
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    result, fmt = check_dir(subsubdir)
                    if result:
                        return result, fmt

    return None, "unknown"


def validate_harness_format(harness: str, detected_format: TaskFormat) -> str | None:
    """Validate that harness matches detected task format.

    Args:
        harness: Requested harness ('harbor' or 'terminus').
        detected_format: Detected task format.

    Returns:
        Error message if validation fails, None if valid.
    """
    if harness == "harbor" and detected_format == "terminus":
        return "Harbor harness requires task.toml file. Detected: task.yaml (Terminus format)"
    if harness == "terminus" and detected_format == "harbor":
        return "Terminus harness requires task.yaml file. Detected: task.toml (Harbor format)"
    return None
