#!/usr/bin/env python3
"""Convert T-Bench 1 (Terminus) tasks to Harbor format.

Usage:
    # Single task conversion
    python convert_tbench1_to_harbor.py /path/to/tbench1_task /path/to/output

    # Batch conversion
    python convert_tbench1_to_harbor.py --batch /path/to/tasks_dir /path/to/output_dir

    # Dry run (preview changes without writing)
    python convert_tbench1_to_harbor.py --dry-run /path/to/task /path/to/output
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def generate_toml_manually(data: dict) -> str:
    """Generate TOML string manually without tomli_w dependency."""
    lines = []

    # Top-level scalar
    if "version" in data:
        lines.append(f'version = "{data["version"]}"')
        lines.append("")

    # [metadata] section
    if "metadata" in data:
        lines.append("[metadata]")
        meta = data["metadata"]
        if "author_name" in meta:
            lines.append(f'author_name = "{meta["author_name"]}"')
        if "author_email" in meta:
            lines.append(f'author_email = "{meta["author_email"]}"')
        if "difficulty" in meta:
            lines.append(f'difficulty = "{meta["difficulty"]}"')
        if "category" in meta:
            lines.append(f'category = "{meta["category"]}"')
        if "tags" in meta:
            tags_str = ", ".join(f'"{t}"' for t in meta["tags"])
            lines.append(f"tags = [{tags_str}]")
        if "expert_time_estimate_min" in meta:
            lines.append(f'expert_time_estimate_min = {meta["expert_time_estimate_min"]}')
        if "junior_time_estimate_min" in meta:
            lines.append(f'junior_time_estimate_min = {meta["junior_time_estimate_min"]}')
        lines.append("")

    # [verifier] section
    if "verifier" in data:
        lines.append("[verifier]")
        verifier = data["verifier"]
        if "timeout_sec" in verifier:
            lines.append(f'timeout_sec = {verifier["timeout_sec"]}')
        lines.append("")

    # [agent] section
    if "agent" in data:
        lines.append("[agent]")
        agent = data["agent"]
        if "timeout_sec" in agent:
            lines.append(f'timeout_sec = {agent["timeout_sec"]}')
        lines.append("")

    # [environment] section
    if "environment" in data:
        lines.append("[environment]")
        env = data["environment"]
        if "build_timeout_sec" in env:
            lines.append(f'build_timeout_sec = {env["build_timeout_sec"]}')
        if "docker_image" in env:
            lines.append(f'docker_image = "{env["docker_image"]}"')
        if "cpus" in env:
            lines.append(f'cpus = {env["cpus"]}')
        if "memory" in env:
            lines.append(f'memory = "{env["memory"]}"')
        if "storage" in env:
            lines.append(f'storage = "{env["storage"]}"')
        lines.append("")

    return "\n".join(lines)


def parse_task_yaml(yaml_path: Path) -> dict[str, Any]:
    """Parse T-Bench 1 task.yaml file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def convert_to_task_toml(task_yaml: dict[str, Any], has_dockerfile: bool = False) -> dict[str, Any]:
    """Convert task.yaml fields to task.toml structure."""
    # Estimate time based on difficulty if not provided
    difficulty_time_map = {
        "easy": (15.0, 60.0),
        "medium": (30.0, 120.0),
        "hard": (60.0, 240.0),
    }

    difficulty = task_yaml.get("difficulty", "medium")
    default_expert, default_junior = difficulty_time_map.get(difficulty, (30.0, 120.0))

    environment = {
        "build_timeout_sec": 600.0,
        "cpus": 1,
        "memory": "2G",
        "storage": "10G",
    }

    # Only include docker_image if there's no Dockerfile
    if not has_dockerfile:
        environment["docker_image"] = ""

    task_toml = {
        "version": "1.0",
        "metadata": {
            "author_name": task_yaml.get("author_name", "Unknown"),
            "author_email": task_yaml.get("author_email", "unknown@example.com"),
            "difficulty": difficulty,
            "category": task_yaml.get("category", "general"),
            "tags": task_yaml.get("tags", []),
            "expert_time_estimate_min": float(task_yaml.get("expert_time_estimate_min", default_expert)),
            "junior_time_estimate_min": float(task_yaml.get("junior_time_estimate_min", default_junior)),
        },
        "verifier": {
            "timeout_sec": float(task_yaml.get("max_test_timeout_sec", 180.0)),
        },
        "agent": {
            "timeout_sec": float(task_yaml.get("max_agent_timeout_sec", 900.0)),
        },
        "environment": environment,
    }

    return task_toml


def extract_instruction(task_yaml: dict[str, Any]) -> str:
    """Extract instruction text from task.yaml."""
    # T-Bench 1 format uses 'instruction' directly
    if "instruction" in task_yaml:
        return task_yaml["instruction"]

    # Alternative: 'descriptions' array format
    if "descriptions" in task_yaml:
        descriptions = task_yaml["descriptions"]
        if descriptions and isinstance(descriptions, list):
            first_desc = descriptions[0]
            if isinstance(first_desc, dict):
                return first_desc.get("description", "")
            return str(first_desc)

    return ""


def convert_solution_yaml_to_bash(yaml_path: Path) -> str:
    """Convert solution.yaml commands to bash script."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    lines = ["#!/bin/bash", "set -e", ""]

    if isinstance(data, list):
        for cmd in data:
            if isinstance(cmd, dict):
                lines.append(cmd.get("command", ""))
            else:
                lines.append(str(cmd))
    elif isinstance(data, dict):
        if "commands" in data:
            for cmd in data["commands"]:
                if isinstance(cmd, dict):
                    lines.append(cmd.get("command", ""))
                else:
                    lines.append(str(cmd))

    return "\n".join(lines) + "\n"


def generate_test_sh(input_dir: Path, parser_name: str = "pytest") -> str:
    """Generate test.sh wrapper script.

    Note: We intentionally don't extract setup commands from run-tests.sh
    because it breaks shell syntax (e.g., extracting partial if/fi blocks).
    Instead, we only extract explicit dependencies.
    """
    import re

    extra_deps = []  # Dependencies to add to uvx

    # Check run-tests.sh for explicit dependencies (uv add package==version)
    run_tests_path = input_dir / "run-tests.sh"
    if run_tests_path.exists():
        with open(run_tests_path) as f:
            content = f.read()
            # Match: uv add package==version (but not uv add "$dep" style)
            for match in re.finditer(r'uv add\s+([a-zA-Z0-9_-]+[=<>]+[0-9.]+)', content):
                dep = match.group(1)
                pkg_name = dep.split("==")[0].split(">=")[0].split("<")[0]
                if pkg_name not in ("pytest", "pytest-json-ctrf"):
                    extra_deps.append(dep)

    # Check requirements.txt for additional dependencies
    requirements_path = input_dir / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    pkg_name = line.split("==")[0].split(">=")[0].split("<")[0]
                    if pkg_name not in ("pytest", "pytest-json-ctrf"):
                        # Only add if not already in extra_deps
                        if not any(d.startswith(pkg_name) for d in extra_deps):
                            extra_deps.append(line)

    # Build uvx command with dependencies
    uvx_deps = ["pytest==8.4.1", "pytest-json-ctrf==0.3.5"] + extra_deps
    deps_lines = " \\\n  ".join(f"-w {dep}" for dep in uvx_deps)

    script = f'''#!/bin/bash
# Auto-generated by T-Bench 1 to Harbor converter

# Install curl
apt-get update
apt-get install -y curl

# Install uv (pinned version for -w flag support)
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env

# Run pytest with CTRF output
# Using Python 3.12 for better wheel compatibility (e.g., psycopg2-binary)
uvx \\
  -p 3.12 \\
  {deps_lines} \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
'''

    return script


def convert_task(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> bool:
    """Convert a single T-Bench 1 task to Harbor format.

    Returns True if conversion succeeded, False otherwise.
    """
    task_yaml_path = input_dir / "task.yaml"

    if not task_yaml_path.exists():
        print(f"Error: No task.yaml found in {input_dir}", file=sys.stderr)
        return False

    if verbose:
        print(f"Converting: {input_dir.name}")

    # Parse task.yaml
    task_yaml = parse_task_yaml(task_yaml_path)

    # Check if Dockerfile exists
    has_dockerfile = (input_dir / "Dockerfile").exists()

    # Convert to Harbor structures
    task_toml_data = convert_to_task_toml(task_yaml, has_dockerfile=has_dockerfile)
    instruction = extract_instruction(task_yaml)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"Task: {input_dir.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        print("\n[task.toml]")
        print(generate_toml_manually(task_toml_data))
        print("\n[instruction.md]")
        print(instruction[:500] + "..." if len(instruction) > 500 else instruction)
        print("\n[Files to copy/create]")
        print("  - environment/Dockerfile")
        print("  - solution/solve.sh")
        print("  - tests/test.sh")
        print("  - tests/* (test files)")
        return True

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment").mkdir(exist_ok=True)
    (output_dir / "solution").mkdir(exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)

    # Write task.toml
    task_toml_path = output_dir / "task.toml"
    if tomli_w:
        with open(task_toml_path, "wb") as f:
            tomli_w.dump(task_toml_data, f)
    else:
        with open(task_toml_path, "w") as f:
            f.write(generate_toml_manually(task_toml_data))

    if verbose:
        print(f"  Created: task.toml")

    # Write instruction.md
    instruction_path = output_dir / "instruction.md"
    with open(instruction_path, "w") as f:
        f.write(instruction + "\n")

    if verbose:
        print(f"  Created: instruction.md")

    # Copy/create Dockerfile
    dockerfile_src = input_dir / "Dockerfile"
    dockerfile_dst = output_dir / "environment" / "Dockerfile"
    if dockerfile_src.exists():
        shutil.copy2(dockerfile_src, dockerfile_dst)
        if verbose:
            print(f"  Copied: environment/Dockerfile")
    else:
        # Check docker-compose.yaml for image info
        compose_path = input_dir / "docker-compose.yaml"
        if compose_path.exists():
            with open(compose_path) as f:
                compose = yaml.safe_load(f)
            # Try to extract image from compose file
            if compose and "services" in compose:
                for service in compose["services"].values():
                    if "image" in service:
                        task_toml_data["environment"]["docker_image"] = service["image"]
                        # Rewrite task.toml with updated image
                        if tomli_w:
                            with open(task_toml_path, "wb") as f:
                                tomli_w.dump(task_toml_data, f)
                        else:
                            with open(task_toml_path, "w") as f:
                                f.write(generate_toml_manually(task_toml_data))
                        break
            if verbose:
                print(f"  Note: No Dockerfile, extracted from docker-compose.yaml")

    # Copy/create solution
    solution_sh_src = input_dir / "solution.sh"
    solution_yaml_src = input_dir / "solution.yaml"
    solution_dst = output_dir / "solution" / "solve.sh"

    if solution_sh_src.exists():
        shutil.copy2(solution_sh_src, solution_dst)
        if verbose:
            print(f"  Copied: solution/solve.sh")
    elif solution_yaml_src.exists():
        bash_content = convert_solution_yaml_to_bash(solution_yaml_src)
        with open(solution_dst, "w") as f:
            f.write(bash_content)
        solution_dst.chmod(0o755)
        if verbose:
            print(f"  Converted: solution/solve.sh (from solution.yaml)")
    else:
        # Create placeholder
        with open(solution_dst, "w") as f:
            f.write("#!/bin/bash\n# No solution provided\nexit 0\n")
        solution_dst.chmod(0o755)
        if verbose:
            print(f"  Created: solution/solve.sh (placeholder)")

    # Generate test.sh
    parser_name = task_yaml.get("parser_name", "pytest")
    test_sh_content = generate_test_sh(input_dir, parser_name)
    test_sh_path = output_dir / "tests" / "test.sh"
    with open(test_sh_path, "w") as f:
        f.write(test_sh_content)
    test_sh_path.chmod(0o755)

    if verbose:
        print(f"  Created: tests/test.sh")

    # Copy test files
    tests_src = input_dir / "tests"
    if tests_src.exists() and tests_src.is_dir():
        for test_file in tests_src.iterdir():
            if test_file.is_file():
                dst = output_dir / "tests" / test_file.name
                shutil.copy2(test_file, dst)
                if verbose:
                    print(f"  Copied: tests/{test_file.name}")

    # Copy additional data files (anything not already handled)
    skip_files = {
        "task.yaml", "Dockerfile", "docker-compose.yaml",
        "solution.sh", "solution.yaml", "run-tests.sh"
    }
    skip_dirs = {"tests", ".git", "__pycache__"}

    for item in input_dir.iterdir():
        if item.name in skip_files or item.name in skip_dirs:
            continue
        if item.name.startswith("."):
            continue

        dst = output_dir / "environment" / item.name
        if item.is_file():
            shutil.copy2(item, dst)
            if verbose:
                print(f"  Copied: environment/{item.name}")
        elif item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
            if verbose:
                print(f"  Copied: environment/{item.name}/")

    if verbose:
        print(f"  Done!")

    return True


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> tuple[int, int]:
    """Convert multiple tasks in batch mode.

    Returns (success_count, failure_count).
    """
    success = 0
    failure = 0

    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "task.yaml").exists():
            continue

        task_output = output_dir / task_dir.name
        if convert_task(task_dir, task_output, dry_run, verbose):
            success += 1
        else:
            failure += 1

    return success, failure


def main():
    parser = argparse.ArgumentParser(
        description="Convert T-Bench 1 (Terminus) tasks to Harbor format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input T-Bench 1 task directory (or tasks directory with --batch)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output Harbor task directory"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all tasks in input directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.batch:
        success, failure = batch_convert(args.input, args.output, args.dry_run, args.verbose)
        print(f"\nBatch conversion complete: {success} succeeded, {failure} failed")
        sys.exit(0 if failure == 0 else 1)
    else:
        success = convert_task(args.input, args.output, args.dry_run, args.verbose)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
