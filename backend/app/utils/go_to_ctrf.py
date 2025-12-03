#!/usr/bin/env python3
"""Convert `go test -json` output to CTRF format.

Usage:
    go test -json ./... | python3 go_to_ctrf.py > /logs/verifier/ctrf.json

Or with a file:
    python3 go_to_ctrf.py input.json output.json

This script reads Go test JSON output (from `go test -json`) and converts it
to CTRF (Common Test Result Format) for integration with TerminalBench.
"""

import json
import sys


def convert_go_json_to_ctrf(input_stream, output_stream):
    """Convert Go test JSON output to CTRF format.

    Args:
        input_stream: File-like object with Go test JSON (one event per line)
        output_stream: File-like object to write CTRF JSON
    """
    tests = []
    passed = 0
    failed = 0

    for line in input_stream:
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = event.get("Action")
        test_name = event.get("Test")

        # Only process test-level pass/fail events (not package-level)
        if not test_name:
            continue

        if action == "pass":
            tests.append({
                "name": test_name,
                "status": "passed",
                "duration": (event.get("Elapsed", 0) or 0) * 1000,  # Convert to ms
            })
            passed += 1
        elif action == "fail":
            tests.append({
                "name": test_name,
                "status": "failed",
                "duration": (event.get("Elapsed", 0) or 0) * 1000,
            })
            failed += 1

    ctrf = {
        "results": {
            "tool": {"name": "go-test"},
            "summary": {
                "tests": len(tests),
                "passed": passed,
                "failed": failed,
                "skipped": 0,
                "pending": 0,
                "other": 0,
            },
            "tests": tests,
        }
    }

    json.dump(ctrf, output_stream, indent=2)
    output_stream.write("\n")


def main():
    if len(sys.argv) == 1:
        # Read from stdin, write to stdout
        convert_go_json_to_ctrf(sys.stdin, sys.stdout)
    elif len(sys.argv) == 3:
        # Read from input file, write to output file
        with open(sys.argv[1], "r") as infile:
            with open(sys.argv[2], "w") as outfile:
                convert_go_json_to_ctrf(infile, outfile)
    else:
        print("Usage: go test -json ./... | python3 go_to_ctrf.py", file=sys.stderr)
        print("   or: python3 go_to_ctrf.py input.json output.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
