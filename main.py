"""
main.py
───────
Command-line entry point for the RegRadar policy diff engine.

Usage
─────
    py main.py <old_policy_file> <new_policy_file> [--output <json_path>]

Example
───────
    py main.py data/policies/data_retention_v1.txt \\
               data/policies/data_retention_v2.txt

The version tag for each document is inferred from the filename stem
(e.g., "data_retention_v1" for data_retention_v1.txt).  You can override
this by editing the load_document() calls below once you have real version
metadata to use.

argparse
────────
Python's built-in `argparse` module parses command-line arguments and
generates a --help message automatically.  No third-party library needed.

Python docs:  https://docs.python.org/3/library/argparse.html
"""

# Note: `from __future__ import annotations` (PEP 563) is intentionally
# absent — see regradar/models.py for the rationale.
import argparse
import sys
from pathlib import Path

from regradar.differ import diff_documents
from regradar.loader import load_document
from regradar.reporter import print_report, write_json_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regradar",
        description=(
            "RegRadar AI — Compare two versions of a policy document and "
            "report what changed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  py main.py data/policies/data_retention_v1.txt "
            "data/policies/data_retention_v2.txt"
        ),
    )
    parser.add_argument(
        "old_file",
        type=Path,
        help="Path to the baseline (old) policy file.",
    )
    parser.add_argument(
        "new_file",
        type=Path,
        help="Path to the updated (new) policy file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/diff_report.json"),
        metavar="JSON_PATH",
        help=(
            "Where to write the JSON report.  "
            "Default: reports/diff_report.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the diff pipeline.  Returns 0 on success, 1 on error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Validate inputs ───────────────────────────────────────────────────────
    for path in (args.old_file, args.new_file):
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1

    # ── Load documents ────────────────────────────────────────────────────────
    # Use the filename stem as the version tag (e.g. "data_retention_v1").
    # In a real project this would come from a metadata file or CLI flag.
    try:
        old_doc = load_document(args.old_file, version_tag=args.old_file.stem)
        new_doc = load_document(args.new_file, version_tag=args.new_file.stem)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading document: {exc}", file=sys.stderr)
        return 1

    # ── Diff ──────────────────────────────────────────────────────────────────
    result = diff_documents(old_doc, new_doc)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(result)

    json_path = write_json_report(result, args.output)
    print(f"JSON report written to: {json_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
