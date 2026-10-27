"""
regradar/reporter.py
────────────────────
Renders a DiffResult to the terminal and to a JSON file.

Two output channels
───────────────────
1. Terminal (stdout): Human-readable, colour-coded via ANSI escape codes.
   Falls back gracefully to plain text if the terminal does not support
   colour (detected via sys.stdout.isatty()).

2. JSON file: Machine-readable report written to a configurable path.
   Uses dataclasses.asdict() to convert the DiffResult — and all nested
   dataclasses — into plain Python dicts, then json.dumps() to serialise.
   No third-party library needed.

ANSI colour codes
─────────────────
Terminals that support colour interpret escape sequences like \\033[32m as
"switch to green".  \\033[0m resets to the default colour.  When stdout is
redirected to a file (not a TTY), we skip these codes so the file stays
readable.

Python docs:  https://docs.python.org/3/library/json.html
              https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from regradar.models import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    DiffResult,
    SectionChange,
)


# ── ANSI colour helpers ───────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Return True if stdout is an interactive terminal that likely supports ANSI colours."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _coloured(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code if colours are supported."""
    if _supports_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def _green(text: str) -> str:
    return _coloured(text, "32")


def _red(text: str) -> str:
    return _coloured(text, "31")


def _yellow(text: str) -> str:
    return _coloured(text, "33")


def _cyan(text: str) -> str:
    return _coloured(text, "36")


def _bold(text: str) -> str:
    return _coloured(text, "1")


# ── Terminal reporter ─────────────────────────────────────────────────────────

# Maps each change type to a label and a colour function.
_CHANGE_STYLE: dict[str, tuple[str, callable]] = {
    CHANGE_ADDED:     ("[ADDED]    ", _green),
    CHANGE_REMOVED:   ("[REMOVED]  ", _red),
    CHANGE_MODIFIED:  ("[MODIFIED] ", _yellow),
    CHANGE_UNCHANGED: ("[UNCHANGED]", _cyan),
}


def print_report(result: DiffResult) -> None:
    """Print a human-readable diff report to stdout."""
    summary = result.summary

    print()
    print(_bold("=" * 60))
    print(_bold("  RegRadar Policy Diff Report"))
    print(_bold("=" * 60))
    print(f"  {result.old_version_tag}  →  {result.new_version_tag}")
    print()

    for change in result.changes:
        _print_section_change(change)

    # Summary footer.
    print(_bold("─" * 60))
    print("  Summary:")
    print(f"    {_green(str(summary[CHANGE_ADDED]))} added   "
          f"  {_red(str(summary[CHANGE_REMOVED]))} removed   "
          f"  {_yellow(str(summary[CHANGE_MODIFIED]))} modified   "
          f"  {_cyan(str(summary[CHANGE_UNCHANGED]))} unchanged")
    print(_bold("=" * 60))
    print()


def _print_section_change(change: SectionChange) -> None:
    """Print one section's change block to stdout."""
    label, colour_fn = _CHANGE_STYLE.get(
        change.change_type,
        (f"[{change.change_type.upper()}]", str),
    )

    print(colour_fn(f"{label}  {change.heading}"))

    if change.change_type == CHANGE_ADDED:
        for line in (change.new_text or "").splitlines():
            print(_green(f"  + {line}"))

    elif change.change_type == CHANGE_REMOVED:
        for line in (change.old_text or "").splitlines():
            print(_red(f"  - {line}"))

    elif change.change_type == CHANGE_MODIFIED:
        # Print only the +/- lines from the unified diff; skip the header lines
        # (--- and +++) and context lines (starting with ' ') for compactness.
        for line in change.diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                print(_green(f"  {line}"))
            elif line.startswith("-") and not line.startswith("---"):
                print(_red(f"  {line}"))

    # UNCHANGED sections: no body printed — the label is enough.

    print()  # Blank line between sections for readability.


# ── JSON reporter ─────────────────────────────────────────────────────────────

def write_json_report(result: DiffResult, output_path: str | Path) -> Path:
    """Serialise a DiffResult to a JSON file.

    Uses dataclasses.asdict() to recursively convert the entire dataclass
    hierarchy (DiffResult → [SectionChange]) to plain dicts, then
    json.dumps() to produce the JSON string.

    Args:
        result:      The DiffResult to serialise.
        output_path: Destination file path (created, along with any missing
                     parent directories).

    Returns:
        The resolved Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # dataclasses.asdict() handles nested dataclasses recursively.
    # The result is a plain dict that json.dumps understands natively.
    data = dataclasses.asdict(result)

    # indent=2 produces readable JSON; ensure_ascii=False preserves any
    # non-ASCII characters in policy text (e.g., accented letters, em-dashes).
    json_text = json.dumps(data, indent=2, ensure_ascii=False)

    output_path.write_text(json_text, encoding="utf-8")
    return output_path
