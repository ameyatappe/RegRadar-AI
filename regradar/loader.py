"""
regradar/loader.py
──────────────────
Parses a plain-text policy file into a DocumentVersion.

Expected file format
────────────────────
Each section begins with a marker on its own line:

    [SECTION: Section Name]
    Body text goes here.
    It can span multiple lines.

    [SECTION: Next Section]
    ...

Rules the loader enforces:
  • Marker lines must match exactly:  [SECTION: <name>]
  • Text before the first marker is ignored with a warning.
  • Duplicate headings within the same file are flagged with a warning.
  • A file with no valid markers raises ValueError (not a silent empty result).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from regradar.models import DocumentVersion, Section


# Regex that matches an entire [SECTION: Name] line.
#
# Breakdown:
#   ^           — start of a line (re.MULTILINE makes ^ match each line)
#   \[SECTION:  — literal "[SECTION:"
#   \s*         — optional whitespace after the colon
#   (.+?)       — capture group: the section name (non-greedy)
#   \s*         — optional trailing whitespace
#   \]          — closing bracket
#   \s*$        — optional whitespace then end of line
#
# The capturing group is key: when you use re.split() with a capturing group,
# Python includes the captured text in the returned list, which lets us
# interleave headings and bodies without a loop.
_SECTION_PATTERN = re.compile(
    r"^\[SECTION:\s*(.+?)\s*\]\s*$",
    re.MULTILINE,
)


def load_document(path: str | Path, version_tag: str) -> DocumentVersion:
    """Read a .txt policy file and return a structured DocumentVersion.

    Args:
        path:        Path to the .txt file.
        version_tag: A label for this version, e.g. "v1" or "2024-rev2".

    Returns:
        A DocumentVersion with one Section per [SECTION: ...] block.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file contains no [SECTION: ...] markers.
    """
    path = Path(path)  # Accept both str and Path objects uniformly.

    raw_text = path.read_text(encoding="utf-8")

    sections = _parse_sections(raw_text)

    if not sections:
        raise ValueError(
            f"No [SECTION: ...] markers found in '{path}'.  "
            "Check the file format — each section must start with a line "
            "like:  [SECTION: Section Name]"
        )

    return DocumentVersion(
        version_tag=version_tag,
        source_path=str(path),
        sections=sections,
    )


def _parse_sections(text: str) -> list[Section]:
    """Split raw text into Section objects using the section marker pattern.

    How re.split works with a capturing group
    ─────────────────────────────────────────
    Given the text:

        preamble
        [SECTION: Alpha]
        Body of Alpha.
        [SECTION: Beta]
        Body of Beta.

    re.split(_SECTION_PATTERN, text) produces:

        ['preamble\\n', 'Alpha', 'Body of Alpha.\\n', 'Beta', 'Body of Beta.\\n']
         ↑ before first marker  ↑ captured name        ↑ body  ↑ captured name  ↑ body

    The list always starts with everything before the first marker (index 0),
    then alternates: name at even-offset indices (1, 3, 5 ...) and body at
    odd-offset indices (2, 4, 6 ...).

    We skip index 0 (preamble) and zip pairs (name, body) from index 1 onward.
    """
    parts = re.split(_SECTION_PATTERN, text)

    # parts[0] is text before the first marker.  Warn if it's non-empty.
    preamble = parts[0].strip()
    if preamble:
        warnings.warn(
            f"Text found before the first [SECTION:] marker will be ignored: "
            f"{preamble[:80]!r}",
            stacklevel=3,
        )

    # Slice off the preamble; what remains alternates [heading, body, heading, body, ...].
    heading_body_pairs = parts[1:]

    # Pair them up: zip stops at the shorter iterable, so an odd number of
    # remaining elements (which would indicate a bug in the regex) is safe.
    sections: list[Section] = []
    seen_headings: set[str] = set()

    for heading, body in zip(heading_body_pairs[0::2], heading_body_pairs[1::2]):
        heading = heading.strip()
        body = body.strip()

        if heading in seen_headings:
            warnings.warn(
                f"Duplicate section heading '{heading}' — only the last "
                "occurrence will be used.",
                stacklevel=3,
            )
        seen_headings.add(heading)

        sections.append(Section(heading=heading, text=body))

    return sections
