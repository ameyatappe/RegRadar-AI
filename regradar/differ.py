"""
regradar/differ.py
──────────────────
Compares two DocumentVersion objects and returns a DiffResult.

Algorithm overview
──────────────────
1. Build a heading → Section map for each document.
2. Determine the union of all headings that appear in either document,
   preserving order (old-document order first, then new-only headings).
3. For each heading in that ordered union:
     • In old only  → REMOVED
     • In new only  → ADDED
     • In both, same text  → UNCHANGED
     • In both, different text  → MODIFIED (compute unified diff)

Why order matters
─────────────────
A naive set() comparison gives correct change types but loses section order,
making the report hard to read.  We preserve old-document order because that
is the mental baseline a reviewer holds while reading the update.

difflib.unified_diff
────────────────────
The standard-library `difflib` module computes the minimal edit sequence
between two sequences of lines.  `unified_diff` formats it like `git diff`:
lines starting with '-' were in the old version, '+' in the new version,
and ' ' (space) are context lines that appear in both.

Python docs:  https://docs.python.org/3/library/difflib.html#difflib.unified_diff
"""

# Note: `from __future__ import annotations` (PEP 563) is intentionally
# absent — see regradar/models.py for the rationale.
import difflib

from regradar.models import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    DiffResult,
    DocumentVersion,
    SectionChange,
)


def diff_documents(
    old_doc: DocumentVersion,
    new_doc: DocumentVersion,
) -> DiffResult:
    """Compare two DocumentVersions and return a structured DiffResult.

    Args:
        old_doc: The baseline (older) document version.
        new_doc: The updated (newer) document version.

    Returns:
        A DiffResult containing one SectionChange per section in either doc.
    """
    old_map = old_doc.sections_by_heading()
    new_map = new_doc.sections_by_heading()

    # Build an ordered list of all headings that appear in either document.
    # dict.fromkeys() preserves insertion order and deduplicates — the
    # "trick" is that dict keys are unique, so merging two lists of keys
    # into one dict gives us a stable, deduplicated ordering.
    all_headings = list(
        dict.fromkeys(
            [s.heading for s in old_doc.sections]  # old order first
            + [s.heading for s in new_doc.sections]  # then new-only headings
        )
    )

    changes: list[SectionChange] = []

    for heading in all_headings:
        in_old = heading in old_map
        in_new = heading in new_map

        if in_old and not in_new:
            # Section existed in the old doc but is gone in the new doc.
            changes.append(
                SectionChange(
                    heading=heading,
                    change_type=CHANGE_REMOVED,
                    old_text=old_map[heading].text,
                    new_text=None,
                )
            )

        elif in_new and not in_old:
            # Section appears for the first time in the new doc.
            changes.append(
                SectionChange(
                    heading=heading,
                    change_type=CHANGE_ADDED,
                    old_text=None,
                    new_text=new_map[heading].text,
                )
            )

        else:
            # Section is present in both versions.
            old_text = old_map[heading].text
            new_text = new_map[heading].text

            if old_text == new_text:
                changes.append(
                    SectionChange(
                        heading=heading,
                        change_type=CHANGE_UNCHANGED,
                        old_text=old_text,
                        new_text=new_text,
                    )
                )
            else:
                # Texts differ — compute a human-readable unified diff.
                diff_lines = _compute_diff(
                    old_text=old_text,
                    new_text=new_text,
                    fromfile=f"{old_doc.version_tag}/{heading}",
                    tofile=f"{new_doc.version_tag}/{heading}",
                )
                changes.append(
                    SectionChange(
                        heading=heading,
                        change_type=CHANGE_MODIFIED,
                        old_text=old_text,
                        new_text=new_text,
                        diff_lines=diff_lines,
                    )
                )

    return DiffResult(
        old_version_tag=old_doc.version_tag,
        new_version_tag=new_doc.version_tag,
        changes=changes,
    )


def _compute_diff(
    old_text: str,
    new_text: str,
    fromfile: str,
    tofile: str,
) -> list[str]:
    """Return unified-diff lines for two text blocks.

    We split on newlines to get line-level diffs.  Each returned string is
    one diff line (including its leading '+', '-', or ' ' character).
    The header lines (--- / +++) are included for completeness.

    Args:
        old_text:  The original text.
        new_text:  The updated text.
        fromfile:  Label for the old side of the diff header.
        tofile:    Label for the new side of the diff header.

    Returns:
        A list of strings, one per diff line (without trailing newlines).
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    # `n=0` means no context lines — show only the changed lines, not
    # the surrounding unchanged lines.  This keeps the diff compact.
    raw = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",  # Suppress extra newlines; we strip them ourselves.
        n=0,
    )

    return [line.rstrip("\n") for line in raw]
