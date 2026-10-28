"""
regradar/models.py
──────────────────
Typed data models for the RegRadar diff pipeline.

All data in this project flows through these dataclasses rather than
through raw strings or plain dicts.  The benefits:

  • Fields have names and types — mistakes are visible when you read the code.
  • `repr()` is generated automatically — useful when debugging in the REPL.
  • `dataclasses.asdict()` converts any instance to a plain dict for JSON
    serialisation without any third-party library.
  • `pytest` assertions like `assert change.change_type == "modified"` are
    readable and precise.

Python docs:  https://docs.python.org/3/library/dataclasses.html
"""

# Note: `from __future__ import annotations` (PEP 563) is intentionally
# absent.  Python 3.14 deprecates it because PEP 649 makes deferred
# evaluation of annotations the default.  All annotation syntax used here
# (e.g. `str | None`, `list[Section]`) is natively supported on Python ≥3.10.
from dataclasses import dataclass, field


# ── Primitive building blocks ─────────────────────────────────────────────────

@dataclass
class Section:
    """One named section extracted from a policy document.

    Attributes:
        heading:  The section name found between [SECTION: ...] markers.
        text:     The body text of that section, stripped of leading/trailing
                  whitespace.
    """
    heading: str
    text: str


@dataclass
class DocumentVersion:
    """A fully parsed policy document with its version metadata.

    Attributes:
        version_tag:  A human-readable label such as "v1" or "2024-rev2".
        source_path:  The file path the document was loaded from.
        sections:     Ordered list of all sections found in the document.
    """
    version_tag: str
    source_path: str
    sections: list[Section] = field(default_factory=list)

    def sections_by_heading(self) -> dict[str, Section]:
        """Return a heading → Section mapping for fast lookup.

        If the same heading appears more than once, the last occurrence wins.
        (Duplicate headings are flagged by the loader as a warning.)
        """
        return {s.heading: s for s in self.sections}


# ── Diff results ──────────────────────────────────────────────────────────────

# Valid values for SectionChange.change_type.
# Using a plain str rather than an Enum keeps JSON serialisation simple.
CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_MODIFIED = "modified"
CHANGE_UNCHANGED = "unchanged"


@dataclass
class SectionChange:
    """Records what happened to a single section between two document versions.

    Attributes:
        heading:      The section heading this change describes.
        change_type:  One of "added", "removed", "modified", "unchanged".
        old_text:     The section body from the old document.
                      None when the section was added (did not exist before).
        new_text:     The section body from the new document.
                      None when the section was removed (no longer exists).
        diff_lines:   For "modified" sections: the unified-diff output lines
                      produced by difflib.  Empty list for other change types.
    """
    heading: str
    change_type: str
    old_text: str | None
    new_text: str | None
    diff_lines: list[str] = field(default_factory=list)


@dataclass
class DiffResult:
    """The complete comparison between two document versions.

    Attributes:
        old_version_tag:  Version label of the baseline (old) document.
        new_version_tag:  Version label of the updated (new) document.
        changes:          Ordered list of SectionChange objects, one per
                          section that appeared in either document.
    """
    old_version_tag: str
    new_version_tag: str
    changes: list[SectionChange] = field(default_factory=list)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def filter_by_type(self, change_type: str) -> list[SectionChange]:
        """Return only the changes whose change_type matches the given value."""
        return [c for c in self.changes if c.change_type == change_type]

    @property
    def summary(self) -> dict[str, int]:
        """Count changes by type.  Useful for quick reporting and assertions."""
        counts: dict[str, int] = {
            CHANGE_ADDED: 0,
            CHANGE_REMOVED: 0,
            CHANGE_MODIFIED: 0,
            CHANGE_UNCHANGED: 0,
        }
        for change in self.changes:
            if change.change_type in counts:
                counts[change.change_type] += 1
        return counts
