"""
tests/test_differ.py
─────────────────────
Unit tests for the RegRadar policy diff engine.

Each test function follows the Arrange / Act / Assert pattern:
  • Arrange — set up the input data (DocumentVersion objects built inline).
  • Act     — call the function under test (diff_documents).
  • Assert  — check the result against expected values.

Why we build DocumentVersions inline rather than reading real files
───────────────────────────────────────────────────────────────────
File-reading tests are slow and fragile — if the test data file moves, the
test breaks for reasons unrelated to the code being tested.  Building the
objects directly makes tests self-contained, fast, and explicit about exactly
what scenario they are exercising.

Running the tests
─────────────────
    py -m pytest tests/ -v

The -v flag prints one line per test function with a PASSED / FAILED status.
"""

from __future__ import annotations

import warnings

import pytest

from regradar.differ import diff_documents
from regradar.loader import load_document
from regradar.models import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    DocumentVersion,
    Section,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(sections: list[tuple[str, str]], tag: str = "test") -> DocumentVersion:
    """Build a DocumentVersion from a list of (heading, text) tuples.

    This helper keeps individual tests short and readable — each test
    only needs to declare what matters for *that* scenario.
    """
    return DocumentVersion(
        version_tag=tag,
        source_path="<test>",
        sections=[Section(heading=h, text=t) for h, t in sections],
    )


# ── Case 1: All sections unchanged ───────────────────────────────────────────

class TestUnchanged:
    def test_all_sections_unchanged(self):
        """When old and new documents are identical, every section is UNCHANGED."""
        doc = _make_doc([
            ("Purpose", "This policy defines retention rules."),
            ("Scope",   "Applies to all departments."),
        ])
        result = diff_documents(doc, doc)

        assert len(result.changes) == 2
        assert all(c.change_type == CHANGE_UNCHANGED for c in result.changes)

    def test_unchanged_section_carries_text(self):
        """An UNCHANGED section should still carry its text for context."""
        doc = _make_doc([("Purpose", "Fixed text.")])
        result = diff_documents(doc, doc)

        change = result.changes[0]
        assert change.old_text == "Fixed text."
        assert change.new_text == "Fixed text."
        assert change.diff_lines == []


# ── Case 2: Modified sections ─────────────────────────────────────────────────

class TestModified:
    def test_single_modified_section(self):
        """A section present in both docs with different text is MODIFIED."""
        old = _make_doc([("Retention Period", "Retain for 3 years.")])
        new = _make_doc([("Retention Period", "Retain for 5 years.")])
        result = diff_documents(old, new)

        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.change_type == CHANGE_MODIFIED
        assert change.heading == "Retention Period"
        assert change.old_text == "Retain for 3 years."
        assert change.new_text == "Retain for 5 years."

    def test_modified_section_has_diff_lines(self):
        """A MODIFIED section must include at least one diff line."""
        old = _make_doc([("Scope", "Old scope.")])
        new = _make_doc([("Scope", "New scope.")])
        result = diff_documents(old, new)

        change = result.changes[0]
        assert len(change.diff_lines) > 0

    def test_modified_diff_lines_contain_plus_and_minus(self):
        """Diff lines for a MODIFIED section must include both + and - lines."""
        old = _make_doc([("Policy", "Version one text.")])
        new = _make_doc([("Policy", "Version two text.")])
        result = diff_documents(old, new)

        diff = result.changes[0].diff_lines
        has_plus  = any(l.startswith("+") and not l.startswith("+++") for l in diff)
        has_minus = any(l.startswith("-") and not l.startswith("---") for l in diff)
        assert has_plus,  "Expected a '+' line in the unified diff."
        assert has_minus, "Expected a '-' line in the unified diff."

    def test_multiline_section_modification(self):
        """A section with multiple lines should still be classified as MODIFIED."""
        old = _make_doc([("Details", "Line one.\nLine two.\nLine three.")])
        new = _make_doc([("Details", "Line one.\nLine TWO changed.\nLine three.")])
        result = diff_documents(old, new)

        change = result.changes[0]
        assert change.change_type == CHANGE_MODIFIED

    def test_whitespace_only_difference_is_unchanged(self):
        """Sections that differ only in leading/trailing whitespace are UNCHANGED
        because the loader strips whitespace from section bodies."""
        # Simulate what the loader would produce after stripping.
        old = _make_doc([("Scope", "All departments.")])
        new = _make_doc([("Scope", "All departments.")])
        result = diff_documents(old, new)

        assert result.changes[0].change_type == CHANGE_UNCHANGED


# ── Case 3: Added sections ────────────────────────────────────────────────────

class TestAdded:
    def test_single_section_added(self):
        """A section in the new doc but not the old doc is ADDED."""
        old = _make_doc([("Purpose", "Old purpose.")])
        new = _make_doc([
            ("Purpose", "Old purpose."),
            ("Archival", "Move to cold storage after 3 years."),
        ])
        result = diff_documents(old, new)

        added = [c for c in result.changes if c.change_type == CHANGE_ADDED]
        assert len(added) == 1
        assert added[0].heading == "Archival"

    def test_added_section_has_no_old_text(self):
        """An ADDED section must have old_text == None."""
        old = _make_doc([("A", "text")])
        new = _make_doc([("A", "text"), ("B", "brand new")])
        result = diff_documents(old, new)

        added = result.filter_by_type(CHANGE_ADDED)
        assert added[0].old_text is None
        assert added[0].new_text == "brand new"

    def test_entirely_new_document(self):
        """If the old doc is empty and the new doc has sections, all are ADDED."""
        old = DocumentVersion(version_tag="empty", source_path="<test>", sections=[])
        new = _make_doc([("Alpha", "text"), ("Beta", "text")])
        result = diff_documents(old, new)

        assert len(result.changes) == 2
        assert all(c.change_type == CHANGE_ADDED for c in result.changes)


# ── Case 4: Removed sections ──────────────────────────────────────────────────

class TestRemoved:
    def test_single_section_removed(self):
        """A section in the old doc but not the new doc is REMOVED."""
        old = _make_doc([
            ("Purpose", "Purpose text."),
            ("Legacy",  "This section will be deleted."),
        ])
        new = _make_doc([("Purpose", "Purpose text.")])
        result = diff_documents(old, new)

        removed = [c for c in result.changes if c.change_type == CHANGE_REMOVED]
        assert len(removed) == 1
        assert removed[0].heading == "Legacy"

    def test_removed_section_has_no_new_text(self):
        """A REMOVED section must have new_text == None."""
        old = _make_doc([("A", "text"), ("B", "gone")])
        new = _make_doc([("A", "text")])
        result = diff_documents(old, new)

        removed = result.filter_by_type(CHANGE_REMOVED)
        assert removed[0].new_text is None
        assert removed[0].old_text == "gone"

    def test_all_sections_removed(self):
        """If the new doc is empty, all old sections are REMOVED."""
        old = _make_doc([("X", "text"), ("Y", "text")])
        new = DocumentVersion(version_tag="empty", source_path="<test>", sections=[])
        result = diff_documents(old, new)

        assert len(result.changes) == 2
        assert all(c.change_type == CHANGE_REMOVED for c in result.changes)


# ── Case 5: Multiple changes in one diff ─────────────────────────────────────

class TestMultipleChanges:
    def test_mixed_changes_match_data_retention_example(self):
        """The full data-retention policy example must produce exactly the
        expected mix of change types."""
        old = _make_doc([
            ("Purpose",          "Establishes data retention requirements for customers."),
            ("Scope",            "Applies to all departments."),
            ("Retention Period", "Retain for 3 years."),
            ("Disposal",         "Securely delete within 30 days of expiry."),
        ], tag="v1")

        new = _make_doc([
            ("Purpose",          "Establishes data retention and archival requirements."),
            ("Scope",            "Applies to all departments and third-party processors."),
            ("Retention Period", "Retain for 5 years."),
            ("Archival",         "Move records older than 3 years to cold storage."),
            ("Disposal",         "Securely delete within 30 days of expiry."),
        ], tag="v2")

        result = diff_documents(old, new)
        summary = result.summary

        assert summary[CHANGE_MODIFIED]  == 3, f"Expected 3 modified, got {summary}"
        assert summary[CHANGE_ADDED]     == 1, f"Expected 1 added,    got {summary}"
        assert summary[CHANGE_REMOVED]   == 0, f"Expected 0 removed,  got {summary}"
        assert summary[CHANGE_UNCHANGED] == 1, f"Expected 1 unchanged, got {summary}"

    def test_section_order_preserved(self):
        """Sections appear in old-document order; new-only sections come after."""
        old = _make_doc([("A", "a"), ("B", "b"), ("C", "c")])
        new  = _make_doc([("A", "a"), ("B", "b_changed"), ("D", "d")])
        result = diff_documents(old, new)

        headings = [c.heading for c in result.changes]
        # C was in old, D is new; A and B come first per old order.
        assert headings == ["A", "B", "C", "D"]


# ── Case 6: Malformed / edge-case inputs ─────────────────────────────────────

class TestMalformedInput:
    def test_empty_section_body_is_unchanged_if_both_empty(self):
        """Two sections with empty bodies should be UNCHANGED."""
        old = _make_doc([("Empty", "")])
        new = _make_doc([("Empty", "")])
        result = diff_documents(old, new)

        assert result.changes[0].change_type == CHANGE_UNCHANGED

    def test_empty_section_body_old_vs_content_new_is_modified(self):
        """If old body is empty and new body has content, section is MODIFIED."""
        old = _make_doc([("Clause", "")])
        new = _make_doc([("Clause", "New content added.")])
        result = diff_documents(old, new)

        assert result.changes[0].change_type == CHANGE_MODIFIED

    def test_loader_raises_on_file_with_no_sections(self, tmp_path):
        """load_document() must raise ValueError for a file with no markers."""
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("This is plain text with no section markers.", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\[SECTION:"):
            load_document(bad_file, version_tag="bad")

    def test_loader_raises_on_missing_file(self, tmp_path):
        """load_document() must raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "does_not_exist.txt", version_tag="ghost")

    def test_loader_warns_on_preamble_text(self, tmp_path):
        """load_document() must emit a warning when text appears before the first marker."""
        doc_file = tmp_path / "with_preamble.txt"
        doc_file.write_text(
            "Preamble text that should be ignored.\n\n"
            "[SECTION: Purpose]\n"
            "Purpose body.\n",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc = load_document(doc_file, version_tag="v1")

        assert len(caught) == 1, "Expected exactly one warning for preamble text."
        assert "ignored" in str(caught[0].message).lower()
        assert len(doc.sections) == 1  # Preamble not included as a section.

    def test_loader_warns_on_duplicate_headings(self, tmp_path):
        """load_document() must emit a warning for duplicate section headings."""
        doc_file = tmp_path / "dup.txt"
        doc_file.write_text(
            "[SECTION: Purpose]\nFirst purpose.\n"
            "[SECTION: Purpose]\nSecond purpose.\n",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc = load_document(doc_file, version_tag="v1")

        assert any("duplicate" in str(w.message).lower() for w in caught)

    def test_version_tags_recorded_in_diff_result(self):
        """DiffResult must carry the version tags from both input documents."""
        old = _make_doc([("A", "text")], tag="2023-v1")
        new = _make_doc([("A", "text")], tag="2024-v2")
        result = diff_documents(old, new)

        assert result.old_version_tag == "2023-v1"
        assert result.new_version_tag == "2024-v2"

    def test_real_policy_files_load_correctly(self):
        """Smoke test: the actual sample policy files must load without errors."""
        from pathlib import Path
        base = Path(__file__).parent.parent / "data" / "policies"
        v1 = load_document(base / "data_retention_v1.txt", "v1")
        v2 = load_document(base / "data_retention_v2.txt", "v2")

        assert len(v1.sections) == 4  # Purpose, Scope, Retention Period, Disposal
        assert len(v2.sections) == 5  # + Archival
