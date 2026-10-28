"""
tests/test_retriever.py
───────────────────────
Unit tests for regradar/retriever.py.

Each test class covers one conceptual concern.  Tests follow the
Arrange / Act / Assert pattern used throughout this project.

Why we build corpora inline
───────────────────────────
Every test constructs its own DocumentVersion from (heading, text) tuples.
This keeps tests self-contained: they break only when the code under test
changes, not because a data file moved or was edited.

BM25 corpus design notes
─────────────────────────
BM25Okapi computes IDF = log(N - df + 0.5) - log(df + 0.5).  When a term
appears in more than half the corpus documents its raw IDF is negative; the
library applies an epsilon floor (0.25 × average_idf) to keep it positive.

If every term in the corpus has a negative IDF the average_idf approaches
zero and the floor collapses to zero — all scores become zero.  To avoid
this degenerate case the ranking test corpus includes unique terms (appearing
in exactly one section) that guarantee a positive average IDF.

Running the tests
─────────────────
    py -m pytest tests/ -v
"""

# Note: `from __future__ import annotations` (PEP 563) is intentionally
# absent — see regradar/models.py for the rationale.
import math

import pytest

from regradar.models import DocumentVersion, RetrievalResult, Section
from regradar.retriever import PolicyRetriever


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_doc(sections: list[tuple[str, str]], tag: str = "test") -> DocumentVersion:
    """Build a DocumentVersion from a list of (heading, text) tuples.

    Uses the real DocumentVersion and Section constructors so any breaking
    change to those classes surfaces immediately in these tests.
    """
    return DocumentVersion(
        version_tag=tag,
        source_path="<test>",
        sections=[Section(heading=h, text=t) for h, t in sections],
    )


# ── TestIndexing ──────────────────────────────────────────────────────────────

class TestIndexing:
    """Verifies that PolicyRetriever can be constructed from various document
    shapes without raising."""

    def test_builds_from_non_empty_document(self):
        doc = _make_doc([("Scope", "data retention policy applies")])
        retriever = PolicyRetriever(doc)
        assert retriever is not None

    def test_builds_from_empty_document(self):
        """Zero sections must not cause a crash at index time.

        BM25Okapi divides by average document length internally; passing an
        empty corpus would produce a ZeroDivisionError.  The constructor guards
        against this by storing None instead of a BM25 instance.
        """
        doc = DocumentVersion(version_tag="v1", source_path="<test>")
        retriever = PolicyRetriever(doc)
        assert retriever is not None


# ── TestRanking ───────────────────────────────────────────────────────────────

class TestRanking:
    """Verifies that query terms produce the expected ordering of results."""

    def test_relevant_section_ranks_first(self):
        """A section whose text contains all query terms must outrank sections
        that contain none of them."""
        doc = _make_doc([
            ("Retention Period", "data retention period five years required applies"),
            ("Access Control",   "user access control permissions management system"),
            ("Disposal",         "secure disposal destruction archival records kept"),
        ])
        retriever = PolicyRetriever(doc)
        # Only "Retention Period" contains "retention", "period", or "years".
        results = retriever.search("retention period years")
        assert len(results) >= 1
        assert results[0].heading == "Retention Period"

    def test_results_ordered_by_descending_score(self):
        """A section with higher term frequency for the query word must
        outrank one with lower term frequency.

        Corpus design rationale
        ───────────────────────
        "audit" appears in two sections (High TF × 3, Low TF × 1).
        "governance", "policy", "framework", "security", "compliance",
        "monitoring" each appear in exactly one section, ensuring a positive
        average IDF and thus a positive BM25 epsilon floor for "audit".

        With a positive IDF and higher TF, "High TF" is guaranteed to score
        above "Low TF" by the BM25 formula.
        """
        doc = _make_doc([
            ("High TF",  "audit audit audit governance policy framework"),
            ("Low TF",   "audit security compliance monitoring system"),
            ("No Match", "annual report summary findings recommendations"),
        ])
        retriever = PolicyRetriever(doc)
        results = retriever.search("audit")

        # Both matching sections must appear; the unrelated one must not.
        headings = [r.heading for r in results]
        assert "High TF" in headings
        assert "Low TF" in headings
        assert "No Match" not in headings

        # Scores must be in non-increasing order.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unrelated_section_excluded_from_results(self):
        """A section with no overlapping tokens must never appear in results."""
        doc = _make_doc([
            ("Retention", "data retention period five years"),
            ("Unrelated", "completely different subject matter entirely"),
        ])
        results = PolicyRetriever(doc).search("retention")
        headings = [r.heading for r in results]
        assert "Unrelated" not in headings


# ── TestResultShape ───────────────────────────────────────────────────────────

class TestResultShape:
    """Verifies every field of the returned RetrievalResult objects."""

    # Build once and reuse across all shape tests in this class.
    _DOC = _make_doc(
        [("Retention Period", "data retention five years applies")],
        tag="v2",
    )

    def _first(self) -> RetrievalResult:
        return PolicyRetriever(self._DOC).search("retention")[0]

    def test_heading_matches_section_heading(self):
        assert self._first().heading == "Retention Period"

    def test_text_is_original_prose_not_tokenised_form(self):
        """text must be the unmodified section body, not the lowercased token list."""
        assert self._first().text == "data retention five years applies"

    def test_version_tag_carried_through_from_document(self):
        assert self._first().version_tag == "v2"

    def test_score_is_plain_python_float_not_numpy_float(self):
        """numpy.float64 is not JSON-serialisable; the retriever must convert it."""
        result = self._first()
        # type() is exact — does not accept subclasses such as numpy.float64.
        assert type(result.score) is float

    def test_score_is_finite_float(self):
        """The score must be a finite float, not NaN or infinite, without assuming positivity."""
        score = self._first().score
        assert type(score) is float
        assert math.isfinite(score)

    def test_rank_of_top_result_is_one(self):
        assert self._first().rank == 1

    def test_ranks_are_one_based_and_consecutive(self):
        """With two matching sections the ranks must be 1, 2 — not 0, 1."""
        doc = _make_doc([
            ("Alpha", "retention policy applies here always annual"),
            ("Beta",  "retention record keeping five years long"),
        ])
        results = PolicyRetriever(doc).search("retention", top_k=5)
        assert len(results) == 2
        ranks = [r.rank for r in results]
        assert ranks == [1, 2]


# ── TestTopK ──────────────────────────────────────────────────────────────────

class TestTopK:
    """Verifies the top_k parameter correctly limits the result list."""

    def test_top_k_limits_number_of_results(self):
        """When all three sections match, top_k=2 must return at most 2 results."""
        doc = _make_doc([
            ("Alpha", "audit retention policy applies"),
            ("Beta",  "audit control review annual"),
            ("Gamma", "audit security compliance monitoring"),
        ])
        results = PolicyRetriever(doc).search("audit", top_k=2)
        assert len(results) <= 2

    def test_top_k_larger_than_matches_returns_all_matches(self):
        """Requesting more results than exist must not raise; return what is there."""
        doc = _make_doc([
            ("Only Match", "retention data policy applies"),
            ("No Match",   "access control permissions system"),
        ])
        results = PolicyRetriever(doc).search("retention", top_k=100)
        headings = [r.heading for r in results]
        assert "Only Match" in headings
        assert len(results) == 1  # only one section contains "retention"

    def test_top_k_zero_returns_empty_list(self):
        doc = _make_doc([("Section", "retention policy five years")])
        assert PolicyRetriever(doc).search("retention", top_k=0) == []


# ── TestEmptyInput ────────────────────────────────────────────────────────────

class TestEmptyInput:
    """Verifies safe handling of empty documents and empty or blank queries."""

    def test_empty_document_returns_empty_list(self):
        """A DocumentVersion with no sections must return [] without crashing."""
        doc = DocumentVersion(version_tag="v1", source_path="<test>")
        assert PolicyRetriever(doc).search("retention") == []

    def test_empty_query_string_returns_empty_list(self):
        doc = _make_doc([("Section", "data retention policy five years")])
        assert PolicyRetriever(doc).search("") == []

    def test_whitespace_only_query_returns_empty_list(self):
        """Whitespace tokenises to an empty token list — equivalent to no query."""
        doc = _make_doc([("Section", "data retention policy five years")])
        assert PolicyRetriever(doc).search("   \t\n  ") == []


# ── TestValidation ────────────────────────────────────────────────────────────

class TestValidation:
    """Verifies that invalid top_k values are rejected with the correct exception."""

    def _retriever(self) -> PolicyRetriever:
        return PolicyRetriever(_make_doc([("S", "retention policy applies")]))

    def test_negative_top_k_raises_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._retriever().search("retention", top_k=-1)

    def test_float_top_k_raises_type_error(self):
        with pytest.raises(TypeError, match="int"):
            self._retriever().search("retention", top_k=2.5)

    def test_string_top_k_raises_type_error(self):
        with pytest.raises(TypeError, match="int"):
            self._retriever().search("retention", top_k="5")

    def test_none_top_k_raises_type_error(self):
        with pytest.raises(TypeError, match="int"):
            self._retriever().search("retention", top_k=None)

    def test_true_raises_type_error(self):
        """bool is a subclass of int in Python, so isinstance(True, int) is True.
        The validator must check for bool before checking for int to reject it."""
        with pytest.raises(TypeError, match="int"):
            self._retriever().search("retention", top_k=True)

    def test_false_raises_type_error(self):
        """False equals 0 and would silently return [] without the bool guard."""
        with pytest.raises(TypeError, match="int"):
            self._retriever().search("retention", top_k=False)


# ── TestOverlapFilter ─────────────────────────────────────────────────────────

class TestOverlapFilter:
    """Verifies the explicit token-overlap filter that decides which sections qualify.

    The filter uses set intersection, not BM25 score sign, to determine whether
    a section is a lexical match.  This class tests both directions:
      • No overlap → excluded (even if BM25 produced a non-zero score by chance).
      • Overlap present → retained (even if BM25 returns 0.0).
    """

    def test_query_with_no_overlapping_tokens_returns_empty(self):
        """A query whose tokens never appear in any section must return []."""
        doc = _make_doc([
            ("Retention", "data retention five years policy"),
            ("Disposal",  "secure disposal destruction archival"),
        ])
        results = PolicyRetriever(doc).search("blockchain cryptocurrency gdpr")
        assert results == []

    def test_zero_bm25_score_with_token_overlap_is_retained(self, monkeypatch):
        """The overlap filter must keep a section even when BM25 returns 0.0.

        We monkeypatch PolicyRetriever._bm25.get_scores to force a score of 0.0
        for a section that genuinely shares the token "data" with the query.
        The old `score > 0.0` filter would have discarded this section; the
        new set-intersection filter must retain it.

        Why a plain Python list is a valid mock return value
        ─────────────────────────────────────────────────────
        The retriever iterates raw_scores with zip() and calls float(score) on
        each element.  Both numpy arrays and Python lists satisfy zip(); float()
        works on Python floats too.  No numpy import is required in the test.
        """
        doc = _make_doc([("Scope", "data retention applies")], tag="v1")
        retriever = PolicyRetriever(doc)

        # Force BM25 to score the section 0.0 despite the "data" token overlap.
        monkeypatch.setattr(
            retriever._bm25,
            "get_scores",
            lambda tokens: [0.0],
        )

        results = retriever.search("data")

        assert len(results) == 1, (
            "A section sharing a token with the query must be retained "
            "regardless of its BM25 score."
        )
        assert results[0].heading == "Scope"
        assert results[0].score == 0.0
        assert results[0].rank == 1


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: empty section body, case folding, and punctuation behaviour."""

    def test_section_with_empty_body_is_not_returned(self):
        """An empty-body section shares no tokens with any query and must be
        excluded.  It must also not cause an index-time or search-time crash."""
        doc = _make_doc([
            ("Empty Section", ""),
            ("Real Section",  "data retention policy applies here"),
        ])
        results = PolicyRetriever(doc).search("retention")
        headings = [r.heading for r in results]
        assert "Empty Section" not in headings
        assert "Real Section" in headings

    def test_search_is_case_insensitive(self):
        """Both the corpus and the query are lowercased by _tokenize.
        An uppercase query must match a section containing the lowercase form."""
        doc = _make_doc([("Policy", "data Retention Policy applies")])
        # "RETENTION" → lowercased to "retention" → matches "retention" in index.
        results = PolicyRetriever(doc).search("RETENTION")
        assert len(results) == 1
        assert results[0].heading == "Policy"
        # Original mixed-case text is preserved in the result.
        assert results[0].text == "data Retention Policy applies"

    def test_punctuation_attached_to_token_prevents_match(self):
        """The tokeniser splits on whitespace only; punctuation is NOT stripped.

        "retention." (with full stop) and "retention" (without) are different
        tokens.  A query of "retention" must NOT match "retention." in the text.

        This test documents a known limitation of the whitespace tokeniser.
        If punctuation normalisation is added in a future milestone, this test
        must be updated to reflect the new behaviour rather than deleted.
        """
        doc = _make_doc([("Section", "retention. policy applies annually.")])
        # The section's token set is {"retention.", "policy", "applies", "annually."}
        # The query token set is {"retention"}.
        # Intersection is empty → no match.
        results = PolicyRetriever(doc).search("retention")
        assert results == [], (
            "Punctuation is not stripped by the current tokeniser; "
            "'retention' must not match 'retention.'"
        )
