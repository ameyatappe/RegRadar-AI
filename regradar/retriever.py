"""
regradar/retriever.py
─────────────────────
BM25 keyword search over a single parsed policy document.

What this module does
─────────────────────
Given a DocumentVersion (produced by loader.load_document), a
PolicyRetriever builds a BM25 index over its sections and answers
free-text queries by returning the sections most likely to contain
the queried terms.

Why BM25 is a good fit for regulatory text
───────────────────────────────────────────
BM25 (Best Match 25) is a term-frequency ranking function.  It rewards:
  • Sections that contain query words frequently (term frequency, TF).
  • Sections that contain query words that are *rare* across the whole
    corpus (inverse document frequency, IDF).

Regulatory documents use precise, domain-specific language — "5-year
retention", "data controller", "Article 17(2)(b)".  Exact-term matching
handles these well.  Semantic similarity (embeddings) would help more
with paraphrases; that is Milestone 3.

Tokenisation
────────────
Tokenisation is the process of splitting text into individual words (tokens)
so BM25 can count term frequencies.

We use the simplest possible tokeniser: lowercase the text then split on
whitespace.  This is good enough for Milestone 2 because:
  • Regulatory text is written in formal, consistent language.
  • More sophisticated preprocessing (stemming, stopword removal) can be
    added later without changing any other module.

The same tokeniser is applied to both the corpus and every query, so the
comparison is always apples-to-apples.

What "score > 0" means
───────────────────────
A BM25 score of exactly 0.0 means the query tokens and the section share
no words at all.  Such sections are not returned — they are not matches.
A positive score means at least one query token appeared in the section,
weighted by how rare that token is across all sections.

Important: a positive score alone does NOT mean the section is certainly
relevant.  BM25 scores are not confidence percentages.  They are relative
rankings within one corpus.  A score of 1.2 only means "more relevant than
a section scoring 0.8" — not "72% confident".  The caller decides what
to do with the ranking.

Python docs:
  dataclasses  — https://docs.python.org/3/library/dataclasses.html
  rank_bm25    — https://github.com/dorianbrown/rank_bm25
"""

from rank_bm25 import BM25Okapi

from regradar.models import DocumentVersion, RetrievalResult


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens by whitespace.

    This function is applied identically to section bodies (at index time)
    and to query strings (at search time).  Consistency is essential: if the
    corpus is indexed with lowercased tokens but the query is not lowercased,
    "Retention" in the query would never match "retention" in the index.

    Args:
        text: Any string — a section body, a heading, or a search query.

    Returns:
        A list of lowercase tokens.  Returns an empty list for empty or
        whitespace-only input (str.split() on whitespace returns []).
    """
    return text.lower().split()


# ── Retriever class ───────────────────────────────────────────────────────────

class PolicyRetriever:
    """BM25 search index over one parsed policy document.

    Build once, query many times.  The BM25 index is constructed in
    ``__init__`` from the document's sections.  Each call to ``search``
    scores every section against the query and returns the top matches.

    Usage::

        doc = load_document("data/policies/data_retention_v2.txt", "v2")
        retriever = PolicyRetriever(doc)
        results = retriever.search("retention period years", top_k=3)
        for r in results:
            print(r.rank, r.heading, r.score)
    """

    def __init__(self, document: DocumentVersion) -> None:
        """Build the BM25 index from all sections in the document.

        Args:
            document: A fully parsed DocumentVersion.  May contain zero
                      sections (e.g. an empty document); in that case the
                      index is empty and all searches return [].
        """
        # Keep a reference to the original Section objects so that search()
        # can return the original heading and text without re-parsing.
        self._sections = document.sections          # list[Section]
        self._version_tag = document.version_tag    # str

        if self._sections:
            # BM25Okapi expects a list of lists of strings.
            # Each inner list is one document's tokens.
            tokenized_corpus = [_tokenize(s.text) for s in self._sections]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenized_corpus)
            # Convert each section's token list to a set for O(1) membership
            # lookup during overlap checks in search().  Built once here and
            # reused on every search call — no re-tokenisation per query.
            self._section_token_sets: list[set[str]] = [
                set(tokens) for tokens in tokenized_corpus
            ]
        else:
            # Empty corpus — BM25Okapi would divide by zero internally.
            # Store None and guard in search().
            self._bm25 = None
            self._section_token_sets = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search the indexed sections for the most relevant matches.

        Args:
            query:  Free-text search string.  Tokenised with the same
                    lowercase-whitespace tokeniser used at index time.
                    An empty or whitespace-only query returns [].
            top_k:  Maximum number of results to return.  Must be >= 0.
                    Passing 0 returns [].  If fewer than top_k sections
                    have a positive score, only those sections are returned.

        Returns:
            A list of RetrievalResult objects sorted by score descending
            (best match first), with one-based rank values.  Sections that
            score 0.0 (no shared tokens with the query) are excluded.

        Raises:
            TypeError:  If top_k is not a plain int (including bool, which is
                        a subclass of int in Python but not a valid count).
            ValueError: If top_k is a negative integer.
        """
        # bool is a subclass of int, so isinstance(True, int) is True.
        # Check bool first to reject it explicitly before the int check.
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(
                f"top_k must be an int, got {type(top_k).__name__!r}."
            )
        if top_k < 0:
            raise ValueError(
                f"top_k must be a non-negative integer, got {top_k!r}."
            )

        # Fast exits — no work needed.
        if top_k == 0 or self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            # Empty or whitespace-only query — no meaningful search possible.
            return []

        # get_scores() returns a numpy array of floats, one per section,
        # in the same order as the corpus passed to __init__.
        raw_scores = self._bm25.get_scores(query_tokens)

        # Build a set of query tokens once; used for O(1) overlap checks below.
        query_token_set = set(query_tokens)

        # Keep only sections that share at least one token with the query.
        # This is an explicit lexical-overlap check — it does not rely on the
        # BM25 score being positive, which would be a fragile assumption if the
        # IDF formula or BM25 variant were ever changed (see code-review notes).
        # self._section_token_sets was built once in __init__; no re-tokenising.
        # Convert numpy float64 → plain Python float for JSON serialisability.
        scored_sections = [
            (section, float(score))
            for section, score, section_tokens in zip(
                self._sections, raw_scores, self._section_token_sets
            )
            if query_token_set & section_tokens  # non-empty intersection = match
        ]

        # Sort by score, highest first.  Python's sort is stable, so sections
        # with equal scores preserve their original document order.
        scored_sections.sort(key=lambda pair: pair[1], reverse=True)

        # Apply the top_k limit.
        scored_sections = scored_sections[:top_k]

        # Build typed RetrievalResult objects.
        # enumerate(..., start=1) gives 1-based rank directly.
        return [
            RetrievalResult(
                heading=section.heading,
                text=section.text,
                score=score,
                version_tag=self._version_tag,
                rank=rank,
            )
            for rank, (section, score) in enumerate(scored_sections, start=1)
        ]
