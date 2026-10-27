# RegRadar AI

> An AI engineering project that investigates regulatory changes and identifies potentially affected internal policies, controls, and business processes.

This is an **educational and engineering tool**, not a legal adviser or compliance certification system. Built as a portfolio project to demonstrate RAG engineering, document analysis, and AI-assisted policy review.

---

## Project status

| Milestone | Description | Status |
|---|---|---|
| **M1** | Structured policy diff — compare two text versions | ✅ Complete |
| M2 | BM25 keyword retrieval over chunked policy corpus | 🔜 Next |
| M3 | Semantic retrieval with local embeddings | 🔜 Planned |
| M4 | Corrective RAG (CRAG) routing + evaluation | 🔜 Planned |
| M5 | FastAPI + UI demo | 🔜 Planned |

---

## Milestone 1 — Policy Diff Engine

Compares two versions of a plain-text policy document and produces:
- A colour-coded terminal summary of every section change (added / removed / modified / unchanged).
- A machine-readable JSON report saved to `reports/`.

### What it demonstrates
- Parsing structured documents with regular expressions.
- Modelling data with Python `dataclasses` instead of raw strings or dicts.
- Generating unified diffs with the standard-library `difflib` module.
- Writing and running unit tests with `pytest`.
- Zero external dependencies beyond `pytest` for testing.

---

## Quick start

### Prerequisites
- Python 3.11 or later (developed on Python 3.14)
- `pytest` for running tests (`py -m pip install pytest`)

### Install (no build step needed)
```bash
# Clone the repo
git clone https://github.com/<your-username>/RegRadar-AI.git
cd RegRadar-AI

# (Optional but recommended) create a virtual environment
py -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# Install dev dependencies
py -m pip install pytest
```

### Run the diff engine
```bash
py main.py data/policies/data_retention_v1.txt data/policies/data_retention_v2.txt
```

The terminal will show a section-by-section change summary. A JSON report is written to `reports/diff_report.json`.

### Run the tests
```bash
py -m pytest tests/ -v
```

Expected output: all tests pass, zero failures.

---

## Project layout

```
RegRadar-AI/
├── data/
│   └── policies/
│       ├── data_retention_v1.txt   # Fictional policy — old version
│       └── data_retention_v2.txt   # Fictional policy — new version
├── regradar/
│   ├── __init__.py
│   ├── models.py    # Typed dataclasses: Section, DocumentVersion, DiffResult
│   ├── loader.py    # Parses a .txt file into a DocumentVersion
│   ├── differ.py    # Compares two DocumentVersions, returns DiffResult
│   └── reporter.py  # Renders DiffResult to terminal and JSON
├── tests/
│   ├── __init__.py
│   └── test_differ.py
├── main.py
├── pyproject.toml
└── README.md
```

---

## Design decisions

**Why `dataclasses` instead of raw dicts?**
Typed fields catch mistakes at definition time, produce readable `repr()` output for debugging, and can be serialised to JSON with `dataclasses.asdict()` without any third-party library.

**Why `[SECTION: Name]` delimiters instead of Markdown headings?**
They are unambiguous — a `##` heading can appear inside a paragraph or a code example, but `[SECTION: Name]` on its own line is a deliberate machine-readable signal. This makes the regex parser reliable without special-casing.

**Why no LangChain in Milestone 1?**
The diff logic is three small functions totalling ~60 lines. Adding a framework to wrap three functions would obscure the logic without adding any capability. Frameworks are introduced in later milestones when they solve a real problem.

---

## Disclaimer

RegRadar AI is an educational engineering project. Nothing it produces constitutes legal advice or a compliance certification. Always consult qualified legal and compliance professionals for regulatory matters.
