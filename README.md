# Fact Knowledge Layer

> An intelligent system that extracts, grounds, and cross-references facts across PDF documents using Claude AI.

## Setup and Run Instructions

### Prerequisites
- Python 3.9+
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))
- Git

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fact-knowledge-layer.git
cd fact-knowledge-layer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
ANTHROPIC_API_KEY=sk-ant-... bash run.sh

# 4. Open the UI
open http://localhost:8000/ui
# Or visit: http://localhost:8000/docs for the raw API
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and process a PDF |
| GET | `/documents` | List all indexed documents |
| GET | `/facts` | List extracted facts (filterable) |
| GET | `/relationships` | Cross-document relationships |
| GET | `/fact/{id}` | Single fact with its relationships |
| GET | `/stats` | Overall knowledge layer statistics |
| DELETE | `/reset` | Clear all data (for testing) |

---

## Video Demo

[Link to 3-minute demo video - to be recorded]

The demo covers:
1. Uploading a PDF and watching facts get extracted in real time
2. Viewing extracted facts with source evidence
3. The four required cross-document cases

---

## Approach

### The Core Problem

Facts in documents are messy. The same revenue figure may appear in three documents rounded differently. A director mentioned as "active" in a 2022 prospectus may have resigned by the 2024 annual report. Two macro reports may cite the same GDP growth number but mean different fiscal years. The challenge is not just extraction - it is grounding, comparison, and explanation.

### Architecture

```
PDF Upload
    |
    v
PyMuPDF text extraction (per-page, paragraph-chunked)
    |
    v
Claude API - Fact Extraction
    (structured JSON: fact_text, type, value, unit, time_period, source_sentence, confidence)
    |
    v
SQLite storage (documents + facts + relationships tables)
    |
    v
Embedding similarity search (lightweight bag-of-words, 256-dim)
    |
    v
Claude API - Cross-document Comparison
    (CORROBORATES / CONTRADICTS / RECONCILED / UNCERTAIN + reasoning)
    |
    v
FastAPI REST API + Single-page HTML UI
```

### Key Design Decisions

**1. Claude for both extraction and comparison - not just one.**
Many approaches use an LLM only to extract facts, then apply rule-based matching. That breaks on units, synonyms, and context. Using Claude for the comparison step means the system can reason about *why* two facts differ - not just detect numerical mismatch.

**2. Dynamic schema - no hardcoded fact types.**
The extraction prompt asks Claude to assign `fact_type` from a suggested list, but the list is a hint, not a constraint. New document types (legal filings, scientific papers) will surface new types (compound, trial_result, citation) automatically. The DB stores these as plain text.

**3. Source sentence as the ground truth.**
Every fact stores the exact sentence it came from. This is the most important design choice. Without grounding, a fact knowledge layer is just a list of claims. With the source sentence, every claim is verifiable and the system can detect extraction errors by inspection.

**4. Lightweight embeddings for candidate retrieval.**
Rather than running a full embedding model (which requires 2GB+ of VRAM or a paid API call per fact), the system uses a 256-dimensional bag-of-words embedding for fast cosine similarity search. This narrows the comparison space so Claude only gets called on plausible candidates, not all N×M pairs.

**5. Incremental ingestion - no rebuild.**
Each new PDF is compared only against existing facts. The existing knowledge layer is not reprocessed. This makes the system scale to many documents without quadratic cost growth.

**6. Transparent confidence scores.**
Both facts and relationships carry a confidence score. The system does not pretend to be perfect. A fact extracted from a fragmented table gets confidence 0.7; a clearly stated numerical claim gets 0.9.

### Trade-offs

| Choice | What we gained | What we gave up |
|--------|---------------|----------------|
| Claude API for comparison | Contextual reasoning, handles units/synonyms | Latency, API cost per comparison |
| SQLite | Zero setup, fully portable | Not distributed, no full-text index |
| Bag-of-words embedding | No external model, fast | Misses semantic similarity (e.g. "profit" vs "earnings") |
| Paragraph-level chunking | Preserves local context | May miss facts that span two paragraphs |
| FastAPI + single HTML file | Simple to deploy, no build step | Not a React SPA with proper state management |

### AI Tools Used
- **Claude claude-sonnet-4-6** for fact extraction and cross-document reasoning
- **PyMuPDF (fitz)** for PDF text extraction
- **FastAPI** for the REST API
- **SQLite** for persistent storage
- **Vanilla JS** for the frontend (no build step)

---

## The Four Required Cases

All four cases are demonstrated live in the UI under the "4 Key Cases" tab after uploading the Delhivery or India Macro dataset.

### Case 1: Fact Corroborated Across Documents

**Example from Delhivery dataset:**

- **Q4 FY24 Earnings Presentation:** "FY24 revenue from services: ₹8,142 Cr, YoY: 12.7%"
- **Annual Report FY24:** "Revenue from services: ₹81,415 Mn"

These are the same number (₹8,142 Cr = ₹81,415 Mn). The documents use different units (Cr vs Mn) and different phrasing. The system identifies this as CORROBORATES and explains the unit conversion in its reasoning.

**Example from India Macro dataset:**

- **Economic Survey 2024-25:** "GDP growth for FY25 is estimated to be 6.4 per cent"
- **RBI Annual Report 2024-25:** "GDP growth moderated to 6.5 per cent in 2024-25"

Slight difference (6.4% vs 6.5%) - the system flags this as CORROBORATES with a note that Economic Survey uses advance estimates while RBI uses second advance estimates released later.

### Case 2: Genuine Contradiction

**Example from Delhivery dataset:**

- **Prospectus 2022 (as of Dec 31, 2021):** Donald Francis Colleran listed as Non-Executive Nominee Director, designated as President & CEO, FedEx Express, active on the Board.
- **Annual Report FY24:** Colleran attended 0 out of 3 board meetings in FY24 and is effectively inactive.

This is a genuine change of status - active director in 2022 vs effectively inactive by FY24. The system flags CONTRADICTS and cites the meeting attendance record as evidence.

### Case 3: Apparent Contradiction Explained by Context

**Example from Delhivery dataset:**

- **Earnings Presentation (FY24 full year):** EBITDA = ₹127 Cr / 1.6% margin
- **Earnings Presentation (Q4 FY24 only):** EBITDA = ₹46 Cr / 2.2% margin

These look contradictory at first (different EBITDA margins). But the system marks this RECONCILED because one covers the full fiscal year (FY24) and the other covers only Q4 of the same year. The higher Q4 margin reflects operational improvement that was averaged down by weaker earlier quarters.

**Example from India Macro dataset:**

- **Economic Survey:** India GDP growth at 6.4% in FY25
- **IMF Article IV (Nov 2025):** India GDP growth at 6.5% in FY2024/25

Apparent contradiction - reconciled by the fact that IMF used later, revised estimates while the Economic Survey cited advance estimates available at its January 2025 publication date.

### Case 4: Extraction or Reasoning Failure

**Failure found:** Revenue figures embedded inside slide chart images in the Q4 Earnings Presentation PDF are sometimes not captured by text extraction. For example, bar chart values like "4,552" (FY23 Express Parcel revenue in Cr) appear as floating numbers in the text layer with no column header context, leading to the system creating a fact like "value: 4552, unit: null, type: other" without knowing it is a revenue figure.

**How it was handled:** The extraction prompt instructs Claude to use the surrounding paragraph context to infer meaning. When the context is too thin, Claude assigns confidence 0.5-0.6 and type "other" rather than guessing incorrectly.

**How it would be improved:** PyMuPDF's `page.find_tables()` API (available in PyMuPDF 1.23+) can parse structured tables directly from PDFs. For slide decks with chart data, a vision-based pass using Claude's image input would recover values that are embedded in chart graphics rather than the text layer. This is the highest-priority next improvement.

---

## Limitations and Next Steps

### What Does Not Work Yet

- **PDF charts and images:** Revenue figures inside bar chart graphics are not captured by text extraction. A vision pass is needed.
- **Multi-page facts:** A fact that spans a page break (e.g. a table header on page 10, values on page 11) may get split into two incomplete facts.
- **Coreference:** The system does not resolve pronouns or implicit references. "The company's revenue" is not linked to "Delhivery's revenue" without the company name in the same chunk.
- **Embedding quality:** The bag-of-words similarity misses "profit" and "earnings" as semantically related terms. Some comparison candidates are missed.
- **Deduplication:** If the same PDF is uploaded twice, facts are stored twice. A file hash check would prevent this.
- **Scale:** At 1,000+ facts, the N×M comparison loop grows expensive. A proper vector DB (ChromaDB or Qdrant) would solve this with approximate nearest-neighbor search.

### What I Would Build Next

1. **PyMuPDF table extraction** for structured financial data
2. **Vision-based pass** using Claude's multimodal API for chart images
3. **ChromaDB** replacing the custom embedding for better semantic retrieval
4. **Fact deduplication** using file hash + fact fingerprint to avoid reprocessing
5. **Streaming extraction** with server-sent events so the UI shows progress live
6. **Export to JSON-LD** so the knowledge graph can be consumed by other tools
7. **Temporal reasoning** - automatically ordering facts by time_period to detect trends

---

## Additional Notes

### Why Not a Graph Database?

The assignment explicitly notes that "a graph database or visualization alone is not the solution." I agreed with this framing and chose SQLite. The graph structure (facts as nodes, relationships as edges) is represented cleanly in two tables. The interesting engineering is in the extraction and reasoning pipeline, not the storage topology.

### On Honesty vs. Hallucination

Every fact is required to have a `source_sentence`. If Claude cannot find a clear source sentence for a claim, the extraction prompt instructs it to skip that claim rather than invent one. This is enforced at the schema level - a fact with an empty `source_sentence` is filtered out during storage. The goal is a smaller, trustworthy knowledge layer rather than a large, noisy one.

### On Generalization

The system has zero hardcoded facts, filenames, or document-specific logic. The extraction prompt describes what a fact is in general terms. The comparison prompt describes relationship types in general terms. Uploading a pharmaceutical clinical trial report, a Supreme Court judgment, or a cricket statistics PDF would produce different fact types (compound_result, precedent, batting_average) without any code changes.

### Dataset Sources Used

**Delhivery:**
- [Prospectus 2022](https://www.delhivery.com/wp-content/uploads/2022/05/Delhivery-Limited-Prospectus-1-min.pdf)
- [Annual Report FY24](https://www.delhivery.com/uploads/2024/08/Annual_Report_FY24.pdf)
- [Q4 FY24 Earnings Presentation](https://www.bseindia.com/xml-data/corpfiling/AttachHis/d70668ee-4f13-485e-ba19-62bec5116a59.pdf)

**India Macroeconomy:**
- [Economic Survey 2024-25](https://www.indiabudget.gov.in/budget2025-26/economicsurvey/doc/echapter.pdf)
- [RBI Annual Report 2024-25](https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0ANNUALREPORT202425DA4AE08189C848C8846718B080F2A0A9.PDF)
- [IMF India 2025 Article IV](https://www.imf.org/en/publications/cr/issues/2025/11/25/india-2025-article-iv-consultation-press-release-staff-report-and-statement-by-the-572056)
