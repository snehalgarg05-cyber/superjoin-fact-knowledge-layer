# Fact Knowledge Layer

> An intelligent system that extracts, grounds, and cross-references facts across PDF documents using Groq AI (GPT-OSS 20B).

## Setup and Run Instructions

### Prerequisites
- Python 3.9+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- Git

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/snehalgarg05-cyber/superjoin-fact-knowledge-layer.git
cd superjoin-fact-knowledge-layer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file in root folder
echo GROQ_API_KEY=gsk_your_key_here > .env

# 4. Start the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. Open the UI
# Visit: http://localhost:8000/ui
```

### Windows Users

1. Open `run_windows.bat` in Notepad
2. Paste your Groq API key where indicated
3. Double-click to run

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
Groq API - Fact Extraction (model: openai/gpt-oss-20b)
    (structured JSON: fact_text, type, value, unit, time_period, source_sentence, confidence)
    |
    v
SQLite storage (documents + facts + relationships tables)
    |
    v
Embedding similarity search (lightweight bag-of-words, 256-dim)
    |
    v
Groq API - Cross-document Comparison (model: openai/gpt-oss-20b)
    (CORROBORATES / CONTRADICTS / RECONCILED / UNCERTAIN + reasoning)
    |
    v
FastAPI REST API + Single-page HTML UI
```

### Key Design Decisions

**1. Groq API (GPT-OSS 20B) for both extraction and comparison.**
Using an LLM for the comparison step means the system can reason about *why* two facts differ - not just detect numerical mismatch. GPT-OSS 20B on Groq's LPU gives fast inference with strong reasoning capability.

**2. Dynamic schema - no hardcoded fact types.**
The extraction prompt asks the model to assign `fact_type` from a suggested list, but the list is a hint, not a constraint. New document types will surface new types automatically. The DB stores these as plain text.

**3. Source sentence as the ground truth.**
Every fact stores the exact sentence it came from. Without grounding, a fact knowledge layer is just a list of claims. With the source sentence, every claim is verifiable.

**4. Lightweight embeddings for candidate retrieval.**
A 256-dimensional bag-of-words embedding for fast cosine similarity search narrows the comparison space so the LLM only gets called on plausible candidates, not all N×M pairs.

**5. Incremental ingestion - no rebuild.**
Each new PDF is compared only against existing facts. The existing knowledge layer is not reprocessed. This makes the system scale to many documents without quadratic cost growth.

**6. Transparent confidence scores.**
Both facts and relationships carry a confidence score. A fact extracted from a fragmented table gets confidence 0.7; a clearly stated numerical claim gets 0.9.

### Trade-offs

| Choice | What we gained | What we gave up |
|--------|---------------|----------------|
| Groq API for comparison | Contextual reasoning, handles units/synonyms | Latency, API cost per comparison |
| SQLite | Zero setup, fully portable | Not distributed, no full-text index |
| Bag-of-words embedding | No external model, fast | Misses semantic similarity |
| Paragraph-level chunking | Preserves local context | May miss facts spanning two paragraphs |
| FastAPI + single HTML file | Simple to deploy, no build step | Not a React SPA |

### AI Tools Used
- **Groq API (openai/gpt-oss-20b)** for fact extraction and cross-document reasoning
- **PyMuPDF (pymupdf)** for PDF text extraction
- **FastAPI** for the REST API
- **SQLite** for persistent storage
- **Vanilla JS** for the frontend (no build step)

---

## The Four Required Cases

All four cases are demonstrated live in the UI under the "4 Key Cases" tab after uploading the Delhivery or India Macro dataset.

### Case 1: Fact Corroborated Across Documents

**Example from Delhivery dataset:**

- **Q4 FY24 Earnings Presentation:** "FY24 revenue from services: Rs. 8,142 Cr, YoY: 12.7%"
- **Annual Report FY24:** "Revenue from services: Rs. 81,415 Mn"

Same number, different units (Cr vs Mn). System identifies this as CORROBORATES and explains the unit conversion.

**Example from India Macro dataset:**

- **Economic Survey 2024-25:** "GDP growth for FY25 is estimated to be 6.4 per cent"
- **RBI Annual Report 2024-25:** "GDP growth moderated to 6.5 per cent in 2024-25"

Slight difference due to different estimate vintages - system flags CORROBORATES with context note.

### Case 2: Genuine Contradiction

**Example from Delhivery dataset:**

- **Prospectus 2022:** Donald Francis Colleran listed as active Non-Executive Nominee Director
- **Annual Report FY24:** Colleran attended 0 out of 3 board meetings - effectively inactive

Active director in 2022 vs inactive by FY24. System flags CONTRADICTS with meeting attendance as evidence.

### Case 3: Apparent Contradiction Explained by Context

**Example from Delhivery dataset:**

- **Earnings Presentation (FY24 full year):** EBITDA = Rs. 127 Cr / 1.6% margin
- **Earnings Presentation (Q4 FY24 only):** EBITDA = Rs. 46 Cr / 2.2% margin

Different margins - RECONCILED because one is full year, other is Q4 only. Higher Q4 margin reflects operational improvement averaged down by weaker earlier quarters.

### Case 4: Extraction or Reasoning Failure

**Failure found:** Revenue figures embedded inside slide chart images in the Q4 Earnings PDF extract as orphaned numbers without column header context.

**How handled:** Low confidence (0.5) assigned, type set to "other" rather than guessing incorrectly.

**Fix:** PyMuPDF's `page.find_tables()` API for structured data + vision pass using multimodal API for chart graphics.

---

## Limitations and Next Steps

### What Does Not Work Yet

- PDF charts and images: values inside bar chart graphics not captured by text extraction
- Multi-page facts: a fact spanning a page break may get split into two incomplete facts
- Coreference: "the company's revenue" not linked to "Delhivery's revenue" without company name in same chunk
- Embedding quality: bag-of-words misses semantic similarity between "profit" and "earnings"
- Deduplication: uploading same PDF twice stores facts twice

### What I Would Build Next

1. PyMuPDF table extraction for structured financial data
2. Vision-based pass using multimodal API for chart images
3. ChromaDB replacing custom embedding for better semantic retrieval
4. Fact deduplication using file hash + fact fingerprint
5. Streaming extraction with server-sent events for live progress
6. Export to JSON-LD for consumption by other tools

---

## Dataset Sources Used

**Delhivery:**
- [Prospectus 2022](https://www.delhivery.com/wp-content/uploads/2022/05/Delhivery-Limited-Prospectus-1-min.pdf)
- [Annual Report FY24](https://www.delhivery.com/uploads/2024/08/Annual_Report_FY24.pdf)
- [Q4 FY24 Earnings Presentation](https://www.bseindia.com/xml-data/corpfiling/AttachHis/d70668ee-4f13-485e-ba19-62bec5116a59.pdf)

**India Macroeconomy:**
- [Economic Survey 2024-25](https://www.indiabudget.gov.in/budget2025-26/economicsurvey/doc/echapter.pdf)
- [RBI Annual Report 2024-25](https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0ANNUALREPORT202425DA4AE08189C848C8846718B080F2A0A9.PDF)
- [IMF India 2025 Article IV](https://www.imf.org/en/publications/cr/issues/2025/11/25/india-2025-article-iv-consultation-press-release-staff-report-and-statement-by-the-572056)