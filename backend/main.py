"""
Fact Knowledge Layer - Backend API
FastAPI + Claude API for intelligent fact extraction and cross-document reasoning.
"""

import os
import json
import uuid
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional
import pymupdf as fitz  # PyMuPDF
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "facts.db"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Fact Knowledge Layer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

client = Groq()  # reads GROQ_API_KEY from env

# ── Database Setup ────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            page_count INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            fact_type TEXT,          -- e.g. revenue, headcount, date, growth_rate, policy, etc.
            value TEXT,              -- normalized value if numerical
            unit TEXT,               -- INR, %, persons, tonnes, etc.
            time_period TEXT,        -- FY24, Q4 FY23, 2024-25, etc.
            source_page INTEGER,
            source_sentence TEXT,    -- exact sentence from PDF
            confidence REAL,
            embedding TEXT,          -- JSON array, dot-product similarity
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            fact_id_a TEXT NOT NULL,
            fact_id_b TEXT NOT NULL,
            relationship TEXT NOT NULL,  -- CORROBORATES | CONTRADICTS | RECONCILED | UNCERTAIN
            reasoning TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (fact_id_a) REFERENCES facts(id),
            FOREIGN KEY (fact_id_b) REFERENCES facts(id)
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_chunks(pdf_path: str) -> list[dict]:
    """Extract text from PDF, returning chunks with page numbers."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if len(text) < 50:
            continue
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
        for para in paragraphs:
            chunks.append({
                "page": page_num + 1,
                "text": para[:2000]  # cap chunk size
            })
    doc.close()
    return chunks

# ── Claude Calls ──────────────────────────────────────────────────────────────
FACT_EXTRACTION_PROMPT = """You are a precise fact extractor for a knowledge graph system.

Given this text passage from a document, extract all meaningful facts.

RULES:
- Extract numerical facts (revenues, growth rates, counts, percentages, dates) AND semantic facts (company descriptions, leadership roles, policies, events).
- For each fact, identify the EXACT sentence it came from.
- Assign a fact_type from: [revenue, growth_rate, headcount, date, operational_metric, policy_rate, gdp, inflation, market_share, leadership, address, legal_status, financial_metric, other]
- For numerical facts: extract value and unit separately.
- Assign time_period if mentioned (e.g. "FY24", "Q4 FY23", "2024-25", "December 2021").
- Confidence: 0.9 if clearly stated, 0.7 if somewhat ambiguous, 0.5 if inferred.
- Skip generic statements that are not verifiable facts.

Respond with ONLY valid JSON, no markdown, no preamble:
{
  "facts": [
    {
      "fact_text": "brief description of the fact",
      "fact_type": "one of the types above",
      "value": "numerical value or null",
      "unit": "unit or null",
      "time_period": "period or null",
      "source_sentence": "exact sentence from the text",
      "confidence": 0.85
    }
  ]
}

Text passage:
"""

COMPARISON_PROMPT = """You are an expert at cross-document fact analysis.

Compare these two facts from different documents and determine their relationship.

Fact A:
- Text: {fact_a_text}
- Type: {fact_a_type}
- Value: {fact_a_value} {fact_a_unit}
- Period: {fact_a_period}
- Source: "{fact_a_sentence}"
- Document: {doc_a}

Fact B:
- Text: {fact_b_text}
- Type: {fact_b_type}
- Value: {fact_b_value} {fact_b_unit}
- Period: {fact_b_period}
- Source: "{fact_b_sentence}"
- Document: {doc_b}

Determine the relationship:
- CORROBORATES: Both facts say the same thing (possibly in different words/units/rounding).
- CONTRADICTS: Facts make conflicting claims about the same thing with no contextual explanation.
- RECONCILED: Apparent contradiction that is explained by context (different time periods, scopes, definitions, or units).
- UNCERTAIN: Cannot determine relationship without more information.

Respond with ONLY valid JSON:
{
  "relationship": "CORROBORATES|CONTRADICTS|RECONCILED|UNCERTAIN",
  "confidence": 0.85,
  "reasoning": "Clear 2-3 sentence explanation of why this relationship holds, citing the specific contextual differences if RECONCILED."
}
"""

def extract_facts_from_chunk(chunk_text: str, page: int) -> list[dict]:
    """Call Groq LLM to extract facts from a text chunk."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": FACT_EXTRACTION_PROMPT + chunk_text
            }]
        )
        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        for fact in data.get("facts", []):
            fact["source_page"] = page
        return data.get("facts", [])
    except Exception as e:
        print(f"Extraction error on page {page}: {e}")
        return []

def compare_facts(fact_a: dict, fact_b: dict, doc_a_name: str, doc_b_name: str) -> Optional[dict]:
    """Use Claude to compare two facts and determine their relationship."""
    prompt = COMPARISON_PROMPT.format(
        fact_a_text=fact_a["fact_text"],
        fact_a_type=fact_a["fact_type"],
        fact_a_value=fact_a.get("value", "N/A"),
        fact_a_unit=fact_a.get("unit", ""),
        fact_a_period=fact_a.get("time_period", "unspecified"),
        fact_a_sentence=fact_a["source_sentence"][:300],
        doc_a=doc_a_name,
        fact_b_text=fact_b["fact_text"],
        fact_b_type=fact_b["fact_type"],
        fact_b_value=fact_b.get("value", "N/A"),
        fact_b_unit=fact_b.get("unit", ""),
        fact_b_period=fact_b.get("time_period", "unspecified"),
        fact_b_sentence=fact_b["source_sentence"][:300],
        doc_b=doc_b_name,
    )
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"Comparison error: {e}")
        return None

def simple_embed(text: str) -> list[float]:
    """
    Simple bag-of-words TF-style embedding (no external model needed).
    Produces a 256-dim vector good enough for candidate retrieval.
    """
    words = text.lower().split()
    vec = [0.0] * 256
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 256
        vec[idx] += 1.0
    norm = (sum(x*x for x in vec) ** 0.5) or 1.0
    return [x / norm for x in vec]

def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def find_similar_facts(new_fact_embedding: list[float], new_fact_type: str,
                       existing_facts: list[dict], threshold: float = 0.25) -> list[dict]:
    """Find existing facts with similar embeddings and same/related type."""
    candidates = []
    for ef in existing_facts:
        if not ef["embedding"]:
            continue
        ef_emb = json.loads(ef["embedding"])
        sim = cosine_sim(new_fact_embedding, ef_emb)
        if sim >= threshold:
            candidates.append({"fact": ef, "similarity": sim})
    candidates.sort(key=lambda x: -x["similarity"])
    return candidates[:2]  # Top 2 candidates for speed

# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Fact Knowledge Layer API running", "version": "1.0.0"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF - extract facts and find relationships."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Save file
    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()
    save_path.write_bytes(content)

    # Store document record
    conn = get_db()
    chunks = extract_text_chunks(str(save_path))
    page_count = max((c["page"] for c in chunks), default=0)

    conn.execute(
        "INSERT INTO documents (id, filename, page_count) VALUES (?, ?, ?)",
        (doc_id, file.filename, page_count)
    )
    conn.commit()

    # Extract facts chunk by chunk
    new_facts = []
    # Process up to 30 chunks to keep things fast (prioritize first + last sections)
    selected_chunks = chunks[:4] + chunks[-4:] if len(chunks) > 8 else chunks
    # Deduplicate by text hash
    seen = set()
    for chunk in selected_chunks:
        h = hashlib.md5(chunk["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        extracted = extract_facts_from_chunk(chunk["text"], chunk["page"])
        for fact in extracted:
            fact["doc_id"] = doc_id
        new_facts.extend(extracted)

    # Store facts in DB
    stored_facts = []
    for fact in new_facts:
        fact_id = str(uuid.uuid4())
        emb = simple_embed(fact["fact_text"] + " " + (fact.get("time_period") or ""))
        conn.execute(
            """INSERT INTO facts
               (id, doc_id, fact_text, fact_type, value, unit, time_period,
                source_page, source_sentence, confidence, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact_id, fact["doc_id"], fact["fact_text"], fact.get("fact_type"),
                fact.get("value"), fact.get("unit"), fact.get("time_period"),
                fact.get("source_page"), fact.get("source_sentence", ""),
                fact.get("confidence", 0.8), json.dumps(emb)
            )
        )
        fact["id"] = fact_id
        stored_facts.append(fact)
    conn.commit()

    # Cross-document relationship detection
    # Fetch all facts from OTHER documents
    existing_rows = conn.execute(
        """SELECT f.*, d.filename
           FROM facts f JOIN documents d ON f.doc_id = d.id
           WHERE f.doc_id != ?""",
        (doc_id,)
    ).fetchall()
    existing_facts = [dict(r) for r in existing_rows]

    new_doc_name = file.filename
    relationships_found = 0
    compared_pairs = set()

    for new_fact in stored_facts:
        if not new_fact.get("id"):
            continue
        emb = simple_embed(new_fact["fact_text"] + " " + (new_fact.get("time_period") or ""))
        candidates = find_similar_facts(emb, new_fact.get("fact_type", ""), existing_facts)

        for cand in candidates:
            ef = cand["fact"]
            pair_key = tuple(sorted([new_fact["id"], ef["id"]]))
            if pair_key in compared_pairs:
                continue
            compared_pairs.add(pair_key)

            result = compare_facts(new_fact, ef, new_doc_name, ef["filename"])
            if result and result.get("relationship") in ("CORROBORATES", "CONTRADICTS", "RECONCILED"):
                rel_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO relationships (id, fact_id_a, fact_id_b, relationship, reasoning, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (rel_id, new_fact["id"], ef["id"],
                     result["relationship"], result["reasoning"], result.get("confidence", 0.8))
                )
                relationships_found += 1

    conn.commit()
    conn.close()

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": page_count,
        "chunks_processed": len(selected_chunks),
        "facts_extracted": len(stored_facts),
        "relationships_found": relationships_found,
        "message": "Processing complete."
    }

@app.get("/documents")
def list_documents():
    conn = get_db()
    rows = conn.execute(
        "SELECT d.*, COUNT(f.id) as fact_count FROM documents d LEFT JOIN facts f ON f.doc_id = d.id GROUP BY d.id ORDER BY d.uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/facts")
def list_facts(doc_id: Optional[str] = None, fact_type: Optional[str] = None, limit: int = 100):
    conn = get_db()
    query = """
        SELECT f.*, d.filename
        FROM facts f JOIN documents d ON f.doc_id = d.id
        WHERE 1=1
    """
    params = []
    if doc_id:
        query += " AND f.doc_id = ?"
        params.append(doc_id)
    if fact_type:
        query += " AND f.fact_type = ?"
        params.append(fact_type)
    query += " ORDER BY f.source_page LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d.pop("embedding", None)  # don't send to frontend
        result.append(d)
    return result

@app.get("/relationships")
def list_relationships(rel_type: Optional[str] = None, limit: int = 100):
    conn = get_db()
    query = """
        SELECT r.*,
               fa.fact_text as fact_a_text, fa.fact_type as fact_a_type,
               fa.value as fact_a_value, fa.unit as fact_a_unit,
               fa.time_period as fact_a_period, fa.source_sentence as fact_a_sentence,
               fa.source_page as fact_a_page,
               da.filename as doc_a,
               fb.fact_text as fact_b_text, fb.fact_type as fact_b_type,
               fb.value as fact_b_value, fb.unit as fact_b_unit,
               fb.time_period as fact_b_period, fb.source_sentence as fact_b_sentence,
               fb.source_page as fact_b_page,
               db.filename as doc_b
        FROM relationships r
        JOIN facts fa ON r.fact_id_a = fa.id
        JOIN facts fb ON r.fact_id_b = fb.id
        JOIN documents da ON fa.doc_id = da.id
        JOIN documents db ON fb.doc_id = db.id
        WHERE 1=1
    """
    params = []
    if rel_type:
        query += " AND r.relationship = ?"
        params.append(rel_type.upper())
    query += " ORDER BY r.confidence DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/stats")
def get_stats():
    conn = get_db()
    stats = {
        "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "facts": conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        "relationships": conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
        "corroborations": conn.execute("SELECT COUNT(*) FROM relationships WHERE relationship='CORROBORATES'").fetchone()[0],
        "contradictions": conn.execute("SELECT COUNT(*) FROM relationships WHERE relationship='CONTRADICTS'").fetchone()[0],
        "reconciled": conn.execute("SELECT COUNT(*) FROM relationships WHERE relationship='RECONCILED'").fetchone()[0],
        "fact_types": [dict(r) for r in conn.execute(
            "SELECT fact_type, COUNT(*) as count FROM facts GROUP BY fact_type ORDER BY count DESC"
        ).fetchall()],
    }
    conn.close()
    return stats

@app.get("/fact/{fact_id}")
def get_fact(fact_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT f.*, d.filename FROM facts f JOIN documents d ON f.doc_id = d.id WHERE f.id = ?",
        (fact_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, "Fact not found")
    d = dict(row)
    d.pop("embedding", None)
    # Get all relationships for this fact
    rels = conn.execute(
        """SELECT r.*,
                  fa.fact_text as fact_a_text, da.filename as doc_a,
                  fb.fact_text as fact_b_text, db.filename as doc_b
           FROM relationships r
           JOIN facts fa ON r.fact_id_a = fa.id
           JOIN facts fb ON r.fact_id_b = fb.id
           JOIN documents da ON fa.doc_id = da.id
           JOIN documents db ON fb.doc_id = db.id
           WHERE r.fact_id_a = ? OR r.fact_id_b = ?""",
        (fact_id, fact_id)
    ).fetchall()
    conn.close()
    d["relationships"] = [dict(r) for r in rels]
    return d

@app.delete("/reset")
def reset_database():
    """Clear all data - useful for testing."""
    conn = get_db()
    conn.execute("DELETE FROM relationships")
    conn.execute("DELETE FROM facts")
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink(missing_ok=True)
    return {"message": "Database reset complete."}