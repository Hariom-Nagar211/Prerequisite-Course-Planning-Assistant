# MIT CS Course Planning Assistant
### Agentic RAG — Assessment 1 Submission
**AI/ML Engineer Intern Assessment | Purple Merit Technologies | March 2026**

---

## Overview

A **Retrieval-Augmented Generation (RAG)** course-planning assistant grounded in the MIT Computer Science and Engineering (Course 6-3) academic catalog. Every answer cites its source. The system refuses to answer questions not covered by the catalog.

| Component | Choice | Reason |
|---|---|---|
| LLM | **Groq `llama-3.1-8b-instant`** | Free tier, ~400 tok/s, deterministic at temp=0 |
| Embeddings | **`all-MiniLM-L6-v2`** (HuggingFace) | Free, local, 384-dim, strong semantic search |
| Vector Store | **FAISS** | Fast, no server needed, ideal for static catalogs |
| Framework | **LangChain** 0.2.x | Stable, well-typed RAG primitives |
| UI | **Streamlit** | Rapid deployment, sidebar examples, chat history |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  INGESTION (ingest.py)                  │
│  .txt catalog files → clean → chunk (600 chars, 120    │
│  overlap) → HuggingFace embeddings → FAISS index       │
└─────────────────────┬───────────────────────────────────┘
                      │ saved to data/faiss_index/
┌─────────────────────▼───────────────────────────────────┐
│                    QUERY TIME (rag_chain.py)             │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  FAISS Retriever (similarity, k=6)                      │
│      │  → top-6 chunks with source metadata            │
│      ▼                                                  │
│  Prompt (system: enforce citations + abstention)        │
│      │                                                  │
│      ▼                                                  │
│  Groq LLM (llama-3.1-8b-instant, temp=0)                     │
│      │                                                  │
│      ▼                                                  │
│  Structured Response:                                   │
│    Answer / Plan | Why | Citations |                    │
│    Clarifying Questions | Assumptions                   │
└─────────────────────────────────────────────────────────┘
```

### Chunking Strategy

- **Chunk size: 600 characters** — large enough to include a full course description (name, prereqs, coreqs, description) but small enough for precise retrieval.
- **Overlap: 120 characters (20%)** — prevents rule text from being cut at chunk boundaries (e.g., a prerequisite condition split across two chunks).
- **Separator priority**: `\n---\n` (course separator) → `\n\n` (paragraphs) → `\n` (lines) → space. This means course blocks are split on the `---` divider first, keeping course data atomic.

---

## Data Sources

| File | URL | Content | Accessed |
|---|---|---|---|
| `mit_cs_catalog.txt` | https://catalog.mit.edu/subjects/6/ | All Foundation, Header, and AUS course descriptions with prereqs | March 2026 |
| `mit_cs_catalog.txt` | https://catalog.mit.edu/degree-charts/computer-science-engineering-course-6-3/ | Course 6-3 degree requirements, credit rules | March 2026 |
| `mit_cs_catalog.txt` | https://catalog.mit.edu/mit/undergraduate-education/academic-policies/ | Grading, repeat, credit overload, transfer policies | March 2026 |
| `mit_cs_catalog_supplement.txt` | https://catalog.mit.edu/subjects/6/ | Additional 6.xxx subjects (NLP, vision, robotics, etc.) | March 2026 |
| `mit_cs_catalog_supplement.txt` | https://www.eecs.mit.edu/academics-admissions/undergraduate-programs/course-6-3-cs/ | Double major rules, AI/ML track guidance, AUS group table | March 2026 |
| `mit_cs_catalog_supplement.txt` | https://registrar.mit.edu/registration/subject-enrollment/ | Registration procedures, waitlist, override policy | March 2026 |
| `mit_cs_catalog.txt` | https://www.eecs.mit.edu/academics-admissions/undergraduate-programs/course-6-3-cs/faq/ | EECS advising FAQ (co-req vs prereq, retake limits) | March 2026 |

**Total words**: ~12,000 across 2 files, 30+ distinct course/section entries.

---

## Prerequisites (System)

- Python 3.10+
- A **free** Groq API key — get one at https://console.groq.com (no credit card)

---

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/mit-rag-assistant.git
cd mit-rag-assistant

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies (pinned to avoid conflicts)
pip install -r requirements.txt

# 4. Set your Groq API key
cp .env.example .env
# Edit .env and paste your key: GROQ_API_KEY=gsk_...

# 5. Build the vector index (run once)
python src/ingest.py

# 6a. Launch the Streamlit UI
streamlit run app.py

# 6b. OR run a quick CLI test
python src/rag_chain.py

# 7. Run the 25-query evaluation
python src/evaluate.py
# Results saved to outputs/evaluation_report.json
```

---

## Project Structure

```
mit_rag/
├── app.py                          # Streamlit UI
├── requirements.txt                # Pinned deps (no conflicts)
├── .env.example                    # API key template
├── README.md
├── src/
│   ├── ingest.py                   # Ingestion pipeline
│   ├── rag_chain.py                # RAG chain + MITCatalogAssistant class
│   └── evaluate.py                 # 25-query evaluation harness
├── data/
│   ├── catalog/
│   │   ├── mit_cs_catalog.txt      # Main catalog (courses + policies)
│   │   └── mit_cs_catalog_supplement.txt  # Additional courses + AUS groups
│   └── faiss_index/               # Generated by ingest.py (git-ignored)
│       ├── index.faiss
│       ├── index.pkl
│       └── chunks_meta.pkl
└── outputs/
    └── evaluation_report.json     # Generated by evaluate.py
```

---

## Evaluation Results (Expected)

| Metric | Score |
|---|---|
| Citation coverage rate | ≥ 90% |
| Abstention accuracy (not-in-docs) | ≥ 80% |
| Structured format compliance | ≥ 95% |
| Avg keyword match (eligibility) | ≥ 70% |
| Overall pass rate | ≥ 80% |

### Per-category breakdown

| Category | # Queries | Focus |
|---|---|---|
| Prerequisite Check | 10 | Eligible / Not eligible binary decisions |
| Prerequisite Chain | 5 | Multi-hop reasoning (2+ steps) |
| Program Requirement | 5 | Credit rules, AUS groups, grade minimums |
| Not in Catalog | 5 | Correct abstention + guidance |

### Key Failure Modes

1. **Semester availability** — catalog notes historical patterns; actual term-by-term schedules are not in docs → correct behavior is abstention.
2. **Instructor override ambiguity** — policy mentions instructor consent exists but does not detail approval steps → partial abstention expected.
3. **Multi-hop chains** — LLM must reason across 3-4 retrieved chunks; occasionally missing an intermediate step in the chain.

### Next Improvements

- Add a **Verifier agent** that re-checks the plan output for citation completeness before returning it.
- Ingest **live MIT Subject Listing** (term-by-term HTML) to answer availability questions.
- Add **re-ranking** (cross-encoder) on top of FAISS for more precise chunk selection.
- Upgrade to `llama3-70b` or `mixtral-8x7b` on Groq for better multi-hop reasoning.

---

## Output Format (every response)

```
**Answer / Plan:**
<answer or course plan>

**Why (requirements / prerequisites satisfied):**
<reasoning with citations>

**Citations:**
[Source: filename | Section: section_name]

**Clarifying Questions (if needed):**
<1-5 questions or "None">

**Assumptions / Not in Catalog:**
<anything assumed or unavailable>
```

---

## Groq Free Tier

- **Rate limit**: 30 requests/min, 14,400 requests/day (as of March 2026)
- **Model**: `llama-3.1-8b-instant` — 8B parameters, 8192 token context
- **Speed**: ~400 tokens/second — near-instant for 1500-token responses
- The evaluation script includes a 1.2s delay between requests to stay within rate limits.
