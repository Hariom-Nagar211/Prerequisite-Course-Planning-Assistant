"""
rag_chain.py — Core RAG pipeline for MIT CS Catalog Assistant
LangChain 0.3.x compatible

Run standalone:  python src/rag_chain.py
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
INDEX_DIR   = Path("data/faiss_index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 6

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise course-planning assistant for the MIT \
Computer Science and Engineering (Course 6-3) undergraduate program.

Your ONLY source of information is the retrieved catalog excerpts provided \
in the context below. You must:

1. NEVER invent, assume, or extrapolate facts not present in the excerpts.
2. ALWAYS cite your source for every claim using the format:
   [Source: <filename> | Section: <section>]
3. If the answer is NOT in the context, respond with:
   "I don't have that information in the provided catalog/policies."
   and suggest where the student can find it (advisor, MIT registrar, etc.).
4. Always produce output in the following exact structured format:

---
**Answer / Plan:**
<your answer or course plan here>

**Why (requirements / prerequisites satisfied):**
<reasoning showing which rules/prereqs are met, with citations>

**Citations:**
<list each source used as: [Source: filename | Section: section_name]>

**Clarifying Questions (if needed):**
<list 1–5 questions if student info is missing; write "None" if complete>

**Assumptions / Not in Catalog:**
<note anything assumed or any information not found in the catalog>
---

Context (retrieved catalog excerpts):
{context}
"""

USER_TEMPLATE = "Student question: {question}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_retriever():
    """Load FAISS index and return a LangChain retriever."""
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_DIR}'. "
            "Run `python src/ingest.py` first."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def format_docs(docs) -> str:
    """Format retrieved docs into a context string with metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source  = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "—")
        url     = doc.metadata.get("source_url", "")
        header  = f"[Excerpt {i} | Source: {source} | Section: {section}"
        if url:
            header += f" | URL: {url}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_chain(retriever, llm):
    """Assemble RAG chain: retrieve → format → prompt → LLM → parse."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Public API ────────────────────────────────────────────────────────────────

class MITCatalogAssistant:
    """High-level wrapper used by both the Streamlit app and evaluator."""

    def __init__(self, groq_api_key: Optional[str] = None):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to .env or pass it directly."
            )
        self.retriever = load_retriever()
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=api_key,
            temperature=0.0,
            max_tokens=1500,
        )
        self.chain = build_chain(self.retriever, self.llm)

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)

    def get_source_chunks(self, question: str):
        return self.retriever.invoke(question)


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MIT CS Catalog RAG — Quick Test")
    print("=" * 60)
    assistant = MITCatalogAssistant()
    test_q = (
        "I have completed 6.1010 and 6.1200 with a B in both. "
        "Can I enroll in 6.1210 next semester?"
    )
    print(f"Question: {test_q}\n")
    print(assistant.ask(test_q))