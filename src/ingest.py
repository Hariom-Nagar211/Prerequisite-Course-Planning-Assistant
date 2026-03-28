"""
ingest.py — Document ingestion pipeline for MIT CS Catalog RAG
Loads catalog .txt files → cleans → chunks → embeds → saves FAISS index

Run: python src/ingest.py
"""

import os
import pickle
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ── Config ─────────────────────────────────────────────────────────────────
CATALOG_DIR   = Path("data/catalog")
INDEX_DIR     = Path("data/faiss_index")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # free, local
CHUNK_SIZE    = 600    # chars — keeps one rule / course block together
CHUNK_OVERLAP = 120    # 20 % overlap so context isn't cut at boundaries

# ── Helpers ────────────────────────────────────────────────────────────────

def load_catalog_files(catalog_dir: Path) -> List[Document]:
    """Load every .txt file from catalog dir as LangChain Documents."""
    docs: List[Document] = []
    for path in sorted(catalog_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        # Store the source file name as metadata
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name,
                "full_path": str(path),
            }
        ))
        print(f"  Loaded: {path.name} ({len(text):,} chars)")
    return docs


def extract_section_metadata(chunk_text: str, source: str) -> dict:
    """
    Best-effort extraction of section heading from chunk text.
    Looks for lines starting with 'SOURCE:' or 'SECTION:'.
    """
    source_url = ""
    section_heading = ""
    for line in chunk_text.splitlines():
        line = line.strip()
        if line.startswith("SOURCE:") and not source_url:
            source_url = line.replace("SOURCE:", "").strip()
        if line.startswith("SECTION:") and not section_heading:
            section_heading = line.replace("SECTION:", "").strip()
        # Also capture course numbers like "6.1210 —"
        if "—" in line and any(f"6.{n}" in line for n in range(10)):
            if not section_heading:
                section_heading = line.split("—")[0].strip()
    return {
        "source": source,
        "source_url": source_url,
        "section": section_heading,
    }


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split raw documents into overlapping chunks.
    Strategy: RecursiveCharacterTextSplitter splitting on paragraph / line
    boundaries so that course blocks stay mostly intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "],
    )
    chunks: List[Document] = []
    for doc in docs:
        raw_chunks = splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(raw_chunks):
            meta = extract_section_metadata(chunk_text, doc.metadata["source"])
            meta["chunk_id"] = f"{doc.metadata['source']}::chunk_{i}"
            chunks.append(Document(page_content=chunk_text, metadata=meta))
    return chunks


def build_index(chunks: List[Document]) -> FAISS:
    """Embed chunks and store in FAISS."""
    print(f"\n  Embedding {len(chunks)} chunks with '{EMBED_MODEL}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    index = FAISS.from_documents(chunks, embeddings)
    return index


def save_index(index: FAISS, chunks: List[Document]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    # Also pickle chunk metadata for citation lookups
    with open(INDEX_DIR / "chunks_meta.pkl", "wb") as f:
        pickle.dump(
            [{"chunk_id": c.metadata.get("chunk_id"), "text": c.page_content,
              "source": c.metadata.get("source"),
              "source_url": c.metadata.get("source_url"),
              "section": c.metadata.get("section")} for c in chunks],
            f
        )
    print(f"\n  ✅ Index saved to '{INDEX_DIR}/'")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MIT CS Catalog — Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading catalog files …")
    docs = load_catalog_files(CATALOG_DIR)
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"  Total: {len(docs)} files, {total_chars:,} characters")

    print("\n[2/4] Chunking documents …")
    print(f"  Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    chunks = chunk_documents(docs)
    print(f"  Total chunks: {len(chunks)}")

    print("\n[3/4] Building FAISS vector index …")
    index = build_index(chunks)

    print("\n[4/4] Saving index …")
    save_index(index, chunks)

    print("\n Done! Summary:")
    print(f"   • Files ingested : {len(docs)}")
    print(f"   • Chunks created : {len(chunks)}")
    print(f"   • Total chars    : {total_chars:,}")
    print(f"   • Embed model    : {EMBED_MODEL}")
    print(f"   • Index location : {INDEX_DIR}/")


if __name__ == "__main__":
    main()
