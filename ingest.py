"""
Ingest PDFs from ./papers into a local ChromaDB collection.

Usage:
    python ingest.py            # rebuilds the index from scratch
"""
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PAPERS_DIR = Path("papers")
DB_DIR = "chroma_db"
COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if piece.strip():
            yield piece
        if start + size >= len(words):
            break


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for each page. Empty text usually means a scanned page."""
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        yield i, (page.extract_text() or "")


def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}/. Add your papers and run again.")
        return

    print(f"Found {len(pdfs)} PDFs. Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    scanned_warnings = []

    for pdf in pdfs:
        empty_pages = 0
        total_pages = 0
        for page_no, text in extract_pages(pdf):
            total_pages += 1
            if not text.strip():
                empty_pages += 1
                continue
            for n, chunk in enumerate(chunk_words(text)):
                ids.append(f"{pdf.stem}-p{page_no}-c{n}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
        if total_pages and empty_pages / total_pages > 0.5:
            scanned_warnings.append(pdf.name)
        print(f"  {pdf.name}: {total_pages} pages")

    if scanned_warnings:
        print("\nWarning: these PDFs appear to be scanned (little or no text). "
              "Run OCR on them first (e.g., ocrmypdf):")
        for name in scanned_warnings:
            print(f"  - {name}")

    if not docs:
        print("No text extracted. Nothing to index.")
        return

    print(f"\nEmbedding {len(docs)} chunks...")
    embeddings = model.encode(docs, batch_size=32, show_progress_bar=True).tolist()

    # Add in batches to stay under Chroma's batch size limit.
    batch = 500
    for i in range(0, len(docs), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=metas[i:i + batch],
            embeddings=embeddings[i:i + batch],
        )

    print(f"Done. Indexed {len(docs)} chunks from {len(pdfs)} PDFs into {DB_DIR}/.")


if __name__ == "__main__":
    main()
