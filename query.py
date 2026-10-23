"""
Retrieve relevant passages from the index and ask a local LLM (Ollama) to answer.
"""
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"  # any model you have pulled with `ollama pull`

_embedder = None
_collection = None


def _load():
    global _embedder, _collection
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION)
    return _embedder, _collection


def retrieve(question: str, k: int = 6):
    embedder, collection = _load()
    q_emb = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return hits


def answer(question: str, k: int = 6):
    hits = retrieve(question, k=k)
    if not hits:
        return "No relevant passages found.", []

    context = "\n\n".join(
        f"[{i}] ({h['source']}, p. {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )

    system = (
        "You are a research assistant. Answer the question using ONLY the numbered passages. "
        "Cite passages inline like [1] or [2][3]. If the passages do not contain the answer, "
        "say so clearly instead of guessing. Do not invent references or numbers."
    )
    user = f"Passages:\n{context}\n\nQuestion: {question}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"], hits


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or input("Question: ")
    text, hits = answer(q)
    print(text)
    print("\nSources:")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] {h['source']}, p. {h['page']}")
