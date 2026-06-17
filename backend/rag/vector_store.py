"""
Vector store — LangChain Chroma + HuggingFaceEmbeddings.

Wraps langchain_chroma.Chroma to manage PGx knowledge base embeddings.
Exposes the same interface as the previous hand-written version so
retriever.py needs no changes.
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DB_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, EMBEDDING_DEVICE


class VectorStore:
    """ChromaDB wrapper using LangChain's Chroma integration."""

    def __init__(self, persist_dir: str | Path = CHROMA_DB_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.available = False
        self.vectorstore = None

        # Init embeddings via LangChain's HuggingFace wrapper
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={
                    "device": EMBEDDING_DEVICE,
                    "local_files_only": True,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            print(f"[WARN] Embedding model '{EMBEDDING_MODEL}' failed to load: {e}")
            print("[WARN] RAG will be unavailable. Falling back to structured-only mode.")
            return

        # Init Chroma via LangChain
        try:
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
                collection_metadata={
                    "description": "PGx knowledge base (CPIC, FDA, clinical guidelines)",
                },
            )
            self.available = True
        except Exception as e:
            print(f"[WARN] ChromaDB initialization failed: {e}")
            print("[WARN] Vector store unavailable. RAG will be skipped.")

    def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """Add text chunks to ChromaDB. Returns count added."""
        if not self.available or self.vectorstore is None:
            print("[WARN] Vector store unavailable. Cannot add chunks.")
            return 0
        try:
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            return len(chunks)
        except Exception as e:
            print(f"[WARN] Failed to add chunks: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Search by semantic similarity. Returns list of result dicts."""
        if not self.available or self.vectorstore is None:
            print("[WARN] Vector store unavailable. Returning empty results.")
            return []
        try:
            # similarity_search_with_score returns List[Tuple[Document, float]]
            # where score is L2/cosine distance (lower = more similar)
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=where,
            )
            output = []
            for i, (doc, score) in enumerate(results):
                # Stable ID from content hash (for dedup in retriever)
                c_hash = hashlib.md5(doc.page_content[:80].encode()).hexdigest()[:8]
                doc_id = doc.metadata.get("id", f"chunk:{c_hash}")
                output.append({
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": float(score),
                })
            return output
        except Exception as e:
            print(f"[WARN] Vector search failed: {e}")
            return []

    def count(self) -> int:
        """Return total chunk count."""
        if not self.available or self.vectorstore is None:
            return 0
        try:
            return len(self.vectorstore.get()["ids"])
        except Exception:
            return 0

    def delete_collection(self):
        """Drop and recreate the collection."""
        if not self.available or self.vectorstore is None:
            return
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
        except Exception as e:
            print(f"[WARN] Failed to delete collection: {e}")
