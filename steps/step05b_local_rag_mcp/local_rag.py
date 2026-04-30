from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextChunk:
    source: str
    text: str


class LocalRagRetriever:
    def __init__(self, paths: List[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        self._chunks = self._load_chunks(paths)
        if not self._chunks:
            raise ValueError("No local documents found for local RAG.")

        self._backend = "semantic"
        self._model = None
        self._index = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        try:
            self._model = SentenceTransformer(model_name)
            embeddings = self._model.encode(
                [chunk.text for chunk in self._chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
        except Exception:
            # Offline fallback: lexical retrieval with no external downloads.
            self._backend = "tfidf"
            self._tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), stop_words="english"
            )
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(
                [chunk.text for chunk in self._chunks]
            )

    def search(self, query: str, top_k: int = 3) -> List[TextChunk]:
        if self._backend == "semantic":
            query_embedding = self._model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            _, indices = self._index.search(query_embedding, top_k)
            return [
                self._chunks[i]
                for i in indices[0].tolist()
                if isinstance(i, int) and 0 <= i < len(self._chunks)
            ]

        # TF-IDF fallback backend.
        query_vector = self._tfidf_vectorizer.transform([query])
        scores = (query_vector @ self._tfidf_matrix.T).toarray()[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._chunks[int(i)] for i in top_indices if scores[int(i)] > 0]

    def _load_chunks(self, paths: List[str]) -> List[TextChunk]:
        files: List[Path] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_dir():
                files.extend(path.rglob("*.md"))
            elif path.is_file():
                files.append(path)

        unique_files = sorted(set(files))
        chunks: List[TextChunk] = []
        for file_path in unique_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunks.extend(self._chunk_markdown(content, str(file_path)))
        return chunks

    def _chunk_markdown(self, content: str, source: str) -> List[TextChunk]:
        raw_blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
        chunks: List[TextChunk] = []
        current = ""
        for block in raw_blocks:
            if len(current) + len(block) + 2 <= 900:
                current = f"{current}\n\n{block}".strip()
                continue
            if current:
                chunks.append(TextChunk(source=source, text=current))
            current = block
        if current:
            chunks.append(TextChunk(source=source, text=current))
        return chunks


def build_local_retriever_from_env() -> LocalRagRetriever:
    configured_paths = os.getenv("LOCAL_RAG_PATHS", "WORKSHOP.md,docs")
    paths = [p.strip() for p in configured_paths.split(",") if p.strip()]
    model_name = os.getenv("LOCAL_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return LocalRagRetriever(paths=paths, model_name=model_name)
