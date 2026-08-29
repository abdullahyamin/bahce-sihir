import json
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunking import Chunk, chunk_document
from src.ingestion.loaders import iter_corpus
from src.text_utils import tokenize_turkish

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25.pkl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


def build_chunks() -> list[Chunk]:
    chunks = []
    for doc in iter_corpus():
        chunks.extend(chunk_document(doc.text, doc.source_file, doc.category))
    return chunks


def build_embeddings_index(chunks: list[Chunk], output_path: Path = EMBEDDINGS_PATH) -> None:
    model = SentenceTransformer(settings.embedding_model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def build_bm25_index(chunks: list[Chunk], output_path: Path = BM25_INDEX_PATH) -> None:
    tokenized = [tokenize_turkish(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)


def save_chunk_metadata(chunks: list[Chunk], output_path: Path = CHUNKS_PATH) -> None:
    metadata = [
        {
            "text": c.text,
            "source_file": c.source_file,
            "category": c.category,
            "section": c.section,
            "article_no": c.article_no,
        }
        for c in chunks
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("Loading and chunking documents...")
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("Saving chunk metadata...")
    save_chunk_metadata(chunks)

    print("Building embeddings index...")
    build_embeddings_index(chunks)

    print("Building BM25 index...")
    build_bm25_index(chunks)

    print("Done.")


if __name__ == "__main__":
    main()
