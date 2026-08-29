import json

from src.ingestion.build_index import CHUNKS_PATH, Chunk, build_bm25_index

with open(CHUNKS_PATH, encoding="utf-8") as f:
    raw_chunks = json.load(f)

chunks = [
    Chunk(
        text=c["text"],
        source_file=c["source_file"],
        category=c["category"],
        section=c["section"],
        article_no=c["article_no"],
    )
    for c in raw_chunks
]

print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
build_bm25_index(chunks)
print("BM25 index built.")
