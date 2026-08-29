import json
from dataclasses import dataclass

from src.config import settings


@dataclass
class ChunkRecord:
    text: str
    source_file: str
    category: str
    section: str | None
    article_no: str | None


class ChunkStore:
    def __init__(self, path: str = settings.chunks_path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [ChunkRecord(**c) for c in raw]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> ChunkRecord:
        return self.chunks[idx]
