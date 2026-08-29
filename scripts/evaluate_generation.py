import json
import time
from pathlib import Path

from scripts.eval_utils import Retrievers, load_benchmark, run_retrieval
from src.generation.generator import generate_answer

OUTPUT_PATH = Path("scripts/generation_eval_results.json")


def main() -> None:
    benchmark = load_benchmark()
    retrievers = Retrievers()

    results = []
    for i, item in enumerate(benchmark):
        if i > 0:
            time.sleep(5)  # stay under gemini-3.5-flash-lite's 15 RPM (2 calls/query)

        chunks = run_retrieval(
            item["query"], retrievers, use_expansion=True, use_hybrid=True, use_rerank=True
        )
        answer = generate_answer(item["query"], chunks)

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "ground_truth_summary": item["ground_truth_summary"],
                "answer": answer,
                "retrieved_sources": [
                    {"source_file": c.source_file, "article_no": c.article_no} for c in chunks
                ],
            }
        )
        print(f"[{item['id']}] done ({i + 1}/{len(benchmark)})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
