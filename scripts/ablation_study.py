import time

from scripts.eval_utils import (
    Retrievers,
    load_benchmark,
    precision_at_k,
    reciprocal_rank,
    run_retrieval,
)

CONFIGURATIONS = [
    ("A: naive single-retriever (dense only, no expansion, no rerank)",
     {"use_expansion": False, "use_hybrid": False, "use_rerank": False}),
    ("B: hybrid retrieval, no reranking (expansion + dense + sparse + RRF)",
     {"use_expansion": True, "use_hybrid": True, "use_rerank": False}),
    ("C: full pipeline (expansion + hybrid + RRF + rerank)",
     {"use_expansion": True, "use_hybrid": True, "use_rerank": True}),
]


def main() -> None:
    benchmark = load_benchmark()
    retrievers = Retrievers()

    print(f"Ablation study over {len(benchmark)} benchmark queries\n")

    results = {}
    for label, flags in CONFIGURATIONS:
        precisions, ranks = [], []
        for item in benchmark:
            if flags["use_expansion"]:
                # gemini-3.5-flash-lite free tier is 15 RPM; pace calls to stay under it
                time.sleep(4)
            chunks = run_retrieval(item["query"], retrievers, **flags)
            precisions.append(
                precision_at_k(chunks, item["expected_source_files"], item["expected_articles"], 5)
            )
            ranks.append(
                reciprocal_rank(chunks, item["expected_source_files"], item["expected_articles"])
            )
        mean_p = sum(precisions) / len(precisions)
        mean_rr = sum(ranks) / len(ranks)
        results[label] = (mean_p, mean_rr)
        print(f"{label}\n  Precision@5 = {mean_p:.3f}   MRR = {mean_rr:.3f}\n")

    print("=== Summary ===")
    for label, (p, rr) in results.items():
        print(f"  {label[:3]}  P@5={p:.3f}  MRR={rr:.3f}")


if __name__ == "__main__":
    main()
