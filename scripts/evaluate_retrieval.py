import time

from scripts.eval_utils import (
    Retrievers,
    is_relevant,
    is_relevant_strict,
    load_benchmark,
    precision_at_k,
    reciprocal_rank,
    run_retrieval,
)


def main() -> None:
    benchmark = load_benchmark()
    print(f"Loaded {len(benchmark)} benchmark queries")

    retrievers = Retrievers()

    precisions, precisions_strict = [], []
    reciprocal_ranks, reciprocal_ranks_strict = [], []
    per_category: dict[str, list[float]] = {}

    for item in benchmark:
        query = item["query"]
        expected_files = item["expected_source_files"]
        expected_articles = item["expected_articles"]

        chunks = run_retrieval(
            query, retrievers, use_expansion=True, use_hybrid=True, use_rerank=True
        )
        time.sleep(5)  # stay under Gemini free tier's 15 RPM (query_expansion makes 1 call/query)

        p_at_5 = precision_at_k(chunks, expected_files, expected_articles, 5)
        p_at_5_strict = precision_at_k(chunks, expected_files, expected_articles, 5, matcher=is_relevant_strict)
        rr = reciprocal_rank(chunks, expected_files, expected_articles)
        rr_strict = reciprocal_rank(chunks, expected_files, expected_articles, matcher=is_relevant_strict)

        precisions.append(p_at_5)
        precisions_strict.append(p_at_5_strict)
        reciprocal_ranks.append(rr)
        reciprocal_ranks_strict.append(rr_strict)
        per_category.setdefault(item["category"], []).append(p_at_5)

        hit_marks = [
            ("X" if is_relevant_strict(c, expected_files, expected_articles) else "x")
            if is_relevant(c, expected_files, expected_articles) else "."
            for c in chunks
        ]
        print(f"[{item['id']:4}] P@5={p_at_5:.2f} RR={rr:.2f} hits={''.join(hit_marks):5}  {query[:55]}")

    n = len(benchmark)
    print()
    print("(X = correct document + article, x = correct document only, . = irrelevant)")
    print()
    print(f"Mean Precision@5 (document-level): {sum(precisions) / n:.3f}")
    print(f"Mean Precision@5 (strict, article) : {sum(precisions_strict) / n:.3f}")
    print(f"MRR (document-level)              : {sum(reciprocal_ranks) / n:.3f}")
    print(f"MRR (strict, article)             : {sum(reciprocal_ranks_strict) / n:.3f}")
    print()
    print("Precision@5 (document-level) by category:")
    for cat, vals in per_category.items():
        print(f"  {cat:12}: {sum(vals) / len(vals):.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
