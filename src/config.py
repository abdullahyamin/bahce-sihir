from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str
    gemini_model: str = "gemini-3.5-flash-lite"

    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    embeddings_path: str = "data/processed/embeddings.npy"
    bm25_index_path: str = "data/processed/bm25.pkl"
    chunks_path: str = "data/processed/chunks.json"

    query_expansion_n: int = 3
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rerank_top_k: int = 5
    rerank_candidate_pool_size: int = 6
    # Cross-encoder raw scores are a relative ranking signal, not a calibrated absolute
    # probability - a threshold of 0.0 was observed to silently discard ALL candidates
    # (including the correct document ranked #1) for some legitimate queries. Kept as a
    # sanity-check floor rather than a meaningful relevance gate; trust top-k reranked
    # order instead of an absolute cutoff.
    min_reranker_score: float = -100.0
    rrf_k: int = 60

    cors_allow_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
