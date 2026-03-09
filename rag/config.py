"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class AppConfig:
    base_url: str
    api_token: str | None
    model_id: str | None
    classifier_model: str | None        # separate model for classification; defaults to model_id
    embedding_model: str
    reranker_model: str
    chunk_size: int
    overlap: int
    dense_top_k: int
    sparse_top_k: int
    fused_top_k: int
    context_top_k: int
    rrf_k: int
    temperature: float
    max_tokens: int
    seed: int | None
    timeout_sec: int
    use_env_proxy: bool
    disable_thinking: bool
    head_chunk_boost_count: int
    min_rerank_score: float             # kept for LexicalFallbackRetriever
    min_confidence: float               # threshold for StructuredAnswer.confidence
    document_language: str             # "auto" | "ru" | "en" | …
    adaptive_retrieval_enabled: bool = True
    evidence_max_sentences: int = 10
    list_coverage_min_items: int = 8
    enable_answer_verifier: bool = False
    verifier_max_tokens: int = 320
    verifier_length_threshold: int = 420

    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv()

        base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL is required in .env")

        token = os.getenv("OLLAMA_API_TOKEN", "").strip() or None
        model_id = os.getenv("OLLAMA_MODEL", "").strip() or None
        classifier_model = os.getenv("CLASSIFIER_MODEL", "").strip() or None
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
        reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()

        seed_raw = os.getenv("OLLAMA_SEED", "").strip()
        seed = int(seed_raw) if seed_raw else None

        return AppConfig(
            base_url=base_url.rstrip("/"),
            api_token=token,
            model_id=model_id,
            classifier_model=classifier_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1100")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "220")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "40")),
            sparse_top_k=int(os.getenv("SPARSE_TOP_K", "40")),
            fused_top_k=int(os.getenv("FUSED_TOP_K", "30")),
            context_top_k=int(os.getenv("CONTEXT_TOP_K", "5")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0").strip()),
            max_tokens=int(os.getenv("OLLAMA_NUM_PREDICT", "420").strip()),
            seed=seed,
            timeout_sec=int(os.getenv("OLLAMA_TIMEOUT_SEC", "180").strip()),
            use_env_proxy=os.getenv("OLLAMA_USE_ENV_PROXY", "0").strip().lower() in ("1", "true", "yes", "on"),
            disable_thinking=os.getenv("OLLAMA_DISABLE_THINKING", "1").strip().lower() in ("1", "true", "yes", "on"),
            head_chunk_boost_count=int(os.getenv("HEAD_CHUNK_BOOST_COUNT", "3").strip()),
            min_rerank_score=float(os.getenv("MIN_RERANK_SCORE", "0.02").strip()),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.3").strip()),
            document_language=os.getenv("DOCUMENT_LANGUAGE", "auto").strip().lower(),
            adaptive_retrieval_enabled=os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on"),
            evidence_max_sentences=int(os.getenv("EVIDENCE_MAX_SENTENCES", "10").strip()),
            list_coverage_min_items=int(os.getenv("LIST_COVERAGE_MIN_ITEMS", "8").strip()),
            enable_answer_verifier=os.getenv("ENABLE_ANSWER_VERIFIER", "0").strip().lower() in ("1", "true", "yes", "on"),
            verifier_max_tokens=int(os.getenv("VERIFIER_MAX_TOKENS", "320").strip()),
            verifier_length_threshold=int(os.getenv("VERIFIER_LENGTH_THRESHOLD", "420").strip()),
        )
