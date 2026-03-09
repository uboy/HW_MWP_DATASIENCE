"""Тонкий orchestrator для RAG: classify → retrieve → generate."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .classifier import QueryClassifier, QueryPlan
from .client import OllamaClient
from .config import AppConfig
from .document import DocumentProcessor
from .generator import AnswerGenerator
from .grounded_rules import try_answer_with_rules
from .index import IndexStore
from .retriever import LexicalFallbackRetriever, RetrievedContext, StrategyRetriever
from .verifier import AnswerVerifier

log = logging.getLogger(__name__)

_NO_DATA = "В документе не найдено данных для ответа."


class RAGPipeline:
    """Оркестрирует классификацию, retrieval и генерацию ответа."""

    def __init__(
        self,
        config: AppConfig,
        classifier: QueryClassifier,
        retriever: StrategyRetriever | LexicalFallbackRetriever,
        generator: AnswerGenerator,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier

    def answer(self, question: str) -> tuple[str, list[RetrievedContext]]:
        # Run classification and dense search in parallel when StrategyRetriever is used
        if isinstance(self.retriever, StrategyRetriever):
            plan, dense_ids = self._parallel_classify_and_dense(question)
            contexts = self.retriever.retrieve_with_dense(question, plan, dense_ids)
        else:
            plan = self.classifier.classify(question)
            contexts = self.retriever.retrieve(question, plan)

        if not contexts:
            return _NO_DATA, []

        rule_answer = try_answer_with_rules(question, contexts, plan)
        if rule_answer:
            return self._finalize_answer(question, rule_answer, contexts, plan), contexts

        structured = self.generator.generate(question, contexts, plan)

        if structured.confidence < self.config.min_confidence:
            return _NO_DATA, contexts

        return self._finalize_answer(question, structured.answer, contexts, plan), contexts

    def _parallel_classify_and_dense(
        self, question: str
    ) -> tuple[QueryPlan, list[int]]:
        pool = self.config.dense_top_k * 5
        with ThreadPoolExecutor(max_workers=2) as executor:
            classify_future = executor.submit(self.classifier.classify, question)
            dense_future = executor.submit(
                self.retriever.dense_search, question, pool  # type: ignore[union-attr]
            )
            plan = classify_future.result()
            dense_ids = dense_future.result()
        return plan, dense_ids

    def _finalize_answer(
        self,
        question: str,
        answer: str,
        contexts: list[RetrievedContext],
        plan: QueryPlan,
    ) -> str:
        if not answer:
            return answer
        if self.verifier is None:
            return answer
        return self.verifier.maybe_revise(question, answer, contexts, plan)


def build_pipeline(config: AppConfig, pdf_path: Path) -> RAGPipeline:
    """Фабрика: загружает документ, строит индексы и связывает компоненты."""
    print(f"Загрузка документа: {pdf_path}")
    processor = DocumentProcessor(chunk_size=config.chunk_size, overlap=config.overlap)
    chunks = processor.load(pdf_path)
    print(f"Загружено чанков: {len(chunks)}")

    client = OllamaClient(config=config)
    classifier = QueryClassifier(client=client, config=config)

    try:
        index = IndexStore(chunks=chunks, config=config)
        model_id = client.resolve_model(config.model_id)
        if index.has_dense:
            print("Режим ретривера: гибридный (bge-m3 + bm25 + rrf + reranker)")
            retriever: StrategyRetriever | LexicalFallbackRetriever = StrategyRetriever(
                index=index, config=config
            )
        else:
            print("Режим ретривера: лексический резервный режим (нет dense-зависимостей)")
            chunk_texts = [c.text for c in chunks]
            retriever = LexicalFallbackRetriever(chunks=chunk_texts, config=config)
    except Exception as exc:
        log.warning("Не удалось собрать IndexStore (%s); используется лексический резервный режим", exc)
        model_id = client.resolve_model(config.model_id)
        chunk_texts = [c.text for c in chunks]
        retriever = LexicalFallbackRetriever(chunks=chunk_texts, config=config)

    generator = AnswerGenerator(client=client, config=config, model_id=model_id)
    verifier = AnswerVerifier(client=client, config=config, model_id=model_id) if config.enable_answer_verifier else None

    return RAGPipeline(
        config=config,
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        verifier=verifier,
    )
