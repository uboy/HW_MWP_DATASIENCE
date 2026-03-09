"""Strategy-aware retriever: uses QueryPlan to choose retrieval strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .classifier import QueryPlan, sanitize_query_for_retrieval
from .config import AppConfig
from .index import IndexStore, tokenize

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", flags=re.UNICODE)
LIST_QUESTION_RE = re.compile(r"^\s*(какие|which|what\s+are)\b", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!;])\s+")
LIST_ITEM_PREFIX_RE = re.compile(r"^(?:[а-яёa-z]\)|\d+\))", flags=re.IGNORECASE)
WHY_QUESTION_RE = re.compile(r"^\s*(почему|why)\b", flags=re.IGNORECASE)
LEGAL_QUERY_RE = re.compile(r"\b(?:закон|законы|фз|law|laws)\b", flags=re.IGNORECASE)
STOPWORDS = {
    "и", "в", "во", "на", "по", "что", "как", "какой", "какие", "какая", "какую",
    "кто", "где", "для", "это", "эта", "этот", "эти", "или", "ли", "из", "под",
    "при", "его", "ее", "их", "the", "what", "which", "how", "when", "where", "why",
}

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


@dataclass
class RetrievedContext:
    chunk_id: int
    text: str
    rerank_score: float


@dataclass(frozen=True)
class RetrievalPolicy:
    pool: int
    fused_limit: int
    rerank_top_k: int
    context_limit: int
    max_units_per_context: int
    unit_budget: int
    list_min_items: int = 0


class StrategyRetriever:
    """Retrieves chunks using dense + sparse + structural + head-boost, fused via RRF."""

    def __init__(self, index: IndexStore, config: AppConfig) -> None:
        self.index = index
        self.config = config
        self._reranker = None
        if _CROSS_ENCODER_AVAILABLE:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.reranker_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dense_search(self, query: str, top_k: int) -> list[int]:
        """Exposed for parallel execution in pipeline."""
        return self.index.dense_search(query, top_k)

    def retrieve(self, query: str, plan: QueryPlan) -> list[RetrievedContext]:
        return self._retrieve_with_dense(query, plan, dense_ids=None)

    def retrieve_with_dense(
        self, query: str, plan: QueryPlan, dense_ids: list[int]
    ) -> list[RetrievedContext]:
        return self._retrieve_with_dense(query, plan, dense_ids=dense_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve_with_dense(
        self, query: str, plan: QueryPlan, dense_ids: list[int] | None
    ) -> list[RetrievedContext]:
        effective_query = sanitize_query_for_retrieval(query) if plan.is_prompt_injection else query
        n_chunks = len(self.index.chunks)
        policy = self._policy_for(query, plan, n_chunks)
        pool = policy.pool
        candidate_queries = self._candidate_queries(query, plan)
        rank_lists: list[list[int]] = []

        primary_query = candidate_queries[0] if candidate_queries else effective_query
        if dense_ids is None or primary_query != effective_query or len(dense_ids) < pool:
            dense_ids = self.index.dense_search(primary_query, pool)
        if dense_ids:
            rank_lists.append(dense_ids)
        primary_sparse_ids = self.index.sparse_search(primary_query, pool)
        if primary_sparse_ids:
            rank_lists.append(primary_sparse_ids)

        for candidate_query in candidate_queries[1:]:
            dense_variant = self.index.dense_search(candidate_query, pool)
            sparse_variant = self.index.sparse_search(candidate_query, pool)
            if dense_variant:
                rank_lists.append(dense_variant)
            if sparse_variant:
                rank_lists.append(sparse_variant)

        # Structural: exact paragraph match — highest priority
        if plan.paragraph_refs:
            structural_ids = self.index.structural.search(plan.paragraph_refs)
            if structural_ids:
                rank_lists.insert(0, structural_ids)

        # Head-chunk boost (document title / header)
        if plan.boost_early_chunks:
            head_ids = list(range(min(self.config.head_chunk_boost_count, n_chunks)))
            rank_lists.append(head_ids)

        fused_ids = self._rrf_fuse(rank_lists, limit=policy.fused_limit)
        if not fused_ids:
            return []

        rerank_query = primary_query
        contexts = self._rerank(rerank_query, fused_ids, top_k=policy.rerank_top_k)
        contexts = self._expand_contexts(rerank_query, plan, contexts)
        contexts = self._ensure_list_coverage(rerank_query, plan, contexts, fused_ids, policy)
        return self._pack_contexts(query, plan, contexts, policy)

    @staticmethod
    def _candidate_queries(query: str, plan: QueryPlan) -> list[str]:
        base_query = sanitize_query_for_retrieval(query) if plan.is_prompt_injection else query
        candidates = [base_query] + list(plan.search_queries)
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" ?!.,:;")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized

    def _rrf_fuse(self, rank_lists: list[list[int]], limit: int | None = None) -> list[int]:
        fused: Dict[int, float] = {}
        for rank_list in rank_lists:
            for rank, doc_id in enumerate(rank_list, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.config.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        final_limit = self.config.fused_top_k if limit is None else limit
        return [doc_id for doc_id, _ in ranked[:final_limit]]

    def _rerank(self, query: str, doc_ids: list[int], top_k: int) -> list[RetrievedContext]:
        texts = self.index.chunk_texts
        if self._reranker is not None:
            pairs = [[query, texts[doc_id]] for doc_id in doc_ids]
            scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            scored = sorted(zip(doc_ids, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        else:
            # Fallback: overlap-based scoring
            q_tokens = set(tokenize(query))
            scored_list: list[tuple[int, float]] = []
            for doc_id in doc_ids:
                overlap = len(q_tokens & set(tokenize(texts[doc_id])))
                scored_list.append((doc_id, float(overlap)))
            scored = sorted(scored_list, key=lambda x: x[1], reverse=True)

        return [
            RetrievedContext(chunk_id=doc_id, text=texts[doc_id], rerank_score=score)
            for doc_id, score in scored[:top_k]
        ]

    def _expand_contexts(
        self,
        query: str,
        plan: QueryPlan,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        if not contexts:
            return contexts
        if not self._should_expand_context(query, plan):
            return self._merge_paragraph_contexts(contexts, plan)

        expanded_ids = [context.chunk_id for context in contexts]
        seen = set(expanded_ids)

        seed_count = 3 if plan.expects_exhaustive_list or plan.query_type == "analytical" else 2
        for context in contexts[:seed_count]:
            metadata = self.index.chunks[context.chunk_id].metadata
            paragraph_numbers = tuple(getattr(metadata, "paragraph_numbers", ()))
            if not paragraph_numbers and metadata.paragraph_number is not None:
                paragraph_numbers = (metadata.paragraph_number,)
            anchor_paragraph = self._anchor_paragraph_number(paragraph_numbers, plan)
            if anchor_paragraph is None:
                continue
            for chunk_id in self.index.structural.chunk_ids_for_paragraph(anchor_paragraph):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                expanded_ids.append(chunk_id)

        if len(expanded_ids) == len(contexts):
            return contexts

        expanded_top_k = min(
            len(expanded_ids),
            max(plan.top_k + 4, self.config.context_top_k + 4),
        )
        expanded_contexts = self._rerank(query, expanded_ids, top_k=expanded_top_k)
        return self._merge_paragraph_contexts(expanded_contexts, plan)

    @staticmethod
    def _should_expand_context(query: str, plan: QueryPlan) -> bool:
        if (
            plan.paragraph_refs
            or plan.query_type in {"analytical", "definitional"}
            or plan.expects_exhaustive_list
        ):
            return True
        return bool(LIST_QUESTION_RE.match(query))

    def _merge_paragraph_contexts(
        self,
        contexts: list[RetrievedContext],
        plan: QueryPlan,
    ) -> list[RetrievedContext]:
        if not contexts:
            return contexts
        if not (
            plan.paragraph_refs
            or plan.query_type in {"analytical", "definitional"}
            or plan.expects_exhaustive_list
        ):
            return contexts

        merged: list[RetrievedContext] = []
        seen_paragraphs: set[int] = set()
        seen_chunks: set[int] = set()

        for context in contexts:
            if context.chunk_id in seen_chunks:
                continue
            metadata = self.index.chunks[context.chunk_id].metadata
            paragraph_numbers = tuple(getattr(metadata, "paragraph_numbers", ()))
            if not paragraph_numbers and metadata.paragraph_number is not None:
                paragraph_numbers = (metadata.paragraph_number,)

            paragraph_number = self._anchor_paragraph_number(paragraph_numbers, plan)
            if paragraph_number is not None:
                if paragraph_number in seen_paragraphs:
                    continue
                sibling_ids = self.index.structural.chunk_ids_for_paragraph(paragraph_number)
                if len(sibling_ids) > 1:
                    if plan.query_type == "definitional" and context.chunk_id in sibling_ids:
                        current_idx = sibling_ids.index(context.chunk_id)
                        sibling_ids = sibling_ids[current_idx : current_idx + 3]
                    merged_text = self._merge_chunk_texts(
                        [self.index.chunk_texts[chunk_id] for chunk_id in sibling_ids]
                    )
                    merged.append(
                        RetrievedContext(
                            chunk_id=context.chunk_id,
                            text=merged_text,
                            rerank_score=context.rerank_score,
                        )
                    )
                    seen_paragraphs.add(paragraph_number)
                    seen_chunks.update(sibling_ids)
                    continue
                seen_paragraphs.add(paragraph_number)

            merged.append(context)
            seen_chunks.add(context.chunk_id)

        limit = max(plan.top_k, self.config.context_top_k)
        if plan.expects_exhaustive_list:
            limit = max(limit, self.config.context_top_k + 4)
        return merged[:limit]

    def _policy_for(
        self,
        query: str,
        plan: QueryPlan,
        n_chunks: int,
    ) -> RetrievalPolicy:
        if not self.config.adaptive_retrieval_enabled:
            base_limit = max(plan.top_k, self.config.context_top_k)
            return RetrievalPolicy(
                pool=min(n_chunks, self.config.dense_top_k * 5),
                fused_limit=min(n_chunks, self.config.fused_top_k),
                rerank_top_k=min(n_chunks, base_limit),
                context_limit=min(n_chunks, base_limit),
                max_units_per_context=2,
                unit_budget=self.config.evidence_max_sentences,
                list_min_items=self.config.list_coverage_min_items if plan.expects_exhaustive_list else 0,
            )

        if plan.expects_exhaustive_list:
            pool_factor = 7
            rerank_top_k = max(12, plan.top_k + 2)
            context_limit = max(4, min(6, self.config.context_top_k + 1))
            max_units_per_context = 12
            unit_budget = max(self.config.evidence_max_sentences, 14)
        elif plan.query_type == "factoid":
            pool_factor = 4
            rerank_top_k = max(6, min(8, plan.top_k + 1))
            context_limit = 3
            max_units_per_context = 1
            unit_budget = min(self.config.evidence_max_sentences, 4)
        elif plan.query_type == "definitional":
            pool_factor = 5
            rerank_top_k = max(8, min(10, plan.top_k + 1))
            context_limit = 2
            max_units_per_context = 3
            unit_budget = max(6, min(self.config.evidence_max_sentences, 8))
        elif plan.query_type == "structural":
            pool_factor = 4
            rerank_top_k = max(7, min(10, plan.top_k + 1))
            context_limit = 2
            max_units_per_context = 3
            unit_budget = max(5, min(self.config.evidence_max_sentences, 7))
        else:
            pool_factor = 6
            rerank_top_k = max(10, min(14, plan.top_k + 2))
            context_limit = 4
            max_units_per_context = 3
            unit_budget = max(8, self.config.evidence_max_sentences)

        if LIST_QUESTION_RE.match(query) and not plan.expects_exhaustive_list:
            context_limit = max(context_limit, 3)
            max_units_per_context = max(max_units_per_context, 2)

        pool = min(n_chunks, max(self.config.dense_top_k * pool_factor, rerank_top_k * 4))
        fused_limit = min(n_chunks, max(self.config.fused_top_k, rerank_top_k + 8))
        return RetrievalPolicy(
            pool=pool,
            fused_limit=fused_limit,
            rerank_top_k=min(n_chunks, rerank_top_k),
            context_limit=min(n_chunks, context_limit),
            max_units_per_context=max_units_per_context,
            unit_budget=unit_budget,
            list_min_items=self.config.list_coverage_min_items if plan.expects_exhaustive_list else 0,
        )

    def _ensure_list_coverage(
        self,
        query: str,
        plan: QueryPlan,
        contexts: list[RetrievedContext],
        fused_ids: list[int],
        policy: RetrievalPolicy,
    ) -> list[RetrievedContext]:
        if not plan.expects_exhaustive_list or not contexts:
            return contexts
        if self._has_sufficient_list_coverage(contexts, policy.list_min_items):
            return contexts

        expanded_ids = [context.chunk_id for context in contexts]
        seen = set(expanded_ids)
        expand_limit = max(policy.rerank_top_k + 4, policy.context_limit + 4)
        for doc_id in fused_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            expanded_ids.append(doc_id)
            if len(expanded_ids) >= expand_limit:
                break

        if len(expanded_ids) == len(contexts):
            return contexts

        original_units = self._count_list_units(contexts)
        expanded_contexts = self._rerank(
            query,
            expanded_ids,
            top_k=min(len(expanded_ids), expand_limit),
        )
        expanded_contexts = self._expand_contexts(query, plan, expanded_contexts)
        expanded_units = self._count_list_units(expanded_contexts)
        if self._has_sufficient_list_coverage(expanded_contexts, policy.list_min_items):
            return expanded_contexts
        if original_units == 0 and expanded_units > 0:
            return expanded_contexts
        return contexts

    def _pack_contexts(
        self,
        query: str,
        plan: QueryPlan,
        contexts: list[RetrievedContext],
        policy: RetrievalPolicy,
    ) -> list[RetrievedContext]:
        if not contexts:
            return contexts
        if not self.config.adaptive_retrieval_enabled:
            return contexts[: policy.context_limit]

        packed: list[RetrievedContext] = []
        used_units = 0
        for context in contexts:
            packed_text, unit_count = self._pack_context_text(query, plan, context.text, policy)
            if not packed_text:
                continue
            packed.append(
                RetrievedContext(
                    chunk_id=context.chunk_id,
                    text=packed_text,
                    rerank_score=context.rerank_score,
                )
            )
            used_units += max(unit_count, 1)
            if len(packed) >= policy.context_limit or used_units >= policy.unit_budget:
                break

        return packed or contexts[: policy.context_limit]

    def _pack_context_text(
        self,
        query: str,
        plan: QueryPlan,
        text: str,
        policy: RetrievalPolicy,
    ) -> tuple[str, int]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return "", 0

        if (
            plan.paragraph_refs
            or plan.query_type in {"definitional", "structural"}
            or LEGAL_QUERY_RE.search(query)
            or (WHY_QUESTION_RE.match(query) and re.search(r"\d", query))
        ):
            unit_count = min(policy.max_units_per_context, max(1, len(self._split_sentences(normalized))))
            return normalized, unit_count

        if plan.expects_exhaustive_list:
            items = self._extract_list_items(normalized)
            if items:
                selected_items = items[: policy.max_units_per_context]
                return "; ".join(selected_items), len(selected_items)

        focus_terms = self._focus_terms(query, plan)
        number_tokens = set(re.findall(r"\d+(?:[.,]\d+)?", query))
        sentences = self._split_sentences(normalized)
        if len(sentences) <= policy.max_units_per_context:
            return normalized, len(sentences)

        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, focus_terms, number_tokens, plan)
            scored.append((score, index, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [
            sentence
            for score, _, sentence in scored[: policy.max_units_per_context]
            if score > 0
        ]
        if not selected:
            selected = sentences[: policy.max_units_per_context]
        return " ".join(selected).strip(), len(selected)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        return [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]

    @staticmethod
    def _extract_list_items(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        parts = [part.strip(" ;") for part in normalized.split(";") if part.strip(" ;")]
        if len(parts) < 2:
            return []

        items: list[str] = []
        seen: set[str] = set()
        for part in parts:
            if len(parts) >= 2 or LIST_ITEM_PREFIX_RE.match(part):
                cleaned = part.strip(" .")
                key = cleaned.lower()
                if not cleaned or key in seen:
                    continue
                seen.add(key)
                items.append(cleaned)
        return items

    @staticmethod
    def _focus_terms(query: str, plan: QueryPlan) -> set[str]:
        tokens = [token.lower() for token in plan.key_tokens if token]
        if len(tokens) < 5:
            for token in WORD_RE.findall(query.lower()):
                if token in STOPWORDS or len(token) < 3 or token.isdigit():
                    continue
                if token not in tokens:
                    tokens.append(token)
        return set(tokens)

    @staticmethod
    def _score_sentence(
        sentence: str,
        focus_terms: set[str],
        number_tokens: set[str],
        plan: QueryPlan,
    ) -> float:
        lowered = sentence.lower()
        tokens = {token for token in WORD_RE.findall(lowered) if token not in STOPWORDS}
        score = float(len(tokens & focus_terms) * 3)
        if number_tokens and any(number in lowered for number in number_tokens):
            score += 2.0
        if plan.query_type == "factoid" and re.search(r"\d", sentence):
            score += 1.0
        if plan.expects_exhaustive_list and LIST_ITEM_PREFIX_RE.match(sentence.strip()):
            score += 2.0
        return score

    @classmethod
    def _count_list_units(cls, contexts: list[RetrievedContext]) -> int:
        total = 0
        seen: set[str] = set()
        for context in contexts:
            for item in cls._extract_list_items(context.text):
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                total += 1
        return total

    @classmethod
    def _has_sufficient_list_coverage(
        cls,
        contexts: list[RetrievedContext],
        minimum_items: int,
    ) -> bool:
        if minimum_items <= 0:
            return True
        return cls._count_list_units(contexts) >= minimum_items

    @staticmethod
    def _merge_chunk_texts(texts: list[str]) -> str:
        if not texts:
            return ""
        merged = texts[0].strip()
        for text in texts[1:]:
            merged = StrategyRetriever._append_unique_text(merged, text)
        return merged

    @staticmethod
    def _append_unique_text(current: str, next_text: str) -> str:
        current_text = current.strip()
        appended_text = next_text.strip()
        if not current_text:
            return appended_text
        if not appended_text:
            return current_text

        next_words = appended_text.split()
        search_window = current_text[-800:]
        max_words = min(len(next_words), 40)
        for size in range(max_words, 7, -1):
            prefix = " ".join(next_words[:size])
            if prefix and prefix in search_window:
                suffix = " ".join(next_words[size:]).strip()
                return current_text if not suffix else f"{current_text} {suffix}"

        max_chars = min(len(current_text), len(appended_text), 220)
        for size in range(max_chars, 39, -1):
            if current_text[-size:] == appended_text[:size]:
                suffix = appended_text[size:].strip()
                return current_text if not suffix else f"{current_text} {suffix}"

        return f"{current_text} {appended_text}"

    @staticmethod
    def _anchor_paragraph_number(
        paragraph_numbers: tuple[int, ...],
        plan: QueryPlan,
    ) -> int | None:
        if not paragraph_numbers:
            return None
        for paragraph_ref in plan.paragraph_refs:
            if paragraph_ref in paragraph_numbers:
                return paragraph_ref
        if len(paragraph_numbers) == 1:
            return paragraph_numbers[0]
        return paragraph_numbers[-1]


# ---------------------------------------------------------------------------
# Lexical fallback retriever (no ML dependencies required)
# ---------------------------------------------------------------------------

class LexicalFallbackRetriever:
    """TF-IDF based retriever for restricted environments without ML libs."""

    def __init__(self, chunks: list[str], config: AppConfig) -> None:
        if not chunks:
            raise ValueError("No chunks found in the source document")
        self.chunks = list(chunks)
        self.config = config
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._token_sets: list[set[str]] = []
        self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        tokenized_docs = [tokenize(chunk) for chunk in self.chunks]
        self._token_sets = [set(toks) for toks in tokenized_docs]
        df: Dict[str, int] = {}
        for toks in tokenized_docs:
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1
            for t in toks:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

        n_docs = len(self.chunks)
        n_vocab = len(self._vocab)
        idf = np.zeros(n_vocab, dtype=np.float32)
        for tok, idx in self._vocab.items():
            idf[idx] = np.log((1.0 + n_docs) / (1.0 + df.get(tok, 0))) + 1.0

        matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for row, toks in enumerate(tokenized_docs):
            if not toks:
                continue
            tf: Dict[int, int] = {}
            for tok in toks:
                col = self._vocab[tok]
                tf[col] = tf.get(col, 0) + 1
            max_tf = max(tf.values())
            for col, cnt in tf.items():
                matrix[row, col] = (cnt / max_tf) * idf[col]

        self._idf = idf
        self._matrix = matrix
        self._norms = np.linalg.norm(matrix, axis=1)

    def _query_vector(self, query: str) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("TF-IDF index not initialized")
        vec = np.zeros_like(self._idf, dtype=np.float32)
        toks = tokenize(query)
        tf: Dict[int, int] = {}
        for t in toks:
            col = self._vocab.get(t)
            if col is not None:
                tf[col] = tf.get(col, 0) + 1
        if not tf:
            return vec
        max_tf = max(tf.values())
        for col, cnt in tf.items():
            vec[col] = (cnt / max_tf) * self._idf[col]
        return vec

    def retrieve(self, query: str, plan: QueryPlan | None = None) -> list[RetrievedContext]:
        if self._matrix is None or self._norms is None:
            raise RuntimeError("TF-IDF index not initialized")

        top_k = plan.top_k if plan else self.config.context_top_k
        boost_early = plan.boost_early_chunks if plan else False
        candidate_queries = (
            StrategyRetriever._candidate_queries(query, plan)  # type: ignore[arg-type]
            if plan is not None
            else [query]
        )

        score_map = {idx: 0.0 for idx in range(len(self.chunks))}
        for candidate_query in candidate_queries:
            qvec = self._query_vector(candidate_query)
            qnorm = float(np.linalg.norm(qvec))
            if qnorm == 0:
                continue
            numer = self._matrix @ qvec
            denom = self._norms * qnorm
            scores = np.divide(
                numer, denom,
                out=np.zeros_like(numer, dtype=np.float32),
                where=denom != 0,
            )
            for idx in range(len(scores)):
                score_map[idx] = max(score_map[idx], float(scores[idx]))

        if not any(score > 0 for score in score_map.values()):
            return []

        # Structural boost via paragraph refs
        if plan and plan.paragraph_refs:
            q_tokens = set()
            for candidate_query in candidate_queries:
                q_tokens.update(tokenize(candidate_query))
            for idx, chunk in enumerate(self.chunks):
                chunk_tokens = set(tokenize(chunk))
                if q_tokens & chunk_tokens:
                    score_map[idx] = score_map.get(idx, 0.0) + 0.3

        # Head-chunk boost
        if boost_early:
            for rank, chunk_id in enumerate(
                range(min(self.config.head_chunk_boost_count, len(self.chunks))),
                start=1,
            ):
                score_map[chunk_id] = score_map.get(chunk_id, 0.0) + 0.2 / rank

        ranked = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)[:top_k]
        return [
            RetrievedContext(
                chunk_id=int(cid),
                text=self.chunks[int(cid)],
                rerank_score=score_map[int(cid)],
            )
            for cid in ranked
            if score_map[int(cid)] > 0
        ]
