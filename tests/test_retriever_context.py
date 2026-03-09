import unittest

from rag.classifier import QueryPlan
from rag.config import AppConfig
from rag.retriever import RetrievalPolicy
from rag.retriever import StrategyRetriever
from rag.retriever import RetrievedContext


def make_config() -> AppConfig:
    return AppConfig(
        base_url="http://localhost:11434",
        api_token=None,
        model_id="qwen",
        classifier_model=None,
        embedding_model="bge",
        reranker_model="reranker",
        chunk_size=1100,
        overlap=220,
        dense_top_k=40,
        sparse_top_k=40,
        fused_top_k=30,
        context_top_k=5,
        rrf_k=60,
        temperature=0.0,
        max_tokens=420,
        seed=7,
        timeout_sec=180,
        use_env_proxy=False,
        disable_thinking=True,
        head_chunk_boost_count=3,
        min_rerank_score=0.02,
        min_confidence=0.3,
        document_language="ru",
        adaptive_retrieval_enabled=True,
        evidence_max_sentences=10,
        list_coverage_min_items=5,
        enable_answer_verifier=False,
        verifier_max_tokens=320,
        verifier_length_threshold=420,
    )


class RetrieverContextTest(unittest.TestCase):
    def test_anchor_paragraph_prefers_explicit_reference_and_otherwise_last(self) -> None:
        structural_plan = QueryPlan(
            query_type="structural",
            top_k=6,
            boost_early_chunks=False,
            paragraph_refs=[1],
            language="ru",
        )
        list_plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            expects_exhaustive_list=True,
        )

        self.assertEqual(
            StrategyRetriever._anchor_paragraph_number((1, 2), structural_plan),
            1,
        )
        self.assertEqual(
            StrategyRetriever._anchor_paragraph_number((26, 27), list_plan),
            27,
        )

    def test_definitional_queries_expand_same_paragraph_context(self) -> None:
        definitional_plan = QueryPlan(
            query_type="definitional",
            top_k=8,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
        )

        self.assertTrue(
            StrategyRetriever._should_expand_context(
                "Что в Стратегии понимается под искусственным интеллектом?",
                definitional_plan,
            )
        )

    def test_merge_chunk_texts_deduplicates_overlap(self) -> None:
        merged = StrategyRetriever._merge_chunk_texts(
            [
                (
                    "а) искусственный интеллект - комплекс технологических решений, позволяющий "
                    "имитировать когнитивные функции человека (включая поиск решений без заранее "
                    "заданного алгоритма) и получать при выполнении конкретных задач результаты, "
                    "сопоставимые с результатами интеллектуальной"
                ),
                (
                    "решений, позволяющий имитировать когнитивные функции человека (включая поиск "
                    "решений без заранее заданного алгоритма) и получать при выполнении конкретных "
                    "задач результаты, сопоставимые с результатами интеллектуальной деятельности "
                    "человека или превосходящие их."
                ),
            ]
        )

        self.assertEqual(merged.count("позволяющий имитировать"), 1)
        self.assertIn("деятельности человека или превосходящие их.", merged)

    def test_adaptive_policy_gives_exhaustive_lists_more_recall_than_factoids(self) -> None:
        retriever = StrategyRetriever.__new__(StrategyRetriever)
        retriever.config = make_config()  # type: ignore[attr-defined]

        factoid_plan = QueryPlan(
            query_type="factoid",
            top_k=6,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["ежегодный", "объем", "услуг"],
        )
        list_plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["направления", "стимулирования", "внедрения"],
            expects_exhaustive_list=True,
        )

        factoid_policy = StrategyRetriever._policy_for(
            retriever,
            "Какой целевой ежегодный объем услуг по ИИ установлен на 2030 год?",
            factoid_plan,
            500,
        )
        list_policy = StrategyRetriever._policy_for(
            retriever,
            "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            list_plan,
            500,
        )

        self.assertGreater(list_policy.pool, factoid_policy.pool)
        self.assertGreaterEqual(list_policy.rerank_top_k, factoid_policy.rerank_top_k)
        self.assertGreater(list_policy.max_units_per_context, factoid_policy.max_units_per_context)

    def test_factoid_context_packing_prefers_requested_metric_sentence(self) -> None:
        retriever = StrategyRetriever.__new__(StrategyRetriever)
        retriever.config = make_config()  # type: ignore[attr-defined]
        plan = QueryPlan(
            query_type="factoid",
            top_k=6,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["ежегодный", "объем", "услуг"],
        )
        policy = RetrievalPolicy(
            pool=40,
            fused_limit=16,
            rerank_top_k=8,
            context_limit=3,
            max_units_per_context=1,
            unit_budget=4,
        )
        packed, unit_count = StrategyRetriever._pack_context_text(
            retriever,
            "Какой целевой ежегодный объем услуг по ИИ установлен на 2030 год?",
            plan,
            (
                "б) совокупный прирост валового внутреннего продукта в 2030 году должен вырасти до 11,2 трлн рублей; "
                "в) ежегодный объем оказанных услуг по разработке и реализации решений в области искусственного интеллекта "
                "в 2030 году должен вырасти не менее чем до 60 млрд рублей по сравнению с 12 млрд рублей в 2022 году; "
                "к) объем затрат организаций на внедрение и использование технологий искусственного интеллекта должен вырасти до 850 млрд рублей."
            ),
            policy,
        )

        self.assertEqual(unit_count, 1)
        self.assertIn("60 млрд рублей", packed)
        self.assertNotIn("850 млрд рублей", packed)

    def test_list_coverage_detection_counts_semicolon_items(self) -> None:
        contexts = [
            RetrievedContext(
                chunk_id=49,
                text=(
                    "стимулирование спроса отраслевых организаций; обязательные требования при субсидиях; "
                    "включение показателей в национальные проекты; внедрение в государственных корпорациях; "
                    "консультирование организаций"
                ),
                rerank_score=1.0,
            )
        ]

        self.assertEqual(StrategyRetriever._count_list_units(contexts), 5)
        self.assertTrue(StrategyRetriever._has_sufficient_list_coverage(contexts, 5))
        self.assertFalse(StrategyRetriever._has_sufficient_list_coverage(contexts, 6))

    def test_list_coverage_detection_counts_two_item_semicolon_lists(self) -> None:
        contexts = [
            RetrievedContext(
                chunk_id=6,
                text="первый показатель; второй показатель",
                rerank_score=1.0,
            )
        ]

        self.assertEqual(StrategyRetriever._count_list_units(contexts), 2)


if __name__ == "__main__":
    unittest.main()
