import unittest

from rag.classifier import QueryPlan, derive_search_queries
from rag.generator import AnswerGenerator
from rag.retriever import StrategyRetriever


class GenericRagRefactorTest(unittest.TestCase):
    def test_derive_search_queries_adds_focused_variants(self) -> None:
        queries = derive_search_queries(
            query="Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            key_tokens=["направления", "стимулирования", "внедрения", "экономики"],
            paragraph_refs=[34],
        )

        self.assertTrue(queries)
        self.assertIn("пункт 34 направления стимулирования внедрения экономики", queries)
        self.assertIn("направления стимулирования внедрения ИИ в отраслях экономики", queries)

    def test_candidate_queries_deduplicates_original_and_rewrites(self) -> None:
        plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["направления", "внедрения"],
            search_queries=[
                "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
                "направления стимулирования внедрения",
            ],
        )

        candidates = StrategyRetriever._candidate_queries(
            "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            plan,
        )

        self.assertEqual(
            candidates,
            [
                "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены",
                "направления стимулирования внедрения",
            ],
        )

    def test_postprocess_answer_compacts_long_exhaustive_lists(self) -> None:
        plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            expects_exhaustive_list=True,
        )
        answer = (
            "Основными направлениями являются: стимулирование спроса отраслевых организаций, "
            "в том числе посредством предоставления грантов; внедрение технологий искусственного "
            "интеллекта в государственных корпорациях, государственных компаниях и акционерных "
            "обществах с государственным участием, в том числе путем приоритетного включения "
            "проектов; популяризация технологий искусственного интеллекта и повышение доверия "
            "граждан к ним."
        )

        compacted = AnswerGenerator._postprocess_answer(
            "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            plan,
            answer,
        )

        self.assertNotIn("предоставления грантов", compacted)
        self.assertNotIn("приоритетного включения проектов", compacted)
        self.assertIn("стимулирование спроса отраслевых организаций", compacted)
        self.assertIn("популяризация технологий искусственного интеллекта", compacted)


if __name__ == "__main__":
    unittest.main()
