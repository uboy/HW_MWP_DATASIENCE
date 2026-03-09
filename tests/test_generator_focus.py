import unittest

from rag.classifier import QueryPlan
from rag.generator import AnswerGenerator
from rag.retriever import RetrievedContext


class GeneratorFocusTest(unittest.TestCase):
    def test_factoid_context_focus_prefers_requested_metric(self) -> None:
        contexts = [
            RetrievedContext(
                chunk_id=30,
                text=(
                    "б) совокупный прирост валового внутреннего продукта в 2030 году "
                    "должен вырасти до 11,2 трлн рублей; "
                    "в) ежегодный объем оказанных услуг по разработке и реализации решений "
                    "в области искусственного интеллекта в 2030 году должен вырасти не менее "
                    "чем до 60 млрд рублей по сравнению с 12 млрд рублей в 2022 году; "
                    "к) объем затрат организаций на внедрение и использование технологий "
                    "искусственного интеллекта должен вырасти до 850 млрд рублей."
                ),
                rerank_score=1.0,
            )
        ]
        plan = QueryPlan(
            query_type="factoid",
            top_k=6,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["ежегодный", "объем", "услуг"],
        )

        block = AnswerGenerator.build_context_block(
            "Какой целевой ежегодный объем услуг по ИИ установлен на 2030 год?",
            contexts,
            plan,
        )

        self.assertIn("60 млрд рублей", block)
        self.assertNotIn("850 млрд рублей", block)

    def test_exhaustive_list_context_keeps_full_paragraph_text(self) -> None:
        contexts = [
            RetrievedContext(
                chunk_id=49,
                text=(
                    "а) стимулирование спроса отраслевых организаций; "
                    "б) обязательные требования при субсидиях; "
                    "в) включение показателей в национальные проекты; "
                    "г) внедрение в государственных корпорациях; "
                    "д) консультирование организаций."
                ),
                rerank_score=1.0,
            )
        ]
        plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            key_tokens=["направления", "стимулирования", "внедрения"],
            expects_exhaustive_list=True,
        )

        block = AnswerGenerator.build_context_block(
            "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            contexts,
            plan,
        )

        self.assertIn("стимулирование спроса", block)
        self.assertIn("обязательные требования", block)
        self.assertIn("включение показателей", block)
        self.assertIn("внедрение в государственных корпорациях", block)
        self.assertIn("консультирование организаций", block)

    def test_extract_list_items_supports_semicolon_only_packed_context(self) -> None:
        items = AnswerGenerator._extract_list_items(
            "стимулирование спроса отраслевых организаций; обязательные требования при субсидиях; "
            "включение показателей в национальные проекты; внедрение в государственных корпорациях"
        )

        self.assertGreaterEqual(len(items), 4)
        self.assertIn("стимулирование спроса отраслевых организаций", items)
        self.assertIn("внедрение в государственных корпорациях", items)

    def test_parse_structured_salvages_partial_answer_field(self) -> None:
        result = AnswerGenerator._parse_structured(  # type: ignore[misc]
            AnswerGenerator.__new__(AnswerGenerator),
            '{"answer":"В документе не указан минимальный оклад для ИИ-специалистов в госсекторе.",',
        )

        self.assertEqual(
            result.answer,
            "В документе не указан минимальный оклад для ИИ-специалистов в госсекторе.",
        )
        self.assertGreaterEqual(result.confidence, 0.6)

    def test_parse_structured_rejects_plain_text_without_answer_field(self) -> None:
        result = AnswerGenerator._parse_structured(  # type: ignore[misc]
            AnswerGenerator.__new__(AnswerGenerator),
            "В документе не указан минимальный оклад для ИИ-специалистов в госсекторе.",
        )

        self.assertEqual(result.answer, "")
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
