import json
import unittest

from rag.classifier import QueryPlan
from rag.config import AppConfig
from rag.evaluator import HomeworkMetricEvaluator
from rag.retriever import RetrievedContext
from rag.verifier import AnswerVerifier


def make_config() -> AppConfig:
    return AppConfig(
        base_url="http://localhost:11434",
        api_token=None,
        model_id="dummy-model",
        classifier_model=None,
        embedding_model="dummy-embedding",
        reranker_model="dummy-reranker",
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
        timeout_sec=30,
        use_env_proxy=False,
        disable_thinking=True,
        head_chunk_boost_count=3,
        min_rerank_score=0.02,
        min_confidence=0.3,
        document_language="ru",
        enable_answer_verifier=True,
        verifier_max_tokens=240,
        verifier_length_threshold=200,
    )


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        json_schema: dict | None = None,
    ) -> str:
        self.calls.append(
            {
                "model_id": model_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.response


class EvaluatorAndVerifierTest(unittest.TestCase):
    def test_weighted_formulas_are_exposed(self) -> None:
        general = HomeworkMetricEvaluator.weighted_general(
            answer_relevancy=0.9,
            answer_correctness=0.8,
            answer_similarity=0.7,
            clarity=0.6,
            safety=1.0,
        )
        correctness_heavy = HomeworkMetricEvaluator.weighted_correctness_heavy(
            answer_correctness=0.8,
            answer_similarity=0.7,
            clarity=0.6,
            safety=1.0,
        )

        self.assertEqual(general, 0.815)
        self.assertEqual(correctness_heavy, 0.76)

    def test_clarity_heuristic_penalizes_bloat(self) -> None:
        concise = HomeworkMetricEvaluator._clarity_score(
            "К 2030 году показатель должен достичь 80 процентов.",
            "К 2030 году показатель должен достичь 80 процентов.",
        )
        verbose = HomeworkMetricEvaluator._clarity_score(
            (
                "К 2030 году показатель должен достичь 80 процентов. "
                "Это значение указывается в документе как целевое. "
                "Также важно отметить, что документ сравнивает его с предыдущим годом и "
                "вообще в целом подробно описывает стратегические цели."
            ),
            "К 2030 году показатель должен достичь 80 процентов.",
        )

        self.assertGreater(concise, verbose)

    def test_verifier_normalizes_leading_enumerator_noise(self) -> None:
        verifier = AnswerVerifier(client=FakeClient("{}"), config=make_config(), model_id="dummy-model")
        plan = QueryPlan(
            query_type="definitional",
            top_k=8,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
        )

        cleaned = verifier._normalize_answer(
            "Что в Стратегии понимается под искусственным интеллектом?",
            "а) искусственный интеллект - комплекс технологических решений.",
            plan,
        )

        self.assertEqual(cleaned, "искусственный интеллект - комплекс технологических решений.")

    def test_verifier_revises_long_list_answers(self) -> None:
        response = json.dumps(
            {
                "answer": "стимулирование спроса; создание пилотных зон; сертификация решений.",
                "confidence": 0.91,
                "issues": ["too verbose"],
            },
            ensure_ascii=False,
        )
        verifier = AnswerVerifier(client=FakeClient(response), config=make_config(), model_id="dummy-model")
        plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            expects_exhaustive_list=True,
        )
        contexts = [
            RetrievedContext(
                chunk_id=34,
                text=(
                    "стимулирование спроса; создание пилотных зон; создание системы сертификации решений."
                ),
                rerank_score=0.9,
            )
        ]

        revised = verifier.maybe_revise(
            "Какие направления стимулирования внедрения ИИ в отраслях экономики выделены?",
            (
                "Основными направлениями являются: стимулирование спроса, в том числе посредством грантов; "
                "создание пилотных зон для апробации и демонстрации разработок; создание системы сертификации решений."
                + (" Очень длинное пояснение о второстепенных юридических, организационных и процедурных деталях."
                   * 20)
            ),
            contexts,
            plan,
        )

        self.assertEqual(
            revised,
            "стимулирование спроса; создание пилотных зон; сертификация решений.",
        )

    def test_verifier_does_not_force_medium_list_through_second_pass(self) -> None:
        verifier = AnswerVerifier(client=FakeClient("{}"), config=make_config(), model_id="dummy-model")
        plan = QueryPlan(
            query_type="analytical",
            top_k=10,
            boost_early_chunks=False,
            paragraph_refs=[],
            language="ru",
            expects_exhaustive_list=True,
        )

        should_verify = verifier._should_verify(
            "Какие показатели используются для оценки достижения целей Стратегии?",
            "показатель 1; показатель 2; показатель 3; показатель 4; показатель 5; показатель 6.",
            plan,
        )

        self.assertFalse(should_verify)

    def test_evaluator_uses_llm_judge_when_available(self) -> None:
        client = FakeClient(
            json.dumps(
                {
                    "answer_relevancy": 0.9,
                    "answer_correctness": 0.8,
                    "answer_similarity": 0.7,
                    "clarity": 0.6,
                    "safety": 1.0,
                    "notes": "ok",
                }
            )
        )
        evaluator = HomeworkMetricEvaluator(
            client=client,
            config=make_config(),
            model_id="dummy-model",
            use_llm_judge=True,
        )

        row = evaluator.evaluate_one(
            question="Какую долю работников с навыками ИИ планируется достичь к 2030 году?",
            gold_answer="К 2030 году доля работников должна вырасти не менее чем до 80 процентов.",
            predicted_answer="К 2030 году доля работников должна вырасти до 80 процентов.",
            row_id="7",
        )

        self.assertEqual(row.id, "7")
        self.assertGreater(row.answer_relevancy, 0.7)
        self.assertGreater(row.answer_correctness, 0.6)
        self.assertEqual(row.notes, "ok")

    def test_evaluator_aligns_predictions_by_id_when_reordered(self) -> None:
        evaluator = HomeworkMetricEvaluator(
            client=None,
            config=None,
            model_id=None,
            use_llm_judge=False,
        )
        gold_rows = [
            {"id": "1", "question": "Q1", "gold_answer": "alpha"},
            {"id": "2", "question": "Q2", "gold_answer": "beta"},
        ]
        prediction_rows = [
            {"id": "2", "answer": "beta"},
            {"id": "1", "answer": "alpha"},
        ]

        rows = evaluator.evaluate_rows(gold_rows=gold_rows, prediction_rows=prediction_rows)

        self.assertEqual([row.id for row in rows], ["1", "2"])
        self.assertEqual(rows[0].predicted_answer, "alpha")
        self.assertEqual(rows[1].predicted_answer, "beta")


if __name__ == "__main__":
    unittest.main()
