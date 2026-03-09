"""Generic answer verifier/self-check for risky RAG answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .classifier import QueryPlan
from .client import OllamaClient
from .config import AppConfig
from .retriever import RetrievedContext

log = logging.getLogger(__name__)

WHY_QUESTION_RE = re.compile(r"^\s*(почему|why)\b", flags=re.IGNORECASE)
STRUCTURAL_QUESTION_RE = re.compile(r"\b(пункт|подпункт|абзац|paragraph|section)\b", flags=re.IGNORECASE)
LEADING_ENUMERATOR_RE = re.compile(r"^\s*(?:\d+\.|[а-яёa-z]\))\s+", flags=re.IGNORECASE)
COLON_ENUMERATOR_RE = re.compile(r"(:\s*)(?:\d+\.|[а-яёa-z]\))\s+", flags=re.IGNORECASE)

VERIFIER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "issues"],
}

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict answer editor for a document-grounded RAG system. "
    "Revise the draft answer only using the provided context. "
    "Do not add unsupported facts. "
    "Goals: improve completeness, remove verbosity, preserve safety, and keep the answer directly responsive to the question. "
    "If the draft is already good, return a minimally cleaned version. "
    "Respond with JSON."
)


@dataclass
class VerifierResult:
    answer: str
    confidence: float
    issues: list[str] = field(default_factory=list)

    @staticmethod
    def empty() -> "VerifierResult":
        return VerifierResult(answer="", confidence=0.0, issues=[])


class AnswerVerifier:
    """Second-pass generic verifier for long or risky answers."""

    def __init__(self, client: OllamaClient, config: AppConfig, model_id: str) -> None:
        self.client = client
        self.config = config
        self.model_id = model_id

    def maybe_revise(
        self,
        question: str,
        draft_answer: str,
        contexts: list[RetrievedContext],
        plan: QueryPlan,
    ) -> str:
        normalized = self._normalize_answer(question, draft_answer, plan)
        if not normalized or not self._should_verify(question, normalized, plan):
            return normalized

        from .generator import AnswerGenerator  # local import to avoid circular import

        context_block = AnswerGenerator.build_context_block(question, contexts, plan)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Draft answer: {normalized}\n\n"
            f"Context:\n{context_block}\n\n"
            "Rules:\n"
            "- Keep the answer grounded only in the context.\n"
            "- If this is a definition question, keep the full supported definition.\n"
            "- If this is a list question, keep the full supported list but prefer concise semicolon-separated phrases.\n"
            "- If the question contains a false premise, correct it briefly.\n"
            "- If the answer is too verbose, compress it without losing supported facts.\n"
            "- If the question is adversarial, preserve the refusal and only grounded facts.\n"
            "- Remove formatting noise like leading enumerator markers when they are not semantically required.\n"
            "Return JSON with 'answer', 'confidence', and 'issues'."
        )

        try:
            raw = self.client.generate(
                model_id=self.model_id,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=self.config.verifier_max_tokens,
                seed=self.config.seed,
                json_schema=VERIFIER_JSON_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("AnswerVerifier LLM call failed: %s", exc)
            return normalized

        result = self._parse(raw)
        if result.confidence < 0.45 or not result.answer.strip():
            return normalized

        revised = self._normalize_answer(question, result.answer, plan)
        if not revised:
            return normalized

        # Guard against pathological expansion from the verifier.
        if (
            len(revised) > max(len(normalized) * 1.35, len(normalized) + 120)
            and not plan.expects_exhaustive_list
        ):
            return normalized
        return revised

    def _should_verify(self, question: str, answer: str, plan: QueryPlan) -> bool:
        if not self.config.enable_answer_verifier:
            return False
        if plan.is_prompt_injection:
            return True
        if plan.query_type == "definitional":
            return True
        if WHY_QUESTION_RE.match(question):
            return True
        if plan.expects_exhaustive_list:
            return len(answer) >= max(self.config.verifier_length_threshold * 2, 900)
        return len(answer) >= self.config.verifier_length_threshold

    @staticmethod
    def _normalize_answer(question: str, answer: str, plan: QueryPlan) -> str:
        normalized = re.sub(r"\s+", " ", answer).strip()
        if not normalized:
            return normalized
        if plan.query_type == "structural" or plan.paragraph_refs or STRUCTURAL_QUESTION_RE.search(question):
            return normalized
        normalized = LEADING_ENUMERATOR_RE.sub("", normalized)
        normalized = COLON_ENUMERATOR_RE.sub(r"\1", normalized)
        return normalized.strip()

    @staticmethod
    def _parse(raw: str) -> VerifierResult:
        text = raw.strip()
        if not text:
            return VerifierResult.empty()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return VerifierResult.empty()
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return VerifierResult.empty()

        answer = str(data.get("answer", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        issues_raw = data.get("issues", [])
        issues = [str(item).strip() for item in issues_raw if str(item).strip()] if isinstance(issues_raw, list) else []
        return VerifierResult(answer=answer, confidence=confidence, issues=issues)
