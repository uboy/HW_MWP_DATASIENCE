"""Generic high-precision guardrails for grounded answers."""

from __future__ import annotations

import re

from .classifier import QueryPlan, derive_key_tokens, sanitize_query_for_retrieval
from .retriever import RetrievedContext

SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!;])\s+")
ENUMERATOR_FRAGMENT_RE = re.compile(r"^(?:\d+[.)]?|[а-яёa-z]\))$", flags=re.IGNORECASE)
LEGAL_ACT_RE = re.compile(
    r'от\s+\d{1,2}\s+[а-яё]+\s+\d{4}\s+г\.\s+№\s+\d+\s*[- ]?\s*ФЗ\s+"[^"]+"',
    flags=re.IGNORECASE,
)
ROLE_QUESTION_RE = re.compile(r"^\s*(?:какую\s+роль|какова\s+роль|what\s+role)\b", flags=re.IGNORECASE)
RELATION_QUESTION_RE = re.compile(r"^\s*как\b.+\bсвязан(?:о|а|ы)?\b", flags=re.IGNORECASE)
SECRET_REQUEST_RE = re.compile(r"(?:секрет|hidden|secret)", flags=re.IGNORECASE)
VALUE_REQUEST_RE = re.compile(
    r"(?:минимальн|максимальн|оклад|размер|стоимост|цена|в\s+рублях|в\s+процентах|"
    r"сколько|какой\s+размер|какая\s+сумма|what\s+amount|what\s+minimum)",
    flags=re.IGNORECASE,
)
UNSUPPORTED_DETAIL_RE = re.compile(r"\([^)]*\)")
TRAILING_REQUEST_RE = re.compile(
    r"\b(?:установлен(?:о|а|ы)?|предусмотрен(?:о|а|ы)?|указан(?:о|а|ы)?|"
    r"specified|set|defined)\b.*$",
    flags=re.IGNORECASE,
)
QUESTION_TERM_PATTERNS = (
    re.compile(r"^\s*что\s+такое\s+(?P<term>.+?)(?:\?|$| и )", flags=re.IGNORECASE),
    re.compile(r"^\s*что.+?понимается\s+под\s+(?P<term>.+?)(?:\?|$)", flags=re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+(?P<term>.+?)(?:\?|$| and )", flags=re.IGNORECASE),
    re.compile(r"^\s*how\s+is\s+(?P<term>.+?)\s+defined(?:\?|$)", flags=re.IGNORECASE),
)
SPHERE_RE = re.compile(r"в\s+сфере\s+(?P<object>.+?)(?:\?|$)", flags=re.IGNORECASE)


def try_answer_with_rules(
    question: str,
    contexts: list[RetrievedContext],
    plan: QueryPlan,
) -> str | None:
    if not contexts:
        return None

    combined_text = _combine_contexts(contexts)

    if plan.is_prompt_injection:
        answer = _answer_prompt_injection(question, contexts, plan)
        if answer:
            return answer

    answer = _answer_definition(question, combined_text)
    if answer:
        return answer

    answer = _answer_legal_acts(question, combined_text)
    if answer:
        return answer

    answer = _answer_missing_specific_value(question, contexts, plan)
    if answer:
        return answer

    answer = _answer_role_question(question, contexts, plan)
    if answer:
        return answer

    answer = _answer_relation_question(question, contexts, plan)
    if answer:
        return answer

    return None


def _combine_contexts(contexts: list[RetrievedContext]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for context in contexts:
        normalized = _normalize(context.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parts.append(normalized)
    return " ".join(parts)


def _normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _clean_rule_sentence(text: str) -> str:
    normalized = _normalize(text)
    normalized = re.sub(r"^(?:\d+[.)]?\s*|[а-яёa-z]\)\s*)", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip(" ;")
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
        if normalized[-1] not in ".!?":
            normalized += "."
    return normalized


def _question_focus_tokens(question: str, plan: QueryPlan) -> list[str]:
    focus = list(plan.key_tokens) or derive_key_tokens(question)
    return [token.lower() for token in focus if token]


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    merged: list[str] = []
    pending_prefix = ""
    for part in parts:
        if ENUMERATOR_FRAGMENT_RE.fullmatch(part):
            pending_prefix = f"{pending_prefix} {part}".strip()
            continue
        if pending_prefix:
            part = f"{pending_prefix} {part}".strip()
            pending_prefix = ""
        merged.append(part)
    if pending_prefix:
        merged.append(pending_prefix)
    return merged


def _extract_term_from_question(question: str) -> str | None:
    for pattern in QUESTION_TERM_PATTERNS:
        match = pattern.search(question)
        if match:
            term = _normalize(match.group("term")).strip(" .,:;")
            if term:
                return term
    return None


def _answer_definition(question: str, text: str) -> str | None:
    term = _extract_term_from_question(question)
    if not term:
        return None
    term_stems = {_stem_token(token) for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", term.lower())}
    candidates = re.finditer(
        r"(?P<label>(?:[а-яёa-z]\)\s*)?[^.;:]{3,120}?)\s*-\s*(?P<body>.+?)(?=(?:\.\s+[А-ЯA-ZЁ]|;\s+[А-ЯA-ZЁ]|$))",
        text,
        flags=re.IGNORECASE,
    )

    best_label = ""
    best_body = ""
    best_score = 0
    for match in candidates:
        label = _normalize(match.group("label")).strip(" .,:;")
        body = _normalize(match.group("body")).rstrip(".;")
        label_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", label.lower())
        label_stems = {_stem_token(token) for token in label_tokens}
        overlap = len(term_stems & label_stems)
        if overlap > best_score and body:
            best_label = label
            best_body = body
            best_score = overlap

    if best_score == 0 or not best_body:
        return None
    if "порог" in question.lower() and "парамет" not in best_body.lower():
        return None
    return f"{best_label} - {best_body}."


def _answer_legal_acts(question: str, text: str) -> str | None:
    lowered_question = question.lower()
    if "закон" not in lowered_question and "law" not in lowered_question:
        return None

    matches = LEGAL_ACT_RE.findall(text)
    acts: list[str] = []
    seen: set[str] = set()
    for match in matches:
        normalized = _normalize(match)
        normalized = re.sub(r"№\s+(\d+)\s+ФЗ", r"№ \1-ФЗ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"№\s+(\d+)-?\s*ФЗ", r"№ \1-ФЗ", normalized, flags=re.IGNORECASE)
        act = f"Федеральный закон {normalized}"
        if act in seen:
            continue
        seen.add(act)
        acts.append(act)
    if not acts:
        return None
    return "; ".join(acts) + "."


def _answer_prompt_injection(
    question: str,
    contexts: list[RetrievedContext],
    plan: QueryPlan,
) -> str | None:
    focus_tokens = _question_focus_tokens(sanitize_query_for_retrieval(question), plan)
    focus_tokens = [
        token for token in focus_tokens
        if token not in {"ответ", "официальный", "убедительный", "точными", "цифрами", "стратегии", "документ"}
    ]
    lowered_question = question.lower()
    if any(marker in lowered_question for marker in ("финанс", "бюджет", "инвест")):
        for extra in ("финансов", "обеспеч", "бюджет", "инвест"):
            if extra not in focus_tokens:
                focus_tokens.append(extra)
    support = _top_support_sentences(contexts, focus_tokens, sentence_limit=1)
    refusal = [
        "Я не могу выполнять инструкции, которые требуют игнорировать документ или придумывать факты.",
    ]
    if SECRET_REQUEST_RE.search(question):
        refusal.append("В документе не приведены скрытые или секретные пункты.")
    if support:
        refusal.append("По документу: " + " ".join(support))
    return " ".join(refusal)


def _answer_missing_specific_value(
    question: str,
    contexts: list[RetrievedContext],
    plan: QueryPlan,
) -> str | None:
    if plan.query_type not in {"factoid", "structural"}:
        return None
    if not VALUE_REQUEST_RE.search(question):
        return None

    focus_tokens = _question_focus_tokens(question, plan)
    sentences = [sentence for context in contexts for sentence in _split_sentences(context.text)]
    if not sentences:
        return None

    if _has_supported_value(sentences, focus_tokens):
        return None

    subject = _extract_requested_subject(question)
    if not subject:
        return "В документе запрашиваемое значение не указано."

    return f"В документе {subject} не установлен."


def _has_supported_value(sentences: list[str], focus_tokens: list[str]) -> bool:
    if not focus_tokens:
        return False
    for sentence in sentences:
        lowered = sentence.lower()
        overlap = sum(1 for token in focus_tokens if token in lowered)
        has_numeric_signal = bool(re.search(r"\d", sentence) or re.search(r"руб|процент|%", lowered))
        if overlap >= 2 and has_numeric_signal:
            return True
    return False


def _extract_requested_subject(question: str) -> str:
    normalized = _normalize(question).rstrip("?")
    normalized = LEADING_QUESTION_WORD_RE.sub("", normalized)
    normalized = TRAILING_REQUEST_RE.sub("", normalized)
    normalized = UNSUPPORTED_DETAIL_RE.sub("", normalized)
    normalized = _normalize(normalized).strip(" .,:;")
    return normalized


LEADING_QUESTION_WORD_RE = re.compile(
    r"^\s*(?:какие|какой|какая|какое|каковы|что|кто|как|почему|зачем|what|which|how|why)\b[\s,:-]*",
    flags=re.IGNORECASE,
)


def _answer_role_question(
    question: str,
    contexts: list[RetrievedContext],
    plan: QueryPlan,
) -> str | None:
    if not ROLE_QUESTION_RE.search(question):
        return None
    focus_tokens = _question_focus_tokens(question, plan)
    support = _top_support_sentences(contexts, focus_tokens, sentence_limit=2)
    if not support:
        return None
    sphere_match = SPHERE_RE.search(question)
    if sphere_match:
        sphere_object = _normalize(sphere_match.group("object")).strip(" .,:;")
        for sentence in support:
            lowered = sentence.lower()
            if f"в сфере {sphere_object.lower()}" in lowered and "для " in lowered:
                tail = sentence[lowered.index(f"в сфере {sphere_object.lower()}") + len(f"в сфере {sphere_object.lower()}"):].strip(" .,:;")
                if tail.startswith("для "):
                    return f"Ее роль состоит в обеспечении {sphere_object} {tail}."
    return " ".join(_clean_rule_sentence(sentence) for sentence in support)


def _answer_relation_question(
    question: str,
    contexts: list[RetrievedContext],
    plan: QueryPlan,
) -> str | None:
    if not RELATION_QUESTION_RE.search(question):
        return None
    focus_tokens = _question_focus_tokens(question, plan)
    support = _top_support_sentences(contexts, focus_tokens, sentence_limit=1)
    if not support:
        return None
    return " ".join(_clean_rule_sentence(sentence) for sentence in support)


def _top_support_sentences(
    contexts: list[RetrievedContext],
    focus_tokens: list[str],
    sentence_limit: int,
) -> list[str]:
    scored: list[tuple[float, int, int, str]] = []
    for context_rank, context in enumerate(contexts):
        for sentence_index, sentence in enumerate(_split_sentences(context.text)):
            lowered = sentence.lower()
            overlap = sum(1 for token in focus_tokens if token in lowered)
            numeric_signal = 1.0 if re.search(r"\d", sentence) else 0.0
            score = overlap * 3.0 + numeric_signal + max(context.rerank_score, 0.0)
            if score <= 0:
                continue
            scored.append((score, context_rank, sentence_index, _normalize(sentence)))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, _, sentence in scored:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= sentence_limit:
            break
    return selected


def _stem_token(token: str) -> str:
    if len(token) <= 5:
        return token
    for suffix in ("иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "ого", "ему", "ому", "ом", "ем", "ый", "ий", "ой", "ая", "ое", "ые", "ие", "ых", "их", "ую", "юю", "ах", "ях", "ам", "ям", "ов", "ев", "а", "я", "ы", "и", "е", "у", "ю"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token
