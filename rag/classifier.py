"""LLM-based query classifier: replaces hardcoded FACTOID_PREFIXES / FACTOID_KEYWORDS."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from .client import OllamaClient
from .config import AppConfig

log = logging.getLogger(__name__)

TOP_K_BY_TYPE: dict[str, int] = {
    "factoid": 6,
    "definitional": 8,
    "structural": 6,
    "analytical": 10,
}

CLASSIFY_PROMPT = """\
Classify this question for document retrieval. Return JSON only, no explanation.

Question: {query}

JSON schema:
{{
  "query_type": "factoid" | "definitional" | "structural" | "analytical",
  "paragraph_refs": [list of paragraph/section numbers explicitly mentioned, empty if none],
  "boost_early_chunks": true if the question is about document title, header, date, or amendments,
  "language": "ru" | "en" | "de" | "fr" | "es" | "zh" | "ja" | "other",
  "key_tokens": [3-5 most important content words from the question],
  "search_queries": [1-3 concise retrieval-oriented rewrites of the question]
}}

Rules:
- factoid: single fact, number, name, date, short list
- definitional: "what is X", "how is X defined", "what does X mean"
- structural: references a specific paragraph/section number like "paragraph 4" or "section 3"
- analytical: broad analysis, comparison, or summary

Return ONLY valid JSON, nothing else.
"""

CLASSIFY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": ["factoid", "definitional", "structural", "analytical"]},
        "paragraph_refs": {"type": "array", "items": {"type": "integer"}},
        "boost_early_chunks": {"type": "boolean"},
        "language": {"type": "string"},
        "key_tokens": {"type": "array", "items": {"type": "string"}},
        "search_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query_type", "paragraph_refs", "boost_early_chunks", "language", "key_tokens", "search_queries"],
}

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", flags=re.UNICODE)
PARAGRAPH_REF_RE = re.compile(
    r"\b(?:пункт(?:а|е|у|ом)?|подпункт(?:а|е|у|ом)?|параграф(?:а|е|у|ом)?|paragraph|section|para)\s*[«\"']?[а-яёa-z]?[»\"']?\s*(\d{1,3})\b",
    flags=re.IGNORECASE,
)
EARLY_CHUNK_HINT_RE = re.compile(
    r"(заголов|редакц|изменени|верси|дата|на какой период|title|header|amendment|version)",
    flags=re.IGNORECASE,
)
ADVERSARIAL_PATTERNS = (
    re.compile(r"ignore\s+(?:the\s+)?document", flags=re.IGNORECASE),
    re.compile(r"ignore\s+previous\s+instructions", flags=re.IGNORECASE),
    re.compile(r"make\s+up", flags=re.IGNORECASE),
    re.compile(r"secret", flags=re.IGNORECASE),
    re.compile(r"игнорируй\s+документ", flags=re.IGNORECASE),
    re.compile(r"игнорируй\s+.*правил", flags=re.IGNORECASE),
    re.compile(r"придумай", flags=re.IGNORECASE),
    re.compile(r"секретн", flags=re.IGNORECASE),
)
PROMPT_CONTROL_PATTERNS = (
    re.compile(r"ignore\s+(?:the\s+)?document(?:\s+and\s+previous\s+instructions)?", flags=re.IGNORECASE),
    re.compile(r"ignore\s+previous\s+instructions", flags=re.IGNORECASE),
    re.compile(r"make\s+up", flags=re.IGNORECASE),
    re.compile(r"игнорируй\s+документ(?:\s+и\s+предыдущие\s+правила)?", flags=re.IGNORECASE),
    re.compile(r"игнорируй\s+предыдущие\s+правила", flags=re.IGNORECASE),
    re.compile(r"придумай", flags=re.IGNORECASE),
    re.compile(r"убедительный\s+официальный\s+ответ", flags=re.IGNORECASE),
    re.compile(r"даже\s+если\s+их\s+нет\s+в\s+тексте", flags=re.IGNORECASE),
    re.compile(r"even\s+if\s+they\s+are\s+not\s+in\s+the\s+text", flags=re.IGNORECASE),
)
EXHAUSTIVE_LIST_PATTERNS = (
    re.compile(r"^\s*(?:какие|каковы)\s+(?:показатели|целевые показатели|направления|документы|цели|задачи|принципы|механизмы|технологии)\b", flags=re.IGNORECASE),
    re.compile(r"перечисл", flags=re.IGNORECASE),
    re.compile(r"что включает", flags=re.IGNORECASE),
    re.compile(r"what are", flags=re.IGNORECASE),
    re.compile(r"which .*indicators", flags=re.IGNORECASE),
)
DEFINITIONAL_PATTERNS = (
    re.compile(r"^\s*что такое\b", flags=re.IGNORECASE),
    re.compile(r"^\s*что .*понимается\b", flags=re.IGNORECASE),
    re.compile(r"^\s*как .*определяется\b", flags=re.IGNORECASE),
    re.compile(r"^\s*как .*определяется\b", flags=re.IGNORECASE),
    re.compile(r"^\s*what is\b", flags=re.IGNORECASE),
    re.compile(r"^\s*how is .*defined\b", flags=re.IGNORECASE),
)
STOPWORDS = {
    "и", "в", "во", "на", "по", "что", "как", "какой", "какие", "какая", "какую",
    "кто", "где", "для", "это", "эта", "этот", "эти", "или", "ли", "из", "под",
    "при", "над", "под", "его", "ее", "их", "the", "what", "which", "how", "when",
    "where", "why", "does", "about", "from", "into", "with", "have", "will",
}
LEADING_QUESTION_WORD_RE = re.compile(
    r"^\s*(?:какие|какой|какая|какое|каковы|что|кто|как|почему|зачем|что такое|"
    r"what|which|how|why|who|when|where)\b[\s,:-]*",
    flags=re.IGNORECASE,
)
TRAILING_META_RE = re.compile(
    r"\b(?:указан(?:о|а|ы)?|установлен(?:о|а|ы)?|выделены|понимается|определяется|"
    r"is\s+defined|is\s+meant|is\s+specified|is\s+set)\b.*$",
    flags=re.IGNORECASE,
)


def extract_explicit_paragraph_refs(query: str) -> list[int]:
    refs: list[int] = []
    for match in PARAGRAPH_REF_RE.finditer(query):
        ref = int(match.group(1))
        if ref not in refs:
            refs.append(ref)
    return refs


def derive_key_tokens(query: str, limit: int = 5) -> list[str]:
    tokens: list[str] = []
    normalized_query = sanitize_query_for_retrieval(query)
    for token in WORD_RE.findall(normalized_query.lower()):
        if token in STOPWORDS or len(token) < 3 or token.isdigit():
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def derive_search_queries(
    query: str,
    key_tokens: list[str],
    paragraph_refs: list[int] | None = None,
    limit: int = 3,
) -> list[str]:
    paragraph_refs = paragraph_refs or []
    normalized_query = sanitize_query_for_retrieval(query)
    stripped_query = LEADING_QUESTION_WORD_RE.sub("", normalized_query)
    stripped_query = TRAILING_META_RE.sub("", stripped_query).strip(" ?!.,:;")
    token_query = " ".join(key_tokens).strip()
    ref_prefix = ""
    if paragraph_refs:
        refs = " ".join(str(ref) for ref in paragraph_refs)
        ref_prefix = f"пункт {refs}".strip()

    candidates = [
        normalized_query,
        stripped_query,
        f"{ref_prefix} {token_query}".strip(),
        token_query,
    ]

    search_queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip(" ?!.,:;")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        search_queries.append(cleaned)
        if len(search_queries) >= limit:
            break
    return search_queries


def looks_like_prompt_injection(query: str) -> bool:
    return any(pattern.search(query) for pattern in ADVERSARIAL_PATTERNS)


def sanitize_query_for_retrieval(query: str) -> str:
    sanitized = query
    for pattern in PROMPT_CONTROL_PATTERNS:
        sanitized = pattern.sub(" ", sanitized)
    sanitized = re.sub(r"[\"'«»“”`]", " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" .,:;")
    return sanitized or query


def expects_exhaustive_list(query: str) -> bool:
    return any(pattern.search(query) for pattern in EXHAUSTIVE_LIST_PATTERNS)


def looks_like_definitional(query: str) -> bool:
    return any(pattern.search(query) for pattern in DEFINITIONAL_PATTERNS)


def _needs_early_chunk_boost(query: str) -> bool:
    return bool(EARLY_CHUNK_HINT_RE.search(query))


@dataclass
class QueryPlan:
    query_type: Literal["factoid", "definitional", "structural", "analytical"]
    top_k: int
    boost_early_chunks: bool
    paragraph_refs: list[int]
    language: str
    key_tokens: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    expects_exhaustive_list: bool = False
    is_prompt_injection: bool = False

    @staticmethod
    def default(language: str = "ru") -> "QueryPlan":
        return QueryPlan(
            query_type="analytical",
            top_k=TOP_K_BY_TYPE["analytical"],
            boost_early_chunks=False,
            paragraph_refs=[],
            language=language,
            key_tokens=[],
            search_queries=[],
            expects_exhaustive_list=False,
            is_prompt_injection=False,
        )


class QueryClassifier:
    """Classifies user questions via a single LLM call into a QueryPlan."""

    def __init__(self, client: OllamaClient, config: AppConfig) -> None:
        self.client = client
        self.config = config
        # Use a dedicated classifier model if configured, else fall back to main model
        self._model_id: str | None = None

    def _get_model_id(self) -> str:
        if self._model_id:
            return self._model_id
        preferred = self.config.classifier_model or self.config.model_id
        self._model_id = self.client.resolve_model(preferred)
        return self._model_id

    def classify(self, query: str) -> QueryPlan:
        fallback_lang = self._resolved_language()
        try:
            raw = self.client.generate(
                model_id=self._get_model_id(),
                system_prompt="You are a query classifier. Return valid JSON only.",
                user_prompt=CLASSIFY_PROMPT.format(query=query),
                temperature=0.0,
                max_tokens=200,
                seed=self.config.seed,
                json_schema=CLASSIFY_JSON_SCHEMA,
            )
            plan = self._parse(raw, query=query, fallback_lang=fallback_lang)
            # If DOCUMENT_LANGUAGE is explicitly set (not "auto"), honour the config
            if self.config.document_language != "auto":
                plan.language = self.config.document_language
            return plan
        except Exception as exc:  # noqa: BLE001
            log.warning("QueryClassifier failed (%s), using default plan", exc)
            return self._fallback_plan(query=query, fallback_lang=fallback_lang)

    def _resolved_language(self) -> str:
        """Returns the language to use when auto-detection is unavailable."""
        lang = self.config.document_language
        return lang if lang != "auto" else "ru"

    def _fallback_plan(self, query: str, fallback_lang: str) -> QueryPlan:
        plan = QueryPlan.default(language=fallback_lang)
        plan.key_tokens = derive_key_tokens(query)
        plan.boost_early_chunks = _needs_early_chunk_boost(query)
        plan.expects_exhaustive_list = expects_exhaustive_list(query)
        plan.is_prompt_injection = looks_like_prompt_injection(query)
        if looks_like_definitional(query):
            plan.query_type = "definitional"
        paragraph_refs = extract_explicit_paragraph_refs(query)
        if paragraph_refs:
            plan.query_type = "structural"
            plan.paragraph_refs = paragraph_refs
        plan.search_queries = derive_search_queries(
            query=query,
            key_tokens=plan.key_tokens,
            paragraph_refs=plan.paragraph_refs,
        )
        if plan.expects_exhaustive_list or plan.is_prompt_injection:
            plan.top_k = max(plan.top_k, TOP_K_BY_TYPE["analytical"])
        else:
            plan.top_k = TOP_K_BY_TYPE[plan.query_type]
        return plan

    def _parse(self, raw: str, query: str, fallback_lang: str) -> QueryPlan:
        # Level 1: direct parse
        data = self._try_parse_json(raw)

        if data is None:
            log.warning("QueryClassifier: invalid JSON response, using default plan")
            return self._fallback_plan(query=query, fallback_lang=fallback_lang)

        query_type = data.get("query_type", "analytical")
        if query_type not in TOP_K_BY_TYPE:
            query_type = "analytical"

        paragraph_refs_raw = data.get("paragraph_refs", [])
        paragraph_refs = [int(x) for x in paragraph_refs_raw if str(x).isdigit() or isinstance(x, int)]
        for ref in extract_explicit_paragraph_refs(query):
            if ref not in paragraph_refs:
                paragraph_refs.append(ref)
        if paragraph_refs:
            query_type = "structural"
        elif looks_like_definitional(query):
            query_type = "definitional"

        language = str(data.get("language", "ru"))
        if not language or language == "other":
            language = fallback_lang

        key_tokens: list[str] = []
        for token in data.get("key_tokens", []):
            cleaned = str(token).strip().lower()
            if not cleaned or cleaned in STOPWORDS:
                continue
            if cleaned not in key_tokens:
                key_tokens.append(cleaned)
        for token in derive_key_tokens(query):
            if token not in key_tokens:
                key_tokens.append(token)
        search_queries: list[str] = []
        for candidate in data.get("search_queries", []):
            cleaned = re.sub(r"\s+", " ", str(candidate)).strip(" ?!.,:;")
            if not cleaned:
                continue
            if cleaned.lower() not in {item.lower() for item in search_queries}:
                search_queries.append(cleaned)
        boost_early_chunks = bool(data.get("boost_early_chunks", False)) or _needs_early_chunk_boost(query)
        exhaustive_list = expects_exhaustive_list(query)
        prompt_injection = looks_like_prompt_injection(query)
        for candidate in derive_search_queries(query, key_tokens, paragraph_refs):
            if candidate.lower() not in {item.lower() for item in search_queries}:
                search_queries.append(candidate)

        top_k = TOP_K_BY_TYPE[query_type]
        if exhaustive_list or prompt_injection:
            top_k = max(top_k, TOP_K_BY_TYPE["analytical"])

        return QueryPlan(
            query_type=query_type,  # type: ignore[arg-type]
            top_k=top_k,
            boost_early_chunks=boost_early_chunks,
            paragraph_refs=paragraph_refs,
            language=language,
            key_tokens=key_tokens[:5],
            search_queries=search_queries[:3],
            expects_exhaustive_list=exhaustive_list,
            is_prompt_injection=prompt_injection,
        )

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        # Level 1: direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Level 2: extract first {...} block
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None
