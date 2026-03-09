"""Ollama HTTP client with retry, dual-endpoint fallback, and structured output support."""

from __future__ import annotations

import json
import time
from typing import Dict, List

import requests

from .config import AppConfig


class OllamaClient:
    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.base_url
        self.token = config.api_token
        self.timeout_sec = config.timeout_sec
        self.model_id = config.model_id
        self.disable_thinking = config.disable_thinking
        self._session = requests.Session()
        self._session.trust_env = config.use_env_proxy

    @property
    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        retries = 3
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=self._url(path),
                    timeout=self.timeout_sec,
                    **kwargs,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

            if resp.status_code in (502, 503, 504) and attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            if resp.status_code >= 400:
                body = (resp.text or "").strip()
                if len(body) > 800:
                    body = body[:800] + "..."
                raise RuntimeError(
                    f"Ollama error {resp.status_code} on {path}. "
                    f"Response body: {body if body else '<empty>'}"
                )
            return resp

        if last_exc is not None:
            raise RuntimeError(f"Ollama request failed: {last_exc}") from last_exc
        raise RuntimeError("Ollama request failed for unknown reason")

    def list_models(self) -> List[str]:
        response = self._request("GET", "/api/tags", headers=self.headers)
        payload = response.json()
        model_ids: List[str] = []
        candidates = payload.get("models", []) if isinstance(payload, dict) else []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            mid = item.get("name") or item.get("model")
            if isinstance(mid, str) and mid.strip():
                model_ids.append(mid.strip())
        return model_ids

    def resolve_model(self, model_id: str | None = None) -> str:
        target = model_id or self.model_id
        if target:
            return target
        models = self.list_models()
        if not models:
            raise RuntimeError(
                "No models returned by Ollama. Set OLLAMA_MODEL explicitly in .env"
            )
        self.model_id = models[0]
        return self.model_id

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
        """Call Ollama /api/chat. Pass json_schema for structured JSON output."""
        options: Dict[str, object] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        payload: Dict[str, object] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }
        if self.disable_thinking:
            payload["think"] = False
        if json_schema is not None:
            payload["format"] = json_schema

        response = self._request("POST", "/api/chat", headers=self.headers, json=payload)
        data = response.json()
        message = data.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                return stripped

        # Fallback: /api/generate for models that don't emit content in /api/chat
        fallback_payload: Dict[str, object] = {
            "model": model_id,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": options,
        }
        if self.disable_thinking:
            fallback_payload["think"] = False
        if json_schema is not None:
            fallback_payload["format"] = json_schema

        fallback_response = self._request(
            "POST", "/api/generate", headers=self.headers, json=fallback_payload
        )
        fallback_data = fallback_response.json()
        fallback_text = fallback_data.get("response")
        if isinstance(fallback_text, str):
            stripped = fallback_text.strip()
            if stripped:
                return stripped

        raise RuntimeError(
            "Unexpected Ollama response. "
            f"/api/chat={json.dumps(data, ensure_ascii=False)}; "
            f"/api/generate={json.dumps(fallback_data, ensure_ascii=False)}"
        )
