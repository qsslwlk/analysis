"""Small LLM client abstraction for optional V2.5 claim extraction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Return a JSON object produced by the model."""


@dataclass
class OpenAIChatClient:
    """Minimal OpenAI-compatible chat completions client using requests."""

    model: str = "gpt-4.1-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", self.base_url).rstrip("/")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM claim extraction.")

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LLM API error {response.status_code}: {response.text[:500]}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


def make_llm_client(provider: str, model: str, api_key: Optional[str] = None) -> Optional[LLMClient]:
    normalized = provider.lower().strip()
    if normalized in {"none", "disabled", "off"}:
        return None
    if normalized in {"openai", "openai_chat", "openai-compatible"}:
        return OpenAIChatClient(model=model, api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")

