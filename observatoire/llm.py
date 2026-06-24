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


def _parse_json_content(content: str) -> Dict[str, Any]:
    """Parse JSON content, tolerating occasional fenced JSON responses."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


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
        return _parse_json_content(content)


@dataclass
class OllamaChatClient:
    """Minimal Ollama chat client using the local /api/chat endpoint."""

    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120

    def __post_init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", self.base_url).rstrip("/")

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text[:500]}")

        data = response.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama response did not include message.content: {str(data)[:500]}")
        return _parse_json_content(content)


def make_llm_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[LLMClient]:
    normalized = provider.lower().strip()
    if normalized in {"none", "disabled", "off"}:
        return None
    if normalized in {"openai", "openai_chat", "openai-compatible"}:
        return OpenAIChatClient(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
    if normalized in {"ollama", "olama"}:
        ollama_model = model if model and model != "gpt-4.1-mini" else "llama3.1:8b"
        return OllamaChatClient(
            model=ollama_model,
            base_url=base_url or "http://localhost:11434",
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
