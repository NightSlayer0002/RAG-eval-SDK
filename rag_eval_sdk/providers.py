"""Small, rate-aware adapters for OpenAI-compatible generation APIs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock, get_ident
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached: bool = False
    response_id: str | None = None


class TextGenerator(Protocol):
    name: str

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> GenerationResult:
        """Generate one response."""


class GoogleGenAIProvider:
    """Lazy Google GenAI adapter with no import-time credential requirement."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        requests_per_minute: float | None = None,
        max_retries: int = 3,
    ) -> None:
        if not model.strip():
            raise ValueError("model is required")
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.requests_per_minute = requests_per_minute
        self.max_retries = max(0, max_retries)
        self.name = f"google-genai:{model}"
        self._client = None
        self._types = None
        self._last_request_started = 0.0
        self._lock = RLock()

    def _load(self):
        with self._lock:
            if self._client is None:
                key = self.api_key or os.getenv(self.api_key_env)
                if not key:
                    raise RuntimeError(f"No API key was provided; set {self.api_key_env}.")
                try:
                    from google import genai
                    from google.genai import types
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "Google GenAI is not installed. Install rag-eval-sdk[google]."
                    ) from exc
                self._client = genai.Client(api_key=key)
                self._types = types
        return self._client, self._types

    def _wait_for_rate_limit(self) -> None:
        if self.requests_per_minute is None:
            return
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        interval = 60.0 / self.requests_per_minute
        with self._lock:
            wait = interval - (time.monotonic() - self._last_request_started)
            if wait > 0:
                time.sleep(wait)
            self._last_request_started = time.monotonic()

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> GenerationResult:
        if not messages:
            raise ValueError("messages cannot be empty")
        client, types = self._load()
        # Role labels preserve intent while remaining compatible with the simple
        # contents form across google-genai releases.
        contents = "\n\n".join(
            f"[{str(message.get('role', 'user')).upper()}]\n{message.get('content', '')}"
            for message in messages
        )
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=float(temperature),
                        max_output_tokens=int(max_tokens),
                    ),
                )
                usage = getattr(response, "usage_metadata", None)
                return GenerationResult(
                    text=str(response.text or "").strip(),
                    model=self.model,
                    latency_seconds=time.perf_counter() - started,
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                    response_id=getattr(response, "response_id", None),
                )
            except Exception as exc:  # pragma: no cover - provider specific
                retryable = any(
                    marker in str(exc).casefold()
                    for marker in ("429", "resource_exhausted", "timeout", "503", "502")
                )
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"Google GenAI request failed: {exc}") from exc
                time.sleep(min(60.0, (2**attempt) + random.random()))
        raise RuntimeError("Generation failed after retries")


class OpenAICompatibleProvider:
    """Standard-library client with bounded retries and an optional RPM limit."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        requests_per_minute: float | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url and model are required")
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.requests_per_minute = requests_per_minute
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.default_headers = dict(default_headers or {})
        self.name = f"openai-compatible:{self.base_url}:{self.model}"
        self._last_request_started = 0.0
        self._rate_lock = RLock()

    @classmethod
    def nvidia_nim(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        requests_per_minute: float | None = None,
        **kwargs: object,
    ) -> "OpenAICompatibleProvider":
        """Configure NVIDIA NIM; the caller still chooses the model and RPM."""

        return cls(
            base_url="https://integrate.api.nvidia.com/v1",
            model=model,
            api_key=api_key,
            api_key_env="NVIDIA_API_KEY",
            requests_per_minute=requests_per_minute,
            **kwargs,
        )

    @classmethod
    def groq(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        requests_per_minute: float | None = None,
        **kwargs: object,
    ) -> "OpenAICompatibleProvider":
        """Configure Groq; the caller still chooses the model and RPM."""

        return cls(
            base_url="https://api.groq.com/openai/v1",
            model=model,
            api_key=api_key,
            api_key_env="GROQ_API_KEY",
            requests_per_minute=requests_per_minute,
            **kwargs,
        )

    def _resolved_key(self) -> str:
        key = self.api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        if not key:
            source = self.api_key_env or "the api_key argument"
            raise RuntimeError(f"No API key was provided; set {source}.")
        return key

    def _wait_for_rate_limit(self) -> None:
        if self.requests_per_minute is None:
            return
        interval = 60.0 / self.requests_per_minute
        with self._rate_lock:
            now = time.monotonic()
            wait = interval - (now - self._last_request_started)
            if wait > 0:
                time.sleep(wait)
            self._last_request_started = time.monotonic()

    @staticmethod
    def _retry_after(headers: object, fallback: float) -> float:
        try:
            value = headers.get("Retry-After")  # type: ignore[union-attr]
            return min(120.0, max(0.0, float(value))) if value else fallback
        except (AttributeError, TypeError, ValueError):
            return fallback

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> GenerationResult:
        if not messages:
            raise ValueError("messages cannot be empty")
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._resolved_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.default_headers,
        }
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                text = str(parsed["choices"][0]["message"]["content"])
                usage = parsed.get("usage") or {}
                return GenerationResult(
                    text=text,
                    model=str(parsed.get("model") or self.model),
                    latency_seconds=time.perf_counter() - started,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    response_id=parsed.get("id"),
                )
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(f"Generation API returned HTTP {exc.code}: {detail}") from exc
                fallback = min(60.0, (2**attempt) + random.random())
                time.sleep(self._retry_after(exc.headers, fallback))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Generation API request failed: {exc}") from exc
                time.sleep(min(60.0, (2**attempt) + random.random()))
        raise RuntimeError("Generation failed after retries")


class DiskCachedGenerator:
    """Content-addressed response cache for reproducible, low-RPM benchmarks."""

    def __init__(self, generator: TextGenerator, cache_directory: str | Path) -> None:
        self.generator = generator
        self.cache_directory = Path(cache_directory)
        self.name = f"disk-cached:{generator.name}"
        self._lock = RLock()

    def _key(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "generator": self.generator.name,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_schema": 1,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> GenerationResult:
        key = self._key(messages, temperature, max_tokens)
        path = self.cache_directory / f"{key}.json"
        with self._lock:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["cached"] = True
                return GenerationResult(**payload)

        result = self.generator.generate(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        payload = asdict(result)
        with self._lock:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.{get_ident()}.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        return result
