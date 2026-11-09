"""Groq API provider implementation with SSE streaming and rate limit telemetry."""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import httpx

from config.settings import settings
from .base import BaseModelProvider, ModelMessage, ModelResponse

logger = logging.getLogger(__name__)


class GroqProvider(BaseModelProvider):
    """Client for live Groq API with header tracking and millisecond timing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        super().__init__(provider_name="groq", source_type="real")
        self.api_key = api_key or settings.GROQ_API_KEY
        self.base_url = (base_url or settings.GROQ_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.GROQ_MODEL
        self.last_rate_limit_headers: Dict[str, str] = {}

    def _extract_rate_limit_headers(self, headers: httpx.Headers) -> Dict[str, str]:
        """Extract and store rate-limit headers from Groq HTTP response."""
        extracted = {}
        for key in [
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "retry-after",
        ]:
            if key in headers:
                extracted[key] = headers[key]
        self.last_rate_limit_headers = extracted
        return extracted

    async def generate(
        self,
        messages: List[ModelMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        stream: bool = True,
    ) -> ModelResponse:
        """Execute request against Groq API, capturing TTFT and full latency."""
        if not self.api_key:
            return ModelResponse(
                content="",
                first_token_ms=0.0,
                total_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                retries=0,
                model_name=model or self.default_model,
                source=self.source_type,
                status="error",
                error_message="GROQ_API_KEY is not configured.",
            )

        chosen_model = model or self.default_model
        payload = {
            "model": chosen_model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Estimate prompt tokens
        prompt_text = " ".join(m.content for m in messages)
        est_tokens_in = self.estimate_tokens(prompt_text)

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        content_parts: List[str] = []
        tokens_out_count = 0

        endpoint = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                if stream:
                    async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                        self._extract_rate_limit_headers(response.headers)

                        if response.status_code == 429:
                            error_body = await response.aread()
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            return ModelResponse(
                                content="",
                                first_token_ms=0.0,
                                total_ms=elapsed_ms,
                                tokens_in=est_tokens_in,
                                tokens_out=0,
                                retries=0,
                                model_name=chosen_model,
                                source=self.source_type,
                                status="rate_limited",
                                rate_limit_headers=self.last_rate_limit_headers,
                                error_message=f"HTTP 429: {error_body.decode('utf-8', errors='ignore')}",
                            )

                        if response.status_code != 200:
                            error_body = await response.aread()
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            return ModelResponse(
                                content="",
                                first_token_ms=0.0,
                                total_ms=elapsed_ms,
                                tokens_in=est_tokens_in,
                                tokens_out=0,
                                retries=0,
                                model_name=chosen_model,
                                source=self.source_type,
                                status="error",
                                rate_limit_headers=self.last_rate_limit_headers,
                                error_message=f"HTTP {response.status_code}: {error_body.decode('utf-8', errors='ignore')}",
                            )

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    delta_content = delta.get("content", "")
                                    if delta_content:
                                        if first_token_time is None:
                                            first_token_time = time.perf_counter()
                                        content_parts.append(delta_content)
                                        tokens_out_count += 1
                                # Also check if usage was included in chunk
                                if "usage" in chunk_json and chunk_json["usage"]:
                                    usage = chunk_json["usage"]
                                    if "prompt_tokens" in usage:
                                        est_tokens_in = usage["prompt_tokens"]
                                    if "completion_tokens" in usage:
                                        tokens_out_count = usage["completion_tokens"]
                            except json.JSONDecodeError:
                                continue

                else:
                    # Non-streaming request
                    response = await client.post(endpoint, headers=headers, json=payload)
                    self._extract_rate_limit_headers(response.headers)
                    if response.status_code == 429:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return ModelResponse(
                            content="",
                            first_token_ms=0.0,
                            total_ms=elapsed_ms,
                            tokens_in=est_tokens_in,
                            tokens_out=0,
                            retries=0,
                            model_name=chosen_model,
                            source=self.source_type,
                            status="rate_limited",
                            rate_limit_headers=self.last_rate_limit_headers,
                            error_message=f"HTTP 429: {response.text}",
                        )

                    if response.status_code != 200:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return ModelResponse(
                            content="",
                            first_token_ms=0.0,
                            total_ms=elapsed_ms,
                            tokens_in=est_tokens_in,
                            tokens_out=0,
                            retries=0,
                            model_name=chosen_model,
                            source=self.source_type,
                            status="error",
                            rate_limit_headers=self.last_rate_limit_headers,
                            error_message=f"HTTP {response.status_code}: {response.text}",
                        )

                    res_json = response.json()
                    first_token_time = time.perf_counter()
                    content = res_json["choices"][0]["message"]["content"]
                    content_parts.append(content)
                    if "usage" in res_json:
                        est_tokens_in = res_json["usage"].get("prompt_tokens", est_tokens_in)
                        tokens_out_count = res_json["usage"].get("completion_tokens", len(content.split()))

            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ModelResponse(
                    content="",
                    first_token_ms=0.0,
                    total_ms=elapsed_ms,
                    tokens_in=est_tokens_in,
                    tokens_out=0,
                    retries=0,
                    model_name=chosen_model,
                    source=self.source_type,
                    status="timeout",
                    error_message=f"Request timed out after {timeout_s}s: {exc}",
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ModelResponse(
                    content="",
                    first_token_ms=0.0,
                    total_ms=elapsed_ms,
                    tokens_in=est_tokens_in,
                    tokens_out=0,
                    retries=0,
                    model_name=chosen_model,
                    source=self.source_type,
                    status="error",
                    error_message=f"Connection/Transport error: {exc}",
                )

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000.0
        first_token_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else total_ms
        full_content = "".join(content_parts)

        # Fallback token estimate if not provided by Groq
        if tokens_out_count == 0 and full_content:
            tokens_out_count = self.estimate_tokens(full_content)

        return ModelResponse(
            content=full_content,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            tokens_in=est_tokens_in,
            tokens_out=tokens_out_count,
            retries=0,
            model_name=chosen_model,
            source=self.source_type,
            status="ok",
            rate_limit_headers=self.last_rate_limit_headers,
        )

    async def stream_tokens(
        self,
        messages: List[ModelMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncGenerator[Tuple[str, Optional[ModelResponse]], None]:
        """Yield tokens as they arrive, emitting the final ModelResponse as the last element."""
        response = await self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            stream=True,
        )
        # Yield the tokens and then the completed ModelResponse
        yield response.content, response
