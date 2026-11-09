"""Abstract Base Class and unified data structures for LLM providers."""

from abc import ABC, abstractmethod
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModelMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ModelResponse(BaseModel):
    content: str
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    model_name: str
    source: str  # "real" or "stand-in"
    status: str  # "ok", "rate_limited", "timeout", "error"
    rate_limit_headers: Dict[str, str] = Field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "ok"


class BaseModelProvider(ABC):
    """Abstract contract for real and simulated model providers."""

    def __init__(self, provider_name: str, source_type: str):
        self.provider_name = provider_name
        self.source_type = source_type  # "real" or "stand-in"

    @abstractmethod
    async def generate(
        self,
        messages: List[ModelMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        stream: bool = True,
    ) -> ModelResponse:
        """Execute completion request and return unified ModelResponse with exact timing."""
        pass

    @abstractmethod
    async def stream_tokens(
        self,
        messages: List[ModelMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncGenerator[Tuple[str, Optional[ModelResponse]], None]:
        """Yield (token_chunk, final_response_or_none) for real-time streaming."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a given string (standard approximation: ~1.3 tokens per word)."""
        words = len(text.split())
        return max(1, int(words * 1.3))
