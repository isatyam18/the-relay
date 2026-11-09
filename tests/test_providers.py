"""Unit tests for Provider abstraction and Calibration Probe."""

import pytest
from src.providers.base import BaseModelProvider, ModelMessage, ModelResponse
from src.providers.groq_provider import GroqProvider
from src.bench.groq_calibration_probe import GroqCalibrationProbe, GroqEmpiricalProfile


def test_provider_token_estimation():
    """Verify standard token estimation logic."""
    provider = GroqProvider(api_key="mock_key")
    text = "The Transformer is the first transduction model relying entirely on self-attention."
    est = provider.estimate_tokens(text)
    assert est > 5
    assert est < 30


@pytest.mark.asyncio
async def test_groq_provider_missing_key():
    """Verify GroqProvider handles missing API key cleanly with an error status instead of crashing."""
    provider = GroqProvider(api_key="")
    messages = [ModelMessage(role="user", content="Hello")]
    resp = await provider.generate(messages=messages)

    assert isinstance(resp, ModelResponse)
    assert resp.status == "error"
    assert resp.source == "real"
    assert "GROQ_API_KEY is not configured" in resp.error_message


def test_calibration_probe_profile_schema(tmp_path):
    """Verify calibration profile generates all fields mandated by Assignment Table 1."""
    profile_path = tmp_path / "test_calibration.json"
    probe = GroqCalibrationProbe(api_key="", output_path=profile_path)
    profile = probe.get_reference_profile()

    assert isinstance(profile, GroqEmpiricalProfile)
    assert profile.rate_limit_rpm >= 30
    assert profile.rate_limit_tpm >= 6000
    assert profile.rate_limit_rpd >= 1000
    assert profile.ttft_p50_ms > 0
    assert profile.total_time_p50_ms > profile.ttft_p50_ms
    assert profile.tokens_per_second_mean > 0
    assert len(profile.rate_limit_error_format) > 0
    assert len(profile.recovery_behavior) > 0
    assert profile.confidence in {"high", "medium", "low"}

    # Verify JSON persistence
    assert profile_path.exists()
    assert "ttft_p50_ms" in profile_path.read_text(encoding="utf-8")
