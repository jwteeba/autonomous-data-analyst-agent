"""Tests for app/llm.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Helpers — build a fresh LLMClient without touching the module-level singleton
def make_client(api_key: str | None = None, has_httpx: bool = True):
    """Return a new LLMClient with controlled env and httpx availability."""
    import app.llm as llm_mod

    with patch.object(llm_mod, "ANTHROPIC_API_KEY", api_key), patch.object(
        llm_mod, "httpx", MagicMock() if has_httpx else None
    ):
        client = llm_mod.LLMClient()
    return client


class TestLLMClientInit:
    def test_live_false_without_key(self):
        client = make_client(api_key=None)
        assert client.live is False
        assert client.provider == "rule-based-fallback"

    def test_live_false_without_httpx(self):
        client = make_client(api_key="sk-ant-test", has_httpx=False)
        assert client.live is False

    def test_live_true_with_key_and_httpx(self):
        client = make_client(api_key="sk-ant-test", has_httpx=True)
        assert client.live is True
        assert client.provider == "anthropic-live"


class TestFallbackComplete:
    def setup_method(self):
        import app.llm as llm_mod

        with patch.object(llm_mod, "ANTHROPIC_API_KEY", None):
            self.client = llm_mod.LLMClient()

    def test_planner_returns_valid_json(self):
        system = "Classify the analytical intent of the user's business question"
        user = "What is our revenue trend?"
        result = self.client._fallback_complete(system, user, json_mode=True)
        plan = json.loads(result)
        assert "needs_sql" in plan
        assert "needs_forecast" in plan
        assert "intent" in plan

    def test_planner_detects_forecast_keyword(self):
        system = "Classify the analytical intent"
        user = "forecast next quarter revenue"
        result = self.client._fallback_complete(system, user, json_mode=False)
        plan = json.loads(result)
        assert plan["needs_forecast"] is True

    def test_planner_detects_correlation_keyword(self):
        system = "Classify the analytical intent"
        user = "show correlation between price and quantity"
        result = self.client._fallback_complete(system, user, json_mode=False)
        plan = json.loads(result)
        assert plan["needs_stats"] is True

    def test_json_mode_returns_json(self):
        result = self.client._fallback_complete(
            "other system", "user msg", json_mode=True
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_non_json_mode_returns_string(self):
        result = self.client._fallback_complete(
            "other system", "user msg", json_mode=False
        )
        assert isinstance(result, str)
        assert "fallback" in result.lower()


class TestCompleteRouting:
    @pytest.mark.asyncio
    async def test_routes_to_fallback_when_not_live(self):
        import app.llm as llm_mod

        with patch.object(llm_mod, "ANTHROPIC_API_KEY", None):
            client = llm_mod.LLMClient()
        result = await client.complete("sys", "usr", json_mode=True)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_routes_to_live_when_live(self):
        import app.llm as llm_mod

        mock_httpx = MagicMock()
        with patch.object(llm_mod, "ANTHROPIC_API_KEY", "sk-ant-test"), patch.object(
            llm_mod, "httpx", mock_httpx
        ):
            client = llm_mod.LLMClient()
            client._live_complete = AsyncMock(return_value="live response")
            result = await client.complete("sys", "usr")
        assert result == "live response"
