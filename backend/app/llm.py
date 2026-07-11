"""
Pluggable LLM client.

Design note: the planning and insight-writing steps in this agent need a
language model. Rather than hard-coding one vendor, this module exposes a
single `LLMClient.complete()` call. If ANTHROPIC_API_KEY is present in the
environment, it calls the real Anthropic Messages API. If not, it uses a
transparent, clearly-labeled rule-based fallback so the rest of the graph
(SQL execution, stats, charts, cleaning) can be fully exercised and tested
without requiring API access. Swap `_fallback_complete` for any other
provider's SDK without touching any node code.
"""

from __future__ import annotations
import json
import os
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


class LLMClient:
    """Thin wrapper so nodes never care which backend answers them."""

    def __init__(self) -> None:
        """Initialize the client, detecting whether a live Anthropic key is available."""
        self.live = bool(ANTHROPIC_API_KEY) and httpx is not None
        self.provider = "anthropic-live" if self.live else "rule-based-fallback"

    async def complete(
        self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1000
    ) -> str:
        """Send a completion request to the configured backend.

        Args:
            system: System prompt text.
            user: User message text.
            json_mode: Hint to the fallback to return a JSON-shaped response.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's response as a plain string.
        """
        if self.live:
            return await self._live_complete(system, user, max_tokens=max_tokens)
        return self._fallback_complete(system, user, json_mode=json_mode)

    async def _live_complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """Call the Anthropic Messages API and return the text response.

        Args:
            system: System prompt text.
            user: User message text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Concatenated text content from the API response.
        """
        assert httpx is not None
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=body
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )

    def _fallback_complete(self, system: str, user: str, *, json_mode: bool) -> str:
        """
        Deterministic, keyword-driven stand-in for an LLM call. It is NOT a
        real language model — it exists so the agent pipeline is genuinely
        runnable end-to-end without network access. Every node that calls
        this degrades gracefully (simpler plan / templated prose) rather
        than silently faking a smarter result.

        Args:
            system: System prompt text.
            user: User message text.
            json_mode: Hint to return a JSON-shaped response.

        Returns:
            A deterministic fake response based on the input.
        """
        text = (system + " " + user).lower()

        if "classify the analytical intent" in system.lower():
            plan = {
                "needs_sql": True,
                "needs_stats": any(
                    k in text
                    for k in [
                        "correlat",
                        "significan",
                        "confidence",
                        "hypothesis",
                        "regress",
                    ]
                ),
                "needs_forecast": any(
                    k in text for k in ["forecast", "predict", "next", "trend"]
                ),
                "needs_segmentation": any(
                    k in text for k in ["segment", "cohort", "churn", "retention"]
                ),
                "intent": "general_analysis",
                "reasoning": "rule-based fallback: keyword match on the question (no live LLM configured)",
            }
            return json.dumps(plan)

        if json_mode:
            return json.dumps(
                {"note": "fallback LLM: no live model configured", "content": []}
            )

        return (
            "[Generated by rule-based fallback — no ANTHROPIC_API_KEY configured. "
            "Set ANTHROPIC_API_KEY to enable live narrative generation.]"
        )


llm_client = LLMClient()
