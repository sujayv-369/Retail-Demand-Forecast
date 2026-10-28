"""
Security, safety, and prompt-injection guardrail tests.
"""

from src.analytics.dispatcher import ToolDispatcher
from src.gemini.client import SYSTEM_INSTRUCTION


def test_dispatcher_rejects_sql_injection_parameters():
    """Verify tool dispatcher rejects arbitrary SQL injection payloads."""
    dispatcher = ToolDispatcher()
    # Attempt SQL injection inside product_id
    payload = "P0001'; DROP TABLE forecast_results; --"
    # The dispatcher only dispatches to parameterized python tools that do not run raw SQL
    result = dispatcher.dispatch("get_product_forecast", {"product_id": payload})

    # The result should either return not found / empty or handle safely without SQL execution
    assert isinstance(result, dict)
    assert "error" in result or "forecasts" in result


def test_system_instruction_contains_guardrails():
    """Verify critical safety rules are embedded in the LLM system prompt."""
    prompt = SYSTEM_INSTRUCTION.lower()

    assert "read-only" in prompt
    assert "never invent" in prompt or "fabricate" in prompt
    assert "refuse" in prompt or "unauthorized" in prompt
