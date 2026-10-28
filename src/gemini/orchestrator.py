"""
Gemini function-calling orchestrator — implements the complete
tool-calling lifecycle with validation, execution, and grounding.
"""

from __future__ import annotations
import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from google.genai import types

from src.config import GEMINI_MODEL, GEMINI_MAX_TOOL_CALLS
from src.gemini.client import get_client, SYSTEM_INSTRUCTION
from src.gemini.function_declarations import get_tool_declarations
from src.gemini.grounding import GroundingValidator
from src.analytics.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


def check_domain_guardrail(query: str) -> str | None:
    """
    Fast deterministic guardrail: blocks programming/code generation requests
    and out-of-scope non-retail queries.
    """
    if not query or not isinstance(query, str):
        return None

    q = query.strip().lower()

    # Retail exception whitelist: queries mentioning retail product codes, store codes, etc.
    # e.g., "what is product code P0001", "give me the promo code", "discount code"
    is_retail_code_context = bool(re.search(
        r"\b(product|sku|item|store|promo|promotion|discount|postal|zip|pin|hsn)\s+codes?\b",
        q
    ))

    # 1. Code generation / programming requests
    code_gen_patterns = [
        r"\b(write|generate|give me|show me|create|build|implement|provide|produce)\b.*\b(python|javascript|typescript|java|c\+\+|c#|php|ruby|golang|rust|swift|kotlin|bash|shell|powershell|html|css|script|algorithm|regex|calculator|fibonacci|bubble sort|binary search)\b",
        r"\b(write|generate|give me|create)\s+(a\s+)?code\b",
        r"\b(code for|code to|script for|script to|program for|program to)\b",
        r"\b(how to (code|program|write a script|write a function|reverse a string|reverse string))\b",
        r"\b(teach me (to code|coding|programming|how to code))\b",
        r"\b(write a program|write a script|write a function|write a class|write an api)\b",
        r"\b(can you (write|generate|give me) (some |a )?code)\b",
        r"\b(reverse (a |the )?string|palindrome check|factorial function|two sum|leetcode)\b",
        r"\b(coding|programmer|programming language)\b",
    ]

    if not is_retail_code_context:
        for pat in code_gen_patterns:
            if re.search(pat, q):
                return (
                    "I am the Retail Demand Forecasting Assistant for this platform. "
                    "My role is strictly dedicated to analyzing retail sales, demand forecasts, "
                    "store performance, pricing, festival trends, and inventory metrics from our retail dataset.\n\n"
                    "I cannot write, generate, or debug programming code, nor can I answer general questions "
                    "outside of retail demand forecasting.\n\n"
                    "Please feel free to ask me about product demand, 4-week forward forecasts, festival sales uplifts, "
                    "top-selling SKUs, or regional store performance!"
                )

    # 2. General non-retail trivia / off-topic queries
    trivia_patterns = [
        r"^\s*(who is the prime minister|who is the president|who is the king|capital of [a-z]+|who won the (world cup|ipl|super bowl)|tell me a joke|write a poem|write an essay|write a story|recipe for)\b",
    ]
    for pat in trivia_patterns:
        if re.search(pat, q):
            return (
                "I am the Retail Demand Forecasting Assistant for this platform. "
                "My role is strictly dedicated to analyzing retail sales, demand forecasts, "
                "store performance, pricing, festival trends, and inventory metrics from our retail dataset.\n\n"
                "I cannot answer general knowledge, political, or off-topic questions outside of retail demand forecasting.\n\n"
                "Please feel free to ask me about product demand, 4-week forward forecasts, festival sales uplifts, "
                "top-selling SKUs, or regional store performance!"
            )

    return None


class GeminiOrchestrator:
    """
    Manages the complete Gemini function-calling lifecycle:
    1. Receive user question
    2. Send to Gemini with tool declarations
    3. Validate and execute function calls
    4. Return function results to Gemini
    5. Get final explanation
    6. Validate grounding of numerical claims
    """

    MAX_RETRIES = 2

    def __init__(self):
        self.dispatcher = ToolDispatcher()
        self.validator = GroundingValidator()
        self.tools = get_tool_declarations()

    def _generate_with_retry(
        self,
        client,
        contents: list,
        config: types.GenerateContentConfig,
        retries: int = 2,
    ):
        """Call Gemini with automatic fallback across active models and retry on transient spikes."""
        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]
        models_to_try = []
        if GEMINI_MODEL and GEMINI_MODEL in candidate_models:
            models_to_try.append(GEMINI_MODEL)
        for m in candidate_models:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for model in models_to_try:
            for attempt in range(retries):
                try:
                    return client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    # If daily quota exhausted or model not found, switch immediately to next candidate
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "404" in err_msg or "NOT_FOUND" in err_msg:
                        logger.warning(
                            "Model %s unavailable (%s). Switching to next candidate model.",
                            model, "quota exhausted" if "429" in err_msg else "not found",
                        )
                        break
                    # If transient server spike (503), sleep and retry
                    if "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg:
                        logger.warning(
                            "Model %s transient error (attempt %d/%d): %s",
                            model, attempt + 1, retries, err_msg[:120],
                        )
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    break
        raise last_error

    def chat(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Process a user question through the full tool-calling lifecycle.

        Returns a dict with:
        - answer: The final grounded response text
        - evidence: Tool results used to ground the answer
        - tools_called: List of tool invocations
        - grounding_status: 'verified' | 'partial' | 'unverified' | 'no_tools_needed'
        - calculation_method: How the answer was computed
        """
        # ── Domain Guardrail: Restrict to Retail Forecasts & Dataset Only ──
        guardrail_refusal = check_domain_guardrail(user_message)
        if guardrail_refusal:
            logger.info("Guardrail triggered: intercepted out-of-scope query: %s", user_message[:60])
            return {
                "answer": guardrail_refusal,
                "evidence": {},
                "tools_called": [],
                "grounding_status": "no_tools_needed",
                "calculation_method": "Domain Scope Guardrail Policy",
            }

        client = get_client()
        tool_trace = []
        all_tool_results = {}

        # ── Build conversation contents ────────────────────────────
        contents = []
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 messages
                role = msg.get("role", "user")
                text = msg.get("text", "")
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)],
                ))

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        ))

        from src.gemini.learning import get_learning_system_instruction
        learning_directive = get_learning_system_instruction()
        effective_system_instruction = SYSTEM_INSTRUCTION
        if learning_directive:
            effective_system_instruction = f"{SYSTEM_INSTRUCTION}\n{learning_directive}"

        config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=effective_system_instruction,
            temperature=0.1,  # Low temperature for factual answers
        )

        # ── Initial call to Gemini ─────────────────────────────────
        try:
            response = self._generate_with_retry(client, contents, config)
        except Exception as e:
            logger.exception("Gemini API call failed")
            return {
                "answer": f"I'm unable to process your request right now. API error: {str(e)}",
                "evidence": {},
                "tools_called": [],
                "grounding_status": "error",
                "error": str(e),
            }

        # ── Tool-calling loop ──────────────────────────────────────
        tool_call_count = 0

        while tool_call_count < GEMINI_MAX_TOOL_CALLS:
            # Check if response has function calls
            if not response.candidates or not response.candidates[0].content:
                break

            parts = response.candidates[0].content.parts
            function_calls = [p for p in parts if p.function_call]

            if not function_calls:
                break  # No more function calls — we have the final answer

            # Execute each function call
            function_response_parts = []

            for part in function_calls:
                fc = part.function_call
                tool_call_count += 1

                logger.info("Tool call %d: %s(%s)", tool_call_count, fc.name, fc.args)

                # Dispatch through allowlisted handler
                result = self.dispatcher.dispatch(fc.name, dict(fc.args) if fc.args else {})

                tool_trace.append({
                    "function": fc.name,
                    "arguments": dict(fc.args) if fc.args else {},
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "non-dict",
                    "has_error": "error" in result if isinstance(result, dict) else False,
                })

                all_tool_results[fc.name] = result

                # Build function response
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": _sanitize_for_json(result)},
                    )
                )

            # Add model's function call and our response to conversation
            contents.append(response.candidates[0].content)
            contents.append(types.Content(
                role="user",
                parts=function_response_parts,
            ))

            # Call Gemini again with function results
            try:
                response = self._generate_with_retry(client, contents, config)
            except Exception as e:
                logger.exception("Gemini API call failed during tool-calling loop")
                return {
                    "answer": f"Error during processing: {str(e)}",
                    "evidence": all_tool_results,
                    "tools_called": tool_trace,
                    "grounding_status": "error",
                    "error": str(e),
                }

        # ── Extract final answer ───────────────────────────────────
        final_text = ""
        if response.candidates and response.candidates[0].content:
            text_parts = [
                p.text for p in response.candidates[0].content.parts
                if p.text
            ]
            final_text = "\n".join(text_parts)

        # ── Grounding validation ───────────────────────────────────
        grounding_status = "no_tools_needed"
        if all_tool_results:
            grounding_result = self.validator.validate(final_text, all_tool_results)
            grounding_status = grounding_result["status"]

            if grounding_status == "failed" and self.MAX_RETRIES > 0:
                # Attempt re-generation with explicit grounding reminder
                logger.warning("Grounding validation failed, attempting correction...")
                correction_prompt = (
                    "Your previous answer contained claims not supported by the tool results. "
                    "Please revise your answer to only include numbers and facts that are "
                    "directly present in the function results provided. Do not add any numbers "
                    "that weren't returned by the tools."
                )
                contents.append(response.candidates[0].content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=correction_prompt)],
                ))
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=config,
                    )
                    if response.candidates and response.candidates[0].content:
                        text_parts = [
                            p.text for p in response.candidates[0].content.parts
                            if p.text
                        ]
                        final_text = "\n".join(text_parts)
                        grounding_result = self.validator.validate(final_text, all_tool_results)
                        grounding_status = grounding_result["status"]
                except Exception:
                    logger.exception("Correction attempt failed")

        # ── Build response ─────────────────────────────────────────
        # Extract calculation method from tool results
        calc_method = ""
        for tool_name, result in all_tool_results.items():
            if isinstance(result, dict) and "calculation_method" in result:
                calc_method = result["calculation_method"]
                break

        return {
            "answer": final_text,
            "evidence": _clean_evidence(all_tool_results),
            "tools_called": tool_trace,
            "grounding_status": grounding_status,
            "calculation_method": calc_method,
            "timestamp": datetime.now().isoformat(),
        }


def _sanitize_for_json(obj: Any) -> Any:
    """Convert non-JSON-serializable types for the API."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    return str(obj)


def _clean_evidence(tool_results: dict) -> dict:
    """Remove internal metadata from evidence for display."""
    cleaned = {}
    for name, result in tool_results.items():
        if isinstance(result, dict):
            cleaned[name] = {
                k: v for k, v in result.items()
                if k != "_tool_metadata"
            }
        else:
            cleaned[name] = result
    return cleaned
