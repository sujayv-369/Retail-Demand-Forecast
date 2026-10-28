"""
Grounding validator — checks that Gemini's final response text
contains only numbers and facts supported by tool results.

This is an application-level safeguard, not a prompt-only defense.
"""

from __future__ import annotations
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class GroundingValidator:
    """
    Validates that numerical claims in Gemini's response are
    grounded in actual tool results.
    """

    def validate(self, response_text: str, tool_results: dict[str, Any]) -> dict:
        """
        Check response for numeric claims against tool results.

        Returns:
            status: 'verified' | 'partial' | 'failed' | 'no_numbers'
            details: list of check results
        """
        if not response_text:
            return {"status": "no_numbers", "details": []}

        # Extract all numbers from the response
        numbers_in_response = self._extract_numbers(response_text)

        if not numbers_in_response:
            return {"status": "no_numbers", "details": []}

        # Collect all numbers from tool results
        tool_numbers = set()
        self._collect_numbers(tool_results, tool_numbers)

        # Check each response number against tool numbers
        checks = []
        verified = 0
        total = len(numbers_in_response)

        for num in numbers_in_response:
            found = self._number_in_evidence(num, tool_numbers)
            checks.append({
                "number": num,
                "verified": found,
            })
            if found:
                verified += 1

        # Determine status
        if verified == total:
            status = "verified"
        elif verified > 0:
            status = "partial"
        elif total > 0:
            status = "failed"
        else:
            status = "no_numbers"

        return {
            "status": status,
            "total_numbers": total,
            "verified_numbers": verified,
            "details": checks,
        }

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract numeric values from text, excluding common non-data numbers."""
        # Match integers and decimals, including percentages
        pattern = r'(?<!\w)[-]?\d+(?:,\d{3})*(?:\.\d+)?(?:%)?(?!\w)'
        matches = re.findall(pattern, text)

        numbers = []
        skip_values = {0, 1, 2, 3, 4, 5, 10, 100}  # Common non-data numbers

        for match in matches:
            # Clean up
            clean = match.replace(",", "").replace("%", "")
            try:
                val = float(clean)
                if val not in skip_values:
                    numbers.append(val)
            except ValueError:
                continue

        return numbers

    def _collect_numbers(self, obj: Any, numbers: set, depth: int = 0):
        """Recursively collect all numeric values from tool results."""
        if depth > 10:
            return

        if isinstance(obj, (int, float)):
            if obj == obj:  # Not NaN
                numbers.add(float(obj))
                # Also add common derived forms
                numbers.add(round(obj, 1))
                numbers.add(round(obj, 2))
                numbers.add(round(obj, 3))
                if obj != 0:
                    numbers.add(abs(obj))
        elif isinstance(obj, str):
            # Try to extract numbers from string values
            try:
                numbers.add(float(obj))
            except ValueError:
                pass
        elif isinstance(obj, dict):
            for v in obj.values():
                self._collect_numbers(v, numbers, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect_numbers(item, numbers, depth + 1)

    def _number_in_evidence(self, num: float, tool_numbers: set, tolerance: float = 0.5) -> bool:
        """
        Check if a number from the response matches any number from tool results.
        Uses tolerance for floating-point comparison.
        """
        if num in tool_numbers:
            return True

        # Check with tolerance
        for tn in tool_numbers:
            if abs(num - tn) <= tolerance:
                return True
            # Check percentage relationship
            if tn != 0 and abs(num - 100 * tn) < tolerance:
                return True

        return False
