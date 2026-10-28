"""
Unit tests for the GroundingValidator.
"""

from src.gemini.grounding import GroundingValidator


def test_grounding_verified_when_numbers_match():
    """Grounding should be verified when all numbers in response exist in tool evidence."""
    validator = GroundingValidator()
    text = "The CatBoost model achieved an MAE of 3.77 and a WAPE of 60.3% over 251650 observations."
    evidence = {
        "compare_forecast_models": {
            "overall_comparison": [
                {"model": "catboost", "mae": 3.767, "wape": 60.288, "n_observations": 251650}
            ]
        }
    }

    result = validator.validate(text, evidence)
    assert result["status"] in ("verified", "partial")
    assert result["verified_numbers"] >= 1


def test_grounding_flags_hallucinated_numbers():
    """Grounding should flag completely fabricated numbers."""
    validator = GroundingValidator()
    text = "Sales surged to 999999 units with profit of 888888 dollars."
    evidence = {
        "tool_result": {
            "units": 150,
            "cost": 25.5
        }
    }

    result = validator.validate(text, evidence)
    assert result["status"] == "failed"
    assert result["verified_numbers"] == 0
    assert result["total_numbers"] >= 2
    # Check details flag ungrounded
    unmatched = [d for d in result["details"] if not d["verified"]]
    assert len(unmatched) >= 2
