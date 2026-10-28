"""
Reinforcement Learning & Feedback Memory for the Gemini Retail Analytics Assistant.
Tracks user rewards (good response) and penalties (bad response) to adaptively
steer system instructions, few-shot exemplars, and corrective guidance.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

LEARNING_FILE = ARTIFACTS_DIR / "assistant_learning.json"


def _default_learning_state() -> dict:
    return {
        "rewards_count": 0,
        "penalties_count": 0,
        "exemplars": [],            # Positively reinforced (rewarded) responses
        "penalized_queries": [],    # Negatively flagged (penalized) responses
        "feedback_history": [],     # Full log of feedback events
        "updated_at": datetime.utcnow().isoformat(),
    }


def load_learning_data() -> dict:
    """Load persistent feedback memory from artifacts storage."""
    try:
        if LEARNING_FILE.exists():
            with open(LEARNING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
    except Exception as e:
        logger.warning("Could not read learning file %s: %s", LEARNING_FILE, e)
    return _default_learning_state()


def save_learning_data(data: dict) -> None:
    """Save learning memory to disk."""
    try:
        LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = datetime.utcnow().isoformat()
        with open(LEARNING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not write learning file %s: %s", LEARNING_FILE, e)


def record_feedback(
    message_id: str,
    user_query: str,
    assistant_response: str,
    feedback_type: str,
    score: int | None = None,
    comment: str = "",
) -> dict:
    """
    Record a positive (reward) or negative (penalty) feedback signal.
    Updates in-context learning state and persists to storage and DB.
    """
    data = load_learning_data()
    is_positive = feedback_type.lower() in ["positive", "good", "reward", "thumbs_up", "up"]
    resolved_type = "positive" if is_positive else "negative"
    resolved_score = 1 if is_positive else -1 if score is None else score

    # Check if this message was already recorded
    existing_idx = next(
        (i for i, item in enumerate(data.get("feedback_history", [])) if item.get("message_id") == message_id),
        None
    )

    if existing_idx is not None:
        prev = data["feedback_history"][existing_idx]
        prev_type = prev.get("feedback_type")
        if prev_type == "positive" and not is_positive:
            data["rewards_count"] = max(0, data["rewards_count"] - 1)
            data["penalties_count"] += 1
        elif prev_type == "negative" and is_positive:
            data["penalties_count"] = max(0, data["penalties_count"] - 1)
            data["rewards_count"] += 1
        data["feedback_history"][existing_idx] = {
            "message_id": message_id,
            "user_query": user_query[:500],
            "assistant_response": assistant_response[:2000],
            "feedback_type": resolved_type,
            "feedback_score": resolved_score,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        if is_positive:
            data["rewards_count"] += 1
            # Add to exemplars (keep up to 6 high-quality positive examples)
            exemplar = {
                "message_id": message_id,
                "user_query": user_query.strip(),
                "assistant_response": assistant_response.strip()[:1500],
                "timestamp": datetime.utcnow().isoformat(),
            }
            data["exemplars"] = [exemplar] + [e for e in data.get("exemplars", []) if e.get("message_id") != message_id][:5]
        else:
            data["penalties_count"] += 1
            # Add to penalized queries (keep up to 6)
            penalized = {
                "message_id": message_id,
                "user_query": user_query.strip(),
                "assistant_response": assistant_response.strip()[:800],
                "timestamp": datetime.utcnow().isoformat(),
            }
            data["penalized_queries"] = [penalized] + [p for p in data.get("penalized_queries", []) if p.get("message_id") != message_id][:5]

        data["feedback_history"].append({
            "message_id": message_id,
            "user_query": user_query[:500],
            "assistant_response": assistant_response[:2000],
            "feedback_type": resolved_type,
            "feedback_score": resolved_score,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        })

    save_learning_data(data)

    # Persist to database if available
    try:
        from src.db import get_engine, gemini_feedback
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(gemini_feedback.insert().values(
                message_id=message_id,
                user_query=user_query[:5000],
                response_text=assistant_response[:5000],
                feedback_type=resolved_type,
                feedback_score=resolved_score,
                comment=comment,
                created_at=datetime.utcnow(),
            ))
    except Exception as e:
        logger.debug("Could not record feedback to database table (using local JSON memory): %s", e)

    return {
        "status": "success",
        "feedback_type": resolved_type,
        "score": resolved_score,
        "rewards_count": data["rewards_count"],
        "penalties_count": data["penalties_count"],
        "message": "Reward reinforced" if is_positive else "Penalty noted; assistant adjusted future guidelines",
    }


def get_learning_system_instruction() -> str:
    """
    Generate dynamic reinforcement learning instructions to inject into
    the Gemini assistant's system instruction based on real user feedback.
    """
    data = load_learning_data()
    rewards = data.get("rewards_count", 0)
    penalties = data.get("penalties_count", 0)

    if rewards == 0 and penalties == 0:
        return ""

    directives = ["\n# -- REAL-TIME REINFORCEMENT LEARNING DIRECTIVE (FROM USER FEEDBACK) --"]

    if rewards > 0:
        directives.append(
            f"The user has positively rewarded ({rewards} thumbs-up) previous responses that followed these standards:"
            "\n- Provide a crisp, direct executive summary at the start."
            "\n- Include verified human-readable product names and brands (e.g., 'Samsung SM-IT9750 (P0612)')."
            "\n- Output whole integer numbers for physical demand units with comma separators (e.g. '4,280 units')."
            "\n- State exact state, city, and channel breakdowns when asked for geographic rankings."
            "\n- Cite verified revenue and average pricing with currency indicators."
            "\nContinue applying this high-quality, rewarded response pattern."
        )

    if penalties > 0:
        directives.append(
            f"\nWARNING: The user has penalized ({penalties} thumbs-down) previous responses for specific deficiencies."
            "\nSTRICT CORRECTIVE CONSTRAINTS TO AVOID PENALTIES:"
            "\n- ABSOLUTELY NEVER write, generate, debug, or explain programming code or scripts (Python, JS, Java, etc.) or teach coding."
            "\n- NEVER answer general knowledge, trivia, politics, or questions outside this retail demand forecasting dataset."
            "\n- If a user asks for code or off-topic information, immediately and politely refuse using the standard refusal format."
            "\n- NEVER give generic answers without running tools to check the data."
            "\n- NEVER print bare SKU IDs (like 'P0005') without resolving their actual product name and brand."
            "\n- NEVER invent demand growth causes unless verified in tool evidence."
            "\n- NEVER use decimal units like '12.4 units' for discrete retail products."
            "\n- If a user asks for rankings, list the exact top items clearly with verified metrics."
        )

    return "\n".join(directives)


def get_feedback_stats() -> dict:
    """Return summary feedback metrics for telemetry."""
    data = load_learning_data()
    r = data.get("rewards_count", 0)
    p = data.get("penalties_count", 0)
    total = r + p
    sat_rate = round((r / total * 100), 1) if total > 0 else 100.0
    return {
        "total_feedback": total,
        "rewards_count": r,
        "penalties_count": p,
        "satisfaction_rate": sat_rate,
        "recent_exemplars_count": len(data.get("exemplars", [])),
    }
