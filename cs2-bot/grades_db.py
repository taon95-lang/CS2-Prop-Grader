"""
grades_db.py — persistent grade history for Elite CS2 Prop Grader Bot.

Stores every grade delivered by the bot (from !grade or !pp) to a JSON file.
Results are logged via !result and !results commands in bot.py.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "grades_history.json")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    if not os.path.exists(_DB_PATH):
        return {"entries": []}
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[grades_db] Failed to load {_DB_PATH}: {e}")
        return {"entries": []}


def _save(db: dict) -> None:
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[grades_db] Failed to save {_DB_PATH}: {e}")


def _norm(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_grade(
    player: str,
    line: float,
    stat: str,
    sim_result: dict,
    opponent: str | None = None,
    baseline_match_id: str | None = None,
) -> str:
    """
    Persist a completed grade.  Returns the entry id.
    sim_result is the dict returned by _analyze_player / run_simulation.
    """
    db = _load()

    now = datetime.now(timezone.utc)
    entry = {
        "id":           str(uuid.uuid4()),
        "ts":           now.timestamp(),
        "date":         now.strftime("%Y-%m-%d"),
        "player":       player.lower(),
        "display":      sim_result.get("player_name") or player,
        "line":         float(line),
        "stat":         stat,
        "opponent":     (opponent or sim_result.get("opponent") or "").strip(),
        "over_pct":     sim_result.get("over_prob"),
        "under_pct":    sim_result.get("under_prob"),
        "grade":        sim_result.get("grade"),
        "grade_label":  sim_result.get("grade_label"),
        "recommendation": sim_result.get("decision") or sim_result.get("recommendation"),
        "sim_median":   sim_result.get("sim_median"),
        "hist_median":  sim_result.get("hist_median"),
        "actual":       None,
        "outcome":      None,   # "over" | "under" | "push" | null
        "baseline_match_id": baseline_match_id,
    }
    db["entries"].append(entry)
    _save(db)
    logger.info(
        f"[grades_db] Saved grade: {entry['display']} {line} {stat} "
        f"(rec={entry['recommendation']}, id={entry['id'][:8]})"
    )
    return entry["id"]


def record_result(player: str, actual: float, entry_id: str | None = None) -> dict | None:
    """
    Record the actual stat total for a player's most recent unresolved grade.
    If entry_id is given, update that specific entry.
    Returns the updated entry dict, or None if not found.
    """
    db = _load()
    entries = db["entries"]

    target = None
    if entry_id:
        target = next((e for e in entries if e["id"].startswith(entry_id)), None)
    else:
        # Find most recent unresolved grade for this player
        norm = _norm(player)
        candidates = [
            e for e in entries
            if _norm(e["player"]) == norm and e["outcome"] is None
        ]
        if candidates:
            target = max(candidates, key=lambda e: e["ts"])

    if not target:
        return None

    line = target["line"]
    target["actual"] = actual
    if actual > line:
        target["outcome"] = "over"
    elif actual < line:
        target["outcome"] = "under"
    else:
        target["outcome"] = "push"

    _save(db)
    logger.info(
        f"[grades_db] Result recorded: {target['display']} {line} {target['stat']} "
        f"actual={actual} → {target['outcome']}"
    )
    return target


def get_entries_for_date(date_str: str) -> list[dict]:
    """Return all entries for a given YYYY-MM-DD date."""
    db = _load()
    return [e for e in db["entries"] if e.get("date") == date_str]


def get_pending_entries() -> list[dict]:
    """Return all entries with no outcome yet."""
    db = _load()
    return [e for e in db["entries"] if e.get("outcome") is None]


def get_recent_entries(days: int = 7) -> list[dict]:
    """Return entries from the last N days."""
    db = _load()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    return [e for e in db["entries"] if e.get("ts", 0) >= cutoff]


def date_label(date_str: str) -> str:
    """Convert YYYY-MM-DD to a friendly label like 'Today', 'Yesterday', 'Apr 5'."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if date_str == today:
        return "Today"
    if date_str == yesterday:
        return "Yesterday"
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%b %-d")
    except Exception:
        return date_str
