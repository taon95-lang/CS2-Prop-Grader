"""
PrizePicks live line fetcher via Apify (zen-studio/prizepicks-player-props).

Public interface:
    get_cs2_lines(player_name=None)  → list[dict]
    get_player_line(player_name, stat_type="Kills")  → dict | None

Each item in the list has at minimum:
    player_name, stat_type, line_score, projection_type_name,
    league, game_start, home_team, away_team, player_team, player_position
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "zen-studio~prizepicks-player-props"
APIFY_BASE     = "https://api.apify.com/v2"

# Cache so we don't hammer Apify every command
_CACHE: dict = {}
_CACHE_TS: float = 0.0
_CACHE_TTL: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _apify_token() -> str | None:
    return os.getenv("APIFY_TOKEN")


def _run_actor_sync(input_payload: dict, timeout: int = 120) -> list[dict]:
    """
    Synchronous Apify actor run (blocks until complete, returns dataset items).
    Uses the /run-sync-get-dataset-items endpoint which is designed for this.
    """
    token = _apify_token()
    if not token:
        raise RuntimeError("APIFY_TOKEN env var not set")

    url = (
        f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
        f"?token={token}&timeout={timeout}&memory=512"
    )
    body = json.dumps(input_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cs2_lines(player_name: str | None = None) -> list[dict]:
    """
    Fetch all CS2 / CSGO props currently on PrizePicks.
    Results are cached for _CACHE_TTL seconds.
    If player_name is given, filters to that player only (case-insensitive).
    """
    global _CACHE, _CACHE_TS

    now = time.time()
    if now - _CACHE_TS < _CACHE_TTL and _CACHE:
        items = _CACHE.get("items", [])
    else:
        logger.info("[prizepicks] Fetching CS2 lines from Apify…")
        try:
            payload: dict = {"league": "CSGO"}
            raw = _run_actor_sync(payload, timeout=90)
            # Normalize: the actor returns a list of items
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict) and "items" in raw:
                items = raw["items"]
            else:
                items = []
            _CACHE = {"items": items}
            _CACHE_TS = now
            logger.info(f"[prizepicks] Got {len(items)} CS2 props")
        except Exception as exc:
            logger.error(f"[prizepicks] Apify fetch failed: {exc}")
            return []

    if player_name:
        needle = player_name.lower().strip()
        items = [
            it for it in items
            if needle in (it.get("player_name") or "").lower()
        ]

    return items


def get_player_line(player_name: str, stat_type: str = "Kills") -> dict | None:
    """
    Return the best-matching PrizePicks line for this player + stat type.

    stat_type can be "Kills" or "HS".  We match on:
      - "Kills"  → stat_display_name contains "kill" (case-insensitive)
      - "HS"     → stat_display_name contains "headshot"
    """
    items = get_cs2_lines(player_name)
    if not items:
        return None

    kw = "headshot" if stat_type.upper() == "HS" else "kill"
    for item in items:
        stat_raw = (item.get("stat_display_name") or item.get("stat_type") or "").lower()
        if kw in stat_raw:
            return item

    # Fallback: return first item regardless of stat type so caller can inspect
    return items[0] if items else None


def get_all_cs2_props() -> list[dict]:
    """Return all CS2 props (no player filter), refreshing cache if stale."""
    return get_cs2_lines()


def invalidate_cache() -> None:
    """Force next call to re-fetch from Apify."""
    global _CACHE, _CACHE_TS
    _CACHE = {}
    _CACHE_TS = 0.0
