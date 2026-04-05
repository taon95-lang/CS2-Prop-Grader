"""
PrizePicks live line fetcher via Apify (zen-studio/prizepicks-player-props).

Strategy:
  1. PRIMARY — read the most recent SUCCEEDED run's dataset (near-instant).
  2. FALLBACK — start a fresh sync run filtered to CS2 if no recent data or cache stale.

CS2 league names seen in PrizePicks data: "CSGO", "CS2", "CS:GO", "Counter-Strike"
(We accept any of these and also filter by stat type as a safety net.)

Cache TTL = 5 minutes.
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

# All known CS2 league name variants (lower-cased for comparison)
CS2_LEAGUE_NAMES = {"cs2", "csgo", "cs:go", "counter-strike", "counter strike"}
# Stats that only exist in CS2 (not NBA/NFL/etc.)
CS2_STAT_KEYWORDS = {"kill", "headshot", "death", "assist", "adr", "kast", "rating"}

_CACHE: dict   = {}
_CACHE_TS: float = 0.0
_CACHE_TTL: int  = 300   # 5 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token() -> str:
    t = os.getenv("APIFY_TOKEN", "")
    if not t:
        raise RuntimeError("APIFY_TOKEN env var not set")
    return t


def _get(url: str, timeout: int = 30) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: int = 110) -> dict | list:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_cs2_item(item: dict) -> bool:
    """Return True only if the item is a CS2/CSGO prop."""
    league = (item.get("league") or item.get("league_name") or "").lower().strip()
    if league in CS2_LEAGUE_NAMES:
        return True
    # Secondary: stat name contains a CS2-specific keyword
    stat = (item.get("stat_display_name") or item.get("stat_type") or "").lower()
    if any(kw in stat for kw in CS2_STAT_KEYWORDS):
        # But guard against false positives: make sure there's NO obvious non-CS2 league
        non_cs2 = {"nba", "nfl", "mlb", "nhl", "nfl", "soccer", "tennis",
                   "mma", "golf", "esports", "valorant", "lol", "dota"}
        if league not in non_cs2:
            return True
    return False


# ---------------------------------------------------------------------------
# Fetch strategies
# ---------------------------------------------------------------------------

def _fetch_last_run_items() -> list[dict] | None:
    """
    Read items from the most recent SUCCEEDED actor run.
    Returns None if no run exists or the result contains no CS2 items.
    """
    try:
        tok = _token()
        runs_url = (
            f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/runs"
            f"?token={tok}&status=SUCCEEDED&limit=1&desc=1"
        )
        data = _get(runs_url, timeout=15)
        runs = (data.get("data", {}).get("items", [])
                if isinstance(data, dict) else [])
        if not runs:
            logger.info("[prizepicks] No recent SUCCEEDED run — will start fresh")
            return None

        dataset_id = runs[0].get("defaultDatasetId")
        if not dataset_id:
            return None

        items_url = (
            f"{APIFY_BASE}/datasets/{dataset_id}/items"
            f"?token={tok}&clean=true&limit=500"
        )
        raw = _get(items_url, timeout=20)
        items = raw if isinstance(raw, list) else []

        # Log what leagues came back so we can tune the filter
        leagues_seen = {(it.get("league") or "?") for it in items}
        logger.info(f"[prizepicks] Last run: {len(items)} items, leagues: {leagues_seen}")

        cs2 = [it for it in items if _is_cs2_item(it)]
        logger.info(f"[prizepicks] CS2 filter: {len(cs2)}/{len(items)} items kept")

        # If we got items but NONE are CS2, the run had no CS2 data — trigger fresh run
        if items and not cs2:
            logger.info("[prizepicks] Last run had no CS2 props — starting fresh run")
            return None

        return cs2

    except Exception as exc:
        logger.warning(f"[prizepicks] _fetch_last_run_items failed: {exc}")
        return None


def _fetch_fresh_run_items() -> list[dict]:
    """
    Start a fresh actor run filtered to CSGO/CS2 and wait for results.
    Tries "CSGO" first, then "CS2" if that returns nothing.
    """
    tok = _token()
    url = (
        f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
        f"?token={tok}&timeout=90&memory=512"
    )
    for league_input in ("CSGO", "CS2"):
        try:
            logger.info(f"[prizepicks] Starting fresh run with league={league_input}…")
            raw = _post(url, {"league": league_input}, timeout=110)
            items = raw if isinstance(raw, list) else []
            logger.info(f"[prizepicks] Fresh run ({league_input}): {len(items)} items")
            cs2 = [it for it in items if _is_cs2_item(it)] if items else items
            if cs2:
                return cs2
        except Exception as exc:
            logger.warning(f"[prizepicks] Fresh run ({league_input}) failed: {exc}")

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cs2_lines(player_name: str | None = None) -> list[dict]:
    """
    Return CS2/CSGO props. Cached for _CACHE_TTL seconds.
    Optionally filters to a specific player name (case-insensitive).
    """
    global _CACHE, _CACHE_TS

    now = time.time()
    if now - _CACHE_TS < _CACHE_TTL and _CACHE:
        items = _CACHE.get("items", [])
        logger.info(f"[prizepicks] Cache hit: {len(items)} CS2 items")
    else:
        items = _fetch_last_run_items()
        if items is None:
            items = _fetch_fresh_run_items()
        _CACHE    = {"items": items or []}
        _CACHE_TS = now

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
    stat_type: "Kills" or "HS"
    """
    items = get_cs2_lines(player_name)
    if not items:
        return None
    kw = "headshot" if stat_type.upper() == "HS" else "kill"
    for item in items:
        stat_raw = (item.get("stat_display_name") or item.get("stat_type") or "").lower()
        if kw in stat_raw:
            return item
    return items[0] if items else None


def get_all_cs2_props() -> list[dict]:
    return get_cs2_lines()


def invalidate_cache() -> None:
    global _CACHE, _CACHE_TS
    _CACHE    = {}
    _CACHE_TS = 0.0
    logger.info("[prizepicks] Cache invalidated")
