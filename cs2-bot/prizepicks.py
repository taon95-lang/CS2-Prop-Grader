"""
PrizePicks live line fetcher via Apify (zen-studio/prizepicks-player-props).

CONFIRMED field names from live CS2 data (2026-04-05):
  "league"      — "CS2"
  "stat"        — "MAPS 1-2 Kills" | "MAPS 1-2 Headshots"
  "line"        — numeric line e.g. 29
  "player_name" — e.g. "shane"
  "player_team" — team name (player_team_name is null for CS2)
  "home_team"   — team name (home_team_name is null for CS2)
  "away_team"   — team name (away_team_name is null for CS2)
  "game_start"  — ISO datetime

CRITICAL INPUT BUG FIXED:
  The Apify actor uses the key "leagues" (array), NOT "league" (string).
  Old broken call: {"league": "CSGO"}  → actor used default {"leagues": ["NBA"]}
  Correct call:    {"leagues": ["CS2"]} → returns 296 real CS2 props

Cache TTL = 5 minutes.
"""

import os
import json
import time
import logging
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "zen-studio~prizepicks-player-props"
APIFY_BASE     = "https://api.apify.com/v2"

_CACHE: dict    = {}
_CACHE_TS: float = 0.0
_CACHE_TTL: int  = 900   # 15 minutes — fresh run cached across back-to-back commands


# ---------------------------------------------------------------------------
# HTTP helpers
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


# ---------------------------------------------------------------------------
# CS2 item filter
# ---------------------------------------------------------------------------

def _is_cs2_item(item: dict) -> bool:
    """Return True if this item is a CS2 prop (league == 'CS2')."""
    league = (item.get("league") or "").strip()
    if league == "CS2":
        return True
    # Broader fallback in case future data uses variant spellings
    league_lc = league.lower()
    if league_lc in ("csgo", "cs:go", "counter-strike", "counter-strike 2"):
        return True
    # Stat-based fallback: "MAPS 1-2 Kills" / "MAPS 1-2 Headshots"
    stat = (item.get("stat") or "").lower()
    if ("kill" in stat or "headshot" in stat) and "maps" in stat:
        return True
    return False


def _dataset_items(dataset_id: str, limit: int = 5000) -> list[dict]:
    """Fetch up to `limit` items from an Apify dataset."""
    tok = _token()
    url = (
        f"{APIFY_BASE}/datasets/{dataset_id}/items"
        f"?token={tok}&clean=true&limit={limit}"
    )
    raw = _get(url, timeout=30)
    return raw if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Fetch strategies
# ---------------------------------------------------------------------------

def _fetch_last_run_cs2() -> list[dict] | None:
    """
    Read items from the most recent SUCCEEDED run that used CS2 input.
    Checks the run's INPUT key-value store to confirm it was a CS2 run.
    Returns None if no suitable run found.
    """
    try:
        tok = _token()
        runs_url = (
            f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/runs"
            f"?token={tok}&status=SUCCEEDED&limit=10&desc=1"
        )
        data = _get(runs_url, timeout=15)
        runs = (data.get("data", {}).get("items", [])
                if isinstance(data, dict) else [])

        for run in runs:
            kv_id = run.get("defaultKeyValueStoreId")
            if not kv_id:
                continue
            # Check if this run used CS2 input
            try:
                kv_url = f"{APIFY_BASE}/key-value-stores/{kv_id}/records/INPUT?token={tok}"
                inp = _get(kv_url, timeout=8)
                leagues_input = inp.get("leagues", []) if isinstance(inp, dict) else []
                if "CS2" not in leagues_input:
                    continue
            except Exception:
                continue

            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                continue

            items = _dataset_items(dataset_id, limit=5000)
            cs2 = [it for it in items if _is_cs2_item(it)]
            logger.info(f"[prizepicks] Last CS2 run: {len(cs2)}/{len(items)} CS2 items (run {run.get('id','?')[:8]})")
            if cs2:
                return cs2

        logger.info("[prizepicks] No recent CS2 run found — starting fresh")
        return None

    except Exception as exc:
        logger.warning(f"[prizepicks] _fetch_last_run_cs2 failed: {exc}")
        return None


def _fetch_fresh_cs2_run() -> list[dict]:
    """
    Start a fresh sync actor run with {"leagues": ["CS2"]} and return items.
    """
    tok = _token()
    url = (
        f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
        f"?token={tok}&timeout=90&memory=512"
    )
    try:
        logger.info("[prizepicks] Starting fresh CS2 run with leagues=['CS2']…")
        raw = _post(url, {"leagues": ["CS2"]}, timeout=110)
        items = raw if isinstance(raw, list) else []

        leagues = {}
        stats = {}
        for it in items:
            lg = it.get("league", "?")
            st = it.get("stat", "?")
            leagues[lg] = leagues.get(lg, 0) + 1
            stats[st]   = stats.get(st, 0) + 1
        logger.info(f"[prizepicks] Fresh CS2 run: {len(items)} items | leagues={leagues} | stats={stats}")

        cs2 = [it for it in items if _is_cs2_item(it)]
        logger.info(f"[prizepicks] CS2 filter kept {len(cs2)}/{len(items)}")
        return cs2

    except Exception as exc:
        logger.warning(f"[prizepicks] Fresh CS2 run failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cs2_lines(player_name: str | None = None) -> list[dict]:
    """
    Return CS2 props. Cached for _CACHE_TTL seconds.
    Optionally filters to a specific player name (case-insensitive substring).
    """
    global _CACHE, _CACHE_TS

    now = time.time()
    if now - _CACHE_TS < _CACHE_TTL and _CACHE:
        items = _CACHE.get("items", [])
        logger.info(f"[prizepicks] Cache hit: {len(items)} CS2 items")
    else:
        # Always run fresh — guarantees we see current PP slate, not stale dataset
        items = _fetch_fresh_cs2_run()
        if not items:
            # Fallback: last stored run if fresh scrape fails
            logger.warning("[prizepicks] Fresh run returned nothing — falling back to last run")
            items = _fetch_last_run_cs2() or []
        _CACHE    = {"items": items}
        _CACHE_TS = now

    # Drop props where the game has already started or is in the past
    now_utc = datetime.now(timezone.utc)
    live_items = []
    dropped = 0
    for it in items:
        gs = it.get("game_start") or ""
        if gs:
            try:
                # ISO format e.g. "2026-04-05T20:00:00Z" or "2026-04-05T20:00:00+00:00"
                gs_clean = gs.replace("Z", "+00:00")
                game_dt  = datetime.fromisoformat(gs_clean)
                if game_dt.tzinfo is None:
                    game_dt = game_dt.replace(tzinfo=timezone.utc)
                if game_dt <= now_utc:
                    dropped += 1
                    continue
            except Exception:
                pass  # can't parse → keep it
        live_items.append(it)
    if dropped:
        logger.info(f"[prizepicks] Dropped {dropped} props with game_start in the past")
    items = live_items

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
    Uses 'stat' field which contains 'MAPS 1-2 Kills' / 'MAPS 1-2 Headshots'.
    """
    items = get_cs2_lines(player_name)
    if not items:
        return None
    kw = "headshot" if stat_type.upper() == "HS" else "kill"
    for item in items:
        stat_raw = (item.get("stat") or "").lower()
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
