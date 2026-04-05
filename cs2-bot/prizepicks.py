"""
PrizePicks live line fetcher via Apify (zen-studio/prizepicks-player-props).

Strategy:
  1. PRIMARY — read the most recent SUCCEEDED run's dataset (near-instant, no charge).
  2. FALLBACK — start a fresh sync run if no recent run exists or data is older than TTL.

Cache TTL = 5 minutes so repeated commands don't hit Apify at all.

Public interface:
    get_cs2_lines(player_name=None)  → list[dict]
    get_player_line(player_name, stat_type="Kills")  → dict | None
    get_all_cs2_props()              → list[dict]
    invalidate_cache()
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


def _post(url: str, payload: dict, timeout: int = 30) -> dict | list:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Fetch strategies
# ---------------------------------------------------------------------------

def _fetch_last_run_items() -> list[dict] | None:
    """
    Read dataset items from the most recent SUCCEEDED actor run.
    This is instant — no new run is started and no Apify credit is consumed.
    Returns None if no successful run exists.
    """
    try:
        tok = _token()
        # Get the most recent succeeded run
        runs_url = (
            f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/runs"
            f"?token={tok}&status=SUCCEEDED&limit=1&desc=1"
        )
        data = _get(runs_url, timeout=15)
        runs = data.get("data", {}).get("items", []) if isinstance(data, dict) else []
        if not runs:
            logger.info("[prizepicks] No recent SUCCEEDED run found — will start fresh run")
            return None

        run = runs[0]
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return None

        # Fetch items from that dataset
        items_url = f"{APIFY_BASE}/datasets/{dataset_id}/items?token={tok}&clean=true&limit=500"
        raw = _get(items_url, timeout=20)
        items = raw if isinstance(raw, list) else raw.get("data", {}).get("items", [])
        logger.info(f"[prizepicks] Loaded {len(items)} items from last run (dataset {dataset_id})")
        return items

    except Exception as exc:
        logger.warning(f"[prizepicks] _fetch_last_run_items failed: {exc}")
        return None


def _fetch_fresh_run_items() -> list[dict]:
    """
    Start a new sync actor run and wait for its output.
    Used as fallback when no recent run exists.
    """
    tok = _token()
    url = (
        f"{APIFY_BASE}/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
        f"?token={tok}&timeout=90&memory=512"
    )
    logger.info("[prizepicks] Starting fresh Apify actor run…")
    raw = _post(url, {}, timeout=110)
    items = raw if isinstance(raw, list) else []
    logger.info(f"[prizepicks] Fresh run returned {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cs2_lines(player_name: str | None = None) -> list[dict]:
    """
    Return CS2/CSGO props from PrizePicks. Results cached for _CACHE_TTL seconds.
    Optionally filter to a specific player name.
    """
    global _CACHE, _CACHE_TS

    now = time.time()
    if now - _CACHE_TS < _CACHE_TTL and _CACHE:
        items = _CACHE.get("items", [])
        logger.info(f"[prizepicks] Returning {len(items)} cached items")
    else:
        # Try last run first (instant), fall back to fresh run
        items = _fetch_last_run_items()
        if items is None:
            items = _fetch_fresh_run_items()
        _CACHE    = {"items": items}
        _CACHE_TS = now

    # Filter by league if needed (actor may return mixed league data)
    cs2_items = [
        it for it in items
        if (it.get("league") or "").upper() in ("CS2", "CSGO", "CS:GO", "COUNTER-STRIKE")
        or "kill" in (it.get("stat_display_name") or it.get("stat_type") or "").lower()
    ]
    # If no CS2-specific items, return all (actor may already filter by league)
    if not cs2_items and items:
        cs2_items = items

    if player_name:
        needle = player_name.lower().strip()
        cs2_items = [
            it for it in cs2_items
            if needle in (it.get("player_name") or "").lower()
        ]

    return cs2_items


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
    """Return all CS2 props (no player filter), refreshing cache if stale."""
    return get_cs2_lines()


def invalidate_cache() -> None:
    """Force next call to re-fetch from Apify."""
    global _CACHE, _CACHE_TS
    _CACHE    = {}
    _CACHE_TS = 0.0
    logger.info("[prizepicks] Cache invalidated")
