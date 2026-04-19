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
# Direct PrizePicks API fallback (no Apify, no token)
# ---------------------------------------------------------------------------

_PP_DIRECT_BASE = "https://api.prizepicks.com"
_PP_DIRECT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
}


def _pp_direct_get(url: str, timeout: int = 70, attempts: int = 2) -> dict | list:
    """
    GET a PrizePicks JSON endpoint. PrizePicks is behind Cloudflare and blocks
    most cloud egress IPs, so we route through ScraperAPI when SCRAPERAPI_KEY
    is set (basic mode — premium returns 500 on api.prizepicks.com). ScraperAPI
    is occasionally slow/flaky, so we retry a few times with backoff.
    """
    sa_key = os.getenv("SCRAPERAPI_KEY", "").strip()
    last_exc: Exception | None = None
    if sa_key:
        proxied = (
            "http://api.scraperapi.com/"
            f"?api_key={sa_key}"
            f"&url={urllib.parse.quote(url, safe='')}"
        )
        for i in range(attempts):
            try:
                req = urllib.request.Request(proxied)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[prizepicks] direct: ScraperAPI attempt {i+1}/{attempts} "
                    f"failed: {exc}"
                )
                time.sleep(2 + i * 3)
        raise last_exc if last_exc else RuntimeError("ScraperAPI exhausted")
    # No proxy — try directly (will likely 403 from cloud IPs)
    req = urllib.request.Request(url, headers=_PP_DIRECT_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


_CS2_LEAGUE_ID_CACHE: str | None = None


def _discover_cs2_league_id() -> str | None:
    """Look up CS2's league_id from /leagues (id changes occasionally)."""
    global _CS2_LEAGUE_ID_CACHE
    if _CS2_LEAGUE_ID_CACHE:
        return _CS2_LEAGUE_ID_CACHE
    try:
        data = _pp_direct_get(f"{_PP_DIRECT_BASE}/leagues")
        for lg in (data.get("data", []) if isinstance(data, dict) else []):
            attrs = lg.get("attributes", {}) or {}
            name = (attrs.get("name") or attrs.get("league") or "").upper()
            if name in ("CS2", "CSGO", "CS:GO", "COUNTER-STRIKE 2", "COUNTER STRIKE 2"):
                _CS2_LEAGUE_ID_CACHE = lg.get("id")
                logger.info(f"[prizepicks] direct: cached CS2 league_id={_CS2_LEAGUE_ID_CACHE}")
                return _CS2_LEAGUE_ID_CACHE
        logger.warning("[prizepicks] direct: CS2 league_id not found in /leagues")
        return None
    except Exception as exc:
        logger.warning(f"[prizepicks] direct: /leagues failed: {exc}")
        return None


def _fetch_direct_cs2() -> list[dict]:
    """
    Free fallback: hit api.prizepicks.com directly. No Apify, no token.
    Maps the JSON:API response into the same item shape Apify returns:
      league, stat, line, player_name, player_team, home_team, away_team, game_start
    """
    try:
        league_id = _discover_cs2_league_id()
        if not league_id:
            return []

        url = (
            f"{_PP_DIRECT_BASE}/projections"
            f"?league_id={league_id}&per_page=1000&single_stat=true"
        )
        logger.info(f"[prizepicks] direct: fetching CS2 projections (league_id={league_id})")
        data = _pp_direct_get(url, timeout=25)
        if not isinstance(data, dict):
            return []

        # Build player lookup from `included`
        players: dict[str, dict] = {}
        for inc in data.get("included", []) or []:
            t = inc.get("type") or ""
            if t in ("new_player", "player"):
                attrs = inc.get("attributes", {}) or {}
                pname = (
                    attrs.get("name")
                    or attrs.get("display_name")
                    or attrs.get("short_name")
                    or ""
                )
                players[inc.get("id")] = {
                    "name": pname,
                    "team": attrs.get("team") or attrs.get("team_name"),
                }

        items: list[dict] = []
        for proj in data.get("data", []) or []:
            attrs = proj.get("attributes", {}) or {}
            rels  = proj.get("relationships", {}) or {}
            pid = (
                (rels.get("new_player") or {}).get("data", {}) or {}
            ).get("id") or (
                (rels.get("player") or {}).get("data", {}) or {}
            ).get("id")
            pinfo = players.get(pid, {}) if pid else {}

            # Stat name on direct API: "MAPS 1-2 Kills" / "MAPS 1-2 Headshots" /
            # sometimes "Kills" / "Headshots" alone — keep as-is, _is_cs2_item
            # / get_player_line already handle both.
            stat_raw = (
                attrs.get("stat_display_name")
                or attrs.get("stat_type")
                or ""
            )

            items.append({
                "league": "CS2",
                "stat": stat_raw,
                "line": attrs.get("line_score"),
                "player_name": pinfo.get("name"),
                "player_team": pinfo.get("team"),
                "home_team": None,
                "away_team": attrs.get("description"),
                "game_start": attrs.get("start_time"),
            })

        cs2 = [it for it in items if _is_cs2_item(it)]
        logger.info(f"[prizepicks] direct: {len(cs2)}/{len(items)} CS2 items")
        return cs2

    except Exception as exc:
        logger.warning(f"[prizepicks] direct: fetch failed: {exc}")
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
        # 1) Apify primary (preferred when token works)
        items = _fetch_fresh_cs2_run()
        # 2) Apify last successful run
        if not items:
            logger.warning("[prizepicks] Fresh run returned nothing — trying last Apify run")
            items = _fetch_last_run_cs2() or []
        # 3) Direct PrizePicks API — free, no token, durable fallback
        if not items:
            logger.warning("[prizepicks] Apify paths exhausted — trying direct PrizePicks API")
            items = _fetch_direct_cs2()
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
