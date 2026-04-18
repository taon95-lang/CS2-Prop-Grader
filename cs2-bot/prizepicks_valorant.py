"""
PrizePicks live Valorant line fetcher.

Hits PrizePicks' public projections API directly via ScraperAPI (Cloudflare-backed
endpoint blocks naked requests). Returns normalised dicts so !vteam / !vgrade can
auto-pull real sportsbook lines.

Cache TTL = 10 minutes — board doesn't move often during a slate.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

VAL_LEAGUE_ID = 159
PP_URL = (
    f"https://api.prizepicks.com/projections"
    f"?league_id={VAL_LEAGUE_ID}&per_page=500&single_stat=true"
)
SA_URL = "http://api.scraperapi.com/"

_CACHE: list[dict] = []
_CACHE_TS: float = 0.0
_CACHE_TTL: int = 600  # 10 minutes


def _scraperapi_key() -> str:
    k = os.getenv("SCRAPERAPI_KEY", "")
    if not k:
        raise RuntimeError("SCRAPERAPI_KEY env var not set")
    return k


def _fetch_pp_valorant() -> list[dict]:
    """
    Pull the live Valorant board from PrizePicks via ScraperAPI.
    Returns a list of normalised prop dicts:
        {
            player_name, player_team, league, stat_type, line,
            description, opponent, start_time, prop_id
        }
    """
    sa = (
        f"{SA_URL}?api_key={_scraperapi_key()}"
        f"&country_code=us&keep_headers=true"
        f"&url={urllib.parse.quote(PP_URL)}"
    )
    req = urllib.request.Request(
        sa,
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            payload = json.loads(r.read())
    except Exception as exc:
        logger.warning(f"[pp-val] fetch failed: {exc}")
        return []

    items = payload.get("data") or []
    incl  = payload.get("included") or []
    players = {
        x["id"]: x.get("attributes", {})
        for x in incl
        if x.get("type") in ("new_player", "player")
    }

    out: list[dict] = []
    for it in items:
        a = it.get("attributes") or {}
        rel = it.get("relationships") or {}
        pid = (rel.get("new_player") or {}).get("data", {}).get("id")
        p = players.get(pid, {}) or {}
        line = a.get("line_score")
        try:
            line = float(line) if line is not None else None
        except (TypeError, ValueError):
            line = None
        out.append({
            "prop_id":     it.get("id"),
            "player_name": p.get("display_name") or p.get("name") or "?",
            "player_team": p.get("team_name") or p.get("team") or "",
            "league":      "VAL",
            "stat_type":   a.get("stat_type") or "?",
            "line":        line,
            "description": a.get("description") or "",
            "start_time":  a.get("start_time") or "",
        })
    logger.info(f"[pp-val] fetched {len(out)} Valorant props from PrizePicks")
    return out


def get_valorant_lines(player_name: str | None = None) -> list[dict]:
    """
    Return live Valorant PrizePicks props. Cached for _CACHE_TTL seconds.
    Optionally filter by player name (case-insensitive substring).
    Drops props whose game has already started.
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if now - _CACHE_TS < _CACHE_TTL and _CACHE:
        items = _CACHE
        logger.info(f"[pp-val] cache hit: {len(items)} items")
    else:
        items = _fetch_pp_valorant()
        if items:
            _CACHE = items
            _CACHE_TS = now

    # Drop props past their start time
    now_utc = datetime.now(timezone.utc)
    live: list[dict] = []
    for it in items:
        gs = it.get("start_time") or ""
        if gs:
            try:
                gs_clean = gs.replace("Z", "+00:00")
                dt = datetime.fromisoformat(gs_clean)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt <= now_utc:
                    continue
            except Exception:
                pass
        live.append(it)

    if player_name:
        nm = player_name.lower().strip()
        live = [it for it in live if nm in (it.get("player_name") or "").lower()]
    return live


def get_valorant_lines_for_team(team_name: str) -> list[dict]:
    """Return all live VAL props whose player_team matches `team_name` (fuzzy)."""
    nm = (team_name or "").lower().replace(" ", "")
    if not nm:
        return []
    out = []
    for it in get_valorant_lines():
        pt = (it.get("player_team") or "").lower().replace(" ", "")
        if not pt:
            continue
        if nm in pt or pt in nm:
            out.append(it)
    return out


def get_valorant_player_line(
    player_name: str,
    stat_keyword: str = "MAPS 1-2 Kills",
) -> Optional[dict]:
    """Return the first prop matching player + stat keyword, or None."""
    nm = (player_name or "").lower().strip()
    sk = stat_keyword.lower()
    for it in get_valorant_lines():
        if nm in (it.get("player_name") or "").lower() and sk in (it.get("stat_type") or "").lower():
            return it
    return None


def invalidate_cache() -> None:
    global _CACHE, _CACHE_TS
    _CACHE = []
    _CACHE_TS = 0.0
