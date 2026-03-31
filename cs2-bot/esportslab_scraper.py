"""
The Esports Lab (theesportslab.com) data scraper — a PrizePicks product.

Confirmed accessible endpoint (no auth, Cloudflare cookie sufficient):
  GET https://api.theesportslab.com/players/list/default_search

This endpoint returns ~14 featured players across CS2 / Dota2 / LoL with
per-player rolling stats covering exactly the last 10 maps:

  stats.kills      — total kills across the last 10 maps
  stats.headshots  — total headshots across the last 10 maps
  stats.kpm        — kills per map (10-map rolling average)
  stats.hspm       — headshots per map (10-map rolling average)
  stats.dpm        — deaths per map
  stats.adm        — assists per map

From these we derive:
  hs_pct = headshots / kills           (last-10-maps headshot kill rate)

Limitations:
  - Only features ~6 CS2 players at any given time (rotates with the website)
  - All parameterized API endpoints are Cloudflare-WAF blocked from server
  - No arbitrary player search — only featured players are accessible

When a player IS found here, the data is very high quality: recent (last 10
maps), per-map granularity, and directly aligned with PrizePicks' own prop lines.
"""

import time
import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ESL_API = "https://api.theesportslab.com"
ESL_SITE = "https://theesportslab.com"
ESL_DEFAULT_SEARCH = f"{ESL_API}/players/list/default_search"

_CACHE_TTL = 30 * 60   # 30-minute TTL — featured players rotate infrequently
_cache_ts: float = 0.0
_cache_data: list[dict] = []

try:
    from curl_cffi import requests as _cr
    _SESSION: Optional[_cr.Session] = None
    _CFFI_OK = True
except ImportError:
    _CFFI_OK = False
    _SESSION = None
    logger.warning("[esl] curl_cffi not available — Esports Lab fallback disabled")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _session() -> Optional["_cr.Session"]:
    global _SESSION
    if not _CFFI_OK:
        return None
    if _SESSION is None:
        _SESSION = _cr.Session(impersonate="chrome120")
    return _SESSION


_HEADERS = {
    "Origin": ESL_SITE,
    "Referer": f"{ESL_SITE}/esports/cs2/stats",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}


# ---------------------------------------------------------------------------
# Data fetch with caching
# ---------------------------------------------------------------------------

def _fetch_featured_players(force: bool = False) -> list[dict]:
    """
    Fetch and cache the default_search player list from The Esports Lab.

    The Cloudflare __cf_bm cookie is set automatically by the first request
    to the API domain and is carried in the session for subsequent calls.

    Returns a list of player dicts (may be empty on failure).
    """
    global _cache_ts, _cache_data

    if not force and _cache_data and (time.time() - _cache_ts) < _CACHE_TTL:
        logger.debug("[esl] Returning cached default_search data")
        return _cache_data

    sess = _session()
    if sess is None:
        return []

    try:
        logger.info(f"[esl] Fetching {ESL_DEFAULT_SEARCH}")
        r = sess.get(
            ESL_DEFAULT_SEARCH,
            headers=_HEADERS,
            timeout=10,
        )
        if r.status_code == 200:
            ct = r.headers.get("content-type", "")
            if "json" in ct and len(r.text) > 10:
                data = json.loads(r.text)
                if isinstance(data, list):
                    _cache_data = data
                    _cache_ts = time.time()
                    cs2_count = sum(
                        1 for p in data
                        if p.get("esport", {}).get("alias") == "cs2"
                    )
                    logger.info(
                        f"[esl] Cached {len(data)} featured players "
                        f"({cs2_count} CS2)"
                    )
                    return _cache_data
        logger.warning(
            f"[esl] default_search returned {r.status_code} — "
            "returning stale cache if available"
        )
        return _cache_data   # stale cache is better than nothing
    except Exception as e:
        logger.warning(f"[esl] Fetch failed: {type(e).__name__}: {e}")
        return _cache_data


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def search_player_esl(player_name: str) -> Optional[dict]:
    """
    Look for a player in The Esports Lab's featured CS2 player list.

    Matching is by normalised nickname (lowercase, alphanumeric only).
    Returns the raw player dict from the API, or None if not found.
    """
    players = _fetch_featured_players()
    if not players:
        return None

    target = _normalise(player_name)
    cs2_players = [
        p for p in players
        if p.get("esport", {}).get("alias") == "cs2"
    ]

    for p in cs2_players:
        nick = _normalise(p.get("nickname", ""))
        if nick == target:
            logger.info(
                f"[esl] Exact match: {p['nickname']!r} (id={p['id']}) "
                f"team={p.get('team',{}).get('name','?')}"
            )
            return p

    # Partial match fallback (player's query may be partial, e.g. "zywoo" → "ZywOo")
    for p in cs2_players:
        nick = _normalise(p.get("nickname", ""))
        if target in nick or nick in target:
            logger.info(
                f"[esl] Partial match: {p['nickname']!r} (id={p['id']}) "
                f"for query {player_name!r}"
            )
            return p

    logger.info(
        f"[esl] {player_name!r} not in featured CS2 list "
        f"(featured: {[p['nickname'] for p in cs2_players]})"
    )
    return None


# ---------------------------------------------------------------------------
# Stats extraction
# ---------------------------------------------------------------------------

def get_esl_stats(player_name: str) -> Optional[dict]:
    """
    Retrieve last-10-maps stats for a player from The Esports Lab.

    Returns a dict or None if the player is not in the featured list:

    {
      'nickname':       str,           # canonical nickname from ESL
      'team':           str,
      'kpm':            float,         # kills per map (10-map avg)
      'hspm':           float,         # headshots per map (10-map avg)
      'dpm':            float,         # deaths per map (10-map avg)
      'kills_total':    float,         # raw kills over last 10 maps
      'headshots_total':float,         # raw headshots over last 10 maps
      'deaths_total':   float,
      'hs_pct':         float,         # headshots / kills  [0, 1]
      'n_maps':         int,           # always 10 (window size)
      'source':         'esportslab.com (last 10 maps)',
    }
    """
    raw = search_player_esl(player_name)
    if raw is None:
        return None

    s = raw.get("stats", {})
    kills = s.get("kills") or 0.0
    hs    = s.get("headshots") or 0.0
    kpm   = s.get("kpm") or 0.0
    hspm  = s.get("hspm") or 0.0
    dpm   = s.get("dpm") or 0.0

    if kills <= 0:
        logger.warning(f"[esl] Zero kills for {raw.get('nickname')!r} — skipping")
        return None

    hs_pct = round(hs / kills, 4)

    result = {
        "nickname":        raw.get("nickname", player_name),
        "team":            raw.get("team", {}).get("name", ""),
        "kpm":             kpm,
        "hspm":            hspm,
        "dpm":             dpm,
        "kills_total":     kills,
        "headshots_total": hs,
        "deaths_total":    s.get("deaths") or 0.0,
        "hs_pct":          hs_pct,
        "n_maps":          10,
        "source":          "theesportslab.com (last 10 maps)",
    }
    logger.info(
        f"[esl] Stats for {result['nickname']!r}: "
        f"kpm={kpm} hspm={hspm} HS%={round(hs_pct*100,1)}%"
    )
    return result
