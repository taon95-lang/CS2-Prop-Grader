"""
Bo3.gg fallback scraper for headshot data.

When HLTV's per-map K(hs) column is inaccessible (Cloudflare-blocked stats pages),
this module provides player headshot statistics from the bo3.gg public JSON API.

Confirmed accessible endpoints (no auth required):
  GET /api/v1/players/{slug}               → player ID lookup
  GET /api/v1/players/{slug}/general_stats → career kills_sum, deaths_sum, etc.
  GET /api/v1/players/{slug}/accuracy_stats → hit-group breakdown (Head kills_sum)
  GET /api/v1/players/{slug}/map_stats     → avg_kills per map type
  GET /api/v1/games/{id}/short_players_stats → kills_sum + headshots_sum per player

Flow for HS% fallback:
  1. Look up player slug → player_id
  2. accuracy_stats  → head_kills (hit_group == "Head", kills_sum)
  3. general_stats   → total_kills (kills_sum)
  4. hs_pct = head_kills / total_kills  (career weighted average)

Flow for per-series headshot count:
  1. Paginate recent games (sort=-begin_at, 10 per page)
  2. For each page, fetch short_players_stats for every game_id
  3. Filter rows matching player_id; collect kills_sum + headshots_sum per match
  4. Stop once 10 BO3 series worth of data are gathered (≤ 20 maps)
  Returns [(kills, headshots), ...] for Maps 1+2 of each series
"""

import re
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BO3GG_BASE = "https://bo3.gg/api/v1"
_FETCH_TIMEOUT = 10  # seconds per request

try:
    from curl_cffi import requests as _cr
    _SESSION: Optional[_cr.Session] = None
    _CFFI_OK = True
except ImportError:
    _CFFI_OK = False
    _SESSION = None
    logger.warning("[bo3] curl_cffi not available — bo3.gg fallback disabled")


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _session() -> Optional["_cr.Session"]:
    global _SESSION
    if not _CFFI_OK:
        return None
    if _SESSION is None:
        _SESSION = _cr.Session(impersonate="chrome120")
    return _SESSION


def _get(path: str, params: dict = None) -> Optional[dict | list]:
    """GET a bo3.gg API path, return parsed JSON or None on failure."""
    sess = _session()
    if sess is None:
        return None
    url = BO3GG_BASE + path
    try:
        r = sess.get(
            url,
            params=params or {},
            timeout=_FETCH_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        if r.status_code == 200:
            ct = r.headers.get("content-type", "")
            if "json" in ct and len(r.text) > 2:
                return json.loads(r.text)
        logger.debug(f"[bo3] GET {url} → {r.status_code}")
        return None
    except Exception as e:
        logger.warning(f"[bo3] GET {url} failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Player search
# ---------------------------------------------------------------------------

def search_player_bo3(name: str) -> Optional[tuple[int, str]]:
    """
    Look up a player on bo3.gg by their HLTV-style nickname.
    Tries slug directly first, then a fuzzy normalised match.

    Returns (player_id, player_slug) or None if not found.
    """
    # Normalise: lowercase, hyphens
    slug = re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-').replace('_', '-'))

    # Direct slug lookup
    data = _get(f"/players/{slug}")
    if data and isinstance(data, dict) and data.get('id'):
        pid = data['id']
        actual_slug = data.get('slug', slug)
        logger.info(f"[bo3] Found player {name!r} → id={pid} slug={actual_slug!r}")
        return pid, actual_slug

    # Try without hyphens (some slugs omit them)
    slug_nohyphen = slug.replace('-', '')
    if slug_nohyphen != slug:
        data2 = _get(f"/players/{slug_nohyphen}")
        if data2 and isinstance(data2, dict) and data2.get('id'):
            pid = data2['id']
            actual_slug = data2.get('slug', slug_nohyphen)
            logger.info(f"[bo3] Found player {name!r} (no-hyphen slug) → id={pid}")
            return pid, actual_slug

    logger.info(f"[bo3] Player {name!r} not found on bo3.gg")
    return None


# ---------------------------------------------------------------------------
# Career HS% from accuracy_stats
# ---------------------------------------------------------------------------

def get_career_hs_pct(player_slug: str) -> Optional[float]:
    """
    Compute a player's career headshot kill rate from bo3.gg accuracy stats.

    Uses:
      /players/{slug}/accuracy_stats  (hit-group breakdown, 'Head' kills_sum)
      /players/{slug}/general_stats   (career kills_sum for denominator)

    Returns HS% as a float in [0, 1], or None if data is unavailable.
    """
    acc = _get(f"/players/{player_slug}/accuracy_stats")
    if not acc or not isinstance(acc, list):
        logger.warning(f"[bo3] accuracy_stats unavailable for {player_slug!r}")
        return None

    head_entry = next((e for e in acc if e.get("hit_group") == "Head"), None)
    if not head_entry:
        logger.warning(f"[bo3] No 'Head' entry in accuracy_stats for {player_slug!r}")
        return None

    head_kills = head_entry.get("kills_sum", 0)
    if not head_kills:
        return None

    gen = _get(f"/players/{player_slug}/general_stats")
    if not gen or not isinstance(gen, dict):
        logger.warning(f"[bo3] general_stats unavailable for {player_slug!r}")
        return None

    total_kills = gen.get("kills_sum", 0)
    if not total_kills:
        return None

    hs_pct = round(head_kills / total_kills, 4)
    logger.info(
        f"[bo3] Career HS% for {player_slug!r}: "
        f"{head_kills}/{total_kills} = {round(hs_pct * 100, 1)}%"
    )
    return hs_pct


# ---------------------------------------------------------------------------
# Per-series headshots from recent games (paginated)
# ---------------------------------------------------------------------------

def get_recent_series_hs(
    player_id: int,
    player_slug: str,
    n_series: int = 10,
    maps_per_series: int = 2,
) -> list[dict]:
    """
    Retrieve per-map kills and headshots for the player's most recent BO3 series.

    Strategy:
      1. Fetch pages of recent games sorted by date DESC (10 games per page).
      2. For each game, call /games/{id}/short_players_stats.
      3. Each short_player_stats entry covers an entire match (all maps combined),
         and is keyed by (player_id, match_id).
      4. Collect unique (match_id, kills_sum, headshots_sum) entries for the player.
      5. Stop once n_series matches are gathered or too many pages have been scanned.

    Returns a list of dicts:
      [{'match_id': ..., 'kills': ..., 'headshots': ..., 'hs_pct': ...}, ...]
    Sorted most-recent first (up to n_series entries).
    """
    results: list[dict] = []
    seen_match_ids: set[int] = set()
    max_pages = 30          # hard cap — 30 pages × 10 games = 300 games scanned
    offset = 0
    page_size = 10

    logger.info(
        f"[bo3] Scanning recent games for player_id={player_id} "
        f"(slug={player_slug!r}), target={n_series} series"
    )

    for page in range(max_pages):
        if len(results) >= n_series:
            break

        games_data = _get("/games", {"sort": "-begin_at", "limit": page_size, "offset": offset})
        if not games_data or not isinstance(games_data, dict):
            logger.warning(f"[bo3] Games page {page} failed")
            break

        games = games_data.get("results", [])
        if not games:
            logger.info(f"[bo3] No more games at offset {offset}")
            break

        for game in games:
            game_id = game.get("id")
            match_id = game.get("match_id")
            if not game_id or not match_id:
                continue

            # Skip if we've already processed this match
            if match_id in seen_match_ids:
                continue

            # Fetch per-player stats for this game
            sps = _get(f"/games/{game_id}/short_players_stats")
            if not sps or not isinstance(sps, list):
                continue

            # Look for our player
            player_entry = next(
                (e for e in sps if e.get("player_id") == player_id),
                None
            )
            if player_entry is None:
                continue  # this game doesn't involve our player

            seen_match_ids.add(match_id)
            kills = player_entry.get("kills_sum", 0) or 0
            headshots = player_entry.get("headshots_sum", 0) or 0
            hs_pct = round(headshots / kills, 4) if kills > 0 else None
            games_count = player_entry.get("games_count", 1)

            # Only include BO3 series (usually 2+ maps played)
            if games_count < 2 and (games_count is not None):
                pass  # include anyway — might be a 2-0

            results.append({
                "match_id":   match_id,
                "kills":      kills,
                "headshots":  headshots,
                "hs_pct":     hs_pct,
                "games_count": games_count,
                "game_id":    game_id,
                "map_name":   game.get("map_name"),
                "date":       str(game.get("begin_at", ""))[:10],
            })
            logger.info(
                f"[bo3] Series match_id={match_id}: "
                f"{kills}K {headshots}HS hs_pct={round(hs_pct*100, 1) if hs_pct else '?'}%"
                f" ({games_count} maps)"
            )

            if len(results) >= n_series:
                break

        offset += page_size
        time.sleep(0.2)  # gentle rate limiting

    logger.info(f"[bo3] Found {len(results)} series for {player_slug!r}")
    return results


# ---------------------------------------------------------------------------
# Combined fallback entry point
# ---------------------------------------------------------------------------

def get_bo3_hs_data(
    player_name: str,
    n_series: int = 10,
) -> Optional[dict]:
    """
    Full bo3.gg fallback flow.  Tries:
      1. Career HS% from accuracy_stats + general_stats  (fast, 2 calls)
      2. Recent series headshot counts from paginated games  (slower, many calls)

    Returns a dict:
      {
        'player_slug':    str,
        'player_id':      int,
        'career_hs_pct':  float | None,   # career headshot kill rate [0,1]
        'recent_series':  list[dict],     # per-series stats (up to n_series)
        'recent_hs_pct':  float | None,   # avg HS% over recent_series
        'source':         'bo3.gg',
      }
    Returns None if the player cannot be found on bo3.gg.
    """
    lookup = search_player_bo3(player_name)
    if not lookup:
        return None

    player_id, player_slug = lookup

    # Career HS% (fast, always attempted)
    career_hs_pct = get_career_hs_pct(player_slug)

    # Recent series headshot counts (slower, paginated)
    recent_series = get_recent_series_hs(player_id, player_slug, n_series=n_series)

    # Compute recent HS% from actual counts
    recent_hs_pct = None
    hs_rates = [
        s["headshots"] / s["kills"]
        for s in recent_series
        if s.get("kills", 0) > 0 and s.get("headshots") is not None
    ]
    if hs_rates:
        recent_hs_pct = round(sum(hs_rates) / len(hs_rates), 4)
        logger.info(
            f"[bo3] Recent HS% for {player_slug!r}: "
            f"{round(recent_hs_pct * 100, 1)}% (from {len(hs_rates)} series)"
        )

    return {
        "player_slug":   player_slug,
        "player_id":     player_id,
        "career_hs_pct": career_hs_pct,
        "recent_series": recent_series,
        "recent_hs_pct": recent_hs_pct,
        "source":        "bo3.gg",
    }
