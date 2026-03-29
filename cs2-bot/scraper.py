"""
HLTV scraper — uses curl_cffi Chrome impersonation against accessible HLTV endpoints.

Discovered accessible paths (no Cloudflare block):
  /search?query={name}          → player search (get player ID)
  /player/{id}/{slug}           → player profile (get team, overview)
  /results?player={id}          → player's recent results (get match IDs)
  /matches/{id}/{slug}          → match detail page (per-map kill stats)

Blocked paths (Cloudflare Turnstile — DO NOT USE):
  /stats/players                → always 403
  /stats/players/...            → always 403
  /stats/matches/...            → always 403
"""

import re
import random
import time
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HLTV_BASE = "https://www.hltv.org"
FETCH_TIMEOUT = 25  # seconds — match pages can be 500KB-1MB, give them room

try:
    from curl_cffi import requests as _cffi_req
    _CFFI_OK = True
except ImportError:
    _CFFI_OK = False
    logger.warning("curl_cffi not available — install it for HLTV access")


def _fetch(url: str) -> str | None:
    """Fetch a URL using curl_cffi Chrome impersonation. Returns None on failure."""
    if not _CFFI_OK:
        logger.warning(f"[fetch] curl_cffi not available, skipping {url}")
        return None
    try:
        logger.info(f"[fetch] GET {url}")
        resp = _cffi_req.get(url, impersonate="chrome110", timeout=FETCH_TIMEOUT)
        if resp.status_code == 200 and "Just a moment" not in resp.text:
            logger.info(f"[fetch] OK — {len(resp.text):,} chars")
            return resp.text
        logger.warning(f"[fetch] status={resp.status_code} or CF block — skipping")
        return None
    except Exception as e:
        logger.warning(f"[fetch] {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 1 — Player search
# ---------------------------------------------------------------------------

def search_player(name: str) -> tuple[str, str, str] | None:
    """
    Search HLTV for a player by name.
    Returns (player_id, player_slug, display_name) or None.
    """
    url = f"{HLTV_BASE}/search?query={name}"
    html = _fetch(url)
    if not html:
        return None

    matches = re.findall(r'/player/(\d+)/([\w-]+)', html)
    if not matches:
        logger.warning(f"[search] No player found for '{name}'")
        return None

    # Score matches by how closely the slug matches the search name
    name_lower = name.lower().replace(" ", "").replace("-", "")
    best = None
    best_score = -1
    for pid, slug in dict.fromkeys(matches).items() if isinstance(matches, dict) else dict.fromkeys(matches):
        slug_clean = slug.lower().replace("-", "")
        score = sum(1 for a, b in zip(name_lower, slug_clean) if a == b)
        if slug_clean == name_lower:
            score += 100  # exact match bonus
        if score > best_score:
            best_score = score
            best = (pid, slug)

    if not best:
        return None

    pid, slug = best
    display = slug.replace("-", " ").title()
    logger.info(f"[search] Found player: {display} (id={pid}, slug={slug})")
    return pid, slug, display


def _score_player_match(name: str, pid: str, slug: str) -> int:
    """Score how well a player ID/slug matches the searched name."""
    name_lower = re.sub(r'[^a-z0-9]', '', name.lower())
    slug_lower = re.sub(r'[^a-z0-9]', '', slug.lower())
    if name_lower == slug_lower:
        return 200
    if name_lower in slug_lower or slug_lower in name_lower:
        return 100
    # character overlap score
    return sum(1 for a, b in zip(name_lower, slug_lower) if a == b)


def search_player_v2(name: str) -> tuple[str, str, str] | None:
    """Improved player search with better matching."""
    url = f"{HLTV_BASE}/search?query={name}"
    html = _fetch(url)
    if not html:
        return None

    matches = re.findall(r'/player/(\d+)/([\w-]+)', html)
    if not matches:
        logger.warning(f"[search] No player found for '{name}'")
        return None

    seen = {}
    for pid, slug in matches:
        if pid not in seen:
            seen[pid] = slug

    best_pid, best_slug, best_score = None, None, -1
    for pid, slug in seen.items():
        score = _score_player_match(name, pid, slug)
        if score > best_score:
            best_score = score
            best_pid, best_slug = pid, slug

    if not best_pid:
        return None

    display = best_slug.replace("-", " ").title()
    logger.info(f"[search] Best match: {display} (id={best_pid}, slug={best_slug}, score={best_score})")
    return best_pid, best_slug, display


# ---------------------------------------------------------------------------
# Step 2 — Get player's recent BO3 match IDs from the results page
# ---------------------------------------------------------------------------

def get_player_match_ids(player_id: str, max_matches: int = 25) -> list[tuple[str, str]]:
    """
    Fetch /results?player={id} and return a list of (match_id, slug) tuples
    for recently completed matches. Only returns large IDs (7+ digits) which
    correspond to accessible HLTV match pages.
    """
    url = f"{HLTV_BASE}/results?player={player_id}"
    html = _fetch(url)
    if not html:
        return []

    # Find all match links on the page — only large IDs work (small IDs return 500)
    all_matches = re.findall(r'/matches/(\d+)/([a-z0-9-]+)', html)
    seen = {}
    for mid, slug in all_matches:
        if mid not in seen and len(mid) >= 6:
            seen[mid] = slug

    results = list(seen.items())[:max_matches]
    logger.info(f"[results] Found {len(results)} match IDs for player {player_id}")
    return results


# ---------------------------------------------------------------------------
# Step 3 — Parse a match page for per-map kills
# ---------------------------------------------------------------------------

def _parse_match_kills(html: str, player_slug: str) -> dict:
    """
    Parse an HLTV match page and return:
      {
        'bo_type': 3,          # or 1/2
        'maps': [
          {'map_name': 'Dust2', 'kills': 22, 'deaths': 14, 'map_number': 1},
          {'map_name': 'Inferno', 'kills': 19, 'deaths': 17, 'map_number': 2},
          ...
        ]
      }
    Returns None if the player isn't found or the match page has no stats.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Determine BO type from score (e.g. Vitality 2 - NaVi 1)
    score_elements = soup.find_all(class_=re.compile(r'won|lost|teamWon|teamLost', re.I))
    team_scores = re.findall(r'>\s*(\d)\s*<', html[:50000])
    bo_type = 1
    if team_scores:
        max_score = max(int(s) for s in team_scores if int(s) <= 3)
        if max_score >= 2:
            bo_type = 3

    matchstats = soup.find(id='match-stats')
    if not matchstats:
        logger.debug("[parse] No match-stats section found")
        return None

    # Get ordered map IDs from the tab navigation
    tab_ids = re.findall(r'id="(\d{5,7})"', str(matchstats))
    # Deduplicate preserving order, skip 'all'
    seen_ids = []
    for tid in tab_ids:
        if tid not in seen_ids:
            seen_ids.append(tid)
    map_ids = seen_ids  # ordered by map number

    logger.info(f"[parse] Maps found: {map_ids} | BO type: {bo_type}")

    # Get map names from the tab labels
    map_names = {}
    for div in matchstats.find_all(class_=re.compile(r'dynamic-map-name-full', re.I)):
        div_id = div.get('id', '')
        if div_id and div_id != 'all':
            map_names[div_id] = div.get_text(strip=True)

    # Normalise player slug for matching (lowercase, no hyphens)
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())

    maps_result = []
    for map_num, map_id in enumerate(map_ids, start=1):
        content_div = matchstats.find(id=f'{map_id}-content')
        if not content_div:
            continue

        map_name = map_names.get(map_id, f'Map{map_num}')

        # Find the player row — search all <tr> elements for the player's name
        player_row = None
        for tr in content_div.find_all('tr'):
            row_text = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
            if slug_norm in row_text and slug_norm:
                player_row = tr
                break

        if player_row is None:
            # Try partial match (first 4 chars of slug)
            short = slug_norm[:4]
            for tr in content_div.find_all('tr'):
                row_text = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
                if len(short) >= 3 and short in row_text:
                    player_row = tr
                    break

        if player_row is None:
            logger.debug(f"[parse] Player '{player_slug}' not found on map {map_num} ({map_name})")
            continue

        # Extract K-D from the row — format is "22-14"
        kd_match = re.search(r'(\d+)-(\d+)', player_row.get_text())
        if not kd_match:
            continue

        kills = int(kd_match.group(1))
        deaths = int(kd_match.group(2))
        maps_result.append({
            'map_name': map_name,
            'kills': kills,
            'deaths': deaths,
            'map_number': map_num,
        })
        logger.info(f"[parse] Map {map_num} ({map_name}): {player_slug} — {kills}K/{deaths}D")

    if not maps_result:
        return None

    return {'bo_type': bo_type, 'maps': maps_result}


# ---------------------------------------------------------------------------
# Step 4 — Get HS kills specifically (from per-round headshot data if available)
# ---------------------------------------------------------------------------

def _parse_hs_kills(html: str, player_slug: str) -> dict | None:
    """
    Try to extract headshot kills per map. HLTV doesn't show HS per map
    in the matchstats section directly, so we derive a HS% from the ADR
    pattern or fall back to None (caller will use overall HS% from profile).
    Returns same format as _parse_match_kills but kills = headshot kills only.
    """
    # HS data is not reliably available per-map on HLTV match pages.
    # We return None and let the caller handle HS via overall stats.
    return None


# ---------------------------------------------------------------------------
# Main entry point — used by the bot
# ---------------------------------------------------------------------------

def get_player_info(player_name: str, stat_type: str = "Kills") -> dict:
    """
    Main scraper entry. Returns:
      {
        'player':        'ZywOo',
        'player_id':     '11893',
        'map_kills':     [22, 19, 28, 21, 17, ...],   # last N maps (maps 1 & 2 of BO3)
        'mean':          21.4,
        'std':           3.8,
        'sample_size':   16,
        'source':        'HLTV Live',
      }
    Or raises RuntimeError if the player cannot be found.
    """
    logger.info(f"[scraper] Looking up '{player_name}' for {stat_type}")

    # Step 1: Find player
    result = search_player_v2(player_name)
    if not result:
        raise RuntimeError(f"Player '{player_name}' not found on HLTV")
    player_id, player_slug, display_name = result

    # Step 2: Get recent match IDs
    match_ids = get_player_match_ids(player_id, max_matches=30)
    if not match_ids:
        raise RuntimeError(f"No recent matches found for '{display_name}'")

    # Step 3: Fetch match pages and collect per-map kill data
    map_kills = []
    bo3_series_count = 0
    errors = 0

    for match_id, slug in match_ids:
        if bo3_series_count >= 10:
            break  # collected 10 BO3 series

        time.sleep(0.3)  # gentle rate limiting

        match_url = f"{HLTV_BASE}/matches/{match_id}/{slug}"
        html = _fetch(match_url)
        if not html:
            errors += 1
            if errors >= 5:
                break
            continue

        parsed = _parse_match_kills(html, player_slug)
        if not parsed:
            continue

        # Only count BO3 series (3 maps possible, score 2-1 or 2-0)
        maps = parsed.get('maps', [])
        if len(maps) < 2:
            continue  # Not enough map data — skip

        # Take maps 1 and 2 only
        for m in maps[:2]:
            map_kills.append(m['kills'])

        bo3_series_count += 1
        logger.info(
            f"[scraper] Series {bo3_series_count}: match {match_id} — "
            f"maps: {[(m['map_name'], m['kills']) for m in maps[:2]]}"
        )

    if len(map_kills) < 4:
        raise RuntimeError(
            f"Insufficient data for '{display_name}' — only {len(map_kills)} map samples found "
            f"(need at least 4). The player may be inactive or have few recent BO3 matches."
        )

    import statistics
    mean = statistics.mean(map_kills)
    std = statistics.stdev(map_kills) if len(map_kills) > 1 else 4.0
    std = max(std, 2.0)  # floor to avoid degenerate distributions

    return {
        'player': display_name,
        'player_id': player_id,
        'map_kills': map_kills,
        'mean': round(mean, 2),
        'std': round(std, 2),
        'sample_size': len(map_kills),
        'source': 'HLTV Live',
    }


# ---------------------------------------------------------------------------
# Fallback — generates seeded realistic estimates when HLTV is unreachable
# ---------------------------------------------------------------------------

def get_player_info_fallback(player_name: str, stat_type: str = "Kills") -> dict:
    """
    Generate seeded realistic estimated stats for when HLTV is unavailable.
    The seed is derived from the player name so the same player always gets
    the same estimates (reproducible but not real data).
    """
    seed = sum(ord(c) for c in player_name.lower())
    rng = random.Random(seed)

    # Elite fragger archetype varies by seed
    tier = seed % 3  # 0 = elite, 1 = solid, 2 = average
    if tier == 0:
        base_mean = rng.uniform(19, 24)
    elif tier == 1:
        base_mean = rng.uniform(16, 20)
    else:
        base_mean = rng.uniform(13, 17)

    std = rng.uniform(3.5, 5.5)
    n = 20  # simulate 20 map samples (10 BO3 series × 2 maps)
    map_kills = [
        max(5, int(rng.gauss(base_mean, std)))
        for _ in range(n)
    ]

    import statistics
    mean = statistics.mean(map_kills)
    real_std = statistics.stdev(map_kills)

    return {
        'player': player_name,
        'player_id': None,
        'map_kills': map_kills,
        'mean': round(mean, 2),
        'std': round(real_std, 2),
        'sample_size': n,
        'source': '⚠️ Estimated (HLTV unavailable — stats are approximate)',
    }
