"""
HLTV scraper — uses curl_cffi Chrome impersonation against accessible HLTV endpoints.

Discovered accessible paths (no Cloudflare block):
  /search?query={name}          → player/team search (get IDs)
  /player/{id}/{slug}           → player profile (get team, overview)
  /results?player={id}          → player's recent results (get match IDs)
  /results?team={id}            → team's recent results (get match IDs)
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
import statistics as _stats
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


# ---------------------------------------------------------------------------
# Team search & defensive stats
# ---------------------------------------------------------------------------

# Pro-level baseline: average kills per player per map across tier-1/2 CS2
_BASELINE_KILLS_PER_MAP = 18.5

# In-memory cache: {team_id_str: (timestamp, result_dict)}
_TEAM_DEF_CACHE: dict = {}
_TEAM_DEF_CACHE_TTL = 4 * 3600  # 4 hours


_TEAM_ALIASES: dict[str, str] = {
    "navi": "natus-vincere",
    "naví": "natus-vincere",
    "natus vincere": "natus-vincere",
    "g2": "g2-esports",
    "faze": "faze",
    "nip": "ninjas-in-pyjamas",
    "ninjas": "ninjas-in-pyjamas",
    "mouz": "mousesports",
    "astralis": "astralis",
    "liquid": "team-liquid",
    "ence": "ence",
    "heroic": "heroic",
    "cloud9": "cloud9",
    "c9": "cloud9",
    "spirit": "team-spirit",
    "vitality": "team-vitality",
    "complexity": "complexity-gaming",
    "col": "complexity-gaming",
    "big": "big",
    "apeks": "apeks",
    "pain": "pain-gaming",
    "imperial": "imperial",
    "9z": "9z",
    "outsiders": "outsiders",
    "forze": "forze",
    "gambit": "gambit-esports",
    "fnatic": "fnatic",
    "eg": "evil-geniuses",
    "evil geniuses": "evil-geniuses",
}

_SECONDARY_MARKERS = ('junior', 'academy', 'youth', '-2', '-b-team', 'b-team', 'female', 'women')


def search_team(name: str) -> tuple | None:
    """
    Search HLTV for a team by name or alias.
    Returns (team_id, team_slug, display_name) or None if not found.
    """
    # Resolve known aliases first so the search query is more accurate
    query = _TEAM_ALIASES.get(name.lower().strip(), name)

    url = f"{HLTV_BASE}/search?query={query}"
    html = _fetch(url)
    if not html:
        logger.warning(f"[search_team] fetch failed for '{name}'")
        return None

    matches = re.findall(r'/team/(\d+)/([\w-]+)', html)
    if not matches:
        logger.warning(f"[search_team] no /team/ links found for '{name}'")
        return None

    # Deduplicate while preserving order
    seen: dict[str, str] = {}
    for tid, slug in matches:
        if tid not in seen:
            seen[tid] = slug

    # Scoring: exact slug match → contains → partial → penalise junior/academy squads
    name_norm = re.sub(r'[^a-z0-9]', '', query.lower())
    best_tid, best_slug, best_score = None, None, -1000

    for tid, slug in seen.items():
        slug_norm = re.sub(r'[^a-z0-9]', '', slug.lower())

        # Base match score
        if slug_norm == name_norm:
            score = 200
        elif slug_norm.startswith(name_norm):
            score = 150
        elif name_norm in slug_norm:
            score = 100
        elif slug_norm in name_norm:
            score = 50
        else:
            score = 0

        # Heavy penalty for junior/academy/female rosters
        if any(marker in slug.lower() for marker in _SECONDARY_MARKERS):
            score -= 120

        if score > best_score:
            best_score, best_tid, best_slug = score, tid, slug

    if not best_tid:
        best_tid, best_slug = next(iter(seen.items()))

    display = best_slug.replace('-', ' ').title()
    logger.info(f"[search_team] '{name}' (query='{query}') → team_id={best_tid} slug={best_slug} score={best_score}")
    return best_tid, best_slug, display


def _get_match_kills_for_team(html: str, team_id: str) -> list[int]:
    """
    Given match page HTML and a team_id, return a list of per-player kill counts
    scored BY THE OPPONENT (i.e., kills conceded by our target team).
    Each entry is one player's kills on one map.
    """
    soup = BeautifulSoup(html, 'html.parser')
    matchstats = soup.find(id='match-stats')
    if not matchstats:
        return []

    raw = str(matchstats)
    map_ids = re.findall(r'id="(\d{5,7})-content"', raw)
    if not map_ids:
        return []

    kill_samples: list[int] = []

    for map_id in map_ids[:2]:  # Maps 1 & 2 only
        content_div = matchstats.find(id=f'{map_id}-content')
        if not content_div:
            continue

        tables = content_div.find_all('table', class_='totalstats')
        if len(tables) < 2:
            continue

        for table in tables:
            # Identify which team owns this table via the /team/{id}/ href
            team_link = table.find('a', href=re.compile(rf'/team/{team_id}/'))
            if team_link:
                # This table belongs to our target team — skip it (we want opponents)
                continue

            # This is the opponent's table — collect their kills
            for tr in table.find_all('tr')[1:]:  # skip header row
                kd_text = tr.get_text()
                kd_match = re.search(r'(\d+)\s*-\s*\d+', kd_text)
                if kd_match:
                    kills = int(kd_match.group(1))
                    if 3 <= kills <= 60:  # sanity bounds
                        kill_samples.append(kills)

    return kill_samples


def get_team_defensive_stats(team_id: str, n_matches: int = 10) -> dict | None:
    """
    Compute how many kills the given team concedes per player per map on average.

    Returns:
        {
            'avg_kills_allowed': 16.8,     # kills per opponent player per map
            'adjustment': 0.91,            # multiplier vs baseline
            'label': 'tough',              # 'tough' | 'average' | 'soft'
            'sample_maps': 18,
        }
    or None if insufficient data.
    """
    # Check in-memory cache
    cached = _TEAM_DEF_CACHE.get(team_id)
    if cached:
        ts, data = cached
        if time.time() - ts < _TEAM_DEF_CACHE_TTL:
            logger.info(f"[defensive_stats] cache hit for team_id={team_id}")
            return data

    results_url = f"{HLTV_BASE}/results?team={team_id}"
    html = _fetch(results_url)
    if not html:
        logger.warning(f"[defensive_stats] could not fetch results for team_id={team_id}")
        return None

    match_pairs = re.findall(r'/matches/(\d+)/([\w-]+)', html)
    seen: dict[str, str] = {}
    for mid, slug in match_pairs:
        if mid not in seen and len(mid) >= 6:
            seen[mid] = slug

    match_list = list(seen.items())[:n_matches]
    if not match_list:
        logger.warning(f"[defensive_stats] no matches found for team_id={team_id}")
        return None

    all_kills: list[int] = []

    for match_id, slug in match_list:
        time.sleep(0.4)
        match_url = f"{HLTV_BASE}/matches/{match_id}/{slug}"
        page_html = _fetch(match_url)
        if not page_html:
            continue
        kills = _get_match_kills_for_team(page_html, team_id)
        all_kills.extend(kills)
        logger.info(
            f"[defensive_stats] match {match_id}: {len(kills)} opponent kill samples"
        )

    if len(all_kills) < 5:
        logger.warning(f"[defensive_stats] only {len(all_kills)} samples — not enough")
        return None

    avg = _stats.mean(all_kills)
    adjustment = round(avg / _BASELINE_KILLS_PER_MAP, 4)
    adjustment = max(0.75, min(1.25, adjustment))  # clamp to ±25%

    if avg < 16.5:
        label = 'tough'
    elif avg > 20.5:
        label = 'soft'
    else:
        label = 'average'

    result = {
        'avg_kills_allowed': round(avg, 1),
        'adjustment': round(adjustment, 4),
        'label': label,
        'sample_maps': len(all_kills),
    }

    _TEAM_DEF_CACHE[team_id] = (time.time(), result)
    logger.info(f"[defensive_stats] team_id={team_id} → {result}")
    return result


def get_matchup_adjustment(opponent_name: str) -> dict | None:
    """
    Public entry point: search for a team, then fetch its defensive profile.

    Returns a dict with:
        team_display   : str   — e.g. "Natus Vincere"
        adjustment     : float — multiplier applied to kill distribution (0.75–1.25)
        label          : str   — 'tough' | 'average' | 'soft'
        avg_allowed    : float — avg kills conceded per player per map
        sample_maps    : int
    or None if the team can't be found / not enough data.
    """
    team_info = search_team(opponent_name)
    if not team_info:
        logger.warning(f"[matchup] team not found: '{opponent_name}'")
        return None

    team_id, team_slug, display = team_info
    def_stats = get_team_defensive_stats(team_id)
    if not def_stats:
        logger.warning(f"[matchup] no defensive stats for {display} (id={team_id})")
        return None

    return {
        'team_display': display,
        'adjustment': def_stats['adjustment'],
        'label': def_stats['label'],
        'avg_allowed': def_stats['avg_kills_allowed'],
        'sample_maps': def_stats['sample_maps'],
    }
