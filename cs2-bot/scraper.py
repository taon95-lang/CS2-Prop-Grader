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
        filtered = [int(s) for s in team_scores if int(s) <= 3]
        if filtered:
            max_score = max(filtered)
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
    _dump_done = False   # only dump once per match for diagnostics
    for map_num, map_id in enumerate(map_ids, start=1):
        content_div = matchstats.find(id=f'{map_id}-content')
        if not content_div:
            continue

        map_name = map_names.get(map_id, f'Map{map_num}')

        # One-time structural dump: find every <td> containing '(' to locate HS cells
        if not _dump_done:
            _dump_done = True
            _hs_cells = []
            for td in content_div.find_all('td'):
                ct = td.get_text(strip=True)
                if '(' in ct and re.search(r'\d+\s*\(\d+\)', ct):
                    _hs_cells.append(repr(ct[:80]))
            logger.info(f"[hs_locate] Map1 cells with '(N)' pattern: {_hs_cells[:10]}")
            # Also dump all unique table classes in this content div
            _tbl_classes = [str(t.get('class', '')) for t in content_div.find_all('table')]
            logger.info(f"[hs_locate] Table classes in content_div: {_tbl_classes}")

        # Find ALL player rows (overview row + per-half row may both exist)
        player_rows = []
        for tr in content_div.find_all('tr'):
            row_text = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
            if slug_norm in row_text and slug_norm:
                player_rows.append(tr)

        # Prefer the row that contains a kills(HS)-deaths cell; fall back to first row found
        player_row = None
        for tr in player_rows:
            for td in tr.find_all('td'):
                if re.search(r'\d+\s*\(\d+\)\s*[-–]\s*\d+', td.get_text()):
                    player_row = tr
                    break
            if player_row:
                break
        if player_row is None and player_rows:
            player_row = player_rows[0]

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

        # Extract K, optional HS count, and D.
        # HLTV match stats have two rows per player per map:
        #   1) Overview row: "22 (8) - 14"  (total kills, HS in parens, total deaths)
        #   2) Per-half row: "11-16\n11-18" (CT kills-deaths \n T kills-deaths)
        # player_row selection above already prefers the overview row when available.
        row_text = player_row.get_text()
        logger.info(f"[parse_row] Map {map_num} ({map_name}) row text: {row_text[:300]!r}")

        headshots = None
        kills     = None
        deaths    = None

        # Strategy A: check each <td> for "kills(HS)-deaths" format (overview row)
        for td in player_row.find_all('td'):
            cell_text = td.get_text(strip=True)
            m = re.search(r'(\d+)\s*\((\d+)\)\s*[-–]\s*(\d+)', cell_text)
            if m:
                kills     = int(m.group(1))
                headshots = int(m.group(2))
                deaths    = int(m.group(3))
                logger.info(f"[parse_row] HS found in td: {cell_text!r} → K={kills} HS={headshots} D={deaths}")
                break

        if kills is None:
            # Strategy B: per-half row — find ALL K-D pairs, sum CT+T halves
            all_kd = re.findall(r'(\d+)\s*[-–]\s*(\d+)', row_text)
            # Filter pairs that look like kills-deaths (both ≤ 60 to exclude round counts)
            kd_pairs = [(int(k), int(d)) for k, d in all_kd if int(k) <= 60 and int(d) <= 60]
            if len(kd_pairs) >= 2:
                # Two pairs = CT half + T half; sum for total
                kills  = kd_pairs[0][0] + kd_pairs[1][0]
                deaths = kd_pairs[0][1] + kd_pairs[1][1]
                logger.info(f"[parse_row] Per-half sum: CT={kd_pairs[0]} T={kd_pairs[1]} → total K={kills} D={deaths}")
            elif len(kd_pairs) == 1:
                kills  = kd_pairs[0][0]
                deaths = kd_pairs[0][1]
            else:
                continue

        # Extract Rating 2.0 — it's a decimal like 1.15 in [0.40, 3.00]
        # found in td cells, typically the rightmost decimal value
        rating = None
        cells = player_row.find_all('td')
        for cell in reversed(cells):
            m = re.match(r'^\s*(\d+\.\d{2})\s*$', cell.get_text())
            if m:
                val = float(m.group(1))
                if 0.40 <= val <= 3.00:
                    rating = val
                    break

        # Extract KAST% — shown as "72%" or "0.72" in a td cell
        kast_pct = None
        for cell in cells:
            ct = cell.get_text(strip=True)
            m = re.match(r'^(\d{2,3})%$', ct)
            if m:
                val = int(m.group(1))
                if 20 <= val <= 100:
                    kast_pct = val
                    break
            # Some pages show as decimal 0.XX
            m2 = re.match(r'^(0\.\d{2})$', ct)
            if m2:
                val = round(float(m2.group(1)) * 100)
                if 20 <= val <= 100:
                    kast_pct = val
                    break

        # Extract ADR — float in range 20-150
        adr = None
        for cell in cells:
            ct = cell.get_text(strip=True)
            m = re.match(r'^(\d{2,3})\.?\d*$', ct)
            if m:
                val = float(ct)
                if 20.0 <= val <= 150.0 and '.' in ct:
                    adr = round(val, 1)
                    break

        # Compute survival rate from deaths (rounds - deaths) / rounds
        rounds_on_map = 22  # default; refined per-map if parseable
        survival_rate = round((rounds_on_map - deaths) / rounds_on_map, 3)

        # Extract FK (First Kills) — integer cell BEFORE the rating
        # FK-FD columns appear on some pages as standalone integers (0-20)
        fk = None
        fd = None
        int_cells = []
        for cell in cells:
            ct = cell.get_text(strip=True)
            if re.match(r'^\d+$', ct) and 0 <= int(ct) <= 40:
                int_cells.append(int(ct))
        # int_cells typically: [kills, deaths, fk, fd, ...]
        # K-D cell shows "22-14" so we skip that; standalone int cells after K-D are fk/fd
        if len(int_cells) >= 2:
            fk = int_cells[0]
            fd = int_cells[1]

        maps_result.append({
            'map_name':      map_name,
            'kills':         kills,
            'headshots':     headshots,   # int or None if not shown on scorecard
            'deaths':        deaths,
            'rating':        rating,
            'kast_pct':      kast_pct,
            'adr':           adr,
            'survival_rate': survival_rate,
            'fk':            fk,
            'fd':            fd,
            'map_number':    map_num,
        })
        _hs_str = f" HS={headshots}" if headshots is not None else ""
        logger.info(
            f"[parse] Map {map_num} ({map_name}): {player_slug} — "
            f"{kills}K{_hs_str}/{deaths}D rating={rating} fk={fk}"
        )

    if not maps_result:
        return None

    return {'bo_type': bo_type, 'maps': maps_result}


def _parse_pistol_stats(html: str, player_slug: str) -> dict:
    """
    Try to extract per-map pistol round kill counts for a player.
    HLTV shows pistol round stats in a separate section within match-stats.
    Returns {map_number: pistol_kills} or {} if not found.

    Fallback: if pistol section not found, estimate from overall kills/rounds
    (2 pistol rounds per half → ~2/22 of total kills per map).
    """
    soup = BeautifulSoup(html, 'html.parser')
    matchstats = soup.find(id='match-stats')
    if not matchstats:
        return {}

    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())
    result = {}

    # Strategy 1: Look for dedicated pistol-round sections
    # HLTV renders pistol stats in divs with class containing 'pistol'
    pistol_sections = matchstats.find_all(
        lambda tag: tag.name in ('div', 'section') and
        any('pistol' in cls.lower() for cls in tag.get('class', []))
    )

    for i, section in enumerate(pistol_sections[:2], start=1):
        for tr in section.find_all('tr'):
            row_text = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
            if slug_norm and slug_norm in row_text:
                kd = re.search(r'(\d+)\s*[-–]\s*(\d+)', tr.get_text())
                if kd:
                    result[i] = int(kd.group(1))
                    break

    if result:
        logger.info(f"[pistol] Scraped pistol kills for {player_slug}: {result}")
        return result

    # Strategy 2: Estimate from overall per-map kill rate
    # ~2 pistol rounds per 22-round map → estimated pistol contribution
    # Walk main stats to get per-map kills and compute estimate
    raw = str(matchstats)
    map_ids = re.findall(r'id="(\d{5,7})-content"', raw)
    for map_num, map_id in enumerate(map_ids[:2], start=1):
        content_div = matchstats.find(id=f'{map_id}-content')
        if not content_div:
            continue
        for tr in content_div.find_all('tr'):
            row_text = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
            if slug_norm and slug_norm in row_text:
                kd = re.search(r'(\d+)-(\d+)', tr.get_text())
                if kd:
                    kills = int(kd.group(1))
                    # Estimate: pistol rounds are ~9% of rounds (2/22)
                    est = round(kills * 2 / 22, 2)
                    result[map_num] = est
                    break

    return result


def get_player_hs_pct(player_id: str, player_slug: str) -> float | None:
    """
    Scrape a player's career headshot % from their HLTV profile page.
    Returns a float in [0.0, 1.0] or None if not found.

    HLTV player page (/player/{id}/{slug}) shows stats including HS%.
    The value appears near a label containing 'headshot' or 'hs'.
    """
    url  = f"{HLTV_BASE}/player/{player_id}/{player_slug}"
    html = _fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Strategy 1: look for a stat row/cell mentioning 'headshot'
    for tag in soup.find_all(['span', 'div', 'td', 'p']):
        text = tag.get_text(strip=True).lower()
        if 'headshot' in text or 'hs%' in text:
            m = re.search(r'(\d{1,2})\.?\d*\s*%', tag.get_text())
            if m:
                val = float(m.group(1))
                if 10 <= val <= 80:
                    logger.info(f"[hs_pct] Scraped HS% for {player_slug}: {val}%")
                    return round(val / 100, 3)
            # Maybe it's in a sibling element
            parent = tag.parent
            if parent:
                sib_text = parent.get_text()
                m2 = re.search(r'(\d{1,2})\.?\d*\s*%', sib_text)
                if m2:
                    val = float(m2.group(1))
                    if 10 <= val <= 80:
                        logger.info(f"[hs_pct] Scraped HS% for {player_slug} (sibling): {val}%")
                        return round(val / 100, 3)

    # Strategy 2: Scan the entire page for lines like "42%" near "headshot"
    raw = html.lower()
    idx = raw.find('headshot')
    if idx != -1:
        snippet = html[max(0, idx - 50): idx + 100]
        m = re.search(r'(\d{1,2})\.?\d*\s*%', snippet)
        if m:
            val = float(m.group(1))
            if 10 <= val <= 80:
                logger.info(f"[hs_pct] Scraped HS% for {player_slug} (text scan): {val}%")
                return round(val / 100, 3)

    logger.info(f"[hs_pct] Could not find HS% for {player_slug} — will use default")
    return None


# ---------------------------------------------------------------------------
# Step 4 — Get HS kills specifically (from per-round headshot data if available)
# ---------------------------------------------------------------------------

def _parse_match_hs_pct(html: str, player_slug: str) -> float | None:
    """
    Extract the player's HS% from a match page's all-maps overview table.

    HLTV match pages show per-player all-map stats (kills, deaths, rating,
    ADR, KAST%, HS%) in a summary section separate from per-map scorecards.

    Strategy: find any row that contains the player's slug, then pull
    percentage cells. KAST% is typically 50-100, HS% is 10-65.
    We take the LAST percentage in the row (KAST tends to be listed before HS).
    Returns a float in [0.0, 1.0] or None.
    """
    soup = BeautifulSoup(html, 'html.parser')
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())

    def _row_pcts(row) -> list[int]:
        """Return all XX% integer values from a <tr> row."""
        vals = []
        for cell in row.find_all(['td', 'th']):
            ct = cell.get_text(strip=True)
            m = re.match(r'^(\d{1,3})%$', ct)
            if m:
                vals.append(int(m.group(1)))
        return vals

    # Strategy 1: scan every table row for the player name
    for row in soup.find_all('tr'):
        row_norm = re.sub(r'[^a-z0-9]', '', row.get_text().lower())
        if slug_norm not in row_norm:
            continue
        pcts = _row_pcts(row)
        if not pcts:
            continue
        # HLTV column order (typical): ...KAST(50-100)...HS(10-65)
        # Take the last percentage that fits the HS range
        hs_candidates = [p for p in pcts if 10 <= p <= 65]
        if hs_candidates:
            val = hs_candidates[-1]
            logger.debug(f"[hs_pct] Row match for {player_slug}: pcts={pcts} → HS={val}%")
            return round(val / 100, 3)

    # Strategy 2: Scan anchors — the player's profile link often has a data-attribute
    # or appears near their stats; look for HS% in siblings
    for a_tag in soup.find_all('a', href=re.compile(rf'/player/\d+/{re.escape(player_slug)}', re.I)):
        row = a_tag.find_parent('tr')
        if row:
            pcts = _row_pcts(row)
            hs_candidates = [p for p in pcts if 10 <= p <= 65]
            if hs_candidates:
                val = hs_candidates[-1]
                logger.debug(f"[hs_pct] Anchor match for {player_slug}: HS={val}%")
                return round(val / 100, 3)

    return None


def _parse_hs_kills(html: str, player_slug: str) -> dict | None:
    """Legacy stub — HS data not available per-map; handled via _parse_match_hs_pct."""
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
    hs_pct_samples: list[float] = []   # per-match HS% — averaged for recent_hs_pct

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

        # Collect HS% from the match-level overview (all-maps combined stats row)
        # This is parsed from the same page we already fetched — no extra requests.
        match_hs = _parse_match_hs_pct(html, player_slug)
        if match_hs is not None:
            hs_pct_samples.append(match_hs)
            logger.info(f"[hs_pct] Match {match_id}: {player_slug} HS%={round(match_hs*100)}%")

        # Attempt pistol round parse for this match
        pistol_data = _parse_pistol_stats(html, player_slug)

        # Take maps 1 and 2 only — store dicts so simulator has stat_value + match_id
        for m in maps[:2]:
            map_num = m.get('map_number', 1)
            map_kills.append({
                'stat_value':    m['kills'],
                'headshots':     m.get('headshots'),  # actual HS count or None
                'rounds':        22,
                'match_id':      match_id,
                'map_name':      m['map_name'].lower(),
                'rating':        m.get('rating'),
                'kast_pct':      m.get('kast_pct'),
                'adr':           m.get('adr'),
                'survival_rate': m.get('survival_rate'),
                'fk':            m.get('fk'),
                'fd':            m.get('fd'),
                'deaths':        m.get('deaths'),
                'pistol_kills':  pistol_data.get(map_num),
            })

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
    kill_values = [m['stat_value'] for m in map_kills]
    mean = statistics.mean(kill_values)
    std = statistics.stdev(kill_values) if len(kill_values) > 1 else 4.0
    std = max(std, 2.0)  # floor to avoid degenerate distributions

    # Prefer actual per-map HS counts (scraped from "kills (HS)" in scorecard) over
    # the all-maps overview HS% when available — it's the ground truth.
    actual_hs_rates = [
        mk['headshots'] / mk['stat_value']
        for mk in map_kills
        if mk.get('headshots') is not None and mk.get('stat_value', 0) > 0
    ]
    if actual_hs_rates:
        recent_hs_pct = round(sum(actual_hs_rates) / len(actual_hs_rates), 3)
        logger.info(
            f"[hs_pct] Actual per-map HS% for {player_slug}: "
            f"{round(recent_hs_pct*100, 1)}% (from {len(actual_hs_rates)} maps with real counts)"
        )
    elif hs_pct_samples:
        recent_hs_pct = round(sum(hs_pct_samples) / len(hs_pct_samples), 3)
        logger.info(
            f"[hs_pct] Fallback overview HS% for {player_slug}: {round(recent_hs_pct*100, 1)}% "
            f"(avg of {len(hs_pct_samples)} match overviews)"
        )
    else:
        recent_hs_pct = None

    return {
        'player':            display_name,
        'player_id':         player_id,
        'player_slug':       player_slug,
        'match_ids':         match_ids,        # full list — used for H2H filtering
        'map_kills':         map_kills,
        'mean':              round(mean, 2),
        'std':               round(std, 2),
        'sample_size':       len(map_kills),
        'source':            'HLTV Live',
        'recent_hs_pct':     recent_hs_pct,   # None if no HS data found on match pages
        'hs_pct_n_matches':  len(hs_pct_samples),
    }


# ---------------------------------------------------------------------------
# Player team lookup (from player profile page)
# ---------------------------------------------------------------------------

def get_player_team(player_id: str, player_slug: str) -> tuple[str, str] | None:
    """
    Fetch the player's profile page and extract their current team ID + slug.
    Returns (team_id, team_slug) or None.
    """
    url = f"{HLTV_BASE}/player/{player_id}/{player_slug}"
    html = _fetch(url)
    if not html:
        return None

    matches = re.findall(r'/team/(\d+)/([\w-]+)', html)
    if not matches:
        return None

    tid, tslug = matches[0]
    logger.info(f"[player_team] player={player_slug} → team_id={tid} slug={tslug}")
    return tid, tslug


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
    raw_kills = [max(5, int(rng.gauss(base_mean, std))) for _ in range(n)]
    # Store as dicts to match the live scraper format expected by the simulator
    map_kills = [
        {'stat_value': k, 'rounds': 22, 'match_id': f'fallback_{i // 2}'}
        for i, k in enumerate(raw_kills)
    ]

    import statistics
    mean = statistics.mean(raw_kills)
    real_std = statistics.stdev(raw_kills)

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
    # "ex-TEAM" aliases — user may type "exruby", "ex-ruby", "ex ruby", etc.
    "exruby": "ruby",
    "ex-ruby": "ruby",
    "ex ruby": "ruby",
    "exgambit": "gambit-esports",
    "ex-gambit": "gambit-esports",
    "exnavi": "natus-vincere",
    "ex-navi": "natus-vincere",
}

_SECONDARY_MARKERS = ('junior', 'academy', 'youth', '-2', '-b-team', 'b-team', 'female', 'women')


def _score_team_candidates(candidates: dict[str, str], name_norm: str) -> tuple[str | None, str | None, int]:
    """
    Score a {team_id: slug} dict against a normalised target name.
    Returns (best_tid, best_slug, best_score).
    """
    best_tid, best_slug, best_score = None, None, -1000
    for tid, slug in candidates.items():
        slug_norm = re.sub(r'[^a-z0-9]', '', slug.lower())
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
        if any(marker in slug.lower() for marker in _SECONDARY_MARKERS):
            score -= 120
        if score > best_score:
            best_score, best_tid, best_slug = score, tid, slug
    return best_tid, best_slug, best_score


def search_team(name: str) -> tuple | None:
    """
    Search HLTV for a team by name or alias.
    Returns (team_id, team_slug, display_name) or None if not found.
    """
    name_clean = name.lower().strip()

    # 1) Resolve explicit aliases first
    query = _TEAM_ALIASES.get(name_clean, name)

    # 2) Auto-normalise "ex-TEAM" / "exTEAM" input that isn't in the alias table.
    #    Strips the leading "ex-" or "ex" prefix and uses the remainder as the query.
    if query == name:   # alias table didn't fire
        m = re.match(r'^ex[-\s]?(.+)$', name_clean)
        if m:
            query = m.group(1)   # e.g. "exruby" → "ruby", "ex natus vincere" → "natus vincere"

    def _search_query(q: str) -> dict[str, str]:
        url = f"{HLTV_BASE}/search?query={q}"
        html = _fetch(url)
        if not html:
            return {}
        seen: dict[str, str] = {}
        for tid, slug in re.findall(r'/team/(\d+)/([\w-]+)', html):
            if tid not in seen:
                seen[tid] = slug
        return seen

    # First attempt with resolved query
    seen = _search_query(query)
    if not seen:
        logger.warning(f"[search_team] no /team/ links found for '{name}' (query='{query}')")
        return None

    name_norm = re.sub(r'[^a-z0-9]', '', query.lower())
    best_tid, best_slug, best_score = _score_team_candidates(seen, name_norm)

    # If no meaningful match, try again with the raw user input as the query
    if best_score <= 0 and query != name:
        seen2 = _search_query(name)
        if seen2:
            raw_norm = re.sub(r'[^a-z0-9]', '', name.lower())
            t2, s2, sc2 = _score_team_candidates(seen2, raw_norm)
            if sc2 > best_score:
                best_tid, best_slug, best_score = t2, s2, sc2
                logger.info(f"[search_team] Retry with raw query improved score to {sc2}")

    # Refuse to return a result when there's no string overlap at all — it would be wrong
    if best_score <= 0:
        logger.warning(
            f"[search_team] '{name}' (query='{query}') — best score was {best_score} "
            f"(slug='{best_slug}'). Refusing to return a mismatched team."
        )
        return None

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
