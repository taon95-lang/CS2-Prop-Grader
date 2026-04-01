"""
HLTV scraper — uses curl_cffi Chrome impersonation against accessible HLTV endpoints.

Discovered accessible paths (no Cloudflare block):
  /search?query={name}          → player/team search (get IDs)
  /player/{id}/{slug}           → player profile (get team, overview)
  /results?player={id}          → player's recent results (get match IDs)
  /results?team={id}            → team's recent results (get match IDs)
  /matches/{id}/{slug}          → match detail page (per-map kill stats)
  /stats/matches/mapstatsid/... → per-map detailed stats with K(hs) column
                                  (requires cookie-warmed session — falls back
                                  gracefully if Cloudflare blocks the request)

Previously blocked but now attempted via session warm-up:
  /stats/players/...            → attempted, falls back if 403
"""

import re
import random
import time
import logging
import statistics as _stats
from datetime import date, timedelta
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


_FETCH_RETRY_DELAYS = [1.0, 2.5, 5.0]   # seconds between retries within one profile

# ---------------------------------------------------------------------------
# Impersonation profile rotation
# Tested live — profiles that return 200 from hltv.org are listed first.
# chrome120/chrome124 currently return 403; rotate away from them on failure.
# ---------------------------------------------------------------------------
_PROFILES = ["chrome116", "safari17_0", "chrome107", "chrome110", "chrome99"]
_profile_idx = 0   # index into _PROFILES — advances on repeated 403s

# ---------------------------------------------------------------------------
# Persistent HLTV session — shared across ALL requests so Cloudflare cookies
# accumulate and the IP is treated as a returning browser, not a fresh bot.
# ---------------------------------------------------------------------------
_HLTV_SESSION: "_cffi_req.Session | None" = None
_HLTV_SESSION_WARMED = False
_HLTV_SESSION_PROFILE: str = _PROFILES[0]


def _make_session(profile: str) -> "_cffi_req.Session | None":
    """Create a fresh curl_cffi Session with the given impersonation profile."""
    try:
        sess = _cffi_req.Session(impersonate=profile)
        logger.info(f"[session] Created session with profile={profile}")
        return sess
    except Exception as e:
        logger.warning(f"[session] Could not create session ({profile}): {e}")
        return None


def _get_hltv_session() -> "_cffi_req.Session | None":
    """Return (or lazily create) the shared persistent HLTV curl_cffi Session."""
    global _HLTV_SESSION, _HLTV_SESSION_PROFILE
    if not _CFFI_OK:
        return None
    if _HLTV_SESSION is None:
        _HLTV_SESSION_PROFILE = _PROFILES[_profile_idx]
        _HLTV_SESSION = _make_session(_HLTV_SESSION_PROFILE)
    return _HLTV_SESSION


def _rotate_session() -> "_cffi_req.Session | None":
    """
    Drop the current session and open a new one using the next profile in the
    rotation list.  Called when the current profile starts returning 403.

    Also resets the stats-page circuit-breaker so the new profile gets a clean
    chance to reach /stats/matches/mapstatsid/ pages (they may be accessible
    under a profile that the previous one couldn't use).
    """
    global _HLTV_SESSION, _HLTV_SESSION_WARMED, _HLTV_SESSION_PROFILE, _profile_idx
    global _STATS_PAGES_BLOCKED, _STATS_SESSION_WARMED
    _profile_idx = (_profile_idx + 1) % len(_PROFILES)
    new_profile = _PROFILES[_profile_idx]
    logger.warning(f"[session] Rotating profile → {new_profile}")
    _HLTV_SESSION = _make_session(new_profile)
    _HLTV_SESSION_PROFILE = new_profile
    _HLTV_SESSION_WARMED = False        # re-warm homepage with new session
    _STATS_SESSION_WARMED = False       # re-warm stats referer with new session
    _STATS_PAGES_BLOCKED = False        # give stats pages a fresh chance
    return _HLTV_SESSION


def _warm_hltv_session() -> None:
    """
    Visit the HLTV homepage to seed Cloudflare cookies in the session.
    Tries every profile in the rotation until one succeeds.
    """
    global _HLTV_SESSION_WARMED
    if _HLTV_SESSION_WARMED:
        return
    if not _CFFI_OK:
        return

    for _ in range(len(_PROFILES)):
        sess = _get_hltv_session()
        if sess is None:
            return
        try:
            r = sess.get(HLTV_BASE + "/", timeout=15)
            logger.info(
                f"[session] Homepage warm-up: {r.status_code} "
                f"(profile={_HLTV_SESSION_PROFILE})"
            )
            if r.status_code == 200 and "Just a moment" not in r.text:
                _HLTV_SESSION_WARMED = True
                time.sleep(0.5)
                return
            # This profile is blocked — rotate and try next
            logger.warning(
                f"[session] Warm-up 403 with {_HLTV_SESSION_PROFILE} — rotating"
            )
            _rotate_session()
        except Exception as e:
            logger.warning(f"[session] Warm-up error ({_HLTV_SESSION_PROFILE}): {e}")
            _rotate_session()

    logger.warning("[session] All profiles failed warm-up — proceeding without cookie seed")


def _fetch(url: str, max_retries: int = 3) -> str | None:
    """
    Fetch a URL using the persistent HLTV session with automatic profile rotation.

    Strategy:
      1. Try the current session/profile up to max_retries times.
      2. On a 403, rotate to the next impersonation profile and retry immediately.
      3. After exhausting all profiles, give up and return None.

    Live-tested working profiles (as of 2025-03): chrome116, safari17_0, chrome107.
    """
    if not _CFFI_OK:
        logger.warning(f"[fetch] curl_cffi not available, skipping {url}")
        return None

    _warm_hltv_session()   # no-op after first successful warm-up

    profiles_tried = 0
    max_profile_rotations = len(_PROFILES)

    while profiles_tried <= max_profile_rotations:
        sess = _get_hltv_session()
        if sess is None:
            return None

        got_403_this_profile = False
        for attempt in range(max_retries):
            try:
                tag = f" (retry {attempt})" if attempt else ""
                logger.info(
                    f"[fetch] GET {url}{tag} [{_HLTV_SESSION_PROFILE}]"
                )
                resp = sess.get(url, timeout=FETCH_TIMEOUT)

                if resp.status_code == 200 and "Just a moment" not in resp.text:
                    logger.info(f"[fetch] OK — {len(resp.text):,} chars [{_HLTV_SESSION_PROFILE}]")
                    return resp.text

                # 403 / CF challenge — note it and stop hammering this profile
                logger.warning(
                    f"[fetch] status={resp.status_code} [{_HLTV_SESSION_PROFILE}]"
                    + (f" — retrying in {_FETCH_RETRY_DELAYS[min(attempt, len(_FETCH_RETRY_DELAYS)-1)]}s"
                       if attempt < max_retries - 1 else " — profile exhausted")
                )
                if resp.status_code == 403:
                    got_403_this_profile = True
                    if attempt < max_retries - 1:
                        time.sleep(_FETCH_RETRY_DELAYS[attempt])
                    else:
                        break   # rotate profile
                else:
                    if attempt < max_retries - 1:
                        time.sleep(_FETCH_RETRY_DELAYS[attempt])

            except Exception as e:
                logger.warning(
                    f"[fetch] {type(e).__name__}: {e} [{_HLTV_SESSION_PROFILE}]"
                )
                if attempt < max_retries - 1:
                    time.sleep(_FETCH_RETRY_DELAYS[attempt])
                else:
                    break

        if got_403_this_profile:
            # Rotate to next profile and try again
            profiles_tried += 1
            if profiles_tried <= max_profile_rotations:
                _rotate_session()
                time.sleep(0.8)
            continue
        else:
            # Non-403 failure (timeout, parse error) — don't rotate, just give up
            break

    logger.warning(f"[fetch] Giving up on {url} after trying {profiles_tried} profile(s)")
    return None


# ---------------------------------------------------------------------------
# Player ID cache — avoids re-running HLTV search on every command
# ---------------------------------------------------------------------------
# Pre-seeded with verified HLTV IDs for commonly graded CS2 players.
# Keys are lowercase normalised nicknames for robust matching.
# Cache grows automatically when new players are successfully looked up.

_PLAYER_ID_CACHE: dict[str, tuple[str, str, str]] = {
    # key            player_id   slug              display_name
    "lake":         ("22921",   "lake",            "Lake"),
    "zywoo":        ("11893",   "zywoo",           "ZywOo"),
    "donk":         ("21202",   "donk",            "donk"),
    "niko":         ("3741",    "niko",            "NiKo"),
    "m0nesy":       ("18943",   "m0nesy",          "m0NESY"),
    "sh1ro":        ("15096",   "sh1ro",           "sh1ro"),
    "b1t":          ("18936",   "b1t",             "b1t"),
    "simple":       ("7998",    "simple",          "s1mple"),
    "s1mple":       ("7998",    "simple",          "s1mple"),
    "twistzz":      ("10394",   "twistzz",         "Twistzz"),
    "elige":        ("9816",    "elige",           "EliGE"),
    "ropz":         ("11816",   "ropz",            "ropz"),
    "yekindar":     ("16957",   "yekindar",        "YEKINDAR"),
    "naf":          ("9176",    "naf",             "NAF"),
    "perfecto":     ("18872",   "perfecto",        "Perfecto"),
    "electronic":   ("8649",    "electronic",      "electronic"),
    "broky":        ("15513",   "broky",           "broky"),
    "rain":         ("3728",    "rain",            "rain"),
    "karrigan":     ("429",     "karrigan",        "karrigan"),
    "frozen":       ("13586",   "frozen",          "frozen"),
    "degster":      ("17072",   "degster",         "degster"),
    "torzsi":       ("17376",   "torzsi",          "torzsi"),
    "idisbalance":  ("22434",   "idisbalance",     "iDISBALANCE"),
    "xant3r":       ("20838",   "xant3r",          "xant3r"),
    "jl":           ("17668",   "jl",              "jL"),
    "malbsmd":      ("21208",   "malbsmd",         "malbsMd"),
    "grim":         ("14119",   "grim",            "Grim"),
    "coldzera":     ("4344",    "coldzera",        "coldzera"),
    "fallen":       ("2023",    "fallen",          "FalleN"),
    "device":       ("1550",    "device",          "dev1ce"),
    "dupreeh":      ("1291",    "dupreeh",         "dupreeh"),
    "magisk":       ("9547",    "magisk",          "Magisk"),
}


def _normalise_player_key(name: str) -> str:
    """Normalise a player name to a cache key (lowercase, alphanumeric only)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Session-based fetcher for /stats/ pages (requires warm cookie session)
# ---------------------------------------------------------------------------

_STATS_SESSION: "_cffi_req.Session | None" = None
_STATS_SESSION_WARMED = False
_STATS_PAGES_BLOCKED = False  # circuit-breaker: stop trying after first confirmed 403


def _get_stats_session() -> "_cffi_req.Session | None":
    """
    Return the shared persistent HLTV session for stats page requests.
    Reuses _HLTV_SESSION so all requests share the same Cloudflare cookies.
    """
    return _get_hltv_session()


def _warm_stats_session(match_url: str) -> bool:
    """
    Warm the shared HLTV session for stats page access by visiting
    the match page (homepage is already done by _warm_hltv_session).
    Returns True when the session is ready.
    """
    global _STATS_SESSION_WARMED, _HLTV_SESSION_WARMED
    if _STATS_SESSION_WARMED:
        return True
    sess = _get_stats_session()
    if sess is None:
        return False
    # Ensure homepage warm-up has happened (sets _HLTV_SESSION_WARMED)
    _warm_hltv_session()
    try:
        r2 = sess.get(match_url, timeout=FETCH_TIMEOUT)
        logger.info(f"[stats_session] Warm-up match page: {r2.status_code} len={len(r2.text)}")
        if r2.status_code == 200:
            _STATS_SESSION_WARMED = True
            return True
    except Exception as e:
        logger.warning(f"[stats_session] Warm-up failed: {e}")
    return False


def _fetch_stats_page(stats_url: str, match_url: str) -> str | None:
    """
    Fetch an HLTV /stats/matches/mapstatsid/ page.

    Cycles through ALL available impersonation profiles before giving up.
    Each profile gets a full warm-up chain (homepage → match page) before
    the stats page is attempted, mimicking real browser navigation.

    Circuit-breaker only trips after every profile has been exhausted.
    """
    global _STATS_PAGES_BLOCKED, _STATS_SESSION_WARMED
    if _STATS_PAGES_BLOCKED:
        return None

    # Build the complete list of profiles to try (current first, then rest)
    all_profiles = list(_PROFILES)
    # Start from current profile index
    start_idx = _profile_idx
    ordered = all_profiles[start_idx:] + all_profiles[:start_idx]

    for profile_attempt, profile in enumerate(ordered):
        # Switch to this profile and build a fresh session
        global _HLTV_SESSION, _HLTV_SESSION_PROFILE, _HLTV_SESSION_WARMED
        if _HLTV_SESSION_PROFILE != profile or _HLTV_SESSION is None:
            _HLTV_SESSION = _make_session(profile)
            _HLTV_SESSION_PROFILE = profile
            _HLTV_SESSION_WARMED = False
            _STATS_SESSION_WARMED = False

        sess = _HLTV_SESSION
        if sess is None:
            continue

        try:
            # Step 1: Homepage warm-up (seeds Cloudflare cookie)
            if not _HLTV_SESSION_WARMED:
                r_home = sess.get(HLTV_BASE + "/", timeout=12)
                if r_home.status_code == 200 and "Just a moment" not in r_home.text:
                    _HLTV_SESSION_WARMED = True
                    logger.info(f"[stats_fetch] Homepage warm-up OK [{profile}]")
                    time.sleep(random.uniform(0.4, 0.9))
                else:
                    logger.warning(f"[stats_fetch] Homepage warm-up {r_home.status_code} [{profile}]")
                    # don't give up — still try the stats page

            # Step 2: Match page warm-up (builds referer chain)
            if not _STATS_SESSION_WARMED and match_url:
                r_match = sess.get(match_url, timeout=15)
                if r_match.status_code == 200:
                    _STATS_SESSION_WARMED = True
                    logger.info(f"[stats_fetch] Match-page warm-up OK [{profile}]")
                    time.sleep(random.uniform(0.3, 0.7))

            # Step 3: Fetch the actual stats page with browser-like headers
            logger.info(
                f"[stats_fetch] GET {stats_url} [{profile}]"
                + (f" (attempt {profile_attempt + 1}/{len(ordered)})" if profile_attempt else "")
            )
            resp = sess.get(
                stats_url,
                headers={
                    "Referer":          match_url or HLTV_BASE + "/",
                    "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language":  "en-US,en;q=0.9",
                    "Accept-Encoding":  "gzip, deflate, br",
                    "Sec-Fetch-Dest":   "document",
                    "Sec-Fetch-Mode":   "navigate",
                    "Sec-Fetch-Site":   "same-origin",
                    "Sec-Fetch-User":   "?1",
                    "Upgrade-Insecure-Requests": "1",
                },
                timeout=15,
            )

            if resp.status_code == 200 and len(resp.text) > 3000 and "Just a moment" not in resp.text:
                logger.info(f"[stats_fetch] OK — {len(resp.text):,} chars [{profile}]")
                return resp.text

            if resp.status_code == 403:
                logger.warning(
                    f"[stats_fetch] 403 on profile={profile} "
                    f"(attempt {profile_attempt + 1}/{len(ordered)}) — rotating"
                )
                _STATS_SESSION_WARMED = False
                # Exponential backoff: 1s, 2s, 3s, ...
                delay = min(1.0 + profile_attempt * 1.0, 4.0) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue

            logger.warning(
                f"[stats_fetch] Unexpected status={resp.status_code} "
                f"len={len(resp.text)} [{profile}]"
            )
            return None

        except Exception as e:
            logger.warning(f"[stats_fetch] {type(e).__name__}: {e} [{profile}]")
            time.sleep(1.0)
            continue

    # All profiles exhausted — trip the circuit-breaker for this session
    logger.warning(
        f"[stats_fetch] All {len(ordered)} profiles returned 403 — "
        "activating circuit-breaker. HS will use calibrated fallback rates."
    )
    _STATS_PAGES_BLOCKED = True
    return None


def _parse_map_stats_hs(
    html: str,
    player_slug: str,
    series_num: int = 0,
    map_num: int = 0,
) -> tuple[int | None, int | None]:
    """
    STRICT extraction of kills and headshots from a /stats/matches/mapstatsid/ page.

    Strict Map-by-Map Path (per user specification):
      1. Locate the stats table on the page (class="stats-table").
      2. Identify the column whose header contains "K" and "hs" — the "K (hs)" column.
      3. Find the row whose player name matches player_slug.
      4. Parse the cell using EXACTLY:
           headshots = int(raw_text.split('(')[1].split(')')[0])
           kills     = int(raw_text.split('(')[0].strip())
      5. If the "(hs)" format is absent from the cell — do NOT guess.
         Emit [ERROR] and return (None, None).

    Returns (kills, headshots) or (None, None) with explicit error logging.
    """
    soup = BeautifulSoup(html, 'html.parser')
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())
    ctx = f"Series {series_num} Map {map_num}"  # for audit / error messages

    # ── Step 1: Locate the stats table ────────────────────────────────────────
    # Priority: class="stats-table" → any table with "stats" in class → all tables
    stats_tables = (
        soup.find_all(class_='stats-table') or
        soup.find_all('table', class_=re.compile(r'stats', re.I)) or
        soup.find_all('table')
    )

    if not stats_tables:
        logger.error(
            f"[HS][{ctx}] No tables found on stats page for {player_slug!r}"
        )
        return None, None

    for tbl in stats_tables:
        all_rows = tbl.find_all('tr')
        if len(all_rows) < 2:
            continue

        header_row = all_rows[0]
        headers    = header_row.find_all(['th', 'td'])

        # ── Step 2: Identify the K (hs) column ────────────────────────────────
        khs_col = None
        for ci, hdr in enumerate(headers):
            ht = hdr.get_text(strip=True).lower()
            # Matches: "k (hs)", "k(hs)", "kills (hs)", "k / hs", etc.
            if re.search(r'k[^a-z]*\(?hs|hs.*\)', ht):
                khs_col = ci
                logger.debug(f"[HS][{ctx}] K(hs) column found at index {ci}, header={ht!r}")
                break

        # Fallback: if no labelled header, probe data rows for "N (N)" pattern
        if khs_col is None:
            all_headers_text = [h.get_text(strip=True) for h in headers]
            logger.debug(f"[HS][{ctx}] No K(hs) header in: {all_headers_text}")
            first_data = all_rows[1]
            data_cells = first_data.find_all('td')
            for probe in (4, 5, 3, 2):
                if probe < len(data_cells):
                    ct = data_cells[probe].get_text(strip=True)
                    if re.search(r'^\d+\s*\(\d+\)$', ct):
                        khs_col = probe
                        logger.debug(
                            f"[HS][{ctx}] K(hs) column probed at index {probe}: {ct!r}"
                        )
                        break

        if khs_col is None:
            logger.debug(f"[HS][{ctx}] K(hs) column not found in this table — trying next")
            continue

        # ── Step 3: Find the player's row ─────────────────────────────────────
        player_row_found = False
        for tr in all_rows[1:]:
            row_text = tr.get_text()
            row_norm = re.sub(r'[^a-z0-9]', '', row_text.lower())
            if slug_norm not in row_norm:
                continue

            player_row_found = True
            cells = tr.find_all('td')
            if khs_col >= len(cells):
                logger.error(
                    f"[HS][{ctx}] {player_slug!r} row found but K(hs) col {khs_col} "
                    f"out of range (row has {len(cells)} cells)"
                )
                return None, None

            raw_text = cells[khs_col].get_text(strip=True)
            logger.info(f"[HS][{ctx}] K(hs) cell for {player_slug!r}: {raw_text!r}")

            # ── Step 4: Strict parse — "21 (11)" format ONLY ──────────────────
            if '(' not in raw_text or ')' not in raw_text:
                # The (hs) format is missing — do NOT guess
                logger.error(
                    f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
                    f"— cell was {raw_text!r}, expected format '21 (11)'"
                )
                print(
                    f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
                    f"(player={player_slug}, cell={raw_text!r})"
                )
                return None, None

            try:
                # Exact user-specified extraction:
                #   headshots = int(raw_text.split('(')[1].split(')')[0])
                #   kills     = int(raw_text.split('(')[0].strip())
                headshots = int(raw_text.split('(')[1].split(')')[0].strip())
                kills_str = raw_text.split('(')[0].strip()
                kills_m   = re.search(r'(\d+)', kills_str)
                kills_hs  = int(kills_m.group(1)) if kills_m else None

                logger.info(
                    f"[HS][{ctx}] Parsed — kills={kills_hs}  headshots={headshots}"
                )
                return kills_hs, headshots

            except (IndexError, ValueError) as e:
                logger.error(
                    f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
                    f"— parse error on {raw_text!r}: {e}"
                )
                print(
                    f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
                    f"(player={player_slug}, cell={raw_text!r}, err={e})"
                )
                return None, None

        if not player_row_found:
            logger.warning(
                f"[HS][{ctx}] Player {player_slug!r} (norm={slug_norm!r}) not found "
                f"in any data row of this table"
            )

    logger.error(
        f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
        f"— player {player_slug!r} not found or no K(hs) column on stats page"
    )
    print(
        f"[ERROR] Missing HS data for Map {map_num} of Series {series_num} "
        f"(player={player_slug})"
    )
    return None, None


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
    """
    Improved player search with cache-first lookup.

    Order:
      1. Check _PLAYER_ID_CACHE (instant — no HTTP).
      2. Fall through to HLTV /search?query={name} with retry.
      3. On success, populate cache so future lookups skip the search.
    """
    key = _normalise_player_key(name)

    # 1. Cache hit — skip search entirely
    if key in _PLAYER_ID_CACHE:
        pid, slug, display = _PLAYER_ID_CACHE[key]
        logger.info(f"[search] Cache hit: {display} (id={pid}, slug={slug})")
        return pid, slug, display

    # 2. Live HLTV search
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

    # 3. Populate cache for future lookups
    _PLAYER_ID_CACHE[key] = (best_pid, best_slug, display)
    slug_key = _normalise_player_key(best_slug)
    if slug_key != key:
        _PLAYER_ID_CACHE[slug_key] = (best_pid, best_slug, display)
    logger.info(f"[search] Cached {name!r} → {display} (id={best_pid})")

    return best_pid, best_slug, display


# ---------------------------------------------------------------------------
# Step 2 — Get player's recent BO3 match IDs from the results page
# ---------------------------------------------------------------------------

# Match ID cache: player_id → (fetched_at_timestamp, [(match_id, slug), ...])
# TTL: 3 hours — matches don't change; player gets new results every few days.
_MATCH_IDS_CACHE: dict[str, tuple[float, list]] = {}
_MATCH_IDS_CACHE_TTL = 3 * 3600  # 3 hours


def get_player_match_ids(player_id: str, max_matches: int = 25) -> list[tuple[str, str]]:
    """
    Fetch /results?player={id} and return a list of (match_id, slug) tuples
    for recently completed matches. Only returns large IDs (7+ digits) which
    correspond to accessible HLTV match pages.

    Results are cached for 3 hours — if HLTV returns 403 on a repeat query,
    we serve the last successful fetch rather than falling to estimated data.
    """
    now = time.time()

    # Serve from cache if fresh
    if player_id in _MATCH_IDS_CACHE:
        cached_at, cached_ids = _MATCH_IDS_CACHE[player_id]
        age_min = round((now - cached_at) / 60)
        if (now - cached_at) < _MATCH_IDS_CACHE_TTL:
            logger.info(
                f"[results] Cache hit for player {player_id}: "
                f"{len(cached_ids)} match IDs (age {age_min}min)"
            )
            return cached_ids
        logger.info(f"[results] Cache stale for player {player_id} ({age_min}min) — refreshing")

    url = f"{HLTV_BASE}/results?player={player_id}"
    html = _fetch(url)
    if not html:
        # Return stale cache rather than nothing
        if player_id in _MATCH_IDS_CACHE:
            _, stale = _MATCH_IDS_CACHE[player_id]
            logger.warning(
                f"[results] Live fetch failed — serving stale cache "
                f"({len(stale)} IDs) for player {player_id}"
            )
            return stale
        return []

    # Find all match links on the page — only large IDs work (small IDs return 500)
    all_matches = re.findall(r'/matches/(\d+)/([a-z0-9-]+)', html)
    seen = {}
    for mid, slug in all_matches:
        if mid not in seen and len(mid) >= 6:
            seen[mid] = slug

    results = list(seen.items())[:max_matches]
    logger.info(f"[results] Found {len(results)} match IDs for player {player_id}")

    # Store in cache
    _MATCH_IDS_CACHE[player_id] = (now, results)
    return results


# ---------------------------------------------------------------------------
# Step 3 — Parse a match page for per-map kills
# ---------------------------------------------------------------------------

def _parse_match_kills(html: str, player_slug: str, match_url: str = "", series_num: int = 0) -> dict:
    """
    Parse an HLTV match page and return:
      {
        'bo_type': 3,          # or 1/2
        'maps': [
          {'map_name': 'Dust2', 'kills': 22, 'deaths': 14, 'headshots': 4, 'map_number': 1},
          {'map_name': 'Inferno', 'kills': 19, 'deaths': 17, 'headshots': None, 'map_number': 2},
          ...
        ]
      }
    Returns None if the player isn't found or the match page has no stats.

    headshots is populated from the /stats/matches/mapstatsid/ page K(hs) column
    when accessible, otherwise None (callers fall back to calibrated HS rates).
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Determine BO type from score (e.g. Vitality 2 - NaVi 1)
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
    seen_ids = []
    for tid in tab_ids:
        if tid not in seen_ids:
            seen_ids.append(tid)
    map_ids = seen_ids  # ordered by map number

    logger.info(f"[parse] Maps found: {map_ids} | BO type: {bo_type}")

    # ── Extract mapstatsid URLs for each map (for Strategy 0 — K(hs) column) ──
    # HLTV embeds links like /stats/matches/mapstatsid/224728/state-vs-bebop
    # in the match page HTML.  We index them in the same order as map_ids.
    _raw_mapstat_links = re.findall(
        r'/stats/matches/mapstatsid/(\d+)/([\w-]+)', html
    )
    # Deduplicate preserving order (each mapstatsid appears once)
    _seen_msid: dict[str, str] = {}
    for msid, msslug in _raw_mapstat_links:
        if msid not in _seen_msid:
            _seen_msid[msid] = msslug
    # Build ordered list aligned to map_ids (best-effort; same count usually)
    _mapstat_urls: list[str] = []
    for msid, msslug in _seen_msid.items():
        _mapstat_urls.append(f"{HLTV_BASE}/stats/matches/mapstatsid/{msid}/{msslug}")
    logger.info(f"[parse] Map stats URLs: {_mapstat_urls}")

    # Get map names from the tab labels
    map_names = {}
    for div in matchstats.find_all(class_=re.compile(r'dynamic-map-name-full', re.I)):
        div_id = div.get('id', '')
        if div_id and div_id != 'all':
            map_names[div_id] = div.get_text(strip=True)

    # Normalise player slug for matching (lowercase, no hyphens)
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())

    # Pre-compute actual round counts per map using mapholder divs.
    # Each mapholder div that has a played result contains:
    #   • A STATS link with href="/stats/matches/mapstatsid/{mid}/..." (gives map_id)
    #   • Two <div class="results-team-score"> elements (one per team) with integer scores
    # Total rounds = score_team_A + score_team_B.
    _map_rounds: dict[str, int] = {}
    for _mh in soup.find_all('div', class_='mapholder'):
        # Get mapstatsid from the STATS href
        _stats_a = _mh.find('a', href=re.compile(r'mapstatsid/(\d+)', re.I))
        if not _stats_a:
            continue
        _mid_m = re.search(r'mapstatsid/(\d+)', _stats_a.get('href', ''))
        if not _mid_m:
            continue
        _mid = _mid_m.group(1)
        # Get team scores
        _score_els = _mh.find_all('div', class_='results-team-score')
        _scores = []
        for _se in _score_els:
            _t = _se.get_text(strip=True)
            if re.match(r'^\d{1,2}$', _t):
                _v = int(_t)
                if 0 <= _v <= 35:
                    _scores.append(_v)
        if len(_scores) >= 2:
            _total = _scores[0] + _scores[1]
            if 13 <= _total <= 60:   # CS2: min 13+0=13, max OT games ~48
                _map_rounds[_mid] = _total
    logger.info(f"[parse] Per-map round counts from mapholders: {_map_rounds}")

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
            _tbl_classes = [str(t.get('class', '')) for t in content_div.find_all('table')]
            logger.info(f"[hs_locate] Table classes in content_div: {_tbl_classes}")

        # ── Extract kills, headshots, deaths ──────────────────────────────────
        #
        # Strategy 0 — /stats/matches/mapstatsid/ page (PRIMARY, highest fidelity)
        #   The dedicated map stats page contains a .stats-table with a "K (hs)"
        #   column formatted as "15 (4)" — Total Kills (Headshots).  We extract:
        #     headshots = text.split('(')[1].replace(')', '')
        #   This is JavaScript-rendered on the match page but available as static
        #   HTML on the mapstatsid sub-page (requires a warmed cookie session).
        #
        # Strategy A — match page Detailed-stats table — K (hs) header in HTML
        #   (present on some older HLTV page versions)
        #
        # Strategy B — match page totalstats table — K-D combined column
        #   HLTV's current format: columns are K-D | eK-eD | Swing | ADR | ...
        #   First number in "K-D" cell is total kills.
        #
        # Strategy C — Regex scan on any row containing the player — last resort.

        headshots  = None
        kills      = None
        deaths     = None
        player_row = None

        # ─ Strategy 0: Fetch per-map stats page and parse K(hs) column ─────────
        # Maps are zero-indexed here: map_num 1 → index 0, map_num 2 → index 1
        _stats_url = _mapstat_urls[map_num - 1] if map_num - 1 < len(_mapstat_urls) else None
        if _stats_url and match_url:
            _stats_html = _fetch_stats_page(_stats_url, match_url)
            if _stats_html:
                _sk, _shs = _parse_map_stats_hs(
                    _stats_html, player_slug,
                    series_num=series_num, map_num=map_num,
                )
                if _sk is not None:
                    kills     = _sk
                    headshots = _shs
                    logger.info(
                        f"[parse_row] Strategy0 K(hs) map{map_num}: "
                        f"{kills}K {headshots}HS (from stats page)"
                    )

        # ─ Strategy A: Detailed-stats table (K (hs) header in match page HTML) ─
        if kills is None:
            for table in content_div.find_all('table'):
                first_tr = table.find('tr')
                if not first_tr:
                    continue
                header_cells = first_tr.find_all(['th', 'td'])

                k_hs_col = None
                d_col    = None
                for ci, hc in enumerate(header_cells):
                    ht = re.sub(r'\s+', '', hc.get_text().lower())
                    if k_hs_col is None and 'k' in ht and ('hs' in ht or 'head' in ht):
                        k_hs_col = ci
                    if d_col is None and 'd' in ht and 't' in ht:
                        d_col = ci

                if k_hs_col is None:
                    continue

                for tr in table.find_all('tr')[1:]:
                    row_norm = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
                    if slug_norm not in row_norm:
                        continue
                    cells_td = tr.find_all('td')
                    if k_hs_col < len(cells_td):
                        ct = cells_td[k_hs_col].get_text(strip=True)
                        m = re.search(r'(\d+)\s*\((\d+)\)', ct)
                        if m:
                            kills     = int(m.group(1))
                            headshots = int(m.group(2))
                            player_row = tr
                            if d_col is not None and d_col < len(cells_td):
                                dm = re.search(r'(\d+)', cells_td[d_col].get_text(strip=True))
                                if dm:
                                    deaths = int(dm.group(1))
                            logger.info(
                                f"[parse_row] StrategyA K(hs) map{map_num}: "
                                f"{kills}K {headshots}HS D={deaths}"
                            )
                            break
                if kills is not None:
                    break

        # ─ Strategy B: totalstats table — K-D combined column ─────────────────
        # HLTV current format: columns are "K-D" | "eK-eD" | "Swing" | ...
        # The first number in the K-D cell is total kills.
        # Also handles legacy pages with separate "K" and "D" columns.
        if kills is None:
            for table in content_div.find_all('table'):
                first_tr = table.find('tr')
                if not first_tr:
                    continue
                header_cells = first_tr.find_all(['th', 'td'])
                k_col = None
                d_col = None
                for ci, hc in enumerate(header_cells):
                    raw = hc.get_text(strip=True)
                    ht  = raw.upper().strip()
                    # "K-D" combined column — kills are first number
                    if k_col is None and re.match(r'^K[-–]D$', ht):
                        k_col = ci
                    # Legacy plain "K" column
                    if k_col is None and ht == 'K':
                        k_col = ci
                    # Legacy plain "D" column
                    if d_col is None and ht == 'D':
                        d_col = ci

                if k_col is None:
                    continue

                for tr in table.find_all('tr')[1:]:
                    row_norm = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
                    if slug_norm not in row_norm:
                        continue
                    cells_td = tr.find_all('td')
                    if k_col < len(cells_td):
                        cell_text = cells_td[k_col].get_text(strip=True)
                        # "15-14" → kills=15, deaths=14 (K-D combined)
                        kd_m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', cell_text)
                        if kd_m:
                            kills  = int(kd_m.group(1))
                            deaths = int(kd_m.group(2))
                            player_row = tr
                        else:
                            # Legacy: plain integer
                            km = re.search(r'(\d+)', cell_text)
                            if km:
                                kills = int(km.group(1))
                                player_row = tr
                            if d_col is not None and d_col < len(cells_td):
                                dm = re.search(r'(\d+)', cells_td[d_col].get_text(strip=True))
                                if dm:
                                    deaths = int(dm.group(1))
                        if kills:
                            logger.info(
                                f"[parse_row] StrategyB K-D map{map_num}: {kills}K D={deaths}"
                            )
                        break
                if kills is not None:
                    break

        # ─ Strategy C: Regex scan (per-half fallback) ─────────────────────────
        if kills is None:
            candidate_rows = []
            for tr in content_div.find_all('tr'):
                rn = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
                if slug_norm in rn:
                    candidate_rows.append(tr)
            if not candidate_rows:
                short = slug_norm[:4]
                for tr in content_div.find_all('tr'):
                    rn = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
                    if len(short) >= 3 and short in rn:
                        candidate_rows.append(tr)
                        break
            for pr in candidate_rows:
                pr_text = pr.get_text()
                kd = re.search(r'(\d+)\s*[-–]\s*(\d+)', pr_text)
                if kd:
                    kills  = int(kd.group(1))
                    deaths = int(kd.group(2))
                    player_row = pr
                    logger.info(
                        f"[parse_row] Regex-fallback K-D map{map_num}: {kills}K {deaths}D"
                    )
                    break

        if kills is None:
            logger.debug(f"[parse] Player '{player_slug}' not found on map {map_num} ({map_name})")
            continue

        # If Strategy 0 found kills/headshots via stats page but left player_row=None,
        # do a best-effort search in the content_div's totalstats table so we can
        # still extract deaths, rating, KAST, and ADR from the match page.
        if player_row is None:
            for _tbl in content_div.find_all('table', class_=re.compile(r'totalstats', re.I)):
                for _tr in _tbl.find_all('tr')[1:]:
                    _rn = re.sub(r'[^a-z0-9]', '', _tr.get_text().lower())
                    if slug_norm in _rn:
                        player_row = _tr
                        # Try to extract deaths from K-D cell
                        if deaths is None:
                            for _td in _tr.find_all('td'):
                                _kd = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', _td.get_text(strip=True))
                                if _kd:
                                    deaths = int(_kd.group(2))
                                    break
                        break
                if player_row:
                    break

        row_text = player_row.get_text() if player_row else ""
        logger.info(f"[parse_row] Map {map_num} ({map_name}) row: {row_text[:200]!r}")

        # Guard: cells is empty list if player_row couldn't be found
        cells = player_row.find_all('td') if player_row else []

        # Extract Rating 2.0 — it's a decimal like 1.15 in [0.40, 3.00]
        # found in td cells, typically the rightmost decimal value
        rating = None
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

        # Use pre-computed round counts from the match page score elements.
        # Falls back to 24 (CS2 regulation max) if score couldn't be parsed.
        rounds_on_map = _map_rounds.get(map_id, 24)
        _deaths_for_sr = deaths if deaths is not None else 0
        survival_rate = round((rounds_on_map - _deaths_for_sr) / rounds_on_map, 3)

        # Extract FK/FD from the eK-eD column (2nd <td> in the HLTV CS2 row).
        # Column layout confirmed: K-D | eK-eD | Swing | ADR | eADR | KAST | eKAST | Rating
        # The "16-7" in the eK-eD cell gives entry kills (16) and entry deaths (7).
        fk = None
        fd = None
        if player_row is not None:
            _row_cells = player_row.find_all('td')
            # Find the data columns: skip the player name cell (which usually has no K-D pattern)
            _data_cells = [c for c in _row_cells if re.match(r'^\d+[-–]\d+$', c.get_text(strip=True))]
            if len(_data_cells) >= 2:
                # _data_cells[0] = K-D, _data_cells[1] = eK-eD
                _ekd_txt = _data_cells[1].get_text(strip=True)
                _ekd_m = re.match(r'^(\d+)[-–](\d+)$', _ekd_txt)
                if _ekd_m:
                    fk = int(_ekd_m.group(1))
                    fd = int(_ekd_m.group(2))

        maps_result.append({
            'map_name':      map_name,
            'kills':         kills,
            'headshots':     headshots,   # int or None if not shown on scorecard
            'deaths':        deaths,
            'rounds':        rounds_on_map,
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
            f"{kills}K{_hs_str}/{deaths}D rounds={rounds_on_map} rating={rating} fk={fk}"
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
# Period stats pages  (/stats/players/{id}/{slug}  and  /stats/teams/{id}/{slug})
# These endpoints accept startDate / endDate query params and return aggregated
# stats (KPR, HS%, Rating 2.0, KAST, ADR, etc.) for that date window.
# We use the same main session (_HLTV_SESSION) — 403s are caught and logged.
# ---------------------------------------------------------------------------

def _store_stat_val(result: dict, label: str, value_text: str) -> None:
    """Map one label→value pair from a stats page row into `result`."""
    label = label.lower().strip()
    vt    = value_text.strip()

    def _flt(s, lo=None, hi=None):
        m = re.search(r'(\d+\.?\d*)', s)
        if not m:
            return None
        v = float(m.group(1))
        if lo is not None and v < lo:
            return None
        if hi is not None and v > hi:
            return None
        return v

    if 'kills / round' in label or label == 'kpr':
        v = _flt(vt, 0.1, 5.0)
        if v:
            result['kpr'] = v
    elif any(x in label for x in ('headshots %', 'hs %', 'headshot %', 'hs%')):
        v = _flt(vt.replace('%', ''), 5.0, 95.0)
        if v:
            result['hs_pct'] = v
    elif any(x in label for x in ('damage / round', 'adr', 'damage/round')):
        v = _flt(vt, 10.0, 250.0)
        if v:
            result['adr'] = v
    elif 'rating 2' in label or (label == 'rating'):
        v = _flt(vt, 0.2, 5.0)
        if v:
            result['rating'] = v
    elif label == 'kast':
        v = _flt(vt.replace('%', ''), 10.0, 100.0)
        if v:
            result['kast'] = v
    elif any(x in label for x in ('k/d ratio', 'k/d')):
        v = _flt(vt, 0.1, 20.0)
        if v:
            result['kd'] = v
    elif 'total kills' in label:
        m = re.search(r'(\d+)', vt)
        if m:
            result['kills'] = int(m.group(1))
    elif 'rounds played' in label:
        m = re.search(r'(\d+)', vt)
        if m:
            result['rounds'] = int(m.group(1))
    elif 'maps played' in label:
        m = re.search(r'(\d+)', vt)
        if m:
            result['maps'] = int(m.group(1))
    elif any(x in label for x in ('win rate', 'w/l')):
        v = _flt(vt.replace('%', ''), 0.0, 100.0)
        if v:
            result['win_rate'] = v
    elif any(x in label for x in ('deaths / round', 'dpr', 'deaths/round')):
        v = _flt(vt, 0.1, 5.0)
        if v:
            result['dpr'] = v


def _parse_stats_page(html: str, slug: str) -> dict:
    """
    Parse a /stats/players/ or /stats/teams/ HLTV page.

    HLTV stats pages use a two-column grid of stat rows:
      <div class="stats-row">
        <span class="stats-row-first">Kills / round</span>
        <span class="bold">0.83</span>
      </div>
    We also fall back to raw-text regex scanning for resilience.
    """
    result: dict = {}
    soup = BeautifulSoup(html, 'html.parser')

    # Strategy 1: stats-row div pattern
    for row in soup.find_all('div', class_='stats-row'):
        spans = row.find_all('span')
        if len(spans) >= 2:
            label_text = spans[0].get_text(strip=True)
            value_text = spans[-1].get_text(strip=True)
            _store_stat_val(result, label_text, value_text)

    # Strategy 2: generic label → next sibling with a numeric value
    # Covers pages that use <p>, <td>, or other elements
    if not result:
        for tag in soup.find_all(['p', 'td', 'div', 'span']):
            label_text = tag.get_text(strip=True)
            if len(label_text) > 50:
                continue
            sibling = tag.find_next_sibling()
            if sibling:
                _store_stat_val(result, label_text, sibling.get_text(strip=True))

    # Strategy 3: raw-text regex scan — last resort
    raw = html.lower()
    _RAW_PATTERNS = [
        ('kpr',    r'kills\s*/\s*round[^<]{0,80}?(\b0\.\d{2,3}\b)'),
        ('hs_pct', r'(?:headshots?\s*%|hs\s*%)[^<]{0,60}?(\d{1,2}\.?\d*)\s*%'),
        ('adr',    r'(?:damage\s*/\s*round|adr)[^<]{0,60}?(\d{2,3}\.?\d*)(?!\s*%)'),
        ('rating', r'rating\s*2\.0[^<]{0,60}?(\d\.\d{2,3})'),
        ('kast',   r'\bkast\b[^<]{0,60}?(\d{2}\.?\d*)\s*%'),
        ('kd',     r'k/d\s*ratio[^<]{0,60}?(\d+\.\d{2})'),
    ]
    for field, pat in _RAW_PATTERNS:
        if field not in result:
            m = re.search(pat, raw)
            if m:
                try:
                    result[field] = float(m.group(1))
                except ValueError:
                    pass

    logger.info(f"[period_stats] parsed slug={slug}: {result}")
    return result


def get_player_period_stats(player_id: str, player_slug: str, days: int = 90) -> dict | None:
    """
    Fetch a player's aggregated HLTV stats for the past `days` days.

    URL: /stats/players/{id}/{slug}?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD

    Returns dict with any of: kpr, hs_pct, rating, kast, adr, kd, kills, rounds
    or None if the page is unavailable.
    """
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)
    url = (
        f"{HLTV_BASE}/stats/players/{player_id}/{player_slug}"
        f"?startDate={start_dt.isoformat()}&endDate={end_dt.isoformat()}"
    )
    logger.info(f"[period_stats] GET player stats: {url}")
    html = _fetch(url)
    if not html:
        logger.warning(f"[period_stats] Could not fetch player stats for {player_slug}")
        return None

    parsed = _parse_stats_page(html, player_slug)
    if not parsed:
        logger.warning(f"[period_stats] No stats parsed for {player_slug}")
        return None

    parsed['url']  = url
    parsed['days'] = days
    return parsed


def get_team_period_stats(team_id: str, team_slug: str, days: int = 90) -> dict | None:
    """
    Fetch a team's aggregated HLTV stats for the past `days` days.

    URL: /stats/teams/{id}/{slug}?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD

    Returns dict with any of: kpr, rating, kast, adr, kd, maps, win_rate
    or None if the page is unavailable.
    """
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)
    url = (
        f"{HLTV_BASE}/stats/teams/{team_id}/{team_slug}"
        f"?startDate={start_dt.isoformat()}&endDate={end_dt.isoformat()}"
    )
    logger.info(f"[period_stats] GET team stats: {url}")
    html = _fetch(url)
    if not html:
        logger.warning(f"[period_stats] Could not fetch team stats for {team_slug}")
        return None

    parsed = _parse_stats_page(html, team_slug)
    if not parsed:
        logger.warning(f"[period_stats] No stats parsed for team {team_slug}")
        return None

    parsed['url']  = url
    parsed['days'] = days
    return parsed


# ---------------------------------------------------------------------------
# Step 4 — Get HS kills specifically (from per-round headshot data if available)
# ---------------------------------------------------------------------------

def _parse_match_hs_pct(html: str, player_slug: str) -> float | None:
    """
    Extract the player's HS% from a match page's ALL-MAPS overview table only.

    HLTV match pages have two stat views per map:
      1) A per-map/per-half breakdown (CT K-D / T K-D / CT ADR / T ADR / CT KAST / T KAST)
      2) An all-maps combined overview in id="all-content" (K(HS)-D / ADR / KAST / HS% / Rating)

    We specifically target the all-maps section to avoid misreading KAST% from per-half rows.
    Strategies (in priority order):
      A) The all-content K-D cell shows "kills(HS)-deaths" → HS% = HS/kills (most accurate)
      B) The all-content row has percentage cells → last in [10,70] range is HS%
    Returns a float in [0.0, 1.0] or None if not found.
    """
    soup = BeautifulSoup(html, 'html.parser')
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())

    def _row_pcts(row) -> list[int]:
        vals = []
        for cell in row.find_all(['td', 'th']):
            ct = cell.get_text(strip=True)
            m = re.match(r'^(\d{1,3})%$', ct)
            if m:
                vals.append(int(m.group(1)))
        return vals

    # ── Step 1: look in the all-maps overview section (id="all-content") ────────
    # HLTV's all-maps detailed stats table has the same columns as per-map:
    #   Op K-D | MKs | KAST | 1vsX | K (hs) | A (f) | D (t) | ADR | Swing | Rating
    # The "K (hs)" cell = "53 (19)" → 53 total kills, 19 headshots across all maps.
    # HS% = HS / kills (then applied per-map as an estimate).
    all_content = soup.find(id='all-content')
    search_scope = all_content if all_content else soup  # fallback to full page

    # Strategy A: find the table with "K (hs)" column in the all-content section
    for table in search_scope.find_all('table'):
        first_tr = table.find('tr')
        if not first_tr:
            continue
        header_cells = first_tr.find_all(['th', 'td'])
        k_hs_col = None
        for ci, hc in enumerate(header_cells):
            ht = re.sub(r'\s+', '', hc.get_text().lower())
            if 'k' in ht and ('hs' in ht or 'head' in ht):
                k_hs_col = ci
                break
        if k_hs_col is None:
            continue

        for tr in table.find_all('tr')[1:]:
            row_norm = re.sub(r'[^a-z0-9]', '', tr.get_text().lower())
            if slug_norm not in row_norm:
                continue
            cells_td = tr.find_all('td')
            if k_hs_col < len(cells_td):
                ct = cells_td[k_hs_col].get_text(strip=True)
                m = re.search(r'(\d+)\s*\((\d+)\)', ct)
                if m:
                    kills = int(m.group(1))
                    hs    = int(m.group(2))
                    if kills > 0:
                        rate = round(hs / kills, 3)
                        logger.info(
                            f"[hs_pct] all-content K(hs) for {player_slug}: "
                            f"{kills}K {hs}HS → {round(rate*100, 1)}% "
                            f"({'all-content' if all_content else 'full-page'})"
                        )
                        return rate

    # Strategy B: percentage columns in any player row of the all-content scope.
    # KAST (50-100) is listed before HS (10-70) in HLTV columns.
    # The LAST percentage in 10-70% range should be HS%.
    for row in search_scope.find_all('tr'):
        row_norm = re.sub(r'[^a-z0-9]', '', row.get_text().lower())
        if slug_norm not in row_norm:
            continue
        pcts = _row_pcts(row)
        if not pcts:
            continue
        hs_candidates = [p for p in pcts if 10 <= p <= 70]
        if hs_candidates:
            val = hs_candidates[-1]
            logger.info(
                f"[hs_pct] pct-scan for {player_slug}: pcts={pcts} → HS≈{val}%"
                + (" (all-content)" if all_content else " (full-page fallback)")
            )
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

        # series_num is 1-indexed before increment (next series will be bo3_series_count+1)
        _this_series = bo3_series_count + 1
        parsed = _parse_match_kills(html, player_slug, match_url, series_num=_this_series)
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
        _series_maps = maps[:2]
        _audit_parts = []
        for m in _series_maps:
            map_num = m.get('map_number', 1)
            _hs_val = m.get('headshots')
            map_kills.append({
                'stat_value':    m['kills'],
                'headshots':     _hs_val,             # actual HS count or None
                'match_hs_pct':  match_hs,            # per-match scraped HS% (all-maps avg) or None
                'rounds':        m.get('rounds', 24),  # actual rounds from score parsing; 24 = CS2 regulation max
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
            # Build the audit line for this map
            _hs_display = str(_hs_val) if _hs_val is not None else "MISSING"
            _audit_parts.append(
                f"Map{map_num}({m['map_name'].title()}): {m['kills']}K / {_hs_display}HS"
            )

        bo3_series_count += 1

        # ── Step 3 Audit: Print total HS per series to console ────────────────
        _series_hs = [
            m.get('headshots') for m in _series_maps if m.get('headshots') is not None
        ]
        _total_hs = sum(_series_hs) if _series_hs else None
        _hs_total_str = str(_total_hs) if _total_hs is not None else "MISSING"
        _audit_line = (
            f"[AUDIT] Series {bo3_series_count} (match {match_id}): "
            + " | ".join(_audit_parts)
            + f" | Total HS (Map1+Map2): {_hs_total_str}"
        )
        print(_audit_line)
        logger.info(_audit_line)

        logger.info(
            f"[scraper] Series {bo3_series_count}: match {match_id} — "
            f"maps: {[(m['map_name'], m['kills']) for m in _series_maps]}"
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
    # MIBR family
    "mibr": "mibr",
    "mibr academy": "mibr-academy",
    "mibraca": "mibr-academy",
    "mibr-academy": "mibr-academy",
    # Other common shorthands
    "furia": "furia",
    "imperial": "imperial",
    "w7m": "w7m-esports",
    "w7mesports": "w7m-esports",
}

_SECONDARY_MARKERS = ('junior', 'academy', 'youth', '-2', '-b-team', 'b-team', 'female', 'women')


def _score_team_candidates(
    candidates: dict[str, str],
    name_norm: str,
    raw_query: str = "",
) -> tuple[str | None, str | None, int]:
    """
    Score a {team_id: slug} dict against a normalised target name.
    Returns (best_tid, best_slug, best_score).

    Secondary-team markers (academy, junior, youth …) are penalised only
    when the user's query does NOT contain that marker.  If the user typed
    "mibr academy" they explicitly want the academy squad, so no penalty.
    """
    raw_lower = raw_query.lower()
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
        # Only penalise secondary-team markers when the user did NOT ask for them
        for marker in _SECONDARY_MARKERS:
            marker_clean = marker.lstrip('-')   # "-b-team" → "b-team" for the check
            if marker in slug.lower() and marker_clean not in raw_lower:
                score -= 120
                break
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
    best_tid, best_slug, best_score = _score_team_candidates(seen, name_norm, raw_query=name)

    # If no meaningful match, try again with the raw user input as the query
    if best_score <= 0 and query != name:
        seen2 = _search_query(name)
        if seen2:
            raw_norm = re.sub(r'[^a-z0-9]', '', name.lower())
            t2, s2, sc2 = _score_team_candidates(seen2, raw_norm, raw_query=name)
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
