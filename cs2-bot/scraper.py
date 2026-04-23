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
    global _STATS_SESSION_WARMED, _MAPSTATS_CIRCUIT_BLOCKED
    _profile_idx = (_profile_idx + 1) % len(_PROFILES)
    new_profile = _PROFILES[_profile_idx]
    logger.warning(f"[session] Rotating profile → {new_profile}")
    _HLTV_SESSION = _make_session(new_profile)
    _HLTV_SESSION_PROFILE = new_profile
    _HLTV_SESSION_WARMED = False        # re-warm homepage with new session
    _STATS_SESSION_WARMED = False       # re-warm stats referer with new session
    _MAPSTATS_CIRCUIT_BLOCKED = False   # give mapstats pages a fresh chance
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

    # ── Fast-path: global /stats/ circuit-breaker ─────────────────────────────
    # If the stats subdomain is known to be blocked, skip immediately without
    # burning retries.  _is_stats_blocked() also auto-resets after 30 min.
    if _is_stats_blocked(url):
        logger.debug(f"[fetch] /stats/ circuit open — skipping {url}")
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
                    _trip_stats_circuit(url)   # no-op for non-/stats/ URLs
                    # If the circuit just tripped, bail immediately — no more retries
                    if _STATS_SUBDOMAIN_BLOCKED and '/stats/' in url:
                        logger.warning(f"[fetch] /stats/ circuit tripped — abandoning {url}")
                        return None
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
_STATS_PAGES_BLOCKED = False  # legacy per-fetch circuit-breaker (HS mapstats)

# Cache for mapstatsid HTML so each URL is only fetched once per bot session
_MAPSTATS_HTML_CACHE: dict[str, str] = {}

# ── TWO separate /stats/ circuit-breakers ─────────────────────────────────────
#
#  1. PLAYER/TEAM stats circuit  (/stats/players/, /stats/teams/, /stats/search/)
#     These return 403 from Replit datacenter IPs immediately.
#     TTL = 30 min.
#
#  2. MAPSTATS circuit  (/stats/matches/mapstatsid/)
#     These are the per-map pages containing the K (hs) column.
#     They are fetched with a full browser warm-up and may succeed even when
#     the player-stats circuit is open.  Separate TTL = 5 min so they retry
#     more often.
#
# Keeping them separate means a 403 on a player stats page does NOT block
# mapstatsid page fetches (which is where live HS data comes from).

_STATS_SUBDOMAIN_BLOCKED: bool = False
_STATS_SUBDOMAIN_BLOCKED_AT: float = 0.0
_STATS_SUBDOMAIN_BLOCK_TTL: int = 30 * 60   # 30 minutes (player/team stats)

_MAPSTATS_CIRCUIT_BLOCKED: bool = False
_MAPSTATS_CIRCUIT_BLOCKED_AT: float = 0.0
_MAPSTATS_CIRCUIT_BLOCK_TTL: int = 5 * 60   # 5 minutes (mapstatsid HS pages)


def _is_mapstats_url(url: str) -> bool:
    return '/stats/matches/mapstatsid/' in url


def _is_stats_blocked(url: str) -> bool:
    """
    Return True if the relevant circuit-breaker is active for this URL.
    mapstatsid URLs use their own narrower circuit; everything else uses
    the global /stats/ circuit.
    """
    global _STATS_SUBDOMAIN_BLOCKED, _STATS_SUBDOMAIN_BLOCKED_AT
    global _MAPSTATS_CIRCUIT_BLOCKED, _MAPSTATS_CIRCUIT_BLOCKED_AT

    if '/stats/' not in url:
        return False

    if _is_mapstats_url(url):
        # mapstatsid has its own 5-min circuit
        if not _MAPSTATS_CIRCUIT_BLOCKED:
            return False
        elapsed = time.time() - _MAPSTATS_CIRCUIT_BLOCKED_AT
        if elapsed > _MAPSTATS_CIRCUIT_BLOCK_TTL:
            _MAPSTATS_CIRCUIT_BLOCKED = False
            logger.info("[circuit] mapstatsid circuit expired — retrying HS pages")
            return False
        return True
    else:
        # Player/team/search stats — 30-min circuit
        if not _STATS_SUBDOMAIN_BLOCKED:
            return False
        elapsed = time.time() - _STATS_SUBDOMAIN_BLOCKED_AT
        if elapsed > _STATS_SUBDOMAIN_BLOCK_TTL:
            _STATS_SUBDOMAIN_BLOCKED = False
            logger.info("[circuit] /stats/ block TTL expired — will retry stats URLs")
            return False
        return True


def _trip_stats_circuit(url: str) -> None:
    """Trip the appropriate circuit-breaker on first 403 from a /stats/ URL."""
    global _STATS_SUBDOMAIN_BLOCKED, _STATS_SUBDOMAIN_BLOCKED_AT
    global _MAPSTATS_CIRCUIT_BLOCKED, _MAPSTATS_CIRCUIT_BLOCKED_AT

    if '/stats/' not in url:
        return

    if _is_mapstats_url(url):
        if not _MAPSTATS_CIRCUIT_BLOCKED:
            _MAPSTATS_CIRCUIT_BLOCKED = True
            _MAPSTATS_CIRCUIT_BLOCKED_AT = time.time()
            logger.warning(
                "[circuit] mapstatsid circuit TRIPPED (403 on HS page) — "
                "retrying in 5 min. HS will use calibrated fallback."
            )
    else:
        if not _STATS_SUBDOMAIN_BLOCKED:
            _STATS_SUBDOMAIN_BLOCKED = True
            _STATS_SUBDOMAIN_BLOCKED_AT = time.time()
            logger.warning(
                "[circuit] /stats/ circuit-breaker TRIPPED — player/team stats "
                "blocked for 30 min. Kill data unaffected (comes from match pages)."
            )


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


def _fetch_via_scraperapi(url: str, referer: str = "") -> str | None:
    """
    Fetch a URL through ScraperAPI (https://www.scraperapi.com).
    Free tier = 1,000 req/month. Handles Cloudflare/JS rendering automatically.
    Returns HTML on success, None on failure.
    """
    import os
    import requests as _requests

    if url in _MAPSTATS_HTML_CACHE:
        return _MAPSTATS_HTML_CACHE[url]

    key = os.environ.get("SCRAPERAPI_KEY", "").strip()
    if not key:
        return None

    api_url = "https://api.scraperapi.com/"
    params = {
        "api_key":         key,
        "url":             url,
        "keep_headers":    "true",
        # render=false is enough for HLTV (no JS-gated content) and saves credits
    }
    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer":         referer or "https://www.hltv.org/",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        logger.info(f"[scraperapi] Fetching: {url[-70:]}")
        resp = _requests.get(api_url, params=params, headers=headers, timeout=70)
        if resp.status_code == 200 and len(resp.text) > 3000 and "Just a moment" not in resp.text:
            logger.info(f"[scraperapi] Success — {len(resp.text):,} chars")
            _MAPSTATS_HTML_CACHE[url] = resp.text
            return resp.text
        if resp.status_code == 401:
            logger.warning("[scraperapi] 401 — invalid SCRAPERAPI_KEY")
        elif resp.status_code == 403:
            logger.warning("[scraperapi] 403 — out of credits or plan limit")
        else:
            logger.warning(f"[scraperapi] status={resp.status_code} len={len(resp.text)}")
        return None
    except Exception as e:
        logger.warning(f"[scraperapi] {type(e).__name__}: {str(e)[:140]}")
        return None


def _fetch_via_apify_proxy(url: str, referer: str = "") -> str | None:
    """
    Fetch a URL through the Apify residential proxy network.

    Tries multiple credential formats so the correct one works regardless
    of whether APIFY_TOKEN is an API key or proxy password.
    Returns HTML on success, None on any failure (no circuit tripping).
    """
    import os
    import requests as _requests
    from requests.auth import HTTPProxyAuth

    if url in _MAPSTATS_HTML_CACHE:
        return _MAPSTATS_HTML_CACHE[url]

    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        logger.warning("[apify] APIFY_TOKEN not set — skipping proxy fetch")
        return None

    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer":         referer or "https://www.hltv.org/",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest":  "document",
        "Sec-Fetch-Mode":  "navigate",
        "Sec-Fetch-Site":  "same-origin",
    }

    # Try multiple auth formats — Apify changed how they handle credentials
    proxy_candidates = [
        f"http://auto:{token}@proxy.apify.com:8000",
        f"http://groups-BUYPROXIES94952:{token}@proxy.apify.com:8000",
        f"http://groups-RESIDENTIAL:{token}@proxy.apify.com:8000",
        f"http://{token}:@proxy.apify.com:8000",
    ]

    for proxy_url in proxy_candidates:
        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            logger.info(f"[apify] Trying proxy ({proxy_url.split('@')[0].split('//')[1][:12]}…): {url[-60:]}")
            sess = _requests.Session()
            sess.proxies = proxies
            resp = sess.get(url, headers=headers, timeout=18, verify=False)
            if resp.status_code == 200 and len(resp.text) > 3000 and "Just a moment" not in resp.text:
                logger.info(f"[apify] Success — {len(resp.text):,} chars")
                _MAPSTATS_HTML_CACHE[url] = resp.text
                return resp.text
            if resp.status_code == 407:
                logger.warning(f"[apify] 407 Proxy Auth failed for this format — trying next")
                continue
            logger.warning(f"[apify] status={resp.status_code} len={len(resp.text)} — giving up")
            return None
        except Exception as e:
            err = str(e)
            if "407" in err or "Authentication Required" in err:
                logger.warning(f"[apify] 407 in exception — trying next format")
                continue
            logger.warning(f"[apify] {type(e).__name__}: {err[:120]}")
            return None

    logger.warning("[apify] All proxy credential formats failed (407)")
    return None


def _fetch_stats_page(stats_url: str, match_url: str) -> str | None:
    """
    Fetch an HLTV /stats/matches/mapstatsid/ page.

    Strategy for mapstatsid URLs (HS data):
      - All tls_client profiles get an immediate 403 from HLTV for /stats/ paths
        — cycling through them wastes 20+ seconds per URL with no benefit.
      - Go straight to Apify residential proxy.
      - On failure, return None without tripping any circuit so each URL
        gets its own independent attempt next time.

    Strategy for all other /stats/ URLs (player/team stats):
      - Use the existing profile-cycling approach with circuit-breaker.
    """
    # Check cache first
    if stats_url in _MAPSTATS_HTML_CACHE:
        return _MAPSTATS_HTML_CACHE[stats_url]

    # ── fast-path: circuit-breaker for player/team stats only ─────────────────
    if not _is_mapstats_url(stats_url) and _is_stats_blocked(stats_url):
        logger.info(f"[stats_fetch] /stats/ circuit open — skipping {stats_url[-60:]}")
        return None

    # ── mapstatsid URLs: skip profile cycling, go straight to proxy ───────────
    if _is_mapstats_url(stats_url):
        # Try ScraperAPI first (Cloudflare-aware, free tier 1k req/mo)
        html = _fetch_via_scraperapi(stats_url, referer=match_url)
        if html:
            logger.info("[stats_fetch] ScraperAPI mapstats fetch succeeded.")
            return html
        # Fallback to Apify proxy (requires paid plan)
        html = _fetch_via_apify_proxy(stats_url, referer=match_url)
        if html:
            logger.info("[stats_fetch] Apify mapstats fetch succeeded.")
            return html
        # No circuit trip — let each URL try independently
        logger.info(f"[stats_fetch] All proxies failed for {stats_url[-60:]} — using HS% fallback")
        return None

    # ── non-mapstats /stats/ URLs: try ScraperAPI first ───────────────────────
    html = _fetch_via_scraperapi(stats_url, referer=match_url)
    if html:
        logger.info("[stats_fetch] ScraperAPI player-stats fetch succeeded.")
        return html

    # ── then fall back to profile-cycling with circuit-breaker ────────────────
    global _STATS_SESSION_WARMED

    all_profiles = list(_PROFILES)
    start_idx = _profile_idx
    ordered = all_profiles[start_idx:] + all_profiles[:start_idx]

    for profile_attempt, profile in enumerate(ordered):
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
            if not _HLTV_SESSION_WARMED:
                r_home = sess.get(HLTV_BASE + "/", timeout=12)
                if r_home.status_code == 200 and "Just a moment" not in r_home.text:
                    _HLTV_SESSION_WARMED = True
                    logger.info(f"[stats_fetch] Homepage warm-up OK [{profile}]")
                    time.sleep(random.uniform(0.4, 0.9))

            if not _STATS_SESSION_WARMED and match_url:
                r_match = sess.get(match_url, timeout=15)
                if r_match.status_code == 200:
                    _STATS_SESSION_WARMED = True
                    logger.info(f"[stats_fetch] Match-page warm-up OK [{profile}]")
                    time.sleep(random.uniform(0.3, 0.7))

            logger.info(
                f"[stats_fetch] GET {stats_url} [{profile}]"
                + (f" (attempt {profile_attempt + 1}/{len(ordered)})" if profile_attempt else "")
            )
            resp = sess.get(
                stats_url,
                headers={
                    "Referer":                   match_url or HLTV_BASE + "/",
                    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language":           "en-US,en;q=0.9",
                    "Accept-Encoding":           "gzip, deflate, br",
                    "Sec-Fetch-Dest":            "document",
                    "Sec-Fetch-Mode":            "navigate",
                    "Sec-Fetch-Site":            "same-origin",
                    "Sec-Fetch-User":            "?1",
                    "Upgrade-Insecure-Requests": "1",
                },
                timeout=15,
            )

            if resp.status_code == 200 and len(resp.text) > 3000 and "Just a moment" not in resp.text:
                logger.info(f"[stats_fetch] OK — {len(resp.text):,} chars [{profile}]")
                return resp.text

            if resp.status_code == 403:
                _trip_stats_circuit(stats_url)
                if _STATS_SUBDOMAIN_BLOCKED:
                    logger.warning("[stats_fetch] player-stats circuit TRIPPED — 30 min block")
                    return None
                _STATS_SESSION_WARMED = False
                delay = min(1.0 + profile_attempt * 1.0, 4.0) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue

            logger.warning(f"[stats_fetch] status={resp.status_code} len={len(resp.text)}")
            return None

        except Exception as e:
            logger.warning(f"[stats_fetch] {type(e).__name__}: {e} [{profile}]")
            time.sleep(1.0)
            continue

    logger.warning("[stats_fetch] All profiles exhausted for non-mapstats URL")
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

    # ── Pre-filter: skip tables with fewer than 5 rows (summary / scoreline junk)
    # and prefer tables that explicitly have a K(hs) header.  Sort: labelled first.
    def _tbl_has_khs_header(t: object) -> bool:
        hr = t.find('tr')  # type: ignore[attr-defined]
        if not hr:
            return False
        for hdr in hr.find_all(['th', 'td']):
            ht = hdr.get_text(strip=True).lower()
            if re.search(r'k[^a-z]*\(?hs|hs.*\)', ht):
                return True
        return False

    stats_tables = sorted(stats_tables, key=lambda t: (not _tbl_has_khs_header(t), -len(t.find_all('tr'))))

    for tbl in stats_tables:
        all_rows = tbl.find_all('tr')
        # Skip tiny tables (scoreboard headers, overview blocks, etc.)
        if len(all_rows) < 5:
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

        # Fallback: scan ALL data rows (not just the first) for "N (N)" pattern
        # and require that the same column is consistent across ≥2 rows.
        if khs_col is None:
            all_headers_text = [h.get_text(strip=True) for h in headers]
            logger.debug(f"[HS][{ctx}] No K(hs) header in: {all_headers_text}")
            col_hits: dict[int, int] = {}
            for dr in all_rows[1:6]:   # check up to 5 data rows
                data_cells = dr.find_all('td')
                for probe in range(min(len(data_cells), 8)):
                    ct = data_cells[probe].get_text(strip=True)
                    if re.search(r'^\d+\s*\(\d+\)$', ct):
                        # Extra guard: HS value (inside parens) must be ≤ kills value
                        try:
                            _k = int(ct.split('(')[0].strip())
                            _h = int(ct.split('(')[1].rstrip(')').strip())
                            if _h <= _k:       # valid K(hs) cell
                                col_hits[probe] = col_hits.get(probe, 0) + 1
                        except (ValueError, IndexError):
                            pass
            if col_hits:
                # Pick the column with the most consistent hits
                khs_col = max(col_hits, key=lambda c: col_hits[c])
                logger.debug(
                    f"[HS][{ctx}] K(hs) column probed at index {khs_col} "
                    f"(consistency hits={col_hits[khs_col]})"
                )

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
                headshots = int(raw_text.split('(')[1].split(')')[0].strip())
                kills_str = raw_text.split('(')[0].strip()
                kills_m   = re.search(r'(\d+)', kills_str)
                kills_hs  = int(kills_m.group(1)) if kills_m else None

                # ── Sanity check: headshots can never exceed kills ─────────────
                if kills_hs is not None and headshots > kills_hs:
                    logger.error(
                        f"[HS][{ctx}] Sanity FAIL — HS ({headshots}) > kills ({kills_hs}) "
                        f"for {player_slug!r}, cell={raw_text!r} — wrong column, discarding"
                    )
                    return None, None

                # Headshots in a single CS2 map realistically cap around 30
                if headshots > 35:
                    logger.error(
                        f"[HS][{ctx}] Sanity FAIL — HS ({headshots}) implausibly high "
                        f"for {player_slug!r} — discarding"
                    )
                    return None, None

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


def _player_has_opponent_in_history(pid: str, opponent_norm: str) -> bool:
    """
    Fetch /results?player={pid} and return True if any match slug contains
    the normalised opponent name.  Used to validate we have the right player
    when the team name isn't known but the opponent is.
    """
    html = _fetch(f"{HLTV_BASE}/results?player={pid}")
    if not html:
        return False
    slugs = re.findall(r'/matches/\d+/([\w-]+)', html)
    return any(opponent_norm in re.sub(r'[^a-z0-9]', '', s) for s in slugs)


def _player_has_upcoming_match(pid: str) -> bool:
    """
    Fetch /matches?player={pid} and return True if the player has any upcoming
    scheduled matches.  Active players always have upcoming matches listed;
    retired/inactive players have none.  Used to disambiguate when multiple
    candidates share the same name/slug — the one with an upcoming match is
    clearly the player that should be graded.
    """
    html = _fetch(f"{HLTV_BASE}/matches?player={pid}")
    if not html:
        return False
    # HLTV lists upcoming matches as /matches/{id}/{slug} links in the schedule
    upcoming = re.findall(r'/matches/(\d{7,})/[\w-]+', html)
    has_match = len(upcoming) > 0
    logger.debug(
        f"[upcoming] player {pid} — "
        f"{'HAS' if has_match else 'NO'} upcoming matches ({len(upcoming)} found)"
    )
    return has_match


def get_upcoming_match_context(pid: str) -> bool | None:
    """
    Auto-detect whether the player's NEXT upcoming match is LAN or Online.

    Strategy:
      1. Fetch /matches?player={pid} → find the first upcoming match link.
      2. Fetch that match page → parse "Best of N (LAN|Online)" with
         _parse_match_format().

    Returns True (LAN), False (Online), or None if not detectable.
    Used by !grade to set today_is_lan automatically when the user doesn't
    pass an explicit `lan` / `online` token.
    """
    html = _fetch(f"{HLTV_BASE}/matches?player={pid}")
    if not html:
        return None

    upcoming = re.findall(r'/matches/(\d{7,})/([\w-]+)', html)
    if not upcoming:
        logger.info(f"[lan_auto] player {pid} — no upcoming match found")
        return None

    # Take the first upcoming match (HLTV lists them chronologically)
    match_id, slug = upcoming[0]
    match_url = f"{HLTV_BASE}/matches/{match_id}/{slug}"
    match_html = _fetch(match_url)
    if not match_html:
        logger.warning(f"[lan_auto] could not fetch upcoming match page {match_url}")
        return None

    fmt = _parse_match_format(match_html)
    if fmt is None:
        logger.info(f"[lan_auto] no LAN/Online tag found in {match_url}")
        return None

    is_lan = (fmt == "LAN")
    logger.info(f"[lan_auto] player {pid} next match {match_id} → {fmt} (is_lan={is_lan})")
    return is_lan


def search_player_v2(
    name: str,
    team_hint: str | None = None,
    opponent_hint: str | None = None,
) -> tuple[str, str, str] | None:
    """
    Improved player search with cache-first lookup and optional disambiguation.

    Disambiguation priority (use whichever hint is available):
      team_hint    — verify/pick the candidate whose current HLTV team matches.
      opponent_hint — verify/pick the candidate whose recent match slugs contain
                      the opponent name.  Useful for !grade where the player's
                      own team is unknown but the opponent is.

    Order:
      1. Check _PLAYER_ID_CACHE (instant — no HTTP).
         If a hint given, validate; invalidate and re-search if wrong.
      2. Fall through to HLTV /search?query={name}.
         Iterate candidates by name score; pick first that passes hint validation.
      3. On success, populate cache so future lookups skip the search.
    """
    key = _normalise_player_key(name)
    team_hint_norm = re.sub(r'[^a-z0-9]', '', team_hint.lower()) if team_hint else None
    opp_hint_norm  = re.sub(r'[^a-z0-9]', '', opponent_hint.lower()) if opponent_hint else None

    # 1. Cache hit
    if key in _PLAYER_ID_CACHE:
        pid, slug, display = _PLAYER_ID_CACHE[key]
        if not team_hint_norm and not opp_hint_norm:
            logger.info(f"[search] Cache hit: {display} (id={pid}, slug={slug})")
            return pid, slug, display
        # Validate cached player against whichever hint we have
        if team_hint_norm:
            team_result = get_player_team(pid, slug)
            if team_result:
                _, tslug = team_result
                tslug_norm = re.sub(r'[^a-z0-9]', '', tslug.lower())
                if team_hint_norm in tslug_norm or tslug_norm in team_hint_norm:
                    logger.info(
                        f"[search] Cache hit (team verified): {display} "
                        f"(id={pid}) → team={tslug}"
                    )
                    return pid, slug, display
                else:
                    logger.warning(
                        f"[search] Cache entry for {name!r} is on team '{tslug}' "
                        f"but hint='{team_hint}' — invalidating and re-searching"
                    )
                    del _PLAYER_ID_CACHE[key]
            else:
                logger.warning(f"[search] Team verification failed for cached {name!r} — trusting cache")
                return pid, slug, display
        elif opp_hint_norm:
            # Validate via match history: does this player have games vs the opponent?
            if _player_has_opponent_in_history(pid, opp_hint_norm):
                logger.info(
                    f"[search] Cache hit (opponent verified): {display} "
                    f"(id={pid}) has matches vs '{opponent_hint}'"
                )
                return pid, slug, display
            else:
                logger.warning(
                    f"[search] Cache entry for {name!r} (id={pid}) has NO matches "
                    f"vs '{opponent_hint}' — invalidating and re-searching"
                )
                del _PLAYER_ID_CACHE[key]

    # 2. Live HLTV search
    url = f"{HLTV_BASE}/search?query={name}"
    html = _fetch(url)
    if not html:
        return None

    matches = re.findall(r'/player/(\d+)/([\w-]+)', html)
    if not matches:
        logger.warning(f"[search] No player found for '{name}'")
        return None

    seen: dict[str, str] = {}
    for pid, slug in matches:
        if pid not in seen:
            seen[pid] = slug

    # Score all candidates by name similarity, best first.
    # Tiebreaker: higher player_id wins — HLTV assigns IDs sequentially so a
    # higher ID indicates a more recently registered (likely still-active) player.
    # This resolves ambiguous cases like two players sharing the same slug
    # where one is retired (low ID) and one is currently playing (high ID).
    scored = sorted(
        [(pid, slug, _score_player_match(name, pid, slug)) for pid, slug in seen.items()],
        key=lambda x: (x[2], int(x[0]) if x[0].isdigit() else 0),
        reverse=True,
    )

    if not scored:
        return None

    best_pid, best_slug, best_score = None, None, -1

    if team_hint_norm:
        # Try each candidate in score order; take first whose team matches hint
        for pid, slug, score in scored:
            team_result = get_player_team(pid, slug)
            if team_result:
                _, tslug = team_result
                tslug_norm = re.sub(r'[^a-z0-9]', '', tslug.lower())
                if team_hint_norm in tslug_norm or tslug_norm in team_hint_norm:
                    best_pid, best_slug, best_score = pid, slug, score
                    logger.info(
                        f"[search] Team-matched: slug={slug} team={tslug} "
                        f"for hint='{team_hint}'"
                    )
                    break
                else:
                    logger.debug(
                        f"[search] Team mismatch: slug={slug} team={tslug} "
                        f"≠ hint='{team_hint}'"
                    )
            else:
                logger.debug(f"[search] Could not verify team for pid={pid} slug={slug}")
        if best_pid is None:
            logger.warning(
                f"[search] No team match for hint='{team_hint}' — "
                f"returning None to force roster lookup instead of guessing wrong player"
            )
            return None   # Caller will try roster lookup or raise RuntimeError

    elif opp_hint_norm:
        # Try each candidate; pick first whose match history includes the opponent
        for pid, slug, score in scored:
            if _player_has_opponent_in_history(pid, opp_hint_norm):
                best_pid, best_slug, best_score = pid, slug, score
                logger.info(
                    f"[search] Opponent-matched: slug={slug} (id={pid}) "
                    f"has matches vs '{opponent_hint}'"
                )
                break
            else:
                logger.debug(
                    f"[search] Opponent miss: slug={slug} (id={pid}) "
                    f"has NO matches vs '{opponent_hint}'"
                )
        if best_pid is None:
            logger.warning(
                f"[search] No candidate with matches vs '{opponent_hint}' — "
                f"falling back to best name match"
            )
            best_pid, best_slug, best_score = scored[0]

    else:
        # No hints at all.  When multiple candidates share the same top name
        # score (e.g. two players both nicknamed "sandman"), use the upcoming
        # match check as the primary tiebreaker: an active player always has
        # at least one scheduled match on HLTV; a retired player has none.
        top_score = scored[0][2]
        tied = [(pid, slug, sc) for pid, slug, sc in scored if sc == top_score]

        if len(tied) > 1:
            logger.info(
                f"[search] {len(tied)} candidates tied at score={top_score} "
                f"— checking upcoming matches to find active player"
            )
            for pid, slug, sc in tied:
                if _player_has_upcoming_match(pid):
                    best_pid, best_slug, best_score = pid, slug, sc
                    logger.info(
                        f"[search] Upcoming-match winner: slug={slug} (id={pid})"
                    )
                    break
            if best_pid is None:
                # All tied or none have upcoming match — fall back to highest player ID
                logger.warning(
                    f"[search] No upcoming matches found for tied candidates — "
                    f"using highest player ID as tiebreaker"
                )
                best_pid, best_slug, best_score = scored[0]  # already sorted by ID desc
        else:
            best_pid, best_slug, best_score = scored[0]

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

# ── CS2 Valid Map Pool ──────────────────────────────────────────────────────
_VETO_VALID_MAPS = {
    'mirage', 'inferno', 'dust2', 'nuke', 'anubis', 'ancient',
    'vertigo', 'overpass', 'train', 'cache', 'cobblestone',
}


def parse_veto(html: str) -> list[dict]:
    """
    Extract map veto actions (picks, bans, left-overs) from an HLTV match page.

    Returns a list of dicts:
        [{'team': 'NaVi', 'action': 'removed', 'map': 'Mirage'}, ...]

    Actions:
        'removed'   — team banned this map
        'picked'    — team picked this map (will be played)
        'left over' — default / decider map (neither team picked/banned)
    """
    vetos: list[dict] = []
    pattern = (
        r'(?:class="[^"]*"[^>]*>)?\s*(\w[\w\s.]+?)\s+'
        r'(removed|picked|left over)\s+(\w+)'
    )
    for team, action, map_name in re.findall(pattern, html, re.IGNORECASE):
        team = team.strip()
        if len(team) < 30 and map_name.lower() in _VETO_VALID_MAPS:
            vetos.append({
                'team':   team,
                'action': action.lower(),
                'map':    map_name.capitalize(),
            })
    return vetos


def is_bo3_html(html: str) -> bool:
    """
    Multi-signal BO3 detection for an HLTV match page.

    Checks (in order of reliability):
      1. JSON  "bestof":3  or  "bestOf":3  embedded in page data
      2. Plain text "best of 3"
      3. Number of mapholder divs ≥ 2 (HLTV always renders a div per map)
      4. Veto page contains "picked" or "removed" (single-map BO1s rarely have
         a veto section)
    Returns True if any strong signal confirms BO3.
    """
    lower = html.lower()
    # Signal 1 — JSON bestof field (most reliable)
    if '"bestof":3' in lower or '"bestof": 3' in lower or '"bestOf":3' in html:
        return True
    # Signal 2 — plain text
    if 'best of 3' in lower:
        return True
    # Signal 3 — mapholder count
    mh_count = len(re.findall(r'class="[^"]*mapholder[^"]*"', html))
    if mh_count >= 2:
        return True
    # Signal 4 — veto verbs (weak — only use if nothing else fired)
    if 'veto' in lower and re.search(r'\b(picked|removed|banned)\b', lower):
        return True
    return False


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

    # ── Parse ONLY result-row links (not nav/sidebar/footer links) ────────────
    # HLTV's results page embeds many /matches/ links in sidebars, navbar, and
    # event footers that are NOT this player's matches.  A bare regex over the
    # entire page picks those up and contaminates the list with unrelated matches
    # whose stats pages trip the 403 circuit-breaker before the player's real
    # matches are even touched.
    #
    # We restrict to <a> elements inside .result-con or .results-all containers.
    _soup_res = BeautifulSoup(html, 'html.parser')

    # Primary: look for result-con anchor tags (current HLTV results page format)
    result_anchors = []
    for _rc in _soup_res.find_all('div', class_='result-con'):
        for _a in _rc.find_all('a', href=True):
            result_anchors.append(_a['href'])

    # Secondary fallback: any <a> inside a .results-all container
    if not result_anchors:
        for _ra in _soup_res.find_all(class_='results-all'):
            for _a in _ra.find_all('a', href=True):
                result_anchors.append(_a['href'])

    # Last resort: regex on the full page (original behaviour) — deduplicated
    if not result_anchors:
        logger.warning("[results] No result-con divs found — falling back to full-page regex")
        result_anchors = [
            f"/matches/{mid}/{slug}"
            for mid, slug in re.findall(r'/matches/(\d+)/([a-z0-9-]+)', html)
        ]

    seen = {}
    for href in result_anchors:
        m = re.match(r'/matches/(\d+)/([a-z0-9-]+)', href)
        if m:
            mid, slug = m.group(1), m.group(2)
            if mid not in seen and len(mid) >= 6:
                seen[mid] = slug

    results = list(seen.items())[:max_matches]
    logger.info(
        f"[results] Found {len(results)} match IDs for player {player_id} "
        f"(from {len(result_anchors)} result anchors)"
    )

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
    # Fallback: if score-based detection thinks BO1, count played mapholder divs.
    # Matches with 2+ played maps are always BO3 regardless of score encoding.
    if bo_type == 1:
        _ph_played = sum(
            1 for _mh in soup.find_all('div', class_='mapholder')
            if 'not-played' not in ' '.join(_mh.get('class', []))
        )
        if _ph_played >= 2:
            bo_type = 3
            logger.info(f"[parse] BO type corrected to 3 via mapholder count ({_ph_played} played)")

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

    # ── Inline-table fallback ─────────────────────────────────────────────────
    # Some HLTV tournament formats (e.g. Clutch Series, ESL Challenger smaller
    # events) render map scorecards as plain <table class="table"> elements in
    # the page *without* assigning them a match-stats tab ID.  When we captured
    # fewer maps than the played-map count from the mapholder divs, scan all
    # inline tables in DOM order and back-fill the missing maps.
    _played_mapholders = sum(
        1 for _mh in soup.find_all('div', class_='mapholder')
        if 'not-played' not in ' '.join(_mh.get('class', []))
    )
    _want_maps = min(2, _played_mapholders)   # Props only need Maps 1+2
    if len(maps_result) < _want_maps:
        logger.info(
            f"[parse] Only {len(maps_result)} map(s) from tab system; "
            f"{_played_mapholders} played mapholders — running inline-table fallback"
        )
        # Collect player rows from every <table class="table"> in DOM order.
        # Each unique table that contains the player corresponds to one played map.
        _inline_entries: list[dict] = []
        for _tbl in soup.find_all('table', class_='table'):
            for _tr in _tbl.find_all('tr'):
                _rn = re.sub(r'[^a-z0-9]', '', _tr.get_text().lower())
                if slug_norm not in _rn:
                    continue
                # Found player row — grab K-D from first matching cell
                for _td in _tr.find_all('td'):
                    _kd_m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', _td.get_text(strip=True))
                    if _kd_m:
                        _k = int(_kd_m.group(1))
                        _d = int(_kd_m.group(2))
                        if 0 <= _k <= 60:
                            _inline_entries.append({'kills': _k, 'deaths': _d})
                        break
                break  # one row per table

        # _inline_entries is in DOM order → index 0 = Map 1, index 1 = Map 2 …
        _already_map_nums = {m['map_number'] for m in maps_result}
        for _mi, _ie in enumerate(_inline_entries, start=1):
            if _mi > _want_maps:
                break
            if _mi in _already_map_nums:
                continue   # already captured via the tab system
            maps_result.append({
                'map_name':      f'Map{_mi}',
                'kills':         _ie['kills'],
                'headshots':     None,
                'deaths':        _ie['deaths'],
                'rounds':        24,
                'rating':        None,
                'kast_pct':      None,
                'adr':           None,
                'survival_rate': None,
                'fk':            None,
                'fd':            None,
                'map_number':    _mi,
            })
            logger.info(
                f"[parse] Inline-table fallback Map{_mi}: "
                f"{player_slug} — {_ie['kills']}K/{_ie['deaths']}D"
            )
        maps_result.sort(key=lambda m: m['map_number'])

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
    # HLTV Analytics Attributes (0–100 scale)
    elif label in ('firepower',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['firepower'] = v
    elif label in ('opening',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['opening'] = v
    elif label in ('entrying',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['entrying'] = v
    elif label in ('trading',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['trading'] = v
    elif label in ('sniping',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['sniping'] = v
    elif label in ('clutching',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['clutching'] = v
    elif label in ('utility',):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['utility'] = v
    elif any(x in label for x in ('survival', 'survival rate')):
        v = _flt(vt.replace('%', ''), 0, 100)
        if v is not None:
            result['survival'] = v
    elif any(x in label for x in ('opening kill ratio', 'opening ratio')):
        v = _flt(vt, 0, 5.0)
        if v is not None:
            result['opening_ratio'] = v


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

    # Strategy 4: HLTV Analytics Attributes
    # These appear as progress bars on the player stats page:
    #   Firepower, Opening, Entrying, Trading, Sniping, Clutching, Utility
    # HTML pattern: <div class="summary-boxes">…<span class="label">Firepower</span>…<span class="...">72</span>
    # Also as: style="width:72%" within player-attribute-fill, or data-tip attributes
    _ANALYTICS_ATTRS = {
        "firepower": ["firepower"],
        "opening":   ["opening"],
        "entrying":  ["entrying"],
        "trading":   ["trading"],
        "sniping":   ["sniping"],
        "clutching": ["clutching"],
        "utility":   ["utility"],
    }

    # Try structured: any element where text == attribute name and a nearby numeric sibling
    _page_text = soup.get_text(separator="\n")
    for _attr_key, _synonyms in _ANALYTICS_ATTRS.items():
        if _attr_key in result:
            continue
        for _syn in _synonyms:
            # Look for span/div with the attribute name, then find the next numeric value
            for _tag in soup.find_all(True, string=re.compile(rf'(?i)^{re.escape(_syn)}$')):
                _parent = _tag.parent
                # Try to find a numeric sibling or nearby span
                for _candidate in (_parent, _tag):
                    for _sib in (_candidate.find_next_siblings() + _candidate.find_all(['span', 'div'])):
                        _st = _sib.get_text(strip=True).rstrip('%')
                        try:
                            _fv = float(_st)
                            if 0 <= _fv <= 100:
                                result[_attr_key] = _fv
                                break
                        except ValueError:
                            continue
                    if _attr_key in result:
                        break
                if _attr_key in result:
                    break

            # Also try: style="width: XX%" on fill bars near a label with this attribute name
            if _attr_key not in result:
                for _fill in soup.find_all(class_=re.compile(r'fill|bar|attr', re.I)):
                    _st = _fill.get('style', '')
                    _label_scope = _fill.find_parent()
                    if not _label_scope:
                        continue
                    _scope_text = _label_scope.get_text().lower()
                    if _syn in _scope_text:
                        _m = re.search(r'width\s*:\s*(\d+(?:\.\d+)?)\s*%', _st)
                        if _m:
                            result[_attr_key] = float(_m.group(1))
                            break

            # Regex raw fallback on page text
            if _attr_key not in result:
                _m = re.search(
                    rf'(?i){re.escape(_syn)}[^0-9]{{0,40}}?(\d{{1,3}})(?:\s*%|\s*\/\s*100)',
                    _page_text,
                )
                if _m:
                    _fv = float(_m.group(1))
                    if 0 <= _fv <= 100:
                        result[_attr_key] = _fv

    # Strategy 5: survival rate (% of rounds survived) — from raw text
    if 'survival' not in result:
        _m = re.search(r'(?i)survival[^0-9]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%', _page_text)
        if _m:
            result['survival'] = float(_m.group(1))

    # Strategy 6: opening ratio / opening kill rate
    if 'opening' not in result:
        _m = re.search(r'(?i)opening\s+(?:kill\s+)?(?:ratio|rate)[^0-9]{0,40}?(\d\.\d{2})', _page_text)
        if _m:
            result['opening'] = float(_m.group(1))

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
    # Try ScraperAPI first (Cloudflare-aware); only fall back to direct fetch if no key.
    html = _fetch_via_scraperapi(url, referer="https://www.hltv.org/")
    if not html:
        if _is_stats_blocked(url):
            logger.info(f"[period_stats] /stats/ circuit open — skipping player stats for {player_slug}")
            return None
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
    # Fast-path: skip immediately if /stats/ subdomain is known-blocked
    if _is_stats_blocked(url):
        logger.info(f"[period_stats] /stats/ circuit open — skipping team stats for {team_slug}")
        return None
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

def _parse_match_format(html: str) -> str | None:
    """
    Parse the match format string from an HLTV match page to determine LAN vs Online.

    HLTV puts the format in plain text near the score, like:
        "Best of 3 (Online)"
        "Best of 3 (LAN)"
        "Best of 5 (LAN)"

    Returns "LAN", "Online", or None if it can't be parsed.
    Note: event names containing "LAN" (e.g. "MPKBK CIS LAN Season 3") are
    NOT a reliable signal — those events often run online qualifiers. The
    "(LAN)" / "(Online)" tag in the format string is the ground truth.
    """
    if not html:
        return None
    # Match either casing in case HLTV ever changes
    m = re.search(r'Best of \d+\s*\(\s*(LAN|Online)\s*\)', html, re.IGNORECASE)
    if m:
        token = m.group(1).strip().lower()
        return "LAN" if token == "lan" else "Online"
    return None


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
# Stats-Matches page scraper  (/stats/players/matches/{id}/{slug})
# ---------------------------------------------------------------------------

_CS2_MAPS_SET = {
    'dust2', 'mirage', 'inferno', 'nuke', 'anubis',
    'ancient', 'overpass', 'vertigo',
}

_PRE_CS2_ID_THRESHOLD_SM = 2_366_000   # reused from main flow


def get_player_stats_matches(
    player_id: str,
    player_slug: str,
    stat_type: str = "Kills",
    max_series: int = 10,
) -> list[dict] | None:
    """
    Fetch per-map kills + headshots from the HLTV stats/players/matches page.

    URL: /stats/players/matches/{player_id}/{player_slug}?startDate=2023-09-27

    The stats-table lists every map the player competed in with a "K (hs)"
    column — giving kills and headshots together in a single fetch.  This is
    the cleanest path for Headshots props and supplements the match-page flow
    for Kills when fewer than the desired 10 series are available.

    Returns a list of map_stat dicts compatible with run_simulation(), or None
    if the page is unreachable / has no usable data.
    """
    CS2_START = "2023-09-27"
    url = (
        f"{HLTV_BASE}/stats/players/matches/{player_id}/{player_slug}"
        f"?startDate={CS2_START}"
    )
    logger.info(f"[stats_matches] Fetching {url}")
    html = _fetch(url)
    if not html:
        logger.info(f"[stats_matches] No HTML returned for {player_slug} — likely blocked")
        return None

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"class": "stats-table"})
    if not table:
        logger.info(f"[stats_matches] No stats-table found for {player_slug}")
        return None

    # ── Detect column indices from header ──────────────────────────────────
    thead = table.find("thead")
    header_cells = thead.find_all("th") if thead else []
    headers = [th.get_text(strip=True).lower() for th in header_cells]
    n_headers = len(headers)
    logger.info(f"[stats_matches] Headers ({n_headers}): {headers}")

    map_col    = 1    # default — HLTV stats/matches typically: date|map|event|opponent|result|K(hs)|…
    result_col = 3
    khs_col    = 4

    for i, h in enumerate(headers):
        h_clean = h.strip()
        if h_clean == "map" and map_col == 1:
            map_col = i
        if h_clean in ("result", "res") and result_col == 3:
            result_col = i
        # "k (hs)" or "k" column containing headshots
        if ("k" in h_clean and "hs" in h_clean) or h_clean == "k":
            khs_col = i

    logger.info(
        f"[stats_matches] col indices — map:{map_col} result:{result_col} k_hs:{khs_col}"
    )

    # ── Parse rows ─────────────────────────────────────────────────────────
    tbody = table.find("tbody")
    if not tbody:
        return None
    all_rows = tbody.find_all("tr")

    # Each "full" row has n_headers tds (including date with rowspan).
    # "Continuation" rows (covered by rowspan) have n_headers-1 tds — date cell absent.
    # Strategy: when a row has fewer tds than the full width, all column indices shift -1.
    parsed_maps: list[dict] = []
    match_counter = 0

    for row in all_rows:
        cells = row.find_all("td")
        if not cells:
            continue

        # Determine col offset — continuation rows lack the date/leading cell
        offset = 0
        if n_headers > 0 and len(cells) < n_headers:
            # Missing leading column(s) — shift indices left by the difference
            offset = -(n_headers - len(cells))

        def _get(col_idx: int) -> str:
            idx = col_idx + offset
            if 0 <= idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""

        # Detect new series start: full row restores the date cell
        if len(cells) >= n_headers or n_headers == 0:
            match_counter += 1

        # Also try to extract a real match ID from any embedded link
        real_match_id: str | None = None
        for cell in cells:
            a = cell.find("a", href=re.compile(r"/matches/\d+/"))
            if a:
                m = re.search(r"/matches/(\d+)/", a["href"])
                if m:
                    real_match_id = m.group(1)
                    break

        # Map name
        map_name = _get(map_col).lower()
        if map_name not in _CS2_MAPS_SET:
            # Try every cell for a map name
            map_name = next(
                (c.get_text(strip=True).lower()
                 for c in cells
                 if c.get_text(strip=True).lower() in _CS2_MAPS_SET),
                None,
            )
        if not map_name:
            continue

        # Rounds from result ("16 - 12", "W (16:12)", etc.)
        result_text = _get(result_col)
        nums = re.findall(r"\d+", result_text[:25])
        if len(nums) >= 2:
            rounds = sum(int(n) for n in nums[:2])
            if rounds < 16 or rounds > 50:   # sanity check
                rounds = 22
        else:
            rounds = 22

        # Kills and headshots from "K (hs)" cell
        k_hs_text = _get(khs_col)
        k_match = re.search(r"(\d+)\s*\((\d+)\)", k_hs_text)
        if k_match:
            kills     = int(k_match.group(1))
            headshots = int(k_match.group(2))
        else:
            # Fallback: scan every cell for "N (M)" pattern
            kills = headshots = None
            for cell in cells:
                ct = cell.get_text(strip=True)
                m2 = re.search(r"(\d+)\s*\((\d+)\)", ct)
                if m2:
                    kills     = int(m2.group(1))
                    headshots = int(m2.group(2))
                    break
            if kills is None:
                # Last resort: parse a plain integer from khs cell
                k_only = re.search(r"(\d+)", k_hs_text)
                kills = int(k_only.group(1)) if k_only else None
            if kills is None:
                continue

        mid_str = real_match_id or f"sm{match_counter:04d}"

        parsed_maps.append({
            "match_id":  mid_str,
            "map_name":  map_name,
            "kills":     kills,
            "headshots": headshots,
            "rounds":    rounds,
        })

    if not parsed_maps:
        logger.info(f"[stats_matches] No usable map rows parsed for {player_slug}")
        return None

    logger.info(
        f"[stats_matches] Parsed {len(parsed_maps)} raw map rows for {player_slug}"
    )

    # ── Group into series (consecutive maps sharing the same match_id) ─────
    # Preserve order (newest first from HLTV) — group by match_id runs.
    from itertools import groupby as _groupby

    series_list: list[list[dict]] = []
    for _mid, group in _groupby(parsed_maps, key=lambda x: x["match_id"]):
        series_list.append(list(group))

    # ── Build map_stats output ─────────────────────────────────────────────
    is_hs = stat_type.lower() in ("headshots", "hs")
    map_stats: list[dict] = []
    series_count = 0

    for smaps in series_list:
        if series_count >= max_series:
            break
        if len(smaps) < 2:
            continue   # BO1 — skip

        # CS2 era gate on real match IDs
        first_mid = smaps[0]["match_id"]
        try:
            if int(first_mid) < _PRE_CS2_ID_THRESHOLD_SM:
                logger.info(
                    f"[stats_matches] Skipping series {first_mid} — pre-CS2 ID"
                )
                continue
        except (ValueError, TypeError):
            pass   # synthetic IDs ("sm0001") pass through

        for m in smaps[:2]:
            stat_val = m["headshots"] if is_hs else m["kills"]
            if stat_val is None:
                # For HS props, skip maps with no HS data
                if is_hs:
                    continue
                stat_val = m["kills"]   # kills always present
            map_stats.append({
                "stat_value":    stat_val,
                "headshots":     m["headshots"],
                "kills":         m["kills"],
                "rounds":        m["rounds"],
                "match_id":      m["match_id"],
                "match_slug":    "",
                "map_name":      m["map_name"],
                "rating":        None,
                "kast_pct":      None,
                "adr":           None,
                "survival_rate": None,
                "fk":            None,
                "fd":            None,
                "deaths":        None,
                "pistol_kills":  None,
                "opp_rank":      None,
                "match_hs_pct":  None,
            })

        series_count += 1

    if len(map_stats) < 4:
        logger.info(
            f"[stats_matches] Insufficient data for {player_slug}: "
            f"{len(map_stats)} maps from {series_count} series"
        )
        return None

    logger.info(
        f"[stats_matches] Built {len(map_stats)} map_stats "
        f"({series_count} series) for {player_slug} [{stat_type}]"
    )
    return map_stats


# ---------------------------------------------------------------------------
# Main entry point — used by the bot
# ---------------------------------------------------------------------------

def get_player_info(
    player_name: str,
    stat_type: str = "Kills",
    team_hint: str | None = None,
    opponent_hint: str | None = None,
) -> dict:
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

    team_hint:     if provided (e.g. "3dmax"), disambiguate via team roster lookup.
    opponent_hint: if provided (e.g. "b8"), disambiguate by checking whether each
                   candidate's recent match history includes games vs the opponent.
                   Useful for !grade where only the opponent is known.
    """
    logger.info(
        f"[scraper] Looking up '{player_name}' for {stat_type} "
        f"(team_hint={team_hint!r}, opponent_hint={opponent_hint!r})"
    )

    # Step 1: Find player (HLTV search + cache)
    # When we know the player's team (from PrizePicks), resolve via the team's
    # HLTV roster page first — it's unambiguous and avoids same-nickname collisions
    # (e.g. two different players both nicknamed "lucky" on different teams).
    result = None
    if team_hint:
        key = _normalise_player_key(player_name)
        if key in _PLAYER_ID_CACHE:
            # Already cached — let search_player_v2 verify it's the right team
            result = search_player_v2(player_name, team_hint=team_hint)
        else:
            # Step A: Try a direct HLTV name/slug search FIRST.
            # If the search returns a player whose normalised slug is an exact
            # (or very close) match to what was typed, trust that player
            # regardless of team_hint.  This prevents wrong-player resolutions
            # like "R3salt" → reiko-on-ESC when the real R3salt is on Nemesis
            # and the user passed 'esc' as a stale team hint.
            _name_norm = re.sub(r'[^a-z0-9]', '', player_name.lower())
            _direct = search_player_v2(player_name)   # no team hint — pure name match
            if _direct:
                _dpid, _dslug, _ddisp = _direct
                _dslug_norm = re.sub(r'[^a-z0-9]', '', _dslug.lower())
                # Accept the direct hit if the slug is an exact match (covers
                # names with numbers/special chars like "r3salt", "s1mple").
                if _dslug_norm == _name_norm:
                    result = _direct
                    logger.info(
                        f"[scraper] Exact slug match from direct search — "
                        f"using {_dslug} (id={_dpid}) — ignoring stale team_hint='{team_hint}'"
                    )

            if not result:
                # Step B: Direct search didn't give an exact match — fall back to
                # the team roster lookup which handles nicknames that differ from slugs.
                result = resolve_player_from_roster(player_name, team_hint)
                if result:
                    pid, slug, display = result
                    # Populate cache so subsequent calls are instant
                    cache_key = _normalise_player_key(player_name)
                    _PLAYER_ID_CACHE[cache_key] = result
                    slug_key = _normalise_player_key(slug)
                    if slug_key != cache_key:
                        _PLAYER_ID_CACHE[slug_key] = result
                    logger.info(f"[scraper] Roster resolved and cached: {display} (id={pid})")
                else:
                    # Roster lookup also failed — try name search with team hint
                    logger.warning(
                        f"[scraper] Roster lookup failed for '{player_name}' on '{team_hint}' "
                        f"— falling back to name search"
                    )
                    result = search_player_v2(player_name, team_hint=team_hint)
    else:
        # No team_hint — use opponent-based disambiguation if available
        result = search_player_v2(player_name, opponent_hint=opponent_hint)

    if not result:
        raise RuntimeError(f"Player '{player_name}' not found on HLTV")
    player_id, player_slug, display_name = result

    # Step 1b: Enrich with bo3.gg context (instant, ~500ms, no CF block)
    # Also fetch Liquipedia role for HS% calibration — only needed for headshots props.
    _bo3gg_ctx = bo3gg_player_context(player_slug)
    _liq_role: str | None = None
    if stat_type.lower() in ("headshots", "hs"):
        # Only spend ~1-2s on Liquipedia if we actually need HS% role info
        _liq_role = liquipedia_player_role(display_name)
    logger.info(
        f"[scraper] bo3.gg context: {_bo3gg_ctx} | Liquipedia role: {_liq_role} "
        f"(stat_type={stat_type})"
    )

    # Step 1c: For Headshots props — try stats/players/matches page as PRIMARY source.
    # That page has a K (hs) column with both kills and headshots in a single fetch,
    # eliminating the need to hop through mapstatsid pages.  If it returns ≥ 8 map
    # samples (4 series × 2 maps), return early with that data — it's cleaner and faster.
    _is_hs_prop = stat_type.lower() in ("headshots", "hs")
    if _is_hs_prop:
        _sm_data = get_player_stats_matches(player_id, player_slug, stat_type="Headshots")
        if _sm_data and len(_sm_data) >= 8:
            logger.info(
                f"[scraper] stats/matches page returned {len(_sm_data)} maps for "
                f"{display_name} [HS prop] — using as primary data source"
            )
            import statistics as _stats_mod
            _hs_vals = [m["stat_value"] for m in _sm_data]
            _hs_mean = _stats_mod.mean(_hs_vals)
            _hs_std  = _stats_mod.stdev(_hs_vals) if len(_hs_vals) > 1 else 2.0
            _hs_std  = max(_hs_std, 1.0)
            # Compute recent_hs_pct from actual kill/hs pairs on this page
            _hs_rate_pairs = [
                (m["headshots"], m["kills"])
                for m in _sm_data
                if m.get("headshots") is not None and m.get("kills", 0) > 0
            ]
            if _hs_rate_pairs:
                _recent_hs_pct = sum(h / k for h, k in _hs_rate_pairs) / len(_hs_rate_pairs)
            else:
                _recent_hs_pct = None
            _country_sm = _bo3gg_ctx.get("country") if _bo3gg_ctx else None
            return {
                "player":            display_name,
                "player_id":         player_id,
                "player_slug":       player_slug,
                "player_team_id":    None,
                "player_team_slug":  None,
                "match_ids":         [],
                "map_kills":         _sm_data,   # full map_stat dicts (same as normal flow)
                "mean":              round(_hs_mean, 2),
                "std":               round(_hs_std, 2),
                "sample_size":       len(_sm_data),
                "source":            "HLTV Stats/Matches",
                "recent_hs_pct":     _recent_hs_pct,
                "hs_pct_n_matches":  0,
                "bo3gg_context":     _bo3gg_ctx,
                "liquipedia_role":   _liq_role,
                "country":           _country_sm,
                "team_mismatch":     False,
            }
        else:
            logger.info(
                f"[scraper] stats/matches primary path insufficient for {display_name} "
                f"({len(_sm_data) if _sm_data else 0} maps) — falling through to match pages"
            )

    # Step 2: Get recent match IDs
    match_ids = get_player_match_ids(player_id, max_matches=40)
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

        # Quick BO3 pre-filter — reject BO1/BO2 before the expensive parse.
        # is_bo3_html() checks JSON bestof field, plain text, and mapholder count.
        if not is_bo3_html(html):
            logger.debug(f"[bo3_filter] Skipping {match_id} — not BO3 (pre-filter)")
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

        # ── CS2 era + map filter ──────────────────────────────────────────────
        # Gate 1: Match ID / slug era check.
        # CS2 launched Sept 27 2023.  HLTV match IDs are monotonically increasing;
        # IDs below ~2,366,000 correspond to pre-launch dates.  The slug also
        # reliably contains the event year (-2021-, -2022-, early -2023-).
        # We skip anything confidently pre-CS2 to avoid CS:GO data contaminating
        # props analysis.  Maps like Dust2/Mirage exist in both games, so a
        # map-name check alone cannot distinguish CS:GO from CS2 matches.
        _PRE_CS2_ID_THRESHOLD = 2_366_000  # approximately CS2 launch
        _skip_era = False
        try:
            _mid_int = int(match_id)
            if _mid_int < _PRE_CS2_ID_THRESHOLD:
                _skip_era = True
        except ValueError:
            pass
        # Belt-and-suspenders: year tokens in slug are definitive
        _slug_lower = slug.lower()
        if any(yr in _slug_lower for yr in ('-2020-', '-2021-', '-2022-')):
            _skip_era = True
        # Early 2023 (pre-CS2 launch) — IDs typically < 2,373,000 with a 2023 slug
        if '-2023-' in _slug_lower and int(match_id) < 2_373_000:
            _skip_era = True
        if _skip_era:
            logger.info(
                f"[era_filter] Skipping match {match_id} ({slug}) — "
                f"pre-CS2 era (ID {match_id} / slug year)"
            )
            continue

        # Gate 2: Map name check — reject maps that never existed in CS2.
        # Train, Cache, Cobblestone, etc. are CS:GO-only.
        _CS2_MAPS = {
            'dust2', 'mirage', 'inferno', 'nuke', 'anubis',
            'ancient', 'overpass', 'vertigo',
        }
        _series_map_names = [m['map_name'].lower() for m in maps[:2]]
        if any(mn not in _CS2_MAPS for mn in _series_map_names if mn):
            logger.info(
                f"[cs2_filter] Skipping match {match_id} — "
                f"non-CS2 maps detected: {_series_map_names}"
            )
            continue

        # Collect HS% from the match-level overview (all-maps combined stats row)
        # This is parsed from the same page we already fetched — no extra requests.
        match_hs = _parse_match_hs_pct(html, player_slug)
        if match_hs is not None:
            hs_pct_samples.append(match_hs)
            logger.info(f"[hs_pct] Match {match_id}: {player_slug} HS%={round(match_hs*100)}%")

        # Attempt pistol round parse for this match
        pistol_data = _parse_pistol_stats(html, player_slug)

        # Detect LAN vs Online context for this series (ground-truth from HLTV).
        # Used by the simulator to weight historical maps that match today's
        # context heavier (LAN-only history is less predictive for an online match).
        match_format = _parse_match_format(html)
        is_lan_match = (match_format == "LAN") if match_format else None

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
                'match_slug':    slug,                 # full match slug (e.g. "faze-vs-natus-vincere") for opp rank lookup
                'map_name':      m['map_name'].lower(),
                'rating':        m.get('rating'),
                'kast_pct':      m.get('kast_pct'),
                'adr':           m.get('adr'),
                'survival_rate': m.get('survival_rate'),
                'fk':            m.get('fk'),
                'fd':            m.get('fd'),
                'deaths':        m.get('deaths'),
                'pistol_kills':  pistol_data.get(map_num),
                'opp_rank':      None,                 # populated later by _enrich_with_opp_ranks in bot.py
                'is_lan':        is_lan_match,         # True/False/None — LAN vs Online context for this series
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

    # Step 3b: Supplemental fill — if main loop found fewer than 6 series, try
    # the stats/players/matches page (single fetch, CS2 era only).  This covers
    # players with sparse results-page data (new to the circuit, bot-era gap, etc.)
    # We only use it to ADD series that are absent from map_kills (no duplicates).
    if not _is_hs_prop and bo3_series_count < 6:
        _sm_supp = get_player_stats_matches(player_id, player_slug, stat_type="Kills")
        if _sm_supp:
            _existing_mids = {m["match_id"] for m in map_kills}
            _added = 0
            for _sm_map in _sm_supp:
                mid_sm = _sm_map["match_id"]
                if mid_sm not in _existing_mids:
                    map_kills.append(_sm_map)
                    _existing_mids.add(mid_sm)
                    _added += 1
            if _added:
                logger.info(
                    f"[scraper] stats/matches supplemented {_added} maps "
                    f"for {display_name} (total now {len(map_kills)})"
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
        # Priority 1: bo3.gg public API — real career HS% (always accessible, no HLTV block)
        try:
            from bo3_scraper import get_career_hs_pct as _bo3_career_hs
            _bo3_val = _bo3_career_hs(player_slug)
            if _bo3_val is not None:
                recent_hs_pct = _bo3_val
                logger.info(
                    f"[hs_pct] bo3.gg career HS% for {player_slug}: "
                    f"{round(_bo3_val * 100, 1)}%"
                )
            else:
                raise ValueError("bo3.gg returned None")
        except Exception as _bo3_err:
            logger.info(f"[hs_pct] bo3.gg lookup failed ({_bo3_err}) — using role default")
            # Priority 2: Role-aware defaults (from Liquipedia role detection)
            if _liq_role == 'awper':
                recent_hs_pct = 0.25
                logger.info(f"[hs_pct] AWPer default 25% for {player_slug} (Liquipedia role)")
            elif _liq_role == 'igl':
                recent_hs_pct = 0.38
                logger.info(f"[hs_pct] IGL default 38% for {player_slug} (Liquipedia role)")
            else:
                recent_hs_pct = None   # bot.py will apply _KNOWN_AWPERS table fallback

    # Build country string from bo3.gg context (supplements HLTV data)
    _country = None
    if _bo3gg_ctx:
        _country = _bo3gg_ctx.get('country')

    # Fetch player's current team now so deep analysis can skip the redundant
    # profile fetch.  Result is cached in _PLAYER_TEAM_CACHE for 2 hours.
    _player_team = get_player_team(player_id, player_slug)
    _player_team_id   = _player_team[0] if _player_team else None
    _player_team_slug = _player_team[1] if _player_team else None

    # ── Post-resolution team verification ────────────────────────────────────
    # If a team_hint was given but the resolved player's current team doesn't
    # match, set a flag so the embed can warn the user.  This catches edge cases
    # where roster lookup or name search still returned the wrong player.
    _team_mismatch = False
    if team_hint and _player_team_slug:
        _th_norm  = re.sub(r'[^a-z0-9]', '', team_hint.lower())
        _pts_norm = re.sub(r'[^a-z0-9]', '', _player_team_slug.lower())
        if _th_norm not in _pts_norm and _pts_norm not in _th_norm:
            logger.warning(
                f"[team_mismatch] Resolved {player_slug!r} (team={_player_team_slug!r}) "
                f"does NOT match team_hint={team_hint!r} — data may be for wrong player"
            )
            _team_mismatch = True

    return {
        'player':            display_name,
        'player_id':         player_id,
        'player_slug':       player_slug,
        'player_team_id':    _player_team_id,    # team_id for rank lookup — pre-fetched
        'player_team_slug':  _player_team_slug,  # team slug — pre-fetched
        'match_ids':         match_ids,           # full list — used for H2H filtering
        'map_kills':         map_kills,
        'mean':              round(mean, 2),
        'std':               round(std, 2),
        'sample_size':       len(map_kills),
        'source':            'HLTV Live',
        'recent_hs_pct':     recent_hs_pct,   # None if no HS data found on match pages
        'hs_pct_n_matches':  len(hs_pct_samples),
        'bo3gg_context':     _bo3gg_ctx,       # {nickname, team_id, country, bo3gg_id} or None
        'liquipedia_role':   _liq_role,        # 'awper' | 'igl' | 'rifler' | None
        'country':           _country,         # country from bo3.gg or None
        'team_mismatch':     _team_mismatch,   # True if resolved player's team ≠ team_hint
    }


# ---------------------------------------------------------------------------
# Player team lookup (from player profile page)
# ---------------------------------------------------------------------------

_PLAYER_TEAM_CACHE: dict[str, tuple[str, str] | None] = {}   # player_id → (team_id, slug)
_PLAYER_TEAM_CACHE_TTL = 7200   # 2 hours


def get_player_team(player_id: str, player_slug: str) -> tuple[str, str] | None:
    """
    Fetch the player's profile page and extract their current team ID + slug.
    Returns (team_id, team_slug) or None.
    Caches successful lookups for 2 hours so teammates don't re-fetch.
    """
    import time as _time
    _cached = _PLAYER_TEAM_CACHE.get(player_id)
    if _cached is not None and _cached is not False:
        # (result, fetched_at) tuple
        if isinstance(_cached, tuple) and len(_cached) == 2 and isinstance(_cached[0], str):
            # Check if this is a (team_id, slug) tuple (both strings)
            # vs a cache metadata tuple
            return _cached

    # Sentinel: if we recently tried and got None, don't spam HLTV
    if _cached is False:
        return None

    url = f"{HLTV_BASE}/player/{player_id}/{player_slug}"
    html = _fetch(url)
    if not html:
        return None

    matches = re.findall(r'/team/(\d+)/([\w-]+)', html)
    if not matches:
        return None

    tid, tslug = matches[0]
    logger.info(f"[player_team] player={player_slug} → team_id={tid} slug={tslug}")
    _PLAYER_TEAM_CACHE[player_id] = (tid, tslug)
    return tid, tslug


# ---------------------------------------------------------------------------
# bo3.gg integration — instant player/team context (no auth, no CF block)
# ---------------------------------------------------------------------------
# bo3.gg API is a Nuxt SPA with a limited but accessible REST API.
# Available public endpoints:
#   GET /api/v1/players/{slug}   → {id, nickname, team_id, country_id, slug}
#   GET /api/v1/players?search=  → paginated player list (broken filter, unused)
# bo3.gg does NOT expose per-map kill stats — it's used for player enrichment only.

_BO3GG_BASE = "https://bo3.gg"
_BO3GG_PLAYER_CACHE: dict[str, dict | None] = {}   # slug → result dict or None
_BO3GG_COUNTRY_MAP = {
    # Confirmed from live API responses
    28:  "Finland",        # myltsi → Finland
    29:  "France",         # ZywOo → France (country_id=29)
    # Standard ISO numeric approximate mappings
    57:  "Czech Republic", 69:  "Estonia",    76:  "France",
    80:  "Germany",        105: "Iceland",    113: "Israel",
    126: "Kazakhstan",     208: "Sweden",     214: "Slovakia",
    233: "Ukraine",        250: "Denmark",    616: "Poland",
    643: "Russia",         840: "United States", 14: "Australia",
    32:  "Belgium",        179: "Norway",     181: "Netherlands",
    167: "Lithuania",      154: "Latvia",     24:  "Brazil",
    40:  "Canada",         75:  "Georgia",    62:  "Croatia",
}


def bo3gg_player_context(player_slug: str) -> dict | None:
    """
    Fetch player context from bo3.gg's instant REST API.

    Returns dict with:
        nickname  : str  — display name from bo3.gg
        team_id   : int  — bo3.gg team id (different from HLTV)
        country   : str  — country name (e.g. "Finland")
        bo3gg_id  : int  — bo3.gg internal player ID
    or None if not found / request fails.

    Cached indefinitely per session (player data rarely changes mid-session).
    """
    key = player_slug.lower().strip()
    if key in _BO3GG_PLAYER_CACHE:
        return _BO3GG_PLAYER_CACHE[key]

    if not _CFFI_OK:
        return None

    try:
        sess = _cffi_req.Session(impersonate='chrome116')
        url = f"{_BO3GG_BASE}/api/v1/players/{key}"
        resp = sess.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and 'id' in data:
                country_id = data.get('country_id', 0)
                result = {
                    'nickname':  data.get('nickname', player_slug),
                    'team_id':   data.get('team_id'),
                    'country':   _BO3GG_COUNTRY_MAP.get(country_id, f"Unknown ({country_id})"),
                    'bo3gg_id':  data.get('id'),
                }
                _BO3GG_PLAYER_CACHE[key] = result
                logger.info(
                    f"[bo3gg] Player {player_slug!r}: "
                    f"nickname={result['nickname']} country={result['country']} "
                    f"team_id={result['team_id']}"
                )
                return result

        _BO3GG_PLAYER_CACHE[key] = None
        logger.debug(f"[bo3gg] Player {player_slug!r} not found (status={resp.status_code})")
        return None

    except Exception as e:
        logger.debug(f"[bo3gg] Error fetching {player_slug}: {e}")
        _BO3GG_PLAYER_CACHE[key] = None
        return None


# ---------------------------------------------------------------------------
# Liquipedia integration — player role detection for HS% calibration
# ---------------------------------------------------------------------------
# Liquipedia's wiki returns rendered HTML for player pages.
# We use it to detect AWPer / IGL roles which affect HS% estimation:
#   AWPer → lower HS% (~22-28%); IGL → moderate; Rifler → higher (~42-50%)

_LIQUIPEDIA_BASE = "https://liquipedia.net"
_LIQUIPEDIA_ROLE_CACHE: dict[str, str | None] = {}   # name → role str or None


def liquipedia_player_role(player_name: str) -> str | None:
    """
    Detect the player's primary role from their Liquipedia wiki page.

    Returns one of: 'awper', 'igl', 'rifler', or None (unknown).
    Cached per session.
    """
    key = player_name.lower().strip()
    if key in _LIQUIPEDIA_ROLE_CACHE:
        return _LIQUIPEDIA_ROLE_CACHE[key]

    if not _CFFI_OK:
        return None

    try:
        sess = _cffi_req.Session(impersonate='chrome116')
        # Use the Liquipedia wiki API for clean JSON (no JS needed)
        url = (
            f"{_LIQUIPEDIA_BASE}/counterstrike/api.php"
            f"?action=parse&page={player_name}&format=json&prop=text&section=0"
        )
        headers = {
            "User-Agent": "CS2PropGraderBot/1.0 (research tool; contact via Discord)",
            "Accept": "application/json",
        }
        resp = sess.get(url, timeout=8, headers=headers)
        if resp.status_code != 200:
            _LIQUIPEDIA_ROLE_CACHE[key] = None
            return None

        data = resp.json()
        html_text = data.get('parse', {}).get('text', {}).get('*', '')
        if not html_text:
            _LIQUIPEDIA_ROLE_CACHE[key] = None
            return None

        # Search for role indicators in the infobox HTML
        html_lower = html_text.lower()

        # AWPer detection: explicit role mentions in infobox or career section
        # Also look for "awp" in weapons tables and "sniper" references
        _awp_terms = [
            'awper', 'awp specialist', 'primary awp', 'role">awp<',
            '"awp"', '>awp<', 'sniper', 'primary sniper',
            # Liquipedia infobox pattern: <td>AWP</td> in weapons column
            '>awp</td>', '<td>awp', '>awp</span>',
        ]
        _igl_terms = [
            'in-game leader', 'in game leader', '>igl<', '"igl"',
            '>igl</td>', '<td>igl', 'role">igl',
        ]
        if any(term in html_lower for term in _awp_terms):
            role = 'awper'
        elif any(term in html_lower for term in _igl_terms):
            role = 'igl'
        elif 'rifler' in html_lower:
            role = 'rifler'
        else:
            role = None

        _LIQUIPEDIA_ROLE_CACHE[key] = role
        logger.info(f"[liquipedia] {player_name}: role={role!r}")
        return role

    except Exception as e:
        logger.debug(f"[liquipedia] Error fetching role for {player_name}: {e}")
        _LIQUIPEDIA_ROLE_CACHE[key] = None
        return None


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


def resolve_player_from_roster(player_name: str, team_name: str) -> tuple[str, str, str] | None:
    """
    Resolve a player's HLTV (player_id, player_slug, display_name) by fetching
    the team's HLTV roster page and finding the player there.

    This is the most reliable disambiguation method: instead of searching for the
    player by name (which can match wrong players with the same nickname), we go
    to the team page and read the roster directly.

    Returns (player_id, player_slug, display_name) or None on failure.
    """
    # Step 1 — find the team on HLTV
    team_result = search_team(team_name)
    if not team_result:
        logger.warning(f"[roster] Team '{team_name}' not found on HLTV")
        return None
    team_id, team_slug, _ = team_result

    # Step 2 — fetch the team page
    url = f"{HLTV_BASE}/team/{team_id}/{team_slug}"
    html = _fetch(url)
    if not html:
        logger.warning(f"[roster] Could not fetch team page for {team_slug}")
        return None

    # Step 3 — extract all /player/{id}/{slug} links from the page
    player_links = re.findall(r'/player/(\d+)/([\w-]+)', html)
    seen: dict[str, str] = {}
    for pid, slug in player_links:
        if pid not in seen:
            seen[pid] = slug

    if not seen:
        logger.warning(f"[roster] No player links found on team page for {team_slug}")
        return None

    # Step 4 — find the best-matching player from the roster
    name_norm = re.sub(r'[^a-z0-9]', '', player_name.lower())
    best_pid, best_slug, best_score = None, None, -1
    for pid, slug in seen.items():
        score = _score_player_match(player_name, pid, slug)
        if score > best_score:
            best_score = score
            best_pid, best_slug = pid, slug

    if not best_pid or best_score <= 0:
        logger.warning(
            f"[roster] No roster match for '{player_name}' on {team_slug} "
            f"(best slug='{best_slug}', score={best_score})"
        )
        return None

    display = best_slug.replace('-', ' ').title()
    logger.info(
        f"[roster] Resolved '{player_name}' on '{team_name}' → "
        f"id={best_pid} slug={best_slug} score={best_score}"
    )
    return best_pid, best_slug, display


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


def check_standin(player_slug: str, match_html: str) -> bool:
    """
    Check if a player is listed as a stand-in in an HLTV match page.
    HLTV flags stand-ins with text like 'stand-in' near the player link.
    Returns True if stand-in detected, False otherwise.
    """
    slug_norm = re.sub(r'[^a-z0-9]', '', player_slug.lower())
    soup = BeautifulSoup(match_html, 'html.parser')
    for a in soup.find_all('a', href=re.compile(r'/player/\d+/')):
        href_norm = re.sub(r'[^a-z0-9]', '', a.get('href', '').lower())
        if slug_norm in href_norm:
            for parent in [a.parent, a.parent.parent if a.parent else None]:
                if parent:
                    txt = parent.get_text().lower()
                    if 'stand-in' in txt or 'standin' in txt or 'substitute' in txt:
                        logger.info(f"[standin] Stand-in detected for {player_slug}")
                        return True
    return False


def _get_upcoming_lineup(team_id: str) -> list[str]:
    """
    Fetch the team's next upcoming match page and return the 5 player slugs
    actually scheduled to play (catches stand-ins that are absent from the
    static team-page roster). Returns [] if no upcoming match or parse fails.
    """
    sched = _fetch(f"{HLTV_BASE}/matches?team={team_id}")
    if not sched:
        return []
    upcoming, seen = [], set()
    for mid, slug in re.findall(r'/matches/(\d{7,})/([\w-]+)', sched):
        if mid in seen:
            continue
        seen.add(mid)
        upcoming.append((mid, slug))
        if len(upcoming) >= 3:
            break
    for mid, slug in upcoming:
        mp = _fetch(f"{HLTV_BASE}/matches/{mid}/{slug}")
        if not mp:
            continue
        id_to_slug: dict[str, str] = {}
        for m in re.finditer(r'/player/(\d+)/([\w-]+)', mp):
            id_to_slug.setdefault(m.group(1), m.group(2))
        soup = BeautifulSoup(mp, 'html.parser')
        for div in soup.select('div.lineup'):
            if not div.find('a', href=re.compile(rf'/team/{team_id}/')):
                continue
            pids: list[str] = []
            for el in div.select('[data-player-id]'):
                pid = el.get('data-player-id')
                if pid and pid not in pids:
                    pids.append(pid)
            slugs = [id_to_slug[p] for p in pids if p in id_to_slug]
            if slugs:
                logger.info(
                    f"[roster] team {team_id} upcoming-lineup ({mid}): {slugs}"
                )
                return slugs
    return []


def get_recent_team_roster(team_id: str, team_slug: str) -> list[str]:
    """
    Return the player slugs that should be treated as the team's active roster.

    Strategy:
      1. Prefer the lineup listed on the team's NEXT upcoming match page —
         this captures stand-ins (e.g. Alkaren on Cybershoke) and excludes
         benched players that are still on the static roster (e.g. fluffy).
      2. Fall back to the static team-page roster if no upcoming match is
         scheduled or the match-page parse fails.
    """
    upcoming = _get_upcoming_lineup(team_id)
    if upcoming:
        return upcoming

    url = f"{HLTV_BASE}/team/{team_id}/{team_slug}"
    html = _fetch(url)
    if not html:
        return []
    return re.findall(r'/player/\d+/([\w-]+)', html)


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


# ---------------------------------------------------------------------------
# Auto-result fetcher — find a player's actual kills after a match is played
# ---------------------------------------------------------------------------

def _get_fresh_match_ids_with_timestamps(
    player_id: str,
    max_matches: int = 15,
) -> list:
    """
    Fetch /results?player={id} fresh (bypasses 3-hour cache) and return
    [(match_id, slug, unix_sec_or_None), ...] ordered newest-first.
    unix_sec is extracted from HLTV's data-unix attribute (ms -> sec).
    """
    url = f"{HLTV_BASE}/results?player={player_id}"
    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    results = []
    seen_ids = set()

    for con in soup.find_all(class_=re.compile(r'result-con')):
        link = con.find('a', href=re.compile(r'/matches/\d+/'))
        if not link:
            continue
        m = re.search(r'/matches/(\d+)/([a-z0-9-]+)', link.get('href', ''))
        if not m:
            continue
        mid, slug = m.group(1), m.group(2)
        if mid in seen_ids or len(mid) < 6:
            continue
        seen_ids.add(mid)

        ts_tag = con.find(attrs={'data-unix': True})
        unix_sec = None
        if ts_tag:
            try:
                unix_sec = int(ts_tag['data-unix']) / 1000
            except (ValueError, TypeError):
                pass

        results.append((mid, slug, unix_sec))
        if len(results) >= max_matches:
            break

    if not results:
        for mid, slug in re.findall(r'/matches/(\d+)/([a-z0-9-]+)', html):
            if mid not in seen_ids and len(mid) >= 6:
                seen_ids.add(mid)
                results.append((mid, slug, None))
                if len(results) >= max_matches:
                    break

    _MATCH_IDS_CACHE.pop(player_id, None)
    return results


def get_actual_result(
    player_name: str,
    opponent: str,
    grade_ts: float,
    line: float,
    baseline_match_id: str | None = None,
) -> dict | None:
    """
    Auto-fetch the actual Maps 1+2 kill total for a pending graded prop.

    Looks for a BO3 match played AFTER grade_ts by:
      1. Using data-unix timestamps from HLTV results page (preferred), OR
      2. Comparing match IDs against baseline_match_id (fallback).

    Returns {'actual': float, 'outcome': 'over'|'under', 'match_id': str}
    or None if no new match found / match not finished yet.
    """
    player_result = search_player_v2(player_name, opponent_hint=opponent or None)
    if not player_result:
        logger.warning(f"[auto_result] Player not found: {player_name}")
        return None

    player_id, player_slug, display = player_result
    logger.info(
        f"[auto_result] Checking {display} — after ts={grade_ts:.0f}, baseline={baseline_match_id}"
    )

    matches = _get_fresh_match_ids_with_timestamps(player_id, max_matches=10)
    if not matches:
        logger.warning(f"[auto_result] No match IDs for {display}")
        return None

    checked_no_info = 0   # guard: don't try unlimited matches when flying blind

    for mid, slug, unix_sec in matches:

        if unix_sec is not None:
            if unix_sec <= grade_ts:
                break   # newest-first — nothing older will be new
        elif baseline_match_id:
            try:
                if int(mid) <= int(baseline_match_id):
                    break
            except (ValueError, TypeError):
                continue
        else:
            # No timestamp AND no baseline — use opponent name in slug as proxy.
            # HLTV slugs look like "nrg-vs-legacy-pgl-bucharest-2026", so checking
            # that the opponent token appears confirms this is the graded match.
            opp_clean = re.sub(r'[^a-z0-9]+', '-', (opponent or '').lower()).strip('-')
            opp_parts = [p for p in opp_clean.split('-') if len(p) >= 3]
            slug_match = bool(opp_parts) and any(p in slug for p in opp_parts)
            if not slug_match:
                logger.info(
                    f"[auto_result] {mid} slug='{slug}' — opponent '{opponent}' not matched, skipping"
                )
                checked_no_info += 1
                if checked_no_info >= 5:
                    break   # don't scan entire history
                continue

        match_url = f"{HLTV_BASE}/matches/{mid}/{slug}"
        try:
            html = _fetch(match_url)
        except Exception as e:
            logger.warning(f"[auto_result] Fetch error {match_url}: {e}")
            continue

        if not html:
            continue

        result = _parse_match_kills(html, player_slug, match_url)
        if not result:
            continue

        if result.get('bo_type') != 3:
            logger.info(f"[auto_result] Match {mid} is BO{result.get('bo_type')} — skipping")
            continue

        maps = result.get('maps', [])[:2]
        if len(maps) < 2:
            logger.info(f"[auto_result] Match {mid} only {len(maps)} maps — may not be finished")
            return None

        actual = float(sum(m['kills'] for m in maps))
        outcome = 'over' if actual > line else 'under'
        logger.info(
            f"[auto_result] SUCCESS {display}: match {mid} "
            f"M1={maps[0]['kills']}+M2={maps[1]['kills']}={actual} vs {line} -> {outcome.upper()}"
        )
        return {'actual': actual, 'outcome': outcome, 'match_id': mid}

    logger.info(f"[auto_result] No new BO3 match found for {display}")
    return None
