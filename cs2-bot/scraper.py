"""
HLTV scraper — cloudscraper with custom User-Agent as primary (Cloudflare bypass),
curl_cffi Chrome impersonation as secondary fallback.
"""

import re
import time
import random
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HLTV_BASE = "https://www.hltv.org"

# Custom User-Agent strings that closely mimic real Chrome browser requests.
# Rotating across multiple helps avoid pattern-based fingerprinting.
CUSTOM_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.128 Safari/537.36",
]

# Full browser-like headers sent alongside the custom User-Agent
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "DNT": "1",
}


def _build_headers() -> dict:
    """Return a headers dict with a randomly chosen custom User-Agent."""
    h = dict(BROWSER_HEADERS)
    h["User-Agent"] = random.choice(CUSTOM_USER_AGENTS)
    return h


FETCH_TIMEOUT = 15  # Hard 15-second timeout per request — exceeding this falls back immediately


def _make_cloudscraper():
    """
    Build a cloudscraper session configured to impersonate Chrome on Windows,
    then inject our custom User-Agent on top.
    """
    import cloudscraper
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "mobile": False,
        },
        delay=3,          # reduced so it fits within the 10s timeout budget
    )
    # Override the UA that cloudscraper sets internally
    scraper.headers.update({"User-Agent": random.choice(CUSTOM_USER_AGENTS)})
    return scraper


def _make_curl_session():
    """Build a curl_cffi session impersonating Chrome (TLS fingerprint level)."""
    from curl_cffi import requests as curl_requests
    return curl_requests.Session(impersonate="chrome124")


def _is_blocked(resp) -> bool:
    """Return True if the response looks like a Cloudflare challenge/block."""
    if resp.status_code in (403, 503, 429):
        return True
    body = resp.text or ""
    return (
        "Just a moment" in body
        or "cf-browser-verification" in body
        or "Enable JavaScript and cookies to continue" in body
        or "Checking your browser" in body
    )


def _fetch(url: str, retries: int = 3, delay: float = 2.5) -> str | None:
    """
    Fetch a URL with Cloudflare bypass.

    Strategy:
      1. cloudscraper + custom User-Agent  (primary)
      2. curl_cffi Chrome impersonation    (fallback if CF still blocks)

    Returns the HTML string or None on total failure.
    """
    headers = _build_headers()

    # --- Primary: cloudscraper + custom User-Agent ---
    try:
        scraper = _make_cloudscraper()
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[cloudscraper] {url} (attempt {attempt}/{retries})")
                resp = scraper.get(url, headers=headers, timeout=FETCH_TIMEOUT)

                if _is_blocked(resp):
                    logger.warning(f"[cloudscraper] CF block on attempt {attempt} — status {resp.status_code}")
                    if attempt < retries:
                        # Rotate UA before next attempt
                        scraper.headers.update({"User-Agent": random.choice(CUSTOM_USER_AGENTS)})
                        time.sleep(delay * attempt + random.uniform(1.5, 4.0))
                    continue

                if resp.status_code != 200:
                    logger.warning(f"[cloudscraper] HTTP {resp.status_code}")
                    if attempt < retries:
                        time.sleep(delay)
                    continue

                logger.info(f"[cloudscraper] OK — {len(resp.text):,} chars")
                return resp.text

            except Exception as e:
                logger.warning(f"[cloudscraper] Timed out or errored on attempt {attempt}: {e}")
                return None  # bail immediately on timeout — don't retry, go to fallback

    except ImportError:
        logger.warning("cloudscraper not installed — skipping to curl_cffi")
    except Exception as e:
        logger.warning(f"cloudscraper session error: {e}")

    # --- Fallback: curl_cffi Chrome TLS impersonation ---
    try:
        curl_session = _make_curl_session()
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[curl_cffi] {url} (attempt {attempt}/{retries})")
                resp = curl_session.get(url, headers=headers, timeout=FETCH_TIMEOUT)

                if _is_blocked(resp):
                    logger.warning(f"[curl_cffi] CF block on attempt {attempt}")
                    if attempt < retries:
                        time.sleep(delay * attempt + random.uniform(1.0, 3.0))
                    continue

                if resp.status_code != 200:
                    logger.warning(f"[curl_cffi] HTTP {resp.status_code}")
                    if attempt < retries:
                        time.sleep(delay)
                    continue

                logger.info(f"[curl_cffi] OK — {len(resp.text):,} chars")
                return resp.text

            except Exception as e:
                logger.warning(f"[curl_cffi] Timed out or errored on attempt {attempt}: {e}")
                return None  # bail immediately on timeout

    except ImportError:
        logger.error("curl_cffi not installed either — no HTTP client available")
    except Exception as e:
        logger.error(f"curl_cffi session error: {e}")

    logger.error(f"All fetch attempts failed for {url}")
    return None


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def search_player(player_name: str) -> dict | None:
    """
    Resolve a player name to their HLTV numeric ID and URL slug.
    Tries the stats search endpoint first, then the global search.
    """
    # Approach 1: stats player list search
    url = f"{HLTV_BASE}/stats/players?name={player_name.replace(' ', '+')}"
    html = _fetch(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Links like /stats/players/player/7998/ZywOo
        links = soup.select("td.playerCol a[href*='/stats/players/player/']")
        if not links:
            links = soup.select("a[href*='/stats/players/player/']")
        for link in links:
            href = link.get("href", "")
            m = re.search(r"/stats/players/player/(\d+)/([^/?#]+)", href)
            if m:
                return {"id": m.group(1), "name": m.group(2), "url": f"{HLTV_BASE}/player/{m.group(1)}/{m.group(2)}"}

    time.sleep(1.5)

    # Approach 2: global search
    url2 = f"{HLTV_BASE}/search?term={player_name.replace(' ', '+')}"
    html2 = _fetch(url2)
    if html2:
        soup2 = BeautifulSoup(html2, "html.parser")
        links2 = soup2.select("a[href*='/player/']")
        for link in links2:
            href = link.get("href", "")
            m = re.search(r"/player/(\d+)/([^/?#]+)", href)
            if m:
                return {"id": m.group(1), "name": m.group(2), "url": f"{HLTV_BASE}/player/{m.group(1)}/{m.group(2)}"}

    logger.warning(f"Could not find player: {player_name}")
    return None


# ---------------------------------------------------------------------------
# Match stats scraping
# ---------------------------------------------------------------------------

def _parse_stat_cell(cells: list, idx: int) -> str:
    try:
        return cells[idx].get_text(strip=True)
    except IndexError:
        return "0"


def _safe_int(text: str) -> int:
    cleaned = re.sub(r"[^\d]", "", text)
    return int(cleaned) if cleaned else 0


def _safe_float(text: str) -> float:
    cleaned = re.sub(r"[^\d.]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def get_player_recent_series(player_id: str, player_slug: str, stat_type: str = "Kills") -> list:
    """
    Fetch last 10 BO3 series for a player (Maps 1 & 2 only) from HLTV.
    Returns a list of per-map stat dicts.

    HLTV stats match table columns (0-indexed):
      0: Date  1: Event  2: Match (link)  3: Map  4: Kills  5: HS
      6: Assists  7: Deaths  8: K/D  9: ADR  10: Rating
    """
    url = (
        f"{HLTV_BASE}/stats/players/matches/{player_id}/{player_slug}"
        f"?matchType=Lan&rankingFilter=Top50"
    )
    html = _fetch(url, retries=3, delay=2.0)

    if not html:
        logger.warning("Failed to fetch player match stats page")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Find the stats table
    table = (
        soup.select_one("table.stats-table")
        or soup.select_one("table#matchesTable")
        or soup.select_one("div.stats-table table")
        or soup.select_one("table")
    )

    if not table:
        logger.warning("No stats table found on HLTV page")
        return []

    rows = table.select("tbody tr")
    if not rows:
        rows = table.select("tr")[1:]  # skip header

    logger.info(f"Found {len(rows)} row(s) in match stats table")

    # Track series: match_id -> count of maps seen so far
    series_map_count: dict[str, int] = {}
    series_order: list[str] = []  # ordered list of unique match IDs seen
    map_stats: list[dict] = []

    for row in rows:
        cells = row.select("td")
        if len(cells) < 5:
            continue

        # --- Extract match ID from any link in the row ---
        match_id = None
        for cell in cells:
            link = cell.select_one("a[href*='/matches/']")
            if link:
                m = re.search(r"/matches/(\d+)/", link.get("href", ""))
                if m:
                    match_id = m.group(1)
                    break

        if not match_id:
            continue

        # Track series order
        if match_id not in series_map_count:
            series_map_count[match_id] = 0
            series_order.append(match_id)

        # Stop once we have 10 series
        if len(series_order) > 10:
            break

        series_map_count[match_id] += 1
        map_num = series_map_count[match_id]

        # Only include maps 1 and 2 (exclude map 3+)
        if map_num > 2:
            continue

        # --- Map name ---
        map_cell = None
        for cell in cells:
            if cell.get("class") and any(
                c in ["mapCol", "statsDetail", "map-td"] for c in cell.get("class", [])
            ):
                map_cell = cell
                break
        map_name = map_cell.get_text(strip=True) if map_cell else cells[3].get_text(strip=True)

        # --- Stat extraction ---
        # Try to detect column positions dynamically via header
        header_row = table.select_one("thead tr")
        col_map = {}
        if header_row:
            for i, th in enumerate(header_row.select("th")):
                txt = th.get_text(strip=True).lower()
                col_map[txt] = i

        def get_col(keys: list[str], default_idx: int) -> str:
            for k in keys:
                if k in col_map:
                    return _parse_stat_cell(cells, col_map[k])
            return _parse_stat_cell(cells, default_idx)

        kills = _safe_int(get_col(["kills", "k"], 4))
        hs    = _safe_int(get_col(["hs", "headshots", "hsk"], 5))
        deaths = _safe_int(get_col(["deaths", "d"], 7))
        adr   = _safe_float(get_col(["adr"], 9))

        # Estimate rounds from deaths (proxy: most players die ~0.65/round)
        if deaths > 0:
            rounds = max(16, min(30, int(deaths / 0.65)))
        else:
            rounds = 22

        stat_value = kills if stat_type == "Kills" else hs

        map_stats.append({
            "match_id": match_id,
            "map_num": map_num,
            "map_name": map_name,
            "kills": kills,
            "hs": hs,
            "deaths": deaths,
            "rounds": rounds,
            "adr": adr,
            "stat_value": stat_value,
        })

    logger.info(
        f"Parsed {len(map_stats)} maps across {len([m for m in series_order if series_map_count[m] > 0])} series"
    )
    return map_stats


# ---------------------------------------------------------------------------
# Match odds
# ---------------------------------------------------------------------------

def get_match_odds(player_name: str = "") -> float:
    """
    Attempt to fetch upcoming match odds from HLTV for round projection.
    Returns the implied probability (0–1) of the stronger side.
    Defaults to 0.55 (slight favourite) if scraping fails.
    """
    try:
        html = _fetch(f"{HLTV_BASE}/matches", retries=2, delay=1.5)
        if not html:
            return 0.55

        soup = BeautifulSoup(html, "html.parser")
        # Look for odds in upcoming match rows
        # HLTV shows odds as percentages in team-rating or odds spans
        odds_spans = soup.select("span.odd, span.oddsCell, div.odds")
        if odds_spans:
            values = []
            for span in odds_spans[:6]:
                txt = span.get_text(strip=True).replace("%", "")
                try:
                    v = float(txt) / 100.0
                    if 0.4 <= v <= 0.9:
                        values.append(v)
                except ValueError:
                    pass
            if values:
                return max(values)

    except Exception as e:
        logger.warning(f"get_match_odds error: {e}")

    return 0.55


# ---------------------------------------------------------------------------
# Fallback data generator
# ---------------------------------------------------------------------------

def get_player_info_fallback(player_name: str, stat_type: str = "Kills") -> list:
    """
    Generate realistic sample data when HLTV is unavailable.
    Uses seeded randomness so the same player always gets the same baseline.
    """
    import random as rnd

    rnd.seed(hash(player_name.lower()) % 100_000)

    baselines = {
        "zywoo": (23, 10), "s1mple": (25, 10), "niko": (21, 9),
        "device": (19, 8), "broky": (18, 8), "electronic": (20, 9),
        "ropz": (19, 8),   "gla1ve": (14, 6), "blameF": (19, 8),
        "jame": (15, 6),   "sh1ro": (20, 9),  "ax1le": (21, 9),
    }
    base_kills, base_hs = baselines.get(player_name.lower(), (17, 7))

    map_stats = []
    for i in range(20):  # 10 series × 2 maps
        kills  = max(5,  int(rnd.gauss(base_kills, 4)))
        hs     = max(1,  int(rnd.gauss(base_hs, 2)))
        deaths = max(8,  int(rnd.gauss(18, 3)))
        rounds = max(16, min(30, int(rnd.gauss(25, 2))))
        adr    = round(rnd.gauss(78, 12), 1)

        stat_value = kills if stat_type == "Kills" else hs
        map_stats.append({
            "match_id": f"est_{i // 2}",
            "map_num": (i % 2) + 1,
            "map_name": ["Mirage", "Inferno", "Nuke", "Ancient", "Vertigo"][i % 5],
            "kills": kills,
            "hs": hs,
            "deaths": deaths,
            "rounds": rounds,
            "adr": adr,
            "stat_value": stat_value,
        })
    return map_stats
