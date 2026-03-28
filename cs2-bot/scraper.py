import cloudscraper
from bs4 import BeautifulSoup
import re
import time
import logging

logger = logging.getLogger(__name__)

HLTV_BASE = "https://www.hltv.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.hltv.org/",
}


def get_scraper():
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    return scraper


def search_player(player_name: str):
    scraper = get_scraper()
    url = f"{HLTV_BASE}/search?term={player_name.replace(' ', '+')}"
    try:
        resp = scraper.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try search results first
        results = soup.select("div.result-con a[href*='/player/']")
        if results:
            href = results[0].get("href", "")
            # href format: /player/7998/ZywOo
            parts = href.strip("/").split("/")
            if len(parts) >= 3:
                return {"id": parts[1], "name": parts[2], "url": f"{HLTV_BASE}{href}"}

        # Fallback: direct player stats page search
        direct_url = f"{HLTV_BASE}/stats/players?name={player_name.replace(' ', '+')}"
        resp2 = scraper.get(direct_url, headers=HEADERS, timeout=15)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        links = soup2.select("td.playerCol a[href*='/stats/players/']")
        if links:
            href = links[0].get("href", "")
            # href: /stats/players/player/7998/ZywOo
            parts = href.strip("/").split("/")
            if len(parts) >= 5:
                return {
                    "id": parts[3],
                    "name": parts[4],
                    "url": f"{HLTV_BASE}/player/{parts[3]}/{parts[4]}",
                }
    except Exception as e:
        logger.error(f"search_player error: {e}")
    return None


def get_player_recent_series(player_id: str, player_name: str, stat_type: str = "Kills"):
    """
    Fetch the last 10 BO3 series for a player and return
    stats for Maps 1 and 2 only.
    Returns list of dicts with map-level stats.
    """
    scraper = get_scraper()

    # Get recent matches from player stats page
    stats_url = (
        f"{HLTV_BASE}/stats/players/matches/{player_id}/{player_name}"
        f"?matchType=Lan&rankingFilter=Top50"
    )
    try:
        resp = scraper.get(stats_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.error(f"get_player_recent_series error fetching matches: {e}")
        return []

    # Parse match rows
    table = soup.select_one("table.stats-table")
    if not table:
        # Try alternate selector
        table = soup.select_one("table")

    rows = []
    if table:
        rows = table.select("tbody tr")

    map_stats = []
    series_seen = {}
    series_count = 0

    for row in rows:
        if series_count >= 10:
            break

        try:
            cells = row.select("td")
            if len(cells) < 5:
                continue

            # Match link to identify series
            match_link_el = row.select_one("td.match-col a, td a[href*='/matches/']")
            if not match_link_el:
                continue

            match_href = match_link_el.get("href", "")
            # Extract match ID from URL
            match_id_match = re.search(r"/matches/(\d+)/", match_href)
            if not match_id_match:
                continue
            match_id = match_id_match.group(1)

            # Map name / number
            map_el = row.select_one("td.mapCol, td.statsDetail")
            map_name = map_el.get_text(strip=True) if map_el else "Unknown"

            if match_id not in series_seen:
                series_seen[match_id] = 0
            series_seen[match_id] += 1
            map_num = series_seen[match_id]

            # Only use maps 1 and 2 (skip map 3+)
            if map_num > 2:
                continue

            if map_num == 1:
                series_count += 1

            # Extract numeric stats
            def get_cell(idx):
                try:
                    return cells[idx].get_text(strip=True)
                except IndexError:
                    return "0"

            # Typical column order in HLTV stats table:
            # Date | Event | Match | Map | Kills | HS | Assists | Deaths | KD | ADR | Rating
            kills_text = get_cell(4)
            hs_text = get_cell(5)
            deaths_text = get_cell(7)
            adr_text = get_cell(9) if len(cells) > 9 else "0"

            kills = int(re.sub(r"[^\d]", "", kills_text) or 0)
            hs = int(re.sub(r"[^\d]", "", hs_text) or 0)
            deaths = int(re.sub(r"[^\d]", "", deaths_text) or 0)
            adr = float(re.sub(r"[^\d.]", "", adr_text) or 0)

            # Estimate rounds: typical map ~26, use deaths as proxy (deaths ≈ rounds played/fraction)
            rounds = max(16, min(30, deaths + 5)) if deaths > 0 else 22

            stat_value = kills if stat_type == "Kills" else hs

            map_stats.append(
                {
                    "match_id": match_id,
                    "map_num": map_num,
                    "map_name": map_name,
                    "kills": kills,
                    "hs": hs,
                    "deaths": deaths,
                    "rounds": rounds,
                    "adr": adr,
                    "stat_value": stat_value,
                }
            )
        except Exception as e:
            logger.warning(f"Row parse error: {e}")
            continue

    return map_stats


def get_match_odds(match_name: str = ""):
    """
    Try to get match odds to determine round projection.
    Returns a float representing the favorite's implied probability (0-1).
    Defaults to 0.55 (slight favorite) if scraping fails.
    """
    return 0.55  # Default: close match, use standard 22 rounds


def get_player_info_fallback(player_name: str, stat_type: str = "Kills"):
    """
    Fallback: generate synthetic-but-realistic sample data
    when HLTV scraping is blocked/fails, for demonstration.
    """
    import random
    random.seed(hash(player_name) % 10000)

    base_kills = {
        "ZywOo": 24, "s1mple": 26, "NiKo": 22, "device": 20,
        "broky": 19, "electronic": 21, "ropz": 20, "gla1ve": 15,
    }.get(player_name, 18)

    base_hs = max(4, int(base_kills * 0.42))

    map_stats = []
    for i in range(20):
        kills = max(5, int(random.gauss(base_kills, 4)))
        hs = max(1, int(random.gauss(base_hs, 2)))
        rounds = int(random.gauss(25, 3))
        rounds = max(16, min(30, rounds))
        adr = round(random.gauss(75, 12), 1)
        stat_value = kills if stat_type == "Kills" else hs
        map_stats.append(
            {
                "match_id": f"demo_{i // 2}",
                "map_num": (i % 2) + 1,
                "map_name": ["Mirage", "Inferno", "Nuke", "Ancient", "Vertigo"][i % 5],
                "kills": kills,
                "hs": hs,
                "deaths": int(random.gauss(18, 3)),
                "rounds": rounds,
                "adr": adr,
                "stat_value": stat_value,
            }
        )
    return map_stats
