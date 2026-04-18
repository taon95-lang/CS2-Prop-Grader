"""
Valorant scraper — pulls last 10 BO3 series (Maps 1 & 2 only) from vlr.gg.

Mirrors the public interface of `scraper.py` (CS2/HLTV) so the bot can plug
Valorant data into the same simulator + grade engine.

Public API:
    search_player(name)                 -> (player_id, slug, display_name) | None
    get_player_info(name, stat_type)    -> dict with map_stats list | None
    get_player_recent_kills(name, n=10) -> shortcut: only returns kills

Strategy:
    * Free-tier ScraperAPI fallback if direct fetch hits 403/cloudflare.
    * Cache match HTML for 30 min to keep ScraperAPI credit usage low.
    * BO3 detection = "Bo3" string present in match HTML AND >=2 content map blocks.
"""

from __future__ import annotations

import logging
import os
import re
import time
import requests
from typing import Optional

logger = logging.getLogger("valorant_scraper")

VLR_BASE = "https://www.vlr.gg"
HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# in-memory caches
_PAGE_CACHE: dict[str, tuple[float, str]] = {}
_PAGE_TTL = 30 * 60  # 30 min


# ─────────────────────────── fetch helpers ──────────────────────────────────

def _fetch(url: str, *, allow_redirects: bool = True) -> Optional[str]:
    """Direct fetch with vlr-friendly headers; ScraperAPI fallback on 403/Cloudflare."""
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and now - cached[0] < _PAGE_TTL:
        return cached[1]

    try:
        r = requests.get(url, headers=HDR, timeout=20, allow_redirects=allow_redirects)
        if r.status_code == 200 and len(r.text) > 1500 and "Just a moment" not in r.text:
            _PAGE_CACHE[url] = (now, r.text)
            return r.text
        if r.status_code in (403, 429, 503):
            logger.info(f"[vlr] {r.status_code} on {url[-60:]} — trying ScraperAPI")
        else:
            logger.warning(f"[vlr] status={r.status_code} len={len(r.text)} url={url[-60:]}")
    except Exception as e:
        logger.warning(f"[vlr] {type(e).__name__}: {str(e)[:120]}")

    # ── ScraperAPI fallback ───────────────────────────────────────────────
    key = os.environ.get("SCRAPERAPI_KEY", "").strip()
    if not key:
        return None
    try:
        api_url = "https://api.scraperapi.com/"
        params = {"api_key": key, "url": url, "keep_headers": "true"}
        logger.info(f"[vlr-scraperapi] {url[-70:]}")
        r = requests.get(api_url, params=params, headers=HDR, timeout=70)
        if r.status_code == 200 and len(r.text) > 1500 and "Just a moment" not in r.text:
            _PAGE_CACHE[url] = (now, r.text)
            return r.text
        logger.warning(f"[vlr-scraperapi] status={r.status_code}")
    except Exception as e:
        logger.warning(f"[vlr-scraperapi] {type(e).__name__}: {str(e)[:120]}")
    return None


# ─────────────────────────── search ─────────────────────────────────────────

def search_player(name: str) -> Optional[tuple[str, str, str]]:
    """
    Look up a Valorant player on vlr.gg.

    Returns (player_id, slug, display_name) or None.
    """
    if not name:
        return None
    q = name.strip()
    url = f"{VLR_BASE}/search?q={requests.utils.quote(q)}&type=players"
    html = _fetch(url)
    if not html:
        logger.warning(f"[vlr search] failed for {name!r}")
        return None

    # Each search result = <a href="/search/r/player/<id>/idx" ...><name></div>
    items = re.findall(
        r'<a href="/search/r/player/(\d+)/idx"[^>]*class="[^"]*search-item[^"]*"[^>]*>'
        r'(.*?)</a>',
        html, re.S,
    )
    if not items:
        logger.warning(f"[vlr search] no player results for {name!r}")
        return None

    # Pull display name out of first match: text inside an inner <div>...</div>
    pid, body = items[0]
    nm_match = re.search(r'>\s*([A-Za-z0-9_\-\.\u0080-\uffff][^<>]{0,40})\s*<', body)
    display_name = (nm_match.group(1).strip() if nm_match else name).strip()

    # Resolve canonical slug by visiting the redirect URL or constructing the player URL
    redirect_url = f"{VLR_BASE}/search/r/player/{pid}/idx"
    try:
        r = requests.get(redirect_url, headers=HDR, timeout=15, allow_redirects=False)
        loc = r.headers.get("Location", "")
        m = re.search(r"/player/(\d+)/([^/?#]+)", loc)
        if m:
            slug = m.group(2)
        else:
            slug = display_name.lower().replace(" ", "")
    except Exception:
        slug = display_name.lower().replace(" ", "")

    logger.info(f"[vlr search] {name!r} -> {display_name} (id={pid}, slug={slug})")
    return (pid, slug, display_name)


# ─────────────────────────── match-page parsing ─────────────────────────────

def _extract_game_blocks(html: str) -> list[tuple[str, str]]:
    """
    Return [(game_id, block_html), ...] for the actual content blocks
    (i.e. the LARGEST occurrence per game_id; the small ones are tab headers).
    Sorted by document order.
    """
    positions = [
        (m.start(), m.group(1))
        for m in re.finditer(
            r'<div[^>]+class="vm-stats-game[^"]*"[^>]+data-game-id="(\d+)"',
            html,
        )
    ]
    if not positions:
        return []
    # Take the LAST occurrence per game-id (the content block, not the tab)
    seen: dict[str, int] = {}
    for pos, gid in positions:
        seen[gid] = pos
    ordered = sorted(seen.items(), key=lambda x: x[1])
    blocks: list[tuple[str, str]] = []
    for i, (gid, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(html)
        block = html[start:end]
        # Skip tiny blocks (those are still tabs, not content)
        if len(block) < 5000:
            continue
        blocks.append((gid, block))
    return blocks


def _first_num(cell_html: str, *, allow_neg: bool = False) -> Optional[float]:
    """Extract the first numeric value (int or %) from a vlr stat cell."""
    txt = re.sub(r"<[^>]+>", " ", cell_html)
    pattern = r"-?\d+\.?\d*" if allow_neg else r"\d+\.?\d*"
    m = re.search(pattern, txt)
    return float(m.group(0)) if m else None


def _player_stats_in_block(block: str, slug: str) -> Optional[dict]:
    """
    Find the player's row in the first overview table of `block` and return
    a dict of all available per-map stats.

    Column layout (after name/agent): R, ACS, K, D, A, +/–, KAST, ADR, HS%, FK, FD, +/–
    Cell format = "<both> <attack> <defense>" — we use the "both" value.
    """
    table_m = re.search(
        r'<table[^>]+class="[^"]*wf-table-inset[^"]*"[^>]*>(.*?)</table>',
        block, re.S,
    )
    if not table_m:
        return None
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_m.group(1), re.S)
    if len(rows) < 2:
        return None

    target_slug = slug.lower()
    for row in rows[1:]:
        href_m = re.search(r'href="/player/(\d+)/([^"]+)"', row)
        if not href_m or href_m.group(2).lower() != target_slug:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(tds) < 13:
            return None
        return {
            "rating":  _first_num(tds[2]),
            "acs":     _first_num(tds[3]),
            "kills":   _first_num(tds[4]),
            "deaths":  _first_num(tds[5]),
            "assists": _first_num(tds[6]),
            "plus_minus": _first_num(tds[7], allow_neg=True),
            "kast":    _first_num(tds[8]),
            "adr":     _first_num(tds[9]),
            "hs_pct":  _first_num(tds[10]),
            "fk":      _first_num(tds[11]),
            "fd":      _first_num(tds[12]),
        }
    return None


def _player_kills_in_block(block: str, slug: str) -> Optional[int]:
    """Backwards-compat shim — returns just kills."""
    s = _player_stats_in_block(block, slug)
    if s is None or s.get("kills") is None:
        return None
    return int(s["kills"])


def _map_rounds_in_block(block: str) -> int:
    """
    Return total rounds played on this map (e.g. "13-7" → 20).
    Falls back to 24 if the score can't be parsed.
    """
    # vlr structure inside a vm-stats-game block:
    #   ...team-name... <div class="score">13</div> ... <div class="score">7</div> ...
    scores = re.findall(r'class="score"[^>]*>\s*(\d+)\s*<', block)
    if len(scores) >= 2:
        # Take the two largest — these are the team final scores
        nums = sorted((int(s) for s in scores), reverse=True)[:2]
        total = sum(nums)
        if 12 <= total <= 60:  # sanity bound (Valorant: min 13-0=13, OT can stretch)
            return total
    # Fallback: scan first 600 chars after the data-game-id line for two integers
    head = block[:1500]
    head_text = re.sub(r"<[^>]+>", " ", head)
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", head_text)]
    cand = [n for n in nums if 0 <= n <= 25]
    if len(cand) >= 2:
        # First two are usually team final scores
        total = cand[0] + cand[1]
        if 12 <= total <= 60:
            return total
    return 24  # safe default for a non-OT Valorant map


def _map_name_in_block(block: str) -> str:
    m = re.search(r'<div class="map"[^>]*>(.*?)</div>', block, re.S)
    if not m:
        return "Map"
    txt = re.sub(r"<[^>]+>", " ", m.group(1))
    txt = re.sub(r"\s+", " ", txt).strip()
    # strip "PICK" / "DECIDER" suffix tags
    txt = re.sub(r"\b(PICK|DECIDER|BAN)\b", "", txt, flags=re.I).strip()
    return txt or "Map"


def _is_bo3(html: str) -> bool:
    return "Bo3" in html


# ─────────────────────────── matches list ───────────────────────────────────

def _fetch_match_list(player_id: str, slug: str) -> list[tuple[str, str]]:
    """Return [(match_id, match_url_path), ...] newest first from /player/matches/."""
    url = f"{VLR_BASE}/player/matches/{player_id}/{slug}"
    html = _fetch(url)
    if not html:
        return []

    pairs = re.findall(
        r'href="/(\d{5,7})/([a-z0-9\-]+)"\s+class="[^"]*wf-card',
        html,
    )
    # de-dupe preserving order
    out: list[tuple[str, str]] = []
    seen = set()
    for mid, mslug in pairs:
        if mid in seen:
            continue
        seen.add(mid)
        out.append((mid, mslug))
    return out


# ─────────────────────────── public: get_player_info ────────────────────────

def get_player_info(name: str, stat_type: str = "Kills", n_series: int = 10) -> Optional[dict]:
    """
    Returns dict mirroring the CS2 scraper output:
        {
            "player_id":   str,
            "slug":        str,
            "display_name": str,
            "team":        None,           # vlr team detection skipped for now
            "country":     None,
            "stat_type":   "Kills",
            "map_stats":   [
                {"match_id": str, "map_num": 1|2, "map_name": str, "stat_value": int},
                ...
            ],
        }

    Only returns the last `n_series` BO3 series (Maps 1 & 2 each → 2*N entries).
    """
    if stat_type.lower() != "kills":
        logger.warning(f"[vlr] unsupported stat_type {stat_type!r} — defaulting to Kills")

    found = search_player(name)
    if not found:
        return None
    pid, slug, display = found

    matches = _fetch_match_list(pid, slug)
    if not matches:
        logger.warning(f"[vlr] no matches found for {display}")
        return {
            "player_id": pid, "slug": slug, "display_name": display,
            "team": None, "country": None, "stat_type": "Kills",
            "map_stats": [],
        }

    map_stats: list[dict] = []
    bo3_seen = 0
    matches_examined = 0
    for mid, mslug in matches:
        if bo3_seen >= n_series:
            break
        if matches_examined >= n_series * 4:  # safety bound
            break
        matches_examined += 1
        match_url = f"{VLR_BASE}/{mid}/{mslug}"
        html = _fetch(match_url)
        if not html:
            continue
        if not _is_bo3(html):
            continue
        blocks = _extract_game_blocks(html)
        if len(blocks) < 2:
            continue
        # Map 1 + Map 2 only
        candidate_rows: list[dict] = []
        skip_series = False
        for i, (gid, block) in enumerate(blocks[:2]):
            stats = _player_stats_in_block(block, slug)
            if stats is None or stats.get("kills") is None:
                logger.info(f"[vlr] {display} not in match {mid} — skipping series")
                skip_series = True
                break
            candidate_rows.append({
                "match_id":   mid,
                "map_num":    i + 1,
                "map_name":   _map_name_in_block(block),
                "rounds":     _map_rounds_in_block(block),
                "stat_value": int(stats["kills"]),
                # Full per-map stat dict (kept under "stats" for downstream aggregation)
                "stats":      stats,
            })
        if skip_series:
            continue
        map_stats.extend(candidate_rows)
        bo3_seen += 1
        logger.info(
            f"[vlr] series {bo3_seen} (match {mid}): "
            + " + ".join(
                f"{m['stat_value']}K/{m['rounds']}r ({m['map_name']})"
                for m in candidate_rows
            )
        )

    return {
        "player_id":    pid,
        "slug":         slug,
        "display_name": display,
        "team":         None,
        "country":      None,
        "stat_type":    "Kills",
        "map_stats":    map_stats,
    }


def get_player_recent_kills(name: str, n: int = 10) -> Optional[dict]:
    """Convenience alias matching CS2 scraper naming."""
    return get_player_info(name, stat_type="Kills", n_series=n)


# ─────────────────────────── analytics aggregator ───────────────────────────

def aggregate_stats(map_stats: list[dict]) -> dict:
    """
    Aggregate the per-map stats dicts into player-level analytics.

    Returns dict with keys:
        kpr   – kills per round       (rate)
        dpr   – deaths per round
        apr   – assists per round
        kd    – kills / deaths ratio
        acs   – avg ACS                (per map)
        adr   – avg ADR
        kast  – avg KAST %
        hs_pct – avg HS%
        rating – avg vlr rating
        fk_rate / fd_rate – first-kill / first-death per round
        fk_share – % of player's first-duel attempts won
        n_maps, n_rounds
    """
    rows = [m for m in map_stats if m.get("stats")]
    if not rows:
        return {}

    total_rounds = sum(m.get("rounds", 0) or 0 for m in rows)
    total_kills  = sum((m["stats"].get("kills")  or 0) for m in rows)
    total_deaths = sum((m["stats"].get("deaths") or 0) for m in rows)
    total_assist = sum((m["stats"].get("assists") or 0) for m in rows)
    total_fk     = sum((m["stats"].get("fk")     or 0) for m in rows)
    total_fd     = sum((m["stats"].get("fd")     or 0) for m in rows)

    def _avg(key: str) -> Optional[float]:
        vals = [m["stats"].get(key) for m in rows if m["stats"].get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    fk_attempts = total_fk + total_fd
    return {
        "n_maps":   len(rows),
        "n_rounds": total_rounds,
        "kpr":      round(total_kills / total_rounds, 3) if total_rounds else None,
        "dpr":      round(total_deaths / total_rounds, 3) if total_rounds else None,
        "apr":      round(total_assist / total_rounds, 3) if total_rounds else None,
        "kd":       round(total_kills / total_deaths, 2) if total_deaths else None,
        "acs":      _avg("acs"),
        "adr":      _avg("adr"),
        "kast":     _avg("kast"),
        "hs_pct":   _avg("hs_pct"),
        "rating":   _avg("rating"),
        "fk_rate":  round(total_fk / total_rounds, 3) if total_rounds else None,
        "fd_rate":  round(total_fd / total_rounds, 3) if total_rounds else None,
        "fk_share": round(100 * total_fk / fk_attempts, 1) if fk_attempts else None,
        "total_kills":  total_kills,
        "total_deaths": total_deaths,
    }


# ─────────────────────────── self-test ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    info = get_player_info("tenz", n_series=10)
    if not info:
        print("No info")
    else:
        print(f"Player: {info['display_name']} (id={info['player_id']}, slug={info['slug']})")
        series: dict[str, list] = {}
        for m in info["map_stats"]:
            series.setdefault(m["match_id"], []).append(m)
        print(f"Series found: {len(series)}")
        for i, (mid, maps) in enumerate(series.items(), 1):
            total = sum(m["stat_value"] for m in maps)
            per = " + ".join(f"{m['stat_value']} ({m['map_name']})" for m in maps)
            print(f"  {i:2d}. match {mid}: {total} kills  ·  {per}")
