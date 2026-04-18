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

def split_recent_vs_older(map_stats: list[dict], recent_n_series: int = 3) -> tuple[float, float, float]:
    """
    Split per-map kill list into 'recent' (last N series → 2N maps) vs 'older' buckets
    and compute trend % = (recent_avg − older_avg) / older_avg × 100.
    Returns (recent_avg, older_avg, trend_pct). Trend = 0 if either bucket is empty.
    """
    if not map_stats:
        return 0.0, 0.0, 0.0
    recent_maps = recent_n_series * 2  # M1+M2 each
    recent = [m["stat_value"] for m in map_stats[:recent_maps]]
    older  = [m["stat_value"] for m in map_stats[recent_maps:]]
    r_avg = sum(recent) / len(recent) if recent else 0.0
    o_avg = sum(older)  / len(older)  if older  else 0.0
    if not recent or not older or o_avg == 0:
        return round(r_avg, 2), round(o_avg, 2), 0.0
    trend_pct = round((r_avg - o_avg) / o_avg * 100, 1)
    return round(r_avg, 2), round(o_avg, 2), trend_pct


def per_map_breakdown(map_stats: list[dict]) -> list[dict]:
    """
    Group per-map kill samples by map name and compute avg / sample / hit-rate-ready stats.
    Returns list of dicts sorted by avg kills DESC. Useful for spotting map-veto edges.
    """
    by_map: dict[str, list[dict]] = {}
    for m in map_stats:
        nm = (m.get("map_name") or "Unknown").strip() or "Unknown"
        by_map.setdefault(nm, []).append(m)
    out = []
    for nm, rows in by_map.items():
        kills = [r["stat_value"] for r in rows]
        rounds = [r.get("rounds", 0) or 0 for r in rows]
        out.append({
            "map_name": nm,
            "n":        len(rows),
            "avg":      round(sum(kills) / len(kills), 2),
            "max":      max(kills),
            "min":      min(kills),
            "kpr":      round(sum(kills) / sum(rounds), 3) if sum(rounds) > 0 else None,
        })
    return sorted(out, key=lambda x: x["avg"], reverse=True)


def infer_role(agg: dict) -> str:
    """
    Cheap role inference from aggregate FK rate + KAST. Not perfect, but useful context.
        Duelist     — high FK rate (>= 0.13) and avg/+ ACS
        Sentinel    — low FK rate (<= 0.07) and high KAST (>= 75)
        Initiator   — mid FK rate, high KAST
        Controller  — mid FK rate, lower KAST
        Flex        — fallback
    """
    fk = (agg or {}).get("fk_rate") or 0
    kast = (agg or {}).get("kast") or 0
    acs  = (agg or {}).get("acs")  or 0
    if fk >= 0.13:
        return "🗡️ Duelist"
    if fk <= 0.07 and kast >= 75:
        return "🛡️ Sentinel"
    if 0.07 < fk < 0.13 and kast >= 73:
        return "🎯 Initiator"
    if 0.07 < fk < 0.13 and acs >= 200:
        return "🧠 Controller"
    return "🔀 Flex"


def confidence_score(
    *,
    edge: float,            # signed: over_prob − 0.5  (e.g. +0.18 = 18% edge to OVER)
    hit_rate: float,        # 0.0..1.0
    n_series: int,
    stability_std: float,   # std-dev of per-series totals
    sample_avg: float,      # series average kills (for CV calc)
    trend_pct: float,
    decision: str,          # "OVER" / "UNDER" / "PASS"
) -> int:
    """
    Universal 0–100 confidence in the recommended direction.
    Combines edge magnitude, hit-rate alignment, sample size, stability, and trend.
    """
    # 1. Edge magnitude — capped at ±30% → 60 pts max
    score = min(60.0, abs(edge) * 200)

    # 2. Hit-rate alignment with decision
    hr_align = (hit_rate - 0.5) if decision == "OVER" else (0.5 - hit_rate) if decision == "UNDER" else 0
    score += max(-10, min(15, hr_align * 30))

    # 3. Sample bonus (capped at 10 pts → fully rewarded at 10 series)
    score += min(10.0, n_series * 1.0)

    # 4. Stability (CV-based) — penalty up to 15 pts if highly volatile
    if sample_avg > 0:
        cv = stability_std / sample_avg
        score -= min(15.0, cv * 30)

    # 5. Trend alignment — bonus if trend supports decision, penalty if opposed
    if decision == "OVER":
        score += max(-10, min(10, trend_pct * 0.4))
    elif decision == "UNDER":
        score += max(-10, min(10, -trend_pct * 0.4))

    if decision == "PASS":
        score = min(score, 50)
    return max(0, min(100, int(round(score))))


def confidence_grade(score: int) -> tuple[str, str]:
    """Map 0–100 → letter grade + label."""
    if score >= 80: return "A", "🏆 High Confidence"
    if score >= 65: return "B", "💪 Solid Confidence"
    if score >= 50: return "C", "⚖️ Fair Confidence"
    if score >= 35: return "D", "⚠️ Low Confidence"
    return "F", "🚫 Unreliable"


# ─────────────────────────── team / roster scraping ─────────────────────────

def search_team(name: str) -> Optional[tuple[str, str]]:
    """
    Search vlr.gg for a team. Returns (team_id, slug) or None.
    vlr.gg search HTML embeds team URLs as `/team/<id>/idx`; we resolve idx →
    the canonical slug via a HEAD/GET to the team page.
    """
    q = name.strip().replace(" ", "+")
    if not q:
        return None
    html = _fetch(f"https://www.vlr.gg/search/?q={q}", allow_redirects=True)
    if not html:
        return None
    matches = re.findall(r"/team/(\d+)/([a-z0-9_-]+)", html)
    if not matches:
        return None
    # Prefer the first non-"idx" slug if present, else resolve idx → slug
    for tid, slug in matches:
        if slug != "idx":
            return tid, slug
    tid = matches[0][0]
    page = _fetch(f"https://www.vlr.gg/team/{tid}/idx", allow_redirects=True)
    if page:
        m = re.search(rf'/team/{tid}/([a-z0-9_-]+)', page)
        if m and m.group(1) != "idx":
            return tid, m.group(1)
    return tid, "idx"


def get_team_roster(team_id: str, slug: str = "idx") -> list[dict]:
    """
    Fetch a team's player roster from vlr.gg. Returns list of dicts:
      [{ "player_id": "...", "slug": "...", "display_name": "..." }, ...]
    Excludes coaches/managers when role is detectable. Returns empty list on error.
    """
    html = _fetch(f"https://www.vlr.gg/team/{team_id}/{slug}", allow_redirects=True)
    if not html:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="/player/(\d+)/([a-z0-9_-]+)"', html):
        pid, pslug = m.group(1), m.group(2)
        if pid in seen:
            continue
        # Look at surrounding context to filter staff
        ctx = html[max(0, m.start()-300): m.end()+400].lower()
        if any(role in ctx for role in ["manager", "head coach", "assistant coach", '"coach"']):
            continue
        # Try to extract display name
        nm = pslug
        nm_m = re.search(r'team-roster-item-name-alias[^>]*>([^<]+)<', ctx)
        if nm_m:
            nm = nm_m.group(1).strip()
        seen.add(pid)
        out.append({"player_id": pid, "slug": pslug, "display_name": nm})
    return out


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


# ─────────────────────────────────────────────────────────────────────────
# Predictive grader — multi-signal stacked model, no Monte Carlo
# ─────────────────────────────────────────────────────────────────────────
#
# Combines four independent predictive signals into a calibrated OVER prob.
# Each signal produces a z-score (signed standardised distance from the line);
# we sum them with empirical weights and pass through a logistic function.
#
#  S1  Projection z-score      weight 2.5   blended KPR × proj rounds → expected
#  S2  Median-gap z-score      weight 1.5   where the line sits vs sample median
#  S3  Recency trend           weight 1.5   recent-3 vs older-7 form swing
#  S4  Bayesian hit rate       weight 1.5   beta-smoothed empirical hit %
#
# Strengths over plain empirical hit rate:
#   • Hit rate alone is binary and high-variance on 5–10 samples → we shrink
#     it toward 50% with a Beta(2,2) prior so 4-of-5 doesn't overfit.
#   • Adds where the line sits relative to projection (a 20.5 line for a
#     guy averaging 28 is a much stronger OVER than hit rate alone shows).
#   • Captures form changes — a player coming off two huge games is
#     differently predictive than one with a flat 10-series average.
#   • Variance-aware: high-CV players get wider z-scores → softer signals
#     → naturally pulled toward PASS.
# ─────────────────────────────────────────────────────────────────────────
def empirical_grade(map_stats: list, line: float, stat_type: str = "Kills") -> dict:
    """
    Predict OVER/UNDER for a kills prop using stacked predictive signals.
    Returns the same shape as the legacy simulator output for embed compatibility.
    """
    import math
    import statistics as _st
    from statistics import mean as _mean, median as _median

    if not map_stats:
        return {"error": "No map data"}

    # ── Series-level (sum kills across Maps 1+2 per match) ───────────────
    by_match: dict = {}
    by_match_rounds: dict = {}
    for m in map_stats:
        mid = m["match_id"]
        by_match.setdefault(mid, []).append(m["stat_value"])
        by_match_rounds.setdefault(mid, []).append(m.get("rounds") or 24)
    # Preserve newest-first order from scraper
    ordered_mids = []
    for m in map_stats:
        if m["match_id"] not in ordered_mids:
            ordered_mids.append(m["match_id"])
    series_totals  = [sum(by_match[mid])         for mid in ordered_mids]
    series_rounds  = [sum(by_match_rounds[mid])  for mid in ordered_mids]

    n_series = len(series_totals)
    if n_series == 0:
        return {"error": "No series data"}

    hist_avg    = _mean(series_totals)
    hist_median = _median(series_totals)
    hist_std    = _st.stdev(series_totals) if n_series > 1 else max(hist_avg * 0.15, 1.0)
    ceiling     = max(series_totals)
    floor       = min(series_totals)

    # Floor std so signals stay finite when a player is freakishly consistent
    sigma = max(hist_std, max(hist_avg * 0.10, 1.0))

    # ── Per-map KPR for projection ───────────────────────────────────────
    valid_maps = [m for m in map_stats if (m.get("rounds") or 0) > 0]
    if valid_maps:
        kpr_all = [m["stat_value"] / m["rounds"] for m in valid_maps]
    else:
        # Fallback: assume 22 rounds/map
        kpr_all = [m["stat_value"] / 22 for m in map_stats]

    # Recent form: most recent 3 series worth of maps (~6)
    recent_n_maps = min(6, len(kpr_all))
    recent_kpr = _mean(kpr_all[:recent_n_maps]) if recent_n_maps else _mean(kpr_all)
    overall_kpr = _mean(kpr_all) if kpr_all else 0.0

    # Blend recent (60%) and overall (40%) — captures form without overfitting
    blended_kpr = 0.60 * recent_kpr + 0.40 * overall_kpr

    # Projected rounds for the next series (avg of player's historical 2-map totals)
    proj_rounds = _mean(series_rounds) if series_rounds else 44.0
    expected_total = blended_kpr * proj_rounds

    # ── Signal 1: Projection z-score (expected vs line) ──────────────────
    z_proj = (expected_total - line) / sigma

    # ── Signal 2: Median-gap z-score (where line sits in distribution) ───
    z_med = (hist_median - line) / sigma

    # ── Signal 3: Recency trend (per-series equivalent) ──────────────────
    if recent_n_maps >= 2 and len(kpr_all) > recent_n_maps:
        older_kpr = _mean(kpr_all[recent_n_maps:]) or recent_kpr
        trend_pct = ((recent_kpr - older_kpr) / max(older_kpr, 0.01)) * 100
    else:
        trend_pct = 0.0
    # Convert trend % into a z-style signal: ±15% trend ≈ ±1σ
    z_trend = trend_pct / 15.0

    # ── Signal 4: Bayesian-smoothed hit rate ─────────────────────────────
    overs  = sum(1 for v in series_totals if v >  line)
    unders = sum(1 for v in series_totals if v <  line)
    pushes = sum(1 for v in series_totals if v == line)
    # Beta(2, 2) prior → posterior mean = (overs + 2) / (n + 4)
    bayes_p = (overs + 2) / (n_series + 4)
    # Convert posterior to z-style signal: 0.5 → 0, 0.80 → ~+2
    z_hit = (bayes_p - 0.5) / 0.15

    # ── Stack signals → logit → P(OVER) ──────────────────────────────────
    W_PROJ, W_MED, W_TREND, W_HIT = 2.5, 1.5, 1.5, 1.5
    raw_score = (
        W_PROJ  * z_proj +
        W_MED   * z_med  +
        W_TREND * z_trend +
        W_HIT   * z_hit
    )
    # Calibration scalar — total max possible ≈ 7 in extremes; divide so
    # logistic input stays in a sensible range (|logit| < 4 → P in 2-98%).
    logit = raw_score / 1.6

    # Sample-size shrinkage — small N pulls probability toward 0.5
    shrink = min(1.0, n_series / 8.0)  # full strength at 8+ series
    logit *= shrink

    over_prob  = 1.0 / (1.0 + math.exp(-logit))
    under_prob = 1.0 - over_prob
    edge       = over_prob - 0.5238

    # ── Empirical reference (for display + EV tie-out) ───────────────────
    emp_over_pct = (overs / n_series) * 100 if n_series else 0.0

    # EV at standard −110 odds
    ev_over  = round(over_prob  * (100/110) - (1 - over_prob),  4)
    ev_under = round(under_prob * (100/110) - (1 - under_prob), 4)

    # Stability bucket
    cv = sigma / hist_avg if hist_avg > 0 else 1.0
    if   sigma > 8 or cv > 0.30: stability_label = "⚡ High Volatility"
    elif sigma > 5 or cv > 0.18: stability_label = "🌊 Moderate Volatility"
    else:                         stability_label = "🎯 Consistent"

    # ── Decision: dual gate (probability AND signal alignment) ───────────
    # Require P ≥ 60 / ≤ 40 AND that the projection actually agrees in sign.
    # Avoids a high P from a strong trend alone when projection contradicts.
    proj_aligned_over  = z_proj > -0.25
    proj_aligned_under = z_proj <  0.25
    decision = "PASS"
    if   over_prob  >= 0.60 and proj_aligned_over  and n_series >= 4: decision = "OVER"
    elif under_prob >= 0.60 and proj_aligned_under and n_series >= 4: decision = "UNDER"

    return {
        "stat_type":       stat_type,
        "n_samples":       len(map_stats),
        "n_series":        n_series,
        "hist_avg":        round(hist_avg, 2),
        "hist_median":     round(hist_median, 2),
        "hit_rate":        round(emp_over_pct, 1),    # raw empirical (display)
        "bayes_hit_rate":  round(bayes_p * 100, 1),   # smoothed (used in model)
        "over_prob":       round(over_prob  * 100, 1),
        "under_prob":      round(under_prob * 100, 1),
        "push_prob":       round((pushes / n_series) * 100, 1) if n_series else 0,
        "edge":            round(edge * 100, 1),
        "decision":        decision,
        "ceiling":         ceiling,
        "floor":           floor,
        "stability_std":   round(sigma, 2),
        "stability_label": stability_label,
        "ev_over":         ev_over,
        "ev_under":        ev_under,
        # ── Predictive model breakdown (transparency) ────────────────────
        "expected_total":  round(expected_total, 2),
        "blended_kpr":     round(blended_kpr, 3),
        "recent_kpr":      round(recent_kpr, 3),
        "proj_rounds":     round(proj_rounds, 1),
        "trend_pct":       round(trend_pct, 1),
        "z_projection":    round(z_proj, 2),
        "z_median":        round(z_med, 2),
        "z_trend":         round(z_trend, 2),
        "z_bayes_hit":     round(z_hit, 2),
        "model_score":     round(raw_score, 2),
        # ── Embed compatibility shims ────────────────────────────────────
        "sim_median":      round(hist_median, 2),
        "sim_std":         round(sigma, 2),
    }


# Keep a friendlier alias for new callers
predict_over_under = empirical_grade
