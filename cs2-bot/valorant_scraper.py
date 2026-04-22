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

# Opponent caches (team rating + roster lookup)
_TEAM_RATING_CACHE: dict[str, tuple[float, int]] = {}   # team_href → (ts, rating)
_TEAM_ROSTER_CACHE: dict[str, tuple[float, set[str]]] = {}  # team_href → (ts, {player_slugs})
_TEAM_TTL = 60 * 60  # 1 hour


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

def _resolve_player_slug(pid: str, fallback_name: str) -> str:
    """Resolve canonical slug via search redirect; fall back to lowercased name."""
    redirect_url = f"{VLR_BASE}/search/r/player/{pid}/idx"
    try:
        r = requests.get(redirect_url, headers=HDR, timeout=15, allow_redirects=False)
        loc = r.headers.get("Location", "")
        m = re.search(r"/player/(\d+)/([^/?#]+)", loc)
        if m:
            return m.group(2)
    except Exception:
        pass
    return fallback_name.lower().replace(" ", "")


def search_player_candidates(name: str, limit: int = 8) -> list[tuple[str, str, str]]:
    """
    Return a ranked list of (player_id, slug, display_name) candidates from
    vlr.gg's player search.  vlr.gg's search treats short queries fuzzily and
    often surfaces unrelated names first (e.g. "Lucas" → "Rojo"), so the
    caller should iterate this list rather than blindly take the first.

    The list is filtered to candidates whose display name contains the
    query as a substring (case-insensitive); if that yields nothing we
    return the unfiltered top results so we still try.
    """
    if not name:
        return []
    q = name.strip()
    url = f"{VLR_BASE}/search?q={requests.utils.quote(q)}&type=players"
    html = _fetch(url)
    if not html:
        logger.warning(f"[vlr search] failed for {name!r}")
        return []

    items = re.findall(
        r'<a href="/search/r/player/(\d+)/idx"[^>]*class="[^"]*search-item[^"]*"[^>]*>'
        r'(.*?)</a>',
        html, re.S,
    )
    if not items:
        logger.warning(f"[vlr search] no player results for {name!r}")
        return []

    candidates: list[tuple[str, str, str]] = []
    for pid, body in items[:limit]:
        nm_match = re.search(r'>\s*([A-Za-z0-9_\-\.\u0080-\uffff][^<>]{0,40})\s*<', body)
        display_name = (nm_match.group(1).strip() if nm_match else name).strip()
        slug = _resolve_player_slug(pid, display_name)
        candidates.append((pid, slug, display_name))

    q_lower = q.lower()
    matched = [c for c in candidates if q_lower in c[2].lower()]
    if matched:
        logger.info(
            f"[vlr search] {name!r} → {len(matched)} name-matched candidates: "
            f"{[c[2] for c in matched]}"
        )
        return matched

    logger.info(
        f"[vlr search] {name!r} → no exact name match in top {len(candidates)}; "
        f"returning fuzzy results: {[c[2] for c in candidates]}"
    )
    return candidates


def search_player(name: str) -> Optional[tuple[str, str, str]]:
    """Backward-compat wrapper — returns the first candidate or None."""
    candidates = search_player_candidates(name, limit=1)
    if not candidates:
        return None
    pid, slug, display_name = candidates[0]
    logger.info(f"[vlr search] {name!r} -> {display_name} (id={pid}, slug={slug})")
    return candidates[0]


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

    candidates = search_player_candidates(name, limit=8)
    if not candidates:
        return None

    # Iterate candidates — vlr.gg search often returns the wrong player first
    # (e.g. "Lucas" → "Rojo"). Pick the first candidate that actually has a
    # recent match list.
    pid = slug = display = None
    matches: list = []
    for cand_pid, cand_slug, cand_display in candidates:
        cand_matches = _fetch_match_list(cand_pid, cand_slug)
        if cand_matches:
            pid, slug, display = cand_pid, cand_slug, cand_display
            matches = cand_matches
            logger.info(
                f"[vlr] picked {display} (id={pid}) — {len(matches)} matches"
            )
            break
        else:
            logger.info(
                f"[vlr] candidate {cand_display} (id={cand_pid}) has no matches — "
                f"trying next"
            )

    if not pid:
        logger.warning(
            f"[vlr] no candidate yielded matches for {name!r} "
            f"(tried: {[c[2] for c in candidates]})"
        )
        return {
            "player_id": candidates[0][0], "slug": candidates[0][1],
            "display_name": candidates[0][2],
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
        # Identify opponent for this match (cached, ~1 extra call per opp team)
        opp_href, opp_rating = _identify_opponent(html, slug)
        opp_slug = opp_href.rsplit("/", 1)[-1] if opp_href else None
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
                "opp_slug":   opp_slug,
                "opp_rating": opp_rating,
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


# ─────────────────────────── opponent scraping ──────────────────────────────

def _extract_match_team_hrefs(html: str) -> list[str]:
    """Pull both team links from a match page (returns up to 2 unique hrefs)."""
    seen: list[str] = []
    for m in re.finditer(r'href="(/team/\d+/[a-z0-9-]+)"', html):
        h = m.group(1)
        if h not in seen:
            seen.append(h)
        if len(seen) == 2:
            break
    return seen


def _team_roster(team_href: str) -> set[str]:
    """Return cached set of player slugs in a team's current/recent roster."""
    now = time.time()
    cached = _TEAM_ROSTER_CACHE.get(team_href)
    if cached and now - cached[0] < _TEAM_TTL:
        return cached[1]
    html = _fetch(f"{VLR_BASE}{team_href}")
    if not html:
        _TEAM_ROSTER_CACHE[team_href] = (now, set())
        return set()
    slugs = {m.group(1) for m in re.finditer(r'/player/\d+/([a-z0-9-]+)', html)}
    _TEAM_ROSTER_CACHE[team_href] = (now, slugs)
    return slugs


def _team_rating(team_href: str) -> int:
    """
    Fetch a team's vlr.gg Elo-style rating (the `rating-num` value on team page).
    Defaults to 1500 (mid-tier) if not found.  Cached for 1 hour.
    """
    now = time.time()
    cached = _TEAM_RATING_CACHE.get(team_href)
    if cached and now - cached[0] < _TEAM_TTL:
        return cached[1]
    html = _fetch(f"{VLR_BASE}{team_href}")
    rating = 1500
    if html:
        m = re.search(r'rating-num[^>]*>\s*(\d{3,5})\s*<', html)
        if m:
            try:
                rating = int(m.group(1))
            except ValueError:
                pass
    _TEAM_RATING_CACHE[team_href] = (now, rating)
    return rating


def _identify_opponent(match_html: str, player_slug: str) -> tuple[Optional[str], Optional[int]]:
    """
    Return (opponent_team_href, opponent_rating) for a given player in a match.
    Determines player's own team by checking which team's roster contains the slug;
    the OTHER team is the opponent.  Returns (None, None) if undetermined.
    """
    teams = _extract_match_team_hrefs(match_html)
    if len(teams) < 2:
        return None, None

    # Try to identify player's team by roster membership
    own = None
    for t in teams:
        if player_slug in _team_roster(t):
            own = t
            break

    # If we can't identify by roster, fall back to "the team that appears
    # NEAREST to player slug in the match HTML" — they're typically grouped
    # in the stat tables together.
    if own is None:
        slug_pos = match_html.find(f"/player/")  # any player anchor
        slug_player_pos = match_html.find(f"/{player_slug}\"")
        if slug_player_pos > 0:
            distances = []
            for t in teams:
                tpos = match_html.find(f'href="{t}"')
                if tpos >= 0:
                    distances.append((abs(tpos - slug_player_pos), t))
            if distances:
                own = min(distances)[1]

    if own is None:
        return None, None
    opp = next((t for t in teams if t != own), None)
    if not opp:
        return None, None
    return opp, _team_rating(opp)


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
def _trimmed_mean(values: list[float], pct: float = 0.10) -> float:
    """Mean after trimming `pct` from each tail (default 10/10). Robust to outliers."""
    if not values:
        return 0.0
    n = len(values)
    k = int(n * pct)
    if n - 2 * k < 1:
        return sum(values) / n
    s = sorted(values)
    trimmed = s[k:n - k]
    return sum(trimmed) / len(trimmed)


def _mad(values: list[float]) -> float:
    """Median Absolute Deviation — robust σ alternative."""
    if not values:
        return 0.0
    from statistics import median as _median
    med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    return _median(abs_dev) * 1.4826  # scale factor to normal-σ equivalent


def empirical_grade(
    map_stats: list,
    line: float,
    stat_type: str = "Kills",
    today_opp_rating: int | None = None,
) -> dict:
    """
    Best-in-class predictive grader for kills props.

    Anti-overestimation stack
    ─────────────────────────
    R1  Robust statistics      — trimmed mean + MAD instead of mean/std,
                                 so one freak game can't inflate the projection.
    R2  Projection clipping    — bounded inside the player's interquartile range
                                 (IQR) of historical totals → never project a
                                 number we've never actually seen.
    R3  Opponent quality filter — drops the bottom-quartile of opponents from
                                 the projection sample (stat-padding games);
                                 also reports avg opp rating per series.
    R4  Disagreement penalty   — when the four signals split, the logit is
                                 multiplied by an alignment factor (0.4–1.0).
                                 Conflicting signals → softer probability →
                                 PASS instead of forced action.
    R5  Temperature softening  — sigmoid temperature T=1.4 squashes extreme
                                 over-confidence on small samples (kills 99%
                                 calls when N<8).
    R6  Push half-credit       — Bayesian smoothing counts pushes as 0.5
                                 OVER / 0.5 UNDER so lines exactly at median
                                 don't bias UNDER.
    R7  Sample-size + edge gate— action requires N≥5, P≥0.62, |edge|≥7%, AND
                                 projection sign agreement.

    Predictive signals
    ──────────────────
    S1  Projection z-score        weight 2.5
    S2  Line-vs-median z-score    weight 1.5
    S3  Recency trend             weight 1.5
    S4  Bayesian hit rate         weight 1.5
    S5  Opponent-tier KPR shift   weight 1.0   (NEW — avg KPR vs strong opps
                                                vs avg KPR vs weak opps)
    """
    import math
    import statistics as _st
    from statistics import mean as _mean, median as _median

    if not map_stats:
        return {"error": "No map data"}

    # ── Build series-level data ──────────────────────────────────────────
    by_match: dict = {}
    by_match_rounds: dict = {}
    by_match_opp: dict = {}
    for m in map_stats:
        mid = m["match_id"]
        by_match.setdefault(mid, []).append(m["stat_value"])
        by_match_rounds.setdefault(mid, []).append(m.get("rounds") or 24)
        if mid not in by_match_opp:
            by_match_opp[mid] = m.get("opp_rating")

    # Preserve newest-first order from scraper
    ordered_mids: list = []
    for m in map_stats:
        if m["match_id"] not in ordered_mids:
            ordered_mids.append(m["match_id"])

    series_totals = [sum(by_match[mid])        for mid in ordered_mids]
    series_rounds = [sum(by_match_rounds[mid]) for mid in ordered_mids]
    series_opp    = [by_match_opp.get(mid)     for mid in ordered_mids]

    n_series = len(series_totals)
    if n_series == 0:
        return {"error": "No series data"}

    # ── R1: Robust central tendency ──────────────────────────────────────
    hist_avg     = _mean(series_totals)
    hist_median  = _median(series_totals)
    trimmed_avg  = _trimmed_mean(series_totals, 0.10)  # 10/10 trim
    sigma_mad    = _mad(series_totals)
    sigma_std    = _st.stdev(series_totals) if n_series > 1 else max(hist_avg * 0.15, 1.0)
    # Use the LARGER of MAD-σ vs std → conservative; stays robust without underestimating risk
    sigma        = max(sigma_mad, sigma_std, max(hist_avg * 0.10, 1.0))
    ceiling      = max(series_totals)
    floor        = min(series_totals)

    # ── R3: Opponent-quality filtering for projection ────────────────────
    # Two modes:
    #   A) today_opp_rating provided → keep series where the opponent was
    #      within ±200 rating of today's opp (similar competition tier).
    #   B) no today_opp_rating → drop bottom-quartile of opponents (filters
    #      stat-padding games against the weakest teams in sample).
    rated_pairs = [(t, r) for t, r in zip(series_totals, series_opp) if r is not None]
    opp_quality_used = False
    proj_totals = list(series_totals)  # default: full sample
    proj_rounds_list = list(series_rounds)

    if today_opp_rating is not None and len(rated_pairs) >= 6:
        # Proximity mode — same-tier opponents only
        WINDOW = 200  # rating points either side of today's opp
        proj_totals = [
            t for t, r in zip(series_totals, series_opp)
            if r is None or abs(r - today_opp_rating) <= WINDOW
        ]
        proj_rounds_list = [
            rd for rd, r in zip(series_rounds, series_opp)
            if r is None or abs(r - today_opp_rating) <= WINDOW
        ]
        if len(proj_totals) >= 4:
            opp_quality_used = True
        else:
            proj_totals = list(series_totals)
            proj_rounds_list = list(series_rounds)
    elif len(rated_pairs) >= 6:
        opp_ratings = [r for _, r in rated_pairs]
        threshold   = sorted(opp_ratings)[len(opp_ratings) // 4]  # bottom 25% cutoff
        proj_totals = [t for t, r in zip(series_totals, series_opp)
                       if r is None or r >= threshold]
        proj_rounds_list = [rd for rd, r in zip(series_rounds, series_opp)
                            if r is None or r >= threshold]
        if len(proj_totals) >= 4:
            opp_quality_used = True
        else:
            proj_totals = list(series_totals)
            proj_rounds_list = list(series_rounds)

    # ── Build per-map KPR (filtered if opp_quality_used) ─────────────────
    valid_maps = [m for m in map_stats if (m.get("rounds") or 0) > 0]
    if opp_quality_used:
        # Drop maps from the filtered-out matches
        kept_mids = set()
        for mid, t in zip(ordered_mids, series_totals):
            if t in proj_totals:  # rough match — totals are unique enough usually
                kept_mids.add(mid)
        valid_maps = [m for m in valid_maps if m["match_id"] in kept_mids] or valid_maps

    if valid_maps:
        kpr_all = [m["stat_value"] / m["rounds"] for m in valid_maps]
    else:
        kpr_all = [m["stat_value"] / 22 for m in map_stats]

    # Recent form: most recent ~6 maps (3 series)
    recent_n_maps = min(6, len(kpr_all))
    recent_kpr  = _mean(kpr_all[:recent_n_maps]) if recent_n_maps else _mean(kpr_all)
    overall_kpr = _mean(kpr_all) if kpr_all else 0.0
    # 50/50 recency blend — prevents 6-map sample noise (e.g. one cold series)
    # from dragging the projection away from the player's true baseline.
    blended_kpr = 0.50 * recent_kpr + 0.50 * overall_kpr

    # ── Match-length adjustment by opponent strength ─────────────────────
    # Mismatch → shorter maps (more 13-3, 13-5 type scores → fewer total
    # rounds → fewer kills available). Closer match → more 13-11/OT rounds.
    # Adjustment caps at ±15% of historical avg to avoid wild swings.
    proj_rounds = _mean(proj_rounds_list) if proj_rounds_list else 44.0
    avg_opp = (
        sum(r for _, r in rated_pairs) / len(rated_pairs)
        if rated_pairs else None
    )
    round_adj_pct = 0.0
    if today_opp_rating is not None and avg_opp is not None:
        gap = today_opp_rating - avg_opp  # +ve = opp stronger than usual
        # Each 200pt rating gap → ~5% round-count shift (closer match → +rounds)
        round_adj_pct = max(-0.15, min(0.15, (gap / 200.0) * 0.05))
        # Sweep risk: if opp is much weaker AND player's team is much stronger,
        # apply extra round downgrade (blowout potential)
        if gap < -300:
            round_adj_pct -= 0.05
    proj_rounds_adj = proj_rounds * (1.0 + round_adj_pct)

    # ── Map-pool aware projection (veto uncertainty hedge) ───────────────
    # Without knowing the actual veto, opponents typically ban into the
    # player's strongest maps. Build per-map kill totals; blend the global
    # KPR projection with a conservative basket (player's lower 50% of maps
    # by avg kills/map) at 65/35. Pulls projections back when a player has
    # weak maps in their pool that opponents will likely force.
    by_map: dict = {}
    for m in valid_maps:
        nm = (m.get("map_name") or "").strip()
        if not nm or nm.lower() in ("unknown", ""):
            continue
        by_map.setdefault(nm, []).append(m["stat_value"])
    map_pool_kpr_blend = blended_kpr
    map_pool_used = False
    if len(by_map) >= 3:
        # avg kills per map for each pool entry, weighted by sample (cap n=3)
        per_map_avg = sorted(_mean(v) for v in by_map.values())
        # Bottom-half basket: maps the player is worst on
        half = max(2, len(per_map_avg) // 2)
        weak_basket_avg = _mean(per_map_avg[:half])  # kills/map on weak maps
        # Convert to per-map KPR equivalent (assume 22 rounds/map)
        weak_kpr_equiv = weak_basket_avg / 22.0
        # 65% global projection / 35% conservative weak-pool basket
        map_pool_kpr_blend = 0.65 * blended_kpr + 0.35 * weak_kpr_equiv
        map_pool_used = True

    expected_total = map_pool_kpr_blend * proj_rounds_adj

    # ── R2: Clip projection to player's IQR ──────────────────────────────
    s_sorted = sorted(series_totals)
    if n_series >= 4:
        q25 = s_sorted[n_series // 4]
        q75 = s_sorted[(3 * n_series) // 4]
        clipped_total = max(q25, min(q75, expected_total))
        if clipped_total != expected_total:
            logger.debug(f"[grader] projection clipped {expected_total:.1f} → {clipped_total:.1f} (IQR {q25}-{q75})")
        expected_total = clipped_total

    # ── Signals ──────────────────────────────────────────────────────────
    z_proj  = (expected_total - line) / sigma
    z_med   = (hist_median   - line) / sigma

    if recent_n_maps >= 2 and len(kpr_all) > recent_n_maps:
        older_kpr = _mean(kpr_all[recent_n_maps:]) or recent_kpr
        trend_pct = ((recent_kpr - older_kpr) / max(older_kpr, 0.01)) * 100
    else:
        trend_pct = 0.0
    z_trend = max(-2.5, min(2.5, trend_pct / 15.0))  # cap at ±2.5σ

    # ── R6: Bayesian hit rate with push half-credit ──────────────────────
    overs   = sum(1 for v in series_totals if v >  line)
    unders  = sum(1 for v in series_totals if v <  line)
    pushes  = sum(1 for v in series_totals if v == line)
    eff_overs = overs + 0.5 * pushes
    bayes_p   = (eff_overs + 2) / (n_series + 4)
    z_hit     = (bayes_p - 0.5) / 0.15

    # ── S5: Opponent-tier KPR shift signal ───────────────────────────────
    z_opp = 0.0
    opp_split_label = "n/a"
    if opp_quality_used and len(rated_pairs) >= 6:
        opp_ratings_sorted = sorted([r for _, r in rated_pairs])
        med_opp = opp_ratings_sorted[len(opp_ratings_sorted) // 2]
        # KPR vs strong (≥med) and weak (<med) opps
        strong_kpr, weak_kpr = [], []
        for m in [mm for mm in map_stats if (mm.get("rounds") or 0) > 0]:
            r = m.get("opp_rating")
            if r is None:
                continue
            kpr = m["stat_value"] / m["rounds"]
            (strong_kpr if r >= med_opp else weak_kpr).append(kpr)
        if strong_kpr and weak_kpr:
            sk = _mean(strong_kpr); wk = _mean(weak_kpr)
            shift_pct = ((sk - wk) / max(wk, 0.01)) * 100
            z_opp = max(-1.5, min(1.5, shift_pct / 20.0))  # 20% shift = 1σ
            opp_split_label = f"vs strong {sk:.2f} kpr · vs weak {wk:.2f} kpr"

    # ── Stack signals → raw score ────────────────────────────────────────
    W_PROJ, W_MED, W_TREND, W_HIT, W_OPP = 2.5, 1.5, 1.5, 1.5, 1.0
    signals = [
        (W_PROJ,  z_proj),
        (W_MED,   z_med),
        (W_TREND, z_trend),
        (W_HIT,   z_hit),
        (W_OPP,   z_opp),
    ]
    raw_score = sum(w * z for w, z in signals)

    # ── R4: Disagreement penalty ─────────────────────────────────────────
    # Count signals voting OVER (>0.25σ) vs UNDER (<-0.25σ); compute alignment.
    over_signals  = sum(1 for w, z in signals if z >  0.25 and w > 0)
    under_signals = sum(1 for w, z in signals if z < -0.25 and w > 0)
    total_voting  = over_signals + under_signals
    if total_voting > 0:
        majority   = max(over_signals, under_signals)
        agreement  = majority / total_voting           # 1.0 = unanimous, 0.5 = split
        # Cap at 0.9 (was 1.0): all 5 signals derive from the same 10-series
        # sample so they're correlated, not independent. Unanimous agreement
        # shouldn't get full credit. Prevents 95%+ overconfidence stacks.
        align_mult = 0.4 + 0.5 * (agreement ** 2)     # 0.4 → 0.9
    else:
        align_mult = 0.5  # nothing voting either way → max softening

    # ── R5: Temperature softening ────────────────────────────────────────
    TEMP = 1.4
    logit = (raw_score * align_mult) / (1.6 * TEMP)

    # Sample-size shrink (full strength at 8+ series)
    shrink = min(1.0, n_series / 8.0)
    logit *= shrink

    over_prob  = 1.0 / (1.0 + math.exp(-logit))

    # ── Adaptive probability cap (sample × volatility × floor) ───────────
    # Stack three caps and take the strictest:
    #   1. Sample size (Wilson CI logic)
    #   2. Volatility — high σ means single-game variance can blow up
    #      any "confident" call; cap aggressively
    #   3. Floor proximity — if the player has historically scored well
    #      below the line, even an "OVER" pick must concede that risk
    # Prevents the Kumi @ 32 case (95% confident on a player whose floor
    # is 23 = 72% of line, with σ=5+).
    if   n_series <= 4:  cap_n = 0.72
    elif n_series <= 6:  cap_n = 0.80
    elif n_series <= 8:  cap_n = 0.85
    elif n_series <= 12: cap_n = 0.88
    elif n_series <= 18: cap_n = 0.91
    else:                cap_n = 0.94

    cv = sigma / hist_avg if hist_avg > 0 else 1.0
    if   sigma >= 8 or cv >= 0.30: cap_vol = 0.72   # ⚡ high vol
    elif sigma >= 5 or cv >= 0.18: cap_vol = 0.80   # 🌊 moderate
    else:                          cap_vol = 0.92   # 🎯 consistent

    # Floor proximity: how far is the player's worst game from the line?
    # OVER bets need floor close to line. UNDER bets need ceiling close.
    floor_ratio   = (floor   / line) if line > 0 else 1.0
    ceiling_ratio = (ceiling / line) if line > 0 else 1.0
    if over_prob >= 0.5:
        # OVER pick — what % of line did the floor reach?
        if   floor_ratio >= 1.0:  cap_floor = 0.95   # floor clears line
        elif floor_ratio >= 0.85: cap_floor = 0.85
        elif floor_ratio >= 0.70: cap_floor = 0.78
        else:                     cap_floor = 0.70   # floor far below line
    else:
        # UNDER pick — what % of line did the ceiling exceed?
        if   ceiling_ratio <= 1.0:  cap_floor = 0.95
        elif ceiling_ratio <= 1.15: cap_floor = 0.85
        elif ceiling_ratio <= 1.30: cap_floor = 0.78
        else:                       cap_floor = 0.70

    prob_cap = min(cap_n, cap_vol, cap_floor)
    over_prob  = max(1.0 - prob_cap, min(prob_cap, over_prob))
    under_prob = 1.0 - over_prob
    edge       = over_prob - 0.5238

    emp_over_pct = (overs / n_series) * 100 if n_series else 0.0

    ev_over  = round(over_prob  * (100/110) - (1 - over_prob),  4)
    ev_under = round(under_prob * (100/110) - (1 - under_prob), 4)

    # Stability label (now using robust σ)
    cv = sigma / hist_avg if hist_avg > 0 else 1.0
    if   sigma > 8 or cv > 0.30: stability_label = "⚡ High Volatility"
    elif sigma > 5 or cv > 0.18: stability_label = "🌊 Moderate Volatility"
    else:                         stability_label = "🎯 Consistent"

    # ── R7: Strict decision gate ─────────────────────────────────────────
    proj_aligned_over  = z_proj > -0.20
    proj_aligned_under = z_proj <  0.20
    # Decision band tightened: require ≥70% probability before firing OVER/UNDER.
    # Anything in the 50–70% range = PASS. Cuts overconfident calls on noisy
    # mid-tier signal stacks (e.g. flyuh 67.6% UNDER → now correctly PASS).
    decision = "PASS"
    is_lock  = False
    lock_reasons: list[str] = []
    if (n_series >= 5 and abs(edge * 100) >= 7.0):
        if   over_prob  >= 0.70 and proj_aligned_over  and over_signals  >= 2:
            decision = "OVER"
        elif under_prob >= 0.70 and proj_aligned_under and under_signals >= 2:
            decision = "UNDER"

    # ── CERTIFIED LOCK gate ──────────────────────────────────────────────
    # Strict, all-or-nothing. Every box must be ticked. No exceptions.
    # Designed to fire only on plays where the model + history + context
    # all align with no contradicting evidence. Realistically 0-3 per slate.
    if decision in ("OVER", "UNDER"):
        prob       = over_prob if decision == "OVER" else under_prob
        sigs       = over_signals if decision == "OVER" else under_signals
        opp_sigs   = under_signals if decision == "OVER" else over_signals
        emp_hit    = (overs / n_series) if decision == "OVER" else ((n_series - overs - pushes) / n_series)
        floor_ok   = (floor  > line) if decision == "OVER" else (ceiling < line)
        stable     = sigma <= 5 and (sigma / hist_avg if hist_avg > 0 else 1.0) <= 0.18
        lock_checks = {
            "Probability ≥85%":          prob >= 0.85,
            "Sample ≥8 series":          n_series >= 8,
            "Hit rate ≥80%":             emp_hit >= 0.80,
            "Edge vs -110 ≥+25pts":      abs(edge * 100) >= 25,
            "Unanimous signals (4+/0)":  sigs >= 4 and opp_sigs == 0,
            "Floor clears line":         floor_ok,
            "Low volatility":            stable,
            "Recency not opposing":      (trend_pct >= -3) if decision == "OVER" else (trend_pct <= 3),
        }
        is_lock = all(lock_checks.values())
        lock_reasons = [k for k, v in lock_checks.items() if not v]

    # Confidence interval on hist_median (robust uncertainty band)
    ci_low  = round(hist_median - sigma, 1)
    ci_high = round(hist_median + sigma, 1)

    avg_opp_rating = round(avg_opp) if avg_opp is not None else None

    return {
        "stat_type":       stat_type,
        "n_samples":       len(map_stats),
        "n_series":        n_series,
        "hist_avg":        round(hist_avg, 2),
        "hist_median":     round(hist_median, 2),
        "trimmed_avg":     round(trimmed_avg, 2),
        "hit_rate":        round(emp_over_pct, 1),
        "bayes_hit_rate":  round(bayes_p * 100, 1),
        "over_prob":       round(over_prob  * 100, 1),
        "under_prob":      round(under_prob * 100, 1),
        "push_prob":       round((pushes / n_series) * 100, 1) if n_series else 0,
        "edge":            round(edge * 100, 1),
        "decision":        decision,
        "is_lock":         is_lock,
        "lock_misses":     lock_reasons,
        "ceiling":         ceiling,
        "floor":           floor,
        "stability_std":   round(sigma, 2),
        "sigma_mad":       round(sigma_mad, 2),
        "stability_label": stability_label,
        "ev_over":         ev_over,
        "ev_under":        ev_under,
        # Predictive transparency
        "expected_total":  round(expected_total, 2),
        "blended_kpr":     round(blended_kpr, 3),
        "map_pool_kpr":    round(map_pool_kpr_blend, 3),
        "map_pool_used":   map_pool_used,
        "recent_kpr":      round(recent_kpr, 3),
        "proj_rounds":     round(proj_rounds, 1),
        "proj_rounds_adj": round(proj_rounds_adj, 1),
        "round_adj_pct":   round(round_adj_pct * 100, 1),
        "prob_cap_used":   round(prob_cap, 2),
        "trend_pct":       round(trend_pct, 1),
        "z_projection":    round(z_proj, 2),
        "z_median":        round(z_med, 2),
        "z_trend":         round(z_trend, 2),
        "z_bayes_hit":     round(z_hit, 2),
        "z_opponent":      round(z_opp, 2),
        "model_score":     round(raw_score, 2),
        "alignment_mult":  round(align_mult, 2),
        "signals_over":    over_signals,
        "signals_under":   under_signals,
        # Opponent context
        "opp_quality_filtered": opp_quality_used,
        "opp_split_label": opp_split_label,
        "avg_opp_rating":  avg_opp_rating,
        "today_opp_rating": today_opp_rating,
        # CI band
        "ci_low":          ci_low,
        "ci_high":         ci_high,
        # Embed compatibility shims
        "sim_median":      round(hist_median, 2),
        "sim_std":         round(sigma, 2),
    }


# Friendly alias for new callers
predict_over_under = empirical_grade


def classify_miss(sim: dict, line: float, actual_total: int,
                  actual_rounds: int | None = None) -> dict:
    """
    Classify why a graded prop missed. Returns a dict with:
      cause:      one of "round-count", "kpr", "map-veto", "role-change",
                  "variance", "correct" (no miss)
      severity:   "minor" (close call) | "major" (model was significantly off)
      details:    human-readable explanation
      delta:      actual - line (how much it missed by)
    """
    decision = sim.get("decision", "PASS")
    over_won  = actual_total >  line
    under_won = actual_total <  line

    # PASS — bot correctly skipped. Still classify which side won + why,
    # so the user can see the model's actual projection vs reality.
    if decision == "PASS":
        winning_side = "OVER" if over_won else ("UNDER" if under_won else "PUSH")
        return {
            "cause":   "skipped",
            "severity": "minor",
            "details": (
                f"Bot did not bet this — current model would PASS. "
                f"Actual {actual_total} vs line {line:g} = {winning_side}. "
                f"Model projection was {sim.get('expected_total', 0):.1f}."
            ),
            "delta":   actual_total - line,
        }

    won = (decision == "OVER" and over_won) or (decision == "UNDER" and under_won)
    if won:
        return {"cause": "correct", "details": "Bot was right", "delta": actual_total - line}

    delta = actual_total - line
    expected = sim.get("expected_total", sim.get("hist_avg", 0)) or 0
    proj_rounds = sim.get("proj_rounds_adj") or sim.get("proj_rounds") or 44
    blended_kpr = sim.get("blended_kpr") or 0
    sigma = sim.get("stability_std") or 1.0

    # ── Round-count miss ─────────────────────────────────────────────────
    # If we know actual rounds and they're way off projection, that's the
    # primary cause regardless of other factors.
    if actual_rounds and proj_rounds:
        round_gap = (actual_rounds - proj_rounds) / proj_rounds
        if abs(round_gap) >= 0.20:
            implied_kills = blended_kpr * actual_rounds
            return {
                "cause":   "round-count",
                "severity": "major" if abs(round_gap) >= 0.30 else "minor",
                "details": (
                    f"Match ran {actual_rounds} rounds vs projected {proj_rounds:.0f} "
                    f"({round_gap*100:+.0f}%). At your KPR ({blended_kpr:.2f}) that "
                    f"changes expected kills by {implied_kills - expected:+.1f}."
                ),
                "delta": delta,
            }

    # ── Variance miss ────────────────────────────────────────────────────
    # Within ~1σ of model — model was right, dice rolled wrong. Common,
    # not a model bug.
    if abs(actual_total - expected) <= sigma:
        return {
            "cause":   "variance",
            "severity": "minor",
            "details": (
                f"Actual ({actual_total}) within 1σ of model expectation "
                f"({expected:.1f} ± {sigma:.1f}). Normal variance — bet was sound."
            ),
            "delta": delta,
        }

    # ── KPR miss (player underperformed/overperformed their baseline) ────
    if actual_rounds:
        actual_kpr = actual_total / actual_rounds
        kpr_gap = (actual_kpr - blended_kpr) / max(blended_kpr, 0.01)
        if abs(kpr_gap) >= 0.20:
            return {
                "cause":   "kpr",
                "severity": "major" if abs(kpr_gap) >= 0.35 else "minor",
                "details": (
                    f"Player's actual KPR ({actual_kpr:.2f}) was {kpr_gap*100:+.0f}% "
                    f"vs projected ({blended_kpr:.2f}). Player had an off-day, OR "
                    f"role/agent shift, OR opponent was stylistically different than "
                    f"expected."
                ),
                "delta": delta,
            }

    # ── Map-veto miss (player got their weak maps) ───────────────────────
    # If model didn't use map-pool weighting OR if the gap is large,
    # likely the actual map pool favored the opponent.
    if not sim.get("map_pool_used"):
        return {
            "cause":   "map-veto",
            "severity": "major",
            "details": (
                "Model used global KPR (insufficient map-pool data). "
                f"Actual ({actual_total}) deviated from projection ({expected:.1f}). "
                "Likely the actual veto landed on player's weak maps."
            ),
            "delta": delta,
        }

    # ── Default: unexplained variance / model error ──────────────────────
    return {
        "cause":   "variance",
        "severity": "major",
        "details": (
            f"Actual ({actual_total}) outside 1σ of model expectation "
            f"({expected:.1f} ± {sigma:.1f}) but no single cause dominates. "
            f"Likely a combination of map veto + KPR + round-count drift."
        ),
        "delta": delta,
    }
