"""
Deep Opponent Analysis for the Elite CS2 Prop Grader.

Dimensions covered:
  1. Defensive Profile    — kills allowed per player per map (last 10 opp matches)
  2. HS Vulnerability     — kills-allowed proxy → Frag Mine / Moderate / Low
  3. CT/T Efficiency      — round win % on each side, T-side aggression modifier
  4. Head-to-Head         — player's actual performance in last 3 matches vs this team
  5. Stomp / Rank Risk    — ranking gap → round projection adjustment
  6. Map Pool             — opponent's most/least played maps → frag boost/suppress
  7. Combined Multiplier  — single scalar applied to kill distribution before simulation
"""

import re
import time
import logging
import statistics as _stats
from bs4 import BeautifulSoup

from scraper import (
    _fetch, HLTV_BASE,
    search_team,
    get_player_team,
    get_team_period_stats,
    _parse_match_kills,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Map type classification
# ---------------------------------------------------------------------------

MAP_TYPE: dict[str, str] = {
    'mirage':       'high_frag',
    'inferno':      'high_frag',
    'dust2':        'high_frag',
    'overpass':     'high_frag',
    'cache':        'high_frag',
    'cobblestone':  'high_frag',
    'ancient':      'average',
    'anubis':       'average',
    'nuke':         'tactical',
    'vertigo':      'tactical',
    'train':        'tactical',
    'faceit':       'average',
}

MAP_KILL_MODIFIER: dict[str, float] = {
    'high_frag': 1.07,
    'average':   1.00,
    'tactical':  0.93,
}

ALL_MAP_NAMES = set(MAP_TYPE.keys())

# CS2 BO3 baseline: avg kills per player per Map 1+2 only (not including Map 3).
# Based on real CS2 data: typical BO3 player averages ~14-16 kills per map.
# 18.5 was the CS:GO-era figure and was causing systematic downward H2H bias.
BASELINE_KILLS = 15.0

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

_RANK_CACHE: dict[str, tuple] = {}    # team_id → (timestamp, rank | None)
_OPP_CACHE: dict[str, tuple] = {}     # team_id → (timestamp, data_dict)
_RANKING_PAGE_CACHE: dict[str, tuple] = {}  # "page" → (timestamp, html)
RANK_TTL = 6 * 3600
OPP_TTL  = 4 * 3600
RANKING_PAGE_TTL = 3600  # 1 hour — ranking page changes infrequently

# ---------------------------------------------------------------------------
# 1. Team ranking
# ---------------------------------------------------------------------------

def _rank_from_team_page(html: str, team_slug: str) -> int | None:
    """
    Parse a team's world ranking from their HLTV profile page HTML.
    Tries multiple extraction strategies in priority order.
    """
    # Strategy 1: JSON field "worldRanking" (most specific — avoids false positives
    # from "individualRanking" or other "ranking" keys that appear in the same JSON)
    m = re.search(r'"worldRanking"\s*:\s*(\d+)', html, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Strategy 2: profile-team-stat div — HLTV renders:
    #   <div class="profile-team-stat">
    #     <b>World ranking</b><a ...>#77</a>
    #   </div>
    # Use BeautifulSoup to find the exact stat block
    try:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html, 'html.parser')
        for stat in soup.find_all(class_='profile-team-stat'):
            label = stat.find('b')
            if label and 'world ranking' in label.get_text(strip=True).lower():
                val = stat.find('a') or stat.find(class_='value')
                if val:
                    m2 = re.search(r'#(\d+)', val.get_text(strip=True))
                    if m2:
                        return int(m2.group(1))
    except Exception:
        pass

    # Strategy 3: generic "World ranking #N" anywhere in page text
    m = re.search(r'World\s+ranking\s*[^<]{0,30}#\s*(\d+)', html, re.IGNORECASE | re.DOTALL)
    if m:
        return int(m.group(1))

    # Strategy 4: teamRanking CSS class containing "#N"
    m = re.search(r'class=["\'][^"\']*teamRanking[^"\']*["\'][^>]*>(?:[^<]{0,20})?#\s*(\d+)', html, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _fetch_ranking_page() -> str | None:
    """Fetch (and cache for 1 hour) the HLTV world team ranking page HTML."""
    cached = _RANKING_PAGE_CACHE.get("page")
    if cached:
        ts, html = cached
        if time.time() - ts < RANKING_PAGE_TTL:
            return html
    url = f"{HLTV_BASE}/ranking/teams"
    html = _fetch(url)
    if html:
        _RANKING_PAGE_CACHE["page"] = (time.time(), html)
    return html


def _rank_from_ranking_page(team_id: str) -> int | None:
    """
    Find a team's world ranking from the HLTV ranking page.
    HLTV renders:
        <div class="ranked-team" data-team-id="XXXXX">
            <div class="ranking-header">
                <span class="position">#77</span>
            </div>
            ...
        </div>
    """
    html = _fetch_ranking_page()
    if not html:
        return None

    try:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html, 'html.parser')

        # Primary: data-team-id attribute on the ranked-team block
        for block in soup.find_all(attrs={"data-team-id": True}):
            if str(block.get("data-team-id")) == str(team_id):
                pos_tag = block.find(class_=re.compile(r'position', re.I))
                if pos_tag:
                    m = re.search(r'(\d+)', pos_tag.get_text())
                    if m:
                        return int(m.group(1))
                # Fallback: look for #N in the block's raw text
                m = re.search(r'#\s*(\d+)', block.get_text())
                if m:
                    return int(m.group(1))

        # Secondary: look for /team/TEAM_ID/ links inside ranked-team divs
        for block in soup.find_all(class_=re.compile(r'ranked-team', re.I)):
            if f'/team/{team_id}/' in str(block):
                pos_tag = block.find(class_=re.compile(r'position', re.I))
                if pos_tag:
                    m = re.search(r'(\d+)', pos_tag.get_text())
                    if m:
                        return int(m.group(1))
    except Exception as e:
        logger.warning(f"[rank] ranking page parse error: {e}")

    # Last resort: simple regex scan for data-team-id="X" ... position ... #N
    m = re.search(
        rf'data-team-id=["\']?{re.escape(str(team_id))}["\']?[^>]*>(.*?)</div>',
        html, re.IGNORECASE | re.DOTALL
    )
    if m:
        m2 = re.search(r'#\s*(\d+)', m.group(1))
        if m2:
            return int(m2.group(1))

    return None


def get_team_rank(team_id: str, team_slug: str) -> int | None:
    cached = _RANK_CACHE.get(team_id)
    if cached:
        ts, rank = cached
        if time.time() - ts < RANK_TTL:
            return rank

    rank = None

    # Tier 1: parse from the team's own profile page
    url = f"{HLTV_BASE}/team/{team_id}/{team_slug}"
    html = _fetch(url)
    if html:
        rank = _rank_from_team_page(html, team_slug)

    # Tier 2: if team page didn't yield a rank, check the live rankings page
    if rank is None:
        logger.info(f"[rank] Team page gave no rank for {team_slug} — trying ranking page")
        rank = _rank_from_ranking_page(team_id)

    if rank is not None:
        # Only cache successful lookups — None results should always be retried
        _RANK_CACHE[team_id] = (time.time(), rank)
        logger.info(f"[rank] {team_slug} → #{rank}")
    else:
        logger.warning(f"[rank] could not determine rank for {team_slug} (team_id={team_id})")
    return rank

# ---------------------------------------------------------------------------
# 2. Map name extraction from a match page
# ---------------------------------------------------------------------------

def _extract_maps_from_page(html: str) -> list[str]:
    """Return lowercase map names found in the match page (max 3)."""
    soup = BeautifulSoup(html, 'html.parser')
    found: list[str] = []

    # Primary: dynamic-map-name-full divs (already used in _parse_match_kills)
    for el in soup.find_all(class_=re.compile(r'dynamic-map-name', re.I)):
        text = el.get_text(strip=True).lower()
        if text in ALL_MAP_NAMES and text not in found:
            found.append(text)

    if not found:
        # Fallback: regex scan
        for name in re.findall(
            r'\b(mirage|inferno|nuke|dust2|vertigo|ancient|anubis|overpass|train|cache|cobblestone)\b',
            html, re.IGNORECASE
        ):
            name = name.lower()
            if name not in found:
                found.append(name)

    return found[:3]

# ---------------------------------------------------------------------------
# 3. CT/T round scores
# ---------------------------------------------------------------------------

def _extract_half_scores(html: str) -> list[dict]:
    """
    Try to pull half-by-half round scores from the match page.
    Looks for patterns like "8-7; 7-5" (two halves) in the score area.
    Returns a list of dicts per map:
      {'ct_a': int, 't_a': int, 'ct_b': int, 't_b': int, 'total': int}
    """
    results = []
    # Pattern: A-B ; C-D  (where each half has two team scores)
    for a, b, c, d in re.findall(r'(\d+)-(\d+)\s*[;,]\s*(\d+)-(\d+)', html)[:4]:
        a, b, c, d = int(a), int(b), int(c), int(d)
        # Sanity: half rounds typically 8-16 in regulation
        if all(0 <= x <= 21 for x in [a, b, c, d]) and (a + b + c + d) >= 16:
            results.append({'h1_a': a, 'h1_b': b, 'h2_a': c, 'h2_b': d,
                            'total': a + b + c + d})
    return results[:2]  # at most 2 maps

# ---------------------------------------------------------------------------
# 4. Opponent deep profile (kills allowed + CT/T + map pool + HS proxy)
# ---------------------------------------------------------------------------

def _fetch_opponent_profile(team_id: str, n_matches: int = 10) -> dict:
    """Fetch opponent match pages and aggregate the full defensive profile."""
    cached = _OPP_CACHE.get(team_id)
    if cached:
        ts, data = cached
        if time.time() - ts < OPP_TTL:
            return data

    results_url = f"{HLTV_BASE}/results?team={team_id}"
    html = _fetch(results_url)
    if not html:
        return {}

    match_pairs = re.findall(r'/matches/(\d+)/([\w-]+)', html)
    seen: dict[str, str] = {}
    for mid, slug in match_pairs:
        if mid not in seen and len(mid) >= 6:
            seen[mid] = slug
    match_list = list(seen.items())[:n_matches]
    if not match_list:
        return {}

    # Aggregation buckets
    opp_kills: list[int] = []          # kills scored AGAINST target team per player per map
    star_kills: list[int] = []         # top-killer's count per map (star suppression metric)
    ct_wins, ct_total = 0, 0           # target team's CT rounds won / played
    t_wins, t_total   = 0, 0           # target team's T rounds won / played
    map_counter: dict[str, int] = {}
    rounds_per_map: list[int] = []

    for match_id, slug in match_list:
        time.sleep(0.35)
        page_html = _fetch(f"{HLTV_BASE}/matches/{match_id}/{slug}")
        if not page_html:
            continue

        soup = BeautifulSoup(page_html, 'html.parser')
        matchstats = soup.find(id='match-stats')

        # --- Kill data + star suppression ---
        if matchstats:
            raw = str(matchstats)
            map_ids = re.findall(r'id="(\d{5,7})-content"', raw)
            for map_id in map_ids[:2]:
                content = matchstats.find(id=f'{map_id}-content')
                if not content:
                    continue
                tables = content.find_all('table', class_='totalstats')
                for table in tables:
                    if table.find('a', href=re.compile(rf'/team/{team_id}/')):
                        continue  # skip target team; collect opponent's kills
                    map_kills_this_table: list[int] = []
                    for tr in table.find_all('tr')[1:]:
                        m = re.search(r'(\d+)\s*-\s*\d+', tr.get_text())
                        if m:
                            k = int(m.group(1))
                            if 3 <= k <= 60:
                                opp_kills.append(k)
                                map_kills_this_table.append(k)
                    # Star suppression: track the highest kill by any player per map
                    if map_kills_this_table:
                        star_kills.append(max(map_kills_this_table))

        # --- CT/T half scores ---
        half_scores = _extract_half_scores(page_html)
        for hs in half_scores:
            total = hs['total']
            if total >= 16:
                rounds_per_map.append(total // 2)
            # h1_a = team A's score in half 1 (CT half)
            # Without knowing team ordering we can still average halves
            ct_wins  += hs['h1_a']
            ct_total += hs['h1_a'] + hs['h1_b']
            t_wins   += hs['h2_a']
            t_total  += hs['h2_a'] + hs['h2_b']

        # --- Map pool ---
        for name in _extract_maps_from_page(page_html):
            map_counter[name] = map_counter.get(name, 0) + 1

    # Compile results
    avg_allowed   = round(_stats.mean(opp_kills),  1) if len(opp_kills)  >= 5 else None
    avg_star_kill = round(_stats.mean(star_kills), 1) if len(star_kills) >= 3 else None
    ct_pct = round(ct_wins / ct_total * 100, 1) if ct_total > 0 else None
    t_pct  = round(t_wins  / t_total  * 100, 1) if t_total  > 0 else None
    avg_rounds = round(_stats.mean(rounds_per_map), 1) if rounds_per_map else 22.0

    sorted_maps = sorted(map_counter.items(), key=lambda x: -x[1])
    most_played  = [m for m, _ in sorted_maps[:3]]
    least_played = [m for m, _ in sorted_maps[-2:]] if len(sorted_maps) >= 4 else []

    data = {
        'avg_kills_allowed': avg_allowed,
        'avg_star_kill':     avg_star_kill,   # top-killer per map average
        'star_suppression':  (avg_star_kill is not None and avg_star_kill < 15.0),
        'sample_kills': len(opp_kills),
        'ct_win_pct': ct_pct,
        't_win_pct':  t_pct,
        'avg_rounds_per_map': avg_rounds,
        'most_played_maps':   most_played,
        'least_played_maps':  least_played,
        'map_counter':        map_counter,
    }
    _OPP_CACHE[team_id] = (time.time(), data)
    logger.info(f"[opp_profile] team_id={team_id} → {data}")
    return data

# ---------------------------------------------------------------------------
# 5. Head-to-head — player's kills in last N matches vs this opponent
# ---------------------------------------------------------------------------

_H2H_MIN_YEAR = 2026  # Only count H2H matches from this year or later
_H2H_MIN_TS   = 1735689600  # Unix timestamp for 2026-01-01 00:00:00 UTC


def get_h2h_stats(
    player_id: str,
    player_slug: str,
    player_match_ids: list[tuple[str, str]],
    opponent_team_id: str,
    opponent_slug: str = "",
    n: int = 3,
    line: float = 0.0,
) -> list[dict]:
    """
    Scan the player's recent match IDs for matches that involved the opponent.
    Returns up to n match records:
      [{'match_id': ..., 'kills_by_map': [22, 18], 'total_kills': 40,
        'avg_kills': 20.0, 'cleared': True, 'maps_found': 2}, ...]

    Opponent matching uses the match URL slug (e.g. 'b8-vs-3dmax-event') which
    is far more reliable than scanning page HTML for a team ID that may appear
    in sidebar/bracket links for teams NOT playing in the match.

    Only 2026+ matches are counted (controlled by _H2H_MIN_YEAR).
    Only records with data from BOTH maps 1 and 2 are counted as valid for the
    cleared check.  Partial records (only 1 map parsed) are flagged but still
    included so callers can at least display them.
    """
    results: list[dict] = []
    opp_slug_norm = re.sub(r'[^a-z0-9]', '', opponent_slug.lower()) if opponent_slug else ""

    for match_id, slug in player_match_ids:
        if len(results) >= n:
            break

        # ── 1. Primary opponent filter — check match URL slug ─────────────────
        # The slug looks like "b8-vs-3dmax-parken-challenger-championship-season-3".
        # We normalise both sides and check for containment so minor slug
        # variations (extra hyphens, numbers) don't cause false negatives.
        slug_norm = re.sub(r'[^a-z0-9]', '', slug.lower())
        if opp_slug_norm and opp_slug_norm not in slug_norm:
            logger.debug(f"[h2h] skip {match_id} — opponent slug '{opp_slug_norm}' not in match slug")
            continue

        time.sleep(0.35)
        _match_url = f"{HLTV_BASE}/matches/{match_id}/{slug}"
        page_html = _fetch(_match_url)
        if not page_html:
            continue

        # ── 2. Secondary opponent guard — team ID in scoreboard section only ──
        # Find the team-header / match-header block (first ~8 KB) and check the
        # team ID there to avoid false positives from sidebar tournament brackets.
        _header_chunk = page_html[:8000]
        if f'/team/{opponent_team_id}/' not in _header_chunk:
            logger.debug(
                f"[h2h] skip {match_id} — team id {opponent_team_id} not in page header "
                f"(slug matched but team not in scoreboard)"
            )
            continue

        # ── 3. Year filter — only use 2026+ H2H data ─────────────────────────
        _unix_m = re.search(r'data-unix=["\'](\d{10,13})["\']', page_html)
        if _unix_m:
            _ts = int(_unix_m.group(1))
            if _ts > 9_999_999_999:   # milliseconds → seconds
                _ts //= 1000
            if _ts < _H2H_MIN_TS:
                logger.info(
                    f"[h2h] skip {match_id} — match is before {_H2H_MIN_YEAR} "
                    f"(ts={_ts})"
                )
                continue
        else:
            logger.debug(f"[h2h] {match_id} — no timestamp found, assuming recent")

        parsed = _parse_match_kills(page_html, player_slug, _match_url)
        if not parsed or not parsed.get('maps'):
            continue

        kills_by_map = [m['kills'] for m in parsed['maps'][:2]]
        if not kills_by_map:
            continue

        maps_found   = len(kills_by_map)
        total_kills  = sum(kills_by_map)
        # Only mark as cleared if we have data from both maps — a single-map
        # result would undercount and produce a false "not cleared" verdict.
        cleared = (maps_found >= 2 and line > 0 and total_kills >= line)

        rec = {
            'match_id':   match_id,
            'kills_by_map': kills_by_map,
            'total_kills': total_kills,
            'avg_kills':  round(_stats.mean(kills_by_map), 1),
            'maps_found': maps_found,
            'cleared':    cleared,
            'partial':    maps_found < 2,
        }
        results.append(rec)
        logger.info(
            f"[h2h] match {match_id} ({slug}): kills={kills_by_map} total={total_kills} "
            f"maps={maps_found} cleared={cleared} (line={line})"
        )

    return results

# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def run_deep_analysis(
    player_id: str,
    player_slug: str,
    player_match_ids: list[tuple[str, str]],
    opponent_name: str,
    stat_type: str,
    baseline_avg: float,
    line: float = 0.0,
) -> dict:
    """
    Orchestrate all analysis dimensions.

    Returns a dict including a 'scouting' sub-dict with:
      - hs_vulnerability:  { rating, label, pct_proxy }
      - role_suppression:  { avg_star_kill, label, clamp_active }
      - h2h_line:          { matches_cleared, of_n, matchup_favorite }
    """
    out: dict = {
        'opponent_display':       None,
        'combined_multiplier':    1.0,
        'components':             {},
        'defensive_profile':      {},
        'rank_info':              {},
        'map_pool':               {},
        'h2h':                    [],
        'h2h_label':              'No H2H data',
        'hs_vulnerability':       {},
        'scouting':               {},
        'matchup_favorite_bonus': False,
        'summary_bullets':        [],
        'error':                  None,
    }

    # ── Find opponent team ──────────────────────────────────────────────────
    team_info = search_team(opponent_name)
    if not team_info:
        out['error'] = f"Team '{opponent_name}' not found on HLTV"
        return out

    opp_id, opp_slug, opp_display = team_info
    out['opponent_display'] = opp_display

    # ── Player's current team (for ranking lookup) ──────────────────────────
    player_team = get_player_team(player_id, player_slug)
    player_team_id   = player_team[0] if player_team else None
    player_team_slug = player_team[1] if player_team else None

    # ── Fetch opponent match data (single pass) ─────────────────────────────
    opp_data = _fetch_opponent_profile(opp_id, n_matches=10)

    # ── Opponent team period stats (90-day aggregate from HLTV /stats/teams/) ─
    # Used as fallback when opp_data has no avg_kills_allowed, and stored on
    # the output dict for optional display.
    team_period: dict | None = None
    try:
        team_period = get_team_period_stats(opp_id, opp_slug, days=90)
        if team_period:
            logger.debug(f"[deep] team period stats {opp_display}: {team_period}")
    except Exception as _tp_e:
        logger.warning(f"[deep] team period stats failed: {_tp_e}")
    out['team_period_stats'] = team_period

    # ── Rankings ────────────────────────────────────────────────────────────
    opp_rank    = get_team_rank(opp_id, opp_slug)
    player_rank = get_team_rank(player_team_id, player_team_slug) if player_team_id and player_team_slug else None

    # ── H2H ─────────────────────────────────────────────────────────────────
    h2h = get_h2h_stats(
        player_id, player_slug, player_match_ids, opp_id,
        opponent_slug=opp_slug, n=3, line=line,
    )
    out['h2h'] = h2h

    # ════════════════════════════════════════════════════════════════════════
    # Compute multipliers
    # ════════════════════════════════════════════════════════════════════════
    components: dict[str, float] = {}
    bullets:    list[str]        = []
    combined = 1.0

    # ── [A] Defensive kills allowed ──────────────────────────────────────────
    avg_allowed = opp_data.get('avg_kills_allowed')

    # If scrape returned no kill data, attempt to estimate avg_allowed from the
    # opponent team's own KPR (90-day period stats).  A team with higher KPR
    # typically plays a more skilled / active CT-side, meaning fewer kills for
    # the opposing player.  We invert the signal:
    #   estimate ≈ BASELINE_KILLS * (1 - (opp_kpr - 0.65) * 0.5)
    # where 0.65 is an approximate pro average KPR.  Clamped to [12, 22].
    if avg_allowed is None and team_period:
        _tp_kpr = team_period.get("kpr")
        if _tp_kpr and 0.10 <= _tp_kpr <= 2.0:
            _baseline_ref = BASELINE_KILLS
            _est = _baseline_ref * (1.0 - (_tp_kpr - 0.65) * 0.50)
            avg_allowed = round(max(12.0, min(22.0, _est)), 1)
            logger.debug(
                f"[deep] avg_kills_allowed estimated from team KPR "
                f"{_tp_kpr:.3f} → {avg_allowed}"
            )

    if avg_allowed:
        # Use the player's own per-map average as the denominator so the
        # adjustment is relative to what THIS player typically scores, not
        # the global tier-1/2 pro baseline (18.5).  A tier-3/4 player who
        # averages 13 kills/map and faces an opponent that allows 12.2 should
        # only get a -6% penalty, not a -25% penalty from 12.2/18.5.
        _def_baseline = baseline_avg if (baseline_avg and baseline_avg > 5) else BASELINE_KILLS
        def_adj = avg_allowed / _def_baseline
        def_adj = max(0.75, min(1.25, def_adj))
        components['defensive'] = round(def_adj, 4)
        combined *= def_adj

        def_pct = round((def_adj - 1) * 100, 1)
        sign = '+' if def_pct >= 0 else ''

        # Label thresholds as percentages of the player baseline
        _tough_thresh = _def_baseline * 0.85   # >15% below player avg = tough
        _soft_thresh  = _def_baseline * 1.10   # >10% above player avg = soft
        if avg_allowed < _tough_thresh:
            def_label = '🛡️ Tough Defense'
            bullets.append(f"Tough defense — only {avg_allowed} kills/player/map allowed ({sign}{def_pct}%)")
        elif avg_allowed > _soft_thresh:
            def_label = '💨 Soft Defense'
            bullets.append(f"Soft defense — {avg_allowed} kills/player/map allowed ({sign}{def_pct}%)")
        else:
            def_label = '⚖️ Average Defense'
    else:
        def_label = '❓ No Data'

    out['defensive_profile'] = {
        'avg_kills_allowed': avg_allowed,
        'label':      def_label,
        'ct_win_pct': opp_data.get('ct_win_pct'),
        't_win_pct':  opp_data.get('t_win_pct'),
        'avg_rounds': opp_data.get('avg_rounds_per_map', 22.0),
        'sample':     opp_data.get('sample_kills', 0),
    }

    # ── [B] CT/T efficiency modifier ─────────────────────────────────────────
    t_pct = opp_data.get('t_win_pct')
    ct_pct = opp_data.get('ct_win_pct')
    t_adj = 1.0
    if t_pct is not None:
        if t_pct >= 55:
            # Aggressive T-side → more opening kills → boost entry fraggers
            t_adj = 1.05
            bullets.append(f"Aggressive T-side ({t_pct}% T-side win rate) → entry openings ↑")
        elif t_pct <= 40:
            t_adj = 0.97
            bullets.append(f"Passive T-side ({t_pct}% T-side win rate) → fewer opening duels")
    if t_adj != 1.0:
        components['t_side'] = round(t_adj, 4)
        combined *= t_adj

    # ── [C] Ranking / stomp risk ─────────────────────────────────────────────
    rank_adj  = 1.0
    stomp     = False
    rank_label = 'Unknown'

    if opp_rank and player_rank:
        diff = opp_rank - player_rank  # positive → player team is ranked better
        rank_label = f"#{player_rank} vs #{opp_rank}"
        if diff >= 20:
            stomp  = True
            rank_adj = 0.88
            rank_label = f"⚠️ Stomp Risk — #{player_rank} vs #{opp_rank}"
            bullets.append(f"Stomp risk: #{player_rank} team vs #{opp_rank} — fewer projected rounds (↓12%)")
        elif diff <= -20:
            rank_adj = 1.08
            rank_label = f"🏆 Elite Clash — #{player_rank} vs #{opp_rank}"
            bullets.append(f"Top-calibre matchup: #{player_rank} vs #{opp_rank} — more rounds projected (↑8%)")
        elif abs(diff) <= 5:
            rank_adj = 1.03
            rank_label = f"⚖️ Even — #{player_rank} vs #{opp_rank}"
            bullets.append(f"Even matchup (#{player_rank} vs #{opp_rank}) — full rounds expected (↑3%)")
    elif opp_rank:
        rank_label = f"Opponent ranked #{opp_rank}"
    elif player_rank:
        rank_label = f"Player's team ranked #{player_rank}"

    if rank_adj != 1.0:
        components['rank'] = round(rank_adj, 4)
    combined *= rank_adj

    raw_gap: int | None = None
    if opp_rank and player_rank:
        raw_gap = abs(opp_rank - player_rank)

    out['rank_info'] = {
        'player_rank': player_rank,
        'opp_rank':    opp_rank,
        'label':       rank_label,
        'stomp_risk':  stomp,
        'rank_gap':    raw_gap,
    }

    # ── [D] Map pool ─────────────────────────────────────────────────────────
    most_played  = opp_data.get('most_played_maps', [])
    least_played = opp_data.get('least_played_maps', [])
    map_adj  = 1.0
    map_label = '⚖️ Mixed / Unknown'

    if most_played:
        types = [MAP_TYPE.get(m, 'average') for m in most_played[:2]]
        modifiers = [MAP_KILL_MODIFIER[t] for t in types]
        map_adj = round(sum(modifiers) / len(modifiers), 4)

        if all(t == 'high_frag' for t in types):
            map_label = f"🔥 High-Frag Pool ({', '.join(most_played[:2]).title()})"
            bullets.append(f"High-frag maps ({', '.join(most_played[:2]).title()}) → kills ↑")
        elif all(t == 'tactical' for t in types):
            map_label = f"🔒 Tactical Pool ({', '.join(most_played[:2]).title()})"
            bullets.append(f"Tactical maps ({', '.join(most_played[:2]).title()}) → kills ↓")
        else:
            map_label = f"⚖️ Mixed ({', '.join(most_played[:2]).title()})"

    if map_adj != 1.0:
        components['map_pool'] = round(map_adj, 4)
    combined *= map_adj

    out['map_pool'] = {
        'most_played':  most_played,
        'least_played': least_played,
        'permaban_hint': least_played[0] if least_played else None,
        'label':         map_label,
    }

    # ── [E] H2H performance ───────────────────────────────────────────────────
    h2h_adj = 1.0
    h2h_label = 'No H2H data found'

    if h2h and baseline_avg > 0:
        h2h_avg   = _stats.mean([m['avg_kills'] for m in h2h])
        h2h_ratio = h2h_avg / baseline_avg
        h2h_adj   = max(0.90, min(1.10, h2h_ratio))  # clamp ±10%
        delta_pct = round((h2h_ratio - 1) * 100, 1)
        sign = '+' if delta_pct >= 0 else ''

        if h2h_ratio >= 1.10:
            h2h_label = f"✅ Farms This Team (H2H avg {round(h2h_avg,1)}K, {sign}{delta_pct}% vs baseline)"
            bullets.append(f"Player historically dominates this matchup — {round(h2h_avg,1)} avg kills H2H")
        elif h2h_ratio <= 0.90:
            h2h_label = f"❌ Struggles Here (H2H avg {round(h2h_avg,1)}K, {sign}{delta_pct}% vs baseline)"
            bullets.append(f"Player historically underperforms vs this team — {round(h2h_avg,1)} avg kills H2H")
        else:
            h2h_label = f"➡️ Neutral (H2H avg {round(h2h_avg,1)}K, {sign}{delta_pct}%)"

        if h2h_adj != 1.0:
            components['h2h'] = round(h2h_adj, 4)
        combined *= h2h_adj

    out['h2h_label'] = h2h_label

    # ── [F] Role Suppression / Defensive Clamp ───────────────────────────────
    avg_star_kill  = opp_data.get('avg_star_kill')
    star_clamp     = opp_data.get('star_suppression', False)
    clamp_label    = '❓ No Data'
    clamp_active   = False

    if avg_star_kill is not None:
        if star_clamp:  # avg star kill < 15
            clamp_active = True
            # Express -1.5 kills as a multiplicative factor relative to baseline
            if baseline_avg > 1.5:
                clamp_factor = (baseline_avg - 1.5) / baseline_avg
                clamp_factor = max(0.80, round(clamp_factor, 4))
            else:
                clamp_factor = 0.88
            components['star_clamp'] = clamp_factor
            combined *= clamp_factor
            clamp_label = f"⚠️ Defensive Clamp (avg top killer held to {avg_star_kill}K/map)"
            bullets.append(
                f"Opponent clamps star players — avg top kill {avg_star_kill}/map "
                f"(↓1.5K projection applied)"
            )
        elif avg_star_kill >= 22:
            clamp_label = f"✅ Star-Friendly — top killers avg {avg_star_kill}K/map"
        elif avg_star_kill >= 18:
            clamp_label = f"⚖️ Average Suppression — top killers avg {avg_star_kill}K/map"
        else:
            clamp_label = f"🛡️ Moderate Clamp — top killers avg {avg_star_kill}K/map"

    # ── [G] HS Vulnerability Index (all props show rating; mult only for HS) ──
    hs_adj      = 1.0
    hs_rating   = '❓ No Data'
    hs_pct_proxy = None

    if avg_allowed is not None:
        avg_rounds_val = opp_data.get('avg_rounds_per_map', 22.0) or 22.0
        # HS proxy: kills allowed per round × 100; typical is ~40-50%
        # We bucket: ≥21 kills/map ≈ "50%+ HS rate" (High); ≤15 ≈ "<42%" (Low)
        if avg_allowed >= 21:
            hs_rating    = '💀 High Vulnerability (50%+ HS rate proxy)'
            hs_pct_proxy = '50%+'
        elif avg_allowed >= 18:
            hs_rating    = '⚠️ Moderate Vulnerability (42–50% proxy)'
            hs_pct_proxy = '42–50%'
        elif avg_allowed >= 15:
            hs_rating    = '⚖️ Average (38–42% proxy)'
            hs_pct_proxy = '38–42%'
        else:
            hs_rating    = '🛡️ Utility Heavy / Low Vulnerability (<38% proxy)'
            hs_pct_proxy = '<38%'

    if stat_type in ('HS', 'hs'):
        # Apply multiplier only for HS props
        if avg_allowed is not None:
            if avg_allowed >= 21:
                hs_adj = 1.12
                bullets.append("Opponent is HS-vulnerable — boosting HS projection (↑12%)")
            elif avg_allowed >= 19:
                hs_adj = 1.04
                bullets.append("Moderate HS vulnerability detected (↑4%)")
            elif avg_allowed <= 15:
                hs_adj = 0.92
                bullets.append("Opponent utility-heavy — HS projection down (↓8%)")
            # T-side aggression bonus for entry fraggers
            if t_pct and t_pct >= 55:
                hs_adj = min(1.25, hs_adj * 1.10)
                bullets.append("Aggressive T-side feeds opening kills → +10% HS bonus")
        if hs_adj != 1.0:
            components['hs_vulnerability'] = round(hs_adj, 4)
        combined *= hs_adj

    out['hs_vulnerability'] = {'label': hs_rating, 'modifier': round(hs_adj, 4)}

    # ── [H] H2H line clearing — Matchup Favorite check ───────────────────────
    # Use the pre-stamped 'cleared' field from get_h2h_stats (which already
    # guards against partial-data false negatives).  Only count records where
    # we had data from BOTH maps; partial records are excluded from of_n so
    # the fraction shown to the user isn't misleading.
    h2h_cleared   = 0
    h2h_of_n      = 0
    h2h_partial   = 0
    matchup_fav   = False

    if h2h and line > 0:
        for rec in h2h[:3]:
            if rec.get('partial'):
                h2h_partial += 1
                continue
            h2h_of_n += 1
            if rec.get('cleared'):
                h2h_cleared += 1
        # Clamp to last 2 complete matches for matchup-favorite decision
        matchup_fav = (h2h_of_n >= 2 and h2h_cleared >= 2)
        if matchup_fav:
            bullets.append(
                f"Matchup Favorite: cleared {line} line in both recent H2H matches → +5% Over"
            )
        if h2h_partial:
            logger.warning(
                f"[h2h] {h2h_partial} H2H match(es) had only 1 map scraped "
                f"— excluded from cleared count to avoid false negatives"
            )

    out['matchup_favorite_bonus'] = matchup_fav

    # ── [I] Economy Impact (Pistol proxy via CT/T win rates) ──────────────────
    ct_pct_val = opp_data.get('ct_win_pct')
    economy_label    = "⚖️ Standard Economy"
    economy_prob_delta = 0.0  # additive % to over_prob (after simulation)

    if ct_pct_val is not None:
        if ct_pct_val < 38:
            economy_label     = "💸 Economy Weak (CT win <38%) → +5% Over"
            economy_prob_delta = +5.0
            bullets.append(f"Weak CT economy ({ct_pct_val}% CT win rate) → eco rounds likely → +5% Over")
        elif ct_pct_val > 58:
            economy_label     = "💪 Economy Elite (CT win >58%) → −5% Over"
            economy_prob_delta = -5.0
            bullets.append(f"Elite CT economy ({ct_pct_val}% CT win rate) → stacks well → −5% Over")
        else:
            economy_label = f"⚖️ Standard Economy (CT {ct_pct_val}%)"

    out['economy_prob_delta'] = economy_prob_delta

    # ── Build Opponent Scouting block ─────────────────────────────────────────
    out['scouting'] = {
        'hs_vulnerability': {
            'rating':    hs_rating,
            'pct_proxy': hs_pct_proxy,
        },
        'role_suppression': {
            'avg_star_kill': avg_star_kill,
            'label':         clamp_label,
            'clamp_active':  clamp_active,
        },
        'h2h_line': {
            'matches_cleared':   h2h_cleared,
            'of_n':              h2h_of_n,
            'h2h_partial':       h2h_partial,
            'matchup_favorite':  matchup_fav,
        },
        'economy_impact': {
            'label':       economy_label,
            'prob_delta':  economy_prob_delta,
            'ct_win_pct':  ct_pct_val,
        },
    }

    # ── Final clamp ───────────────────────────────────────────────────────────
    combined = max(0.70, min(1.40, combined))
    out['combined_multiplier'] = round(combined, 4)
    out['components'] = components
    out['summary_bullets'] = bullets

    total_pct = round((combined - 1) * 100, 1)
    sign = '+' if total_pct >= 0 else ''
    logger.info(
        f"[deep_analysis] {opp_display}: multiplier={combined} ({sign}{total_pct}%) "
        f"components={components} matchup_fav={matchup_fav}"
    )
    return out
