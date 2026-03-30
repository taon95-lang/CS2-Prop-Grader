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

# Pro baseline: avg kills per player per map (tier 1/2 CS2)
BASELINE_KILLS = 18.5

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

_RANK_CACHE: dict[str, tuple] = {}    # team_id → (timestamp, rank | None)
_OPP_CACHE: dict[str, tuple] = {}     # team_id → (timestamp, data_dict)
RANK_TTL = 6 * 3600
OPP_TTL  = 4 * 3600

# ---------------------------------------------------------------------------
# 1. Team ranking
# ---------------------------------------------------------------------------

def get_team_rank(team_id: str, team_slug: str) -> int | None:
    cached = _RANK_CACHE.get(team_id)
    if cached:
        ts, rank = cached
        if time.time() - ts < RANK_TTL:
            return rank

    url = f"{HLTV_BASE}/team/{team_id}/{team_slug}"
    html = _fetch(url)
    if not html:
        _RANK_CACHE[team_id] = (time.time(), None)
        return None

    patterns = [
        r'"ranking":\s*(\d+)',
        r'teamRanking[^<]{0,100}#\s*(\d+)',
        r'World\s+ranking\s*#\s*(\d+)',
        r'ranked\s*#\s*(\d+)',
        r'#(\d+)\s+in the world',
        r'(?:rank|ranking)[^<]{0,60}>\s*#\s*(\d+)',
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            rank = int(m.group(1))
            _RANK_CACHE[team_id] = (time.time(), rank)
            logger.info(f"[rank] {team_slug} → #{rank}")
            return rank

    _RANK_CACHE[team_id] = (time.time(), None)
    logger.warning(f"[rank] could not parse rank for {team_slug}")
    return None

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
    ct_wins, ct_total = 0, 0           # target team's CT rounds won / played
    t_wins, t_total   = 0, 0           # target team's T rounds won / played
    map_counter: dict[str, int] = {}
    rounds_per_map: list[int] = []

    soup_cache = BeautifulSoup('', 'html.parser')  # dummy, reused per page

    for match_id, slug in match_list:
        time.sleep(0.35)
        page_html = _fetch(f"{HLTV_BASE}/matches/{match_id}/{slug}")
        if not page_html:
            continue

        soup = BeautifulSoup(page_html, 'html.parser')
        matchstats = soup.find(id='match-stats')

        # --- Kill data ---
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
                        continue  # skip target team table; want opponent kills
                    for tr in table.find_all('tr')[1:]:
                        m = re.search(r'(\d+)\s*-\s*\d+', tr.get_text())
                        if m:
                            k = int(m.group(1))
                            if 3 <= k <= 60:
                                opp_kills.append(k)

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
    avg_allowed = round(_stats.mean(opp_kills), 1) if len(opp_kills) >= 5 else None
    ct_pct = round(ct_wins / ct_total * 100, 1) if ct_total > 0 else None
    t_pct  = round(t_wins  / t_total  * 100, 1) if t_total  > 0 else None
    avg_rounds = round(_stats.mean(rounds_per_map), 1) if rounds_per_map else 22.0

    sorted_maps = sorted(map_counter.items(), key=lambda x: -x[1])
    most_played  = [m for m, _ in sorted_maps[:3]]
    least_played = [m for m, _ in sorted_maps[-2:]] if len(sorted_maps) >= 4 else []

    data = {
        'avg_kills_allowed': avg_allowed,
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

def get_h2h_stats(
    player_id: str,
    player_slug: str,
    player_match_ids: list[tuple[str, str]],
    opponent_team_id: str,
    n: int = 3,
) -> list[dict]:
    """
    Scan the player's recent match IDs for matches that involved the opponent.
    Returns up to n match records:
      [{'match_id': ..., 'kills_by_map': [22, 18], 'avg_kills': 20.0}, ...]
    """
    results: list[dict] = []

    for match_id, slug in player_match_ids:
        if len(results) >= n:
            break
        time.sleep(0.35)
        page_html = _fetch(f"{HLTV_BASE}/matches/{match_id}/{slug}")
        if not page_html:
            continue

        # Quick check — does this match involve the opponent?
        if f'/team/{opponent_team_id}/' not in page_html:
            continue

        parsed = _parse_match_kills(page_html, player_slug)
        if not parsed or not parsed.get('maps'):
            continue

        kills_by_map = [m['kills'] for m in parsed['maps'][:2]]
        if not kills_by_map:
            continue

        results.append({
            'match_id': match_id,
            'kills_by_map': kills_by_map,
            'avg_kills': round(_stats.mean(kills_by_map), 1),
        })
        logger.info(f"[h2h] match {match_id}: kills={kills_by_map}")

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
) -> dict:
    """
    Orchestrate all analysis dimensions.

    Returns a dict:
      {
        'opponent_display':    'Natus Vincere',
        'combined_multiplier': 0.93,
        'components': { 'defensive': 0.89, 'rank': 1.04, 'map': 1.07, 'h2h': 0.98, 'hs_vuln': 1.0 },

        'defensive_profile':   { avg_kills_allowed, label, ct_win_pct, t_win_pct, ... },
        'rank_info':           { player_rank, opp_rank, label, stomp_risk },
        'map_pool':            { most_played, least_played, map_label },
        'h2h':                 [ {avg_kills, kills_by_map, match_id}, ... ],
        'h2h_label':           '✅ Farms This Team',
        'hs_vulnerability':    { label, modifier },
        'summary_bullets':     [ str, ... ],

        'error':               None | str,  (set only if critical failure)
      }
    """
    out: dict = {
        'opponent_display':    None,
        'combined_multiplier': 1.0,
        'components':          {},
        'defensive_profile':   {},
        'rank_info':           {},
        'map_pool':            {},
        'h2h':                 [],
        'h2h_label':           'No H2H data',
        'hs_vulnerability':    {},
        'summary_bullets':     [],
        'error':               None,
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

    # ── Rankings ────────────────────────────────────────────────────────────
    opp_rank    = get_team_rank(opp_id, opp_slug)
    player_rank = get_team_rank(player_team_id, player_team_slug) if player_team_id and player_team_slug else None

    # ── H2H ─────────────────────────────────────────────────────────────────
    h2h = get_h2h_stats(player_id, player_slug, player_match_ids, opp_id, n=3)
    out['h2h'] = h2h

    # ════════════════════════════════════════════════════════════════════════
    # Compute multipliers
    # ════════════════════════════════════════════════════════════════════════
    components: dict[str, float] = {}
    bullets:    list[str]        = []
    combined = 1.0

    # ── [A] Defensive kills allowed ──────────────────────────────────────────
    avg_allowed = opp_data.get('avg_kills_allowed')
    if avg_allowed:
        def_adj = avg_allowed / BASELINE_KILLS
        def_adj = max(0.75, min(1.25, def_adj))
        components['defensive'] = round(def_adj, 4)
        combined *= def_adj

        def_pct = round((def_adj - 1) * 100, 1)
        sign = '+' if def_pct >= 0 else ''

        if avg_allowed < 16.5:
            def_label = '🛡️ Tough Defense'
            bullets.append(f"Tough defense — only {avg_allowed} kills/player/map allowed ({sign}{def_pct}%)")
        elif avg_allowed > 20.5:
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

    out['rank_info'] = {
        'player_rank': player_rank,
        'opp_rank':    opp_rank,
        'label':       rank_label,
        'stomp_risk':  stomp,
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

    # ── [F] HS Vulnerability Index ────────────────────────────────────────────
    hs_adj   = 1.0
    hs_label = 'N/A (Kills prop)'

    if stat_type in ('HS', 'hs'):
        if avg_allowed is not None:
            if avg_allowed >= 21:
                hs_adj   = 1.12
                hs_label = '💀 Frag Mine — High Vulnerability (50%+ HS rate proxy)'
                bullets.append("Opponent is HS-vulnerable — boosting HS projection (↑12%)")
            elif avg_allowed >= 19:
                hs_adj   = 1.04
                hs_label = '⚠️ Moderate HS Vulnerability'
                bullets.append("Moderate HS vulnerability detected (↑4%)")
            elif avg_allowed <= 15:
                hs_adj   = 0.92
                hs_label = '🛡️ Low HS Vulnerability'
                bullets.append("Opponent gives up few kills — HS projection down (↓8%)")
            else:
                hs_label = '⚖️ Average HS Vulnerability'

            # T-side aggression bonus for entry fraggers (HS prop)
            if t_pct and t_pct >= 55:
                hs_adj = min(1.25, hs_adj * 1.10)
                bullets.append("Aggressive T-side feeds entry kills → +10% HS bonus for Entry fraggers")
        else:
            hs_label = '❓ Insufficient data'

        if hs_adj != 1.0:
            components['hs_vulnerability'] = round(hs_adj, 4)
        combined *= hs_adj

    out['hs_vulnerability'] = {
        'label':    hs_label,
        'modifier': round(hs_adj, 4),
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
        f"components={components}"
    )
    return out
