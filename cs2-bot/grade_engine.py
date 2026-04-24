"""
Grade Engine — The brain behind every !grade call.

Provides:
  compute_form_streak()      – Consecutive hit/miss streak + last-N summary
  compute_variance_tier()    – LOW / MEDIUM / HIGH / VERY HIGH
  compute_confidence_score() – 0–100 integer from weighted signals
  compute_edge_pct()         – Betting edge vs -110 vig (52.38% implied)
  compute_map_intel()        – Per-map kill averages + projected overlay
  compute_risk_flags()       – Active risk strings the bettor should know
  build_verdict_reason()     – One-line justification for the call
  run_lines_table()          – Multi-line over/under table for line shopping
  build_prob_bar()           – Discord-friendly ASCII probability bar
"""

from __future__ import annotations
from statistics import mean, stdev, median
import math
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Form Streak
# ─────────────────────────────────────────────────────────────────────────────

def compute_form_streak(map_stats: list, line: float) -> dict:
    """
    Group map_stats into series by match_id, sum stat_value per series,
    then analyse the hit/miss sequence against `line`.

    Returns:
      type          – 'HOT' | 'COLD' | 'NEUTRAL'
      streak        – length of current consecutive run
      streak_dir    – True=hits, False=misses
      label         – emoji string e.g. '🔥 4 straight hits'
      last4_hits    – hits in last 4 series
      last4_n       – number of series checked (up to 4)
      series_totals – list of per-series stat sums (newest first)
      hits          – list of booleans (newest first)
    """
    # Group by match_id preserving insertion order (newest-first from scraper)
    series_order: list[str] = []
    seen: dict[str, list] = {}
    for m in map_stats:
        mid = str(m.get("match_id", ""))
        if not mid:
            continue
        if mid not in seen:
            seen[mid] = []
            series_order.append(mid)
        seen[mid].append(m["stat_value"])

    series_totals = [sum(seen[mid]) for mid in series_order]
    if not series_totals:
        return {
            "type": "NEUTRAL", "streak": 0, "streak_dir": True,
            "label": "No series data", "last4_hits": 0,
            "last4_n": 0, "series_totals": [], "hits": [],
        }

    hits = [t > line for t in series_totals]

    # Consecutive streak from the front (most recent)
    streak_dir = hits[0]
    streak = 0
    for h in hits:
        if h == streak_dir:
            streak += 1
        else:
            break

    # Last-4 window
    last4    = hits[:4]
    last4_n  = len(last4)
    last4_hits = sum(last4)

    # Classify
    if streak >= 3 and streak_dir:
        form_type = "HOT"
    elif streak >= 3 and not streak_dir:
        form_type = "COLD"
    elif last4_hits >= 3:
        form_type = "HOT"
    elif last4_hits <= 1:
        form_type = "COLD"
    else:
        form_type = "NEUTRAL"

    # Label
    if streak >= 2 and streak_dir:
        label = f"🔥 {streak} straight hits"
    elif streak >= 2 and not streak_dir:
        label = f"❄️ {streak} straight misses"
    else:
        label = f"{last4_hits}/{last4_n} of last {last4_n} series hit"

    return {
        "type":          form_type,
        "streak":        streak,
        "streak_dir":    streak_dir,
        "label":         label,
        "last4_hits":    last4_hits,
        "last4_n":       last4_n,
        "series_totals": series_totals,
        "hits":          hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variance Tier
# ─────────────────────────────────────────────────────────────────────────────

def compute_variance_tier(series_totals: list) -> dict:
    """
    Coefficient of Variation (CV = σ/μ) bucketed into four tiers.
    """
    if len(series_totals) < 2:
        return {"tier": "UNKNOWN", "label": "❓ Low Sample", "std": 0.0, "cv": 0.0}

    mu  = mean(series_totals)
    std = stdev(series_totals)
    cv  = std / mu if mu > 0 else 0.0

    if cv < 0.16:
        tier  = "LOW"
        label = "✅ Low Variance"
    elif cv < 0.24:
        tier  = "MEDIUM"
        label = "🔶 Medium Variance"
    elif cv < 0.32:
        tier  = "HIGH"
        label = "⚠️ High Variance"
    else:
        tier  = "VERY_HIGH"
        label = "🚨 Very High Variance"

    return {
        "tier":  tier,
        "label": label,
        "std":   round(std, 1),
        "cv":    round(cv * 100, 1),   # stored as percent for readability
        "mean":  round(mu, 1),
        "floor": round(min(series_totals), 1),
        "ceil":  round(max(series_totals), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_score(
    sim_result: dict,
    map_stats: list,
    form: dict,
    variance: dict,
    deep: dict | None,
    period_stats: dict | None,
    decision: str,
) -> int:
    """
    Weighted multi-signal confidence score — 0 to 100.

    Signals and their max contributions:
      Simulation probability  ±15
      Historical alignment    ±10
      Hit rate                ±12
      Variance tier           ±10
      Form streak             ±10
      Stomp trap              –20 (hard penalty)
      Deep opponent analysis  ±8
      H2H record              ±5
      Period stats alignment  ±5
      Sample size             ±5
      KAST adjustment         ±3
    """
    score = 50

    # ── Simulation probability ────────────────────────────────────────────────
    over_prob  = sim_result.get("over_prob", 50)
    under_prob = sim_result.get("under_prob", 50)
    # Probability in the direction of the call
    dir_prob = over_prob if decision == "OVER" else under_prob if decision == "UNDER" else 50.0

    if dir_prob >= 70:      score += 15
    elif dir_prob >= 62:    score += 9
    elif dir_prob >= 56:    score += 4
    elif dir_prob <= 30:    score -= 15
    elif dir_prob <= 38:    score -= 9
    elif dir_prob <= 44:    score -= 4

    # ── Historical alignment ──────────────────────────────────────────────────
    line       = sim_result.get("line", 0) or 0
    hist_avg   = sim_result.get("hist_avg", 0) or 0
    hist_med   = sim_result.get("hist_median", 0) or 0

    avg_above = hist_avg > line
    med_above = hist_med > line

    if decision == "OVER":
        if avg_above and med_above:     score += 10
        elif avg_above != med_above:    score -= 6   # split
        elif not avg_above:             score -= 10  # both against direction
    elif decision == "UNDER":
        if not avg_above and not med_above: score += 10
        elif avg_above != med_above:        score -= 6
        elif avg_above:                     score -= 10
    else:
        if avg_above != med_above:      score -= 4

    # ── Hit rate (direction-aware) ────────────────────────────────────────────
    # For an OVER call: high hit rate = good, low hit rate = bad.
    # For an UNDER call: low hit rate = good (player rarely clears = under is likely),
    #   high hit rate = bad (player usually clears = under is risky).
    hit_rate = sim_result.get("hit_rate", 50) or 50
    under_rate = 100 - hit_rate  # how often the prop went UNDER historically

    if decision == "OVER":
        if hit_rate >= 75:      score += 12
        elif hit_rate >= 65:    score += 8
        elif hit_rate >= 57:    score += 4
        elif hit_rate <= 30:    score -= 14
        elif hit_rate <= 40:    score -= 10
        elif hit_rate <= 48:    score -= 5
    elif decision == "UNDER":
        if under_rate >= 75:    score += 12   # player clears < 25% of the time
        elif under_rate >= 65:  score += 8
        elif under_rate >= 57:  score += 4
        elif under_rate <= 30:  score -= 14   # player clears > 70% — bad for under
        elif under_rate <= 40:  score -= 10
        elif under_rate <= 48:  score -= 5
    else:
        # PASS/MISPRICED — neutral directional penalty for extreme mismatch
        if hit_rate <= 30 or hit_rate >= 75:
            score -= 3

    # ── Variance ─────────────────────────────────────────────────────────────
    vtier = variance.get("tier", "MEDIUM")
    if vtier == "LOW":          score += 10
    elif vtier == "MEDIUM":     score += 0
    elif vtier == "HIGH":       score -= 10   # was -7; σ>6 is meaningful volatility
    elif vtier == "VERY_HIGH":  score -= 16   # was -12; σ>9 makes props nearly random

    # ── Form streak ──────────────────────────────────────────────────────────
    ftype   = form.get("type", "NEUTRAL")
    fstreak = form.get("streak", 0)
    fdir    = form.get("streak_dir", True)

    if decision in ("OVER", "PASS"):
        if ftype == "HOT":
            score += 8 if fstreak >= 3 else 4
        elif ftype == "COLD":
            score -= 8 if fstreak >= 3 else 4
    elif decision == "UNDER":
        if ftype == "COLD":
            score += 8 if fstreak >= 3 else 4
        elif ftype == "HOT":
            score -= 8 if fstreak >= 3 else 4

    # ── Stomp trap ───────────────────────────────────────────────────────────
    if sim_result.get("stomp_via_rank") and sim_result.get("stat_type", "") == "Kills":
        score -= 18

    # ── Deep opponent analysis ────────────────────────────────────────────────
    if deep and not deep.get("error"):
        comb = deep.get("combined_multiplier", 1.0) or 1.0
        pct  = (comb - 1.0) * 100
        if decision == "OVER":
            if pct >= 8:    score += 8
            elif pct >= 4:  score += 4
            elif pct <= -8: score -= 8
            elif pct <= -4: score -= 4
        elif decision == "UNDER":
            if pct <= -8:   score += 8
            elif pct <= -4: score += 4
            elif pct >= 8:  score -= 8
            elif pct >= 4:  score -= 4

        # H2H record
        h2h = deep.get("h2h", [])
        if h2h:
            h2h_clears = sum(1 for s in h2h if s.get("cleared"))
            h2h_total  = len(h2h)
            if h2h_total > 0:
                h2h_rate = h2h_clears / h2h_total
                if decision == "OVER":
                    if h2h_rate >= 0.8:   score += 5
                    elif h2h_rate <= 0.3: score -= 5
                elif decision == "UNDER":
                    if h2h_rate <= 0.3:   score += 5
                    elif h2h_rate >= 0.8: score -= 5

    # ── Period stats KPR alignment ────────────────────────────────────────────
    if period_stats:
        pkpr = period_stats.get("kpr")
        if pkpr and 0.1 <= pkpr <= 4.0:
            period_expected = pkpr * 44   # ~22 rounds × 2 maps
            pct_vs_line = (period_expected - line) / max(line, 1) * 100
            if decision == "OVER":
                if pct_vs_line >= 8:    score += 5
                elif pct_vs_line <= -8: score -= 5
            elif decision == "UNDER":
                if pct_vs_line <= -8:   score += 5
                elif pct_vs_line >= 8:  score -= 5

    # ── Sample size ───────────────────────────────────────────────────────────
    n_series = sim_result.get("n_series", 0) or 0
    if n_series >= 9:     score += 5
    elif n_series >= 7:   score += 2
    elif n_series <= 4:   score -= 5
    elif n_series <= 6:   score -= 2

    # ── KAST boosts already applied to over_prob — reflect small confidence lift ─
    if sim_result.get("kast_adj_applied"):
        score += 3 if decision == "OVER" else -3

    # Clamp 5–95 (never claim certainty either way)
    return max(5, min(95, score))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_pct(over_prob: float, decision: str) -> float:
    """
    Edge vs standard -110 vig (implied = 52.38%).
    Returns % edge for the direction of the call (positive = value).
    """
    IMPLIED = 0.5238
    if decision == "OVER":
        return round((over_prob / 100.0 - IMPLIED) * 100, 1)
    elif decision == "UNDER":
        return round(((100.0 - over_prob) / 100.0 - IMPLIED) * 100, 1)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Map Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def compute_map_intel(map_stats: list, likely_maps: list | None, line: float) -> dict:
    """
    Per-map kill averages from historical data.
    Overlays the likely_maps projection when available.
    """
    per_map: dict[str, list] = {}
    for m in map_stats:
        mn = m.get("map_name", "").lower()
        if mn and mn not in ("unknown", ""):
            per_map.setdefault(mn, []).append(m["stat_value"])

    # Per-map averages
    map_avgs: dict[str, float] = {}
    for mn, vals in per_map.items():
        if vals:
            map_avgs[mn] = round(mean(vals), 1)

    sorted_maps = sorted(map_avgs.items(), key=lambda x: x[1], reverse=True)
    best_map  = sorted_maps[0]  if sorted_maps else None
    worst_map = sorted_maps[-1] if len(sorted_maps) > 1 else None

    # Projected map overlay
    projected_vals: list   = []
    projected_labels: list = []
    if likely_maps:
        for lm in likely_maps[:3]:
            mn = lm.lower()
            if mn in per_map and per_map[mn]:
                avg = map_avgs[mn]
                projected_vals.extend(per_map[mn])
                # Arrow compares projected series total (avg × 2 maps) vs series line
                series_proj = avg * 2
                arrow = "↑" if series_proj > line else ("↓" if series_proj < line else "→")
                projected_labels.append(f"{lm.title()} `{avg}` {arrow}")

    projected_avg     = round(mean(projected_vals), 1) if projected_vals else None
    projected_vs_line = None
    if projected_avg is not None and line:
        # projected_avg is per-map; multiply by 2 to get expected series total
        projected_series = projected_avg * 2
        pct  = round((projected_series - line) / max(line, 1) * 100, 1)
        sign = "+" if pct >= 0 else ""
        projected_vs_line = f"{sign}{pct}% vs line"

    projected_series = round(projected_avg * 2, 1) if projected_avg is not None else None

    # Last-10 raw kill values per map (newest-first as scraped)
    per_map_samples: dict[str, list] = {
        mn: [int(round(v)) for v in vals[:10]]
        for mn, vals in per_map.items()
    }

    return {
        "per_map":           map_avgs,
        "per_map_samples":   per_map_samples,
        "sorted_maps":       sorted_maps,
        "best_map":          best_map,
        "worst_map":         worst_map,
        "projected_avg":     projected_avg,      # per-map avg (used internally)
        "projected_series":  projected_series,   # series total projection (avg × 2 maps)
        "projected_labels":  projected_labels,
        "projected_vs_line": projected_vs_line,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Risk Flags
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_flags(
    sim_result: dict,
    variance: dict,
    form: dict,
    deep: dict | None,
    line: float,
) -> list[str]:
    """Returns a list of risk warning strings (empty = no flags)."""
    flags: list[str] = []

    # Stomp trap
    if sim_result.get("stomp_via_rank"):
        rg = sim_result.get("rank_gap", "?")
        flags.append(f"⚠️ Stomp risk — rank gap {rg}, maps may end ~19 rounds")

    # OT risk
    if sim_result.get("close_via_rank"):
        flags.append("⚠️ Close clash — overtime rounds could inflate totals")

    # High variance
    vtier = variance.get("tier", "MEDIUM")
    if vtier == "VERY_HIGH":
        flags.append(f"🚨 Boom/bust player — {variance.get('label')} σ={variance.get('std')}")
    elif vtier == "HIGH":
        flags.append(f"⚠️ High variance — σ={variance.get('std')} (range: {variance.get('floor')}–{variance.get('ceil')})")

    # Cold streak
    if form.get("type") == "COLD" and form.get("streak", 0) >= 2:
        flags.append(f"❄️ Cold streak — {form.get('label', '')}")

    # Split signals (avg & median disagree)
    hist_avg = sim_result.get("hist_avg", 0) or 0
    hist_med = sim_result.get("hist_median", 0) or 0
    if line and (hist_avg > line) != (hist_med > line):
        flags.append(f"⚠️ Split signals — avg {hist_avg} vs median {hist_med} disagree on direction")

    # Small sample
    n_series = sim_result.get("n_series", 10) or 10
    if n_series < 6:
        flags.append(f"⚠️ Thin sample — only {n_series} BO3 series found")

    # Tough opponent
    if deep and not deep.get("error"):
        comb = deep.get("combined_multiplier", 1.0) or 1.0
        if comb < 0.90:
            flags.append(f"🛡️ Tough matchup — deep analysis: {round((comb-1)*100)}% projected adjustment")

    # Very low hit rate (direction-aware)
    hr = sim_result.get("hit_rate", 50) or 50
    decision_rf = sim_result.get("decision", "PASS")
    if decision_rf == "OVER" and hr < 40:
        flags.append(f"⚠️ Low hit rate — only {hr}% cleared this line historically (OVER risk)")
    elif decision_rf == "UNDER" and hr > 65:
        flags.append(f"⚠️ High hit rate — player cleared {hr}% of the time (UNDER risk)")

    # Map intelligence warning — projected series total below the line
    map_intel_obj = sim_result.get("_map_intel_warning")
    if map_intel_obj:
        flags.append(map_intel_obj)

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# 7. Verdict Reason
# ─────────────────────────────────────────────────────────────────────────────

def build_verdict_reason(
    decision: str,
    form: dict,
    variance: dict,
    deep: dict | None,
    sim_result: dict,
    flags: list[str],
) -> str:
    """One-line justification (up to 3 bullet points joined by ·)."""
    reasons: list[str] = []
    line      = sim_result.get("line", 0) or 0
    hit_rate  = sim_result.get("hit_rate", 50) or 50
    over_prob = sim_result.get("over_prob", 50) or 50

    # Form
    ftype = form.get("type", "NEUTRAL")
    if ftype == "HOT" and decision in ("OVER",):
        reasons.append(form.get("label", "hot streak"))
    elif ftype == "COLD" and decision == "UNDER":
        reasons.append(form.get("label", "cold streak"))
    elif ftype == "COLD" and decision == "OVER":
        reasons.append("❄️ cold streak — bet carefully")
    elif ftype == "HOT" and decision == "UNDER":
        reasons.append("🔥 hot streak — UNDER plays against momentum")

    # Hit rate
    if hit_rate >= 65 and decision == "OVER":
        reasons.append(f"{hit_rate:.0f}% hit rate")
    elif hit_rate <= 35 and decision == "UNDER":
        reasons.append(f"only {hit_rate:.0f}% cleared historically")

    # Opponent analysis
    if deep and not deep.get("error"):
        comb = deep.get("combined_multiplier", 1.0) or 1.0
        adj  = round((comb - 1) * 100)
        sign = "+" if adj >= 0 else ""
        if abs(adj) >= 4:
            def_lbl = deep.get("defensive_profile", {}).get("label", "")
            reasons.append(f"{def_lbl} ({sign}{adj}%)")

        h2h = deep.get("h2h", [])
        if h2h:
            clears = sum(1 for s in h2h if s.get("cleared"))
            total  = len(h2h)
            if total >= 2:
                h2h_rate = clears / total
                if h2h_rate >= 0.8 and decision == "OVER":
                    reasons.append(f"H2H {clears}/{total} ✅")
                elif h2h_rate <= 0.3 and decision == "UNDER":
                    reasons.append(f"H2H {clears}/{total} ❌")

    # Variance (only if noteworthy)
    vtier = variance.get("tier", "MEDIUM")
    if vtier == "LOW" and decision in ("OVER", "UNDER"):
        reasons.append("consistent player ✅")
    elif vtier == "VERY_HIGH":
        reasons.append("⚠️ boom/bust risk")

    # Simulation if nothing else
    if not reasons:
        if decision == "OVER" and over_prob >= 58:
            reasons.append(f"simulation {over_prob}% OVER")
        elif decision == "UNDER" and (100 - over_prob) >= 58:
            reasons.append(f"simulation {100-over_prob:.0f}% UNDER")
        elif decision == "PASS":
            reasons.append("signals too mixed for a strong call")
        else:
            reasons.append("marginal edge")

    return " · ".join(reasons[:3])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Multi-Line Probability Table
# ─────────────────────────────────────────────────────────────────────────────

def run_lines_table(
    map_stats: list,
    base_line: float,
    stat_type: str,
    favorite_prob: float,
    likely_maps: list | None,
    rank_gap: int | None,
    period_kpr: float | None,
    step: float = 1.0,
    spread: int = 3,
    period_rating: float | None = None,
    period_adr: float | None = None,
) -> list[dict]:
    """
    Run simulation for base_line ± spread * step increments.
    Returns a list of row dicts for display as a table.
    """
    from simulator import run_simulation

    results: list[dict] = []
    lines_to_check = [round(base_line + (i - spread) * step, 1) for i in range(spread * 2 + 1)]

    for lv in lines_to_check:
        try:
            sim = run_simulation(
                map_stats=map_stats,
                line=lv,
                stat_type=stat_type,
                favorite_prob=favorite_prob,
                likely_maps=likely_maps,
                rank_gap=rank_gap,
                period_kpr=period_kpr,
                period_rating=period_rating,
                period_adr=period_adr,
            )
        except Exception as e:
            logger.warning(f"[lines_table] sim failed for line {lv}: {e}")
            continue

        op = sim.get("over_prob", 50.0)
        up = sim.get("under_prob", 50.0)

        # Value indicator vs -110 vig (52.38% implied)
        IMPLIED = 52.38
        if op >= IMPLIED + 12:       over_val = "🟢🟢"
        elif op >= IMPLIED + 6:      over_val = "🟢"
        elif op >= IMPLIED:          over_val = "⚪"
        else:                        over_val = ""

        if up >= IMPLIED + 12:       under_val = "🔴🔴"
        elif up >= IMPLIED + 6:      under_val = "🔴"
        elif up >= IMPLIED:          under_val = "⚪"
        else:                        under_val = ""

        results.append({
            "line":      lv,
            "over":      round(op, 1),
            "under":     round(up, 1),
            "over_val":  over_val,
            "under_val": under_val,
            "is_base":   abs(lv - base_line) < 0.01,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. ASCII Probability Bar
# ─────────────────────────────────────────────────────────────────────────────

def build_prob_bar(probability: float, width: int = 12) -> str:
    """
    Build an ASCII bar representing a 0–1 probability.
    E.g. probability=0.68, width=12 → '████████░░░░'
    """
    prob = max(0.0, min(1.0, probability))
    filled = round(prob * width)
    empty  = width - filled
    return "█" * filled + "░" * empty


# ─────────────────────────────────────────────────────────────────────────────
# 10. Player Role Fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def determine_role(
    player_slug: str,
    known_awpers: dict,
    avg_kpr: float | None,
    avg_fk_rate: float | None,
    avg_survival: float | None,
    hs_rate: float | None,
) -> tuple[str, str]:
    """
    Returns (role_tag, role_emoji) based on available signals.
    """
    slug = player_slug.lower()

    # AWPer override
    if slug in known_awpers and (known_awpers[slug] < 0.32):
        return "AWPer", "🎯"

    # Aggressive entry fragger: high FK rate, lower survival
    if avg_fk_rate is not None and avg_fk_rate > 0.25:
        if avg_survival is not None and avg_survival < 0.50:
            return "Entry Fragger", "⚡"

    # Passive support/exit: high survival, lower KPR
    if avg_survival is not None and avg_survival > 0.62:
        if avg_kpr is not None and avg_kpr < 0.68:
            return "Support", "🤫"

    # Star rifler: high KPR + high HS%
    if avg_kpr is not None and avg_kpr > 0.80:
        if hs_rate is not None and hs_rate > 0.40:
            return "Star Rifler", "⭐"

    return "Rifler", "🔫"


# ─────────────────────────────────────────────────────────────────────────────
# 11a. Round Swing (Opportunity Metric)
# ─────────────────────────────────────────────────────────────────────────────

def compute_round_swing(
    period_stats: dict | None,
    series_totals: list[float],
    avg_kpr: float | None = None,
) -> dict:
    """
    Round Swing = a player's ability to produce in compressed/short maps.

    HIGH  → stable production regardless of map length (AWPers, opening fraggers,
             high-KAST support with clutch ability)
    MEDIUM → typical — scales reasonably with round count
    LOW   → fragile — output collapses when maps are short

    Signals used (in priority order):
      1. HLTV period stats: kast, kpr, opening_ratio, survival
      2. Series consistency: coefficient of variation (LOW CV = high swing stability)
      3. Avg KPR passed from scraper

    Returns a dict with:
      level   – "HIGH" | "MEDIUM" | "LOW"
      label   – display string
      rationale – one-line explanation
    """
    score = 0  # accumulate signal points; ≥4 = HIGH, ≤-4 = LOW

    kpr      = avg_kpr
    kast     = None
    survival = None
    opening  = None

    if period_stats:
        kpr      = period_stats.get("kpr") or kpr
        kast     = period_stats.get("kast")
        survival = period_stats.get("survival")
        opening  = period_stats.get("opening") or period_stats.get("opening_ratio") or period_stats.get("entrying")

    # KPR: high output per round = survives compression
    if kpr is not None:
        if kpr >= 0.82:   score += 3
        elif kpr >= 0.72: score += 1
        elif kpr <= 0.58: score -= 2

    # KAST: round participation / survival breadth
    if kast is not None:
        kast_val = float(kast) / 100 if float(kast) > 1 else float(kast)
        if kast_val >= 0.75:  score += 2
        elif kast_val >= 0.68: score += 1
        elif kast_val <= 0.55: score -= 2

    # Survival rate: high survival = still alive late in rounds = more kill opportunities
    if survival is not None:
        surv_val = float(survival) / 100 if float(survival) > 1 else float(survival)
        if surv_val >= 0.55:  score += 2
        elif surv_val >= 0.45: score += 0
        elif surv_val <= 0.35: score -= 2

    # Opening rate: opening fraggers guarantee early-round involvement even in short maps
    if opening is not None:
        op_val = float(opening)
        if op_val >= 0.18:  score += 2
        elif op_val >= 0.12: score += 1

    # Series consistency (low CV = reliable production = stable round swing)
    if len(series_totals) >= 4:
        mu  = mean(series_totals)
        std = stdev(series_totals)
        cv  = std / mu if mu > 0 else 1.0
        if cv <= 0.18:   score += 2
        elif cv <= 0.24: score += 1
        elif cv >= 0.32: score -= 1
        elif cv >= 0.40: score -= 2

    # Classify
    if score >= 4:
        level = "HIGH"
        label = "🟢 HIGH Round Swing"
        rationale = "Stable per-round production — survives compressed maps well"
    elif score <= -3:
        level = "LOW"
        label = "🔴 LOW Round Swing"
        rationale = "Output depends on map length — short maps compress this player's ceiling"
    else:
        level = "MEDIUM"
        label = "🟡 MEDIUM Round Swing"
        rationale = "Typical output scaling — moderate match-length sensitivity"

    return {
        "level":     level,
        "label":     label,
        "rationale": rationale,
        "score":     score,
        "_kpr":      kpr,
        "_kast":     kast,
        "_survival": survival,
        "_opening":  opening,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11b. Multi-kill Ceiling (Conversion Metric)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multikill_ceiling(
    series_totals: list[float],
    period_stats: dict | None,
    avg_kpr: float | None = None,
) -> dict:
    """
    Multi-kill = ability to convert rounds into 2K/3K+ kills.

    HIGH  → routinely posts big rounds (spikes well above average, high peak)
    MEDIUM → occasional multi-kills but not dominant ceiling
    LOW   → linear accumulator — rarely spikes, needs round volume

    Signals:
      1. Peak/mean ratio in series_totals  (high peak relative to avg = multi-kill)
      2. Max series total vs line gap
      3. Period stats: firepower, kpr
      4. Variance tier (HIGH variance in a good direction = ceiling)
    """
    if not series_totals or len(series_totals) < 2:
        return {
            "level": "MEDIUM", "label": "🟡 MEDIUM Multi-kill",
            "rationale": "Insufficient data to determine ceiling", "score": 0,
        }

    mu   = mean(series_totals)
    peak = max(series_totals)
    peak_ratio = peak / mu if mu > 0 else 1.0

    score = 0

    # Peak ratio: how much can this player exceed their average?
    if peak_ratio >= 1.65:   score += 4
    elif peak_ratio >= 1.40: score += 2
    elif peak_ratio >= 1.22: score += 1
    elif peak_ratio <= 1.10: score -= 2

    # Top-quartile concentration: does performance cluster at the top?
    sorted_totals = sorted(series_totals, reverse=True)
    top_quarter   = sorted_totals[:max(1, len(sorted_totals)//4)]
    top_avg       = mean(top_quarter)
    top_vs_mean   = (top_avg - mu) / mu if mu > 0 else 0
    if top_vs_mean >= 0.40:   score += 3
    elif top_vs_mean >= 0.25: score += 1

    # KPR: high rate = more multi-kill rounds
    kpr = avg_kpr
    if period_stats:
        kpr = period_stats.get("kpr") or kpr
    if kpr is not None:
        if kpr >= 0.82:   score += 2
        elif kpr >= 0.72: score += 1
        elif kpr <= 0.58: score -= 1

    # Firepower from period_stats (HLTV attribute score 0–100)
    if period_stats:
        fp = period_stats.get("firepower")
        if fp is not None:
            if float(fp) >= 75:   score += 2
            elif float(fp) >= 60: score += 1
            elif float(fp) <= 40: score -= 1

    if score >= 5:
        level = "HIGH"
        label = "🟢 HIGH Multi-kill"
        rationale = f"Ceiling play is real — peak {round(peak, 1)} vs avg {round(mu, 1)} ({round((peak_ratio-1)*100)}% above avg)"
    elif score <= -2:
        level = "LOW"
        label = "🔴 LOW Multi-kill"
        rationale = "Linear accumulator — rarely spikes; needs round volume to hit line"
    else:
        level = "MEDIUM"
        label = "🟡 MEDIUM Multi-kill"
        rationale = "Moderate ceiling — occasional big rounds but not consistently elite"

    return {
        "level":     level,
        "label":     label,
        "rationale": rationale,
        "score":     score,
        "peak":      round(peak, 1),
        "peak_ratio": round(peak_ratio, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11c. Player Profile Classification
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_MATRIX = {
    ("HIGH", "HIGH"):   ("STAR",     "⭐ Star Profile",     "Best OVER — excels in any map length, ceiling is real"),
    ("HIGH", "MEDIUM"): ("STABLE",   "🔒 Stable Producer",  "Reliable OVER — consistent but capped upside"),
    ("HIGH", "LOW"):    ("GRINDER",  "⚙️ Grinder",          "Safe floor — grinds volume but needs rounds to get there"),
    ("MEDIUM","HIGH"):  ("VOLATILE", "⚡ Volatile Spike",   "High ceiling but needs the right map to hit it"),
    ("MEDIUM","MEDIUM"):("BALANCED", "⚖️ Balanced",         "Middle of the road — small edges only at right line"),
    ("MEDIUM","LOW"):   ("SOFT",     "🔵 Soft Accumulator", "Needs round volume — short maps hurt this profile"),
    ("LOW",  "HIGH"):   ("BOOM_BUST","💣 Boom/Bust",        "High ceiling but boom/bust — short maps = Under risk"),
    ("LOW",  "MEDIUM"): ("FRAGILE",  "⚠️ Fragile",          "Depends on long maps — compressed maps = clear Under"),
    ("LOW",  "LOW"):    ("FADE",     "❌ Fade",             "Clear Under profile — weak round swing AND ceiling"),
}

def classify_player_profile(round_swing: dict, multikill: dict) -> dict:
    """
    Matrix:
      High Swing + High Multi → STAR (best Over)
      High Swing + Low Multi  → STABLE (safe but capped)
      Low Swing  + High Multi → BOOM_BUST (volatile)
      Low Swing  + Low Multi  → FADE (clear Under)
    """
    rs_level = round_swing.get("level", "MEDIUM")
    mk_level = multikill.get("level",    "MEDIUM")
    key      = (rs_level, mk_level)
    code, label, desc = _PROFILE_MATRIX.get(key, ("BALANCED", "⚖️ Balanced", "Mixed signals"))
    return {"code": code, "label": label, "description": desc, "swing": rs_level, "multi": mk_level}


# ─────────────────────────────────────────────────────────────────────────────
# 11c-bis. 100-Point Weighted Scoring Model (Doc 1 — strict edge-detection)
# ─────────────────────────────────────────────────────────────────────────────
#
# "Scoring should run on a 100-point weighted model where ceiling (how often
#  the player hits 30+ or 35+) is the most important driver, followed by
#  hit rate, multi-kill ability, round swing, match-length risk, role, and
#  consistency, with averages and medians heavily de-emphasized since kills
#  props are distribution and ceiling-driven rather than mean-driven."
#
# Component weights (sum = 100):
#   Ceiling frequency .... 25  (line+3 hit rate × 0.5  +  line+8 hit rate × 0.5)
#   Hit rate ............. 20  (over-line conversion)
#   Multi-kill ........... 15  (HIGH/MED/LOW from grade_pkg)
#   Round swing .......... 12  (HIGH/MED/LOW from grade_pkg)
#   Match-length risk .... 12  (favorite_prob, stomp, projected_rounds)
#   Role ................. 8   (Star/AWPer/Entry > Support > Rifler)
#   Consistency .......... 8   (σ ≤ 4 best, σ ≥ 9 worst)
#
# Direction-aware: for UNDER decisions, ceiling/hit-rate/role components are
# inverted (low MK + low swing + plain rifler + short maps = strong UNDER score).

_WS_WEIGHTS = {
    "ceiling":      25,
    "hit_rate":     20,
    "multikill":    15,
    "round_swing":  12,
    "match_length": 12,
    "role":          8,
    "consistency":   8,
}

def _ws_label(total: float) -> tuple[str, str]:
    if total >= 85: return "🟢 ELITE",     "Conviction play — all signals align"
    if total >= 75: return "✅ STRONG",    "Strong play — supports elite tiers"
    if total >= 65: return "🟡 LEAN",      "Playable lean — needs price"
    if total >= 50: return "⚪ MARGINAL",  "Marginal — skip unless price is plus"
    return "🚫 NO BET", "Below threshold — auto-skip enforced"


def _level_to_unit(level: str, invert: bool = False) -> float:
    """HIGH/MEDIUM/LOW level → 0..1 unit score, optionally inverted.
    MEDIUM centred at 0.5 so OVER and UNDER are symmetric (no built-in bias).
    """
    table = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
    base = table.get((level or "MEDIUM").upper(), 0.5)
    return (1.0 - base) if invert else base


def compute_weighted_score_100(
    series_totals:    list[float],
    line:             float,
    decision:         str,
    hit_rate:         float,
    round_swing:      dict,
    multikill:        dict,
    favorite_prob:    float,
    stomp_via_rank:   bool,
    projected_rounds: int | float | None,
    role_tag:         str | None,
    stability_std:    float,
) -> dict:
    """
    Direction-aware 100-point weighted score.

    Returns:
      {
        "total":        float (0-100),
        "label":        str,
        "verdict":      str,
        "components": {
            "<name>": {"score": float, "weight": int, "points": float, "detail": str},
            ...
        },
        "direction":    "OVER" | "UNDER",
        "ceiling_pct":  float (ceiling component as % of its max — used for
                                elite-tier confirmation gate),
      }
    """
    direction = decision if decision in {"OVER", "UNDER"} else "OVER"
    is_over   = direction == "OVER"

    # ── 1. Ceiling frequency (25 pts) — HARSHER ──────────────────────────────
    # Higher bars (line+5 / line+10) and peak weighted more (it's harder to fake).
    # Multiplicative penalty: if peak threshold has never been hit, ceiling capped
    # at 0.45 — a player who never spikes can't earn full ceiling credit.
    if series_totals:
        n = len(series_totals)
        if is_over:
            clear_pct = sum(1 for x in series_totals if x >= line + 5)  / n
            peak_pct  = sum(1 for x in series_totals if x >= line + 10) / n
            detail_str = f"{clear_pct*100:.0f}% ≥ line+5, {peak_pct*100:.0f}% ≥ line+10"
        else:
            clear_pct = sum(1 for x in series_totals if x <= line - 5)  / n
            peak_pct  = sum(1 for x in series_totals if x <= line - 10) / n
            detail_str = f"{clear_pct*100:.0f}% ≤ line-5, {peak_pct*100:.0f}% ≤ line-10"
        ceiling_score = 0.35 * clear_pct + 0.65 * peak_pct
        # Hard cap when peak threshold never hit (player has no real ceiling)
        if peak_pct < 0.10:
            ceiling_score = min(ceiling_score, 0.45)
    else:
        ceiling_score = 0.0
        detail_str = "no series data"

    # ── 2. Hit rate (20 pts) — PENALIZES HR ≤ 50% ────────────────────────────
    # hit_rate is over-side conversion (0..1). For UNDER, invert.
    # Coinflip (50%) and below = active penalty. Neutral pivot moved to 55%
    # so that 50% itself yields a small negative (-4 pts).
    # 75%+ → full credit (+20). 25% or worse → -20 pts.
    hr = hit_rate if is_over else (1.0 - hit_rate)
    if hr >= 0.55:
        hr_score = min(1.0, (hr - 0.55) / 0.20)        # 55→0, 75→1.0
    else:
        hr_score = -min(1.0, (0.55 - hr) / 0.30)       # 55→0, 50→-0.17, 25→-1.0
    hr_detail = f"{hr*100:.0f}% {'over' if is_over else 'under'} conversion"
    if hr <= 0.50:
        hr_detail += " ⚠️ penalty"

    # ── 3. Multi-kill (15 pts) ──────────────────────────────────────────────
    mk_level   = multikill.get("level", "MEDIUM")
    mk_score   = _level_to_unit(mk_level, invert=not is_over)
    mk_detail  = f"{mk_level} multi-kill"

    # ── 4. Round swing (12 pts) ─────────────────────────────────────────────
    rs_level   = round_swing.get("level", "MEDIUM")
    rs_score   = _level_to_unit(rs_level, invert=not is_over)
    rs_detail  = f"{rs_level} round swing"

    # ── 5. Match-length risk (12 pts) — STOMP HARSHER ───────────────────────
    pr = projected_rounds or 44
    if is_over:
        # Long maps + competitive odds = high score
        if   pr >= 46:                       length_score = 1.00
        elif pr >= 44:                       length_score = 0.85
        elif pr >= 42:                       length_score = 0.65
        elif pr >= 40:                       length_score = 0.40
        else:                                length_score = 0.15
        # Stomp condition cripples OVER match-length score (was 0.20 → 0.05)
        if stomp_via_rank or favorite_prob >= 0.72:
            length_score = min(length_score, 0.05)
        elif favorite_prob >= 0.65:
            length_score = min(length_score, 0.30)
        ml_detail = f"~{pr} rds, fav {favorite_prob*100:.0f}%"
        if stomp_via_rank: ml_detail += " 🚨 stomp"
    else:
        # Short maps benefit UNDERS
        if   pr <= 38:                       length_score = 1.00
        elif pr <= 40:                       length_score = 0.85
        elif pr <= 42:                       length_score = 0.65
        elif pr <= 44:                       length_score = 0.40
        else:                                length_score = 0.15
        if stomp_via_rank or favorite_prob >= 0.72:
            length_score = max(length_score, 0.90)   # stomp favors UNDER
        ml_detail = f"~{pr} rds, fav {favorite_prob*100:.0f}%"
        if stomp_via_rank: ml_detail += " ✅ stomp helps"

    # ── 6. Role (8 pts) — SYMMETRIC (no OVER/UNDER bias) ────────────────────
    # Both directions can hit 1.0 max for their best-fit role.
    role  = (role_tag or "").lower()
    if is_over:
        role_score = {
            "awper":         1.00,
            "star rifler":   1.00,
            "entry fragger": 0.95,
            "support":       0.40,
            "rifler":        0.40,
        }.get(role, 0.50)
    else:
        # Mirror — role types that fade well get full credit on UNDER
        role_score = {
            "awper":         0.40,
            "star rifler":   0.40,
            "entry fragger": 0.45,
            "support":       1.00,
            "rifler":        1.00,
        }.get(role, 0.50)
    role_detail = f"{role_tag or '—'}"

    # ── 7. Consistency (8 pts) — TIGHTER, ALLOWS NEGATIVE ───────────────────
    # σ ≤ 3 → full credit; σ ≥ 9 → -0.5 (active penalty for chaotic players).
    s = stability_std or 0.0
    if   s <= 3:  cons_score = 1.00
    elif s <= 5:  cons_score = 0.65
    elif s <= 7:  cons_score = 0.30
    elif s <= 9:  cons_score = 0.00
    else:         cons_score = -0.50
    cons_detail = f"σ={s:.1f}"
    if cons_score < 0: cons_detail += " ⚠️ penalty"

    # ── Assemble ─────────────────────────────────────────────────────────────
    raw = {
        "ceiling":      (ceiling_score, detail_str),
        "hit_rate":     (hr_score,      hr_detail),
        "multikill":    (mk_score,      mk_detail),
        "round_swing":  (rs_score,      rs_detail),
        "match_length": (length_score,  ml_detail),
        "role":         (role_score,    role_detail),
        "consistency":  (cons_score,    cons_detail),
    }

    components: dict = {}
    total = 0.0
    for k, (sc, det) in raw.items():
        w   = _WS_WEIGHTS[k]
        pts = round(sc * w, 1)
        total += pts
        components[k] = {
            "score":  round(sc, 3),
            "weight": w,
            "points": pts,
            "detail": det,
        }

    total = round(total, 1)
    label, verdict = _ws_label(total)

    ceiling_pct = (components["ceiling"]["points"] / _WS_WEIGHTS["ceiling"]) * 100

    return {
        "total":       total,
        "label":       label,
        "verdict":     verdict,
        "components":  components,
        "direction":   direction,
        "ceiling_pct": round(ceiling_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11d. Scenario Projections (Short-map vs Normal-map)
# ─────────────────────────────────────────────────────────────────────────────

# Typical round counts per map by scenario
_ROUNDS_SHORT  = 19   # stomp/blowout scenario (rank gap > 15)
_ROUNDS_NORMAL = 26   # standard competitive map
_ROUNDS_LONG   = 30   # close match / OT risk

def compute_scenario_projections(
    kpr:           float | None,
    hist_avg:      float,
    line:          float,
    round_swing:   dict,
    multikill:     dict,
    stomp_via_rank: bool = False,
    close_via_rank: bool = False,
    n_maps:        int   = 2,
) -> dict:
    """
    Build short-map and normal-map projected series totals.

    Formula: kpr × rounds_per_map × n_maps
    If KPR unavailable, fallback to hist_avg ratio scaling.

    Round Swing modifies short-map projection:
      HIGH swing  → short_proj not further penalized (player survives compression)
      LOW swing   → short_proj capped at 90% of kpr-based value

    Multi-kill modifies the upper ceiling estimate:
      HIGH multi  → ceiling = normal_proj × 1.20
      LOW multi   → ceiling = normal_proj × 1.05
    """
    rounds_short  = _ROUNDS_SHORT
    rounds_normal = _ROUNDS_NORMAL

    # If stomp risk is known, short scenario is more likely
    if stomp_via_rank:
        rounds_short  = 18
        rounds_normal = 23   # even normal maps trend shorter vs weak opponents

    if close_via_rank:
        rounds_normal = 28   # close matches go longer
        rounds_short  = 23

    rs_level = round_swing.get("level", "MEDIUM")
    mk_level = multikill.get("level",   "MEDIUM")

    if kpr and kpr > 0:
        short_proj_raw  = round(kpr * rounds_short  * n_maps, 1)
        normal_proj_raw = round(kpr * rounds_normal * n_maps, 1)
    else:
        # Fallback: scale from hist_avg using round ratio
        ratio = rounds_short / rounds_normal
        short_proj_raw  = round(hist_avg * ratio, 1)
        normal_proj_raw = round(hist_avg, 1)

    # Round Swing adjustment
    short_proj = short_proj_raw
    if rs_level == "LOW":
        short_proj = round(short_proj_raw * 0.88, 1)   # compression penalty
    elif rs_level == "HIGH":
        short_proj = round(short_proj_raw * 1.04, 1)   # stable bonus

    normal_proj = normal_proj_raw

    # Multi-kill ceiling
    if mk_level == "HIGH":
        ceiling = round(normal_proj * 1.22, 1)
    elif mk_level == "LOW":
        ceiling = round(normal_proj * 1.05, 1)
    else:
        ceiling = round(normal_proj * 1.12, 1)

    def _edge(proj: float) -> str:
        if line <= 0:
            return "—"
        pct = round((proj - line) / line * 100, 1)
        sign = "+" if pct >= 0 else ""
        clears = "✅ CLEARS" if proj > line else "❌ FALLS SHORT"
        return f"{clears} ({sign}{pct}%)"

    return {
        "short_proj":       short_proj,
        "short_clears":     short_proj > line,
        "short_edge_str":   _edge(short_proj),
        "normal_proj":      normal_proj,
        "normal_clears":    normal_proj > line,
        "normal_edge_str":  _edge(normal_proj),
        "ceiling":          ceiling,
        "rounds_short":     rounds_short,
        "rounds_normal":    rounds_normal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11e. Mispriced Prop Classification (Guide Framework)
# ─────────────────────────────────────────────────────────────────────────────

def classify_misprice(
    hist_avg:      float,
    hist_median:   float,
    hit_rate:      float,
    line:          float,
    round_swing:   dict,
    multikill:     dict,
    scenario:      dict,
    decision:      str,
) -> dict:
    """
    Apply the guide's Mispriced Prop Identification framework.

    Returns:
      misprice_type – "OVER_VALUE" | "UNDER_VALUE" | "CLEAR_ERROR" | "LEAN" | "NONE"
      label         – display string
      bullets       – list of reasons
    """
    avg_above  = hist_avg    > line
    med_above  = hist_median > line
    hit_over   = hit_rate / 100
    rs         = round_swing.get("level", "MEDIUM")
    mk         = multikill.get("level",   "MEDIUM")
    short_ok   = scenario.get("short_clears", False)
    normal_ok  = scenario.get("normal_clears", False)
    bullets: list[str] = []

    if decision == "OVER":
        # OVER VALUE conditions from guide
        strong_avg = avg_above and med_above
        strong_hr  = hit_over >= 0.60
        ok_swing   = rs in ("HIGH", "MEDIUM")
        real_ceil  = mk in ("HIGH", "MEDIUM")
        short_pass = short_ok

        met = sum([strong_avg, strong_hr, ok_swing, real_ceil, short_pass])

        if strong_avg:   bullets.append(f"✅ Avg ({hist_avg}) AND Median ({hist_median}) both above line ({line})")
        if strong_hr:    bullets.append(f"✅ Hit rate {hit_over*100:.0f}% — strong historical conversion")
        if ok_swing:     bullets.append(f"✅ Round Swing: {rs} — opportunity is real")
        if real_ceil:    bullets.append(f"✅ Multi-kill: {mk} — ceiling supports the line")
        if short_pass:   bullets.append(f"✅ Clears short-map projection ({scenario['short_proj']} kills @ {scenario['rounds_short']} rds/map)")
        else:            bullets.append(f"⚠️ Short-map risk — only {scenario['short_proj']} projected in compressed maps")

        if not avg_above: bullets.append(f"❌ Avg {hist_avg} below line {line}")
        if not med_above: bullets.append(f"❌ Median {hist_median} below line {line}")

        gap = round(abs(hist_avg - line), 1)
        if gap >= line * 0.12 and (not avg_above or not med_above):
            return {
                "misprice_type": "UNDER_VALUE",
                "label":         "❌ Line favors UNDER — avg/median distant from line",
                "bullets":       bullets,
            }

        if met >= 4:
            return {"misprice_type": "OVER_VALUE",  "label": "🟢 OVER VALUE — Multiple layers confirm edge", "bullets": bullets}
        if met == 3:
            return {"misprice_type": "LEAN",        "label": "🔵 LEAN OVER — Partial alignment", "bullets": bullets}
        if gap >= line * 0.15 and avg_above and med_above:
            return {"misprice_type": "CLEAR_ERROR", "label": "🚨 CLEAR ERROR — Line far below avg + median", "bullets": bullets}
        return {"misprice_type": "NONE", "label": "⚪ No clear edge", "bullets": bullets}

    elif decision == "UNDER":
        avg_weak   = not avg_above
        med_weak   = not med_above
        low_hr     = hit_over <= 0.45
        weak_swing = rs == "LOW"
        weak_ceil  = mk == "LOW"
        short_fail = not short_ok

        met = sum([avg_weak, med_weak, low_hr, weak_swing, short_fail])

        if avg_weak:     bullets.append(f"✅ Avg ({hist_avg}) below line ({line})")
        if med_weak:     bullets.append(f"✅ Median ({hist_median}) below line ({line})")
        if low_hr:       bullets.append(f"✅ Only {hit_over*100:.0f}% cleared this line historically")
        if weak_swing:   bullets.append("✅ Low Round Swing — output collapses in compressed maps")
        if weak_ceil:    bullets.append("✅ Low Multi-kill — needs round volume to accumulate")
        if short_fail:   bullets.append(f"✅ Fails short-map ({scenario['short_proj']} < {line})")

        gap = round(abs(hist_avg - line), 1)
        if gap >= line * 0.15 and avg_weak and med_weak:
            return {"misprice_type": "CLEAR_ERROR", "label": "🚨 CLEAR ERROR — Line far above avg + median", "bullets": bullets}
        if met >= 3:
            return {"misprice_type": "UNDER_VALUE", "label": "🔴 UNDER VALUE — Multiple layers confirm edge", "bullets": bullets}
        if met == 2:
            return {"misprice_type": "LEAN",        "label": "🔵 LEAN UNDER — Partial alignment", "bullets": bullets}
        return {"misprice_type": "NONE", "label": "⚪ No clear edge", "bullets": bullets}

    return {"misprice_type": "NONE", "label": "⚪ PASS — No directional alignment", "bullets": []}


# ─────────────────────────────────────────────────────────────────────────────
# 11. Full Grade Package — convenience wrapper called by bot.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_grade_package(
    sim_result: dict,
    map_stats: list,
    deep: dict | None,
    period_stats: dict | None,
) -> dict:
    """
    Top-level entry point. Wraps all analysis functions into one dict.
    Now includes the full CS2 prop grading framework:
      - Round Swing (opportunity metric)
      - Multi-kill (conversion metric / ceiling)
      - Player Profile matrix (Swing × Multi)
      - Scenario projections (short-map vs normal-map)
      - Mispriced prop classification (Over Value / Under Value / Clear Error)

    Attach as sim_result['grade_pkg'] in bot.py for embed access.
    """
    line       = sim_result.get("line", 0) or 0
    decision   = sim_result.get("decision", "PASS")
    over_prob  = sim_result.get("over_prob", 50.0) or 50.0
    hist_avg   = sim_result.get("hist_avg",   0) or 0
    hist_med   = sim_result.get("hist_median", 0) or 0
    hit_rate   = sim_result.get("hit_rate",   50) or 50
    stomp      = sim_result.get("stomp_via_rank", False)
    close      = sim_result.get("close_via_rank", False)

    # ── Core computations ────────────────────────────────────────────────────
    series_totals = _extract_series_totals(map_stats)
    form          = compute_form_streak(map_stats, line)
    variance      = compute_variance_tier(series_totals)
    likely_maps   = None
    if deep and not deep.get("error"):
        mp = deep.get("map_pool", {})
        likely_maps = mp.get("most_played", [])

    map_intel  = compute_map_intel(map_stats, likely_maps, line)

    # ── Framework metrics (guide) ─────────────────────────────────────────────
    # Pull KPR from period_stats or sim_result
    avg_kpr = None
    if period_stats:
        avg_kpr = period_stats.get("kpr")
    if avg_kpr is None:
        avg_kpr = sim_result.get("avg_kpr")

    round_swing = compute_round_swing(
        period_stats  = period_stats,
        series_totals = series_totals,
        avg_kpr       = avg_kpr,
    )
    multikill = compute_multikill_ceiling(
        series_totals = series_totals,
        period_stats  = period_stats,
        avg_kpr       = avg_kpr,
    )
    player_profile = classify_player_profile(round_swing, multikill)

    # ── 100-Point Weighted Score (Doc 1) ─────────────────────────────────────
    # Reads role tag + favorite_prob from sim_result (passed through bot.py).
    weighted_score = None
    try:
        _stab_std = sim_result.get("sim_std") or sim_result.get("sigma_mad") or 0.0
        weighted_score = compute_weighted_score_100(
            series_totals    = series_totals,
            line             = float(line),
            decision         = decision,
            hit_rate         = float(hit_rate) / 100.0 if hit_rate > 1 else float(hit_rate),
            round_swing      = round_swing,
            multikill        = multikill,
            favorite_prob    = float(sim_result.get("favorite_prob", 0.55) or 0.55),
            stomp_via_rank   = bool(stomp),
            projected_rounds = sim_result.get("total_projected_rounds")
                               or sim_result.get("projected_rounds"),
            role_tag         = sim_result.get("role_tag"),
            stability_std    = float(_stab_std),
        )
    except Exception:
        weighted_score = None

    scenario       = compute_scenario_projections(
        kpr            = avg_kpr,
        hist_avg       = float(hist_avg),
        line           = float(line),
        round_swing    = round_swing,
        multikill      = multikill,
        stomp_via_rank = bool(stomp),
        close_via_rank = bool(close),
        n_maps         = 2,
    )
    misprice = classify_misprice(
        hist_avg    = float(hist_avg),
        hist_median = float(hist_med),
        hit_rate    = float(hit_rate),
        line        = float(line),
        round_swing = round_swing,
        multikill   = multikill,
        scenario    = scenario,
        decision    = decision,
    )

    # ── Risk flags, confidence, edge ─────────────────────────────────────────
    flags = compute_risk_flags(sim_result, variance, form, deep, line)

    # Augment flags from framework
    rs_level = round_swing.get("level", "MEDIUM")
    mk_level = multikill.get("level",   "MEDIUM")
    if rs_level == "LOW" and not scenario.get("short_clears") and decision == "OVER":
        flags.insert(0, f"⚠️ Match-length risk — fails short-map projection ({scenario['short_proj']} < {line})")
    if mk_level == "LOW" and decision == "OVER":
        flags.append("⚠️ Low Multi-kill ceiling — linear accumulator needs round volume")
    if player_profile.get("code") in ("FADE", "FRAGILE") and decision == "OVER":
        flags.insert(0, f"🚨 Player profile mismatch — {player_profile['label']} profile taking an OVER")

    confidence = compute_confidence_score(
        sim_result   = sim_result,
        map_stats    = map_stats,
        form         = form,
        variance     = variance,
        deep         = deep,
        period_stats = period_stats,
        decision     = decision,
    )

    # Framework confidence adjustments
    if rs_level == "HIGH" and mk_level == "HIGH" and decision == "OVER":
        confidence = min(95, confidence + 6)
    elif rs_level == "LOW" and mk_level == "LOW" and decision == "OVER":
        confidence = max(5, confidence - 8)
    elif rs_level == "LOW" and decision == "OVER" and not scenario.get("short_clears"):
        confidence = max(5, confidence - 5)
    if misprice.get("misprice_type") == "CLEAR_ERROR":
        confidence = min(95, confidence + 5)

    edge_pct = compute_edge_pct(over_prob, decision)
    reason   = build_verdict_reason(decision, form, variance, deep, sim_result, flags)

    return {
        "form":           form,
        "variance":       variance,
        "map_intel":      map_intel,
        "flags":          flags,
        "confidence":     confidence,
        "edge_pct":       edge_pct,
        "reason":         reason,
        "series_totals":  series_totals,
        # ── New framework metrics ──
        "round_swing":    round_swing,
        "multikill":      multikill,
        "player_profile": player_profile,
        "scenario":       scenario,
        "misprice":       misprice,
        "avg_kpr":        avg_kpr,
        "weighted_score": weighted_score,
    }


def build_analysis_blurb(
    player_name: str,
    sim_result: dict,
    form: dict,
    variance: dict,
    deep: dict | None,
    period_stats: dict | None,
    map_intel: dict | None,
) -> str:
    """
    Return a short 2–3 sentence analyst-style narrative explaining why the
    player is going OVER or UNDER. Written in plain English, not bullets.
    References the player's unique traits, form, and matchup context.
    """
    decision   = sim_result.get("decision", "PASS")
    line       = sim_result.get("line", 0) or 0
    over_prob  = sim_result.get("over_prob", 50) or 50
    under_prob = 100 - over_prob
    hist_avg   = sim_result.get("hist_avg", 0) or 0
    hist_med   = sim_result.get("hist_median", 0) or 0
    hit_rate   = sim_result.get("hit_rate", 50) or 50
    n_series   = sim_result.get("n_series", 0) or 0
    trend_pct  = sim_result.get("trend_pct", 0) or 0
    kpr        = (period_stats or {}).get("kpr")
    adr_val    = (period_stats or {}).get("adr")
    rating     = (period_stats or {}).get("rating")
    hs_pct     = (period_stats or {}).get("hs_pct")
    total_rds  = sim_result.get("total_projected_rounds")
    stomp      = sim_result.get("stomp_via_rank", False)
    n_name     = player_name.strip().title() if player_name else "This player"

    # ── Sentence 1: Player profile + role + key attribute ─────────────────────
    sentences: list[str] = []

    ftype  = form.get("type", "NEUTRAL")
    streak = form.get("streak", 0)
    vtier  = variance.get("tier", "MEDIUM")

    # Build a role descriptor from the available signals
    if kpr is not None and rating is not None:
        if kpr >= 0.85 and rating >= 1.15:
            role_desc = "elite star player with consistently high output"
        elif kpr >= 0.75:
            role_desc = "high-volume rifler who produces at a strong rate across maps"
        elif kpr is not None and kpr < 0.60:
            role_desc = "support-oriented player whose kill numbers run lean"
        else:
            role_desc = "reliable contributor in his team's system"
    elif kpr is not None:
        if kpr >= 0.80:
            role_desc = "high-volume fragger who racks up kills consistently"
        elif kpr < 0.60:
            role_desc = "low-KPR player whose role limits his kill ceiling"
        else:
            role_desc = "mid-range producer across BO3 series"
    else:
        role_desc = "player whose historical output is the primary signal here"

    # HS rate flavour
    hs_flavour = ""
    if hs_pct is not None:
        if hs_pct >= 58:
            hs_flavour = f" His {hs_pct:.0f}% headshot rate signals a dueling, entry-first style."
        elif hs_pct >= 45:
            hs_flavour = f" A {hs_pct:.0f}% HS rate suggests he trades efficiently in aim duels."

    # Variance flavour
    var_flavour = ""
    if vtier == "VERY_HIGH":
        var_flavour = " He is a boom-or-bust type — the ceiling is real but so is the floor."
    elif vtier == "LOW":
        var_flavour = " He runs with low variance, meaning his output is predictable and steady."
    elif vtier == "HIGH":
        var_flavour = " His numbers swing series-to-series, so the range matters as much as the average."

    sentences.append(
        f"{n_name} is a {role_desc}.{hs_flavour}{var_flavour}"
    )

    # ── Sentence 2: Form + average vs line — the core 'why' ───────────────────
    avg_gap_pct = ((hist_avg - line) / max(line, 1) * 100) if line else 0

    core_why = ""
    if decision == "OVER":
        if ftype == "HOT" and streak >= 2:
            core_why = (
                f"He's on a {streak}-series hot streak and his recent average of "
                f"{hist_avg} kills sits {avg_gap_pct:.0f}% above the {line} line, "
                f"with the simulation returning a {over_prob:.0f}% OVER probability."
            )
        elif hist_avg > line * 1.06:
            core_why = (
                f"His recent average of {hist_avg} kills clears this line "
                f"comfortably, hitting the OVER in {hit_rate:.0f}% of his last "
                f"{n_series} series."
            )
        elif trend_pct >= 12:
            core_why = (
                f"His output is trending {trend_pct:.0f}% above his career baseline, "
                f"and the sim projects an OVER {over_prob:.0f}% of the time."
            )
        else:
            core_why = (
                f"The simulation gives a {over_prob:.0f}% OVER probability, "
                f"with a recent average of {hist_avg} vs the line at {line}."
            )
    elif decision == "UNDER":
        if ftype == "COLD" and streak >= 2:
            core_why = (
                f"He's on a {streak}-series cold streak, and his recent average of "
                f"{hist_avg} kills falls short of the {line} line. "
                f"The simulation returns a {under_prob:.0f}% UNDER probability."
            )
        elif hist_avg < line * 0.94:
            core_why = (
                f"His recent average of {hist_avg} puts him {abs(avg_gap_pct):.0f}% "
                f"below the {line} line, and he's only cleared it "
                f"{hit_rate:.0f}% of the time across {n_series} series."
            )
        elif trend_pct <= -12:
            core_why = (
                f"Output has dropped {abs(trend_pct):.0f}% from his career baseline, "
                f"and the sim agrees — {under_prob:.0f}% UNDER probability."
            )
        else:
            core_why = (
                f"The simulation gives a {under_prob:.0f}% UNDER probability, "
                f"with a recent average of {hist_avg} vs the line at {line}."
            )
    else:
        core_why = (
            f"His recent average of {hist_avg} sits near the {line} line "
            f"and signals are split — the simulation shows no clear edge."
        )

    if core_why:
        sentences.append(core_why)

    # ── Sentence 3: Matchup / round context / risk ────────────────────────────
    context = ""

    if deep and not deep.get("error"):
        comb_pct  = round(((deep.get("combined_multiplier", 1.0) or 1.0) - 1) * 100, 1)
        def_prof  = (deep.get("defensive_profile") or {}).get("label", "")
        h2h_recs  = deep.get("h2h", [])
        h2h_cmpl  = [s for s in h2h_recs if not s.get("partial")]
        h2h_n     = len(h2h_cmpl)
        h2h_clrs  = sum(1 for s in h2h_cmpl if s.get("cleared"))

        opp_name_d = deep.get("opponent_display", "the opponent") or "the opponent"

        if stomp:
            rg = sim_result.get("rank_gap", "")
            context = (
                f"The rank gap against {opp_name_d} introduces a stomp risk "
                f"that could shorten maps and suppress his total{' (' + str(rg) + ' positions)' if rg else ''}."
            )
        elif comb_pct <= -7 and decision in ("OVER",):
            context = (
                f"{opp_name_d}'s defensive structure suppresses output by {abs(comb_pct):.0f}% "
                f"on average, which is the main headwind for this OVER."
                + (f" He's cleared {h2h_clrs}/{h2h_n} times H2H." if h2h_n >= 2 else "")
            )
        elif comb_pct >= 6 and decision in ("OVER",):
            context = (
                f"{opp_name_d} plays an open style that inflates kill totals, "
                f"giving a {comb_pct:.0f}% expected boost to his output."
                + (f" He's cleared {h2h_clrs}/{h2h_n} times H2H." if h2h_n >= 2 else "")
            )
        elif comb_pct <= -6 and decision in ("UNDER",):
            context = (
                f"{opp_name_d} runs a disciplined defence ({def_prof}), "
                f"suppressing kill volume by {abs(comb_pct):.0f}% — a key driver for this UNDER."
            )
        elif h2h_n >= 2:
            h2h_rate = h2h_clrs / h2h_n
            if h2h_rate >= 0.70 and decision == "OVER":
                context = f"He's cleared this line {h2h_clrs}/{h2h_n} times against {opp_name_d} historically, showing genuine H2H comfort."
            elif h2h_rate <= 0.33 and decision == "UNDER":
                context = f"He's only cleared this line {h2h_clrs}/{h2h_n} times against {opp_name_d} historically, backing the UNDER."
        elif total_rds and total_rds >= 52:
            context = f"Full, competitive maps are projected ({total_rds} rounds), giving his volume every opportunity to hit."
        elif total_rds and total_rds <= 44:
            context = f"Shorter maps are projected ({total_rds} rounds), capping the kill ceiling and increasing UNDER probability."

    elif stomp:
        context = "A significant rank mismatch introduces stomp risk — shorter maps could cap his output."
    elif total_rds and total_rds >= 52:
        context = f"Full, competitive maps are projected ({total_rds} rounds), which is ideal for volume."
    elif total_rds and total_rds <= 44:
        context = f"Shorter maps projected ({total_rds} rounds) — the kill ceiling may be lower than the line assumes."

    if context:
        sentences.append(context)

    narrative = " ".join(sentences)

    # ── Player Strengths ──────────────────────────────────────────────────────
    p_strengths: list[str] = []

    if kpr is not None:
        if kpr >= 0.85:   p_strengths.append(f"Elite KPR ({kpr:.2f})")
        elif kpr >= 0.75: p_strengths.append(f"Strong KPR ({kpr:.2f})")
    if rating is not None:
        if rating >= 1.15:   p_strengths.append(f"Top-tier rating ({rating:.2f})")
        elif rating >= 1.05: p_strengths.append(f"Above-avg rating ({rating:.2f})")
    if adr_val is not None and adr_val >= 80:
        p_strengths.append(f"High ADR ({adr_val:.0f})")
    if hs_pct is not None and hs_pct >= 52:
        p_strengths.append(f"{hs_pct:.0f}% HS rate")
    if ftype == "HOT" and streak >= 2:
        p_strengths.append(f"Currently hot — {streak} straight hits")
    elif ftype == "COLD" and streak >= 2 and decision == "UNDER":
        p_strengths.append(f"Cold run supports the UNDER")
    if vtier == "LOW":
        p_strengths.append("Consistent / low variance")
    if total_rds and total_rds >= 50 and not stomp:
        p_strengths.append(f"Full maps projected ({total_rds} rds)")
    if kpr is not None and kpr < 0.60:
        p_strengths.append("Support role — lean kill profile")

    # ── Opponent Strengths & Weaknesses ───────────────────────────────────────
    opp_strengths:   list[str] = []
    opp_weaknesses:  list[str] = []
    opp_label = ""

    if deep and not deep.get("error"):
        opp_label  = deep.get("opponent_display", "") or ""
        def_prof   = deep.get("defensive_profile") or {}
        ct_win_pct = def_prof.get("ct_win_pct")
        t_win_pct  = def_prof.get("t_win_pct")
        avg_kills_allowed = def_prof.get("avg_kills_allowed")
        rank_info  = deep.get("rank_info") or {}
        opp_rank   = rank_info.get("opp_rank")
        plr_rank   = rank_info.get("player_rank")
        map_pool   = deep.get("map_pool") or {}
        most_maps  = map_pool.get("most_played", [])
        least_maps = map_pool.get("least_played", [])
        hs_vuln    = (deep.get("hs_vulnerability") or {}).get("label", "")
        h2h_recs   = deep.get("h2h", [])
        h2h_cmpl   = [s for s in h2h_recs if not s.get("partial")]
        h2h_n      = len(h2h_cmpl)
        h2h_clrs   = sum(1 for s in h2h_cmpl if s.get("cleared"))

        # Strengths (things the opponent does well → bad for OVER / good for UNDER)
        if ct_win_pct is not None and ct_win_pct >= 53:
            opp_strengths.append(f"Strong CT side ({ct_win_pct:.0f}% win rate)")
        if t_win_pct is not None and t_win_pct >= 53:
            opp_strengths.append(f"Aggressive T-side ({t_win_pct:.0f}% win rate)")
        if opp_rank and opp_rank <= 10:
            opp_strengths.append(f"Top-{opp_rank} world ranking")
        elif opp_rank and opp_rank <= 20:
            opp_strengths.append(f"Top-20 ranked (#{opp_rank})")
        if avg_kills_allowed is not None and avg_kills_allowed < 14.0:
            opp_strengths.append("Tight defensive structure — low kills allowed")
        if least_maps:
            opp_strengths.append(f"Avoids {least_maps[0].title()} — controlled pool")
        if h2h_n >= 2 and (h2h_clrs / h2h_n) <= 0.33:
            opp_strengths.append(f"H2H dominance — player cleared only {h2h_clrs}/{h2h_n}")

        # Weaknesses (things the opponent does poorly → good for OVER / bad for UNDER)
        if ct_win_pct is not None and ct_win_pct <= 45:
            opp_weaknesses.append(f"Leaky CT side ({ct_win_pct:.0f}% win rate)")
        if t_win_pct is not None and t_win_pct <= 40:
            opp_weaknesses.append(f"Passive T-side ({t_win_pct:.0f}% win rate) — fewer duels")
        if opp_rank and plr_rank and opp_rank > plr_rank + 15:
            opp_weaknesses.append(f"Outranked (#{opp_rank} vs #{plr_rank}) — stomp risk")
        if avg_kills_allowed is not None and avg_kills_allowed >= 16.5:
            opp_weaknesses.append("Generous with kills — high avg allowed per player")
        if most_maps:
            top2 = ", ".join(m.title() for m in most_maps[:2])
            _HIGH_FRAG = {"mirage", "inferno", "dust2", "overpass", "cache", "cobblestone"}
            if any(m.lower() in _HIGH_FRAG for m in most_maps[:2]):
                opp_weaknesses.append(f"High-frag map pool ({top2}) inflates kill totals")
        if hs_vuln and "frag mine" in hs_vuln.lower():
            opp_weaknesses.append("HS-vulnerable defence")
        if h2h_n >= 2 and (h2h_clrs / h2h_n) >= 0.70:
            opp_weaknesses.append(f"Player proven vs them — {h2h_clrs}/{h2h_n} H2H clears")

    # ── Assemble final output ─────────────────────────────────────────────────
    parts = [narrative]

    if p_strengths:
        parts.append(f"**💪 Player Strengths:** {', '.join(p_strengths[:4])}")

    if opp_label and (opp_strengths or opp_weaknesses):
        opp_header = f"**vs {opp_label}**"
        if opp_strengths:
            parts.append(f"{opp_header} — **Strengths:** {', '.join(opp_strengths[:3])}")
        if opp_weaknesses:
            label = "  **Weaknesses:**" if opp_strengths else f"{opp_header} — **Weaknesses:**"
            parts.append(f"{label} {', '.join(opp_weaknesses[:3])}")

    return "\n".join(parts)


def _extract_series_totals(map_stats: list) -> list[float]:
    """Group map_stats by match_id and sum stat_value per series."""
    seen: dict[str, float] = {}
    order: list[str] = []
    for m in map_stats:
        mid = str(m.get("match_id", ""))
        if not mid:
            continue
        if mid not in seen:
            seen[mid] = 0.0
            order.append(mid)
        seen[mid] += m["stat_value"]
    return [seen[mid] for mid in order]


# ---------------------------------------------------------------------------
# Dog Line / Market Efficiency Detection
# ---------------------------------------------------------------------------

def detect_dog_line(
    sim_prob: float,
    book_implied: float,
    hist_avg: float,
    line: float,
    decision: str,
) -> dict | None:
    """
    Detect mispriced 'dog' lines — where the bookmaker underprices a player
    but our simulation shows a meaningful edge.

    A 'dog' line occurs when:
      - Book implied probability is low (player priced as unlikely to hit)
      - But sim probability is significantly higher (we disagree with the book)

    Returns a dict with type/edge/label or None if no misprice detected.
    """
    if decision not in ("OVER", "UNDER"):
        return None

    edge = sim_prob / 100 - book_implied

    # Strong mispriced dog: sim says 62%+ but book implies under 45%
    if sim_prob >= 62 and book_implied <= 0.45:
        return {
            "type":  "MISPRICED DOG",
            "edge":  round(edge * 100, 1),
            "label": (
                f"🐕 MISPRICED DOG — Book implies only {round(book_implied*100)}% "
                f"but data says {round(sim_prob)}%. Bookmaker undervalued this line."
            ),
        }
    # Moderate dog value: sim 58%+ vs book under 48%
    if sim_prob >= 58 and book_implied <= 0.48:
        return {
            "type":  "DOG VALUE",
            "edge":  round(edge * 100, 1),
            "label": (
                f"🎯 DOG VALUE — {round(sim_prob)}% sim vs "
                f"{round(book_implied*100)}% implied. Edge: +{round(edge*100,1)}%"
            ),
        }
    # Line value: player avg clearly above/below line but book didn't adjust
    if decision == "OVER" and hist_avg > line * 1.12 and sim_prob >= 60:
        return {
            "type":  "LINE VALUE",
            "edge":  round(edge * 100, 1),
            "label": (
                f"💰 LINE VALUE — Avg {hist_avg} is {round((hist_avg/line-1)*100)}% "
                f"above line {line}. Book hasn't priced the volume correctly."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Correlated Parlay Scoring
# ---------------------------------------------------------------------------

def score_correlated_parlay(grades: list[dict]) -> list[dict]:
    """
    Given a list of grade result dicts (same team, same match), score and
    rank them by combined EV and correlation strength.

    Players from the same team in the same match are positively correlated —
    if one player has a long map, all players benefit from the extra rounds.
    Returns the grades sorted by EV descending with correlation notes added.
    """
    ev_plays = []
    for g in grades:
        decision = g.get("decision", "PASS")
        if decision not in ("OVER", "UNDER"):
            continue
        ev_key = "ev_over" if decision == "OVER" else "ev_under"
        ev = g.get(ev_key) or g.get("edge", 0) or 0
        pkg = g.get("grade_pkg") or {}
        conf = pkg.get("confidence", 50)
        ev_plays.append({**g, "_ev": float(ev), "_conf": conf})

    ev_plays.sort(key=lambda x: x["_ev"], reverse=True)

    # Tag correlation notes
    for i, p in enumerate(ev_plays):
        total_rounds = p.get("total_projected_rounds")
        if total_rounds and total_rounds >= 50:
            p["_corr_note"] = "✅ Full maps projected — correlated round volume"
        elif total_rounds and total_rounds <= 44:
            p["_corr_note"] = "⚠️ Short maps risk — negative correlation for OVER legs"
        else:
            p["_corr_note"] = "➖ Neutral map pace"

    return ev_plays
