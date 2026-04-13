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

    return {
        "per_map":           map_avgs,
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
    Attach as sim_result['grade_pkg'] in bot.py for embed access.
    """
    line      = sim_result.get("line", 0) or 0
    decision  = sim_result.get("decision", "PASS")
    over_prob = sim_result.get("over_prob", 50.0) or 50.0

    # --- Core computations --------------------------------------------------
    series_totals = _extract_series_totals(map_stats)
    form          = compute_form_streak(map_stats, line)
    variance      = compute_variance_tier(series_totals)
    likely_maps   = None
    if deep and not deep.get("error"):
        mp = deep.get("map_pool", {})
        likely_maps = mp.get("most_played", [])

    map_intel = compute_map_intel(map_stats, likely_maps, line)
    flags     = compute_risk_flags(sim_result, variance, form, deep, line)
    confidence = compute_confidence_score(
        sim_result=sim_result,
        map_stats=map_stats,
        form=form,
        variance=variance,
        deep=deep,
        period_stats=period_stats,
        decision=decision,
    )
    edge_pct = compute_edge_pct(over_prob, decision)
    reason   = build_verdict_reason(decision, form, variance, deep, sim_result, flags)

    return {
        "form":       form,
        "variance":   variance,
        "map_intel":  map_intel,
        "flags":      flags,
        "confidence": confidence,
        "edge_pct":   edge_pct,
        "reason":     reason,
        "series_totals": series_totals,
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
