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

    # ── Hit rate ─────────────────────────────────────────────────────────────
    hit_rate = sim_result.get("hit_rate", 50) or 50
    if hit_rate >= 75:      score += 12
    elif hit_rate >= 65:    score += 8
    elif hit_rate >= 57:    score += 4
    elif hit_rate <= 30:    score -= 12
    elif hit_rate <= 40:    score -= 8
    elif hit_rate <= 48:    score -= 4

    # ── Variance ─────────────────────────────────────────────────────────────
    vtier = variance.get("tier", "MEDIUM")
    if vtier == "LOW":          score += 10
    elif vtier == "MEDIUM":     score += 0
    elif vtier == "HIGH":       score -= 7
    elif vtier == "VERY_HIGH":  score -= 12

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
                arrow = "↑" if avg > line else ("↓" if avg < line else "→")
                projected_labels.append(f"{lm.title()} `{avg}` {arrow}")

    projected_avg     = round(mean(projected_vals), 1) if projected_vals else None
    projected_vs_line = None
    if projected_avg is not None and line:
        pct  = round((projected_avg - line) / max(line, 1) * 100, 1)
        sign = "+" if pct >= 0 else ""
        projected_vs_line = f"{sign}{pct}% vs line"

    return {
        "per_map":          map_avgs,
        "sorted_maps":      sorted_maps,
        "best_map":         best_map,
        "worst_map":        worst_map,
        "projected_avg":    projected_avg,
        "projected_labels": projected_labels,
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

    # Very low hit rate
    hr = sim_result.get("hit_rate", 50) or 50
    if hr < 35:
        flags.append(f"⚠️ Low historical hit rate — only {hr}% cleared this line")

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
