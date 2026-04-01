import numpy as np
from scipy.stats import nbinom
from scipy.special import gammaln
from statistics import median, mean
import statistics
import math
import logging

logger = logging.getLogger(__name__)

N_SIMULATIONS = 10_000


def calculate_kpr(map_stats: list) -> float:
    """Kills Per Round average across all map samples."""
    valid = [m for m in map_stats if m["rounds"] > 0]
    if not valid:
        return 0.7
    return mean(m["kills"] / m["rounds"] for m in valid)


def fit_negative_binomial(values: list):
    """
    Fit a Negative Binomial distribution to a list of kill counts.
    Returns (r, p) parameters for scipy.stats.nbinom.
    """
    if not values or len(values) < 2:
        return 5.0, 0.3

    mu = mean(values)
    var = np.var(values, ddof=1) if len(values) > 1 else mu * 1.5

    if var <= mu or mu <= 0:
        # Fallback: use Poisson-like (r large, p ≈ mu/(mu+r))
        r = max(mu * 2, 1.0)
        p = r / (r + mu)
        return r, p

    # Method of moments for NB: mu = r*(1-p)/p, var = r*(1-p)/p^2
    # => r = mu^2 / (var - mu), p = mu / var
    r = (mu ** 2) / (var - mu)
    p = mu / var
    r = max(r, 0.1)
    p = max(min(p, 0.9999), 0.0001)
    return r, p


def run_simulation(
    map_stats: list,
    line: float,
    stat_type: str = "Kills",
    favorite_prob: float = 0.55,
    likely_maps: list = None,
    rank_gap: int = None,
    period_kpr: float = None,
) -> dict:
    """
    Run 100,000 Monte Carlo simulations using Negative Binomial distribution.
    Returns a dict with all computed statistics.
    """
    stat_values = [m["stat_value"] for m in map_stats]
    rounds_per_map = [m["rounds"] for m in map_stats]

    if not stat_values:
        return {"error": "No stat data available"}

    # --- Series Totals (needed early for stability + grade penalty) ---
    total_series: dict = {}
    for m in map_stats:
        mid = m["match_id"]
        total_series.setdefault(mid, 0)
        total_series[mid] += m["stat_value"]
    series_totals = list(total_series.values())

    # --- Stability Score ---
    stability_std = statistics.stdev(series_totals) if len(series_totals) > 1 else 0.0
    ceiling_val   = max(series_totals) if series_totals else 0
    floor_val     = min(series_totals) if series_totals else 0
    if stability_std > 8:
        stability_label = "⚡ High Volatility"
    elif stability_std > 5:
        stability_label = "🌊 Moderate Volatility"
    else:
        stability_label = "🎯 Consistent"

    # --- Round Projection ---
    # Base: 21 rounds per map (42 total for 2 maps).
    # Closeness factor: matches near 50/50 odds run longer; lopsided matches run shorter.
    # Formula mirrors CS2PropGrader: base + (1 - |implied - 0.5|) * 6, capped at [18, 25].
    stomp_via_rank = False
    close_via_rank = False

    BASE_ROUNDS_PER_MAP = 21

    # Rank gap overrides odds-based projection when available (more precise)
    if rank_gap is not None:
        if rank_gap > 100:
            rounds_per_map_projected = 19
            match_context = f"Heavy Stomp (Rank gap {rank_gap}) — very short match risk"
            stomp_via_rank = True
        elif rank_gap > 50:
            rounds_per_map_projected = 20
            match_context = f"Stomp Mismatch (Rank gap {rank_gap}) — short match risk"
            stomp_via_rank = True
        elif rank_gap < 15:
            rounds_per_map_projected = 24
            match_context = f"Tight Clash (Rank gap {rank_gap}) — OT risk"
            close_via_rank = True
        else:
            # Moderate rank gap — use odds-based closeness formula
            closeness = 1.0 - abs(favorite_prob - 0.5)
            rounds_float = BASE_ROUNDS_PER_MAP + closeness * 3
            if favorite_prob >= 0.70:
                rounds_float -= 2
                stomp_via_rank = True
            rounds_per_map_projected = int(round(max(18, min(25, rounds_float))))
            match_context = f"Rank gap {rank_gap} — {round(closeness*100)}% closeness"
    else:
        # No rank data — pure odds-based closeness formula
        closeness = 1.0 - abs(favorite_prob - 0.5)
        rounds_float = BASE_ROUNDS_PER_MAP + closeness * 3
        if favorite_prob >= 0.70:
            rounds_float -= 2
            stomp_via_rank = True
            match_context = "Heavy Favorite (short match risk)"
        elif favorite_prob <= 0.55:
            match_context = "Coinflip Match (full maps likely)"
        else:
            match_context = "Moderate Favorite"
        rounds_per_map_projected = int(round(max(18, min(25, rounds_float))))

    # Stomp adjusts rounds down for kills (shorter maps = fewer kills)
    if stomp_via_rank and stat_type == "Kills":
        rounds_per_map_projected = max(18, rounds_per_map_projected - 1)

    total_projected_rounds = rounds_per_map_projected * 2
    per_map_values = stat_values

    # --- Per-map KPR ---
    kpr_values: list = []
    map_kpr: dict = {}
    for m in map_stats:
        r = m["rounds"]
        if r > 0:
            kpr = m["stat_value"] / r
            kpr_values.append(kpr)
            mn = m.get("map_name", "unknown").lower()
            if mn and mn != "unknown":
                map_kpr.setdefault(mn, []).append(kpr)

    # --- Map-Weighted Projection ---
    map_projection_note = "Overall average"
    avg_kpr = mean(kpr_values) if kpr_values else (mean(stat_values) / 22)
    overall_avg_kpr = avg_kpr  # preserve pre-map-weighted baseline for trend calc

    if likely_maps:
        weighted: list = []
        matched_maps: list = []
        for mn in likely_maps[:3]:
            key = mn.lower()
            if key in map_kpr:
                weighted.extend(map_kpr[key])
                matched_maps.append(mn.title())
        if len(weighted) >= 2:
            avg_kpr = mean(weighted)
            map_projection_note = f"Map-weighted ({', '.join(matched_maps)})"

    # --- Recency Weighting (60% recent / 40% historical) ---
    # kpr_values are ordered newest-first (scraper fetches results in reverse chron order)
    # Take the most recent 4 map samples (~2 series) as "recent form"
    n_kpr = len(kpr_values)
    recent_n = max(2, min(4, n_kpr // 3 + 1))
    recent_kpr_vals = kpr_values[:recent_n] if n_kpr >= recent_n else kpr_values
    recent_avg_kpr = mean(recent_kpr_vals)

    # Blend: 70% recent form, 30% map-weighted (or overall) avg
    blended_kpr = 0.70 * recent_avg_kpr + 0.30 * avg_kpr

    # If HLTV period stats provides an aggregate KPR (90-day), fold it in as a
    # 3rd signal. It captures a broader date window than the 10-series scrape
    # and is already normalised to all rounds (not just BO3 Maps 1&2).
    # We weight it conservatively (25%) so it calibrates without dominating.
    if period_kpr and 0.1 <= period_kpr <= 4.0:
        blended_kpr = 0.75 * blended_kpr + 0.25 * period_kpr
        logger.debug(f"[sim] period_kpr={period_kpr:.3f} blended into kpr → {blended_kpr:.3f}")

    # Trend signal (vs overall average, not map-weighted)
    trend_pct = round((recent_avg_kpr - overall_avg_kpr) / max(overall_avg_kpr, 0.01) * 100, 1)
    recent_avg_kills = round(recent_avg_kpr * 22, 1)  # per-map kills equivalent
    if trend_pct >= 12:
        trend_label = f"📈 Hot Form (+{trend_pct:.0f}% vs avg)"
    elif trend_pct <= -12:
        trend_label = f"📉 Cold Form ({trend_pct:.0f}% vs avg)"
    else:
        trend_label = f"➡️ Neutral ({trend_pct:+.0f}% vs avg)"

    expected_total = blended_kpr * total_projected_rounds

    # --- NB Fit + Simulation ---
    r_param, p_param = fit_negative_binomial(per_map_values)
    r_total = r_param * 2
    mean_nb = r_total * (1 - p_param) / p_param
    target_mean = expected_total
    r_total_adj = r_total
    p_adj = r_total_adj / (r_total_adj + target_mean) if target_mean > 0 else 0.5
    p_adj = max(min(p_adj, 0.9999), 0.0001)

    np.random.seed(42)
    samples = nbinom.rvs(r_total_adj, p_adj, size=N_SIMULATIONS)

    sim_mean   = float(np.mean(samples))
    sim_std    = float(np.std(samples))
    sim_median = float(np.median(samples))

    # Over / Under / Push
    over_prob  = float(np.mean(samples > line))
    under_prob = float(np.mean(samples < line))
    push_prob  = float(np.mean(samples == int(line)))
    # Percentile: what % of simulated totals fall AT OR BELOW the line
    # < 50 → model says lean OVER (line is below median projection)
    line_percentile = round(float(np.mean(samples <= line)) * 100, 1)

    # --- Fair Line & Misprice ---
    fair_line  = round(sim_median, 1)
    line_gap   = fair_line - line
    abs_gap    = abs(line_gap)
    direction  = "+" if line_gap > 0 else ""
    if abs_gap > 4:
        misprice_label = f"🚨 MASSIVE MISPRICE ({direction}{line_gap:.1f})"
    elif abs_gap > 2:
        misprice_label = f"⚠️ Mispriced ({direction}{line_gap:.1f})"
    else:
        misprice_label = "✅ Fair Line"

    # --- Expected Value (EV) vs standard -110 odds ---
    # EV = (over_prob * (100/110)) - (under_prob * 1.0)
    # Positive EV = value on OVER side at -110
    ev_over  = round(over_prob * (100 / 110) - under_prob, 4)
    ev_under = round(under_prob * (100 / 110) - over_prob, 4)

    # --- Percentile ceiling / floor from simulation (p90/p10) ---
    sim_p10 = float(np.percentile(samples, 10))
    sim_p90 = float(np.percentile(samples, 90))

    # --- Historical stats ---
    hit_rate   = (
        sum(1 for v in series_totals if v > line) / len(series_totals)
        if series_totals else over_prob
    )
    hist_avg    = mean(series_totals) if series_totals else mean(stat_values) * 2
    hist_median = median(series_totals) if series_totals else median(stat_values) * 2

    # --- Grading ---
    edge = over_prob - 0.5

    grade, recommendation, decision = calculate_grade(
        edge=edge,
        over_prob=over_prob,
        hist_avg=hist_avg,
        hist_median=hist_median,
        line=line,
        hit_rate=hit_rate,
        favorite_prob=favorite_prob,
        stat_type=stat_type,
        stability_std=stability_std,
        trend_pct=trend_pct,
        stomp_via_rank=stomp_via_rank,
    )

    return {
        "stat_type":              stat_type,
        "n_samples":              len(map_stats),
        "n_series":               len(series_totals),
        "hist_avg":               round(hist_avg, 2),
        "hist_median":            round(hist_median, 2),
        "hit_rate":               round(hit_rate * 100, 1),
        "rounds_per_map":         rounds_per_map_projected,
        "total_projected_rounds": total_projected_rounds,
        "expected_total":         round(expected_total, 2),
        "sim_mean":               round(sim_mean, 2),
        "sim_std":                round(sim_std, 2),
        "sim_median":             round(sim_median, 2),
        "over_prob":              round(over_prob * 100, 1),
        "under_prob":             round(under_prob * 100, 1),
        "push_prob":              round(push_prob * 100, 1),
        "edge":                   round(edge * 100, 1),
        "grade":                  grade,
        "recommendation":         recommendation,
        "decision":               decision,
        "match_context":          match_context,
        "r_param":                round(r_total_adj, 3),
        "p_param":                round(p_adj, 4),
        "n_simulations":          N_SIMULATIONS,
        # --- New fields ---
        "fair_line":              fair_line,
        "misprice_label":         misprice_label,
        "stability_std":          round(stability_std, 2),
        "stability_label":        stability_label,
        "ceiling":                ceiling_val,
        "floor":                  floor_val,
        "map_projection_note":    map_projection_note,
        "map_kpr":                {k: round(mean(v), 3) for k, v in map_kpr.items()},
        "stomp_via_rank":         stomp_via_rank,
        "close_via_rank":         close_via_rank,
        "rank_gap":               rank_gap,
        # --- Recency / trend ---
        "trend_pct":              trend_pct,
        "trend_label":            trend_label,
        "recent_avg_kills":       recent_avg_kills,
        "recent_n_maps":          recent_n,
        # --- Line context ---
        "line_percentile":        line_percentile,
        # --- EV at standard -110 odds ---
        "ev_over":                round(ev_over, 4),
        "ev_under":               round(ev_under, 4),
        # --- Simulation percentile range (p10/p90 = floor/ceiling) ---
        "sim_p10":                round(sim_p10, 1),
        "sim_p90":                round(sim_p90, 1),
    }


def calculate_grade(
    edge: float,
    over_prob: float,
    hist_avg: float,
    hist_median: float,
    line: float,
    hit_rate: float,
    favorite_prob: float,
    stat_type: str,
    stability_std: float = 0.0,
    trend_pct: float = 0.0,
    stomp_via_rank: bool = False,
) -> tuple:
    """
    Apply grading scale and decision logic.
    Returns (grade_str, recommendation_str, decision_str).
    """
    # --- Decision Logic ---
    # stomp_trap fires either from favorite_prob OR from a rank-based stomp
    # projection (stomp_via_rank). When it's active the primary OVER/UNDER
    # paths based on historical data are suppressed — the round-count model
    # may have artificially moved over_prob away from 50 without historical
    # backing, so PASS is the safe call.
    stomp_trap = (favorite_prob >= 0.72 or stomp_via_rank) and stat_type == "Kills"
    avg_above = hist_avg > line
    median_above = hist_median > line
    hot_form  = trend_pct >= 12   # recent 4 maps running 12%+ above average
    cold_form = trend_pct <= -12  # recent 4 maps running 12%+ below average
    # Sportsbooks shade lines slightly low — lower OVER bar, tighten UNDER bar
    strong_hit_rate = hit_rate >= 0.55   # was 0.60
    weak_hit_rate   = hit_rate < 0.35    # was 0.40 — need stronger evidence for UNDER

    if avg_above and median_above and strong_hit_rate and not stomp_trap:
        decision = "OVER"
    elif not avg_above and not median_above and weak_hit_rate and not hot_form and not stomp_trap:
        # Don't call UNDER if player is trending hot or if stomp projection
        # may have artificially depressed over_prob
        decision = "UNDER"
    elif stomp_trap and avg_above:
        decision = "PASS"
    elif stomp_trap and not avg_above and not median_above and weak_hit_rate and not hot_form:
        # Historical data agrees with UNDER even accounting for stomp context
        decision = "UNDER"
    else:
        # Edge-based decision — lower threshold when trend confirms direction
        over_threshold  = 0.05 if hot_form  else 0.08
        under_threshold = 0.05 if cold_form else 0.08
        if edge > over_threshold:
            decision = "OVER"
        elif edge < -under_threshold:
            # Guard: if the stomp projection drove the edge negative but
            # historical data (hit_rate, hist_avg) still leans OVER, call
            # PASS rather than UNDER — the model and history contradict.
            if stomp_via_rank and avg_above and median_above:
                decision = "PASS"
            else:
                decision = "UNDER"
        elif hot_form and edge >= 0:
            # Player on a hot streak and edge is non-negative → lean OVER
            decision = "OVER"
        elif cold_form and edge <= 0:
            # Player on a cold streak and edge is non-positive → lean UNDER
            decision = "UNDER"
        elif abs(line - hist_avg) / max(hist_avg, 1) > 0.12:
            decision = "MISPRICED"
        else:
            decision = "PASS"

    # --- Misprice check (from CS2PropGrader avg/median gap logic) ---
    avg_gap    = hist_avg - line       # positive = avg above line
    median_gap = hist_median - line    # positive = median above line
    # "Prop Error" if avg is >6 above line; "Mispriced" if avg±median both >3 off
    prop_error  = avg_gap > 6 and median_gap > 2
    mispriced   = (abs(avg_gap) > 3 and abs(median_gap) > 2) or (abs(line - hist_avg) / max(hist_avg, 1) > 0.15)

    # --- Grade Scale (1-10) with rich labels ---
    edge_abs = abs(edge)

    if edge_abs >= 0.22:
        grade_num = 10
        grade_label = "Elite edge"
    elif edge_abs >= 0.18:
        grade_num = 9
        grade_label = "Very strong edge"
    elif edge_abs >= 0.14:
        grade_num = 8
        grade_label = "Strong playable edge"
    elif edge_abs >= 0.10:
        grade_num = 7
        grade_label = "Solid lean"
    elif edge_abs >= 0.07:
        grade_num = 6
        grade_label = "Small edge"
    elif edge_abs >= 0.05:
        grade_num = 5
        grade_label = "Borderline"
    elif edge_abs >= 0.03:
        grade_num = 4
        grade_label = "Marginal"
    elif edge_abs >= 0.02:
        grade_num = 3
        grade_label = "Very thin"
    elif edge_abs >= 0.01:
        grade_num = 2
        grade_label = "Noise level"
    else:
        grade_num = 1
        grade_label = "No edge"

    # Boost grade when line is clearly mispriced
    if mispriced and grade_num >= 6:
        grade_num = min(grade_num + 1, 10)
    # Extra boost for prop error (wildly off line)
    if prop_error and grade_num >= 7:
        grade_num = min(grade_num + 1, 10)
        grade_label = "Prop Error (Wildly off)"

    # Stability penalty: High Volatility + thin edge → drop grade by 1
    if stability_std > 8 and edge_abs < 0.10:
        grade_num = max(1, grade_num - 1)

    grade_str = f"{grade_num}/10 ({grade_label})"

    # Recommendation
    if grade_num >= 8:
        if decision == "OVER":
            rec = "STRONG BET — OVER"
        elif decision == "UNDER":
            rec = "STRONG BET — UNDER"
        else:
            rec = "PASS (conflicting signals)"
    elif grade_num >= 6:
        if decision in ("OVER", "UNDER"):
            rec = f"LEAN {decision} (value play)"
        else:
            rec = "SKIP — low edge"
    elif grade_num >= 4:
        rec = "MARGINAL — proceed with caution"
    else:
        rec = "PASS — insufficient edge"

    if prop_error:
        rec = "PROP ERROR — line wildly off, investigate before betting"
    elif mispriced and decision == "MISPRICED":
        rec = "LINE APPEARS MISPRICED — investigate before betting"

    return grade_str, rec, decision
