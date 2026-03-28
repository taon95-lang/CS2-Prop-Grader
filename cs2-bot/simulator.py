import numpy as np
from scipy.stats import nbinom
from scipy.special import gammaln
from statistics import median, mean
import math
import logging

logger = logging.getLogger(__name__)

N_SIMULATIONS = 100_000


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
) -> dict:
    """
    Run 100,000 Monte Carlo simulations using Negative Binomial distribution.

    Returns a dict with all computed statistics.
    """
    stat_values = [m["stat_value"] for m in map_stats]
    rounds_per_map = [m["rounds"] for m in map_stats]

    if not stat_values:
        return {"error": "No stat data available"}

    # --- Round Projection ---
    if favorite_prob >= 0.70:
        # Heavy favorite: shorter match expected
        rounds_per_map_projected = 19
        match_context = "Heavy Favorite (short match risk)"
    elif favorite_prob <= 0.55:
        # Coinflip: longer match
        rounds_per_map_projected = 23
        match_context = "Coinflip Match (full maps likely)"
    else:
        rounds_per_map_projected = 22
        match_context = "Moderate Favorite"

    total_projected_rounds = rounds_per_map_projected * 2

    # --- Per-map stat values for 2 maps ---
    per_map_values = stat_values

    # Scale stat values to projected rounds
    kpr_values = []
    for i, m in enumerate(map_stats):
        r = m["rounds"]
        if r > 0:
            kpr_values.append(m["stat_value"] / r)

    avg_kpr = mean(kpr_values) if kpr_values else (mean(stat_values) / 22)

    # Expected total stats for 2 maps at projected rounds
    expected_total = avg_kpr * total_projected_rounds

    # Fit NB to per-map distribution
    r_param, p_param = fit_negative_binomial(per_map_values)

    # Scale NB params to total 2-map projection
    # For sum of 2 NB(r, p), result is NB(2r, p)
    r_total = r_param * 2
    mean_nb = r_total * (1 - p_param) / p_param

    # Adjust mean to match expected total
    scale_factor = expected_total / mean_nb if mean_nb > 0 else 1.0
    r_total_adj = r_total
    # Adjust p to match target mean: p = r / (r + mean)
    target_mean = expected_total
    p_adj = r_total_adj / (r_total_adj + target_mean)
    p_adj = max(min(p_adj, 0.9999), 0.0001)

    # --- Monte Carlo Simulation ---
    np.random.seed(42)
    samples = nbinom.rvs(r_total_adj, p_adj, size=N_SIMULATIONS)

    sim_mean = float(np.mean(samples))
    sim_std = float(np.std(samples))
    sim_median = float(np.median(samples))

    # Over/Under probabilities
    over_prob = float(np.mean(samples > line))
    under_prob = float(np.mean(samples < line))
    push_prob = float(np.mean(samples == int(line)))

    # Hit rate against line from historical data
    total_series = {}
    for m in map_stats:
        mid = m["match_id"]
        if mid not in total_series:
            total_series[mid] = 0
        total_series[mid] += m["stat_value"]

    series_totals = list(total_series.values())
    hit_rate = (
        sum(1 for v in series_totals if v > line) / len(series_totals)
        if series_totals
        else over_prob
    )

    # Historical average and median (per-series totals)
    hist_avg = mean(series_totals) if series_totals else mean(stat_values) * 2
    hist_median = median(series_totals) if series_totals else median(stat_values) * 2

    # --- Grading ---
    implied_line_prob = 0.5  # Default: line is set at 50%

    edge = over_prob - implied_line_prob

    grade, recommendation, decision = calculate_grade(
        edge=edge,
        over_prob=over_prob,
        hist_avg=hist_avg,
        hist_median=hist_median,
        line=line,
        hit_rate=hit_rate,
        favorite_prob=favorite_prob,
        stat_type=stat_type,
    )

    return {
        "stat_type": stat_type,
        "n_samples": len(map_stats),
        "n_series": len(series_totals),
        "hist_avg": round(hist_avg, 2),
        "hist_median": round(hist_median, 2),
        "hit_rate": round(hit_rate * 100, 1),
        "rounds_per_map": rounds_per_map_projected,
        "total_projected_rounds": total_projected_rounds,
        "expected_total": round(expected_total, 2),
        "sim_mean": round(sim_mean, 2),
        "sim_std": round(sim_std, 2),
        "sim_median": round(sim_median, 2),
        "over_prob": round(over_prob * 100, 1),
        "under_prob": round(under_prob * 100, 1),
        "push_prob": round(push_prob * 100, 1),
        "edge": round(edge * 100, 1),
        "grade": grade,
        "recommendation": recommendation,
        "decision": decision,
        "match_context": match_context,
        "r_param": round(r_total_adj, 3),
        "p_param": round(p_adj, 4),
        "n_simulations": N_SIMULATIONS,
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
) -> tuple:
    """
    Apply grading scale and decision logic.
    Returns (grade_str, recommendation_str, decision_str).
    """
    # --- Decision Logic ---
    stomp_trap = favorite_prob >= 0.72 and stat_type == "Kills"
    avg_above = hist_avg > line
    median_above = hist_median > line
    strong_hit_rate = hit_rate >= 0.60
    weak_hit_rate = hit_rate < 0.40

    if avg_above and median_above and strong_hit_rate and not stomp_trap:
        decision = "OVER"
    elif not avg_above and not median_above and weak_hit_rate:
        decision = "UNDER"
    elif stomp_trap and avg_above:
        decision = "PASS"
    else:
        # Edge-based decision
        if edge > 0.08:
            decision = "OVER"
        elif edge < -0.08:
            decision = "UNDER"
        elif abs(line - hist_avg) / max(hist_avg, 1) > 0.12:
            decision = "MISPRICED"
        else:
            decision = "PASS"

    # Mispriced check
    mispriced = abs(line - hist_avg) / max(hist_avg, 1) > 0.15

    # --- Grade Scale (1-10) ---
    edge_abs = abs(edge)

    if edge_abs >= 0.22:
        grade_num = 10
    elif edge_abs >= 0.18:
        grade_num = 9
    elif edge_abs >= 0.14:
        grade_num = 8
    elif edge_abs >= 0.10:
        grade_num = 7
    elif edge_abs >= 0.07:
        grade_num = 6
    elif edge_abs >= 0.05:
        grade_num = 5
    elif edge_abs >= 0.03:
        grade_num = 4
    elif edge_abs >= 0.02:
        grade_num = 3
    elif edge_abs >= 0.01:
        grade_num = 2
    else:
        grade_num = 1

    # Adjust for mispriced
    if mispriced and grade_num >= 6:
        grade_num = min(grade_num + 1, 10)

    grade_str = f"{grade_num}/10"

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

    if mispriced and decision == "MISPRICED":
        rec = "LINE APPEARS MISPRICED — investigate before betting"

    return grade_str, rec, decision
