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


def _opp_quality_weights(map_stats: list, today_opp_rank: int | None) -> list[float]:
    """
    Per-map weights based on how closely the historical opponent's rank
    matches today's opponent's rank.

    Tighter rank proximity → higher weight (those matches are more predictive).
    Maps with no opp_rank data get a neutral weight of 1.0.

    If today_opp_rank is None (rank unknown), all maps get weight 1.0 (no change).
    """
    if today_opp_rank is None:
        return [1.0] * len(map_stats)

    weights = []
    any_ranked = False
    for m in map_stats:
        hist_rank = m.get('opp_rank')
        if hist_rank is None:
            weights.append(1.0)
            continue
        any_ranked = True
        diff = abs(today_opp_rank - hist_rank)
        if diff <= 10:
            w = 4.0    # nearly identical quality tier
        elif diff <= 25:
            w = 2.5    # similar quality
        elif diff <= 50:
            w = 1.0    # baseline — roughly same competitive level
        elif diff <= 80:
            w = 0.4    # noticeably different tier
        else:
            w = 0.15   # wildly different — discount heavily
        weights.append(w)

    # If no maps had opp_rank data, weighting is meaningless — return flat
    if not any_ranked:
        return [1.0] * len(map_stats)
    return weights


def _weighted_mean_var(values: list, weights: list) -> tuple[float, float]:
    """Compute reliability-weighted mean and variance for NB distribution fitting."""
    total_w = sum(weights)
    if total_w <= 0:
        return mean(values), statistics.variance(values) if len(values) > 1 else mean(values) * 1.5
    w_mean = sum(w * v for w, v in zip(weights, values)) / total_w
    # Reliability-weighted variance (Bessel correction via effective sample size)
    eff_n = total_w ** 2 / sum(w ** 2 for w in weights)
    correction = eff_n / max(eff_n - 1, 1)
    w_var = sum(w * (v - w_mean) ** 2 for w, v in zip(weights, values)) / total_w * correction
    return w_mean, max(w_var, w_mean * 0.5)  # floor variance to avoid degenerate NB


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
    today_opp_rank: int | None = None,
) -> dict:
    """
    Run 100,000 Monte Carlo simulations using Negative Binomial distribution.
    Returns a dict with all computed statistics.

    today_opp_rank: HLTV world rank of today's opponent.  When provided, maps
    played against similarly-ranked opponents are weighted more heavily in the
    distribution fit.  Maps vs much stronger/weaker teams are down-weighted so
    they don't distort the projection for today's specific matchup.
    """
    stat_values = [m["stat_value"] for m in map_stats]
    rounds_per_map = [m["rounds"] for m in map_stats]

    # --- Opponent Quality Weighting ---
    # Compute per-map reliability weights based on rank proximity to today's opp.
    _opp_weights = _opp_quality_weights(map_stats, today_opp_rank)
    _any_weighted = today_opp_rank is not None and any(w != 1.0 for w in _opp_weights)
    if _any_weighted:
        _covered = sum(1 for m in map_stats if m.get('opp_rank') is not None)
        logger.info(
            f"[sim] Opp quality weighting active — today opp #{today_opp_rank}, "
            f"{_covered}/{len(map_stats)} maps have historical opp_rank. "
            f"Weight range: {min(_opp_weights):.2f}–{max(_opp_weights):.2f}"
        )

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
    kpr_weights: list = []   # parallel weight list aligned to kpr_values
    map_kpr: dict = {}
    for m, _w in zip(map_stats, _opp_weights):
        r = m["rounds"]
        if r > 0:
            kpr = m["stat_value"] / r
            kpr_values.append(kpr)
            kpr_weights.append(_w)
            mn = m.get("map_name", "unknown").lower()
            if mn and mn != "unknown":
                map_kpr.setdefault(mn, []).append(kpr)

    # --- Map-Weighted Projection ---
    map_projection_note = "Overall average"
    if kpr_values:
        if _any_weighted and any(w != 1.0 for w in kpr_weights):
            # Opponent-quality weighted KPR average
            _kpr_total_w = sum(kpr_weights)
            avg_kpr = sum(k * w for k, w in zip(kpr_values, kpr_weights)) / _kpr_total_w
        else:
            avg_kpr = mean(kpr_values)
    else:
        avg_kpr = mean(stat_values) / 22
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

    # Blend: 60% recent form, 40% map-weighted (or overall) avg
    # (was 70/30 — reduced because 2-4 map samples were over-weighting hot streaks)
    blended_kpr = 0.60 * recent_avg_kpr + 0.40 * avg_kpr

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
    # If opponent quality weighting is active, use weighted mean/variance
    # to fit the NB distribution so similar-quality historical maps dominate.
    if _any_weighted:
        _w_mean, _w_var = _weighted_mean_var(per_map_values, _opp_weights)
        # Fit NB from weighted moments (method of moments: mu=r(1-p)/p, var=r(1-p)/p^2)
        if _w_var > _w_mean > 0:
            _r_w = (_w_mean ** 2) / (_w_var - _w_mean)
            _p_w = _w_mean / _w_var
            r_param = max(_r_w, 0.1)
            p_param = max(min(_p_w, 0.9999), 0.0001)
        else:
            r_param, p_param = fit_negative_binomial(per_map_values)
        logger.info(
            f"[sim] Weighted NB fit — w_mean={_w_mean:.2f} w_var={_w_var:.2f} "
            f"→ r={r_param:.2f} p={p_param:.4f}"
        )
    else:
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
        # --- Opponent quality weighting metadata ---
        "opp_quality_weighted":   _any_weighted,
        "today_opp_rank":         today_opp_rank,
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
    Evidence-stacking grade engine — v2.

    Four independent signals each vote OVER (+) or UNDER (-) with weights.
    The weighted consensus score drives the decision.  No single signal
    can override everything else.  The opponent-quality multiplier is no
    longer applied to the kill distribution — it is a display signal only.

    Signals and weights
    ───────────────────
    S1  Historical hit rate    weight 3   Has the player actually cleared this line?
    S2  Median vs line gap     weight 2   Is the median structurally above/below?
    S3  Recent form / trend    weight 2   Is the player running hot or cold right now?
    S4  Simulation probability weight 1   Does the raw kill distribution confirm?

    Max raw score ≈ 19  (all four signals at maximum in one direction)
    Typical strong edge: 11–14

    Stomp modifier: if stomp risk detected → total × 0.70
      Preserves dominant edges (≥ 10 pre-stomp → ≥ 7 post-stomp = action).
      Kills borderline leans that only squeak past threshold.

    Action threshold: |total| ≥ 7 → OVER / UNDER,  else PASS.

    Returns (grade_str, recommendation_str, decision_str).
    """

    # ── S1: Historical hit rate (weight 3) ───────────────────────────────────
    # Most reliable signal — direct measurement: has this player cleared this
    # exact line in past BO3 series?
    hr_pct = hit_rate * 100
    if   hr_pct >= 70:   s1 = 3
    elif hr_pct >= 62:   s1 = 2
    elif hr_pct >= 55:   s1 = 1
    elif hr_pct >= 45:   s1 = 0
    elif hr_pct >= 38:   s1 = -1
    elif hr_pct >= 30:   s1 = -2
    else:                s1 = -3

    # ── S2: Median vs line gap (weight 2) ────────────────────────────────────
    # Uses raw/unscaled hist_median — the number the player actually produced,
    # not an opponent-adjusted projection.
    gap_pct = (hist_median - line) / max(line, 1)
    if   gap_pct >= 0.12:    s2 = 2
    elif gap_pct >= 0.04:    s2 = 1
    elif gap_pct >= -0.04:   s2 = 0
    elif gap_pct >= -0.12:   s2 = -1
    else:                    s2 = -2

    # ── S3: Recent trend / form (weight 2) ───────────────────────────────────
    # trend_pct = (recent 4-map avg – career avg) / career avg × 100
    if   trend_pct >= 15:   s3 = 2
    elif trend_pct >=  5:   s3 = 1
    elif trend_pct >= -5:   s3 = 0
    elif trend_pct >= -15:  s3 = -1
    else:                   s3 = -2

    # ── S4: Raw simulation probability (weight 1) ────────────────────────────
    # Runs on un-inflated historical data — minor confirmation signal.
    if   over_prob >= 0.68:  s4 = 2
    elif over_prob >= 0.60:  s4 = 1
    elif over_prob >= 0.40:  s4 = 0
    elif over_prob >= 0.32:  s4 = -1
    else:                    s4 = -2

    # ── Weighted consensus ───────────────────────────────────────────────────
    raw_total = (s1 * 3) + (s2 * 2) + (s3 * 2) + (s4 * 1)

    # ── Stomp modifier ───────────────────────────────────────────────────────
    # Stomps shorten maps ~10% (≈21→19 rounds).  A 30% score reduction keeps
    # dominant historical edges alive while cancelling borderline leans.
    stomp = (stomp_via_rank or favorite_prob >= 0.72) and stat_type == "Kills"
    if stomp:
        total = int(raw_total * 0.70)
    else:
        total = raw_total

    # ── Decision ─────────────────────────────────────────────────────────────
    if total >= 7:
        decision = "OVER"
    elif total <= -7:
        decision = "UNDER"
    else:
        decision = "PASS"

    # ── Grade scale (based on raw signal strength, not stomp-adjusted) ───────
    abs_r = abs(raw_total)
    if decision == "PASS":
        grade_str = "N/A"
        rec = "⏸️ PASS — Signals too mixed for a confident call"
    else:
        if   abs_r >= 15:   grade_num, edge_label = 10, "Elite edge"
        elif abs_r >= 12:   grade_num, edge_label = 9,  "Elite edge"
        elif abs_r >= 10:   grade_num, edge_label = 8,  "Strong edge"
        elif abs_r >= 8:    grade_num, edge_label = 7,  "Solid lean"
        else:               grade_num, edge_label = 6,  "Lean"

        grade_str = f"{grade_num}/10 ({edge_label})"
        sign = "✅" if decision == "OVER" else "❌"
        rec  = f"{sign} {decision} — {grade_str}"

    return grade_str, rec, decision
