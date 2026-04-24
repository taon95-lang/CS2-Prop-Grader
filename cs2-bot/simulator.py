import numpy as np
# NB / scipy.stats removed — empirical model only (per user request)
from statistics import median, mean
import statistics
import math
import logging

logger = logging.getLogger(__name__)

# N_SIMULATIONS removed — empirical model has no Monte Carlo runs.
# Field is kept in the output dict as 0 for backward-compat with grades_db.
N_SIMULATIONS = 0

# Standard sportsbook vig (-110 both sides = 52.38% implied probability)
BOOK_IMPLIED_PROB = 0.5238

# ── Bayesian regression-to-the-mean for projections ─────────────────────────
# Diagnosis (Apr 2026 audit, n=139 settled grades):
#   • sim_median had +3.95 mean error (bot consistently OVER-projected)
#   • Bot's projection MAE (7.52) was WORSE than the bookmaker's line MAE (6.34)
#   • Top-tier "Elite Edge" calls hit 54%, vs "Small Edge" hitting 83% — i.e.
#     the bot's confidence was inversely correlated with hit rate.
#
# Root cause: with n≥8 series, the existing book-shrink does nothing, leaving
# raw recent-form to drive over_prob with zero regression to the mean.
#
# Fix: blend the projection toward a population mean using prior strength K.
# At n=10 series, K=12 → 45% player / 55% population weight.
# Backtested on the same 139 settled grades: lifted hit rate from 60% → 66%
# and units from +10 to +18 (Policy C, k=12).
POP_MEAN_KILLS_BO3 = 27.65   # empirical mean of actual BO3 maps 1+2 kill totals
POP_MEAN_HS_BO3    = 11.0    # placeholder — refine when HS settles land
PROJ_SHRINK_K      = 12      # prior strength (higher = more shrinkage)


def _trimmed_mean(values: list, pct: float = 0.10) -> float:
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


def _mad(values: list) -> float:
    """Median Absolute Deviation — robust σ alternative (scaled to normal-σ)."""
    if not values:
        return 0.0
    med = median(values)
    abs_dev = [abs(v - med) for v in values]
    return median(abs_dev) * 1.4826


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


def _lan_context_weights(map_stats: list, today_is_lan: bool | None) -> list[float]:
    """
    Per-map weights based on whether the historical match was played in the
    same context (LAN vs Online) as today's match.

    Same-context maps are more predictive because peeker's advantage,
    crowd pressure, ping variance, and equipment reliability all differ.

    - Today is LAN, history is LAN  → weight 1.5 (boost LAN form)
    - Today is LAN, history is Online → weight 0.6 (online form is less reliable on LAN)
    - Today is Online, history is Online → weight 1.5
    - Today is Online, history is LAN  → weight 0.7 (LAN form often inflates online projection)
    - Unknown context (either side) → weight 1.0 (no change)

    If today_is_lan is None, all maps get weight 1.0 (no weighting applied).
    If no historical maps have is_lan set, returns flat weights (no signal).
    """
    if today_is_lan is None:
        return [1.0] * len(map_stats)

    weights = []
    any_known = False
    for m in map_stats:
        hist_lan = m.get('is_lan')
        if hist_lan is None:
            weights.append(1.0)
            continue
        any_known = True
        if today_is_lan and hist_lan:
            w = 1.5
        elif today_is_lan and not hist_lan:
            w = 0.6
        elif (not today_is_lan) and (not hist_lan):
            w = 1.5
        else:  # today online, history LAN
            w = 0.7
        weights.append(w)

    if not any_known:
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


# fit_negative_binomial removed (NB model deprecated — empirical only)


def compute_kill_quality_multiplier(
    period_kpr: float | None,
    period_rating: float | None,
    period_adr: float | None,
) -> tuple[float, str, dict]:
    """
    April 2026 overhaul — Doc 1 #1: Quality-of-Kill (eco-adjustment) multiplier.

    Detects stat-padding against eco rounds WITHOUT new HLTV scraping by
    cross-checking three already-collected period metrics:

      1. Rating-vs-KPR divergence:
         HLTV's Rating 2.1 is a regression-based all-in-one impact score
         (multikills, opening kills, clutches, traded deaths, KAST, ADR).
         Eco-padded kills add to KPR but barely move Rating because they
         lack opening-duel weight, multikill weight, and impact weight.
         Empirical baseline (Apr-2026 calibration on top-30 HLTV rifling
         cohort): expected_rating ≈ 0.44 + 0.86 × KPR. Players above this
         line are converting kills into impact; those below are padding.

      2. Damage-per-kill (ADR / KPR / rounds_per_map proxy):
         Pure eco kills register ~85-100 dmg before the kill (low-HP
         opponents, no armor). Quality kills against full buys land in
         the 115-135 dmg/kill band. We use ADR ÷ KPR as a sample-stable
         proxy for dmg-per-kill. <100 = soft, >120 = hard.

      3. Both signals are blended; the combined quality_score maps to a
         multiplier in [0.93, 1.05]. We cap downside at -7% and upside
         at +5% so a single noisy 90-day window can never wipe out or
         double a projection.

    Returns (multiplier, label, details_dict).
    """
    if not period_kpr or not period_rating or period_kpr <= 0:
        return 1.0, "➖ Neutral (insufficient data)", {}

    # Signal 1 — Rating vs expected from KPR
    expected_rating = 0.44 + 0.86 * period_kpr
    rating_delta = period_rating - expected_rating  # +0.05 elite, -0.05 padded

    # Signal 2 — ADR per kill (only if ADR present and KPR > 0)
    adr_factor = 0.0
    dmg_per_kill = None
    if period_adr and period_adr > 0:
        # ADR per kill = (ADR × 1 round) / (KPR × 1 round) = ADR / KPR
        dmg_per_kill = period_adr / period_kpr
        # Center at 110 dmg/kill (typical), 1 unit ≈ 1 dmg/kill above center
        # Each 10 dmg/kill shift = 0.05 quality score
        adr_factor = (dmg_per_kill - 110.0) / 200.0   # ±0.10 typical range

    # Composite: rating_delta is the dominant signal (0.7 weight)
    quality_score = 0.7 * rating_delta + 0.3 * adr_factor

    # Map to multiplier — asymmetric clamp (-7% downside, +5% upside)
    raw_mult = 1.0 + quality_score * 0.6   # 0.10 score → ~6% shift
    multiplier = max(0.93, min(1.05, raw_mult))

    if multiplier >= 1.025:
        label = f"🎯 HIGH (impactful kills, {multiplier:.3f}×)"
    elif multiplier <= 0.975:
        label = f"⚠️ LOW (eco-padded, {multiplier:.3f}×)"
    else:
        label = f"➖ Neutral ({multiplier:.3f}×)"

    details = {
        "rating": round(period_rating, 3),
        "expected_rating": round(expected_rating, 3),
        "rating_delta": round(rating_delta, 3),
        "dmg_per_kill": round(dmg_per_kill, 1) if dmg_per_kill else None,
        "quality_score": round(quality_score, 3),
        "multiplier": round(multiplier, 3),
    }
    return multiplier, label, details


def run_simulation(
    map_stats: list,
    line: float,
    stat_type: str = "Kills",
    favorite_prob: float = 0.55,
    likely_maps: list = None,
    rank_gap: int = None,
    period_kpr: float = None,
    period_rating: float = None,
    period_adr: float = None,
    today_opp_rank: int | None = None,
    today_is_lan: bool | None = None,
    book_implied_prob: float = BOOK_IMPLIED_PROB,
) -> dict:
    """
    Run 100,000 Monte Carlo simulations using Negative Binomial distribution.
    Returns a dict with all computed statistics.

    today_opp_rank: HLTV world rank of today's opponent.  When provided, maps
    played against similarly-ranked opponents are weighted more heavily in the
    distribution fit.

    today_is_lan: True if today's match is LAN, False if Online, None if unknown.
    When provided, historical maps from the same context are weighted heavier.
    """
    stat_values = [m["stat_value"] for m in map_stats]
    rounds_per_map = [m["rounds"] for m in map_stats]

    # --- Opponent Quality Weighting ---
    _opp_weights = _opp_quality_weights(map_stats, today_opp_rank)
    _any_weighted = today_opp_rank is not None and any(w != 1.0 for w in _opp_weights)
    if _any_weighted:
        _covered = sum(1 for m in map_stats if m.get('opp_rank') is not None)
        logger.info(
            f"[sim] Opp quality weighting active — today opp #{today_opp_rank}, "
            f"{_covered}/{len(map_stats)} maps have historical opp_rank. "
            f"Weight range: {min(_opp_weights):.2f}–{max(_opp_weights):.2f}"
        )

    # --- LAN/Online Context Weighting ---
    _lan_weights = _lan_context_weights(map_stats, today_is_lan)
    _lan_active = today_is_lan is not None and any(w != 1.0 for w in _lan_weights)
    if _lan_active:
        _lan_covered = sum(1 for m in map_stats if m.get('is_lan') is not None)
        _lan_match_count = sum(1 for m in map_stats if m.get('is_lan') is True)
        ctx = "LAN" if today_is_lan else "Online"
        logger.info(
            f"[sim] LAN context weighting active — today={ctx}, "
            f"{_lan_covered}/{len(map_stats)} maps have known context "
            f"({_lan_match_count} LAN, {_lan_covered - _lan_match_count} Online). "
            f"Weight range: {min(_lan_weights):.2f}–{max(_lan_weights):.2f}"
        )

    # Combine opp + LAN context weights multiplicatively
    _opp_weights = [o * l for o, l in zip(_opp_weights, _lan_weights)]

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

    # --- Robust stats (R1): trimmed mean + MAD-σ ---
    # Used for projection clipping; raw mean/median still drive the embed
    # display so users see what they're used to.
    trimmed_avg_val = _trimmed_mean(series_totals, 0.10) if series_totals else 0.0
    sigma_mad_val   = _mad(series_totals) if series_totals else 0.0

    # --- DPR (Deaths Per Round) ---
    _dpr_valid = [m for m in map_stats if m.get("rounds", 0) > 0 and m.get("deaths") is not None]
    if _dpr_valid:
        dpr = round(sum(m["deaths"] / m["rounds"] for m in _dpr_valid) / len(_dpr_valid), 3)
    else:
        dpr = None

    # --- Outlier Detection ---
    # Flag when one monster or disaster series is distorting the average.
    # Standard: any series total more than 2 std devs from mean is an outlier.
    outlier_detected  = False
    outlier_note      = ""
    outlier_series    = None
    avg_without       = None
    med_without       = None
    if len(series_totals) >= 4 and stability_std > 0:
        series_mean = mean(series_totals)
        for _sv in series_totals:
            if abs(_sv - series_mean) > 2.0 * stability_std:
                outlier_detected = True
                outlier_series   = _sv
                _without = [v for v in series_totals if v != _sv]
                if _without:
                    avg_without = round(mean(_without), 2)
                    med_without = round(median(_without), 2)
                    outlier_note = (
                        f"⚠️ Outlier detected: {_sv} kills series. "
                        f"Avg without: {avg_without} · Median without: {med_without}"
                    )
                break  # flag first outlier only
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

    # Fix 2: Cap rounds projection at avg historical rounds + 2 per map.
    # Coinflip odds formula can produce 24 rounds/map (OT risk), but if a player's
    # historical maps averaged 20 rounds, projecting 24 inflates expected kills by 20%.
    # Cap prevents this compounding with any KPR inflation.
    if rounds_per_map:
        _avg_hist_rounds = sum(rounds_per_map) / len(rounds_per_map)
        _round_cap = int(_avg_hist_rounds) + 2
        if rounds_per_map_projected > _round_cap:
            logger.debug(
                f"[sim] Round cap applied: {rounds_per_map_projected} → {_round_cap} "
                f"(hist avg={_avg_hist_rounds:.1f})"
            )
            rounds_per_map_projected = _round_cap

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

    # --- EWMA Recency Weighting (Apr 2026 overhaul, Doc 1 #2) ──────────────
    # Replaces the previous "simple mean of last 2-4 maps" — which over-fit to
    # tiny samples and let single hot/cold games dominate. EWMA with half-life
    # 5 maps gives the last 5 maps ~45% of total weight, last 10 ~70%, last
    # 20 ~95%. This captures genuine form shifts while staying anchored to
    # the player's larger floor.
    # kpr_values are newest-first.
    n_kpr = len(kpr_values)
    if n_kpr >= 2:
        EWMA_HALF_LIFE_MAPS = 5
        _lam = 0.5 ** (1.0 / EWMA_HALF_LIFE_MAPS)   # ≈ 0.871
        _weights = [_lam ** i for i in range(n_kpr)]
        recent_avg_kpr = sum(w * v for w, v in zip(_weights, kpr_values)) / sum(_weights)
    else:
        recent_avg_kpr = mean(kpr_values) if kpr_values else 0.0
    # Display sample (last 5) — embed shows what "recent" means to the user
    recent_kpr_vals = kpr_values[:5]

    # Blend: 60% EWMA recent / 40% map-weighted historical anchor
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

    # ── Quality-of-Kill multiplier (Apr 2026 overhaul, Doc 1 #1) ────────────
    # Applies BEFORE the robust anchor / IQR / R2M shrinkage so all downstream
    # logic sees the eco-adjusted projection. Critically: NOT applied to
    # series_totals (those are facts — what the player actually delivered).
    # The series_totals stay raw so historical hit-rate and IQR remain anchored
    # to reality; only the forward projection is shifted.
    quality_mult, quality_label, quality_details = compute_kill_quality_multiplier(
        period_kpr=period_kpr,
        period_rating=period_rating,
        period_adr=period_adr,
    )
    if quality_mult != 1.0 and stat_type == "Kills":
        _pre_quality = expected_total
        expected_total = expected_total * quality_mult
        logger.debug(
            f"[sim] Quality-of-Kill: {_pre_quality:.2f} → {expected_total:.2f} "
            f"(mult={quality_mult:.3f}, {quality_label})"
        )

    # ── Robust anchor (R1+R2): trimmed-mean band + IQR clipping ─────────────
    # A single freak game can pull both the median and the mean noticeably.
    # We anchor projection to the trimmed mean (R1) and then clamp it inside
    # the player's interquartile range of historical 2-map totals (R2) so we
    # can never project a number above what the player has actually delivered
    # in the upper-half of their sample.
    iqr_clipped = False
    iqr_band: tuple | None = None
    if series_totals:
        _anchor = trimmed_avg_val if trimmed_avg_val > 0 else median(series_totals)
        # ±8% / ±10% trimmed-mean band — same intent as the old median anchor
        # but driven by the robust center.
        expected_total = max(
            _anchor * 0.90,
            min(_anchor * 1.08, expected_total),
        )
        # R2: hard IQR clamp (only when we have enough series to define one)
        if len(series_totals) >= 4:
            _s = sorted(series_totals)
            n = len(_s)
            q25 = _s[n // 4]
            q75 = _s[(3 * n) // 4]
            iqr_band = (q25, q75)
            _pre_clip = expected_total
            expected_total = max(q25, min(q75, expected_total))
            if expected_total != _pre_clip:
                iqr_clipped = True
                logger.debug(
                    f"[sim] IQR clip {_pre_clip:.1f} → {expected_total:.1f} "
                    f"(IQR {q25}-{q75})"
                )

    # ──────────────────────────────────────────────────────────────────────
    # EMPIRICAL PROJECTION (NB model removed per user request)
    # Probabilities and percentiles come straight from observed series totals.
    # No Monte Carlo, no NB sampling. Trimmed mean drives center, MAD-σ drives
    # dispersion, historical hit-rate (with shrink) drives over/under prob.
    # ──────────────────────────────────────────────────────────────────────

    # If opponent-quality weighting is active, blend the weighted map mean
    # into expected_total so today's matchup quality still moves the projection.
    if _any_weighted:
        _w_mean, _w_var = _weighted_mean_var(per_map_values, _opp_weights)
        # weighted per-map mean → series total (×2 maps)
        _w_total_proj = _w_mean * 2
        # 60% empirical anchor / 40% opp-weighted projection
        expected_total = 0.6 * expected_total + 0.4 * _w_total_proj
        # Re-clamp to IQR after the blend so the opp-weighted adjustment can't
        # push expected_total back outside the historical IQR band (R2 must be
        # the final guardrail).
        if iqr_band is not None:
            _pre_reclip = expected_total
            expected_total = max(iqr_band[0], min(iqr_band[1], expected_total))
            if expected_total != _pre_reclip:
                iqr_clipped = True
                logger.debug(
                    f"[sim] Post-blend IQR re-clip {_pre_reclip:.1f} → "
                    f"{expected_total:.1f} (IQR {iqr_band[0]}-{iqr_band[1]})"
                )
        logger.info(
            f"[sim] Empirical+weighted blend — anchor={expected_total:.2f} "
            f"w_mean={_w_mean:.2f} w_total={_w_total_proj:.2f}"
        )

    # ── Bayesian regression-to-the-mean (Apr 2026 calibration fix) ──────────
    # Without this, players coming off a hot run (n=10 series with raw mean
    # well above population) get projected at their hot-form rate, generating
    # phantom OVER edges. See header constants for diagnosis details.
    if series_totals:
        # Pick prior based on stat type; default to kills if unknown
        if stat_type and "head" in stat_type.lower():
            pop_mean = POP_MEAN_HS_BO3
        else:
            pop_mean = POP_MEAN_KILLS_BO3
        n_series = len(series_totals)
        w_player = n_series / (n_series + PROJ_SHRINK_K)
        pre_shrink = expected_total
        expected_total = w_player * expected_total + (1.0 - w_player) * pop_mean
        # Propagate shrinkage to series_totals so the empirical over/under
        # count below inherits the regression — otherwise the projection and
        # the probability disagree.
        delta = expected_total - pre_shrink
        if abs(delta) > 0.01:
            series_totals = [v + delta for v in series_totals]
            logger.info(
                f"[sim] R2M shrinkage: {pre_shrink:.2f} → {expected_total:.2f} "
                f"(pop_mean={pop_mean}, n={n_series}, w_player={w_player:.2f}, Δ={delta:+.2f})"
            )

    # Sim-mean / sim-std / sim-median now come from empirical stats.
    # We keep the field names (`sim_*`) for downstream backward-compat.
    sim_mean   = float(expected_total)
    sim_std    = float(sigma_mad_val) if sigma_mad_val > 0 else (
        float(np.std(series_totals)) if len(series_totals) > 1 else 0.0
    )
    sim_median = float(median(series_totals)) if series_totals else float(expected_total)

    # ── Empirical Over / Under / Push from historical series totals ────────
    # Push is only possible when line is an integer (and a series total
    # equals it exactly). For half-lines, push is mathematically impossible.
    if series_totals:
        n_st = len(series_totals)
        _overs_emp  = sum(1 for v in series_totals if v >  line)
        _unders_emp = sum(1 for v in series_totals if v <  line)
        if float(line).is_integer():
            _pushes_emp = sum(1 for v in series_totals if v == line)
        else:
            _pushes_emp = 0
        raw_over_prob = _overs_emp  / n_st
        under_prob    = _unders_emp / n_st
        push_prob     = _pushes_emp / n_st
    else:
        # No series data — fall back to book-implied prior
        raw_over_prob = book_implied_prob
        under_prob    = 1.0 - book_implied_prob
        push_prob     = 0.0

    # ── R5: small-sample shrink toward book-implied probability ────────────
    # Pulls the empirical OVER% toward 52.38% when N_series < 8 (full strength
    # at 8+). Prevents 100%-confidence calls off 3 series.
    n_series_for_shrink = len(series_totals)
    shrink_factor       = min(1.0, n_series_for_shrink / 8.0)
    over_prob = shrink_factor * raw_over_prob + (1.0 - shrink_factor) * book_implied_prob
    under_prob = max(0.0, 1.0 - over_prob - push_prob)
    # Defensive renormalize so (over, under, push) sums to exactly 1.0
    _total = over_prob + under_prob + push_prob
    if _total > 0:
        over_prob  /= _total
        under_prob /= _total
        push_prob  /= _total

    # Percentile of line within historical series totals
    if series_totals:
        line_percentile = round(
            sum(1 for v in series_totals if v <= line) / len(series_totals) * 100, 1
        )
    else:
        line_percentile = 50.0

    # NB-era params kept as 0 for output-shape backward compat
    r_total_adj = 0.0
    p_adj       = 0.0

    # --- Fair Line & Misprice Classification ---
    # Documents define:
    #   Prop Error  — line off by 5+ kills/headshots from fair line
    #   Mispriced   — line off by 2–4 from fair line
    #   Trap        — avg/median above line but stomp risk or declining form
    #   Fair Line   — within 2 of fair line, no strong signals
    fair_line  = round(sim_median, 1)
    line_gap   = fair_line - line
    abs_gap    = abs(line_gap)
    direction  = "+" if line_gap > 0 else ""

    # Determine misprice type
    hist_avg_raw    = mean(series_totals) if series_totals else 0
    hist_median_raw = median(series_totals) if series_totals else 0
    _avg_above_line = hist_avg_raw > line and hist_median_raw > line
    _stomp_or_cold  = stomp_via_rank or trend_pct <= -10

    if abs_gap >= 5.0:
        misprice_type  = "Prop Error"
        misprice_label = f"🚨 PROP ERROR ({direction}{line_gap:.1f}) — book used wrong data"
    elif abs_gap >= 2.0:
        if _avg_above_line and _stomp_or_cold:
            misprice_type  = "Trap"
            misprice_label = f"⚠️ TRAP ({direction}{line_gap:.1f}) — looks beatable but context kills it"
        else:
            misprice_type  = "Mispriced"
            misprice_label = f"⚠️ Mispriced ({direction}{line_gap:.1f}) — book underweighted context"
    elif _avg_above_line and _stomp_or_cold:
        misprice_type  = "Trap"
        misprice_label = "🪤 TRAP — Average above line but stomp/cold-form risk"
    else:
        misprice_type  = "Fair Line"
        misprice_label = "✅ Fair Line"

    # --- Expected Value (EV) vs standard -110 odds ---
    # EV = (over_prob * (100/110)) - (under_prob * 1.0)
    # Positive EV = value on OVER side at -110
    ev_over  = round(over_prob * (100 / 110) - under_prob, 4)
    ev_under = round(under_prob * (100 / 110) - over_prob, 4)

    # --- Percentile ceiling / floor from EMPIRICAL series totals ----------
    # Replaces NB-sample percentiles. With <4 series, fall back to MAD-σ
    # bands around the trimmed mean so embeds still get sensible numbers.
    if len(series_totals) >= 4:
        sim_p10 = float(np.percentile(series_totals, 10))
        sim_p25 = float(np.percentile(series_totals, 25))
        sim_p75 = float(np.percentile(series_totals, 75))
        sim_p90 = float(np.percentile(series_totals, 90))
    else:
        _spread = max(sim_std, 1.0)
        sim_p10 = max(0.0, sim_mean - 1.28 * _spread)
        sim_p25 = max(0.0, sim_mean - 0.67 * _spread)
        sim_p75 = sim_mean + 0.67 * _spread
        sim_p90 = sim_mean + 1.28 * _spread

    # --- Historical stats (R6: push half-credit) ──────────────────────────
    # Lines exactly at a player's median used to bias UNDER because pushes
    # were dropped from the numerator. Counting them as 0.5 OVER fixes that.
    if series_totals:
        _overs  = sum(1 for v in series_totals if v >  line)
        _pushes = sum(1 for v in series_totals if v == line)
        hit_rate = (_overs + 0.5 * _pushes) / len(series_totals)
    else:
        hit_rate = over_prob
    hist_avg    = mean(series_totals) if series_totals else mean(stat_values) * 2
    hist_median = median(series_totals) if series_totals else median(stat_values) * 2

    # ── Adaptive probability cap (CS2: sample × volatility × floor) ─────
    # Mirrors Valorant's three-cap stacking. Stops 90%+ confidence calls
    # on noisy samples or players whose floor is far below the line.
    # Example: Smash @ 26.5 — floor was 11 (a 13-3 sweep). Floor / line =
    # 0.42 → cap_floor = 0.70 instead of letting the model claim 80%+.
    if series_totals:
        _floor   = min(series_totals)
        _ceiling = max(series_totals)
        _hist_avg_real = mean(series_totals)
        _sigma   = float(np.std(series_totals)) if len(series_totals) > 1 else 0.0
        _cv      = (_sigma / _hist_avg_real) if _hist_avg_real > 0 else 1.0
        _n       = len(series_totals)

        if   _n <= 4:  cap_n = 0.72
        elif _n <= 6:  cap_n = 0.80
        elif _n <= 8:  cap_n = 0.85
        elif _n <= 12: cap_n = 0.88
        elif _n <= 18: cap_n = 0.91
        else:          cap_n = 0.94

        if   _sigma >= 8 or _cv >= 0.30: cap_vol = 0.72
        elif _sigma >= 5 or _cv >= 0.18: cap_vol = 0.80
        else:                            cap_vol = 0.92

        if line > 0:
            _floor_ratio   = _floor   / line
            _ceiling_ratio = _ceiling / line
            if over_prob >= 0.5:
                if   _floor_ratio >= 1.0:  cap_floor = 0.95
                elif _floor_ratio >= 0.85: cap_floor = 0.85
                elif _floor_ratio >= 0.70: cap_floor = 0.78
                else:                      cap_floor = 0.70
            else:
                if   _ceiling_ratio <= 1.0:  cap_floor = 0.95
                elif _ceiling_ratio <= 1.15: cap_floor = 0.85
                elif _ceiling_ratio <= 1.30: cap_floor = 0.78
                else:                        cap_floor = 0.70
        else:
            cap_floor = 0.95

        prob_cap = min(cap_n, cap_vol, cap_floor)
        _orig_over = over_prob
        # Reserve push mass first; over+under share what's left of probability
        # space. This keeps push_prob exactly unchanged across the cap.
        _avail        = max(0.0, 1.0 - push_prob)
        _cap_hi       = min(prob_cap, _avail)
        _cap_lo       = max(_avail - prob_cap, 0.0)
        over_prob     = max(_cap_lo, min(_cap_hi, over_prob))
        under_prob    = max(0.0, _avail - over_prob)
        if abs(over_prob - _orig_over) > 0.001:
            logger.info(
                f"[sim] CS2 prob cap applied: {_orig_over*100:.1f}% → {over_prob*100:.1f}% "
                f"(cap_n={cap_n} cap_vol={cap_vol} cap_floor={cap_floor}, push={push_prob*100:.1f}%)"
            )
        # Invariants: push unchanged, all in [0,1], sum ≈ 1
        assert abs((over_prob + under_prob + push_prob) - 1.0) < 1e-6, (
            f"prob mass broken: over={over_prob} under={under_prob} push={push_prob}"
        )
    else:
        prob_cap = 0.85

    # --- Grading ---
    edge = over_prob - 0.5

    grade, recommendation, decision, vote_tally = calculate_grade(
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
        book_implied_prob=book_implied_prob,
        series_totals=series_totals,
        n_series=len(series_totals),
        total_projected_rounds=total_projected_rounds,
        recent_avg_kpr=recent_avg_kpr,
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
        "recent_n_maps":          len(recent_kpr_vals),
        # --- Quality of Kill (Apr 2026 overhaul, Doc 1 #1) ---
        "quality_multiplier":     quality_mult,
        "quality_label":          quality_label,
        "quality_details":        quality_details,
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
        # --- New fields (document integration) ---
        "sim_p25":                round(sim_p25, 1),
        "sim_p75":                round(sim_p75, 1),
        "dpr":                    dpr,
        "misprice_type":          misprice_type,
        "outlier_detected":       outlier_detected,
        "outlier_note":           outlier_note,
        "outlier_series":         outlier_series,
        "avg_without_outlier":    avg_without,
        "med_without_outlier":    med_without,
        # --- Anti-overestimation fields (R1, R2, R5, R7) ──
        "trimmed_avg":            round(trimmed_avg_val, 2),
        "sigma_mad":              round(sigma_mad_val, 2),
        "raw_over_prob":          round(raw_over_prob * 100, 1),
        "shrink_factor":          round(shrink_factor, 2),
        "iqr_clipped":            iqr_clipped,
        "iqr_band":               (round(iqr_band[0], 1), round(iqr_band[1], 1)) if iqr_band else None,
        "vote_tally":             vote_tally,
    }


def apply_tier_caps(
    grade_num: int,
    decision: str,
    *,
    hit_rate: float,
    over_prob: float,
    hist_median: float,
    recent_avg_per_series: float,
    line: float,
    projected_rounds: int,
    stomp_via_rank: bool,
    favorite_prob: float,
    stability_std: float,
    sub_signal_alignment: int,   # 0-4: how many of s1-s4 agree with decision
    stat_type: str = "Kills",
) -> tuple:
    """
    April 2026 overhaul tier-cap stack — merged from Doc 1 (#1, #3, #5, #6) and
    Doc 2 (#1-9). Each cap can ONLY lower the grade. The starting `grade_num`
    is computed from edge-vs-book; these gates are then applied so that strong
    edges still need supporting evidence to earn elite tiers.

    Returns (post_cap_grade, list_of_cap_reasons).
    """
    caps = []
    g = grade_num

    def cap(new_max: int, reason: str):
        nonlocal g
        if g > new_max:
            caps.append(f"{reason}→cap {new_max}")
            g = new_max

    side_prob = over_prob if decision == "OVER" else (1.0 - over_prob)
    sp_pct = side_prob * 100
    hr_pct = hit_rate * 100

    # C1: Hit-rate hard filter (Doc 2 #4)
    if   hr_pct < 50: cap(5, f"HR {hr_pct:.0f}%<50%")
    elif hr_pct < 60: cap(6, f"HR {hr_pct:.0f}%<60%")
    elif hr_pct < 65: cap(8, f"HR {hr_pct:.0f}%<65%")
    elif hr_pct < 70: cap(9, f"HR {hr_pct:.0f}%<70%")

    # C2: Stomp-risk veto on elite tiers (Doc 1 #3, Doc 2 #2)
    # Stomp shortens maps → hurts OVERS only.  UNDERS benefit, so no cap there.
    if decision == "OVER" and stat_type == "Kills":
        if stomp_via_rank or favorite_prob >= 0.72:
            cap(7, "stomp risk")
        elif favorite_prob >= 0.65:
            cap(8, f"medium stomp ({favorite_prob*100:.0f}% fav)")

    # C3: Round-volume cap (Doc 1 #3, Doc 2 #6) — OVERS only
    if decision == "OVER" and stat_type == "Kills" and projected_rounds:
        if   projected_rounds < 40: cap(6, f"rounds {projected_rounds}<40")
        elif projected_rounds < 42: cap(7, f"rounds {projected_rounds}<42")
        elif projected_rounds < 44: cap(8, f"rounds {projected_rounds}<44")

    # C4: Side-probability band (Doc 2 #5)
    if   sp_pct < 55: cap(5, f"prob {sp_pct:.0f}%<55%")
    elif sp_pct < 60: cap(6, f"prob {sp_pct:.0f}%<60%")
    elif sp_pct < 65: cap(7, f"prob {sp_pct:.0f}%<65%")
    elif sp_pct < 70: cap(9, f"prob {sp_pct:.0f}%<70%")

    # C6: Median + recent-avg gap requirement for elite tiers (Doc 2 #1)
    if decision == "OVER":
        gap_med = hist_median - line
        gap_rec = recent_avg_per_series - line
    else:
        gap_med = line - hist_median
        gap_rec = line - recent_avg_per_series
    if g >= 10 and (gap_med < 2.0 or gap_rec < 3.0):
        cap(9, f"gap insufficient for 10 (med {gap_med:+.1f}, rec {gap_rec:+.1f})")
    if g >= 9 and (gap_med < 1.5 or gap_rec < 2.5):
        cap(8, f"gap insufficient for 9 (med {gap_med:+.1f}, rec {gap_rec:+.1f})")

    # C7: Volatility cap (Doc 1 #6, Doc 2 #8)
    if stability_std > 8.5:
        cap(8, f"σ={stability_std:.1f}>8.5")

    # C8: Multi-signal convergence requirement (Doc 1 #5, Doc 2 #3)
    if g >= 10 and sub_signal_alignment < 4:
        cap(9, f"convergence {sub_signal_alignment}/4")
    if g >= 9 and sub_signal_alignment < 3:
        cap(8, f"convergence {sub_signal_alignment}/4")

    return g, caps


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
    book_implied_prob: float = BOOK_IMPLIED_PROB,
    series_totals: list = None,
    n_series: int = 0,
    total_projected_rounds: int = 0,
    recent_avg_kpr: float = 0.0,
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

    # ── S4: Raw simulation probability (weight 2) ────────────────────────────
    # 100 K Monte Carlo runs are the most statistically robust signal in the
    # stack.  Weight raised from 1→2 so a clear 76% UNDER sim result cannot
    # be outvoted to PASS by a noisy hit-rate reading from 3–4 series.
    if   over_prob >= 0.68:  s4 = 2
    elif over_prob >= 0.60:  s4 = 1
    elif over_prob >= 0.40:  s4 = 0
    elif over_prob >= 0.32:  s4 = -1
    else:                    s4 = -2

    # ── Fix 3: Consecutive cold-series penalty ───────────────────────────────
    # series_totals[0] = most recent completed series (newest-first ordering).
    # If the 2 most recent series are BOTH under the line the player is clearly
    # cold at this exact number.  Penalise the raw score so borderline OVER
    # calls flip to PASS and strong OVER calls lose at least one grade tier.
    cold_penalty = 0
    if series_totals and len(series_totals) >= 2:
        n_cold = sum(1 for v in series_totals[:3] if v < line)
        if series_totals[0] < line and series_totals[1] < line:
            cold_penalty = -3 if n_cold >= 3 else -2

    # ── Weighted consensus ───────────────────────────────────────────────────
    raw_total = (s1 * 3) + (s2 * 2) + (s3 * 2) + (s4 * 2) + cold_penalty

    # ── Stomp modifier ───────────────────────────────────────────────────────
    # Stomps shorten maps ~10% (≈21→19 rounds).  A 30% score reduction keeps
    # dominant historical edges alive while cancelling borderline leans.
    stomp = (stomp_via_rank or favorite_prob >= 0.72) and stat_type == "Kills"
    if stomp:
        total = int(raw_total * 0.70)
    else:
        total = raw_total

    # ── Decision (threshold lowered 7→6) ─────────────────────────────────────
    # Old ±7 left clear directional signals (e.g. 76% under) as PASS when the
    # evidence stack landed at -6.  Lowering to ±6 lets the strongest combos
    # of hist evidence + sim probability convert into UNDER/OVER calls.
    if total >= 6:
        decision = "OVER"
    elif total <= -6:
        decision = "UNDER"
    else:
        decision = "PASS"

    # ── R7: Strict gate + signal alignment ──────────────────────────────────
    # Vote tally from the four sub-signals (s1=hit-rate, s2=median-gap,
    # s3=trend, s4=sim-prob). +1 = OVER vote, -1 = UNDER vote, 0 = neutral.
    over_votes  = sum(1 for s in (s1, s2, s3, s4) if s >  0)
    under_votes = sum(1 for s in (s1, s2, s3, s4) if s <  0)
    pass_reason: str | None = None

    # 1. Sample size: need at least 5 series for any non-PASS verdict
    if decision != "PASS" and n_series < 5:
        pass_reason = f"sample-size gate (N={n_series}<5)"
        decision = "PASS"

    # 2. Edge size: need ≥ 7% directional edge vs book-implied
    if decision != "PASS":
        if decision == "OVER":
            _dir_edge_check = (over_prob - book_implied_prob)
        else:
            _dir_edge_check = ((1.0 - over_prob) - book_implied_prob)
        if _dir_edge_check < 0.07:
            pass_reason = f"edge gate (|edge|={_dir_edge_check*100:.1f}%<7%)"
            decision = "PASS"

    # 3. Signal alignment: majority of sub-signals must point the same way
    if decision == "OVER" and over_votes <= under_votes:
        pass_reason = f"signals split ({over_votes}🟢/{under_votes}🔴)"
        decision = "PASS"
    elif decision == "UNDER" and under_votes <= over_votes:
        pass_reason = f"signals split ({over_votes}🟢/{under_votes}🔴)"
        decision = "PASS"
    # If we landed at PASS via base threshold AND signals are split, surface
    # that to the user too so the embed always explains why it's a PASS.
    if decision == "PASS" and pass_reason is None and over_votes == under_votes:
        pass_reason = f"signals split ({over_votes}🟢/{under_votes}🔴)"

    vote_tally = {
        "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "over_votes":  over_votes,
        "under_votes": under_votes,
        "pass_reason": pass_reason,
    }

    # ── Grade scale — edge % vs -110 (52.38% implied) ────────────────────────
    # Documents define grade by edge vs book, NOT by raw evidence stack score.
    # Evidence stack drives DECISION only; grade number reflects betting value.
    #
    #   10/10  edge ≥ 15%   Prop error / extreme misprice
    #   9/10   edge ≥ 12%   Very strong, book made a clear mistake
    #   8/10   edge ≥  8%   Strong, playable with supporting context
    #   7/10   edge ≥  5%   Solid value, worth a bet if context confirms
    #   6/10   edge ≥  3%   Marginal, playable only with strong conviction
    #   5/10   edge ≥  0%   Borderline — line is roughly fair
    #   4/10   negative     Wrong side or no edge — do not bet
    if decision == "PASS":
        grade_str = "N/A"
        rec = "⏸️ PASS — Signals too mixed for a confident call"
    else:
        # Directional edge vs book implied probability (default -110 = 52.38%)
        if decision == "OVER":
            dir_edge_pct = (over_prob - book_implied_prob) * 100
        else:  # UNDER
            dir_edge_pct = ((1.0 - over_prob) - book_implied_prob) * 100

        if   dir_edge_pct >= 15: grade_num, edge_label = 10, "Elite edge"
        elif dir_edge_pct >= 12: grade_num, edge_label = 9,  "Elite edge"
        elif dir_edge_pct >= 8:  grade_num, edge_label = 8,  "Strong edge"
        elif dir_edge_pct >= 5:  grade_num, edge_label = 7,  "Solid lean"
        elif dir_edge_pct >= 3:  grade_num, edge_label = 6,  "Marginal edge"
        elif dir_edge_pct >= 0:  grade_num, edge_label = 5,  "Fair line"
        else:                    grade_num, edge_label = 4,  "Negative edge"

        # ── Apr 2026 overhaul: tier-cap stack (merged Doc 1 + Doc 2) ─────────
        # Compute how many of the 4 sub-signals agree with the chosen direction
        if decision == "OVER":
            sub_align = sum(1 for s in (s1, s2, s3, s4) if s > 0)
        else:
            sub_align = sum(1 for s in (s1, s2, s3, s4) if s < 0)
        # Recent-avg in per-series units (recent_avg_kpr × total_projected_rounds)
        recent_avg_per_series = (recent_avg_kpr * total_projected_rounds) if recent_avg_kpr else hist_avg
        pre_cap_grade = grade_num
        grade_num, caps_applied = apply_tier_caps(
            grade_num=grade_num,
            decision=decision,
            hit_rate=hit_rate,
            over_prob=over_prob,
            hist_median=hist_median,
            recent_avg_per_series=recent_avg_per_series,
            line=line,
            projected_rounds=total_projected_rounds,
            stomp_via_rank=stomp_via_rank,
            favorite_prob=favorite_prob,
            stability_std=stability_std,
            sub_signal_alignment=sub_align,
            stat_type=stat_type,
        )
        if caps_applied:
            edge_label = f"{edge_label} (capped)"
        vote_tally["caps_applied"] = caps_applied
        vote_tally["pre_cap_grade"] = pre_cap_grade
        vote_tally["sub_signal_alignment"] = sub_align

        grade_str = f"{grade_num}/10 ({edge_label})"
        sign = "✅" if decision == "OVER" else "❌"
        rec  = f"{sign} {decision} — {grade_str}"

    return grade_str, rec, decision, vote_tally
