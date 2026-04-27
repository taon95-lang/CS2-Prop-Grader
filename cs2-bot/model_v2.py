"""
model_v2.py — alternative grading model for !slip2 / !modelv2

Implements the user's revised grading logic side-by-side with the
existing build_and_format_slips pipeline. The bot's main !slip /
!autoslip commands are unchanged.

Pipeline:
  1. grade_prop(p)       — single-prop grade with stronger NO BET filter,
                            HARD FAIL on fragile overs, strict A gate,
                            UNDER boost on short maps.
  2. limit_A_plays(...)  — caps A-grade plays at top 3 by edge.
  3. build_slips(...)    — sizes [2,3,4,6], score = avg_prob*0.5 +
                            avg_edge*0.5, no same-team within slip.
  4. format_for_discord  — text output ready for Discord code-block.

Required input fields per play:
  player, team, opponent, line
  over_prob, under_prob, edge       (all 0–100 percent floats)
  variance      ('low'|'medium'|'high')   ← canonical variance signal
  avg           (float, projected mean for the stat)
  normal_map_proj (float, projected mean on a normal-length map)
  stomp         (bool)
  hit_rate      (0–100 percent float)
  short_map_proj (float, projected stat on shorter maps)

Variance score (categorical → fixed numeric):
  low  → 3
  medium → 6
  high → 10

Edge handling:
  Profile-based stepwise penalties applied independently:
    weak hit-rate (<50%)               → −5
    short-map projection below line    → −5
    opponent stomp risk                → −3
  adjusted_edge = edge − (sum of triggered penalties)
  Worst case (all three) = edge − 13.
  Variance is NOT subtracted from the edge — it's enforced via
  separate var_score gates in the value classifier and the coin-
  flip NO BET filter. The raw `edge` is preserved on the play
  dict for display.

Decision is now driven by SIGNAL COUNTING, not by raw simulator probability.
Pipeline order inside grade_prop:
  1. decide_side(data)  →  one of OVER / LEAN OVER / UNDER / LEAN / NO BET
                            using avg, median, normal_proj, short_proj,
                            sim probs (only as 1 of 4 signals), round_swing,
                            stomp_risk, sample_size.
  2. Map verdict to dict `decision`:
       OVER / LEAN OVER  →  decision = 'OVER'   (raw_verdict preserved on dict)
       UNDER             →  decision = 'UNDER'
       LEAN              →  decision = 'NO BET' (coin-flip pass)
       NO BET            →  decision = 'NO BET' (return early)
  3. Existing pipeline (variance score, adjusted_edge, score, value
     classification, A/B grading, NO BET filters, force-A) runs on top
     of the new decision exactly as before.

decide_side's own NO BET conditions:
  • sample_size < 8                                          (insufficient data)
  • avg/median straddle the line in opposite directions      (conflicting core)
  • stomp_risk == True AND round_swing == 'LOW'              (no upside)
  • signals tie → 'LEAN' → mapped to NO BET in grade_prop    (coin-flip)

Existing hard NO BET filters (still applied AFTER decide_side picks a side):
  1. hit_rate < 40 AND variance == 'high'        → NO BET / NO VALUE
  2. adjusted_edge < 6 OR pick prob < 55         → NO BET
  3. var_score ≥ 9 AND |avg − line| < 2          → NO BET (coin-flip trap)
  4. OVER + short-map fail + (high var OR stomp) → NO BET (fragile over)

Score adjustments (applied inside grade_prop, before grade is set):
  Score starts at `prob` (0–100). Profile penalties may reduce it:
    UNDER + variance == 'high'                   → score −5
  The master filter in run_model later trips NO BET on score < 55,
  so a 5-point dock can flip a borderline UNDER. Score is display
  only — see the per-play status fields section below.

Grade overrides (applied inside grade_prop, after the strict-A gate
and after the variance/short-map downgrades):
  OVER + adjusted_edge ≥ 10 AND prob ≥ 58        → grade = 'A' (forced)
    Strong-edge OVERs guarantee A regardless of variance / short-map
    state, provided the pick prob clears 58 (blocks coin-flip OVERs
    from being elevated on edge alone). Earlier hard NO BET filters
    still rule out the worst fragile overs (filter #4), so this only
    fires on plays that already survive the NO BET gates.
  UNDER + short-map fail + currently grade B     → grade = 'A' (boost)
    Short maps favor unders; reward survival of all earlier filters.

Value classification (independent of grade):
  STRONG VALUE   — adjusted_edge ≥ 10 AND hit_rate ≥ 60
  MODERATE VALUE — adjusted_edge ≥ 6
  LOW VALUE      — otherwise
  NO VALUE       — only set by hard-NO-BET filter (1) above

Value demotion (post-classification):
  If ANY profile penalty triggered (hit_rate<50, short_map<line,
  stomp), STRONG VALUE is demoted to MODERATE — STRONG status
  requires a clean profile.

Mispriced flag (diagnostic):
  |normal_map_proj − line| ≥ 3 AND hit_rate ≥ 60
  Surfaces "the line looks soft" plays without changing grade.

Per-play status fields written by run_model's consistency pass
(applied in order; later rules override earlier ones):
  score        — 0–100 CONFIDENCE on the chosen direction (= prob).
                 Display-only signal — does NOT trigger or gate bets.
                 Set to None when final_label == 'NO BET' so it is
                 never shown as justification for an unqualified play.
  can_bet      — Soft eligibility flag (always set). True if
                 hit_rate ≥ 50 AND short_map_proj > line AND
                 normal_map_proj > line. Score is intentionally
                 excluded — score is confidence, not a bet trigger.
                 Diagnostic only; does not gate any other field.
  final_label  — Set to 'NO BET' by MASTER FINAL FILTER when any
                 fail condition holds (else unset).
  bet_size     — 0 when MASTER FINAL FILTER fires (also set to 0
                 by Rule 2 NO BET, 0.5 by Rule 1 weak grade).
  confidence   — 0 when MASTER FINAL FILTER fires (else unset).

MASTER FINAL FILTER (HARD OVERRIDE — runs last, authoritative):
  If score < 55 OR hit_rate < 50 OR short_map_proj < line
  OR normal_map_proj < line:
      final_label = 'NO BET', bet_size = 0, confidence = 0
"""

import itertools


# ─────────────────────────────────────────────────────────────────────
# Signal-counting decision (drives grade_prop's `decision` field)
# ─────────────────────────────────────────────────────────────────────
def decide_side(data: dict) -> str:
    """
    Pure signal-counting decision. Returns one of:
        'OVER'  /  'UNDER'  /  'LEAN OVER'  /  'LEAN'  /  'NO BET'

    Data fields expected (see _to_decision_data for the mapping from
    grade_prop's input dict):
        avg, median, line
        normal_proj, short_proj
        over_prob, under_prob          (0–100)
        round_swing  ('LOW'|'MEDIUM'|'HIGH')
        multi_kill   ('LOW'|'MEDIUM'|'HIGH')   (carried but unused here)
        stomp_risk   (bool)
        variance     ('LOW'|'MEDIUM'|'HIGH')   (carried but unused here)
        sample_size  (int)
        hit_rate     (0–100, carried but unused here)
        side         ('OVER'|'UNDER',          carried but unused here)

    Philosophy:
      • Real per-map data (avg, median, normal_proj) drives the decision.
      • Simulator probability is just one of four equal signals.
      • NO BET only triggers on TRUE conflicts or insufficient data.
      • No 100-point composite score.
    """
    # 1. HARD NO BET FILTERS (only true edge killers) ────────────────
    if data["sample_size"] < 8:
        return "NO BET"

    conflicting_core = (
        (data["avg"] > data["line"] and data["median"] < data["line"]) or
        (data["avg"] < data["line"] and data["median"] > data["line"])
    )
    extreme_stomp_low_swing = (
        data["stomp_risk"] and data["round_swing"] == "LOW"
    )
    if conflicting_core or extreme_stomp_low_swing:
        return "NO BET"

    # 2. MATCH-LENGTH ADJUSTMENT (round-swing scaled penalty) ────────
    short_map_penalty = 0.0
    if data["short_proj"] < data["line"]:
        short_map_penalty = 1.0
        if data["round_swing"] == "HIGH":
            short_map_penalty *= 0.5
        elif data["round_swing"] == "MEDIUM":
            short_map_penalty *= 0.75

    # 3. BASE SIGNAL (real data first) ───────────────────────────────
    over_signal = 0
    under_signal = 0

    if data["avg"] > data["line"]:
        over_signal += 1
    else:
        under_signal += 1

    if data["median"] > data["line"]:
        over_signal += 1
    else:
        under_signal += 1

    if data["normal_proj"] > data["line"]:
        over_signal += 1
    else:
        under_signal += 1

    # 4. SIMULATION (used last, just one signal) ─────────────────────
    if data["over_prob"] >= 55:
        over_signal += 1
    elif data["under_prob"] >= 55:
        under_signal += 1

    # 5. STRONG OVER OVERRIDE (avg/median/normal all ≥ line) ────────
    strong_over = (
        data["avg"] >= data["line"] and
        data["median"] >= data["line"] and
        data["normal_proj"] >= data["line"] and
        data["round_swing"] in ("MEDIUM", "HIGH")
    )
    if strong_over:
        return "LEAN OVER" if data["over_prob"] < 60 else "OVER"

    # 6. FINAL DECISION (signal counts, not 100-point score) ─────────
    over_score = over_signal - short_map_penalty
    under_score = under_signal - short_map_penalty

    if over_score > under_score:
        return "OVER"
    elif under_score > over_score:
        return "UNDER"
    else:
        return "LEAN"


def _to_decision_data(p: dict) -> dict:
    """
    Map a grade_prop input dict into the field shape that decide_side
    expects: renames legacy keys, normalizes case, fills safe defaults
    for fields that aren't yet wired through bot.py's data pipeline
    (median, round_swing, multi_kill).

    Defaults are deliberately neutral — they keep the new logic active
    without spuriously triggering its NO BET / strong-over rules:
      • median       → defaults to avg (symmetric distribution)
      • round_swing  → defaults to 'MEDIUM' (disables LOW + stomp NO BET)
      • multi_kill   → defaults to 'MEDIUM' (currently unused by decide_side)
    """
    variance_upper = (str(p.get("variance") or "medium")).upper()
    if variance_upper not in ("LOW", "MEDIUM", "HIGH"):
        variance_upper = "MEDIUM"

    rs = (str(p.get("round_swing") or "MEDIUM")).upper()
    if rs not in ("LOW", "MEDIUM", "HIGH"):
        rs = "MEDIUM"

    mk = (str(p.get("multi_kill") or "MEDIUM")).upper()
    if mk not in ("LOW", "MEDIUM", "HIGH"):
        mk = "MEDIUM"

    avg = float(p.get("avg",
                      p.get("normal_map_proj", p.get("line", 0.0))))
    median_val = p.get("median")
    median = float(median_val) if isinstance(median_val, (int, float)) else avg

    op = float(p.get("over_prob") or 0.0)
    up = float(p.get("under_prob") or 0.0)

    return {
        "avg":          avg,
        "median":       median,
        "line":         float(p.get("line", 0.0)),
        "normal_proj":  float(p.get("normal_map_proj", p.get("line", 0.0))),
        "short_proj":   float(p.get("short_map_proj", p.get("line", 0.0))),
        "over_prob":    op,
        "under_prob":   up,
        "round_swing":  rs,
        "multi_kill":   mk,
        "stomp_risk":   bool(p.get("stomp", False)),
        "variance":     variance_upper,
        "sample_size":  int(p.get("sample_size",
                                  p.get("history_n", 10)) or 10),
        "hit_rate":     float(p.get("hit_rate", 0.0)),
        "side":         "OVER" if op >= up else "UNDER",
    }


# ─────────────────────────────────────────────────────────────────────
# Grade ONE prop
# ─────────────────────────────────────────────────────────────────────
def grade_prop(p: dict) -> dict:
    # ── SIGNAL-COUNTING DECISION (replaces the old prob-based one) ──
    # decide_side returns one of OVER / LEAN OVER / UNDER / LEAN /
    # NO BET. We map LEAN OVER → OVER (carrying raw_verdict for the
    # display layer), LEAN → NO BET (coin-flip pass), and NO BET →
    # NO BET (return early with a minimal extras-shaped dict).
    raw_verdict = decide_side(_to_decision_data(p))
    p["raw_verdict"] = raw_verdict

    if raw_verdict in ("NO BET", "LEAN"):
        return {
            **p,
            "decision":      "NO BET",
            "grade":         "NO BET",
            "value":         "NO VALUE",
            "mispriced":     False,
            "adjusted_edge": float(p.get("edge", 0.0)),
            "var_score":     {"low": 3, "medium": 6, "high": 10}.get(
                                str(p.get("variance") or "medium").lower(), 6),
            "score":         None,
        }

    decision = "OVER" if raw_verdict in ("OVER", "LEAN OVER") else "UNDER"
    prob = p["over_prob"] if decision == "OVER" else p["under_prob"]

    short_fail = p["short_map_proj"] < p["line"]
    stomp = bool(p.get("stomp", False))

    # ── VARIANCE SCORE (categorical bucket → fixed numeric score) ────
    # Still used by the value classification (STRONG requires var≤7)
    # and the coin-flip NO BET rule (var≥9 + |avg-line|<2).
    var_score = {"low": 3, "medium": 6, "high": 10}.get(p["variance"], 6)

    # ── ADJUSTED EDGE ────────────────────────────────────────────────
    # Profile-based stepwise penalties — each triggers independently:
    #   • weak hit-rate (<50%)               → −5
    #   • short-map projection below line    → −5
    #   • opponent stomp risk                → −3
    # Worst case (all three) = edge − 13. Variance is NOT subtracted
    # from the edge anymore; it's enforced separately via var_score
    # gates in the value classifier and coin-flip filter below.
    adjusted_edge = p["edge"]
    if p["hit_rate"] < 50:
        adjusted_edge -= 5
    if p["short_map_proj"] < p["line"]:
        adjusted_edge -= 5
    if p.get("stomp"):
        adjusted_edge -= 3

    # ── PLAY-LEVEL SCORE ─────────────────────────────────────────────
    # Canonical 0–100 confidence on the chosen direction. Starts at
    # `prob`; profile-based penalties below mutate it before it gets
    # frozen onto the play. Score is display/confidence only — see
    # "Score adjustments" in the module docstring.
    score = prob

    # UNDER + HIGH VARIANCE penalty: UNDER bets on high-variance
    # plays are noisier than prob alone suggests. Drop 5 points so
    # borderline UNDERs slide under the master filter's <55 cut.
    # Mirrors the -5 adjusted_edge profile penalties in scale.
    if decision == "UNDER" and p["variance"] == "high":
        score -= 5

    # Pin canonical `normal_map_proj` onto the play so all downstream
    # consumers (consistency rules, exports) read the same value.
    normal_proj = float(p.get("normal_map_proj", p.get("avg", p["line"])))
    p["normal_map_proj"] = normal_proj

    # ── HARD NO BET: weak history + high variance ────────────────────
    # If the model has weak hit-rate evidence (<40%) AND variance is
    # high, no edge calculation can save it — refuse outright.
    if p["hit_rate"] < 40 and p["variance"] == "high":
        return {**p, "decision": "NO BET", "grade": "NO BET",
                "value": "NO VALUE", "mispriced": False,
                "adjusted_edge": adjusted_edge, "var_score": var_score,
                "score": score}

    # ── MISPRICED CHECK ──────────────────────────────────────────────
    # Model disagrees with line by ≥3 AND historical hit-rate is
    # reliable (≥60%). Pure diagnostic flag — does NOT change grade
    # by itself, just surfaces "the line looks soft" plays.
    mispriced = (
        abs(normal_proj - float(p["line"])) >= 3
        and p["hit_rate"] >= 60
    )

    # ── VALUE CLASSIFICATION ─────────────────────────────────────────
    # STRONG now requires reliable history (hit_rate ≥ 60) instead of
    # the previous variance gate. High-variance plays can still earn
    # STRONG provided their historical hit-rate backs them up.
    if adjusted_edge >= 10 and p["hit_rate"] >= 60:
        value = "STRONG VALUE"
    elif adjusted_edge >= 6:
        value = "MODERATE VALUE"
    else:
        value = "LOW VALUE"

    # ── VALUE DEMOTION (profile penalty) ─────────────────────────────
    # If ANY of the three profile penalties triggered above, demote
    # STRONG VALUE → MODERATE — STRONG status requires a clean profile.
    if (
        p["hit_rate"] < 50
        or p["short_map_proj"] < p["line"]
        or p.get("stomp")
    ):
        if value == "STRONG VALUE":
            value = "MODERATE VALUE"

    # Common return scaffold — every code path below carries these
    extras = {
        "value":         value,
        "mispriced":     mispriced,
        "adjusted_edge": adjusted_edge,
        "var_score":     var_score,
        "score":         score,
    }

    # ── NO BET: adjusted edge / pick prob too low ────────────────────
    if adjusted_edge < 6 or prob < 55:
        return {**p, "decision": "NO BET", "grade": "NO BET", **extras}

    # ── NO BET: coin-flip-with-high-variance trap ────────────────────
    # var_score ≥ 9 (i.e. high variance) AND |avg − line| < 2.
    avg = float(p.get("avg", p["line"]))
    if var_score >= 9 and abs(avg - float(p["line"])) < 2:
        return {**p, "decision": "NO BET", "grade": "NO BET", **extras}

    # ── NO BET: fragile overs (short-map fail + high var OR stomp) ──
    if decision == "OVER" and short_fail and (p["variance"] == "high" or stomp):
        return {**p, "decision": "NO BET", "grade": "NO BET", **extras}

    grade = "B"

    # ── A grade (strict, ADJUSTED edge) ──────────────────────────────
    if (
        adjusted_edge >= 10
        and prob >= 65
        and p["hit_rate"] >= 60
        and p["variance"] != "high"
    ):
        if not short_fail:
            grade = "A"

    # ── Downgrades ───────────────────────────────────────────────────
    if short_fail and decision == "OVER":
        grade = "B"
    if p["variance"] == "high":
        grade = "B"

    # ── OVER + STRONG EDGE → grade A (overrides downgrades) ─────────
    # If the model still shows ≥10 EV after every profile penalty
    # has been subtracted (weak hit, short-map fail, stomp) AND
    # the pick prob is at least 58, the raw edge is doing the work
    # and we guarantee grade A. The prob≥58 floor blocks coin-flip
    # OVERs from being elevated on edge alone. Note: the earlier
    # "OVER + short_fail + (high var OR stomp)" hard NO BET (above)
    # still rules out the worst fragile overs, so this guarantee
    # only applies to plays that survive that gate.
    if decision == "OVER" and adjusted_edge >= 10 and prob >= 58:
        grade = "A"

    # ── UNDER boost (short maps favor unders) ────────────────────────
    if decision == "UNDER" and short_fail and grade == "B":
        grade = "A"

    return {**p, "decision": decision, "grade": grade, **extras}


# ─────────────────────────────────────────────────────────────────────
# Limit A plays to top-N by edge (so slips aren't overloaded with A's)
# ─────────────────────────────────────────────────────────────────────
def limit_A_plays(graded_props: list[dict], max_A: int = 3) -> list[dict]:
    A_plays = [p for p in graded_props if p["grade"] == "A"]
    # Rank by ADJUSTED edge so high-variance plays don't beat clean ones.
    A_plays.sort(key=lambda x: x.get("adjusted_edge", x["edge"]), reverse=True)
    keep_A = A_plays[:max_A]
    keep_A_ids = {id(x) for x in keep_A}
    for p in graded_props:
        if p["grade"] == "A" and id(p) not in keep_A_ids:
            p["grade"] = "B"
    return graded_props


# ─────────────────────────────────────────────────────────────────────
# Build slips (smart-value, no same team, no NO BET)
# ─────────────────────────────────────────────────────────────────────
def build_slips(props: list[dict], sizes=(2, 3, 4, 6)) -> list[dict]:
    def get_prob(p):
        return p["over_prob"] if p["decision"] == "OVER" else p["under_prob"]

    def get_adj_edge(p):
        return p.get("adjusted_edge", p["edge"])

    playable = [
        p for p in props
        if p["decision"] != "NO BET"
        and p["grade"] in ("A", "B")
        and p.get("value") in ("STRONG VALUE", "MODERATE VALUE")
        and get_prob(p) >= 60
        and get_adj_edge(p) >= 6   # ADJUSTED edge gate
    ]

    slips: list[dict] = []
    for size in sizes:
        if len(playable) < size:
            continue
        for combo in itertools.combinations(playable, size):
            teams = [p["team"] for p in combo]
            if len(set(teams)) != len(teams):
                continue  # same-team inside slip
            probs     = [get_prob(p)     for p in combo]
            adj_edges = [get_adj_edge(p) for p in combo]
            raw_edges = [p["edge"]       for p in combo]
            avg_prob     = sum(probs)     / len(probs)
            avg_adj_edge = sum(adj_edges) / len(adj_edges)
            avg_raw_edge = sum(raw_edges) / len(raw_edges)
            # Score uses ADJUSTED edge so high-variance combos drop in rank.
            score = (avg_prob * 0.5) + (avg_adj_edge * 0.5)
            slips.append({
                "legs": list(combo),
                "score": score,
                "avg_prob": avg_prob,
                "avg_edge": avg_raw_edge,        # raw edge for display
                "avg_adj_edge": avg_adj_edge,    # adjusted edge for transparency
                "size": size,
            })
    slips.sort(key=lambda x: x["score"], reverse=True)
    return slips


# ─────────────────────────────────────────────────────────────────────
# Format for Discord (raw text intended to live in a code block)
# ─────────────────────────────────────────────────────────────────────
def format_for_discord(
    graded: list[dict],
    slips: list[dict],
    *,
    top_n: int = 5,
    show_grades: bool = True,
    disp_team_key: str = "team",
) -> str:
    """
    Format graded plays + top slips for Discord.

    Pass `disp_team_key="_disp_team"` if your adapter stored a clean
    display version of the team alongside the dedup key.
    """
    out: list[str] = []
    out.append("🏁 MODEL v2 — FINAL GRADES\n")

    if show_grades:
        for p in sorted(
            graded,
            key=lambda x: (
                {"A": 0, "B": 1, "NO BET": 2}.get(x["grade"], 3),
                -max(x["over_prob"], x["under_prob"]),
            ),
        ):
            prob = p["over_prob"] if p["decision"] == "OVER" else p["under_prob"]
            tag  = "🟢" if p["decision"] == "OVER" else ("🔴" if p["decision"] == "UNDER" else "⚪")
            grade_label = p["grade"]
            raw_e = p["edge"]
            adj_e = p.get("adjusted_edge", raw_e)
            edge_str = (
                f"Edge: +{raw_e:.1f}% (adj +{adj_e:.1f}%)"
                if abs(adj_e - raw_e) >= 0.05 else
                f"Edge: +{raw_e:.1f}%"
            )
            value_str = p.get("value", "")
            value_suffix = f" | {value_str}" if value_str else ""
            mispriced_suffix = " · 💎 MISPRICED" if p.get("mispriced") else ""
            out.append(
                f"{tag} {p['player']} vs {p['opponent']} | "
                f"{p['decision']} {p['line']} | "
                f"{prob:.0f}% | Grade: {grade_label} | "
                f"{edge_str}{value_suffix}{mispriced_suffix}"
            )
        out.append("")

    out.append("🔥 BEST SLIPS (model v2)\n")
    if not slips:
        out.append("(no slips passed all gates — try a wider time window)")
        return "\n".join(out)

    for i, slip in enumerate(slips[:top_n], 1):
        out.append(
            f"\nSlip #{i} ({slip['size']} Leg) | Score {slip['score']:.1f} | "
            f"Avg Prob {slip['avg_prob']:.1f}% | Avg Edge +{slip['avg_edge']:.1f}%"
        )
        for leg in slip["legs"]:
            prob = leg["over_prob"] if leg["decision"] == "OVER" else leg["under_prob"]
            tag  = "🟢" if leg["decision"] == "OVER" else "🔴"
            team_show = leg.get(disp_team_key) or leg.get("team") or "?"
            value_str = leg.get("value", "")
            value_suffix = f" · {value_str}" if value_str else ""
            mispriced_suffix = " · 💎" if leg.get("mispriced") else ""
            out.append(
                f"  {tag} {leg['player']} ({team_show}) vs {leg['opponent']} | "
                f"{leg['decision']} {leg['line']} ({prob:.0f}%) | "
                f"Grade {leg['grade']}{value_suffix}{mispriced_suffix}"
            )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
# Convenience wrapper used by the bot command
# ─────────────────────────────────────────────────────────────────────
def run_model(
    props: list[dict],
    *,
    sizes=(2, 3, 4, 6),
    max_A: int = 3,
    top_n: int = 5,
    show_grades: bool = True,
    disp_team_key: str = "team",
) -> tuple[list[dict], list[dict], str]:
    """
    Returns (graded, slips, formatted_text).
    Does NOT print — the bot handles Discord delivery.
    """
    graded = [grade_prop(dict(p)) for p in props]
    graded = limit_A_plays(graded, max_A=max_A)

    # ── FINAL CONSISTENCY CHECK ──────────────────────────────────────
    # Walks every graded play and normalises its status fields so
    # downstream consumers (slip builder, display, exports) never see
    # drift between grade / decision / value / sizing flags.
    #
    # Rule order matters:
    #   1. Weak grade (C/D/F) → cap value at LOW, half size
    #   2. NO BET           → zero size, value = NO PLAY, normalised
    #                          final_call + weighted-score label
    #   3. Hard block       → STRONG VALUE is reserved for grade A;
    #                          any STRONG on a non-A play is demoted
    #                          to MODERATE so slips can never carry a
    #                          "STRONG" leg below A grade.
    for p in graded:
        # 0. CAN_BET (soft eligibility flag) ────────────────────────
        # Always set on the play as a diagnostic flag. SCORE DOES
        # NOT TRIGGER BETS — score is confidence/display only.
        # Eligibility = solid history AND both projections beat
        # the line.
        p["can_bet"] = (
            p["hit_rate"] >= 50
            and p["short_map_proj"]  > p["line"]
            and p["normal_map_proj"] > p["line"]
        )

        # 1. weak-grade cap
        if p["grade"] in ("C", "D", "F"):
            p["bet_size"] = 0.5
            p["value"]    = "LOW VALUE"

        # 2. NO BET normalisation
        if p["decision"] == "NO BET":
            p["bet_size"]             = 0
            p["value"]                = "NO PLAY"
            p["final_call"]           = "NO BET"
            p["weighted_score_label"] = "N/A"

        # 3. HARD BLOCK: no elite value on weak grades
        if p["grade"] != "A" and p.get("value") == "STRONG VALUE":
            p["value"] = "MODERATE VALUE"

        # 4. MASTER FINAL FILTER (HARD OVERRIDE) ────────────────────
        # Authoritative last word. Any single fragility signal
        # zeroes out the play regardless of what earlier rules set:
        # weak score (<55), weak history (<50), short-map fail, or
        # normal-map projection below the line.
        fail_conditions = (
            p["score"] < 55
            or p["hit_rate"] < 50
            or p["short_map_proj"] < p["line"]
            or p["normal_map_proj"] < p["line"]
        )
        if fail_conditions:
            p["final_label"] = "NO BET"
            p["bet_size"]    = 0
            p["confidence"]  = 0

        # 5. SCORE ONLY USED FOR CONFIDENCE ─────────────────────────
        # Score does not trigger bets — it's just a display-time
        # confidence signal. Null it on NO BET so it can never be
        # mistaken for justification of a play that didn't qualify.
        # .get() is used because final_label is only set by the
        # master filter above (plays that pass have no such field).
        if p.get("final_label") == "NO BET":
            p["score"] = None

    slips = build_slips(graded, sizes=sizes)
    text = format_for_discord(
        graded, slips,
        top_n=top_n,
        show_grades=show_grades,
        disp_team_key=disp_team_key,
    )
    return graded, slips, text
