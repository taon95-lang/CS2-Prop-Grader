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

Hard NO BET filters (in order):
  1. hit_rate < 40 AND variance == 'high'        → NO BET / NO VALUE
  2. adjusted_edge < 6 OR pick prob < 55         → NO BET
  3. var_score ≥ 9 AND |avg − line| < 2          → NO BET (coin-flip trap)
  4. OVER + short-map fail + (high var OR stomp) → NO BET (fragile over)

Value classification (independent of grade):
  STRONG VALUE   — adjusted_edge ≥ 10 AND hit_rate ≥ 60
  MODERATE VALUE — adjusted_edge ≥ 6
  LOW VALUE      — otherwise
  NO VALUE       — only set by hard-NO-BET filter (1) above

Elite/POTD block (post-classification):
  If ANY profile penalty triggered (hit_rate<50, short_map<line,
  stomp), the play cannot be POTD (potd=False) and any STRONG
  VALUE label is demoted to MODERATE — elite status requires
  a clean profile.

Mispriced flag (diagnostic):
  |normal_map_proj − line| ≥ 3 AND hit_rate ≥ 60
  Surfaces "the line looks soft" plays without changing grade.

Per-play status fields written by run_model's consistency pass
(applied in order; later rules override earlier ones):
  score        — 0–100 confidence on the chosen direction (= prob)
  potd         — Set by ELITE FILTER. True only if all five hold:
                   adjusted_edge ≥ 10
                   hit_rate     ≥ 65
                   short_map_proj  > line
                   normal_map_proj > line
                   score        ≥ 70
                 Else False.
  lock         — Set by LOCK rule. True only if potd AND
                 variance != 'high'. Else False.
  final_label  — Set to 'NO BET' by MASTER FINAL FILTER when any
                 fail condition holds (else unset).
  bet_size     — 0 when MASTER FINAL FILTER fires (also set to 0
                 by Rule 2 NO BET, 0.5 by Rule 1 weak grade).
  confidence   — 0 when MASTER FINAL FILTER fires (else unset).
  can_bet      — Soft eligibility flag (always set). True if
                 score ≥ 55 AND hit_rate ≥ 50 AND (short_map_proj
                 > line OR normal_map_proj > line). Looser than
                 the master filter (OR between projections, not
                 AND), so a play can have can_bet=True even when
                 the master filter fires.

MASTER FINAL FILTER (HARD OVERRIDE — runs last, authoritative):
  If score < 55 OR hit_rate < 50 OR short_map_proj < line
  OR normal_map_proj < line:
      final_label = 'NO BET', bet_size = 0, confidence = 0,
      potd = False, lock = False
"""

import itertools


# ─────────────────────────────────────────────────────────────────────
# Grade ONE prop
# ─────────────────────────────────────────────────────────────────────
def grade_prop(p: dict) -> dict:
    decision = "OVER" if p["over_prob"] > p["under_prob"] else "UNDER"
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
    # Canonical 0–100 confidence on the chosen direction. For now
    # equals `prob`; kept as a separate field so it can evolve into
    # a richer composite later without breaking downstream rules.
    score = prob

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

    # ── POTD / ELITE BLOCK ───────────────────────────────────────────
    # If ANY of the three profile penalties were triggered above, the
    # play cannot be Pick-Of-The-Day and any STRONG VALUE label is
    # demoted to MODERATE — elite status requires a clean profile.
    if (
        p["hit_rate"] < 50
        or p["short_map_proj"] < p["line"]
        or p.get("stomp")
    ):
        p["potd"] = False
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
    #   1. Weak grade (C/D/F) → cap value at LOW, half size, no POTD
    #   2. NO BET           → zero size, value = NO PLAY, normalised
    #                          final_call + weighted-score label
    #   3. Hard block       → STRONG VALUE is reserved for grade A;
    #                          any STRONG on a non-A play is demoted
    #                          to MODERATE so slips can never carry a
    #                          "STRONG" leg below A grade.
    for p in graded:
        # 1. weak-grade cap
        if p["grade"] in ("C", "D", "F"):
            p["potd"]     = False
            p["bet_size"] = 0.5
            p["value"]    = "LOW VALUE"

        # 2. NO BET normalisation
        if p["decision"] == "NO BET":
            p["potd"]                 = False
            p["bet_size"]             = 0
            p["value"]                = "NO PLAY"
            p["final_call"]           = "NO BET"
            p["weighted_score_label"] = "N/A"

        # 3. HARD BLOCK: no elite value on weak grades
        if p["grade"] != "A" and p.get("value") == "STRONG VALUE":
            p["value"] = "MODERATE VALUE"

        # 4. ELITE FILTER (very strict) ─────────────────────────────
        # POTD is granted ONLY if all five elite criteria hold; any
        # miss zeros it out. Re-evaluates potd from scratch.
        if (
            p["adjusted_edge"]   >= 10
            and p["hit_rate"]    >= 65
            and p["short_map_proj"]  > p["line"]
            and p["normal_map_proj"] > p["line"]
            and p["score"]       >= 70
        ):
            p["potd"] = True
        else:
            p["potd"] = False

        # 5. LOCK ───────────────────────────────────────────────────
        # Lock = POTD AND not high-variance. Variance can take a
        # POTD play down a tier even when every other signal aligns.
        if p["potd"] and p["variance"] != "high":
            p["lock"] = True
        else:
            p["lock"] = False

        # 6. MASTER FINAL FILTER (HARD OVERRIDE) ────────────────────
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
            p["potd"]        = False
            p["lock"]        = False

        # 7. CAN_BET (soft eligibility flag) ────────────────────────
        # Looser than the master filter — needs only ONE projection
        # above the line (OR), not both. Diagnostic field that flags
        # whether the play meets minimum betting criteria at all,
        # regardless of POTD / LOCK / master-filter results.
        p["can_bet"] = (
            p["score"]    >= 55
            and p["hit_rate"] >= 50
            and (
                p["short_map_proj"]  > p["line"]
                or p["normal_map_proj"] > p["line"]
            )
        )

    slips = build_slips(graded, sizes=sizes)
    text = format_for_discord(
        graded, slips,
        top_n=top_n,
        show_grades=show_grades,
        disp_team_key=disp_team_key,
    )
    return graded, slips, text
