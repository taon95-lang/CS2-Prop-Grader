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
  variance      ('low'|'medium'|'high')
  variance_num  (float, raw σ — used for adjusted_edge & coin-flip rule)
  avg           (float, projected mean for the stat)
  stomp         (bool)
  hit_rate      (0–100 percent float)
  short_map_proj (float, projected stat on shorter maps)

Edge handling:
  adjusted_edge = max(0, edge − variance_num × 0.5)
  All edge gates (NO BET threshold, A grade, slip-build) use the
  ADJUSTED edge so high-variance plays are penalised. The raw `edge`
  is preserved on the play dict for display.

Coin-flip-with-high-variance NO BET rule:
  variance_num ≥ 9 AND |avg − line| < 2  →  forced NO BET.
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

    # Adjusted edge — penalise high-variance plays. Used for ALL edge
    # gates below; raw edge stays on the dict for display.
    variance_num = float(p.get("variance_num", 6.0))
    adjusted_edge = max(0.0, float(p["edge"]) - variance_num * 0.5)

    # ── NO BET filter (uses ADJUSTED edge) ───────────────────────────
    if adjusted_edge < 6 or prob < 55:
        return {**p, "decision": "NO BET", "grade": "NO BET",
                "adjusted_edge": adjusted_edge}

    # NEW: coin-flip-with-high-variance trap
    #   variance_num ≥ 9 AND |avg − line| < 2  →  forced NO BET
    avg = float(p.get("avg", p["line"]))
    if variance_num >= 9 and abs(avg - float(p["line"])) < 2:
        return {**p, "decision": "NO BET", "grade": "NO BET",
                "adjusted_edge": adjusted_edge}

    # HARD FAIL: fragile overs (short-map fail + high variance OR stomp)
    if decision == "OVER" and short_fail and (p["variance"] == "high" or stomp):
        return {**p, "decision": "NO BET", "grade": "NO BET",
                "adjusted_edge": adjusted_edge}

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

    return {**p, "decision": decision, "grade": grade,
            "adjusted_edge": adjusted_edge}


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
            out.append(
                f"{tag} {p['player']} vs {p['opponent']} | "
                f"{p['decision']} {p['line']} | "
                f"{prob:.0f}% | Grade: {grade_label} | {edge_str}"
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
            out.append(
                f"  {tag} {leg['player']} ({team_show}) vs {leg['opponent']} | "
                f"{leg['decision']} {leg['line']} ({prob:.0f}%) | Grade {leg['grade']}"
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
    slips = build_slips(graded, sizes=sizes)
    text = format_for_discord(
        graded, slips,
        top_n=top_n,
        show_grades=show_grades,
        disp_team_key=disp_team_key,
    )
    return graded, slips, text
