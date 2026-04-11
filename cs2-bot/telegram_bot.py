"""
Elite CS2 Prop Grader — Telegram Bot
=====================================
Mirrors every command from the Discord bot (bot.py) but sends plain text
messages instead of Discord embeds. All scraping, simulation, and grading
logic is shared — imported directly from the same files.

Commands
--------
/grade  [Player] [Line] [kills/hs] [Team?] [vs Opponent?]
/scout  [Player]
/lines
/pp
/pphs
/ppkills
/ppstop
/result [Player] [Opponent]
/results
/fetchresults
/help
"""

import asyncio
import logging
import os
import re
import time as _time
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Shared analysis engine (same as Discord bot)
# ---------------------------------------------------------------------------
# Import the core analysis function and supporting constants from bot.py.
# _analyze_player() is pure data — it never touches Discord objects.
from bot import (
    _analyze_player,
    _KNOWN_AWPERS,
    _session_rank_gap,
)

from scraper import get_actual_result as _scraper_get_actual_result
from prizepicks import (
    get_cs2_lines,
    get_player_line,
    get_all_cs2_props,
    invalidate_cache as pp_invalidate,
)
from grades_db import (
    save_grade,
    record_result,
    get_entries_for_date,
    get_pending_entries,
    get_recent_entries,
    date_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Cancel flag for /ppstop
_pp_cancel: bool = False


# ---------------------------------------------------------------------------
# Text formatter — converts result dict → Telegram Markdown message
# ---------------------------------------------------------------------------

def _fmt_grade_message(player_name: str, line: float, stat_type: str, result: dict) -> str:
    """Build a Telegram-compatible Markdown string from a grade result dict."""
    decision   = result.get("decision", "PASS")
    used_fb    = result.get("used_fallback", False)
    is_lock    = result.get("is_lock", False)
    pkg        = result.get("grade_pkg") or {}
    deep       = result.get("deep") or {}

    form       = pkg.get("form", {})
    variance   = pkg.get("variance", {})
    confidence = pkg.get("confidence", result.get("confidence_score", 50))
    edge_pct   = pkg.get("edge_pct", result.get("edge", 0))

    opp_display = deep.get("opponent_display") if deep and not deep.get("error") else None
    opp_name    = result.get("opponent")
    opp_str     = f" vs *{opp_display or opp_name}*" if (opp_display or opp_name) else ""

    grade_str  = result.get("grade", "N/A")
    unit_rec   = result.get("unit_recommendation", "0u — Pass")
    hs_src     = result.get("hs_rate_src")
    fair_ln    = result.get("fair_line", result.get("sim_median", "N/A"))

    if   confidence >= 80: conf_grade = "A"
    elif confidence >= 65: conf_grade = "B"
    elif confidence >= 50: conf_grade = "C"
    elif confidence >= 35: conf_grade = "D"
    else:                  conf_grade = "F"

    conf_labels = {"A": "High", "B": "Moderate", "C": "Fair", "D": "Low", "F": "Unreliable"}
    conf_label  = conf_labels[conf_grade]

    if decision == "OVER":
        dec_icon = "✅"
        dec_word = "BET MORE (OVER)"
    elif decision == "UNDER":
        dec_icon = "❌"
        dec_word = "BET LESS (UNDER)"
    else:
        dec_icon = "⏸"
        dec_word = "NO BET (PASS)"

    lock_header = "🔒 *LOCK* — " if is_lock else ""

    # Core stats
    n_series  = result.get("n_series", 0) or 0
    hist_avg  = result.get("hist_avg", "N/A")
    hist_med  = result.get("hist_median", "N/A")
    hit_rate  = result.get("hit_rate", 0) or 0
    over_p    = result.get("over_prob", "N/A")
    under_p   = result.get("under_prob", "N/A")
    sim_mean  = result.get("sim_mean", "N/A")
    sim_p10   = result.get("sim_p10")
    sim_p90   = result.get("sim_p90")

    if n_series > 0 and isinstance(hit_rate, (int, float)):
        hits_n  = round(hit_rate / 100 * n_series)
        hit_str = f"{hits_n}/{n_series}"
    else:
        hit_str = f"{hit_rate}%"

    range_str = ""
    if sim_p10 is not None and sim_p90 is not None:
        range_str = f"\n• *Range (p10–p90):* `{sim_p10:.0f}–{sim_p90:.0f}`"

    # H2H and deep analysis summary
    deep_str = ""
    if deep and not deep.get("error"):
        comb_mult = deep.get("combined_multiplier", 1.0) or 1.0
        comb_pct  = round((comb_mult - 1) * 100, 1)
        sign      = "+" if comb_pct >= 0 else ""
        h2h_recs  = deep.get("h2h", [])
        h2h_n     = len([s for s in h2h_recs if not s.get("partial")])
        h2h_clrs  = sum(1 for s in h2h_recs if s.get("cleared") and not s.get("partial"))
        h2h_note  = f" · H2H {h2h_clrs}/{h2h_n}" if h2h_n else ""
        deep_str  = f"\n• *Matchup:* {sign}{comb_pct}%{h2h_note} vs *{opp_display or opp_name}*"

    # Series breakdown
    breakdown_parts = []
    for s in (result.get("series_breakdown") or [])[:10]:
        maps_str = ", ".join(s.get("per_map", []))
        breakdown_parts.append(f"  Series: {s['total']} ({maps_str})")
    breakdown_str = "\n".join(breakdown_parts)

    # Form streak
    form_label = form.get("label", "")
    var_label  = variance.get("label", "")
    var_std    = variance.get("std", "")

    # HS source note
    hs_note = f"\n_HS rate: {hs_src}_" if hs_src else ""

    # Fallback warning
    fb_warn = "\n⚠️ _Estimated data only — HLTV unavailable. No directional call._" if used_fb else ""

    SEP = "━━━━━━━━━━━━━━━━━━━━"

    msg = (
        f"🏆 *[GURU] GRADE & PROJECTIONS*\n"
        f"{SEP}\n"
        f"*Player:* {player_name}{opp_str}\n"
        f"*Prop:* Maps 1–2 {stat_type} | *Line:* `{line}`"
        f"{fb_warn}\n"
        f"{SEP}\n"
        f"{lock_header}{dec_icon} *{dec_word}*\n"
        f"*Grade:* `{grade_str}` · *Confidence:* `{conf_grade}` ({conf_label})\n"
        f"*Unit Rec:* {unit_rec}\n"
        f"*Fair Line:* `{fair_ln}`\n"
        f"{SEP}\n"
        f"📊 *Simulation ({n_series} series)*\n"
        f"• *OVER prob:* `{over_p}%`  |  *UNDER prob:* `{under_p}%`\n"
        f"• *Sim mean:* `{sim_mean}`{range_str}\n"
        f"{SEP}\n"
        f"📈 *Historical Stats*\n"
        f"• *Avg (last {n_series}):* `{hist_avg}`\n"
        f"• *Median:* `{hist_med}`\n"
        f"• *Hit rate:* `{hit_str}`\n"
        f"• *Form:* {form_label}\n"
        f"• *Variance:* {var_label} (σ={var_std})"
        f"{deep_str}\n"
    )

    if breakdown_str:
        msg += f"{SEP}\n📋 *Series Breakdown*\n{breakdown_str}\n"

    msg += f"{hs_note}\n_Data: HLTV.org · Not financial advice_"
    return msg


def _fmt_error(title: str, detail: str) -> str:
    return f"❌ *{title}*\n{detail}"


# ---------------------------------------------------------------------------
# Argument parser — mirrors the Discord bot's !grade argument parser
# ---------------------------------------------------------------------------

def _parse_grade_args(args: list[str]) -> dict:
    """
    Parse Telegram /grade arguments into a structured dict.
    Supports: player line [kills/hs] [odds] [team] [vs opponent]
    Returns: {player, line, stat_type, team_hint, opponent, book_odds_raw, book_implied}
    """
    if len(args) < 2:
        return {}

    player_name = args[0]
    line_raw    = args[1]
    remaining   = list(args[2:])

    stat_type = "Kills"
    if remaining and remaining[0].lower() in ("kills", "hs", "headshots"):
        tok = remaining.pop(0).lower()
        stat_type = "HS" if tok in ("hs", "headshots") else "Kills"

    def _parse_odds(token: str) -> Optional[float]:
        if not re.match(r'^[+-]\d{2,4}$', token):
            return None
        try:
            v = int(token)
            return 100 / (v + 100) if v > 0 else abs(v) / (abs(v) + 100)
        except ValueError:
            return None

    book_odds_raw = None
    book_implied  = 0.5238
    odds_idx = [i for i, a in enumerate(remaining) if _parse_odds(a) is not None]
    if odds_idx:
        book_odds_raw = remaining.pop(odds_idx[0])
        book_implied  = _parse_odds(book_odds_raw)

    vs_indices = [i for i, a in enumerate(remaining) if a.lower() == "vs"]
    team_hint = None
    opponent  = None
    if vs_indices:
        vi = vs_indices[0]
        team_hint = " ".join(remaining[:vi]).strip() or None
        opponent  = " ".join(remaining[vi + 1:]).strip() or None
    else:
        team_hint = " ".join(remaining).strip() or None

    try:
        line_val = float(line_raw)
    except ValueError:
        return {}

    return {
        "player":        player_name,
        "line":          line_val,
        "stat_type":     stat_type,
        "team_hint":     team_hint,
        "opponent":      opponent,
        "book_odds_raw": book_odds_raw,
        "book_implied":  book_implied,
    }


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "🎮 *Elite CS2 Prop Grader — Help*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*Grade a prop:*\n"
        "`/grade [Player] [Line] [kills/hs]`\n"
        "`/grade ZywOo 38.5 kills`\n"
        "`/grade ZywOo 38.5 kills vs NaVi`\n"
        "`/grade ZywOo 38.5 kills -115 vs NaVi`\n"
        "`/grade sandman 28.5 kills LAG vs Surge`\n\n"
        "*Auto-fetch PrizePicks line:*\n"
        "`/grade ZywOo` ← fetches live line automatically\n\n"
        "*PrizePicks bulk grading:*\n"
        "`/pp`      — grade all live CS2 kills props\n"
        "`/pphs`    — grade all live CS2 headshot props\n"
        "`/ppkills` — grade all live CS2 kills props\n"
        "`/ppstop`  — cancel a running /pp batch\n\n"
        "*Scouting:*\n"
        "`/scout ZywOo` — last 10 series raw kill data\n"
        "`/lines`       — all live PrizePicks CS2 lines\n\n"
        "*Results:*\n"
        "`/result ZywOo NaVi` — fetch actual result for a graded prop\n"
        "`/results`           — today's graded props\n"
        "`/fetchresults`      — auto-fetch all pending results\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*Grade Scale:*\n"
        "🟢 8-10 — Strong bet\n"
        "🟠 6-7  — Lean / value play\n"
        "🟡 4-5  — Marginal\n"
        "🔴 1-3  — Pass\n\n"
        "_Data: HLTV.org · Not financial advice_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /grade
# ---------------------------------------------------------------------------

async def cmd_grade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []

    if not args:
        await update.message.reply_text(
            _fmt_error(
                "Usage Error",
                "Usage: `/grade [Player] [Line] [kills/hs] [Team?] [vs Opponent?]`\n\n"
                "Examples:\n"
                "`/grade ZywOo 38.5 kills`\n"
                "`/grade ZywOo 38.5 kills vs NaVi`\n"
                "`/grade sandman 28.5 kills LAG vs Surge`\n"
                "`/grade ZywOo` ← auto-fetches PrizePicks line",
            ),
            parse_mode="Markdown",
        )
        return

    # Auto-fetch line when only player name is given
    if len(args) == 1:
        player_name = args[0]
        stat_type   = "Kills"
        thinking    = await update.message.reply_text(f"🔍 Looking up PrizePicks line for *{player_name}*...", parse_mode="Markdown")
        try:
            pp_item = await asyncio.to_thread(get_player_line, player_name, stat_type)
            if pp_item:
                raw_score = pp_item.get("line_score") or pp_item.get("line")
                if raw_score is not None:
                    args = [player_name, str(raw_score)]
                    logger.info(f"[tg_grade] Auto-fetched line {raw_score} for {player_name}")
        except Exception as e:
            logger.warning(f"[tg_grade] PrizePicks auto-fetch failed: {e}")

        if len(args) == 1:
            await thinking.edit_text(
                _fmt_error("No Line Found", f"No PrizePicks line found for *{player_name}*. Provide it manually: `/grade {player_name} 22.5`"),
                parse_mode="Markdown",
            )
            return

    parsed = _parse_grade_args(args)
    if not parsed:
        await update.message.reply_text(
            _fmt_error("Invalid Arguments", "Could not parse arguments. Example: `/grade ZywOo 38.5 kills`"),
            parse_mode="Markdown",
        )
        return

    player_name   = parsed["player"]
    line_val      = parsed["line"]
    stat_type     = parsed["stat_type"]
    team_hint     = parsed["team_hint"]
    opponent      = parsed["opponent"]
    book_odds_raw = parsed["book_odds_raw"]
    book_implied  = parsed["book_implied"]

    opp_note = f" vs *{opponent}*" if opponent else ""
    thinking = await update.message.reply_text(
        f"⚙️ Analysing *{player_name}*{opp_note} — `{line_val}` {stat_type}...\n"
        f"_This takes ~30–45s, please wait._",
        parse_mode="Markdown",
    )

    try:
        result = await asyncio.to_thread(
            _analyze_player,
            player_name,
            line_val,
            stat_type,
            opponent,
            team_hint,
            book_implied,
            book_odds_raw,
        )
    except Exception as e:
        logger.error(f"[tg_grade] Executor error: {e}", exc_info=True)
        await thinking.edit_text(
            _fmt_error("Analysis Error", f"`{type(e).__name__}: {str(e)[:300]}`"),
            parse_mode="Markdown",
        )
        return

    if result and ("sim_error" in result or "error" in result):
        err = result.get("error") or result.get("sim_error", "Unknown error")
        await thinking.edit_text(_fmt_error("Error", str(err)), parse_mode="Markdown")
        return

    msg = _fmt_grade_message(player_name, line_val, stat_type, result)
    await thinking.edit_text(msg, parse_mode="Markdown")

    # Save to grades history
    try:
        _baseline_mid = None
        _mk = result.get("map_kills") or []
        if _mk:
            _baseline_mid = str(_mk[0].get("match_id", ""))
        save_grade(player_name, line_val, stat_type, result, opponent=opponent, baseline_match_id=_baseline_mid)
    except Exception as e:
        logger.warning(f"[tg_grade] grades_db save failed: {e}")


# ---------------------------------------------------------------------------
# /scout
# ---------------------------------------------------------------------------

async def cmd_scout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text(
            _fmt_error("Usage", "Usage: `/scout [PlayerName]`\nExample: `/scout ZywOo`"),
            parse_mode="Markdown",
        )
        return

    player_name = " ".join(args)
    thinking = await update.message.reply_text(
        f"🔍 Scouting *{player_name}* — fetching last 10 BO3 series...",
        parse_mode="Markdown",
    )

    from scraper import get_player_info

    try:
        info = await asyncio.to_thread(get_player_info, player_name, "Kills")
    except Exception as e:
        await thinking.edit_text(_fmt_error("Scout Error", str(e)), parse_mode="Markdown")
        return

    map_stats = info.get("map_kills", [])
    if not map_stats:
        await thinking.edit_text(_fmt_error("No Data", f"No match data found for *{player_name}*."), parse_mode="Markdown")
        return

    # Group into series
    series: dict[str, list] = {}
    for m in map_stats:
        series.setdefault(m["match_id"], []).append(m)

    lines = [f"🔍 *Scout: {player_name}*\n_Last {len(series)} BO3 series (Maps 1 & 2)_\n━━━━━━━━━━━━━━━━━━━━"]
    for i, (mid, maps) in enumerate(series.items(), 1):
        total = sum(m["stat_value"] for m in maps)
        per   = " + ".join(str(round(m["stat_value"])) for m in maps)
        lines.append(f"Series {i}: `{total}` ({per}) — {mid}")

    lines.append(f"\n*Avg:* `{info.get('mean')}` · *Std:* `{info.get('std')}` · *Samples:* {info.get('sample_size')}")
    lines.append(f"_Source: {info.get('source')}_")

    await thinking.edit_text("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /lines
# ---------------------------------------------------------------------------

async def cmd_lines(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    thinking = await update.message.reply_text("📡 Fetching live PrizePicks CS2 lines...", parse_mode="Markdown")
    try:
        props = await asyncio.to_thread(get_all_cs2_props)
    except Exception as e:
        await thinking.edit_text(_fmt_error("PrizePicks Error", str(e)), parse_mode="Markdown")
        return

    if not props:
        await thinking.edit_text("📭 No CS2 props on PrizePicks right now.", parse_mode="Markdown")
        return

    lines = ["📋 *Live CS2 PrizePicks Lines*\n━━━━━━━━━━━━━━━━━━━━"]
    for p in props:
        name      = p.get("player_name", "?")
        line_val  = p.get("line_score") or p.get("line", "?")
        stat      = p.get("stat_type", "?")
        team      = p.get("team", "")
        team_str  = f" ({team})" if team else ""
        lines.append(f"• *{name}*{team_str} — `{line_val}` {stat}")

    await thinking.edit_text("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /pp, /ppkills, /pphs  (bulk PrizePicks grading)
# ---------------------------------------------------------------------------

async def _run_pp_batch(update: Update, stat_filter: str) -> None:
    global _pp_cancel
    _pp_cancel = False

    thinking = await update.message.reply_text(
        f"📡 Fetching PrizePicks CS2 {stat_filter} props...",
        parse_mode="Markdown",
    )

    try:
        props = await asyncio.to_thread(get_all_cs2_props)
    except Exception as e:
        await thinking.edit_text(_fmt_error("PrizePicks Error", str(e)), parse_mode="Markdown")
        return

    # Filter by stat type
    stat_lower = stat_filter.lower()
    if stat_lower in ("kills", "headshots", "hs"):
        if stat_lower == "kills":
            props = [p for p in props if "kill" in (p.get("stat_type") or "").lower()]
        else:
            props = [p for p in props if "head" in (p.get("stat_type") or "").lower()]

    if not props:
        await thinking.edit_text(f"📭 No CS2 {stat_filter} props found on PrizePicks.", parse_mode="Markdown")
        return

    await thinking.edit_text(
        f"✅ Found *{len(props)}* {stat_filter} props. Grading each one...\n"
        f"_Use /ppstop to cancel._",
        parse_mode="Markdown",
    )

    results = []
    for i, prop in enumerate(props, 1):
        if _pp_cancel:
            await update.message.reply_text("🛑 Batch grading cancelled by /ppstop.")
            return

        player_name = prop.get("player_name", "")
        line_val    = prop.get("line_score") or prop.get("line")
        team_hint   = prop.get("team") or None
        stat_type   = "HS" if "head" in (prop.get("stat_type") or "").lower() else "Kills"

        if not player_name or line_val is None:
            continue

        await update.message.reply_text(
            f"⚙️ [{i}/{len(props)}] Grading *{player_name}* — `{line_val}` {stat_type}...",
            parse_mode="Markdown",
        )

        try:
            result = await asyncio.to_thread(
                _analyze_player,
                player_name,
                float(line_val),
                stat_type,
                None,
                team_hint,
                0.5238,
                None,
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ *{player_name}* — Error: `{str(e)[:200]}`",
                parse_mode="Markdown",
            )
            continue

        if result and ("sim_error" in result or "error" in result):
            err = result.get("error") or result.get("sim_error", "?")
            await update.message.reply_text(
                f"❌ *{player_name}* — {err}",
                parse_mode="Markdown",
            )
            continue

        decision = result.get("decision", "PASS")
        grade    = result.get("grade", "N/A")
        over_p   = result.get("over_prob", "?")
        unit_rec = result.get("unit_recommendation", "0u")

        icon = "✅" if decision == "OVER" else ("❌" if decision == "UNDER" else "⏸")
        summary = (
            f"{icon} *{player_name}* `{line_val}` {stat_type}\n"
            f"  Grade: `{grade}` · OVER: `{over_p}%` · {unit_rec}"
        )
        await update.message.reply_text(summary, parse_mode="Markdown")
        results.append((player_name, line_val, stat_type, decision, grade))

        try:
            save_grade(player_name, float(line_val), stat_type, result)
        except Exception:
            pass

        await asyncio.sleep(1)

    # Summary
    overs  = [r for r in results if r[3] == "OVER"]
    unders = [r for r in results if r[3] == "UNDER"]
    passes = [r for r in results if r[3] == "PASS"]

    await update.message.reply_text(
        f"✅ *Batch complete — {len(results)} props graded*\n"
        f"• OVER: {len(overs)} · UNDER: {len(unders)} · PASS: {len(passes)}",
        parse_mode="Markdown",
    )


async def cmd_pp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_pp_batch(update, "kills")


async def cmd_ppkills(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_pp_batch(update, "kills")


async def cmd_pphs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_pp_batch(update, "headshots")


async def cmd_ppstop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _pp_cancel
    _pp_cancel = True
    await update.message.reply_text("🛑 Cancelling batch grading after current player...")


# ---------------------------------------------------------------------------
# /result
# ---------------------------------------------------------------------------

async def cmd_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text(
            _fmt_error("Usage", "Usage: `/result [Player] [Opponent]`\nExample: `/result ZywOo NaVi`"),
            parse_mode="Markdown",
        )
        return

    player_name = args[0]
    opponent    = " ".join(args[1:]) if len(args) > 1 else ""

    thinking = await update.message.reply_text(
        f"🔍 Looking up actual result for *{player_name}*...",
        parse_mode="Markdown",
    )

    # Find the most recent pending grade for this player
    try:
        pending = get_pending_entries()
    except Exception:
        pending = []

    match_entry = None
    for e in pending:
        if player_name.lower() in (e.get("player") or "").lower():
            match_entry = e
            break

    grade_ts       = match_entry["timestamp"] if match_entry else (_time.time() - 86400)
    line_val       = match_entry["line"]      if match_entry else 0
    baseline_mid   = match_entry.get("baseline_match_id") if match_entry else None
    opp_for_lookup = opponent or (match_entry.get("opponent") if match_entry else None) or ""

    try:
        res = await asyncio.to_thread(
            _scraper_get_actual_result,
            player_name,
            opp_for_lookup,
            grade_ts,
            line_val,
            baseline_mid,
        )
    except Exception as e:
        await thinking.edit_text(_fmt_error("Result Error", str(e)), parse_mode="Markdown")
        return

    if not res:
        await thinking.edit_text(
            f"📭 No new BO3 result found for *{player_name}*.\n"
            f"_Match may not have been played yet, or results aren't on HLTV._",
            parse_mode="Markdown",
        )
        return

    actual  = res["actual"]
    outcome = res["outcome"].upper()
    mid     = res["match_id"]
    icon    = "✅" if outcome == "OVER" else "❌"

    msg = (
        f"🎯 *Result: {player_name}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Actual (Maps 1+2):* `{actual}`\n"
        f"*Line:* `{line_val}`\n"
        f"*Outcome:* {icon} *{outcome}*\n"
        f"_Match ID: {mid}_"
    )
    await thinking.edit_text(msg, parse_mode="Markdown")

    if match_entry:
        try:
            record_result(match_entry["id"], actual, outcome.lower())
        except Exception as e:
            logger.warning(f"[tg_result] record_result failed: {e}")


# ---------------------------------------------------------------------------
# /results
# ---------------------------------------------------------------------------

async def cmd_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        entries = get_recent_entries(days=1)
    except Exception as e:
        await update.message.reply_text(_fmt_error("DB Error", str(e)), parse_mode="Markdown")
        return

    if not entries:
        await update.message.reply_text("📭 No graded props today.", parse_mode="Markdown")
        return

    lines = [f"📋 *Today's Graded Props*\n━━━━━━━━━━━━━━━━━━━━"]
    for e in entries:
        player  = e.get("player", "?")
        line    = e.get("line", "?")
        stat    = e.get("stat_type", "Kills")
        dec     = e.get("decision", "PASS")
        grade   = e.get("grade", "N/A")
        actual  = e.get("actual_result")
        outcome = e.get("outcome", "")

        dec_icon    = "✅" if dec == "OVER" else ("❌" if dec == "UNDER" else "⏸")
        result_str  = f" → `{actual}` {outcome.upper()}" if actual is not None else " _(pending)_"
        lines.append(f"{dec_icon} *{player}* `{line}` {stat} · Grade: `{grade}`{result_str}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /fetchresults
# ---------------------------------------------------------------------------

async def cmd_fetchresults(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    thinking = await update.message.reply_text(
        "🔄 Auto-fetching results for all pending props...",
        parse_mode="Markdown",
    )

    try:
        pending = get_pending_entries()
    except Exception as e:
        await thinking.edit_text(_fmt_error("DB Error", str(e)), parse_mode="Markdown")
        return

    if not pending:
        await thinking.edit_text("📭 No pending props to check.", parse_mode="Markdown")
        return

    await thinking.edit_text(
        f"🔄 Checking *{len(pending)}* pending prop(s)...",
        parse_mode="Markdown",
    )

    resolved = 0
    for entry in pending:
        player     = entry.get("player", "")
        line       = entry.get("line", 0)
        opponent   = entry.get("opponent") or ""
        grade_ts   = entry.get("timestamp", _time.time() - 86400)
        baseline   = entry.get("baseline_match_id")
        entry_id   = entry.get("id")

        try:
            res = await asyncio.to_thread(
                _scraper_get_actual_result,
                player, opponent, grade_ts, line, baseline,
            )
        except Exception as e:
            logger.warning(f"[fetchresults] {player}: {e}")
            continue

        if not res:
            continue

        actual  = res["actual"]
        outcome = res["outcome"]
        icon    = "✅" if outcome == "over" else "❌"

        await update.message.reply_text(
            f"{icon} *{player}* `{line}` → Actual: `{actual}` — *{outcome.upper()}*",
            parse_mode="Markdown",
        )

        if entry_id:
            try:
                record_result(entry_id, actual, outcome)
            except Exception as e:
                logger.warning(f"[fetchresults] record_result failed: {e}")

        resolved += 1
        await asyncio.sleep(0.5)

    summary = f"✅ *Done* — resolved *{resolved}/{len(pending)}* pending props."
    if resolved < len(pending):
        summary += f"\n_{len(pending) - resolved} not found yet (match may not be played)._"
    await update.message.reply_text(summary, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_TOKEN environment variable is not set. "
            "Add your Telegram bot token as a secret named TELEGRAM_TOKEN."
        )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("help",         cmd_help))
    app.add_handler(CommandHandler("grade",        cmd_grade))
    app.add_handler(CommandHandler("scout",        cmd_scout))
    app.add_handler(CommandHandler("lines",        cmd_lines))
    app.add_handler(CommandHandler("pp",           cmd_pp))
    app.add_handler(CommandHandler("ppkills",      cmd_ppkills))
    app.add_handler(CommandHandler("pphs",         cmd_pphs))
    app.add_handler(CommandHandler("ppstop",       cmd_ppstop))
    app.add_handler(CommandHandler("result",       cmd_result))
    app.add_handler(CommandHandler("results",      cmd_results))
    app.add_handler(CommandHandler("fetchresults", cmd_fetchresults))

    logger.info("Elite CS2 Prop Grader Telegram Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
