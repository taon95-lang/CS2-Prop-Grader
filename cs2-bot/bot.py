import discord
from discord.ext import commands
from discord import app_commands
from typing import Literal
import os
import asyncio
import logging
from scraper import (
    get_player_info,
    get_player_info_fallback,
)
from deep_analysis import run_deep_analysis
from simulator import run_simulation
from keep_alive import keep_alive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DECISION_COLORS = {
    "OVER": 0x2ECC40,
    "UNDER": 0xFF4136,
    "PASS": 0xFFDC00,
    "MISPRICED": 0xFF851B,
}

GRADE_EMOJIS = {
    range(1, 4): "🔴",
    range(4, 6): "🟡",
    range(6, 8): "🟠",
    range(8, 11): "🟢",
}


def grade_emoji(grade_str: str) -> str:
    try:
        num = int(grade_str.split("/")[0])
        for r, emoji in GRADE_EMOJIS.items():
            if num in r:
                return emoji
    except Exception:
        pass
    return "⚪"


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")

    # Sync slash commands so /grade and /help appear in Discord
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash command(s) globally")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="CS2 props | !grade [Player] [Line] [Kills/HS]",
        )
    )


# ---------------------------------------------------------------------------
# Prefix command: !grade
# ---------------------------------------------------------------------------

@bot.command(name="grade")
async def grade_prop(ctx, player_name: str = None, line: str = None, stat_type: str = "Kills", opponent: str = None):
    if not player_name or not line:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description=(
                    "**Correct usage:**\n"
                    "`!grade [Player Name] [Line] [Kills/HS] [Opponent?]`\n\n"
                    "**Examples:**\n"
                    "`!grade ZywOo 38.5 Kills`\n"
                    "`!grade ZywOo 38.5 Kills NaVi`\n"
                    "`!grade s1mple 14.5 HS FaZe`"
                ),
                color=0xFF4136,
            )
        )
        return

    try:
        line_val = float(line)
    except ValueError:
        await ctx.send(f"❌ Invalid line `{line}`. Please provide a number, e.g. `38.5`.")
        return

    stat_type = stat_type.capitalize()
    if stat_type not in ("Kills", "Hs"):
        stat_type = "Kills"
    if stat_type == "Hs":
        stat_type = "HS"

    # Normalise opponent (strip quotes if the user wrapped it)
    if opponent:
        opponent = opponent.strip('"\'').strip()

    # Progress stages shown in the embed while the scraper and sim are running
    _STAGES_BASE = [
        "🔍 Searching HLTV for player profile...",
        "📡 Fetching recent match pages (Maps 1 & 2)...",
        "📡 Parsing per-map kill data from match stats...",
        "📊 Running 25,000 Monte Carlo simulations...",
        "⏳ Almost there — finalising results...",
    ]
    _STAGES_OPP = [
        "🔍 Searching HLTV for player profile...",
        "📡 Fetching recent match pages (Maps 1 & 2)...",
        f"🛡️ Analysing {opponent}'s defensive profile (10 matches)...",
        f"🔎 Pulling H2H records vs {opponent} + ranking comparison...",
        "📊 Applying deep adjustments + running 25,000 simulations...",
    ]
    _STAGES = _STAGES_OPP if opponent else _STAGES_BASE

    ETA = "~35s" if not opponent else "~45s"

    def _stage_embed(elapsed: int, stage_idx: int) -> discord.Embed:
        filled = min(10, elapsed // 4)
        bar = "▓" * filled + "░" * (10 - filled)
        matchup_line = f" vs **{opponent}**" if opponent else ""
        return discord.Embed(
            title="⚙️ Analyzing...",
            description=(
                f"**Player:** {player_name}{matchup_line}\n"
                f"**Prop:** {line_val} {stat_type}\n\n"
                f"{_STAGES[stage_idx]}\n\n"
                f"`[{bar}]` ⏱️ {elapsed}s elapsed  _(ETA {ETA} — please wait)_"
            ),
            color=0x7289DA,
        )

    # Send the initial embed immediately — bot is visibly "alive" from the start
    thinking_msg = await ctx.send(embed=_stage_embed(0, 0))

    # Launch the blocking analysis in a thread so the event loop stays free
    import time as _time
    loop = asyncio.get_running_loop()
    fut = asyncio.ensure_future(
        loop.run_in_executor(
            None,
            lambda: _analyze_player(player_name, line_val, stat_type, opponent),
        )
    )

    start = _time.monotonic()
    TOTAL_TIMEOUT = 150
    result = None
    logger.info(f"[grade] Started analysis: {player_name} {line_val} {stat_type} opp={opponent}")

    while True:
        elapsed = int(_time.monotonic() - start)

        if elapsed >= TOTAL_TIMEOUT:
            fut.cancel()
            logger.error(f"[grade] Timed out after {TOTAL_TIMEOUT}s for {player_name}")
            await thinking_msg.edit(
                embed=discord.Embed(
                    title="❌ Timed Out",
                    description=(
                        f"Analysis exceeded {TOTAL_TIMEOUT}s and was cancelled.\n"
                        "HLTV may be slow — try again in a moment."
                    ),
                    color=0xFF4136,
                )
            )
            return

        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=5)
            logger.info(f"[grade] Analysis finished in {elapsed}s for {player_name}")
            break
        except asyncio.TimeoutError:
            elapsed = int(_time.monotonic() - start)
            stage_idx = min(len(_STAGES) - 1, elapsed // 15)
            logger.info(f"[grade] Still running — {elapsed}s elapsed (stage {stage_idx})")
            try:
                await thinking_msg.edit(embed=_stage_embed(elapsed, stage_idx))
            except Exception as edit_err:
                logger.warning(f"[grade] Progress edit failed: {edit_err}")
        except Exception as e:
            logger.error(f"[grade] Executor error ({type(e).__name__}): {e}", exc_info=True)
            await thinking_msg.edit(
                embed=discord.Embed(
                    title="❌ Analysis Error",
                    description=f"**{type(e).__name__}**\n```{str(e)[:300]}```",
                    color=0xFF4136,
                )
            )
            return

    # Handle error dicts
    if result and ("sim_error" in result or "error" in result):
        err_msg = result.get("error") or result.get("sim_error", "Unknown error")
        logger.error(f"[grade] Result error for {player_name}: {err_msg}")
        await thinking_msg.edit(
            embed=discord.Embed(title="❌ Error", description=str(err_msg), color=0xFF4136)
        )
        return

    # Build and deliver the result embed
    logger.info(f"[grade] Building embed for {player_name}...")
    try:
        embed = build_result_embed(player_name, line_val, stat_type, result)
        # Delete the progress message and send a fresh one (triggers notification)
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await ctx.reply(embed=embed, mention_author=False)
        logger.info(f"[grade] ✅ Grade delivered — {player_name} {line_val} {stat_type} "
                    f"over={result.get('over_prob')}% grade={result.get('grade')}")
    except Exception as e:
        logger.error(f"[grade] Embed failed ({type(e).__name__}): {e}", exc_info=True)
        await ctx.send(
            embed=discord.Embed(
                title="❌ Display Error",
                description=f"Analysis done but display failed.\n```{type(e).__name__}: {str(e)[:300]}```",
                color=0xFF4136,
            )
        )


# ---------------------------------------------------------------------------
# Slash command: /help
# ---------------------------------------------------------------------------

@bot.tree.command(name="help", description="Show how to use the Elite CS2 Prop Grader")
async def help_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=False)
    embed = discord.Embed(
        title="🎮 Elite CS2 Prop Grader — Help",
        description="Analyze CS2 player props using HLTV data and Monte Carlo simulation.",
        color=0x7289DA,
    )
    embed.add_field(
        name="Command",
        value="`!grade [Player Name] [Line] [Kills/HS] [Opponent?]`",
        inline=False,
    )
    embed.add_field(
        name="Examples",
        value=(
            "`!grade ZywOo 38.5 Kills`\n"
            "`!grade ZywOo 38.5 Kills NaVi`\n"
            "`!grade s1mple 14.5 HS FaZe`\n"
            "`!grade NiKo 22.5 Kills Vitality`"
        ),
        inline=False,
    )
    embed.add_field(
        name="Opponent (optional)",
        value=(
            "Add the opposing team name to factor in their defensive profile.\n"
            "The bot fetches how many kills that team concedes per map and adjusts\n"
            "the simulation accordingly (±25% max)."
        ),
        inline=False,
    )
    embed.add_field(
        name="How it works",
        value=(
            "1️⃣ Fetches last 10 BO3 series (Maps 1 & 2 only) from HLTV\n"
            "2️⃣ (Optional) Fetches opponent's defensive kills-allowed profile\n"
            "3️⃣ Fits a Negative Binomial distribution to kill data\n"
            "4️⃣ Runs 25,000 Monte Carlo simulations\n"
            "5️⃣ Grades the prop on a 1-10 scale based on edge\n"
            "⚠️ If HLTV is unavailable, falls back to Estimated Stats automatically"
        ),
        inline=False,
    )
    embed.add_field(
        name="Grade Scale",
        value=(
            "🟢 **8-10** — Strong bet\n"
            "🟠 **6-7** — Lean / value play\n"
            "🟡 **4-5** — Marginal\n"
            "🔴 **1-3** — Pass"
        ),
        inline=False,
    )
    embed.set_footer(text="Data sourced from HLTV.org | Not financial advice")
    await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Core analysis (blocking — always run via executor)
# ---------------------------------------------------------------------------

def _analyze_player(
    player_name: str,
    line: float,
    stat_type: str,
    opponent: str | None = None,
) -> dict:
    """
    All blocking I/O and CPU work lives here.
    Run via loop.run_in_executor so the async event loop is never blocked.

    Scraping strategy:
      1. get_player_info() — searches HLTV, fetches accessible match pages,
         extracts per-map kills from matchstats HTML section.
      2. get_player_info_fallback() — seeded estimated stats if HLTV fails.
      3. If opponent provided, fetch their defensive profile and scale the
         kill distribution by (avg_kills_allowed / baseline).
    """
    internal_stat = "Kills" if stat_type in ("Kills", "kills") else "HS"

    # --- Step 1: Try live HLTV data ---
    map_stats = []
    data_source = "HLTV Live"
    used_fallback = False
    info: dict = {}  # populated on success; used for player_id / match_ids in step 3

    try:
        info = get_player_info(player_name, stat_type=internal_stat)
        map_stats = info["map_kills"]
        data_source = info["source"]
        logger.info(
            f"HLTV returned {len(map_stats)} map samples | "
            f"mean={info['mean']} std={info['std']} source={info['source']}"
        )
    except RuntimeError as e:
        logger.warning(f"HLTV scrape failed (RuntimeError): {e}")
        map_stats = []
    except Exception as e:
        logger.warning(f"HLTV scrape raised unexpected error ({type(e).__name__}): {e}")
        map_stats = []

    # --- Step 2: Fallback if not enough data ---
    if len(map_stats) < 4:
        logger.warning("Insufficient HLTV data — using Estimated Stats fallback")
        try:
            fallback = get_player_info_fallback(player_name, stat_type=internal_stat)
            map_stats = fallback["map_kills"]
            data_source = fallback["source"]
            used_fallback = True
        except Exception as e:
            logger.error(f"Fallback generator failed ({type(e).__name__}): {e}")
            return {"error": "Both HLTV and fallback data sources failed. Please try again."}

    if not map_stats:
        return {"error": "No data available. Check the player name spelling."}

    # --- Step 3 (optional): Deep opponent analysis ---
    deep: dict | None = None
    if opponent:
        player_id   = info.get("player_id")   if not used_fallback else None
        player_slug = info.get("player_slug")  if not used_fallback else None
        match_ids   = info.get("match_ids", []) if not used_fallback else []
        baseline_avg = info.get("mean", sum(m["stat_value"] for m in map_stats) / len(map_stats)) if map_stats else 0

        if player_id and player_slug:
            try:
                deep = run_deep_analysis(
                    player_id=player_id,
                    player_slug=player_slug,
                    player_match_ids=match_ids,
                    opponent_name=opponent,
                    stat_type=stat_type,
                    baseline_avg=baseline_avg,
                    line=line,
                )
                if deep and not deep.get("error"):
                    adj = deep["combined_multiplier"]
                    map_stats = [{**m, "stat_value": round(m["stat_value"] * adj, 2)} for m in map_stats]
                    total_pct = round((adj - 1) * 100, 1)
                    sign = "+" if total_pct >= 0 else ""
                    logger.info(
                        f"Deep analysis for '{opponent}': "
                        f"combined multiplier ×{adj} ({sign}{total_pct}%)"
                    )
                else:
                    logger.warning(
                        f"Deep analysis returned error for '{opponent}': "
                        f"{deep.get('error') if deep else 'None'}"
                    )
            except Exception as e:
                logger.warning(f"Deep analysis failed ({type(e).__name__}): {e}")
                deep = None
        else:
            logger.warning("Cannot run deep analysis without player_id (fallback data in use)")

    # --- Step 4: Monte Carlo simulation ---
    favorite_prob = 0.55
    likely_maps: list = []
    rank_gap: int | None = None
    if deep:
        mp = deep.get("map_pool", {})
        likely_maps = mp.get("most_played", []) or []
        rank_gap = deep.get("rank_info", {}).get("rank_gap")
    try:
        sim_result = run_simulation(
            map_stats=map_stats,
            line=line,
            stat_type=stat_type,
            favorite_prob=favorite_prob,
            likely_maps=likely_maps if likely_maps else None,
            rank_gap=rank_gap,
        )
    except Exception as e:
        err_name = type(e).__name__
        logger.error(f"Simulation failed ({err_name}): {e}")
        return {"sim_error": err_name, "error": str(e)[:300]}

    sim_result["data_source"] = data_source
    sim_result["used_fallback"] = used_fallback
    sim_result["player_name"] = player_name
    sim_result["line"] = line
    sim_result["deep"] = deep

    # If using estimated fallback data — override to PASS, never make directional calls
    # on invented stats. The grade stays for context but direction is unreliable.
    if used_fallback:
        sim_result["decision"] = "PASS"
        sim_result["recommendation"] = "⚠️ PASS — Using estimated stats (HLTV unavailable)"
        sim_result["grade"] = "N/A"
        logger.info(f"[grade] Fallback data → forced PASS for {player_name}")

    # Apply +5% Over bonus for confirmed matchup favorites
    if deep and deep.get("matchup_favorite_bonus"):
        over_p  = sim_result.get("over_prob",  50)
        under_p = sim_result.get("under_prob", 50)
        push_p  = sim_result.get("push_prob",   0)
        sim_result["over_prob"]  = min(95, over_p + 5)
        sim_result["under_prob"] = max(5,  under_p - 5)
        sim_result["push_prob"]  = push_p

    # Apply Economy Impact probability adjustment
    if deep:
        economy_delta = deep.get("economy_prob_delta", 0.0)
        if economy_delta:
            over_p  = sim_result.get("over_prob",  50)
            under_p = sim_result.get("under_prob", 50)
            sim_result["over_prob"]  = round(min(95, max(5, over_p + economy_delta)), 1)
            sim_result["under_prob"] = round(min(95, max(5, under_p - economy_delta)), 1)
            sim_result["economy_adjusted"] = True
        # Recalculate edge if present
        if "edge" in sim_result:
            sim_result["edge"] = round(sim_result["over_prob"] - 50, 1)

    # --- Step 5: Impact Profile (from Rating 2.0 scraped off match scorecards) ---
    ratings = [m["rating"] for m in map_stats if m.get("rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    if avg_rating is not None:
        if avg_rating >= 1.20:
            impact_label = "🏆 Clutch Performer"
            # Boost Over slightly in coinflip matches where skill floors matter
            if favorite_prob <= 0.58:
                over_p = sim_result.get("over_prob", 50)
                sim_result["over_prob"]  = round(min(95, over_p + 2.5), 1)
                sim_result["under_prob"] = round(max(5, sim_result.get("under_prob", 50) - 2.5), 1)
                sim_result["impact_adjusted"] = True
        elif avg_rating < 0.95:
            impact_label = "🚪 Exit Fragger"
            impact_note  = "⚠️ Low rating despite kills — caution on Over"
        else:
            impact_label = "📊 Standard Entry"
        if avg_rating >= 1.20:
            impact_note = f"Rating {avg_rating} → boosted Over (+2.5% in coinflip)" if favorite_prob <= 0.58 else f"Rating {avg_rating}"
        elif avg_rating < 0.95:
            impact_note = f"Rating {avg_rating} → dying early / exits"
        else:
            impact_note = f"Rating {avg_rating}"
    else:
        impact_label = "❓ No Rating Data"
        impact_note  = "_Rating not available on these match pages_"

    sim_result["impact_label"]  = impact_label
    sim_result["impact_note"]   = impact_note
    sim_result["avg_rating"]    = avg_rating

    # --- Opening Duel Profile (from FK/FD scraped off match scorecards) ---
    fk_vals = [m["fk"] for m in map_stats if m.get("fk") is not None]
    fd_vals = [m["fd"] for m in map_stats if m.get("fd") is not None]

    if len(fk_vals) >= 4 and len(fd_vals) >= 4:
        avg_fk = sum(fk_vals) / len(fk_vals)
        avg_fd = sum(fd_vals) / len(fd_vals)
        rounds_used = 22  # default rounds per map
        fk_attempt_rate = avg_fk / rounds_used
        total_duels = avg_fk + avg_fd
        fk_success_rate = avg_fk / total_duels if total_duels > 0 else 0

        if fk_attempt_rate > 0.25 and fk_success_rate > 0.55:
            duel_label = "🎯 High-Impact Opener"
            duel_note  = f"FK rate {fk_attempt_rate:.0%} | Win rate {fk_success_rate:.0%} → high ceiling"
        elif fk_attempt_rate > 0.25 and fk_success_rate < 0.45:
            duel_label = "⚡ Volatility Risk"
            duel_note  = f"FK rate {fk_attempt_rate:.0%} | Win rate {fk_success_rate:.0%} → dies early on bad maps"
        elif fk_attempt_rate <= 0.10:
            duel_label = "🤫 Passive / Support"
            duel_note  = f"FK rate {fk_attempt_rate:.0%} | Win rate {fk_success_rate:.0%} → accumulates late kills"
        else:
            duel_label = "📋 Standard Fragger"
            duel_note  = f"FK rate {fk_attempt_rate:.0%} | Win rate {fk_success_rate:.0%}"
    else:
        duel_label = "❓ No Opening Duel Data"
        duel_note  = "_FK/FD not available on these match pages_"
        avg_fk = avg_fd = None

    sim_result["duel_label"]  = duel_label
    sim_result["duel_note"]   = duel_note
    sim_result["avg_fk"]      = round(avg_fk, 1) if avg_fk is not None else None
    sim_result["avg_fd"]      = round(avg_fd, 1) if avg_fd is not None else None

    # Stomp + high line warning
    if rank_gap is not None and rank_gap > 50 and line > 35.5:
        sim_result["stomp_high_line_warning"] = True

    return sim_result


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------

def build_result_embed(
    player_name: str, line: float, stat_type: str, result: dict
) -> discord.Embed:
    decision = result.get("decision", "PASS")
    color = DECISION_COLORS.get(decision, 0x7289DA)
    grade = result.get("grade", "N/A")
    emoji = grade_emoji(grade)

    deep = result.get("deep")
    opp_display = deep["opponent_display"] if deep and deep.get("opponent_display") else None

    title = f"{emoji} CS2 Prop Grade — {player_name}"
    if opp_display:
        title += f" vs {opp_display}"

    description = (
        f"**Prop:** `{line} {stat_type}` ({'Over' if decision == 'OVER' else 'Under' if decision == 'UNDER' else decision})\n"
        f"**Data:** {result.get('data_source', 'HLTV')}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

    # ── Deep Opponent Analysis (only when opponent was provided) ──────────────
    if deep and not deep.get("error") and opp_display:
        combined     = deep.get("combined_multiplier", 1.0)
        total_pct    = round((combined - 1) * 100, 1)
        total_sign   = "+" if total_pct >= 0 else ""
        components   = deep.get("components", {})

        def fmt_comp(key, label):
            v = components.get(key)
            if v is None:
                return None
            p = round((v - 1) * 100, 1)
            s = "+" if p >= 0 else ""
            return f"`{label}` {s}{p}%"

        comp_parts = list(filter(None, [
            fmt_comp("defensive",       "Defense"),
            fmt_comp("t_side",          "T-Side"),
            fmt_comp("rank",            "Ranking"),
            fmt_comp("map_pool",        "Map Pool"),
            fmt_comp("h2h",             "H2H"),
            fmt_comp("star_clamp",      "Role Clamp"),
            fmt_comp("hs_vulnerability","HS Vuln"),
        ]))

        # Summary section
        bullets = deep.get("summary_bullets", [])
        bullet_text = "\n".join(f"• {b}" for b in bullets[:4]) if bullets else "_No significant factors detected_"

        embed.add_field(
            name=f"🔬 Deep Opponent Analysis — {opp_display}",
            value=(
                f"**Combined Adjustment:** `{total_sign}{total_pct}%`\n"
                f"{chr(10).join(comp_parts) if comp_parts else '_No component adjustments_'}\n\n"
                f"**Key Factors:**\n{bullet_text}"
            ),
            inline=False,
        )

        # Defensive Profile
        def_profile = deep.get("defensive_profile", {})
        rank_info   = deep.get("rank_info", {})
        map_pool    = deep.get("map_pool", {})

        def_val_parts = [f"**{def_profile.get('label','?')}**"]
        if def_profile.get("avg_kills_allowed"):
            def_val_parts.append(f"Avg kills allowed/map: `{def_profile['avg_kills_allowed']}`")
        if def_profile.get("ct_win_pct") is not None:
            def_val_parts.append(f"CT win %: `{def_profile['ct_win_pct']}%`")
        if def_profile.get("t_win_pct") is not None:
            def_val_parts.append(f"T win %: `{def_profile['t_win_pct']}%`")
        if def_profile.get("sample"):
            def_val_parts.append(f"Sample: {def_profile['sample']} player-maps")

        embed.add_field(
            name="🛡️ Defensive Profile",
            value="\n".join(def_val_parts) or "No data",
            inline=True,
        )

        # Ranking & Map Pool
        rank_label    = rank_info.get("label", "Unknown")
        stomp_text    = " ⚠️" if rank_info.get("stomp_risk") else ""
        most_played   = map_pool.get("most_played", [])
        permaban_hint = map_pool.get("permaban_hint")

        map_val = f"**{map_pool.get('label','?')}**"
        if most_played:
            map_val += f"\nMost Played: `{', '.join(m.title() for m in most_played[:3])}`"
        if permaban_hint:
            map_val += f"\nPermaban Hint: `{permaban_hint.title()}`"

        embed.add_field(
            name="🗺️ Ranking & Map Pool",
            value=f"**{rank_label}{stomp_text}**\n{map_val}",
            inline=True,
        )

        # H2H
        h2h_records = deep.get("h2h", [])
        h2h_label   = deep.get("h2h_label", "No H2H data")
        if h2h_records:
            h2h_lines = []
            for i, rec in enumerate(h2h_records, 1):
                maps_str = " / ".join(str(k) for k in rec.get("kills_by_map", []))
                h2h_lines.append(f"Match {i}: `{maps_str}` kills → avg `{rec['avg_kills']}`")
            h2h_val = f"{h2h_label}\n" + "\n".join(h2h_lines)
        else:
            h2h_val = h2h_label

        embed.add_field(
            name="🆚 Head-to-Head (Last 3)",
            value=h2h_val,
            inline=False,
        )

        # ── 🛡️ Opponent Scouting ──────────────────────────────────────────────
        scouting = deep.get("scouting", {})
        hs_sc   = scouting.get("hs_vulnerability", {})
        role_sc = scouting.get("role_suppression", {})
        h2h_sc  = scouting.get("h2h_line", {})

        hs_line = hs_sc.get("rating") or "❓ No Data"
        if hs_sc.get("pct_proxy"):
            hs_line += f"  `({hs_sc['pct_proxy']})`"

        role_line = role_sc.get("label") or "❓ No Data"

        cleared   = h2h_sc.get("matches_cleared", 0)
        of_n      = h2h_sc.get("of_n", 0)
        fav       = h2h_sc.get("matchup_favorite", False)
        if of_n > 0:
            h2h_line_txt = f"`{cleared}/{of_n}` H2H matches cleared `{line}` line"
            if fav:
                h2h_line_txt += "  ✅ **+5% Over bonus applied**"
        else:
            h2h_line_txt = "_No H2H data for line check_"

        econ_sc  = scouting.get("economy_impact", {})
        econ_line = econ_sc.get("label") or "⚖️ No Economy Data"

        embed.add_field(
            name="🛡️ Opponent Scouting",
            value=(
                f"**HS Vulnerability:** {hs_line}\n"
                f"**Role Suppression:** {role_line}\n"
                f"**H2H vs Line:** {h2h_line_txt}\n"
                f"**Economy Impact:** {econ_line}"
            ),
            inline=False,
        )

    n_series = result.get("n_series", 0)
    n_maps = result.get("n_samples", 0)
    embed.add_field(
        name="📋 Recent Sample",
        value=(
            f"**Series:** Last {n_series} BO3 (Maps 1 & 2 only)\n"
            f"**Total Maps:** {n_maps}"
        ),
        inline=False,
    )

    # ── 🔥 Recent Form ────────────────────────────────────────────────────────
    trend_label      = result.get("trend_label", "➡️ Neutral")
    recent_avg_kills = result.get("recent_avg_kills", "N/A")
    recent_n_maps    = result.get("recent_n_maps", 4)
    trend_pct        = result.get("trend_pct", 0)
    hist_avg_kills   = round(result.get("hist_avg", 0) / 2, 1)  # per-map from series total
    embed.add_field(
        name="🔥 Recent Form",
        value=(
            f"**Trend:** {trend_label}\n"
            f"**Recent avg (last {recent_n_maps} maps):** `{recent_avg_kills}` kills/map\n"
            f"**Overall avg per map:** `{hist_avg_kills}` kills/map\n"
            f"_Simulation is 60% weighted to recent form_"
        ),
        inline=False,
    )

    embed.add_field(
        name="📊 Historical Stats (2-map totals)",
        value=(
            f"**Average:** `{result.get('hist_avg', 'N/A')}`\n"
            f"**Median:** `{result.get('hist_median', 'N/A')}`\n"
            f"**Hit Rate vs {line}:** `{result.get('hit_rate', 'N/A')}%`"
        ),
        inline=True,
    )

    map_note = result.get("map_projection_note", "Overall average")
    embed.add_field(
        name="🎯 Projection",
        value=(
            f"**Rounds/Map:** `{result.get('rounds_per_map', 22)}`\n"
            f"**Total Rounds:** `{result.get('total_projected_rounds', 44)}`\n"
            f"**Expected {stat_type}:** `{result.get('expected_total', 'N/A')}`\n"
            f"**Basis:** {map_note}\n"
            f"**Context:** {result.get('match_context', 'Standard')}"
        ),
        inline=True,
    )

    embed.add_field(
        name=f"🎲 Monte Carlo ({result.get('n_simulations', 25000):,} runs)",
        value=(
            f"**Sim Mean:** `{result.get('sim_mean', 'N/A')}`\n"
            f"**Std Dev:** `±{result.get('sim_std', 'N/A')}`\n"
            f"**Sim Median:** `{result.get('sim_median', 'N/A')}`\n"
            f"**Model:** Negative Binomial"
        ),
        inline=True,
    )

    over_p = result.get("over_prob", 0)
    under_p = result.get("under_prob", 0)
    econ_adj_tag = "  _(economy-adjusted)_" if result.get("economy_adjusted") else ""
    over_bar = build_prob_bar(over_p / 100)
    embed.add_field(
        name="📈 Simulated Probabilities",
        value=(
            f"**Over {line}:** `{over_p}%`{econ_adj_tag} {over_bar}\n"
            f"**Under {line}:** `{under_p}%`\n"
            f"**Push:** `{result.get('push_prob', 0)}%`"
        ),
        inline=False,
    )

    # ── 📉 Stability Score ────────────────────────────────────────────────────
    stab_std   = result.get("stability_std", 0)
    stab_label = result.get("stability_label", "🎯 Consistent")
    ceiling    = result.get("ceiling", "N/A")
    floor_v    = result.get("floor", "N/A")
    map_kpr    = result.get("map_kpr", {})
    map_kpr_lines = "\n".join(
        f"  `{mn.title()}`: {v} KPR" for mn, v in sorted(map_kpr.items())
    ) if map_kpr else "  _No map-specific data_"
    embed.add_field(
        name="📉 Stability Score",
        value=(
            f"**Std Dev (series):** `±{stab_std}` kills  —  {stab_label}\n"
            f"**Ceiling:** `{ceiling}` kills  |  **Floor:** `{floor_v}` kills\n"
            f"**Per-Map KPR:**\n{map_kpr_lines}"
        ),
        inline=False,
    )

    # ── ⚔️ Impact & Opening Duel Profile ─────────────────────────────────────
    impact_label = result.get("impact_label", "❓ No Rating Data")
    impact_note  = result.get("impact_note", "")
    duel_label   = result.get("duel_label",  "❓ No Opening Duel Data")
    duel_note    = result.get("duel_note",   "")
    avg_fk       = result.get("avg_fk")
    avg_fd       = result.get("avg_fd")
    avg_rating   = result.get("avg_rating")

    impact_adj_tag = "  _(impact-adjusted)_" if result.get("impact_adjusted") else ""
    fk_stat_line = f"Avg FK: `{avg_fk}` / FD: `{avg_fd}`" if avg_fk is not None else ""

    stomp_warning = ""
    if result.get("stomp_high_line_warning"):
        stomp_warning = "\n⛔ **STOMP RISK + HIGH LINE — Lean UNDER**"

    embed.add_field(
        name="⚔️ Impact & Opening Duel",
        value=(
            f"**Impact Profile:** {impact_label}{impact_adj_tag}\n"
            f"_{impact_note}_\n"
            f"**Opening Duels:** {duel_label}\n"
            f"_{duel_note}_"
            + (f"\n{fk_stat_line}" if fk_stat_line else "")
            + stomp_warning
        ),
        inline=False,
    )

    # ── 💰 Fair Line Analysis ──────────────────────────────────────────────────
    fair_line_val  = result.get("fair_line", result.get("sim_median", "N/A"))
    misprice_label = result.get("misprice_label", "✅ Fair Line")
    embed.add_field(
        name="💰 Fair Line Analysis",
        value=(
            f"**Fair Line (50/50):** `{fair_line_val}` kills\n"
            f"**Sportsbook Line:** `{line}` kills\n"
            f"**Assessment:** {misprice_label}"
        ),
        inline=False,
    )

    edge_val = result.get("edge", 0)
    edge_sign = "+" if edge_val >= 0 else ""
    embed.add_field(
        name="💡 Edge & Verdict",
        value=(
            f"**Edge vs Line:** `{edge_sign}{edge_val}%` (vs 50% implied)\n"
            f"**Grade:** `{grade}` {emoji}\n"
            f"**Decision:** `{decision}`"
        ),
        inline=True,
    )

    rec = result.get("recommendation", "N/A")
    embed.add_field(
        name="✅ Recommendation",
        value=f"**{rec}**",
        inline=True,
    )

    embed.set_footer(
        text=(
            "Elite CS2 Prop Grader | Negative Binomial Model | "
            "Data: HLTV (Last 10 BO3, Maps 1-2 only) | "
            "Not financial advice."
        )
    )
    return embed


def build_prob_bar(prob: float, length: int = 10) -> str:
    filled = round(prob * length)
    return "█" * filled + "░" * (length - filled)


# ---------------------------------------------------------------------------
# Error handler for prefix commands
# ---------------------------------------------------------------------------

@bot.event
async def on_command_error(ctx, error):
    err_name = type(error).__name__
    logger.error(f"Prefix command error ({err_name}): {error}", exc_info=True)
    try:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Command Error",
                description=f"**{err_name}:** {str(error)[:300]}",
                color=0xFF4136,
            )
        )
    except Exception:
        pass


# Error handler for app commands
# ---------------------------------------------------------------------------

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    err_name = type(error).__name__
    logger.error(f"App command error ({err_name}): {error}")
    msg = f"**Simulation Error: {err_name}**\n```{str(error)[:300]}```"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                embed=discord.Embed(title="❌ Command Error", description=msg, color=0xFF4136)
            )
        else:
            await interaction.response.send_message(
                embed=discord.Embed(title="❌ Command Error", description=msg, color=0xFF4136),
                ephemeral=True,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN secret is not set!")

    keep_alive()
    logger.info("Starting Elite CS2 Prop Grader bot...")
    bot.run(token)
