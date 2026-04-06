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
    get_player_hs_pct,
    get_player_period_stats,
    get_team_period_stats,
    _warm_hltv_session,
)
from deep_analysis import run_deep_analysis
from simulator import run_simulation
from keep_alive import keep_alive
from prizepicks import get_cs2_lines, get_player_line, get_all_cs2_props, invalidate_cache as pp_invalidate
from grade_engine import (
    compute_grade_package,
    run_lines_table,
    build_prob_bar as ge_prob_bar,
    determine_role,
)

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

# ---------------------------------------------------------------------------
# AWPer HS% database
# ---------------------------------------------------------------------------
# AWP kills can be body shots (no headshot needed), so AWPers have structurally
# lower HS% than riflers (riflers: 38-55%, AWPers: 15-28%).
# Values below are conservative typical rates sourced from career HLTV stats.
# If a player is listed here AND the scraped HS% is unrealistically high
# (> _AWPER_HS_CAP), the scraping result is treated as an artifact and this
# value is used instead.
_KNOWN_AWPERS: dict[str, float] = {
    # ── Pure AWPers ─────────────────────────────────────────────────────────
    # AWP body-shots push HS% far below rifler norms (typical 18-28%).
    # Override fires when scraped value is None OR above _AWPER_HS_CAP (55%).
    "sh1ro":       0.20,
    "zywoo":       0.26,
    "s1mple":      0.28,   # hybrid but still low for AWP role periods
    "device":      0.21,
    "jl":          0.24,
    "maden":       0.23,
    "snappi":      0.24,   # AWP support
    "mezii":       0.24,
    "floppyfish":  0.23,
    "syrson":      0.22,
    "headtr1ck":   0.24,
    "nitro":       0.24,
    "broky":       0.23,
    "azr":         0.25,
    "imorim":      0.22,
    "degster":     0.28,   # AWP / hybrid
    "nawwk":       0.22,
    "mantuu":      0.21,
    # ── Hybrid / occasional AWP ─────────────────────────────────────────────
    "ropz":        0.35,
    "electronic":  0.48,   # mostly rifles; measured ~47-50% on HLTV career
    # ── High-HS riflers ──────────────────────────────────────────────────────
    # These players have genuinely high HS% as aggressive rifles.  The known
    # rate prevents the 40% generic default from under-projecting HS props.
    # Override fires when scraped value is None OR above the cap (KAST misread).
    "fl1t":        0.50,   # C9 entry fragger — career ~48-52% on HLTV
    "ax1le":       0.47,   # C9 rifler — career ~45-49%
    "b1t":         0.50,   # NaVi rifler — career ~48-52%
    "buster":      0.50,   # aggressive entry — career ~48-52%
    "nafany":      0.46,   # entry fragger — career ~44-48%
    "xantares":    0.52,   # high-aggression rifler — career ~50-54%
    "frozen":      0.48,   # MC rifler — career ~46-50%
    "torzsi":      0.30,   # AWP/hybrid — career ~28-32%
    "donk":        0.53,   # star rifler — measured high HS rate
    "niko":        0.46,   # G2 star rifler — career ~44-48%
    "hunter-":     0.46,   # G2 rifler — career ~44-48%
    "monesy":      0.44,   # G2 rifler — career ~42-46%
    # ── Riflers: measured from real match data ───────────────────────────────
    "idisbalance": 0.36,   # rifler — measured 35.8% vs State (19 HS / 53 kills)
}

# HS% above this threshold for a known AWPer is almost certainly a scraping
# artefact (often KAST% misread as HS%).  Override with the role-based estimate.
# Set at 0.55 so genuine high-HS rifler readings are NOT blocked.
_AWPER_HS_CAP = 0.55

# Cancellation flag for !ppstop — set True to abort an in-progress !pp run
_pp_cancel: bool = False

# Session-level rank_gap cache keyed by opponent name (lower).  Reset at the
# start of every !pp run.  Ensures all players on the same team facing the same
# opponent receive the SAME stomp/OT match context even if individual HLTV rank
# fetches are flaky.
_session_rank_gap: dict[str, int | None] = {}

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

    # Pre-warm the persistent HLTV session so Cloudflare cookies are set
    # before the first user command arrives. Run in executor to avoid blocking.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _warm_hltv_session)

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
    if not player_name:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description=(
                    "**Correct usage:**\n"
                    "`!grade [Player Name] [Line] [Kills/HS] [Opponent?]`\n\n"
                    "**Examples:**\n"
                    "`!grade ZywOo 38.5 Kills`\n"
                    "`!grade ZywOo 38.5 Kills NaVi`\n"
                    "`!grade s1mple 14.5 HS FaZe`\n"
                    "`!grade ZywOo` ← auto-fetches live line from PrizePicks"
                ),
                color=0xFF4136,
            )
        )
        return

    stat_type = stat_type.capitalize()
    if stat_type not in ("Kills", "Hs"):
        stat_type = "Kills"
    if stat_type == "Hs":
        stat_type = "HS"

    # If no line provided, try to pull it from PrizePicks live board
    pp_line_fetched = False
    if not line:
        try:
            pp_item = await asyncio.to_thread(get_player_line, player_name, stat_type)
            if pp_item:
                raw_score = pp_item.get("line_score") or pp_item.get("line")
                if raw_score is not None:
                    line = str(raw_score)
                    pp_line_fetched = True
                    logger.info(f"[grade] Auto-fetched PrizePicks line {line} for {player_name}")
        except Exception as _pp_exc:
            logger.warning(f"[grade] PrizePicks auto-fetch failed: {_pp_exc}")

    if not line:
        await ctx.send(
            embed=discord.Embed(
                title="❌ No Line Found",
                description=(
                    f"No line provided and no PrizePicks line found for **{player_name}** ({stat_type}).\n"
                    "Please provide a line manually: `!grade ZywOo 38.5 Kills`"
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
        pp_note = " _(line auto-fetched from PrizePicks)_" if pp_line_fetched else ""
        return discord.Embed(
            title="⚙️ Analyzing...",
            description=(
                f"**Player:** {player_name}{matchup_line}\n"
                f"**Prop:** {line_val} {stat_type}{pp_note}\n\n"
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
    player_team_hint: str | None = None,
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
        info = get_player_info(player_name, stat_type=internal_stat, team_hint=player_team_hint)
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

    # --- Step 2.5: Period stats from HLTV stats page (last 90 days) ---
    # Always fetched when we have real HLTV data (not fallback).
    # Provides: KPR, HS%, Rating 2.0, KAST, ADR — all date-filtered.
    period_stats: dict | None = None
    if not used_fallback:
        _pid_ps   = info.get("player_id")
        _pslug_ps = info.get("player_slug")
        if _pid_ps and _pslug_ps:
            try:
                period_stats = get_player_period_stats(_pid_ps, _pslug_ps, days=90)
            except Exception as _e:
                logger.warning(f"[period_stats] Fetch failed: {_e}")

    # --- Step 2.6: HS% Scaling (only for HS props) ---
    # The scraper always returns kill data. For HS props we convert kills → HS
    # using the best available HS rate in priority order:
    #   P0: Period stats page HS% (90-day aggregate, most accurate & date-filtered)
    #   P1: recent HS% averaged from match-level overview rows (~10 matches)
    #   P2: career HS% from /player/{id}/{slug} profile page
    #   P3: AWPer known-rate override (if player is in calibrated table)
    #   P4: default 45% (rifler estimate)
    hs_rate      = None
    hs_rate_src  = None
    is_awper     = False
    awper_warn   = False   # True when AWPer override fires

    pslug = info.get("player_slug", "").lower() if not used_fallback else ""

    if stat_type == "HS" and not used_fallback:
        n_hs_matches = info.get("hs_pct_n_matches", 0)

        # Priority 0: period stats page HS% (90-day aggregate, most reliable)
        if period_stats and period_stats.get("hs_pct") is not None:
            _ps_hs  = period_stats["hs_pct"] / 100.0
            hs_rate     = _ps_hs
            hs_rate_src = f"HLTV stats page 90d ({round(period_stats['hs_pct'])}%)"
            logger.info(f"[hs_scale] Using period stats HS%: {hs_rate_src}")

        # Priority 1: recent HS% derived from actual match pages (last ~10 matches)
        if hs_rate is None:
            recent_hs = info.get("recent_hs_pct")
            if recent_hs is not None:
                hs_rate     = recent_hs
                hs_rate_src = (
                    f"last {n_hs_matches} matches avg "
                    f"({round(recent_hs * 100)}%)"
                )
                logger.info(f"[hs_scale] Using recent match HS%: {hs_rate_src}")

        # Priority 2: career HS% from bo3.gg public API (always accessible)
        # HLTV player profile pages don't expose HS% in their HTML, so the
        # previous get_player_hs_pct() call always returned None.  bo3.gg's
        # public accuracy API is a reliable replacement.
        if hs_rate is None:
            _pslug = info.get("player_slug")
            if _pslug:
                try:
                    from bo3_scraper import get_career_hs_pct as _bo3_career_hs_bot
                    profile_rate = _bo3_career_hs_bot(_pslug)
                    if profile_rate is not None:
                        hs_rate     = profile_rate
                        hs_rate_src = f"bo3.gg career avg ({round(profile_rate * 100)}%)"
                        logger.info(
                            f"[hs_scale] bo3.gg career HS% for {_pslug}: "
                            f"{round(profile_rate * 100, 1)}%"
                        )
                except Exception as _e:
                    logger.warning(f"bo3.gg HS% lookup failed ({type(_e).__name__}): {_e}")

    if stat_type == "HS":
        # --- Known-rate override ---
        # HLTV's detailed stats (K/hs column) are JavaScript-rendered and not
        # available in static HTML.  Per-match HS% scraping often returns KAST%
        # (~65-70%) by mistake.  For players in _KNOWN_AWPERS we use a
        # pre-measured rate whenever the scraped value is above _AWPER_HS_CAP.
        # This covers both true AWPers (low HS%) and riflers whose accurate rate
        # has been manually verified from real match data.
        if pslug in _KNOWN_AWPERS:
            is_awper = True
            awper_known_rate = _KNOWN_AWPERS[pslug]
            if hs_rate is None or hs_rate > _AWPER_HS_CAP:
                logger.info(
                    f"[hs_scale] Known rate override {pslug}: scraped={hs_rate} "
                    f"> cap={_AWPER_HS_CAP} → using measured {awper_known_rate}"
                )
                hs_rate     = awper_known_rate
                hs_rate_src = (
                    f"measured rate ({round(awper_known_rate * 100)}%)"
                )
                awper_warn = True

        if hs_rate is None:
            # Generic default for players not in the calibrated-rate table.
            # Rifler career average on HLTV is ~44-46%; 45% is a conservative
            # centre. AWPers should already be captured in the table above.
            hs_rate     = 0.45
            hs_rate_src = "default (45% — generic rifler estimate)"

        # Apply HS scaling per-map using the best available rate in priority order:
        #   1) Actual HS count from scorecard  → use directly
        #   2) Per-match scraped HS% (only for players NOT in _KNOWN_AWPERS,
        #      because known-rate players use pre-verified rates that are more
        #      reliable than HLTV's JS-rendered detailed stats scraped via fallback)
        #   3) Global hs_rate (known rate override or recent avg)  → kills × hs_rate
        use_known_rate = pslug in _KNOWN_AWPERS
        scaled_maps     = []
        n_actual        = 0
        n_match_pct     = 0
        n_global        = 0
        for m in map_stats:
            kills = m["stat_value"]
            if m.get("headshots") is not None:
                scaled_maps.append({**m, "stat_value": float(m["headshots"])})
                n_actual += 1
            elif m.get("match_hs_pct") is not None and not use_known_rate:
                est = round(kills * m["match_hs_pct"], 2)
                scaled_maps.append({**m, "stat_value": est})
                n_match_pct += 1
            else:
                scaled_maps.append({**m, "stat_value": round(kills * hs_rate, 2)})
                n_global += 1
        map_stats = scaled_maps

        # Build a readable source label for the embed note
        parts = []
        if n_actual:     parts.append(f"{n_actual} actual scorecard")
        if n_match_pct:  parts.append(f"{n_match_pct} per-match HS%")
        if n_global:
            # Describe what the global hs_rate actually came from
            if awper_warn:
                parts.append(f"{n_global} AWPer measured rate ({round(hs_rate*100)}%)")
            elif hs_rate_src and "career profile" in hs_rate_src:
                parts.append(f"{n_global} career HS% ({round(hs_rate*100)}%)")
            elif hs_rate_src and "last " in hs_rate_src:
                parts.append(f"{n_global} recent avg HS% ({round(hs_rate*100)}%)")
            else:
                parts.append(f"{n_global} default estimate ({round(hs_rate*100)}%)")
        hs_rate_src = " + ".join(parts) if parts else hs_rate_src
        logger.info(f"[hs_scale] HS sources: {hs_rate_src}")
    else:
        hs_rate_src = None

    # Snapshot map_stats BEFORE any opponent multiplier is applied.
    # Used for historical displays (form streak, variance, map intel) so they
    # always reflect what the player actually did, not opponent-adjusted values.
    map_stats_hist = list(map_stats)

    # Build per-series breakdown (kills or estimated HS) for the embed.
    # Groups maps back into their original series (2 maps per series).
    _series_breakdown: list[dict] = []
    _seen_series: dict[str, list] = {}
    for _m in map_stats:
        _seen_series.setdefault(_m["match_id"], []).append(_m)
    for _mid, _maps in _seen_series.items():
        _series_breakdown.append({
            "match_id": _mid,
            "total":    round(sum(m["stat_value"] for m in _maps), 1),
            "per_map":  [f"{m.get('map_name', f'Map {j+1}')} {round(m['stat_value'], 1)}" for j, m in enumerate(_maps)],
        })

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

    # ── Session rank_gap cache — keeps stomp/OT flags consistent for teammates ─
    # Key on the opponent name so all players facing the same team get the same
    # rank_gap even when individual HLTV rank fetches are flaky.
    _opp_key = (opponent or "").strip().lower()
    if _opp_key:
        if rank_gap is not None:
            # Successful lookup — store for later teammates
            _session_rank_gap[_opp_key] = rank_gap
            logger.debug(f"[rank_gap_cache] stored rank_gap={rank_gap} for '{_opp_key}'")
        elif _opp_key in _session_rank_gap:
            # Failed lookup — reuse earlier result from a teammate
            rank_gap = _session_rank_gap[_opp_key]
            logger.info(
                f"[rank_gap_cache] rank_gap fetch failed for '{_opp_key}' — "
                f"reusing cached {rank_gap} from earlier teammate grade"
            )

    _period_kpr = (period_stats or {}).get("kpr")
    try:
        sim_result = run_simulation(
            map_stats=map_stats,
            line=line,
            stat_type=stat_type,
            favorite_prob=favorite_prob,
            likely_maps=likely_maps if likely_maps else None,
            rank_gap=rank_gap,
            period_kpr=_period_kpr,
        )
    except Exception as e:
        err_name = type(e).__name__
        logger.error(f"Simulation failed ({err_name}): {e}")
        return {"sim_error": err_name, "error": str(e)[:300]}

    # Override hist_avg / hist_median with RAW unscaled values so the displayed
    # average matches what the user sees in the Series Breakdown.
    # The opponent-adjustment multiplier is already captured in over_prob / edge
    # — the average shown in the embed should reflect actual history, not
    # inflated/deflated numbers the player never actually produced.
    if map_stats_hist:
        from statistics import mean as _mean, median as _median
        _series_ids_raw: dict[str, list] = {}
        for _m in map_stats_hist:
            _series_ids_raw.setdefault(_m["match_id"], []).append(_m["stat_value"])
        _raw_totals = [sum(v) for v in _series_ids_raw.values()]
        if _raw_totals:
            sim_result["hist_avg"]    = round(_mean(_raw_totals), 2)
            sim_result["hist_median"] = round(_median(_raw_totals), 2)

    sim_result["data_source"]   = data_source
    sim_result["used_fallback"] = used_fallback
    sim_result["player_name"]   = player_name
    sim_result["line"]          = line
    sim_result["deep"]          = deep
    sim_result["opponent"]      = opponent     # raw user input — None if not supplied
    sim_result["hs_rate_src"]   = hs_rate_src  # None for kills props, str for HS props
    sim_result["period_stats"]  = period_stats  # HLTV 90-day aggregate stats
    sim_result["is_awper"]          = is_awper
    sim_result["awper_warn"]        = awper_warn
    sim_result["series_breakdown"]  = _series_breakdown   # per-series stat totals
    # bo3.gg enrichment (country, role) — available when scraper is successful
    sim_result["country"]          = (info or {}).get("country")
    sim_result["liquipedia_role"]  = (info or {}).get("liquipedia_role")
    sim_result["bo3gg_context"]    = (info or {}).get("bo3gg_context")

    # If using estimated fallback data — override to PASS, never make directional calls
    # on invented stats. The grade stays for context but direction is unreliable.
    if used_fallback:
        sim_result["decision"] = "PASS"
        sim_result["recommendation"] = "⚠️ PASS — Using estimated stats (HLTV unavailable)"
        sim_result["grade"] = "N/A"
        logger.info(f"[grade] Fallback data → forced PASS for {player_name}")

    # --- Map Intelligence Override ---
    # When the player's historical per-map average on the specific maps expected
    # in this match projects a series total materially BELOW the line, the OVER
    # call is contradicted by map-specific data regardless of the overall average.
    # This prevents the global mean from overriding map-pool reality.
    if (
        sim_result.get("decision") == "OVER"
        and likely_maps
        and map_stats_hist
    ):
        _likely_set = {m.lower() for m in likely_maps}
        _map_intel_vals = [
            m["stat_value"] for m in map_stats_hist
            if m.get("map_name", "").lower() in _likely_set
        ]
        if len(_map_intel_vals) >= 4:  # need at least 4 data points (2 series × 2 maps)
            _map_intel_per_map = sum(_map_intel_vals) / len(_map_intel_vals)
            # Multiply by 2 (Maps 1+2) to get projected series total
            _map_intel_series_proj = _map_intel_per_map * 2
            _map_intel_gap_pct = (_map_intel_series_proj - line) / max(line, 1)
            if _map_intel_gap_pct < -0.10:  # projected total >10% below the line
                _proj_str = round(_map_intel_series_proj, 1)
                _pct_str  = round(_map_intel_gap_pct * 100, 1)
                sim_result["decision"] = "PASS"
                sim_result["recommendation"] = (
                    f"⚠️ PASS — Map intel projects {_proj_str} "
                    f"on expected maps ({_pct_str}% vs line)"
                )
                sim_result["_map_intel_warning"] = (
                    f"🗺️ Map intel override — projected series total {_proj_str} "
                    f"({_pct_str}% vs line {line}) suppressed OVER call"
                )
                logger.info(
                    f"[map_intel_override] {player_name}: projected series total "
                    f"{_proj_str} is {_pct_str}% vs line {line} "
                    f"— OVER suppressed to PASS"
                )

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

    # --- Step 6: Survival Rate vs Win Rate → Exit Fragger Analysis ---
    survival_vals = [m["survival_rate"] for m in map_stats if m.get("survival_rate") is not None]
    deaths_vals   = [m["deaths"]        for m in map_stats if m.get("deaths")        is not None]
    kills_vals    = [m["stat_value"]    for m in map_stats]

    if len(survival_vals) >= 4:
        avg_survival = round(sum(survival_vals) / len(survival_vals), 3)
        total_kills  = sum(kills_vals)
        total_deaths = sum(deaths_vals) if deaths_vals else 1
        avg_kd_ratio = round(total_kills / max(total_deaths, 1), 2)

        # Exit Fragger: survives often but doesn't win duels (dying late in rounds)
        is_exit_fragger = avg_survival > 0.60 and avg_kd_ratio < 1.05

        if is_exit_fragger:
            ef_label = "🚪 Exit Fragger"
            # In blowout scenarios the passive player still racks up late kills
            if rank_gap is not None and rank_gap > 50:
                over_p = sim_result.get("over_prob", 50)
                sim_result["over_prob"]  = round(min(95, over_p + 3.0), 1)
                sim_result["under_prob"] = round(max(5, sim_result.get("under_prob", 50) - 3.0), 1)
                ef_note = (f"Survival {avg_survival:.0%} | K/D {avg_kd_ratio} → "
                           f"passive style helps in blowouts (+3% Over)")
            elif rank_gap is not None and rank_gap < 15:
                over_p = sim_result.get("over_prob", 50)
                sim_result["over_prob"]  = round(max(5, over_p - 2.0), 1)
                sim_result["under_prob"] = round(min(95, sim_result.get("under_prob", 50) + 2.0), 1)
                ef_note = (f"Survival {avg_survival:.0%} | K/D {avg_kd_ratio} → "
                           f"tight rounds limit late-kill opps (-2% Over)")
            else:
                ef_note = f"Survival {avg_survival:.0%} | K/D {avg_kd_ratio} → passive accumulator"
        else:
            ef_label = "⚔️ Active Fragger"
            ef_note  = f"Survival {avg_survival:.0%} | K/D {avg_kd_ratio}"

        sim_result["exit_fragger_label"] = ef_label
        sim_result["exit_fragger_note"]  = ef_note
        sim_result["avg_survival_rate"]  = avg_survival
        sim_result["avg_kd_ratio"]       = avg_kd_ratio
    else:
        sim_result["exit_fragger_label"] = "❓ Survival Data Unavailable"
        sim_result["exit_fragger_note"]  = "_deaths data not scraped on these pages_"
        sim_result["avg_survival_rate"]  = None
        sim_result["avg_kd_ratio"]       = None

    # --- Step 7: Pistol Round KPR → Floor & Security Buffer ---
    pistol_vals = [m["pistol_kills"] for m in map_stats if m.get("pistol_kills") is not None]
    # Determine if values are real (integer from scrape) or estimated (float from formula)
    real_pistol = [v for v in pistol_vals if isinstance(v, int)]
    est_pistol  = [v for v in pistol_vals if isinstance(v, float)]
    pistol_source = "scraped" if real_pistol else ("estimated" if est_pistol else None)

    if pistol_vals:
        avg_pistol_kpr = round(sum(pistol_vals) / len(pistol_vals), 2)
        # Real data threshold: >1.5 kills/pistol round pair
        # Estimated threshold: >0.18 (equivalent to ~2 kills per 22-round map)
        threshold = 1.5 if pistol_source == "scraped" else 0.18
        strong_pistol = avg_pistol_kpr > threshold

        if strong_pistol:
            pistol_label = "🔫 Strong Pistol Player"
            # Security Buffer: +1.5 kills to expected total if team CT win% > 55%
            ct_win_pct = None
            if deep:
                def_profile = deep.get("defensive_profile", {})
                ct_win_pct  = def_profile.get("ct_win_pct") if def_profile else None
                # Also check opponent's CT win (attacker's T side vs opponent's CT)
                opp_ct = deep.get("scouting", {}).get("economy_impact", {}).get("ct_win_pct")

            pistol_win_proxy = ct_win_pct and ct_win_pct > 55
            if pistol_win_proxy:
                old_exp = sim_result.get("expected_total", 0)
                sim_result["expected_total"] = round(old_exp + 1.5, 2)
                sim_result["pistol_buffer_applied"] = True
                pistol_note = (f"Avg pistol KPR: `{avg_pistol_kpr}` ({pistol_source}) | "
                               f"CT win {ct_win_pct}% → +1.5 Security Buffer applied")
            else:
                sim_result["pistol_buffer_applied"] = False
                pistol_note = (f"Avg pistol KPR: `{avg_pistol_kpr}` ({pistol_source}) | "
                               f"Pistol CT% unavailable — buffer not applied")
        else:
            pistol_label = "📉 Weak/Average Pistol"
            pistol_note  = f"Avg pistol KPR: `{avg_pistol_kpr}` ({pistol_source}) — below floor threshold"
    else:
        avg_pistol_kpr = None
        pistol_label   = "❓ No Pistol Data"
        pistol_note    = "_Pistol round stats not available_"

    sim_result["pistol_label"]   = pistol_label
    sim_result["pistol_note"]    = pistol_note
    sim_result["avg_pistol_kpr"] = avg_pistol_kpr

    # --- Step 8: KAST% Consistency Engine + ADR Kill Conversion ---
    kast_vals = [m["kast_pct"] for m in map_stats if m.get("kast_pct") is not None]
    adr_vals  = [m["adr"]      for m in map_stats if m.get("adr")      is not None]

    avg_kast   = None
    kill_eff   = None
    avg_adr    = None
    kast_adj_applied = False

    if len(kast_vals) >= 3:
        avg_kast = round(sum(kast_vals) / len(kast_vals), 1)
        if avg_kast >= 72:
            kast_label = "📊 Consistent Contributor"
            # High KAST = player impacts rounds reliably → tighter kill floor → lean Over
            over_p = sim_result.get("over_prob", 50)
            sim_result["over_prob"]  = round(min(95, over_p + 2.0), 1)
            sim_result["under_prob"] = round(max(5, sim_result.get("under_prob", 50) - 2.0), 1)
            kast_note = f"Avg KAST `{avg_kast}%` → reliable impact (+2% Over floor)"
            kast_adj_applied = True
        elif avg_kast < 58:
            kast_label = "⚠️ Round-Dependent"
            kast_note  = f"Avg KAST `{avg_kast}%` → inconsistent impact, high variance"
        else:
            kast_label = "🎮 Standard KAST"
            kast_note  = f"Avg KAST `{avg_kast}%`"
    else:
        kast_label = "❓ KAST Not Available"
        kast_note  = "_KAST% not found on these match pages_"

    if len(adr_vals) >= 3:
        avg_adr = round(sum(adr_vals) / len(adr_vals), 1)
        total_kills_all  = sum(m["stat_value"] for m in map_stats)
        # Use actual scraped round counts instead of a fixed 22 per map
        total_rounds_all = max(sum(m.get("rounds", 22) for m in map_stats), 1)
        avg_kpr_all      = total_kills_all / total_rounds_all
        kill_eff         = round(avg_kpr_all / (avg_adr / 100), 3) if avg_adr > 0 else None

        if kill_eff and kill_eff > 0.28:
            conv_label = "🎯 Clean Finisher"
            conv_note  = f"ADR `{avg_adr}` | Efficiency `{kill_eff:.2f}` → kills damage efficiently"
        elif kill_eff and kill_eff < 0.18:
            conv_label = "💥 Damage Dealer"
            conv_note  = f"ADR `{avg_adr}` | Efficiency `{kill_eff:.2f}` → chunks but under-converts on kills"
        else:
            conv_label = "📋 Standard Converter"
            conv_note  = f"ADR `{avg_adr}` | Efficiency `{kill_eff:.2f}`" if kill_eff else f"ADR `{avg_adr}`"
    else:
        conv_label = "❓ No ADR Data"
        conv_note  = "_ADR not available on these match pages_"

    sim_result["kast_label"]       = kast_label
    sim_result["kast_note"]        = kast_note
    sim_result["avg_kast"]         = avg_kast
    sim_result["kast_adj_applied"] = kast_adj_applied
    sim_result["conv_label"]       = conv_label
    sim_result["conv_note"]        = conv_note
    sim_result["avg_adr"]          = avg_adr
    sim_result["kill_eff"]         = kill_eff

    # --- Step 9: Confidence Score (A–F) + Unit Sizing ---
    conf_score = 50  # baseline

    n_series_val = sim_result.get("n_series", 0)
    if n_series_val >= 8:    conf_score += 15
    elif n_series_val >= 6:  conf_score += 8
    elif n_series_val < 4:   conf_score -= 15

    stab_std_val = sim_result.get("stability_std", 0)
    if stab_std_val < 4:     conf_score += 10
    elif stab_std_val > 8:   conf_score -= 12
    elif stab_std_val > 6:   conf_score -= 5

    if avg_kast is not None:
        if avg_kast >= 72:   conf_score += 8
        elif avg_kast < 58:  conf_score -= 8

    decision_val = sim_result.get("decision", "PASS")
    trend_pct_val = sim_result.get("trend_pct", 0)
    hot_form_val  = trend_pct_val >= 12
    cold_form_val = trend_pct_val <= -12
    if (decision_val == "OVER"  and hot_form_val)  or (decision_val == "UNDER" and cold_form_val):
        conf_score += 10
    elif (decision_val == "OVER" and cold_form_val) or (decision_val == "UNDER" and hot_form_val):
        conf_score -= 10

    fair_line_num = sim_result.get("fair_line", line)
    try:
        misprice_gap = abs(float(fair_line_num) - line)
    except (TypeError, ValueError):
        misprice_gap = 0
    if misprice_gap > 4:     conf_score += 12
    elif misprice_gap > 2:   conf_score += 6

    if kill_eff is not None:
        if kill_eff > 0.28:  conf_score += 5
        elif kill_eff < 0.18: conf_score -= 5

    if used_fallback:
        conf_score = 0

    conf_score = max(0, min(100, conf_score))

    if conf_score >= 80:     conf_grade = "A"
    elif conf_score >= 65:   conf_grade = "B"
    elif conf_score >= 50:   conf_grade = "C"
    elif conf_score >= 35:   conf_grade = "D"
    else:                    conf_grade = "F"

    _conf_labels = {
        "A": "🟢 High Confidence",
        "B": "🟡 Moderate Confidence",
        "C": "🟠 Fair Confidence",
        "D": "🔴 Low Confidence",
        "F": "⚫ Unreliable",
    }
    conf_label_final = _conf_labels[conf_grade]

    # Unit Sizing: only for directional calls with enough conviction
    grade_str_val = sim_result.get("grade", "0/10")
    try:
        grade_num_val = int(str(grade_str_val).split("/")[0]) if "/" in str(grade_str_val) else 0
    except (ValueError, TypeError):
        grade_num_val = 0

    sim_result["confidence_score"] = conf_score
    sim_result["confidence_grade"] = conf_grade
    sim_result["confidence_label"] = conf_label_final

    # --- Step 10: Grade Engine Package (new analytics layer) ---
    try:
        grade_pkg = compute_grade_package(
            sim_result=sim_result,
            map_stats=map_stats_hist,
            deep=deep,
            period_stats=period_stats,
        )
        sim_result["grade_pkg"] = grade_pkg
    except Exception as _ge_err:
        logger.warning(f"[grade_engine] Failed: {_ge_err}")
        sim_result["grade_pkg"] = {}

    # --- Unit Sizing (after grade_pkg so confidence is consistent with the embed) ---
    # Use the full grade_engine confidence score (same number displayed to the user)
    # rather than the simplified Step 9 score so unit sizing matches displayed confidence.
    pkg_conf = (sim_result.get("grade_pkg") or {}).get("confidence", conf_score)
    if pkg_conf >= 80:     unit_conf_grade = "A"
    elif pkg_conf >= 65:   unit_conf_grade = "B"
    elif pkg_conf >= 50:   unit_conf_grade = "C"
    elif pkg_conf >= 35:   unit_conf_grade = "D"
    else:                  unit_conf_grade = "F"

    if decision_val in ("OVER", "UNDER"):
        if grade_num_val >= 8 and unit_conf_grade == "A":
            unit_rec = "💰 2u — Strong Play"
        elif grade_num_val >= 6 and unit_conf_grade in ("A", "B"):
            unit_rec = "💵 1u — Value Play"
        elif grade_num_val >= 4 and unit_conf_grade in ("A", "B", "C"):
            unit_rec = "🪙 0.5u — Marginal"
        else:
            unit_rec = "🚫 0u — Skip (low grade or confidence)"
    else:
        unit_rec = "🚫 0u — Pass"

    sim_result["unit_recommendation"] = unit_rec

    return sim_result


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------

def _fmt_comp(comps: dict, key: str, lbl: str) -> str | None:
    v = comps.get(key)
    if v is None:
        return None
    p = round((v - 1) * 100, 1)
    s = "+" if p >= 0 else ""
    return f"`{lbl}` {s}{p}%"


def build_result_embed(
    player_name: str, line: float, stat_type: str, result: dict
) -> discord.Embed:
    # ── Core data extraction ─────────────────────────────────────────────────
    decision  = result.get("decision", "PASS")
    color     = DECISION_COLORS.get(decision, 0x7289DA)
    stat_unit = result.get("stat_type", stat_type)
    used_fb   = result.get("used_fallback", False)

    deep        = result.get("deep") or {}
    opp_display = deep.get("opponent_display") if deep and not deep.get("error") else None
    opp_name    = result.get("opponent")
    pkg         = result.get("grade_pkg") or {}

    form      = pkg.get("form",      {})
    variance  = pkg.get("variance",  {})
    map_intel = pkg.get("map_intel", {})
    flags     = pkg.get("flags",     [])
    confidence = pkg.get("confidence", result.get("confidence_score", 50))
    edge_pct   = pkg.get("edge_pct",   result.get("edge", 0))
    reason     = pkg.get("reason",     result.get("recommendation", ""))

    # ── Title ────────────────────────────────────────────────────────────────
    d_icons = {"OVER": "✅", "UNDER": "❌", "PASS": "⏸️", "MISPRICED": "⚠️"}
    title = f"🎯  {player_name}  ·  {line} {stat_unit}"

    # ── Description — verdict banner ─────────────────────────────────────────
    conf_bar  = ge_prob_bar(confidence / 100, width=8)
    edge_sign = "+" if edge_pct >= 0 else ""
    verdict_line = (
        f"**{d_icons.get(decision, '📊')} {decision}**  ·  "
        f"Confidence: **{confidence}/100** `{conf_bar}`  ·  "
        f"Edge: **{edge_sign}{edge_pct}%**"
    )

    ctx_parts = []
    if opp_display:
        ctx_parts.append(f"vs **{opp_display}**")
    match_ctx = result.get("match_context", "")
    if match_ctx:
        ctx_parts.append(f"_{match_ctx}_")
    if used_fb:
        ctx_parts.append("⚠️ _Estimated data only_")

    SEP = "━" * 30
    description = (
        ("  ·  ".join(ctx_parts) + "\n" if ctx_parts else "")
        + f"{SEP}\n{verdict_line}\n{SEP}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

    # ── HLTV unavailable warning ─────────────────────────────────────────────
    if used_fb:
        embed.add_field(
            name="🚫 HLTV Data Unavailable",
            value="Live data could not be fetched. Stats are **estimated** — no directional call is made.",
            inline=False,
        )

    if deep and deep.get("error") and opp_name:
        embed.add_field(
            name="⚠️ Opponent Not Found",
            value=(
                f"**{opp_name}** wasn't found on HLTV — opponent analysis skipped.\n"
                f"Try exact team name (e.g. `!grade {player_name} {line} {stat_unit} NaVi`)."
            ),
            inline=False,
        )

    # ── 1. Historical Stats ───────────────────────────────────────────────────
    n_series  = result.get("n_series",    "?")
    hist_avg  = result.get("hist_avg",    "N/A")
    hist_med  = result.get("hist_median", "N/A")
    hit_rate  = result.get("hit_rate",    "N/A")
    var_label = variance.get("label", "")
    var_std   = variance.get("std",   "")
    var_floor = variance.get("floor", "")
    var_ceil  = variance.get("ceil",  "")
    form_lbl  = form.get("label", "")
    last4_h   = form.get("last4_hits", "?")
    last4_n   = form.get("last4_n",    "?")
    hist_val  = (
        f"**Avg:** `{hist_avg}` · **Median:** `{hist_med}` · **Hit Rate:** `{hit_rate}%`\n"
        f"**Variance:** {var_label} · σ={var_std} · Range: {var_floor}–{var_ceil}\n"
        f"{form_lbl}  ·  Last {last4_n}: {last4_h}/{last4_n} hit"
    )
    embed.add_field(name=f"📊 Historical — {n_series} BO3 Series", value=hist_val, inline=False)

    # ── 2. Simulation ─────────────────────────────────────────────────────────
    over_p   = result.get("over_prob",  "N/A")
    under_p  = result.get("under_prob", "N/A")
    push_p   = result.get("push_prob",  0)
    sim_mean = result.get("sim_mean",   "N/A")
    sim_std  = result.get("sim_std",    "N/A")
    fair_ln  = result.get("fair_line",  result.get("sim_median", "N/A"))
    n_sims   = result.get("n_simulations", 10000)
    eco_tag  = " _(eco-adj)_" if result.get("economy_adjusted") else ""
    over_bar = ge_prob_bar((over_p or 0) / 100) if isinstance(over_p, (int, float)) else ""
    # EV display (positive EV = value, shown as +X.XXu)
    decision_for_ev = result.get("decision", "PASS")
    if decision_for_ev == "OVER":
        ev_raw = result.get("ev_over", None)
    elif decision_for_ev == "UNDER":
        ev_raw = result.get("ev_under", None)
    else:
        ev_raw = None
    ev_str = f" · EV: **{'+' if ev_raw and ev_raw>=0 else ''}{ev_raw:.3f}u**" if ev_raw is not None else ""
    # p10/p90 floor/ceiling from simulation
    sim_p10 = result.get("sim_p10")
    sim_p90 = result.get("sim_p90")
    range_str = f"\nRange (p10–p90): **{sim_p10:.0f}–{sim_p90:.0f}**" if sim_p10 is not None and sim_p90 is not None else ""
    sim_val  = (
        f"OVER `{line}`: **{over_p}%** {eco_tag} `{over_bar}`\n"
        f"UNDER: **{under_p}%** · Push: **{push_p}%**\n"
        f"Mean: **{sim_mean}** · ±**{sim_std}** · Fair: **{fair_ln}**{ev_str}"
        f"{range_str}"
    )
    embed.add_field(name=f"🎲 Simulation ({n_sims:,} runs)", value=sim_val, inline=True)

    # ── 3. HLTV 90-Day Stats ──────────────────────────────────────────────────
    ps = result.get("period_stats") or {}
    if ps and any(ps.get(k) is not None for k in ("kpr", "rating", "kast", "adr")):
        ps_lines = []
        if ps.get("kpr")    is not None: ps_lines.append(f"KPR: **{ps['kpr']:.2f}**")
        if ps.get("rating") is not None: ps_lines.append(f"Rating: **{ps['rating']:.2f}**")
        if ps.get("kast")   is not None: ps_lines.append(f"KAST: **{ps['kast']:.0f}%**")
        if ps.get("adr")    is not None: ps_lines.append(f"ADR: **{ps['adr']:.0f}**")
        if ps.get("kd")     is not None: ps_lines.append(f"K/D: **{ps['kd']:.2f}**")
        if ps.get("hs_pct") is not None: ps_lines.append(f"HS%: **{ps['hs_pct']:.0f}%**")
        ps_val   = "\n".join(ps_lines)
        ps_label = f"📋 HLTV {ps.get('days', 90)}d Stats"
    else:
        ps_val   = "_HLTV stats page unavailable_"
        ps_label = "📋 HLTV Stats"
    embed.add_field(name=ps_label, value=ps_val, inline=True)

    # ── 4. Map Intelligence (if data available) ────────────────────────────────
    mi_parts = []
    if map_intel.get("projected_labels"):
        mi_parts.append("**Expected:** " + " · ".join(map_intel["projected_labels"][:3]))
    if map_intel.get("projected_series") is not None:
        pvs = map_intel.get("projected_vs_line", "")
        # Show projected series total (per-map avg × 2) not the raw per-map avg
        mi_parts.append(f"**Series proj on these maps:** `{map_intel['projected_series']}` {pvs}")
    if map_intel.get("best_map") and map_intel.get("worst_map"):
        bm = map_intel["best_map"]
        wm = map_intel["worst_map"]
        mi_parts.append(f"Best: {bm[0].title()} `{bm[1]}` ↑ · Worst: {wm[0].title()} `{wm[1]}` ↓")
    if mi_parts:
        embed.add_field(name="🗺️ Map Intelligence", value="\n".join(mi_parts), inline=False)

    # ── 5. Opponent Deep Analysis ─────────────────────────────────────────────
    if deep and not deep.get("error") and opp_display:
        combined   = deep.get("combined_multiplier", 1.0) or 1.0
        total_pct  = round((combined - 1) * 100, 1)
        tot_sign   = "+" if total_pct >= 0 else ""
        components = deep.get("components", {})

        comp_str = "  ".join(filter(None, [
            _fmt_comp(components, "defensive", "Def"),
            _fmt_comp(components, "t_side",    "T-Side"),
            _fmt_comp(components, "rank",      "Rank"),
            _fmt_comp(components, "map_pool",  "Maps"),
            _fmt_comp(components, "h2h",       "H2H"),
        ]))

        h2h_records = deep.get("h2h", [])
        if h2h_records:
            # Only count complete-data records (both maps scraped) in the cleared tally
            _h2h_complete = [s for s in h2h_records if not s.get("partial")]
            h2h_clears    = sum(1 for s in _h2h_complete if s.get("cleared"))
            _h2h_n        = len(_h2h_complete)
            _partial_n    = len(h2h_records) - _h2h_n
            _partial_tag  = f" (+{_partial_n} partial)" if _partial_n else ""
            h2h_str = (
                f"H2H **{h2h_clears}/{_h2h_n}** "
                f"{'✅' if _h2h_n > 0 and h2h_clears == _h2h_n else '⚠️'}"
                f"{_partial_tag}"
            )
        else:
            h2h_str = "H2H: no data"

        def_lbl  = (deep.get("defensive_profile") or {}).get("label", "")
        rank_lbl = (deep.get("rank_info") or {}).get("label", "")

        opp_val = (
            f"**Combined:** `{tot_sign}{total_pct}%`  ·  {def_lbl}  ·  {h2h_str}\n"
            f"{comp_str}\n"
            f"_{rank_lbl}_"
        )

        otp = deep.get("team_period_stats") or {}
        if otp and any(otp.get(k) is not None for k in ("kpr", "rating", "adr")):
            otp_parts = []
            if otp.get("kpr")    is not None: otp_parts.append(f"KPR {otp['kpr']:.2f}")
            if otp.get("rating") is not None: otp_parts.append(f"Rtg {otp['rating']:.2f}")
            if otp.get("adr")    is not None: otp_parts.append(f"ADR {otp['adr']:.0f}")
            opp_val += f"\n_Opp 90d: {' · '.join(otp_parts)}_"

        # H2H vs line
        scouting = deep.get("scouting", {})
        h2h_sc   = scouting.get("h2h_line", {})
        cleared  = h2h_sc.get("matches_cleared", 0)
        of_n     = h2h_sc.get("of_n", 0)
        partial  = h2h_sc.get("h2h_partial", 0)
        if of_n > 0 or partial > 0:
            bonus_tag   = "  ✅ +5% Over bonus" if h2h_sc.get("matchup_favorite") else ""
            partial_tag = f"  ⚠️ {partial} match(es) incomplete data" if partial else ""
            opp_val += f"\n_H2H vs line: {cleared}/{of_n} cleared{bonus_tag}{partial_tag}_"

        # Per-match H2H kill totals
        if h2h_records:
            _line_val = result.get("line", 0)
            _h2h_rows = []
            for _i, _rec in enumerate(h2h_records, 1):
                _total  = _rec.get("total_kills") or sum(_rec.get("kills_by_map", []))
                _maps   = _rec.get("kills_by_map", [])
                _map_str = " + ".join(str(k) for k in _maps)
                if _rec.get("partial"):
                    _icon = "⚠️"
                    _note = " (partial)"
                elif _rec.get("cleared"):
                    _icon = "✅"
                    _note = ""
                else:
                    _icon = "❌"
                    _note = ""
                _h2h_rows.append(f"{_icon} H2H {_i}: **{_total}** kills ({_map_str}){_note}")
            opp_val += "\n" + "\n".join(_h2h_rows)

        if result.get("stomp_high_line_warning"):
            opp_val += "\n⛔ **STOMP RISK + HIGH LINE — Lean UNDER**"

        embed.add_field(name=f"🔬 vs {opp_display}", value=opp_val, inline=False)

    # ── 6. Risk Flags (only if active) ────────────────────────────────────────
    active_flags = [f for f in flags if not f.startswith("✅")]
    if active_flags:
        embed.add_field(
            name="⚠️ Risk Flags",
            value="\n".join(f"• {f}" for f in active_flags[:4]),
            inline=False,
        )

    # ── 7. Per-Series Breakdown ────────────────────────────────────────────────
    _breakdown  = result.get("series_breakdown", [])
    _is_hs_prop = stat_unit == "HS"
    _hs_src     = result.get("hs_rate_src", "")
    if used_fb:
        # Fallback data is generated/estimated — never show it as real historical series
        embed.add_field(
            name="📋 Series Breakdown",
            value="_No real match history found — breakdown unavailable.\nUsing estimated stats only._",
            inline=False,
        )
    elif _breakdown:
        rows = []
        for i, s in enumerate(_breakdown[:10], 1):
            _maps_str = " + ".join(s.get("per_map", []))
            _total    = s.get("total", 0)
            _tick     = "✅" if _total > line else "❌"
            rows.append(f"S{i}: {_maps_str} = **{_total}** {_tick}")
        rows.append(f"_Line {line} → need >{int(line)}_")
        if _is_hs_prop and _hs_src:
            _src_str = str(_hs_src)
            if "actual scorecard" in _src_str:
                rows.append("_Actual HS counts from HLTV scorecard_")
            elif "estimate" in _src_str or "AWPer" in _src_str:
                rows.append("_AWPer/default HS rate estimate applied_")
        embed.add_field(name="📋 Series Breakdown", value="\n".join(rows), inline=False)

    # ── 8. Verdict ────────────────────────────────────────────────────────────
    unit_rec = result.get("unit_recommendation", "🚫 0u — Pass")
    hs_src   = result.get("hs_rate_src")
    is_awper = result.get("is_awper", False)
    awper_warn = result.get("awper_warn", False)

    if decision in ("OVER", "UNDER"):
        verdict_text = f"**PLAY {decision} {line}**"
    else:
        verdict_text = f"**{decision}**"

    hs_note = ""
    if hs_src and awper_warn:
        hs_note = f"\n⚠️ AWPer detected — {hs_src}"
    elif hs_src and is_awper:
        hs_note = f"\n🎯 AWPer — {hs_src}"
    elif hs_src:
        hs_note = f"\n_HS rate: {hs_src}_"

    grade_str = result.get("grade", "N/A")
    embed.add_field(
        name="✅ Verdict",
        value=(
            f"{verdict_text}\n"
            f"_{reason}_\n"
            f"{unit_rec}  ·  Grade: `{grade_str}`" + hs_note
        ),
        inline=False,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    data_note = "Estimated (HLTV unavailable)" if used_fb else "HLTV Live — Last 10 BO3, Maps 1&2 only"
    # Enrich footer with country (from bo3.gg) and role (from Liquipedia) when available
    country = result.get("country")
    liq_role = result.get("liquipedia_role")
    enrichment_parts = []
    if country:
        enrichment_parts.append(f"🌍 {country}")
    if liq_role:
        role_icons = {"awper": "🎯 AWPer", "igl": "🧠 IGL", "rifler": "⚡ Rifler"}
        enrichment_parts.append(role_icons.get(liq_role, liq_role))
    enrichment_str = "  ·  " + "  ·  ".join(enrichment_parts) if enrichment_parts else ""
    embed.set_footer(
        text=(
            f"Elite CS2 Prop Grader  ·  Negative Binomial Model  ·  "
            f"{data_note}{enrichment_str}  ·  Not financial advice"
        )
    )
    return embed


def build_prob_bar(prob: float, length: int = 10) -> str:
    """Thin wrapper around grade_engine's bar builder."""
    return ge_prob_bar(prob, width=length)


# ---------------------------------------------------------------------------
# !scout — Player scouting card
# ---------------------------------------------------------------------------

@bot.command(name="scout")
async def cmd_scout(ctx, *, player_arg: str = ""):
    """
    Usage: !scout <Player>
    Returns a compact player scouting card from HLTV.
    """
    if not player_arg.strip():
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description="Usage: `!scout <Player>`\nExample: `!scout ZywOo`",
                color=0xFF4136,
            )
        )
        return

    player_name = player_arg.strip()
    status_msg  = await ctx.send(
        embed=discord.Embed(
            title=f"🔍 Scouting {player_name}…",
            description="Fetching HLTV data — please wait.",
            color=0x7289DA,
        )
    )

    try:
        # Use get_player_info for search + scrape in one call
        info = await asyncio.to_thread(get_player_info, player_name, "Kills")
        map_stats     = info.get("map_kills", [])
        resolved_name = info.get("player_name", player_name)
        team_name     = info.get("team_name",   "Unknown")
        player_id     = info.get("player_id",   "")
        used_fallback_scout = info.get("used_fallback", False)

        if not map_stats and not used_fallback_scout:
            await status_msg.edit(
                embed=discord.Embed(
                    title="❌ Player Not Found",
                    description=f"Could not find **{player_name}** on HLTV. Check the spelling.",
                    color=0xFF4136,
                )
            )
            return

        # Period stats
        ps = None
        if player_id:
            try:
                player_slug = info.get("player_slug", player_name.lower())
                ps = await asyncio.to_thread(
                    get_player_period_stats, player_id, player_slug, 90
                )
            except Exception:
                pass

        if not map_stats:
            await status_msg.edit(
                embed=discord.Embed(
                    title="❌ No Data",
                    description=f"No recent BO3 match data for **{resolved_name}**.",
                    color=0xFF4136,
                )
            )
            return

        # Compute stats
        from grade_engine import (
            compute_form_streak, compute_variance_tier, compute_map_intel,
            _extract_series_totals,
        )
        series_totals = _extract_series_totals(map_stats)
        variance = compute_variance_tier(series_totals)

        hist_avg  = round(sum(series_totals) / len(series_totals), 1) if series_totals else 0
        hist_med  = round(sorted(series_totals)[len(series_totals) // 2], 1) if series_totals else 0
        per_map_avgs: dict = {}
        for m in map_stats:
            mn = m.get("map_name", "").lower()
            if mn and mn != "unknown":
                per_map_avgs.setdefault(mn, []).append(m["stat_value"])
        sorted_maps = sorted(
            [(mn, round(sum(v)/len(v), 1)) for mn, v in per_map_avgs.items() if v],
            key=lambda x: x[1], reverse=True
        )

        # Role fingerprint
        known_awpers = _KNOWN_AWPERS
        kpr_vals = [m["stat_value"] / max(m["rounds"], 1) for m in map_stats if m.get("rounds")]
        avg_kpr  = round(sum(kpr_vals) / len(kpr_vals), 3) if kpr_vals else None
        hs_rate  = ps.get("hs_pct") / 100 if ps and ps.get("hs_pct") is not None else None
        role, role_emoji = determine_role(
            resolved_name.lower(), known_awpers, avg_kpr,
            avg_fk_rate=None, avg_survival=None, hs_rate=hs_rate
        )

        # Build embed
        color = 0x00B4FF
        title = f"👤  {resolved_name}  ·  {team_name or 'Unknown Team'}"
        desc  = f"{role_emoji} **{role}**  ·  Last {len(series_totals)} BO3 Series (Maps 1&2)"

        embed = discord.Embed(title=title, description=desc, color=color)

        # Historical
        hist_val = (
            f"**Avg:** `{hist_avg}` · **Median:** `{hist_med}`\n"
            f"**Variance:** {variance.get('label','?')} · σ={variance.get('std','?')}\n"
            f"**Range:** {variance.get('floor','?')}–{variance.get('ceil','?')}"
        )
        embed.add_field(name=f"📊 Kill Totals (Maps 1+2, {len(series_totals)} series)", value=hist_val, inline=False)

        # Map strengths
        if sorted_maps:
            top3 = sorted_maps[:3]
            worst = sorted_maps[-1] if len(sorted_maps) > 1 else None
            map_lines = [f"**{mn.title()}:** `{avg}` avg" for mn, avg in top3]
            if worst and worst[0] != top3[-1][0]:
                map_lines.append(f"🔴 Worst: **{worst[0].title()}:** `{worst[1]}` avg")
            embed.add_field(name="🗺️ Map Pool (by kill avg)", value="\n".join(map_lines), inline=True)

        # HLTV 90d
        if ps and any(ps.get(k) is not None for k in ("kpr", "rating", "kast", "adr")):
            ps_lines = []
            if ps.get("kpr")    is not None: ps_lines.append(f"KPR: **{ps['kpr']:.2f}**")
            if ps.get("rating") is not None: ps_lines.append(f"Rating: **{ps['rating']:.2f}**")
            if ps.get("kast")   is not None: ps_lines.append(f"KAST: **{ps['kast']:.0f}%**")
            if ps.get("adr")    is not None: ps_lines.append(f"ADR: **{ps['adr']:.0f}**")
            if ps.get("kd")     is not None: ps_lines.append(f"K/D: **{ps['kd']:.2f}**")
            if ps.get("hs_pct") is not None: ps_lines.append(f"HS%: **{ps['hs_pct']:.0f}%**")
            embed.add_field(name=f"📋 HLTV {ps.get('days',90)}d Stats", value="\n".join(ps_lines), inline=True)

        embed.add_field(
            name="💡 Quick Grade",
            value=f"Try `!grade {resolved_name} {{line}} Kills` to grade a prop",
            inline=False,
        )
        embed.set_footer(text="Elite CS2 Prop Grader  ·  HLTV Live Data  ·  Not financial advice")

        await status_msg.edit(embed=embed)

    except Exception as e:
        logger.exception(f"[scout] Error for {player_name}: {e}")
        await status_msg.edit(
            embed=discord.Embed(
                title="❌ Error",
                description=f"Failed to scout **{player_name}**: {str(e)[:200]}",
                color=0xFF4136,
            )
        )


# ---------------------------------------------------------------------------
# !lines — Multi-line probability table
# ---------------------------------------------------------------------------

@bot.command(name="lines")
async def cmd_lines(ctx, player_arg: str = "", stat_type_arg: str = "Kills"):
    """
    Usage: !lines <Player> [Kills|HS]
    Returns a probability table for ±3 lines around the fair line.
    """
    if not player_arg.strip():
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description="Usage: `!lines <Player> [Kills|HS]`\nExample: `!lines ZywOo Kills`",
                color=0xFF4136,
            )
        )
        return

    player_name = player_arg.strip()
    stat_type   = "HS" if stat_type_arg.upper() in ("HS", "HEADSHOTS", "HEADSHOT") else "Kills"

    status_msg = await ctx.send(
        embed=discord.Embed(
            title=f"📊 Building Lines Table — {player_name}",
            description="Running simulations across 7 line values…",
            color=0x7289DA,
        )
    )

    try:
        info = await asyncio.to_thread(get_player_info, player_name, stat_type)
        map_stats     = info.get("map_kills", [])
        resolved_name = info.get("player_name", player_name)
        player_id     = info.get("player_id", "")
        used_fallback = info.get("used_fallback", False)

        if not map_stats and not used_fallback:
            await status_msg.edit(
                embed=discord.Embed(
                    title="❌ Player Not Found",
                    description=f"**{player_name}** not found on HLTV.",
                    color=0xFF4136,
                )
            )
            return

        if not map_stats:
            await status_msg.edit(
                embed=discord.Embed(
                    title="❌ No Data",
                    description=f"No recent BO3 match data for **{resolved_name}**.",
                    color=0xFF4136,
                )
            )
            return

        # Compute fair line from initial sim (base = median series total)
        from grade_engine import _extract_series_totals
        series_totals = _extract_series_totals(map_stats)
        # Fair line = median series total (already in 2-map units), rounded to 0.5
        med = sorted(series_totals)[len(series_totals)//2] if series_totals else 40.0
        base_line = round(med * 2) / 2

        period_kpr = None
        if player_id:
            try:
                player_slug = info.get("player_slug", player_name.lower())
                ps_lines_cmd = await asyncio.to_thread(
                    get_player_period_stats, player_id, player_slug, 90
                )
                if ps_lines_cmd:
                    period_kpr = ps_lines_cmd.get("kpr")
            except Exception:
                pass

        rows = await asyncio.to_thread(
            run_lines_table,
            map_stats, base_line, stat_type,
            0.60, None, None, period_kpr,
            1.0, 3
        )

        if not rows:
            await status_msg.edit(
                embed=discord.Embed(
                    title="❌ Simulation Failed",
                    description="Could not generate line table. Try again later.",
                    color=0xFF4136,
                )
            )
            return

        # Build table text
        header = f"{'Line':>6}  {'OVER%':>6}  {'UNDER%':>7}  {'Value':>5}"
        divider = "─" * len(header)
        table_lines = [f"```", header, divider]
        for row in rows:
            marker = "►" if row["is_base"] else " "
            val_str = (row["over_val"] or row["under_val"] or "⚪").strip()
            table_lines.append(
                f"{marker}{row['line']:>5.1f}  {row['over']:>5.1f}%  {row['under']:>6.1f}%  {val_str}"
            )
        table_lines.append("```")
        table_lines.append("🟢🟢 Strong OVER · 🟢 Lean OVER · ⚪ Toss-up · 🔴 Lean UNDER · 🔴🔴 Strong UNDER")
        table_lines.append(f"_► = projected fair line  ·  vs -110 vig (52.38% implied)_")

        embed = discord.Embed(
            title=f"📊  {resolved_name}  ·  {stat_type} Lines Table",
            description="\n".join(table_lines),
            color=0x00B4FF,
        )
        embed.set_footer(text="Elite CS2 Prop Grader  ·  Negative Binomial Model  ·  Not financial advice")
        await status_msg.edit(embed=embed)

    except Exception as e:
        logger.exception(f"[lines] Error for {player_name}: {e}")
        await status_msg.edit(
            embed=discord.Embed(
                title="❌ Error",
                description=f"Lines table failed for **{player_name}**: {str(e)[:200]}",
                color=0xFF4136,
            )
        )


# ---------------------------------------------------------------------------
# !pp — PrizePicks live CS2 lines + auto-grade entire slate
# ---------------------------------------------------------------------------

def _pp_stat_type(item: dict) -> str:
    """Map a PrizePicks item to our internal stat_type string.
    Confirmed CS2 values: 'MAPS 1-2 Kills' | 'MAPS 1-2 Headshots'
    """
    raw = (item.get("stat") or "").lower()
    if "headshot" in raw:
        return "HS"
    return "Kills"


def _pp_line_score(item: dict) -> float | None:
    """Extract the numeric line from a PrizePicks item."""
    val = item.get("line_score") or item.get("line")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pp_game_time(item: dict) -> str:
    """Return a short human-readable game start time string."""
    gstart = item.get("game_start") or ""
    if not gstart:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(gstart)
        return dt.strftime("%b %d  %H:%M UTC")
    except Exception:
        return gstart[:16]


def _decision_icon(decision: str) -> str:
    return {"OVER": "✅", "UNDER": "❌", "PASS": "⏸️"}.get(decision, "❓")


@bot.command(name="pp", aliases=["pphs", "ppkills"])
async def cmd_pp(ctx, *, player_arg: str = ""):
    """
    !pp               — grade all live CS2 PrizePicks props (Kills + HS)
    !pphs             — grade only Headshots props
    !ppkills          — grade only Kills props
    !pp <Player>      — grade that player's live PrizePicks line
    !pp refresh       — force-refresh the PrizePicks cache
    """
    global _pp_cancel
    arg = player_arg.strip()

    # Determine stat filter from the command alias used
    _invoked = ctx.invoked_with.lower()
    stat_filter: str | None = (
        "HS"    if _invoked == "pphs"    else
        "Kills" if _invoked == "ppkills" else
        None    # !pp → all stats
    )

    # ── refresh shortcut ────────────────────────────────────────────────────
    if arg.lower() == "refresh":
        pp_invalidate()
        await ctx.send(
            embed=discord.Embed(
                title="🔄 Cache Cleared",
                description="PrizePicks cache cleared — next `!pp` pulls fresh data.",
                color=0x7289DA,
            )
        )
        return

    # ── fetch slate (always fresh scrape, cached 15 min) ─────────────────────
    _slate_label = (
        f"CS2 {stat_filter} Props" if stat_filter else "CS2 Lines"
    )
    status_msg = await ctx.send(
        embed=discord.Embed(
            title=f"📡 Fetching PrizePicks {_slate_label}…",
            description=(
                "Running a live scrape of PrizePicks — this takes ~60s to pull fresh data.\n"
                "Subsequent commands within 15 minutes reuse this result instantly."
            ),
            color=0x7289DA,
        )
    )

    try:
        raw_items = await asyncio.to_thread(get_cs2_lines, arg if arg else None)
    except Exception as exc:
        logger.exception(f"[pp] Apify fetch failed: {exc}")
        await status_msg.edit(
            embed=discord.Embed(
                title="❌ Fetch Failed",
                description=f"Could not pull PrizePicks data: `{str(exc)[:200]}`",
                color=0xFF4136,
            )
        )
        return

    if not raw_items:
        desc = (
            f"No CS2 props found for **{arg}**. Check spelling or run `!pp` for all lines."
            if arg else
            "No CS2/CSGO props are live on PrizePicks right now.\n"
            "Props are usually posted a few hours before matches start.\n"
            "Run `!pp refresh` then try again if you think there should be lines."
        )
        await status_msg.edit(
            embed=discord.Embed(title="📭 No Lines Found", description=desc, color=0xFFDC00)
        )
        return

    # De-duplicate: one grade job per (player_name, stat_type)
    # Apply stat_filter when invoked as !pphs or !ppkills
    seen: set = set()
    jobs: list[dict] = []
    for item in raw_items:
        pname = (item.get("player_name") or "").strip()
        stat  = _pp_stat_type(item)
        score = _pp_line_score(item)
        if not pname or score is None:
            continue
        if stat_filter and stat != stat_filter:
            continue

        key = (pname.lower(), stat)
        if key in seen:
            continue
        seen.add(key)

        # Resolve opponent: whichever team is NOT the player's team
        player_team = (item.get("player_team") or "").strip().lower()
        home        = (item.get("home_team_name") or item.get("home_team") or "").strip()
        away        = (item.get("away_team_name") or item.get("away_team") or "").strip()
        if player_team and home and away:
            opponent = away if home.lower() == player_team else home
        else:
            opponent = home or away or None   # best guess if player_team missing

        jobs.append({"player": pname, "stat": stat, "line": score, "item": item, "opponent": opponent})

    if not jobs:
        _no_props_desc = (
            f"No **{stat_filter}** props found on the current slate. "
            f"Try `!pp` to see all available props."
            if stat_filter else
            "Props found but none had parseable player names or line scores."
        )
        await status_msg.edit(
            embed=discord.Embed(
                title="📭 No Gradeable Props",
                description=_no_props_desc,
                color=0xFFDC00,
            )
        )
        return

    n = len(jobs)
    _grade_label = f"CS2 {stat_filter} Prop" if stat_filter else "CS2 Prop"
    await status_msg.edit(
        embed=discord.Embed(
            title=f"⚙️ Grading {n} {_grade_label}{'s' if n != 1 else ''}…",
            description=(
                f"Found **{n}** {_grade_label.lower()}{'s' if n != 1 else ''} on the slate.\n"
                f"Grading one at a time — detail cards stream in as each player finishes (~30s each)."
            ),
            color=0x7289DA,
        )
    )

    # ── shared state ─────────────────────────────────────────────────────────
    import time as _t
    _start = _t.monotonic()
    HARD_TIMEOUT = 3600  # 60 minutes max — one at a time, ~30s each, 60 props ≈ 30 min

    results: list[dict | None] = [None] * n
    completed: list[int] = []           # indices finished so far

    # Live scoreboard lines: index → formatted row string
    live_rows: dict[int, str] = {}

    def _fmt_row(idx: int, job: dict, res: dict | None) -> str:
        """Build a one-line summary for the running scoreboard."""
        if res is None:
            return f"⏳ **{job['player']}** `{job['line']} {job['stat']}` — grading…"
        if "error" in res:
            return f"⚠️ **{job['player']}** — failed"
        if res.get("used_fallback"):
            return f"🚫 **{job['player']}** `{job['line']} {job['stat']}` — no HLTV profile, skipped"
        decision   = res.get("decision", "PASS")
        over_p     = res.get("over_prob", 50)
        grade_str  = res.get("grade", "?/10")
        pkg        = res.get("grade_pkg") or {}
        conf       = pkg.get("confidence", res.get("confidence_score", 50))
        icon       = _decision_icon(decision)
        item       = job["item"]
        home       = item.get("home_team_name") or item.get("home_team") or "?"
        away       = item.get("away_team_name") or item.get("away_team") or "?"
        gtime      = _pp_game_time(item)
        match_str  = f"{away} @ {home}" + (f" · {gtime}" if gtime else "")
        opp_str    = f" vs **{job['opponent']}**" if job.get("opponent") else ""
        return (
            f"{icon} **{job['player']}**{opp_str} · `{job['line']} {job['stat']}`\n"
            f"  OVER {over_p}% · Grade {grade_str} · Conf {conf}/100\n"
            f"  _{match_str}_"
        )

    async def _update_scoreboard():
        """
        Rebuild and edit the status embed.
        For large slates (>20 props) shows a compact dashboard to stay
        under Discord's 4096-char description limit.
        """
        done    = len(completed)
        pending = n - done
        color   = 0x7289DA if pending > 0 else 0x2ECC40

        # Tally results so far
        over_lines, under_lines, pass_lines, err_lines = [], [], [], []
        active_lines = []   # currently grading (in semaphore)
        for i in range(n):
            if i not in live_rows:
                continue
            row = live_rows[i]
            res = results[i]
            if res is None:
                # Currently being graded (semaphore acquired)
                active_lines.append(f"⏳ **{jobs[i]['player']}** `{jobs[i]['line']} {jobs[i]['stat']}`")
            elif "error" in res:
                err_lines.append(f"⚠️ **{jobs[i]['player']}**")
            else:
                dec = res.get("decision", "PASS")
                over_p    = res.get("over_prob", 50)
                grade_str = res.get("grade", "?/10")
                pkg       = res.get("grade_pkg") or {}
                conf      = pkg.get("confidence", res.get("confidence_score", 50))
                icon      = _decision_icon(dec)
                entry     = f"{icon} **{jobs[i]['player']}** · `{jobs[i]['line']} {jobs[i]['stat']}` · OVER {over_p}% · {grade_str} · {conf}/100"
                if dec == "OVER":
                    over_lines.append(entry)
                elif dec == "UNDER":
                    under_lines.append(entry)
                else:
                    pass_lines.append(entry)

        # Build compact embed for large slates, full list for small ones
        if n <= 20:
            # Small slate: show every row
            rows_text = "\n\n".join(
                live_rows.get(i, f"⏳ **{jobs[i]['player']}** — queued…")
                for i in range(n)
            )
            desc = rows_text or "Starting…"
        else:
            # Large slate: dashboard view
            parts = []
            parts.append(
                f"**Progress:** {done}/{n} graded"
                + (f"  ·  ~{pending * 10}s remaining" if pending > 0 else "  ·  All done!")
            )
            parts.append(
                f"✅ OVER: **{len(over_lines)}**  ·  "
                f"❌ UNDER: **{len(under_lines)}**  ·  "
                f"⏸️ PASS: **{len(pass_lines)}**"
                + (f"  ·  ⚠️ Errors: {len(err_lines)}" if err_lines else "")
            )
            if active_lines:
                parts.append("**Currently grading:**\n" + "\n".join(active_lines[:3]))

            # Show last 8 completed results
            recent_done = [i for i in reversed(completed[-8:])]
            if recent_done:
                recent_rows = []
                for i in recent_done:
                    res = results[i]
                    if res and "error" not in res:
                        dec       = res.get("decision", "PASS")
                        over_p    = res.get("over_prob", 50)
                        grade_str = res.get("grade", "?/10")
                        icon      = _decision_icon(dec)
                        pkg_r     = results[i].get("grade_pkg") or {}
                        conf_r    = pkg_r.get("confidence", results[i].get("confidence_score", 50))
                        recent_rows.append(
                            f"{icon} **{jobs[i]['player']}** `{jobs[i]['line']} {jobs[i]['stat']}` "
                            f"OVER {over_p}% · Grade {grade_str} · Conf {conf_r}/100"
                        )
                    else:
                        recent_rows.append(f"⚠️ **{jobs[i]['player']}** — failed")
                parts.append("**Recent results:**\n" + "\n".join(recent_rows))

            desc = "\n\n".join(parts)

        emb = discord.Embed(
            title=f"⚙️ Grading {n} {_grade_label}s — {done}/{n} done" + (" ✅" if pending == 0 else "…"),
            description=desc[:4000],   # hard cap for safety
            color=color,
        )
        if pending == 0:
            emb.set_footer(text=f"✅ {len(over_lines)} OVER  ❌ {len(under_lines)} UNDER  ⏸️ {len(pass_lines)} PASS · Strong plays follow · Not financial advice")
        else:
            emb.set_footer(text=f"{pending} props still in queue · grading one at a time")
        try:
            await status_msg.edit(embed=emb)
        except Exception:
            pass

    # Reset cancel flag and session rank_gap cache for this run
    _pp_cancel = False
    _session_rank_gap.clear()

    # ── grade props one at a time to avoid HLTV rate limits ──────────────────
    sem = asyncio.Semaphore(1)

    async def _grade_one(idx: int, job: dict):
        if _pp_cancel:
            results[idx] = {"error": "cancelled"}
            completed.append(idx)
            live_rows[idx] = f"⛔ **{job['player']}** — stopped"
            return
        async with sem:
            if _pp_cancel:
                results[idx] = {"error": "cancelled"}
                completed.append(idx)
                live_rows[idx] = f"⛔ **{job['player']}** — stopped"
                return
            # Show "grading…" immediately when this slot starts
            live_rows[idx] = _fmt_row(idx, job, None)
            await _update_scoreboard()
            try:
                _team_hint = (job.get("item") or {}).get("player_team") or None
                res = await asyncio.wait_for(
                    asyncio.to_thread(
                        _analyze_player,
                        job["player"],
                        job["line"],
                        job["stat"],
                        job.get("opponent"),
                        _team_hint,
                    ),
                    timeout=90,   # fixed 90s per player — independent of slate size
                )
                results[idx] = res
            except asyncio.TimeoutError:
                logger.warning(f"[pp] Timed out grading {job['player']}")
                results[idx] = {"error": "timed out"}
            except Exception as exc:
                logger.warning(f"[pp] Grade failed for {job['player']}: {exc}")
                results[idx] = {"error": str(exc)}

        completed.append(idx)
        res = results[idx]
        live_rows[idx] = _fmt_row(idx, job, res)

        # Stream full detail embed — skip players with no HLTV data (fallback PASS)
        if res and "error" not in res and not res.get("used_fallback"):
            try:
                await ctx.send(embed=build_result_embed(job["player"], job["line"], job["stat"], res))
            except Exception as _e:
                logger.warning(f"[pp] Failed to send detail embed for {job['player']}: {_e}")

        await _update_scoreboard()

    tasks = [asyncio.create_task(_grade_one(i, j)) for i, j in enumerate(jobs)]

    # Hard-timeout watchdog
    async def _watchdog():
        await asyncio.sleep(HARD_TIMEOUT)
        for t in tasks:
            if not t.done():
                t.cancel()

    watchdog = asyncio.create_task(_watchdog())
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        watchdog.cancel()

    # ── final scoreboard update ──────────────────────────────────────────────
    await _update_scoreboard()

    # ── for large slates, send a full recap so nothing gets lost ─────────────
    if n > 20:
        over_rec, under_rec, pass_rec, err_rec = [], [], [], []
        for i, job in enumerate(jobs):
            res = results[i]
            if not res:
                continue
            if "error" in res:
                err_rec.append(f"⚠️ **{job['player']}**")
                continue
            if res.get("used_fallback"):
                continue   # no HLTV data — omit from recap entirely
            dec       = res.get("decision", "PASS")
            over_p    = res.get("over_prob", 50)
            grade_str = res.get("grade", "?/10")
            pkg_rec   = res.get("grade_pkg") or {}
            conf_rec  = pkg_rec.get("confidence", res.get("confidence_score", 50))
            icon      = _decision_icon(dec)
            entry     = (
                f"{icon} **{job['player']}** `{job['line']} {job['stat']}` · "
                f"OVER {over_p}% · Grade {grade_str} · Conf {conf_rec}/100"
            )
            if dec == "OVER":
                over_rec.append(entry)
            elif dec == "UNDER":
                under_rec.append(entry)
            else:
                pass_rec.append(entry)

        def _chunk_embed(title: str, lines: list, color: int):
            """Split a list of lines into ≤4000-char embeds."""
            embeds, buf = [], []
            for line in lines:
                candidate = "\n".join(buf + [line])
                if len(candidate) > 3800:
                    embeds.append(discord.Embed(title=title, description="\n".join(buf), color=color))
                    buf = [line]
                else:
                    buf.append(line)
            if buf:
                embeds.append(discord.Embed(title=title, description="\n".join(buf), color=color))
            return embeds

        _sfx = f" — {stat_filter}" if stat_filter else ""
        if over_rec:
            for emb in _chunk_embed(f"✅ OVER Calls ({len(over_rec)}){_sfx}", over_rec, 0x2ECC40):
                await ctx.send(embed=emb)
        if under_rec:
            for emb in _chunk_embed(f"❌ UNDER Calls ({len(under_rec)}){_sfx}", under_rec, 0xFF4136):
                await ctx.send(embed=emb)



# ---------------------------------------------------------------------------
# !ppstop — cancel an in-progress !pp grading run
# ---------------------------------------------------------------------------

@bot.command(name="ppstop")
async def cmd_ppstop(ctx):
    """Stop the currently running !pp grading run after the current player finishes."""
    global _pp_cancel
    _pp_cancel = True
    await ctx.send(
        embed=discord.Embed(
            title="⛔ Grading Stopped",
            description=(
                "The current `!pp` run will stop after the player being graded right now finishes.\n"
                "Run `!pp` again to start a fresh grading session."
            ),
            color=0xFF4136,
        )
    )


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
