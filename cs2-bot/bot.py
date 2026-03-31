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
    _warm_hltv_session,
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
    # Pure AWPers — low HS% because AWP kills land on body
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
    # Hybrid / entry riflers who occasionally AWP — higher HS% than pure AWPers
    "ropz":        0.35,
    # Riflers: scraped HS% is unreliable (HLTV detailed stats JS-rendered).
    # Rates below are measured from real match data.
    "idisbalance": 0.36,   # rifler — measured 35.8% vs State (19 HS / 53 kills)
}

# HS% above this threshold for a known AWPer is almost certainly a scraping
# artefact (often KAST% misread as HS%).  Override with the role-based estimate.
# Set at 0.55 so genuine high-HS rifler readings are NOT blocked.
_AWPER_HS_CAP = 0.55

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

    # --- Step 2.5: HS% Scaling (only for HS props) ---
    # The scraper always returns kill data. For HS props we convert kills → HS
    # using the player's recent match HS% (or career profile, or role default).
    hs_rate      = None
    hs_rate_src  = None
    is_awper     = False
    awper_warn   = False   # True when AWPer override fires

    pslug = info.get("player_slug", "").lower() if not used_fallback else ""

    if stat_type == "HS" and not used_fallback:
        n_hs_matches = info.get("hs_pct_n_matches", 0)

        # Priority 1: recent HS% derived from actual match pages (last ~10 matches)
        recent_hs = info.get("recent_hs_pct")
        if recent_hs is not None:
            hs_rate     = recent_hs
            hs_rate_src = (
                f"last {n_hs_matches} matches avg "
                f"({round(recent_hs * 100)}%)"
            )
            logger.info(f"[hs_scale] Using recent match HS%: {hs_rate_src}")

        # Priority 2: career HS% from their HLTV profile page (separate request)
        if hs_rate is None:
            _pid   = info.get("player_id")
            _pslug = info.get("player_slug")
            if _pid and _pslug:
                try:
                    profile_rate = get_player_hs_pct(_pid, _pslug)
                    if profile_rate is not None:
                        hs_rate     = profile_rate
                        hs_rate_src = f"career profile ({round(profile_rate * 100)}%)"
                except Exception as _e:
                    logger.warning(f"HS% profile scrape failed ({type(_e).__name__}): {_e}")

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
            # Generic default — warn if it looks high relative to AWPer norms
            hs_rate     = 0.40
            hs_rate_src = "default (40% — rifler average; lower for AWPers)"

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
        if n_global:     parts.append(f"{n_global} AWPer/default estimate")
        hs_rate_src = " + ".join(parts) if parts else hs_rate_src
        logger.info(f"[hs_scale] HS sources: {hs_rate_src}")
    else:
        hs_rate_src = None

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

    sim_result["data_source"]  = data_source
    sim_result["used_fallback"] = used_fallback
    sim_result["player_name"]  = player_name
    sim_result["line"]         = line
    sim_result["deep"]         = deep
    sim_result["opponent"]     = opponent     # raw user input — None if not supplied
    sim_result["hs_rate_src"]  = hs_rate_src  # None for kills props, str for HS props
    sim_result["is_awper"]          = is_awper
    sim_result["awper_warn"]        = awper_warn
    sim_result["series_breakdown"]  = _series_breakdown   # per-series stat totals

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
        total_rounds_all = max(len(map_stats) * 22, 1)
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

    if decision_val in ("OVER", "UNDER"):
        if grade_num_val >= 8 and conf_grade == "A":
            unit_rec = "💰 2u — Strong Play"
        elif grade_num_val >= 6 and conf_grade in ("A", "B"):
            unit_rec = "💵 1u — Value Play"
        elif grade_num_val >= 4 and conf_grade in ("A", "B", "C"):
            unit_rec = "🪙 0.5u — Marginal"
        else:
            unit_rec = "🚫 0u — Skip (low grade or confidence)"
    else:
        unit_rec = "🚫 0u — Pass"

    sim_result["confidence_score"] = conf_score
    sim_result["confidence_grade"] = conf_grade
    sim_result["confidence_label"] = conf_label_final
    sim_result["unit_recommendation"] = unit_rec

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
    # Use the stat_type from result (set by simulator) so all labels match the prop type
    stat_unit = result.get("stat_type", stat_type)   # "Kills" or "HS"

    deep = result.get("deep")
    opp_display = deep["opponent_display"] if deep and deep.get("opponent_display") else None

    title = f"{emoji} CS2 Prop Grade — {player_name}"
    if opp_display:
        title += f" vs {opp_display}"

    used_fallback = result.get("used_fallback", False)

    if used_fallback:
        data_line = "⚠️ HLTV data unavailable — estimated stats only"
    else:
        data_line = f"HLTV Live ({result.get('data_source', 'HLTV')})"

    description = (
        f"**Prop:** `{line} {stat_type}` ({'Over' if decision == 'OVER' else 'Under' if decision == 'UNDER' else decision})\n"
        f"**Data:** {data_line}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

    # ── HLTV unavailable banner ───────────────────────────────────────────────
    if used_fallback:
        embed.add_field(
            name="🚫 HLTV Data Unavailable",
            value=(
                "Could not retrieve live match data from HLTV for this player.\n"
                "Stats below are **estimated** and no directional call has been made.\n"
                "Check back later or try a different player."
            ),
            inline=False,
        )

    # ── Warn user when the opponent team name was not found on HLTV ────────────
    if deep and deep.get("error") and result.get("opponent"):
        embed.add_field(
            name="⚠️ Opponent Not Found",
            value=(
                f"Could not locate **{result['opponent']}** on HLTV — "
                f"opponent analysis was skipped.\n"
                f"Try the exact team name (e.g. `!grade {player_name} {line} {stat_type} Team-Liquid`)."
            ),
            inline=False,
        )

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
            f"**Recent avg (last {recent_n_maps} maps):** `{recent_avg_kills}` {stat_unit}/map\n"
            f"**Overall avg per map:** `{hist_avg_kills}` {stat_unit}/map\n"
            f"_Simulation is 70% weighted to recent form_"
        ),
        inline=False,
    )

    # Per-series breakdown (compact — one entry per series)
    _breakdown   = result.get("series_breakdown", [])
    _is_hs_prop  = stat_type == "HS"
    _hs_rate_val = result.get("hs_rate_src", "")   # e.g. "AWPer role estimate (22%...)"
    if _breakdown:
        _series_lines = []
        for i, s in enumerate(_breakdown, 1):
            _maps_str = " + ".join(s["per_map"])
            _total_str = f"`{s['total']}`"
            # Mark series that beat the prop line
            _hit = "✅" if s["total"] > line else "❌"
            _series_lines.append(f"S{i}: {_maps_str} = **{s['total']}** {_hit}")
        _breakdown_str = "\n".join(_series_lines)
        if _is_hs_prop:
            _src = str(_hs_rate_val)
            if "actual scorecard" in _src and "per-match" not in _src and "estimate" not in _src:
                _breakdown_note = "\n_Actual HS counts from HLTV scorecard_"
            elif "per-match" in _src or "actual scorecard" in _src:
                _breakdown_note = "\n_Per-match HS% from HLTV overview applied per series_"
            else:
                _breakdown_note = "\n_AWPer/default HS rate applied (per-match data unavailable)_"
        else:
            _breakdown_note = ""
    else:
        _breakdown_str = "_No breakdown available_"
        _breakdown_note = ""

    embed.add_field(
        name=f"📊 Per-Series {stat_unit} Totals (last {len(_breakdown)} series)",
        value=_breakdown_str + _breakdown_note,
        inline=False,
    )

    embed.add_field(
        name="📈 Historical Stats (2-map totals)",
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
    # Show stat/map (KPR × 22) alongside KPR so it's human-readable
    map_kpr_lines = "\n".join(
        f"  `{mn.title()}`: `{round(v * 22, 1)}` {stat_unit}/map  ({v} KPR)"
        for mn, v in sorted(map_kpr.items(), key=lambda x: -x[1])
    ) if map_kpr else "  _No map-specific data_"
    embed.add_field(
        name="📉 Stability Score",
        value=(
            f"**Std Dev (series):** `±{stab_std}` {stat_unit}  —  {stab_label}\n"
            f"**Ceiling:** `{ceiling}` {stat_unit}  |  **Floor:** `{floor_v}` {stat_unit}\n"
            f"**Per-Map Avg:**\n{map_kpr_lines}"
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

    # ── 🏃 Survival Rate / Exit Fragger ──────────────────────────────────────
    ef_label = result.get("exit_fragger_label", "❓ No Data")
    ef_note  = result.get("exit_fragger_note",  "")
    avg_surv = result.get("avg_survival_rate")
    avg_kd   = result.get("avg_kd_ratio")
    surv_line = f"Avg Survival: `{avg_surv:.0%}` | K/D: `{avg_kd}`" if avg_surv is not None else ""
    embed.add_field(
        name="🏃 Survival Profile",
        value=(
            f"**Style:** {ef_label}\n"
            + (f"{surv_line}\n" if surv_line else "")
            + f"_{ef_note}_"
        ),
        inline=True,
    )

    # ── 🔫 Pistol Floor ───────────────────────────────────────────────────────
    pistol_label  = result.get("pistol_label",  "❓ No Pistol Data")
    pistol_note   = result.get("pistol_note",   "")
    buffer_tag    = "  ✅ **+1.5 buffer applied**" if result.get("pistol_buffer_applied") else ""
    embed.add_field(
        name="🔫 Pistol Floor",
        value=(
            f"**{pistol_label}**{buffer_tag}\n"
            f"_{pistol_note}_"
        ),
        inline=True,
    )

    # ── 🎮 KAST% Consistency + ADR Kill Conversion ───────────────────────────
    kast_label_e = result.get("kast_label",  "❓ KAST Not Available")
    kast_note_e  = result.get("kast_note",   "")
    conv_label_e = result.get("conv_label",  "❓ No ADR Data")
    conv_note_e  = result.get("conv_note",   "")
    kast_adj_tag = "  _(+2% Over applied)_" if result.get("kast_adj_applied") else ""
    embed.add_field(
        name="🎮 KAST & ADR Conversion",
        value=(
            f"**Consistency:** {kast_label_e}{kast_adj_tag}\n"
            f"_{kast_note_e}_\n"
            f"**Kill Conversion:** {conv_label_e}\n"
            f"_{conv_note_e}_"
        ),
        inline=False,
    )

    # ── 💰 Fair Line Analysis ──────────────────────────────────────────────────
    fair_line_val  = result.get("fair_line", result.get("sim_median", "N/A"))
    misprice_label = result.get("misprice_label", "✅ Fair Line")
    line_pct       = result.get("line_percentile")
    hs_rate_src    = result.get("hs_rate_src")
    is_awper       = result.get("is_awper", False)
    awper_warn     = result.get("awper_warn", False)
    if line_pct is not None:
        pct_direction = "⬆️ Lean OVER" if line_pct < 50 else ("⬇️ Lean UNDER" if line_pct > 50 else "⚖️ Coin Flip")
        pct_line = f"**Line Percentile:** `{line_pct}%` of sims at/below line — {pct_direction}"
    else:
        pct_line = ""
    if hs_rate_src and awper_warn:
        hs_note = f"\n⚠️ **AWPer detected** — {hs_rate_src}"
    elif hs_rate_src and is_awper:
        hs_note = f"\n🎯 **AWPer** — HS rate: {hs_rate_src}"
    elif hs_rate_src:
        hs_note = f"\n_HS rate: {hs_rate_src}_"
    else:
        hs_note = ""
    embed.add_field(
        name="💰 Fair Line Analysis",
        value=(
            f"**Fair Line (50/50):** `{fair_line_val}` {stat_unit}\n"
            f"**Sportsbook Line:** `{line}` {stat_unit}\n"
            f"**Assessment:** {misprice_label}\n"
            + (f"{pct_line}" if pct_line else "")
            + hs_note
        ),
        inline=False,
    )

    conf_grade  = result.get("confidence_grade", "?")
    conf_label  = result.get("confidence_label", "❓ Unknown")
    conf_score  = result.get("confidence_score", 0)
    unit_rec    = result.get("unit_recommendation", "🚫 0u — Pass")
    edge_val    = result.get("edge", 0)
    edge_sign   = "+" if edge_val >= 0 else ""
    embed.add_field(
        name="💡 Edge & Verdict",
        value=(
            f"**Edge vs Line:** `{edge_sign}{edge_val}%`\n"
            f"**Grade:** `{grade}` {emoji}\n"
            f"**Decision:** `{decision}`\n"
            f"**Confidence:** {conf_label} `({conf_score}/100)`"
        ),
        inline=True,
    )

    rec = result.get("recommendation", "N/A")
    embed.add_field(
        name="✅ Recommendation",
        value=(
            f"**{rec}**\n"
            f"**Unit Size:** {unit_rec}"
        ),
        inline=True,
    )

    footer_data = (
        "Data: Estimated (HLTV unavailable)"
        if used_fallback
        else "Data: HLTV (Last 10 BO3, Maps 1-2 only)"
    )
    embed.set_footer(
        text=f"Elite CS2 Prop Grader | Negative Binomial Model | {footer_data} | Not financial advice."
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
