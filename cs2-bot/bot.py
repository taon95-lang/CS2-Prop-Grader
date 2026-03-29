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
    get_matchup_adjustment,
)
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
        f"🛡️ Fetching {opponent}'s defensive profile...",
        "📊 Applying matchup adjustment + running simulations...",
        "⏳ Almost there — finalising results...",
    ]
    _STAGES = _STAGES_OPP if opponent else _STAGES_BASE

    def _stage_embed(elapsed: int, stage_idx: int) -> discord.Embed:
        bar = "▓" * min(10, elapsed // 8) + "░" * max(0, 10 - elapsed // 8)
        matchup_line = f" vs **{opponent}**" if opponent else ""
        return discord.Embed(
            title="⚙️ Analyzing...",
            description=(
                f"**Player:** {player_name}{matchup_line}\n"
                f"**Prop:** {line_val} {stat_type}\n\n"
                f"{_STAGES[stage_idx]}\n\n"
                f"`[{bar}]` ⏱️ {elapsed}s elapsed"
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
    TOTAL_TIMEOUT = 150  # ScraperAPI render can take up to 70s + sim time
    result = None

    while True:
        elapsed = int(_time.monotonic() - start)

        # Hard ceiling — give up after TOTAL_TIMEOUT seconds
        if elapsed >= TOTAL_TIMEOUT:
            fut.cancel()
            logger.error(f"Analysis timed out after {TOTAL_TIMEOUT}s")
            await thinking_msg.edit(
                embed=discord.Embed(
                    title="❌ Timed Out",
                    description=(
                        f"Analysis exceeded {TOTAL_TIMEOUT}s and was cancelled.\n"
                        "HLTV data is loading slowly — try again in a moment."
                    ),
                    color=0xFF4136,
                )
            )
            return

        # Poll the future every 20 seconds; update the embed on each tick
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=20)
            break  # Analysis finished — exit the loop
        except asyncio.TimeoutError:
            # Still running — update the progress embed
            elapsed = int(_time.monotonic() - start)
            stage_idx = min(len(_STAGES) - 1, elapsed // 25)
            try:
                await thinking_msg.edit(embed=_stage_embed(elapsed, stage_idx))
            except Exception:
                pass
        except Exception as e:
            err_name = type(e).__name__
            logger.error(f"Executor error ({err_name}): {e}")
            await thinking_msg.edit(
                embed=discord.Embed(
                    title="❌ Analysis Error",
                    description=f"**Simulation Error: {err_name}**\n```{str(e)[:300]}```",
                    color=0xFF4136,
                )
            )
            return

    # Simulation-level error (returned as dict, not raised)
    if "sim_error" in result:
        err_name = result["sim_error"]
        logger.error(f"Simulation returned error: {err_name}")
        await thinking_msg.edit(
            embed=discord.Embed(
                title="❌ Simulation Error",
                description=f"**Simulation Error: {err_name}**\n{result.get('error', '')}",
                color=0xFF4136,
            )
        )
        return

    if "error" in result:
        await thinking_msg.edit(
            embed=discord.Embed(
                title="❌ Error",
                description=result["error"],
                color=0xFF4136,
            )
        )
        return

    embed = build_result_embed(player_name, line_val, stat_type, result)
    await thinking_msg.edit(embed=embed)


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

    # --- Step 3 (optional): Opponent matchup adjustment ---
    matchup_info: dict | None = None
    if opponent:
        try:
            matchup_info = get_matchup_adjustment(opponent)
            if matchup_info:
                adj = matchup_info["adjustment"]
                map_stats = [round(k * adj, 2) for k in map_stats]
                logger.info(
                    f"Matchup adjustment for '{opponent}': "
                    f"×{adj} ({matchup_info['label']}) — "
                    f"{matchup_info['sample_maps']} sample maps"
                )
            else:
                logger.warning(f"Could not fetch defensive stats for '{opponent}' — no adjustment applied")
        except Exception as e:
            logger.warning(f"Matchup adjustment failed ({type(e).__name__}): {e}")
            matchup_info = None

    # --- Step 4: Monte Carlo simulation ---
    favorite_prob = 0.55  # default edge assumption
    try:
        sim_result = run_simulation(
            map_stats=map_stats,
            line=line,
            stat_type=stat_type,
            favorite_prob=favorite_prob,
        )
    except Exception as e:
        err_name = type(e).__name__
        logger.error(f"Simulation failed ({err_name}): {e}")
        return {"sim_error": err_name, "error": str(e)[:300]}

    sim_result["data_source"] = data_source
    sim_result["used_fallback"] = used_fallback
    sim_result["player_name"] = player_name
    sim_result["line"] = line
    sim_result["matchup"] = matchup_info  # None if no opponent given
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

    matchup = result.get("matchup")
    opp_display = matchup["team_display"] if matchup else None

    title = f"{emoji} CS2 Prop Grade — {player_name}"
    if opp_display:
        title += f" vs {opp_display}"

    description = (
        f"**Prop:** `{line} {stat_type}` ({'Over' if decision == 'OVER' else 'Under' if decision == 'UNDER' else decision})\n"
        f"**Data:** {result.get('data_source', 'HLTV')}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

    # Matchup section (only when opponent was provided)
    if matchup:
        adj = matchup["adjustment"]
        adj_pct = round((adj - 1) * 100, 1)
        adj_sign = "+" if adj_pct >= 0 else ""
        label = matchup["label"]
        label_emoji = {"tough": "🛡️ Tough", "average": "⚖️ Average", "soft": "💨 Soft"}.get(label, label)
        embed.add_field(
            name=f"🎯 Matchup — vs {opp_display}",
            value=(
                f"**Defense:** {label_emoji}\n"
                f"**Avg Kills Allowed/Map:** `{matchup['avg_allowed']}`\n"
                f"**Adjustment:** `{adj_sign}{adj_pct}%` applied to kill distribution\n"
                f"**Sample:** {matchup['sample_maps']} maps"
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

    embed.add_field(
        name="📊 Historical Stats (2-map totals)",
        value=(
            f"**Average:** `{result.get('hist_avg', 'N/A')}`\n"
            f"**Median:** `{result.get('hist_median', 'N/A')}`\n"
            f"**Hit Rate vs {line}:** `{result.get('hit_rate', 'N/A')}%`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🎯 Projection",
        value=(
            f"**Rounds/Map:** `{result.get('rounds_per_map', 22)}`\n"
            f"**Total Rounds:** `{result.get('total_projected_rounds', 44)}`\n"
            f"**Expected {stat_type}:** `{result.get('expected_total', 'N/A')}`\n"
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
    over_bar = build_prob_bar(over_p / 100)
    embed.add_field(
        name="📈 Simulated Probabilities",
        value=(
            f"**Over {line}:** `{over_p}%` {over_bar}\n"
            f"**Under {line}:** `{under_p}%`\n"
            f"**Push:** `{result.get('push_prob', 0)}%`"
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
