import discord
from discord.ext import commands
from discord import app_commands
from typing import Literal
import os
import asyncio
import logging
from scraper import (
    search_player,
    get_player_recent_series,
    get_match_odds,
    get_player_info_fallback,
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
async def grade_prop(ctx, player_name: str = None, line: str = None, stat_type: str = "Kills"):
    if not player_name or not line:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description=(
                    "**Correct usage:**\n"
                    "`!grade [Player Name] [Line] [Kills/HS]`\n\n"
                    "**Examples:**\n"
                    "`!grade ZywOo 38.5 Kills`\n"
                    "`!grade s1mple 14.5 HS`"
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

    # Immediately send a visible acknowledgement — keeps Discord from looking dead
    thinking_msg = await ctx.send(
        embed=discord.Embed(
            title="⚙️ Analyzing...",
            description=(
                f"**Player:** {player_name}\n"
                f"**Prop:** {line_val} {stat_type}\n\n"
                f"🔍 Fetching HLTV data (15s timeout, then falls back to Estimated)...\n"
                f"📊 Running 25,000 Monte Carlo simulations once data is ready."
            ),
            color=0x7289DA,
        )
    )

    # Run all blocking work in a thread executor — event loop stays free
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _analyze_player(player_name, line_val, stat_type),
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        logger.error("Analysis timed out after 60s")
        await thinking_msg.edit(
            embed=discord.Embed(
                title="❌ Timed Out",
                description="The analysis took longer than 60 seconds and was cancelled. Try again shortly.",
                color=0xFF4136,
            )
        )
        return
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
        value="`!grade [Player Name] [Line] [Kills/HS]`",
        inline=False,
    )
    embed.add_field(
        name="Examples",
        value=(
            "`!grade ZywOo 38.5 Kills`\n"
            "`!grade s1mple 14.5 HS`\n"
            "`!grade NiKo 22.5 Kills`"
        ),
        inline=False,
    )
    embed.add_field(
        name="How it works",
        value=(
            "1️⃣ Fetches last 10 BO3 series (Maps 1 & 2 only) from HLTV\n"
            "2️⃣ Adjusts round projections based on match odds\n"
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

def _analyze_player(player_name: str, line: float, stat_type: str) -> dict:
    """
    All blocking I/O and CPU work lives here.
    Run via loop.run_in_executor so the async event loop is never blocked.

    Fix 3 — Simulation Guard:
    If HLTV scraping fails at any point, the function always falls through
    to Estimated Stats and always proceeds to the simulation. It never hangs
    or returns early without a result.
    """
    internal_stat = "Kills" if stat_type in ("Kills", "kills") else "HS"

    # --- Step 1: Try HLTV (15s timeout enforced inside scraper) ---
    map_stats = []
    data_source = "HLTV Live (cloudscraper)"
    used_fallback = False

    try:
        player_info = search_player(player_name)
        logger.info(f"Player search: {player_info}")

        if player_info:
            map_stats = get_player_recent_series(
                player_info["id"], player_info["name"], stat_type=internal_stat
            )
            logger.info(f"HLTV returned {len(map_stats)} map(s)")
    except Exception as e:
        logger.warning(f"HLTV scrape raised exception ({type(e).__name__}): {e}")
        map_stats = []

    # --- Fix 3: Simulation Guard — always fall through to estimated if needed ---
    if not map_stats or len(map_stats) < 4:
        logger.warning("Insufficient HLTV data — using Estimated Stats fallback")
        try:
            map_stats = get_player_info_fallback(player_name, stat_type=internal_stat)
            data_source = "⚠️ Estimated (HLTV unavailable — stats are approximate)"
            used_fallback = True
        except Exception as e:
            logger.error(f"Fallback generator failed ({type(e).__name__}): {e}")
            return {"error": "Both HLTV and fallback data sources failed. Please try again."}

    if not map_stats:
        return {"error": "No data available. Check the player name spelling."}

    # --- Step 2: Match odds (safe, always has a default) ---
    try:
        favorite_prob = get_match_odds(player_name)
    except Exception:
        favorite_prob = 0.55  # default: slight favourite

    # --- Step 3: Monte Carlo simulation ---
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

    title = f"{emoji} CS2 Prop Grade — {player_name}"
    description = (
        f"**Prop:** `{line} {stat_type}` ({'Over' if decision == 'OVER' else 'Under' if decision == 'UNDER' else decision})\n"
        f"**Data:** {result.get('data_source', 'HLTV')}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

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
