import discord
from discord.ext import commands
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
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="CS2 props | !grade [Player] [Line] [Kills/HS]",
        )
    )


@bot.command(name="grade")
async def grade_prop(ctx, player_name: str = None, line: str = None, stat_type: str = "Kills"):
    if not player_name or not line:
        embed = discord.Embed(
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
        await ctx.send(embed=embed)
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

    # Acknowledge receipt
    thinking_embed = discord.Embed(
        title="⚙️ Analyzing...",
        description=(
            f"**Player:** {player_name}\n"
            f"**Prop:** {line_val} {stat_type}\n\n"
            f"🔍 Fetching last 10 BO3 series from HLTV (Chrome TLS bypass)...\n"
            f"📊 Running 100,000 Monte Carlo simulations (Negative Binomial)..."
        ),
        color=0x7289DA,
    )
    thinking_msg = await ctx.send(embed=thinking_embed)

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _analyze_player(player_name, line_val, stat_type),
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        await thinking_msg.edit(
            embed=discord.Embed(
                title="❌ Error",
                description=f"An error occurred while analyzing: `{str(e)[:200]}`",
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


def _analyze_player(player_name: str, line: float, stat_type: str) -> dict:
    """Blocking function — run in executor."""
    internal_stat = "Kills" if stat_type in ("Kills", "kills") else "HS"

    # 1. Search for player on HLTV (using curl_cffi Chrome fingerprint)
    player_info = search_player(player_name)
    logger.info(f"Player search result: {player_info}")

    map_stats = []
    data_source = "HLTV Live (cloudscraper)"
    used_fallback = False

    if player_info:
        map_stats = get_player_recent_series(
            player_info["id"], player_info["name"], stat_type=internal_stat
        )
        logger.info(f"Scraped {len(map_stats)} map(s) from HLTV")

    if not map_stats or len(map_stats) < 4:
        logger.warning(
            f"HLTV returned {len(map_stats)} maps — using fallback estimates"
        )
        map_stats = get_player_info_fallback(player_name, stat_type=internal_stat)
        data_source = "⚠️ Estimated (HLTV blocked — stats are approximate)"
        used_fallback = True

    if not map_stats:
        return {"error": "Could not retrieve player data. Check the player name spelling."}

    # 2. Get match odds for round projection
    favorite_prob = get_match_odds(player_name)

    # 3. Monte Carlo simulation
    sim_result = run_simulation(
        map_stats=map_stats,
        line=line,
        stat_type=stat_type,
        favorite_prob=favorite_prob,
    )

    sim_result["data_source"] = data_source
    sim_result["used_fallback"] = used_fallback
    sim_result["player_name"] = player_name
    sim_result["line"] = line
    return sim_result


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

    # Sample Info
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

    # Historical Stats
    embed.add_field(
        name="📊 Historical Stats (2-map totals)",
        value=(
            f"**Average:** `{result.get('hist_avg', 'N/A')}`\n"
            f"**Median:** `{result.get('hist_median', 'N/A')}`\n"
            f"**Hit Rate vs {line}:** `{result.get('hit_rate', 'N/A')}%`"
        ),
        inline=True,
    )

    # Projection
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

    # Simulation
    embed.add_field(
        name=f"🎲 Monte Carlo ({result.get('n_simulations', 100000):,} runs)",
        value=(
            f"**Sim Mean:** `{result.get('sim_mean', 'N/A')}`\n"
            f"**Std Dev:** `±{result.get('sim_std', 'N/A')}`\n"
            f"**Sim Median:** `{result.get('sim_median', 'N/A')}`\n"
            f"**Model:** Negative Binomial"
        ),
        inline=True,
    )

    # Probabilities
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

    # Edge & Grade
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

    # Recommendation
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


@bot.command(name="help")
async def help_cmd(ctx):
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
            "4️⃣ Runs 100,000 Monte Carlo simulations\n"
            "5️⃣ Grades the prop on a 1-10 scale based on edge"
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
    await ctx.send(embed=embed)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Missing arguments. Use `!grade [Player] [Line] [Kills/HS]`")
    elif isinstance(error, commands.CommandNotFound):
        pass
    else:
        logger.error(f"Command error: {error}")


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN secret is not set!")

    keep_alive()
    logger.info("Starting Elite CS2 Prop Grader bot...")
    bot.run(token)
