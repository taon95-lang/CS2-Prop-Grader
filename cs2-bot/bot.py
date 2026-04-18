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
from grades_db import save_grade, record_result, get_entries_for_date, get_pending_entries, get_recent_entries, date_label
from scraper import get_actual_result as _scraper_get_actual_result
from prizepicks import get_cs2_lines, get_player_line, get_all_cs2_props, invalidate_cache as pp_invalidate
from grade_engine import (
    compute_grade_package,
    build_analysis_blurb,
    run_lines_table,
    build_prob_bar as ge_prob_bar,
    determine_role,
    detect_dog_line,
    score_correlated_parlay,
)
from scraper import check_standin, get_recent_team_roster

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
async def grade_prop(ctx, player_name: str = None, line: str = None, *args):
    if not player_name:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Usage Error",
                description=(
                    "**Correct usage:**\n"
                    "`!grade [Player] [Line] [Kills/HS] [Team?] [vs Opponent?]`\n\n"
                    "**Examples:**\n"
                    "`!grade ZywOo 38.5 Kills`\n"
                    "`!grade sandman 28.5 Kills LAG` ← specify player's own team\n"
                    "`!grade ZywOo 38.5 Kills vs NaVi` ← specify opponent\n"
                    "`!grade sandman 28.5 Kills LAG vs Surge` ← both\n"
                    "`!grade ZywOo 38.5 Kills -115` ← real book odds (any position)\n"
                    "`!grade ZywOo` ← auto-fetches live line from PrizePicks"
                ),
                color=0xFF4136,
            )
        )
        return

    # ── Parse remaining args: [stat?] [odds?] [team?] [vs opponent?] ──────────
    # "vs" is the explicit separator between the player's own team and the
    # opponent.  Without "vs", the trailing token is a team hint (the player's
    # own team) — NOT the opponent — which fixes the bug where
    # "!grade sandman 28.5 kills lag" was treating LAG as the opposing team.
    #
    # Odds tokens: -110, -115, +105, +110, etc. — detected by leading +/- and digits.
    # Example: !grade ZywOo 38.5 kills -115 vs NaVi
    remaining = list(args)
    stat_type = "Kills"
    team_hint: str | None = None
    opponent: str | None = None
    book_odds_raw: str | None = None
    book_implied: float = 0.5238  # default -110 both sides

    def _parse_odds_implied(token: str) -> float | None:
        """Convert American odds string to implied probability. Returns None if not odds."""
        import re
        if not re.match(r'^[+-]\d{2,4}$', token):
            return None
        try:
            v = int(token)
            if v > 0:
                return 100 / (v + 100)
            else:
                return abs(v) / (abs(v) + 100)
        except ValueError:
            return None

    # Pull stat type from the front if it's one of the recognised tokens
    if remaining and remaining[0].lower() in ("kills", "hs", "headshots"):
        stat_raw = remaining.pop(0).lower()
        stat_type = "HS" if stat_raw in ("hs", "headshots") else "Kills"

    # Pull odds token if present (can be anywhere before vs separator)
    _odds_indices = [i for i, a in enumerate(remaining) if _parse_odds_implied(a) is not None]
    if _odds_indices:
        _oi = _odds_indices[0]
        book_odds_raw = remaining.pop(_oi)
        book_implied  = _parse_odds_implied(book_odds_raw)
        logger.info(f"[grade] Book odds parsed: {book_odds_raw} → implied {book_implied:.4f}")

    # Split on "vs" (case-insensitive)
    vs_indices = [i for i, a in enumerate(remaining) if a.lower() == "vs"]
    if vs_indices:
        vs_idx = vs_indices[0]
        team_parts = remaining[:vs_idx]
        opp_parts  = remaining[vs_idx + 1:]
        team_hint  = " ".join(team_parts).strip() or None
        opponent   = " ".join(opp_parts).strip() or None
    else:
        # No "vs" — everything left is the player's own team (disambiguation)
        team_hint = " ".join(remaining).strip() or None
        opponent  = None

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
            lambda: _analyze_player(
                player_name, line_val, stat_type, opponent, team_hint,
                book_implied=book_implied, book_odds_raw=book_odds_raw,
            ),
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
        try:
            _baseline_mid = None
            _mk = result.get('map_kills') or []
            if _mk:
                _baseline_mid = str(_mk[0].get('match_id', ''))
            save_grade(player_name, line_val, stat_type, result, opponent=opponent,
                       baseline_match_id=_baseline_mid)
        except Exception as _ge:
            logger.warning(f"[grade] grades_db save failed: {_ge}")
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
# Opponent quality enrichment — adds opp_rank to historical map entries
# ---------------------------------------------------------------------------

def _enrich_with_opp_ranks(
    map_stats: list,
    player_team_slug: str | None,
) -> None:
    """
    In-place: look up the historical opponent's world rank for each map entry
    and store it as map_entry['opp_rank'] (int or None).

    Uses the HLTV ranking page (already cached by deep_analysis) — no new
    HTTP requests when the ranking page is warm from a prior deep analysis call.

    How it works:
      • Each map entry has 'match_slug' like 'faze-vs-natus-vincere'.
      • We split on '-vs-' to get two team slugs.
      • We compare each slug against player_team_slug to identify the opponent.
      • We call rank_by_team_slug() (ranking-page lookup, cached) for the opponent.
    """
    try:
        from deep_analysis import rank_by_team_slug as _rank_by_slug
    except ImportError:
        return

    import re as _re

    def _norm(s: str) -> str:
        return _re.sub(r'[^a-z0-9]', '', s.lower())

    pt_norm = _norm(player_team_slug) if player_team_slug else None

    # Cache per match_slug to avoid duplicate lookups across two maps in one series
    _slug_cache: dict[str, int | None] = {}

    for m in map_stats:
        match_slug = m.get('match_slug', '')
        if not match_slug or '-vs-' not in match_slug:
            continue

        if match_slug in _slug_cache:
            m['opp_rank'] = _slug_cache[match_slug]
            continue

        parts = match_slug.split('-vs-', 1)
        slug_a, slug_b = parts[0].strip(), parts[1].strip()

        opp_slug = None
        if pt_norm:
            norm_a = _norm(slug_a)
            norm_b = _norm(slug_b)
            # Opponent is whichever side does NOT match the player's team
            if pt_norm in norm_a or norm_a in pt_norm:
                opp_slug = slug_b
            elif pt_norm in norm_b or norm_b in pt_norm:
                opp_slug = slug_a
            else:
                # Neither side clearly matches — skip (will get weight 1.0)
                _slug_cache[match_slug] = None
                continue
        else:
            # No team hint: can't determine opponent side — skip
            _slug_cache[match_slug] = None
            continue

        try:
            rank = _rank_by_slug(opp_slug)
        except Exception as _e:
            logger.debug(f"[opp_rank] rank_by_team_slug({opp_slug!r}) failed: {_e}")
            rank = None

        _slug_cache[match_slug] = rank
        m['opp_rank'] = rank


# ---------------------------------------------------------------------------
# Core analysis (blocking — always run via executor)
# ---------------------------------------------------------------------------

def _analyze_player(
    player_name: str,
    line: float,
    stat_type: str,
    opponent: str | None = None,
    player_team_hint: str | None = None,
    book_implied: float = 0.5238,
    book_odds_raw: str | None = None,
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
        info = get_player_info(
            player_name,
            stat_type=internal_stat,
            team_hint=player_team_hint,
            opponent_hint=opponent,   # always pass — opponent is the strongest
                                      # disambiguator when two players share a name
        )
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
                # Pass pre-fetched team info so deep analysis skips the redundant
                # HLTV profile fetch — ensures all teammates get consistent rank_gap.
                _pt_id   = info.get("player_team_id")
                _pt_slug = info.get("player_team_slug")
                _player_team_arg = (_pt_id, _pt_slug) if _pt_id and _pt_slug else None
                deep = run_deep_analysis(
                    player_id=player_id,
                    player_slug=player_slug,
                    player_match_ids=match_ids,
                    opponent_name=opponent,
                    stat_type=stat_type,
                    baseline_avg=baseline_avg,
                    line=line,
                    player_team=_player_team_arg,
                )
                if deep and not deep.get("error"):
                    adj = deep["combined_multiplier"]
                    # The multiplier is retained for display in the embed
                    # (defensive profile, H2H context) but is NO LONGER applied
                    # to the kill distribution.  The grading engine now uses
                    # raw historical data as its baseline — the props line is
                    # already priced with opponent quality factored in, so
                    # scaling the distribution caused double-counting and
                    # systematically inflated OVER predictions.
                    total_pct = round((adj - 1) * 100, 1)
                    sign = "+" if total_pct >= 0 else ""
                    logger.info(
                        f"Deep analysis for '{opponent}': "
                        f"multiplier ×{adj} ({sign}{total_pct}%) — display only, not applied to distribution"
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

    # --- Step 3.5: Opponent quality enrichment ---
    # Add opp_rank to each historical map entry so the simulator can weight
    # maps played against similar-quality opponents more heavily.
    # Only runs when: real HLTV data (not fallback) + player team is known.
    today_opp_rank: int | None = None
    if not used_fallback and map_stats:
        # Derive the player's team slug for -vs- matching in match slugs.
        # Best source: player_team_hint (known from !pp PrizePicks data).
        # Fallback: nothing (enrichment will silently skip matches it can't resolve).
        _pt_slug = player_team_hint   # e.g. "3DMax" — normalised inside _enrich_*
        try:
            _enrich_with_opp_ranks(map_stats, _pt_slug)
            _ranked_count = sum(1 for m in map_stats if m.get('opp_rank') is not None)
            logger.info(
                f"[opp_rank] Enriched {_ranked_count}/{len(map_stats)} maps "
                f"with historical opp_rank (team_hint={_pt_slug!r})"
            )
        except Exception as _oe:
            logger.warning(f"[opp_rank] Enrichment failed: {_oe}")

    # --- Step 4: Monte Carlo simulation ---
    favorite_prob = 0.55
    likely_maps: list = []
    rank_gap: int | None = None
    if deep:
        mp = deep.get("map_pool", {})
        likely_maps = mp.get("most_played", []) or []
        rank_gap = deep.get("rank_info", {}).get("rank_gap")
        today_opp_rank = deep.get("rank_info", {}).get("opp_rank")

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
            today_opp_rank=today_opp_rank,
            book_implied_prob=book_implied,
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

    # --- Step 3b: Specific-Matchup Veto ---
    # S1-S4 in the evidence stack use the player's FULL historical record.
    # When a specific opponent is analysed, two additional matchup-specific
    # signals are available: the deep combined multiplier and the H2H rate.
    # If BOTH of these simultaneously disagree with the current decision at
    # meaningful thresholds, the call is downgraded to PASS.  One signal
    # alone is never enough to override a strong multi-season edge.
    #
    # Example that triggered this fix: Sandman vs Boss — 70% overall hit rate
    # says OVER, but Boss allows -23.6% kills AND H2H is 1/3 (33%) vs this
    # specific opponent.  Both signals agree → downgrade to PASS.
    if deep and not deep.get("error") and not used_fallback:
        _comb_mult  = deep.get("combined_multiplier", 1.0) or 1.0
        _comb_pct   = (_comb_mult - 1.0) * 100          # negative = fewer kills vs this opp
        _h2h        = deep.get("h2h", [])
        _h2h_n      = len(_h2h)
        _h2h_clears = sum(1 for s in _h2h if s.get("cleared"))
        _h2h_rate   = _h2h_clears / _h2h_n if _h2h_n >= 2 else None  # need ≥2 samples

        _cur_dec    = sim_result.get("decision", "PASS")

        if _cur_dec == "OVER":
            # Veto OVER if defense is very tough AND H2H rate is low
            _deep_bearish = _comb_pct <= -12          # -12% or worse defensive suppression
            _h2h_bearish  = _h2h_rate is not None and _h2h_rate <= 0.40
            if _deep_bearish and _h2h_bearish:
                sim_result["decision"] = "PASS"
                sim_result["recommendation"] = (
                    f"⏸️ PASS — Historical edge vetoed by specific matchup: "
                    f"defense {round(_comb_pct)}% · H2H {_h2h_clears}/{_h2h_n} cleared"
                )
                logger.info(
                    f"[matchup_veto] OVER→PASS: deep={round(_comb_pct)}% "
                    f"h2h={_h2h_clears}/{_h2h_n} ({round((_h2h_rate or 0)*100)}%)"
                )

        elif _cur_dec == "UNDER":
            # Veto UNDER if defense is weak (allows many kills) AND H2H rate is high
            _deep_bullish = _comb_pct >= 12           # +12% or more kills expected
            _h2h_bullish  = _h2h_rate is not None and _h2h_rate >= 0.60
            if _deep_bullish and _h2h_bullish:
                sim_result["decision"] = "PASS"
                sim_result["recommendation"] = (
                    f"⏸️ PASS — Historical edge vetoed by specific matchup: "
                    f"defense {round(_comb_pct)}% · H2H {_h2h_clears}/{_h2h_n} cleared"
                )
                logger.info(
                    f"[matchup_veto] UNDER→PASS: deep={round(_comb_pct)}% "
                    f"h2h={_h2h_clears}/{_h2h_n} ({round((_h2h_rate or 0)*100)}%)"
                )
    sim_result["hs_rate_src"]   = hs_rate_src  # None for kills props, str for HS props
    sim_result["period_stats"]  = period_stats  # HLTV 90-day aggregate stats
    sim_result["is_awper"]          = is_awper
    sim_result["awper_warn"]        = awper_warn
    sim_result["series_breakdown"]  = _series_breakdown   # per-series stat totals
    # bo3.gg enrichment (country, role) — available when scraper is successful
    sim_result["country"]          = (info or {}).get("country")
    sim_result["liquipedia_role"]  = (info or {}).get("liquipedia_role")
    sim_result["bo3gg_context"]    = (info or {}).get("bo3gg_context")

    # --- Stand-in Detection ---
    # Check the most recent match page for stand-in markers.
    # A stand-in typically performs differently from a regular roster member —
    # this is a critical esports-specific risk flag.
    _standin_detected = False
    if not used_fallback and info:
        try:
            _pslug_si = info.get("player_slug", "")
            _mids_si  = info.get("match_ids", [])
            if _pslug_si and _mids_si:
                _latest_mid, _latest_slug = _mids_si[0][0], _mids_si[0][1]
                from scraper import _fetch as _scraper_fetch, HLTV_BASE as _HLTV_BASE
                _si_html = _scraper_fetch(f"{_HLTV_BASE}/matches/{_latest_mid}/{_latest_slug}")
                if _si_html:
                    _standin_detected = check_standin(_pslug_si, _si_html)
        except Exception as _sie:
            logger.debug(f"[standin] Check failed: {_sie}")
    sim_result["standin_detected"] = _standin_detected
    if _standin_detected:
        logger.info(f"[standin] ⚠️ {player_name} flagged as stand-in in most recent match")

    # --- Dog Line / Market Efficiency Detection ---
    _dir_sim_prob = (
        sim_result.get("over_prob",  0) if sim_result.get("decision") == "OVER"
        else sim_result.get("under_prob", 0) if sim_result.get("decision") == "UNDER"
        else 0
    )
    _dog = detect_dog_line(
        sim_prob     = float(_dir_sim_prob or 0),
        book_implied = book_implied,
        hist_avg     = float(sim_result.get("hist_avg") or 0),
        line         = line,
        decision     = sim_result.get("decision", "PASS"),
    )
    sim_result["dog_line"] = _dog

    # If using estimated fallback data — override to PASS, never make directional calls
    # on invented stats. The grade stays for context but direction is unreliable.
    if used_fallback:
        sim_result["decision"] = "PASS"
        # Explain the specific cause so the user knows why they're seeing PASS
        _n_real = len(map_stats)  # how many real maps were found before fallback kicked in
        if _n_real == 0:
            _fb_reason = (
                f"⚠️ PASS — No BO3 match data found for **{player_name}** on HLTV "
                f"(0 valid maps). The player may be inactive, or today's match "
                f"stats haven't been posted yet. Try again after the event has results."
            )
        else:
            _fb_reason = (
                f"⚠️ PASS — Only {_n_real} map sample(s) found (need 4+). "
                f"HLTV may be rate-limiting stats pages — try again in 5–10 min."
            )
        sim_result["recommendation"] = _fb_reason
        sim_result["grade"] = "N/A"
        logger.info(f"[grade] Fallback data → forced PASS for {player_name} ({_n_real} real maps found)")

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

    # --- Step 8b: Decision Coherence Check ---
    # All probability adjustments (matchup bonus, economy delta, rating impact)
    # happen AFTER run_simulation locks in the decision.  If those adjustments
    # move over_prob below 50% when decision is OVER — the embed would show
    # "✅ OVER" alongside a sub-50% probability, which is contradictory.
    #
    # Thresholds (post-adjustment):
    #   • If OVER but over_prob < 44% (clearly UNDER territory) → flip to UNDER
    #   • If OVER but over_prob ∈ [44%, 50%) (mildly contradicted) → PASS
    #   • Symmetric for UNDER → OVER / PASS
    #
    # The 44% cutoff preserves PASSes for genuinely mixed signals while letting
    # clear-directional calls survive as OVER/UNDER rather than all defaulting PASS.
    if not sim_result.get("used_fallback"):
        _adj_over  = sim_result.get("over_prob",  50)
        _adj_under = sim_result.get("under_prob", 50)
        _cur_dec   = sim_result.get("decision", "PASS")

        if _cur_dec == "OVER" and _adj_over < _adj_under:
            if _adj_over < 44:
                # Probability strongly favours UNDER — flip the decision
                sim_result["decision"] = "UNDER"
                sim_result["recommendation"] = (
                    f"❌ UNDER — Evidence stack said OVER but sim strongly disagrees "
                    f"(under {_adj_under}% vs over {_adj_over}%). Sim takes precedence."
                )
                logger.info(
                    f"[coherence] OVER→UNDER (sim dominant): "
                    f"over_prob={_adj_over}% under_prob={_adj_under}%"
                )
            else:
                # Mildly contradicted — signals genuinely mixed
                sim_result["decision"] = "PASS"
                sim_result["recommendation"] = (
                    "⏸️ PASS — Evidence stack and simulation disagree "
                    f"(over {_adj_over}% vs under {_adj_under}%)"
                )
                logger.info(
                    f"[coherence] OVER→PASS (borderline): "
                    f"over_prob={_adj_over}% under_prob={_adj_under}%"
                )

        elif _cur_dec == "UNDER" and _adj_under < _adj_over:
            if _adj_under < 44:
                # Probability strongly favours OVER — flip the decision
                sim_result["decision"] = "OVER"
                sim_result["recommendation"] = (
                    f"✅ OVER — Evidence stack said UNDER but sim strongly disagrees "
                    f"(over {_adj_over}% vs under {_adj_under}%). Sim takes precedence."
                )
                logger.info(
                    f"[coherence] UNDER→OVER (sim dominant): "
                    f"over_prob={_adj_over}% under_prob={_adj_under}%"
                )
            else:
                sim_result["decision"] = "PASS"
                sim_result["recommendation"] = (
                    "⏸️ PASS — Evidence stack and simulation disagree "
                    f"(under {_adj_under}% vs over {_adj_over}%)"
                )
                logger.info(
                    f"[coherence] UNDER→PASS (borderline): "
                    f"over_prob={_adj_over}% under_prob={_adj_under}%"
                )

    # --- Step 8c: Recompute Grade String (post-adjustment) ---
    # run_simulation baked the grade before matchup veto / coherence check /
    # probability deltas changed decision or over_prob.  Recompute now so the
    # displayed grade always matches the final decision and final probability.
    _final_dec = sim_result.get("decision", "PASS")
    if _final_dec == "PASS" or sim_result.get("used_fallback"):
        sim_result["grade"] = "N/A"
    else:
        _final_op   = sim_result.get("over_prob", 50.0) / 100.0   # % → fraction
        _book_imp   = 0.5238                                        # -110 implied
        if _final_dec == "OVER":
            _de = (_final_op - _book_imp) * 100
        else:  # UNDER
            _de = ((1.0 - _final_op) - _book_imp) * 100
        if   _de >= 15: sim_result["grade"] = "10/10 (Elite edge)"
        elif _de >= 12: sim_result["grade"] = "9/10 (Elite edge)"
        elif _de >= 8:  sim_result["grade"] = "8/10 (Strong edge)"
        elif _de >= 5:  sim_result["grade"] = "7/10 (Solid lean)"
        elif _de >= 3:  sim_result["grade"] = "6/10 (Marginal edge)"
        elif _de >= 0:  sim_result["grade"] = "5/10 (Fair line)"
        else:           sim_result["grade"] = "4/10 (Negative edge)"

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

    # --- Step 11: Lock Classification ---
    # A "Lock" is reserved for calls where every available signal aligns
    # decisively in one direction.  All five gates must pass:
    #   1. Directional call (not PASS)
    #   2. Grade ≥ 9/10 — evidence stack score ≥ 12
    #   3. Confidence ≥ 80 — data quality + stability gate
    #   4. Sim prob ≥ 68% for the directional side
    #   5. Real HLTV data (not estimated fallback)
    # The matchup veto (Step 3b) and coherence check (Step 8b) guard the
    # upstream decision so if we reach OVER/UNDER here it has already
    # survived those filters.
    _dir_prob = (
        sim_result.get("over_prob",  0) if decision_val == "OVER"
        else sim_result.get("under_prob", 0) if decision_val == "UNDER"
        else 0
    )
    is_lock = (
        decision_val in ("OVER", "UNDER")
        and grade_num_val >= 9
        and pkg_conf >= 80
        and _dir_prob >= 68
        and not used_fallback
    )
    sim_result["is_lock"] = is_lock
    if is_lock:
        logger.info(
            f"[lock] 🔒 LOCK — {decision_val} | grade={grade_num_val}/10 "
            f"conf={pkg_conf} prob={_dir_prob}%"
        )

    # --- Book odds (user-supplied or defaulted to -110) ---
    sim_result["book_implied"]  = book_implied
    sim_result["book_odds_raw"] = book_odds_raw  # e.g. "-115", "+105", None

    # --- Player / team identity fields (from info dict) ---
    # These are needed by the embed builder but were never forwarded from info.
    if not used_fallback and info:
        sim_result.setdefault("player_team_id",   info.get("player_team_id"))
        sim_result.setdefault("player_team_slug",  info.get("player_team_slug"))
        sim_result.setdefault("team_mismatch",     info.get("team_mismatch", False))
        sim_result.setdefault("player_slug",       info.get("player_slug"))

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
    is_lock   = result.get("is_lock", False)

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

    # Lock → gold embed border
    if is_lock:
        color = 0xFFD700

    # ── Player / team / opponent labels ──────────────────────────────────────
    liq_role   = result.get("liquipedia_role")
    country    = result.get("country")
    is_awper   = result.get("is_awper", False)
    awper_warn = result.get("awper_warn", False)

    role_icons = {"awper": "🎯 AWPer", "igl": "🧠 IGL", "rifler": "⚡ Rifler"}
    role_str   = role_icons.get(liq_role, "⚡ Rifler") if liq_role else ("🎯 AWPer" if is_awper else "⚡ Rifler")

    # Team: try deep analysis's player-team display first, fall back to opponent team info
    player_team_str = ""
    pt_id   = result.get("player_team_id")
    pt_slug = result.get("player_team_slug")
    if pt_slug:
        player_team_str = f" ({pt_slug})"

    # Team mismatch warning — resolved player is on a different team than hinted
    _team_mismatch_warn = ""
    if result.get("team_mismatch"):
        _resolved_slug = result.get("player_slug", player_name)
        _team_mismatch_warn = (
            f"\n⚠️ **PLAYER MISMATCH** — Could not find **{player_name}** on "
            f"the hinted team. Stats shown are for **{_resolved_slug}** ({pt_slug}). "
            f"Try `!grade <exact_hltv_tag> ...` for correct results."
        )

    opp_str = f" vs. **{opp_display}**" if opp_display else (f" vs. **{opp_name}**" if opp_name else "")

    # ── Projection labels ─────────────────────────────────────────────────────
    if decision == "OVER":
        proj_word  = "MORE"
        proj_icon  = "✅"
    elif decision == "UNDER":
        proj_word  = "LESS"
        proj_icon  = "❌"
    else:
        proj_word  = "NO BET"
        proj_icon  = "⏸️"

    conf_level_map = {
        "A": "High Confidence",
        "B": "Moderate Confidence",
        "C": "Fair Confidence",
        "D": "Low Confidence",
        "F": "Unreliable",
    }
    conf_grade_chr = result.get("confidence_grade", "C")
    # Recalculate from numeric score for accuracy
    if   confidence >= 80: conf_grade_chr = "A"
    elif confidence >= 65: conf_grade_chr = "B"
    elif confidence >= 50: conf_grade_chr = "C"
    elif confidence >= 35: conf_grade_chr = "D"
    else:                  conf_grade_chr = "F"
    conf_level_str = conf_level_map[conf_grade_chr]

    # ── Metric table data ─────────────────────────────────────────────────────
    n_series   = result.get("n_series",    0) or 0
    hist_avg   = result.get("hist_avg",    "N/A")
    hist_med   = result.get("hist_median", "N/A")
    hit_rate   = result.get("hit_rate",    0) or 0   # stored as percentage (e.g. 70.0)

    # Hit rate as "X/N" fraction
    if n_series > 0 and isinstance(hit_rate, (int, float)):
        _hits_n = round(hit_rate / 100 * n_series)
        hit_str = f"{_hits_n}/{n_series}"
    else:
        hit_str = f"{hit_rate}%"

    # Indicators for avg/median vs line
    def _line_indicator(val, lne):
        try:
            v = float(val)
            gap = (v - lne) / max(lne, 1)
            if gap >= 0.10:    return "✅ Above Line"
            elif gap >= 0.02:  return "⚠️ Near Line"
            elif gap >= -0.02: return "➖ At Line"
            elif gap >= -0.10: return "⚠️ Near Line"
            else:              return "❌ Below Line"
        except (TypeError, ValueError):
            return "—"

    avg_ind = _line_indicator(hist_avg, line)
    med_ind = _line_indicator(hist_med, line)

    # Hit rate indicator
    if isinstance(hit_rate, (int, float)):
        if hit_rate >= 70:   hr_ind = "✅ Strong"
        elif hit_rate >= 50: hr_ind = "⚠️ Moderate"
        else:                hr_ind = "❌ Weak"
    else:
        hr_ind = "—"

    # Projected rounds indicator
    total_rounds = result.get("total_projected_rounds")
    if total_rounds:
        if total_rounds >= 50:   rounds_pace = "✅ Normal+"
        elif total_rounds >= 44: rounds_pace = "⚠️ Competitive"
        else:                    rounds_pace = "❌ Stomp Risk"
        rounds_str = str(total_rounds)
    else:
        rounds_str  = "N/A"
        rounds_pace = "—"

    # Role / Map Pool indicator (from deep analysis map_pool component)
    mp_comp = (deep.get("components") or {}).get("map_pool", 1.0) or 1.0
    if   mp_comp > 1.05:  map_ind = "✅ Favorable"
    elif mp_comp > 0.95:  map_ind = "➖ Neutral"
    else:                 map_ind = "❌ Unfavorable"

    # ── GURU COMMENTARY — synthesized narrative ───────────────────────────────
    commentary_parts = []

    # Opponent defensive context
    if deep and not deep.get("error"):
        comb_mult = deep.get("combined_multiplier", 1.0) or 1.0
        comb_pct  = round((comb_mult - 1) * 100, 1)
        def_lbl   = (deep.get("defensive_profile") or {}).get("label", "")
        rank_lbl  = (deep.get("rank_info") or {}).get("label", "")
        h2h_recs  = deep.get("h2h", [])
        h2h_cmpl  = [s for s in h2h_recs if not s.get("partial")]
        h2h_n     = len(h2h_cmpl)
        h2h_clrs  = sum(1 for s in h2h_cmpl if s.get("cleared"))
        sign_str  = "+" if comb_pct >= 0 else ""
        h2h_note  = f", H2H {h2h_clrs}/{h2h_n} cleared" if h2h_n else ""
        commentary_parts.append(
            f"vs **{opp_display}** ({sign_str}{comb_pct}% combined{h2h_note}). "
            f"{def_lbl}. {rank_lbl}."
        )

    # Stomp / rounds context
    stomp = result.get("stomp_via_rank", False)
    match_ctx = result.get("match_context", "")
    if stomp:
        commentary_parts.append(f"⚠️ **Stomp risk** — projected {rounds_str} rounds ({match_ctx}).")
    elif total_rounds and total_rounds >= 50:
        commentary_parts.append(f"Competitive pace projected ({rounds_str} rounds).")

    # Role context
    if awper_warn:
        commentary_parts.append(f"⚠️ AWPer — HS props use estimated HS rate.")
    elif liq_role or is_awper:
        role_label = role_icons.get(liq_role, "AWPer" if is_awper else "Rifler")
        map_pool_note = "favorable map pool" if mp_comp > 1.05 else ("neutral map pool" if mp_comp > 0.95 else "unfavorable map pool")
        commentary_parts.append(f"{role_label} with {map_pool_note}.")

    # Form context
    trend_pct = result.get("trend_pct", 0) or 0
    if trend_pct >= 12:
        commentary_parts.append("🔥 Currently running **hot** (recent form above career avg).")
    elif trend_pct <= -12:
        commentary_parts.append("🧊 Currently running **cold** (recent form below career avg).")

    # Variance context
    var_std = variance.get("std", "")
    var_lbl = variance.get("label", "")
    if var_lbl:
        commentary_parts.append(f"{var_lbl} · σ={var_std}.")

    # Matchup veto notice
    rec_text = result.get("recommendation", "")
    if "vetoed by specific matchup" in (rec_text or ""):
        commentary_parts.append(f"⚠️ Historical edge overridden — {rec_text.split('—',1)[-1].strip()}")

    guru_commentary = " ".join(commentary_parts) if commentary_parts else "_No additional matchup context available._"

    # ── Final Bet Recommendation text ─────────────────────────────────────────
    unit_rec  = result.get("unit_recommendation", "🚫 0u — Pass")
    grade_str = result.get("grade", "N/A")
    hs_src    = result.get("hs_rate_src")
    edge_sign = "+" if edge_pct >= 0 else ""
    fair_ln   = result.get("fair_line", result.get("sim_median", "N/A"))

    if is_lock:
        final_rec_name  = "🔒 LOCK"
        final_rec_value = f"🔒 **LOCK — BET {proj_word}** `{line}`\n{unit_rec}  ·  Grade: `{grade_str}`  ·  Fair Line: `{fair_ln}`"
    elif decision == "OVER":
        final_rec_name  = "🎯 FINAL BET RECOMMENDATION"
        final_rec_value = f"**✅ BET MORE** — {grade_str}\n{unit_rec}  ·  Fair Line: `{fair_ln}`"
    elif decision == "UNDER":
        final_rec_name  = "🎯 FINAL BET RECOMMENDATION"
        final_rec_value = f"**❌ BET LESS** — {grade_str}\n{unit_rec}  ·  Fair Line: `{fair_ln}`"
    else:
        final_rec_name  = "🎯 FINAL BET RECOMMENDATION"
        final_rec_value = f"**⏸️ NO BET** — Signals too mixed for a confident call\n{unit_rec}"

    if hs_src and awper_warn:
        final_rec_value += f"\n⚠️ AWPer detected — {hs_src}"
    elif hs_src:
        final_rec_value += f"\n_HS rate: {hs_src}_"

    # ── Title + Description (GURU header block) ───────────────────────────────
    lock_pfx = "🔒 " if is_lock else ""
    title = f"{lock_pfx}🏆 [GURU] GRADES & PROJECTIONS"

    SEP = "━" * 30
    fb_warn = "\n⚠️ _Estimated data only — no directional call_" if used_fb else ""

    # Metric table (Discord renders markdown tables in embed descriptions)
    metric_table = (
        "| Metric | Value | Status |\n"
        "| :--- | :--- | :--- |\n"
        f"| **Recent Avg (Last {n_series})** | {hist_avg} | {avg_ind} |\n"
        f"| **Recent Median** | {hist_med} | {med_ind} |\n"
        f"| **Hit Rate** | {hit_str} | {hr_ind} |\n"
        f"| **Projected Rounds** | {rounds_str} | {rounds_pace} |\n"
        f"| **Role / Map Pool** | {role_str} | {map_ind} |"
    )

    description = (
        f"**PLAYER:** {player_name}{player_team_str}{opp_str}\n"
        f"**MATCH:** Maps 1–2 {stat_unit} | **PROP LINE:** `{line}`"
        f"{fb_warn}{_team_mismatch_warn}\n"
        f"{SEP}\n"
        f"**GRADE:** `{conf_grade_chr}`\n"
        f"**PROJECTION:** {proj_icon} **{proj_word}** ({conf_level_str})\n"
        f"{SEP}\n"
        f"{metric_table}"
    )

    embed = discord.Embed(title=title, description=description, color=color)

    # ── Warnings ──────────────────────────────────────────────────────────────
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

    # ── Simulation field ──────────────────────────────────────────────────────
    over_p    = result.get("over_prob",  "N/A")
    under_p   = result.get("under_prob", "N/A")
    push_p    = result.get("push_prob",  0)
    sim_mean  = result.get("sim_mean",   "N/A")
    sim_std   = result.get("sim_std",    "N/A")
    n_sims    = result.get("n_simulations", 10000)
    eco_tag   = " _(eco-adj)_" if result.get("economy_adjusted") else ""
    sim_p10   = result.get("sim_p10")
    sim_p25   = result.get("sim_p25")
    sim_p75   = result.get("sim_p75")
    sim_p90   = result.get("sim_p90")
    dpr_val   = result.get("dpr")
    hist_ceil = result.get("ceiling")
    hist_flr  = result.get("floor")
    misprice_type_val = result.get("misprice_type", "")
    outlier_note_val  = result.get("outlier_note", "")
    book_odds_raw_val = result.get("book_odds_raw")
    book_implied_val  = result.get("book_implied", 0.5238)
    over_bar = ge_prob_bar((over_p or 0) / 100) if isinstance(over_p, (int, float)) else ""
    if decision == "OVER":
        ev_raw = result.get("ev_over", None)
    elif decision == "UNDER":
        ev_raw = result.get("ev_under", None)
    else:
        ev_raw = None

    # Build EV string — label with actual book odds if user supplied them
    if ev_raw is not None:
        _odds_label = f"vs {book_odds_raw_val}" if book_odds_raw_val else "vs -110"
        ev_str = f"\n• **EV ({_odds_label}):** `{'+' if ev_raw >= 0 else ''}{ev_raw:.3f}u`"
    else:
        ev_str = ""

    # Percentile range: p10–p90 headline + p25–p75 IQR below it
    if sim_p10 is not None and sim_p90 is not None:
        range_str = f"\n• **Range (p10–p90):** `{sim_p10:.0f}–{sim_p90:.0f}`"
        if sim_p25 is not None and sim_p75 is not None:
            range_str += f"  ·  **IQR (p25–p75):** `{sim_p25:.0f}–{sim_p75:.0f}`"
    else:
        range_str = ""

    # DPR line
    dpr_str = f"\n• **Deaths/Round (DPR):** `{dpr_val:.3f}`" if dpr_val is not None else ""

    # Misprice type badge (show only when not Fair Line to avoid clutter)
    if misprice_type_val and misprice_type_val != "Fair Line":
        misprice_badge = f"\n• **Misprice Type:** `{misprice_type_val}`"
    else:
        misprice_badge = ""

    # Outlier note
    outlier_str = f"\n• {outlier_note_val}" if outlier_note_val else ""

    # Historical series ceiling / floor (actual observed max/min M1+M2 kills)
    if hist_ceil is not None and hist_flr is not None:
        hist_range_str = f"\n• **Historical Ceiling/Floor:** `{hist_flr}–{hist_ceil}`"
    else:
        hist_range_str = ""

    sim_val = (
        f"• **Simulated Mean:** `{sim_mean}`  ·  **σ:** `{sim_std}`\n"
        f"• **Over Probability:** `{over_p}%` {eco_tag} `{over_bar}`\n"
        f"• **Under Probability:** `{under_p}%`  ·  Push: `{push_p}%`\n"
        f"• **Edge vs. Line:** `{edge_sign}{edge_pct}%`  ·  **Fair Line:** `{fair_ln}`"
        f"{ev_str}{range_str}{hist_range_str}{dpr_str}{misprice_badge}{outlier_str}"
    )
    embed.add_field(name=f"📊 SIMULATION ({n_sims:,} RUNS)", value=sim_val, inline=False)

    # ── Robustness panel (anti-overestimation discipline) ───────────────────
    _trim_avg = result.get("trimmed_avg")
    _sig_mad  = result.get("sigma_mad")
    _shrink   = result.get("shrink_factor")
    _raw_op   = result.get("raw_over_prob")
    _iqr_clip = result.get("iqr_clipped", False)
    _iqr_band = result.get("iqr_band")
    _votes    = result.get("vote_tally") or {}
    if _trim_avg is not None and _sig_mad is not None and _votes:
        _ov, _uv = _votes.get("over_votes", 0), _votes.get("under_votes", 0)
        _pass_reason = _votes.get("pass_reason")
        _shrink_pct = int(round((1.0 if _shrink is None else _shrink) * 100))
        _shrink_note = f"`{_shrink_pct}%`"
        if _shrink and _shrink < 1.0 and _raw_op is not None:
            _shrink_note += f"  (raw {_raw_op}% → {over_p}%)"
        _iqr_str = "—"
        if _iqr_band:
            _clip_tag = " ⚓ clipped" if _iqr_clip else ""
            _iqr_str = f"`{_iqr_band[0]}–{_iqr_band[1]}`{_clip_tag}"
        _vote_arrow = "🟢 OVER" if _ov > _uv else ("🔴 UNDER" if _uv > _ov else "⚖️ split")
        _vote_line  = f"`{_ov}🟢/{_uv}🔴` → {_vote_arrow}"
        if _pass_reason:
            _vote_line += f"  ·  ⏸️ {_pass_reason}"
        robust_val = (
            f"• **Trimmed Avg:** `{_trim_avg:.1f}`  ·  **MAD-σ:** `{_sig_mad:.1f}`  ·  **IQR:** {_iqr_str}\n"
            f"• **Sample-shrink:** {_shrink_note}\n"
            f"• **Sub-signals:** {_vote_line}"
        )
        embed.add_field(name="🛡️ ROBUSTNESS", value=robust_val, inline=False)

    # ── HLTV 90-Day Stats (compact inline) ────────────────────────────────────
    ps = result.get("period_stats") or {}
    if ps and any(ps.get(k) is not None for k in ("kpr", "rating", "kast", "adr")):
        ps_parts = []
        if ps.get("kpr")    is not None: ps_parts.append(f"KPR `{ps['kpr']:.2f}`")
        if ps.get("rating") is not None: ps_parts.append(f"Rtg `{ps['rating']:.2f}`")
        if ps.get("kast")   is not None: ps_parts.append(f"KAST `{ps['kast']:.0f}%`")
        if ps.get("adr")    is not None: ps_parts.append(f"ADR `{ps['adr']:.0f}`")
        if ps.get("kd")     is not None: ps_parts.append(f"K/D `{ps['kd']:.2f}`")
        if ps.get("hs_pct") is not None: ps_parts.append(f"HS% `{ps['hs_pct']:.0f}%`")
        ps_val   = "  ·  ".join(ps_parts)
        ps_label = f"📋 HLTV {ps.get('days', 90)}d Stats"
        embed.add_field(name=ps_label, value=ps_val, inline=False)

    # ── Framework: HLTV Analytics (Firepower, Opening, etc.) ─────────────────
    # These are HLTV attribute scores from the period stats page (0–100 scale)
    _ps_fw = result.get("period_stats") or {}
    _hltv_attrs: list[str] = []
    _attr_map = [
        ("firepower",  "🔥 Firepower"),
        ("opening",    "🚪 Opening"),
        ("entrying",   "⚡ Entrying"),
        ("trading",    "🔄 Trading"),
        ("sniping",    "🎯 Sniping"),
        ("clutching",  "🧠 Clutching"),
        ("utility",    "💡 Utility"),
    ]
    for _attr_key, _attr_label in _attr_map:
        _v = _ps_fw.get(_attr_key)
        if _v is not None:
            _bar_val = min(100, max(0, float(_v)))
            _bar_filled = round(_bar_val / 10)
            _bar_empty  = 10 - _bar_filled
            _bar_str    = "█" * _bar_filled + "░" * _bar_empty
            _hltv_attrs.append(f"{_attr_label}: `{_bar_str}` `{round(_v)}`")
    if _hltv_attrs:
        embed.add_field(
            name="📊 HLTV Analytics",
            value="\n".join(_hltv_attrs),
            inline=False,
        )

    # ── Framework: Round Swing + Multi-kill + Player Profile ─────────────────
    _pkg_fw = result.get("grade_pkg") or {}
    _rs     = _pkg_fw.get("round_swing")    or {}
    _mk     = _pkg_fw.get("multikill")      or {}
    _pp     = _pkg_fw.get("player_profile") or {}
    _sc     = _pkg_fw.get("scenario")       or {}
    _misp   = _pkg_fw.get("misprice")       or {}

    if _rs.get("level") or _mk.get("level"):
        rs_val = (
            f"**{_rs.get('label', '—')}**\n"
            f"_{_rs.get('rationale', '')}_ \n\n"
            f"**{_mk.get('label', '—')}**\n"
            f"_{_mk.get('rationale', '')}_"
        )
        if _pp.get("label"):
            rs_val += f"\n\n**Player Profile: {_pp['label']}**\n_{_pp.get('description', '')}_"
        embed.add_field(
            name="⚙️ ROUND SWING  ·  MULTI-KILL  ·  PLAYER PROFILE",
            value=rs_val[:1020],
            inline=False,
        )

    # ── Framework: Scenario Projections ────────────────────────────────────────
    if _sc.get("short_proj") is not None:
        short_p  = _sc.get("short_proj",  "—")
        normal_p = _sc.get("normal_proj", "—")
        ceil_p   = _sc.get("ceiling",     "—")
        sc_val = (
            f"**Short-map Projection** (~{_sc.get('rounds_short', 19)} rds/map): "
            f"`{short_p}` kills  →  {_sc.get('short_edge_str', '—')}\n"
            f"**Normal-map Projection** (~{_sc.get('rounds_normal', 26)} rds/map): "
            f"`{normal_p}` kills  →  {_sc.get('normal_edge_str', '—')}\n"
            f"**Ceiling estimate:** `{ceil_p}` kills"
        )
        embed.add_field(
            name="📐 MATCH-LENGTH SCENARIOS",
            value=sc_val,
            inline=False,
        )

    # ── Framework: Mispriced Prop Verdict ─────────────────────────────────────
    if _misp.get("misprice_type") and _misp["misprice_type"] != "NONE":
        misp_bullets = _misp.get("bullets") or []
        misp_body    = "\n".join(misp_bullets[:8])
        misp_val     = f"**{_misp.get('label', '')}**\n{misp_body}"
        embed.add_field(
            name="🔎 PROP ASSESSMENT",
            value=misp_val[:1020],
            inline=False,
        )

    # ── Map Intelligence ──────────────────────────────────────────────────────
    mi_parts = []
    if map_intel.get("projected_labels"):
        mi_parts.append("**Expected:** " + " · ".join(map_intel["projected_labels"][:3]))
    if map_intel.get("projected_series") is not None:
        pvs = map_intel.get("projected_vs_line", "")
        mi_parts.append(f"**Series proj on these maps:** `{map_intel['projected_series']}` {pvs}")
    if map_intel.get("best_map") and map_intel.get("worst_map"):
        bm = map_intel["best_map"]
        wm = map_intel["worst_map"]
        mi_parts.append(f"Best: {bm[0].title()} `{bm[1]}` ↑ · Worst: {wm[0].title()} `{wm[1]}` ↓")
    # Per-map KPR breakdown (from simulator — shows kill efficiency per map name)
    _map_kpr = result.get("map_kpr") or {}
    if _map_kpr:
        _mkpr_sorted = sorted(_map_kpr.items(), key=lambda x: x[1], reverse=True)
        _mkpr_parts  = [f"{mn.title()} `{kpr:.2f}`" for mn, kpr in _mkpr_sorted[:5]]
        mi_parts.append("**KPR by Map:** " + " · ".join(_mkpr_parts))
    if mi_parts:
        embed.add_field(name="🗺️ Map Intelligence", value="\n".join(mi_parts), inline=False)

    # ── Analysis Blurb ────────────────────────────────────────────────────────
    try:
        _blurb = build_analysis_blurb(
            player_name  = player_name,
            sim_result   = result,
            form         = form,
            variance     = variance,
            deep         = deep,
            period_stats = result.get("period_stats"),
            map_intel    = map_intel if map_intel else {},
        )
        if _blurb:
            if len(_blurb) > 1020:
                _blurb = _blurb[:1017] + "…"
            embed.add_field(name="🔍 ANALYSIS", value=_blurb, inline=False)
    except Exception as _gse:
        logger.warning(f"[analysis_blurb] Failed: {_gse}")

    # ── Opponent Deep Analysis ────────────────────────────────────────────────
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
        _h2h_cmpl   = [s for s in h2h_records if not s.get("partial")]
        _h2h_n      = len(_h2h_cmpl)
        _partial_n  = len(h2h_records) - _h2h_n
        _partial_tag = f" (+{_partial_n} partial)" if _partial_n else ""
        if _h2h_n:
            _h2h_clrs = sum(1 for s in _h2h_cmpl if s.get("cleared"))
            h2h_str = f"H2H **{_h2h_clrs}/{_h2h_n}** {'✅' if _h2h_clrs == _h2h_n else '⚠️'}{_partial_tag}"
        else:
            h2h_str = "H2H: no data"
        def_lbl  = (deep.get("defensive_profile") or {}).get("label", "")
        rank_lbl = (deep.get("rank_info") or {}).get("label", "")
        opp_val  = (
            f"**Combined:** `{tot_sign}{total_pct}%`  ·  {def_lbl}  ·  {h2h_str}\n"
            f"{comp_str}\n_{rank_lbl}_"
        )
        otp = deep.get("team_period_stats") or {}
        if otp and any(otp.get(k) is not None for k in ("kpr", "rating", "adr")):
            otp_parts = []
            if otp.get("kpr")    is not None: otp_parts.append(f"KPR {otp['kpr']:.2f}")
            if otp.get("rating") is not None: otp_parts.append(f"Rtg {otp['rating']:.2f}")
            if otp.get("adr")    is not None: otp_parts.append(f"ADR {otp['adr']:.0f}")
            opp_val += f"\n_Opp 90d: {' · '.join(otp_parts)}_"
        scouting = deep.get("scouting", {})
        h2h_sc   = scouting.get("h2h_line", {})
        cleared  = h2h_sc.get("matches_cleared", 0)
        of_n     = h2h_sc.get("of_n", 0)
        partial  = h2h_sc.get("h2h_partial", 0)
        if of_n > 0 or partial > 0:
            bonus_tag   = "  ✅ +5% Over bonus" if h2h_sc.get("matchup_favorite") else ""
            partial_tag = f"  ⚠️ {partial} match(es) incomplete data" if partial else ""
            opp_val += f"\n_H2H vs line: {cleared}/{of_n} cleared{bonus_tag}{partial_tag}_"
        if h2h_records:
            _h2h_rows = []
            for _i, _rec in enumerate(h2h_records, 1):
                _total   = _rec.get("total_kills") or sum(_rec.get("kills_by_map", []))
                _maps    = _rec.get("kills_by_map", [])
                _map_str = " + ".join(str(k) for k in _maps)
                if _rec.get("partial"):
                    _icon = "⚠️"; _note = " (partial)"
                elif _rec.get("cleared"):
                    _icon = "✅"; _note = ""
                else:
                    _icon = "❌"; _note = ""
                _h2h_rows.append(f"{_icon} H2H {_i}: **{_total}** kills ({_map_str}){_note}")
            opp_val += "\n" + "\n".join(_h2h_rows)
        if result.get("stomp_high_line_warning"):
            opp_val += "\n⛔ **STOMP RISK + HIGH LINE — Lean UNDER**"
        embed.add_field(name=f"🔬 vs {opp_display}", value=opp_val, inline=False)

    # ── GURU Commentary ───────────────────────────────────────────────────────
    embed.add_field(name="💬 GURU COMMENTARY", value=guru_commentary, inline=False)

    # ── Risk Flags ────────────────────────────────────────────────────────────
    active_flags = [f for f in flags if not f.startswith("✅")]
    if active_flags:
        embed.add_field(
            name="⚠️ Risk Flags",
            value="\n".join(f"• {f}" for f in active_flags[:4]),
            inline=False,
        )

    # ── Per-Series Breakdown ──────────────────────────────────────────────────
    _breakdown  = result.get("series_breakdown", [])
    _is_hs_prop = stat_unit == "HS"
    _hs_src     = result.get("hs_rate_src", "")
    if used_fb:
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

    # ── Stand-in Warning ──────────────────────────────────────────────────────
    if result.get("standin_detected"):
        embed.add_field(
            name="🔄 ROSTER ALERT — Stand-in Detected",
            value=(
                f"⚠️ **{player_name}** appears to be playing as a **stand-in** in their most recent match.\n"
                "Stand-ins often underperform their historical averages due to limited practice "
                "with the team's system. Factor this into your bet sizing — consider reducing to 0.5u."
            ),
            inline=False,
        )

    # ── Dog Line / Market Efficiency Badge ────────────────────────────────────
    _dog = result.get("dog_line")
    if _dog:
        embed.add_field(
            name=f"💰 MARKET EDGE — {_dog['type']}",
            value=(
                f"{_dog['label']}\n"
                f"Edge vs book: **+{_dog['edge']}%**\n"
                "_Bookmaker inefficiency detected — this is the type of spot sharp bettors target._"
            ),
            inline=False,
        )

    # ── Final Bet Recommendation ──────────────────────────────────────────────
    embed.add_field(name=final_rec_name, value=final_rec_value, inline=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    data_note = "Estimated (HLTV unavailable)" if used_fb else "HLTV Live — Last 10 BO3, Maps 1&2 only"
    enrichment_parts = []
    if country:
        enrichment_parts.append(f"🌍 {country}")
    if liq_role:
        enrichment_parts.append(role_icons.get(liq_role, liq_role))
    enrichment_str = "  ·  " + "  ·  ".join(enrichment_parts) if enrichment_parts else ""
    embed.set_footer(
        text=(
            f"Elite CS2 Prop Grader  ·  Esports Betting Guru  ·  "
            f"EV+ Focus · Data-Driven · No Fluff  ·  "
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
                try:
                    save_grade(job["player"], job["line"], job["stat"], res, opponent=job.get("opponent"))
                except Exception as _ge:
                    logger.warning(f"[pp] grades_db save failed for {job['player']}: {_ge}")
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

    # ── EV+ Summary — always sent at the end of every !pp run ─────────────────
    ev_plays: list[tuple] = []
    for i, job in enumerate(jobs):
        res = results[i]
        if not res or "error" in res or res.get("used_fallback"):
            continue
        dec    = res.get("decision", "PASS")
        if dec not in ("OVER", "UNDER"):
            continue
        ev_key = "ev_over" if dec == "OVER" else "ev_under"
        ev_val = res.get(ev_key)
        if ev_val is None:
            continue
        pkg_ev = res.get("grade_pkg") or {}
        conf_ev = pkg_ev.get("confidence", res.get("confidence_score", 50))
        dog = res.get("dog_line")
        dog_tag = f"  🐕 {dog['type']}" if dog else ""
        ev_plays.append((ev_val, job, res, conf_ev, dog_tag))

    if ev_plays:
        ev_plays.sort(key=lambda x: x[0], reverse=True)
        top_ev  = [p for p in ev_plays if p[0] > 0]
        neg_ev  = [p for p in ev_plays if p[0] <= 0]
        ev_lines = []
        for ev_val, job, res, conf_ev, dog_tag in top_ev[:8]:
            dec   = res.get("decision", "?")
            grade = res.get("grade", "?")
            icon  = "✅" if dec == "OVER" else "❌"
            ev_lines.append(
                f"{icon} **{job['player']}** `{job['line']} {job['stat']}`  "
                f"EV: `+{ev_val:.3f}u`  Grade: `{grade}`  Conf: {conf_ev}/100{dog_tag}"
            )
        if neg_ev:
            ev_lines.append(f"\n_+ {len(neg_ev)} negative-EV calls omitted_")

        ev_embed = discord.Embed(
            title="🎯 EV+ PLAYS — Best Value on the Slate",
            description=(
                "\n".join(ev_lines) if ev_lines else "_No positive-EV plays found._"
            ),
            color=0xFFD700,
        )
        ev_embed.set_footer(
            text=(
                "Ranked by Expected Value vs -110 vig  ·  "
                "Esports Betting Guru  ·  EV+ Focus · Data-Driven · No Fluff  ·  Not financial advice"
            )
        )
        await ctx.send(embed=ev_embed)



# ---------------------------------------------------------------------------
# !ev — EV+ hunting: show only the best positive-EV plays from live slate
# ---------------------------------------------------------------------------

@bot.command(name="ev")
async def cmd_ev(ctx, stat_arg: str = ""):
    """
    !ev         — fetch live PrizePicks slate, show only EV+ plays sorted by edge
    !ev kills   — kills props only
    !ev hs      — headshots props only
    """
    stat_filter = None
    if stat_arg.lower() in ("hs", "headshots"):
        stat_filter = "HS"
    elif stat_arg.lower() in ("kills", "kill"):
        stat_filter = "Kills"

    status_msg = await ctx.send(
        embed=discord.Embed(
            title="🎯 EV+ Hunter — Scanning Live Slate…",
            description="Fetching PrizePicks lines and running simulations to find positive-EV plays…",
            color=0xFFD700,
        )
    )

    try:
        raw_items = await asyncio.to_thread(get_cs2_lines, None)
    except Exception as exc:
        await status_msg.edit(embed=discord.Embed(title="❌ Fetch Failed", description=str(exc)[:200], color=0xFF4136))
        return

    if not raw_items:
        await status_msg.edit(embed=discord.Embed(title="📭 No Props Found", description="No CS2 props live right now.", color=0xFFDC00))
        return

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
        player_team = (item.get("player_team") or "").strip().lower()
        home = (item.get("home_team_name") or item.get("home_team") or "").strip()
        away = (item.get("away_team_name") or item.get("away_team") or "").strip()
        opponent = away if home.lower() == player_team else home if player_team else (home or away or None)
        jobs.append({"player": pname, "stat": stat, "line": score, "item": item, "opponent": opponent})

    if not jobs:
        await status_msg.edit(embed=discord.Embed(title="📭 No Props", description="No gradeable props found.", color=0xFFDC00))
        return

    await status_msg.edit(
        embed=discord.Embed(
            title=f"🎯 EV+ Hunter — Grading {len(jobs)} Props…",
            description=f"Running analysis on {len(jobs)} props. Results stream in as each finishes (~30s each).\n_Only EV+ plays will appear in the final summary._",
            color=0xFFD700,
        )
    )

    ev_results: list[tuple] = []
    sem = asyncio.Semaphore(1)

    async def _ev_grade_one(job: dict):
        async with sem:
            try:
                _team_hint = (job.get("item") or {}).get("player_team") or None
                res = await asyncio.wait_for(
                    asyncio.to_thread(_analyze_player, job["player"], job["line"], job["stat"], job.get("opponent"), _team_hint),
                    timeout=90,
                )
            except Exception:
                return
        if not res or "error" in res or res.get("used_fallback"):
            return
        dec = res.get("decision", "PASS")
        if dec not in ("OVER", "UNDER"):
            return
        ev_key = "ev_over" if dec == "OVER" else "ev_under"
        ev_val = res.get(ev_key)
        if ev_val is None or ev_val <= 0:
            return
        pkg_r  = res.get("grade_pkg") or {}
        conf_r = pkg_r.get("confidence", 50)
        dog    = res.get("dog_line")
        ev_results.append((ev_val, job, res, conf_r, dog))

    tasks = [asyncio.create_task(_ev_grade_one(j)) for j in jobs]
    await asyncio.gather(*tasks)

    if not ev_results:
        await ctx.send(embed=discord.Embed(
            title="📭 No EV+ Plays Found",
            description="Graded all props — none cleared the positive-EV threshold vs -110 vig.",
            color=0x95A5A6,
        ))
        return

    ev_results.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for ev_val, job, res, conf_r, dog in ev_results:
        dec   = res.get("decision", "?")
        grade = res.get("grade", "?")
        icon  = "✅" if dec == "OVER" else "❌"
        dog_tag = f"  🐕 **{dog['type']}**" if dog else ""
        lines.append(
            f"{icon} **{job['player']}** `{job['line']} {job['stat']}`\n"
            f"  EV: `+{ev_val:.3f}u`  ·  Grade: `{grade}`  ·  Conf: {conf_r}/100{dog_tag}"
        )

    embed = discord.Embed(
        title=f"🎯 EV+ PLAYS — {len(ev_results)} Found ({len(jobs)} Props Scanned)",
        description="\n\n".join(lines[:10]),
        color=0xFFD700,
    )
    embed.set_footer(text="Ranked by EV vs -110 vig  ·  Esports Betting Guru  ·  Data-Driven · No Fluff  ·  Not financial advice")
    await ctx.send(embed=embed)


# ---------------------------------------------------------------------------
# !parlay — correlated parlay builder for a team's props
# ---------------------------------------------------------------------------

@bot.command(name="parlay")
async def cmd_parlay(ctx, team_name: str = "", *, extra: str = ""):
    """
    !parlay <Team>
    Grade all PrizePicks props for players on a given team playing today.
    Identifies correlated legs (same match = round correlation) and ranks
    the best combinations by combined EV.

    Example: !parlay Vitality
    """
    if not team_name:
        await ctx.send(embed=discord.Embed(
            title="❌ Usage",
            description=(
                "**Usage:** `!parlay <Team>`\n"
                "**Example:** `!parlay Vitality`\n\n"
                "Grades all PrizePicks props for players on that team, "
                "then suggests the strongest correlated parlay legs."
            ),
            color=0xFF4136,
        ))
        return

    status_msg = await ctx.send(embed=discord.Embed(
        title=f"🔗 Building Correlated Parlay — {team_name}",
        description="Fetching PrizePicks slate and finding props for this team…",
        color=0x7289DA,
    ))

    try:
        raw_items = await asyncio.to_thread(get_cs2_lines, None)
    except Exception as exc:
        await status_msg.edit(embed=discord.Embed(title="❌ Fetch Failed", description=str(exc)[:200], color=0xFF4136))
        return

    team_norm = team_name.lower().replace(" ", "")
    team_jobs: list[dict] = []
    seen: set = set()
    for item in raw_items:
        pname       = (item.get("player_name") or "").strip()
        player_team = (item.get("player_team") or "").strip().lower().replace(" ", "")
        stat        = _pp_stat_type(item)
        score       = _pp_line_score(item)
        if not pname or score is None:
            continue
        if team_norm not in player_team and player_team not in team_norm:
            continue
        key = (pname.lower(), stat)
        if key in seen:
            continue
        seen.add(key)
        home = (item.get("home_team_name") or item.get("home_team") or "").strip()
        away = (item.get("away_team_name") or item.get("away_team") or "").strip()
        opponent = away if home.lower().replace(" ", "") == player_team else home
        team_jobs.append({"player": pname, "stat": stat, "line": score, "item": item, "opponent": opponent or ""})

    if not team_jobs:
        await status_msg.edit(embed=discord.Embed(
            title="📭 No Props Found",
            description=f"No PrizePicks CS2 props found for **{team_name}** today.\nTry the exact team name from `!lines`.",
            color=0xFFDC00,
        ))
        return

    opponent_name = team_jobs[0].get("opponent", "") if team_jobs else ""
    await status_msg.edit(embed=discord.Embed(
        title=f"🔗 Correlated Parlay — {team_name}",
        description=(
            f"Found **{len(team_jobs)}** prop(s) for {team_name} players"
            + (f" vs **{opponent_name}**" if opponent_name else "") + ".\n"
            "Grading each leg — this takes ~30s per player…"
        ),
        color=0x7289DA,
    ))

    graded: list[dict] = []
    for job in team_jobs:
        try:
            _team_hint = (job.get("item") or {}).get("player_team") or None
            res = await asyncio.wait_for(
                asyncio.to_thread(_analyze_player, job["player"], job["line"], job["stat"], job.get("opponent"), _team_hint),
                timeout=90,
            )
            res["_job"] = job
            graded.append(res)
            # Stream each result as it finishes
            if "error" not in res and not res.get("used_fallback"):
                await ctx.send(embed=build_result_embed(job["player"], job["line"], job["stat"], res))
                try:
                    save_grade(job["player"], job["line"], job["stat"], res, opponent=job.get("opponent"))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"[parlay] Grade failed for {job['player']}: {exc}")

    if not graded:
        await ctx.send(embed=discord.Embed(title="❌ No results", description="All grades failed.", color=0xFF4136))
        return

    # Score correlated parlay legs using grade_engine
    scored = score_correlated_parlay(graded)

    if not scored:
        await ctx.send(embed=discord.Embed(
            title="⏸️ No Directional Legs",
            description=f"All {len(graded)} props graded PASS — no correlated parlay recommended.",
            color=0xFFDC00,
        ))
        return

    # Build parlay recommendation embed
    parlay_lines = []
    total_ev = 0.0
    for p in scored:
        job   = p.get("_job", {})
        dec   = p.get("decision", "PASS")
        grade = p.get("grade", "?")
        ev_key = "ev_over" if dec == "OVER" else "ev_under"
        ev    = p.get(ev_key) or 0
        total_ev += float(ev)
        icon  = "✅" if dec == "OVER" else "❌"
        corr  = p.get("_corr_note", "")
        dog   = p.get("dog_line")
        dog_tag = f"  🐕 {dog['type']}" if dog else ""
        parlay_lines.append(
            f"{icon} **{job.get('player','?')}** `{job.get('line','?')} {job.get('stat','?')}` — "
            f"Grade `{grade}`  EV: `{'+' if ev >= 0 else ''}{ev:.3f}u`{dog_tag}\n"
            f"  _{corr}_"
        )

    # Correlation note: same team = positively correlated (round volume)
    rounds_note = ""
    sample_rounds = scored[0].get("total_projected_rounds")
    if sample_rounds:
        if sample_rounds >= 50:
            rounds_note = f"\n✅ **Round correlation ACTIVE** — {sample_rounds} rounds projected. More rounds = more kills for all legs."
        elif sample_rounds <= 44:
            rounds_note = f"\n⚠️ **Short maps risk** — only {sample_rounds} rounds projected. Consider reducing parlay size."

    ev_sign = "+" if total_ev >= 0 else ""
    embed = discord.Embed(
        title=f"🔗 Correlated Parlay — {team_name}" + (f" vs {opponent_name}" if opponent_name else ""),
        description=(
            f"**{len(scored)} Legs Graded** · Combined EV: `{ev_sign}{total_ev:.3f}u`"
            f"{rounds_note}\n\n"
            + "\n\n".join(parlay_lines)
            + "\n\n_Same-team legs share round-count correlation — "
            "when the match goes long, all players benefit._"
        ),
        color=0x7289DA,
    )
    embed.set_footer(text="Correlated Parlay  ·  Esports Betting Guru  ·  EV+ Focus · Data-Driven  ·  Not financial advice")
    await ctx.send(embed=embed)


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
# !result  — log an actual stat total against a graded prop
# ---------------------------------------------------------------------------

@bot.command(name="result")
async def cmd_result(ctx, player: str = None, actual: str = None):
    """
    Record the actual outcome for a graded prop.
    Usage: !result <player> <actual>
    Example: !result lucky 32
    """
    if not player or actual is None:
        await ctx.send(
            embed=discord.Embed(
                title="Usage",
                description="`!result <player> <actual_kills>`\nExample: `!result lucky 32`",
                color=0x95A5A6,
            )
        )
        return

    try:
        actual_val = float(actual)
    except ValueError:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Invalid value",
                description=f"`{actual}` is not a valid number.",
                color=0xFF4136,
            )
        )
        return

    entry = record_result(player, actual_val)
    if not entry:
        await ctx.send(
            embed=discord.Embed(
                title="❌ Not found",
                description=f"No pending grade found for **{player}**. Run `!grade` or `!pp` first.",
                color=0xFF4136,
            )
        )
        return

    outcome = entry["outcome"]
    line    = entry["line"]
    rec     = (entry.get("recommendation") or "?").upper()
    stat    = entry.get("stat", "Kills")
    opp     = entry.get("opponent") or "?"

    outcome_icon = {"over": "📈", "under": "📉", "push": "➡️"}.get(outcome, "❓")
    outcome_label = outcome.upper() if outcome else "?"

    # Did the bot's recommendation match the outcome?
    bot_correct = (rec == "OVER" and outcome == "over") or (rec == "UNDER" and outcome == "under")
    correct_icon = "✅" if bot_correct else ("❌" if rec != "PASS" else "—")

    color = 0x2ECC71 if bot_correct else (0xFF4136 if rec != "PASS" else 0x95A5A6)

    embed = discord.Embed(
        title=f"{outcome_icon} Result Logged — {entry['display']}",
        color=color,
    )
    embed.add_field(name="Line",     value=f"{line} {stat}",  inline=True)
    embed.add_field(name="Actual",   value=str(actual_val),   inline=True)
    embed.add_field(name="Outcome",  value=outcome_label,     inline=True)
    embed.add_field(name="Bot Pick", value=rec,               inline=True)
    embed.add_field(name="Opponent", value=opp,               inline=True)
    embed.add_field(name="Result",   value=f"{correct_icon} {'HIT' if bot_correct else ('MISS' if rec != 'PASS' else 'N/A')}",
                    inline=True)

    await ctx.send(embed=embed)


# ---------------------------------------------------------------------------
# !results — show grade history with outcomes
# ---------------------------------------------------------------------------

@bot.command(name="results")
async def cmd_results(ctx, when: str = "today"):
    """
    Show grade history with outcomes.
    Usage: !results [today|yesterday|pending|week]
    """
    from datetime import datetime, timezone, timedelta

    when_lower = when.lower()
    today_str     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    if when_lower in ("today",):
        target_date = today_str
        entries = get_entries_for_date(target_date)
        title = f"📊 Results — {date_label(target_date)}"
    elif when_lower in ("yesterday", "yday", "ytd"):
        target_date = yesterday_str
        entries = get_entries_for_date(target_date)
        title = f"📊 Results — {date_label(target_date)}"
    elif when_lower in ("pending", "open"):
        entries = get_pending_entries()
        title = "⏳ Pending Grades (no result entered yet)"
    elif when_lower in ("week", "7d"):
        entries = get_recent_entries(days=7)
        title = "📊 Results — Last 7 Days"
    else:
        # Try parsing as a date like "Apr5" or "04-05"
        target_date = yesterday_str
        entries = get_entries_for_date(target_date)
        title = f"📊 Results — {date_label(target_date)}"

    if not entries:
        await ctx.send(
            embed=discord.Embed(
                title=title,
                description="No grades found for that period.",
                color=0x95A5A6,
            )
        )
        return

    # Build rows
    lines_out = []
    correct = wrong = pending = 0

    for e in sorted(entries, key=lambda x: x["ts"]):
        rec     = (e.get("recommendation") or "?").upper()
        outcome = e.get("outcome")
        actual  = e.get("actual")
        line    = e["line"]
        stat    = e.get("stat", "Kills")
        player  = e.get("display") or e["player"]
        opp     = e.get("opponent") or "?"

        if outcome is None:
            status = "⏳"
            actual_str = "—"
            pending += 1
        elif (rec == "OVER" and outcome == "over") or (rec == "UNDER" and outcome == "under"):
            status = "✅"
            actual_str = str(actual)
            correct += 1
        elif rec == "PASS":
            status = "—"
            actual_str = str(actual)
        else:
            status = "❌"
            actual_str = str(actual)
            wrong += 1

        over_pct  = e.get("over_pct")
        grade_str = f" ({e['grade']})" if e.get("grade") and e["grade"] != "N/A" else ""

        lines_out.append(
            f"{status} **{player}** {line} {stat} vs {opp}\n"
            f"   Bot: **{rec}** {f'({over_pct}% over)' if over_pct else ''}{grade_str} | "
            f"Actual: **{actual_str}** {('**' + outcome.upper() + '**') if outcome else ''}"
        )

    total_decided = correct + wrong
    record_str = f"{correct}W-{wrong}L" if total_decided else "No results yet"
    if pending:
        record_str += f" ({pending} pending)"

    embed = discord.Embed(title=title, color=0x3498DB)
    embed.description = "\n\n".join(lines_out)
    embed.set_footer(text=f"Record: {record_str}  |  Use !result <player> <actual> to log outcomes")

    # Discord embeds have a 4096-char description limit — split if needed
    if len(embed.description) > 4096:
        chunks = []
        chunk = []
        for row in lines_out:
            if sum(len(r) + 2 for r in chunk) + len(row) > 4000:
                chunks.append(chunk)
                chunk = []
            chunk.append(row)
        if chunk:
            chunks.append(chunk)
        for i, ch in enumerate(chunks):
            e2 = discord.Embed(
                title=title if i == 0 else f"{title} (cont.)",
                description="\n\n".join(ch),
                color=0x3498DB,
            )
            if i == len(chunks) - 1:
                e2.set_footer(text=f"Record: {record_str}  |  !result <player> <actual> to log outcomes")
            await ctx.send(embed=e2)
        return

    await ctx.send(embed=embed)


# ---------------------------------------------------------------------------
# !fetchresults — auto-scrape HLTV to fill in pending outcomes
# ---------------------------------------------------------------------------

@bot.command(name="fetchresults")
async def cmd_fetchresults(ctx, when: str = "today"):
    """
    Auto-fetch actual match results from HLTV for all pending graded props.
    Usage: !fetchresults [today|yesterday]
    """
    from datetime import datetime, timezone, timedelta

    today_str     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    target_date   = yesterday_str if when.lower() in ("yesterday", "yday") else today_str

    pending = [e for e in get_entries_for_date(target_date) if e.get("outcome") is None]

    if not pending:
        await ctx.send(embed=discord.Embed(
            title="✅ Nothing Pending",
            description=f"No unresolved grades for {target_date}.",
            color=0x2ECC71,
        ))
        return

    msg = await ctx.send(embed=discord.Embed(
        title="🔍 Fetching Results…",
        description=f"Checking HLTV for {len(pending)} pending grade(s). This may take a minute…",
        color=0xF39C12,
    ))

    found = 0
    skipped = 0
    not_ready = 0
    rows = []

    for entry in pending:
        player  = entry["player"]
        display = entry.get("display") or player
        opp     = entry.get("opponent") or ""
        line    = entry["line"]
        grade_ts = entry["ts"]
        baseline = entry.get("baseline_match_id")

        try:
            res = await asyncio.to_thread(
                _scraper_get_actual_result,
                player, opp, grade_ts, line, baseline
            )
        except Exception as ex:
            logger.warning(f"[fetchresults] Error for {player}: {ex}")
            res = None

        if res is None:
            not_ready += 1
            rows.append(f"⏳ **{display}** {line} — not played / not finished yet")
            continue

        actual  = res["actual"]
        outcome = res["outcome"]
        rec     = (entry.get("recommendation") or "?").upper()

        updated = record_result(player, actual, entry_id=entry["id"])
        if updated:
            found += 1
            if rec == "PASS":
                icon = "—"
            elif (rec == "OVER" and outcome == "over") or (rec == "UNDER" and outcome == "under"):
                icon = "✅"
            else:
                icon = "❌"
            rows.append(f"{icon} **{display}** {line} | Bot: **{rec}** | Actual: **{actual}** ({outcome.upper()})")
        else:
            skipped += 1
            rows.append(f"⚠️ **{display}** — could not update record")

    summary = f"Updated **{found}** result(s). {not_ready} not ready yet."
    if skipped:
        summary += f" {skipped} skipped."

    embed = discord.Embed(
        title=f"📊 Auto-Results — {target_date}",
        description="\n".join(rows) if rows else "Nothing to show.",
        color=0x2ECC71 if found else 0x95A5A6,
    )
    embed.set_footer(text=summary)

    try:
        await msg.edit(embed=embed)
    except Exception:
        await ctx.send(embed=embed)


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


# ===========================================================================
# VALORANT COMMANDS (vlr.gg data, kills only, last 10 BO3 series Maps 1+2)
# ===========================================================================

import valorant_scraper as _vlr  # noqa: E402
import prizepicks_valorant as _ppv  # noqa: E402

VAL_DECISION_COLORS = {
    "OVER":   0x2ECC71,
    "UNDER":  0xE74C3C,
    "PASS":   0x95A5A6,
}


def _build_valorant_embed(
    player_name: str,
    line: float,
    info: dict,
    sim: dict,
    book_implied: float,
) -> discord.Embed:
    """Compact Valorant prop-grading embed (kills only)."""
    decision = sim.get("decision", "PASS")
    color    = VAL_DECISION_COLORS.get(decision, 0x7289DA)

    # Group map_stats into series for display
    series: dict = {}
    for m in info["map_stats"]:
        series.setdefault(m["match_id"], []).append(m)
    series_list = list(series.values())  # newest-first

    # Header projection words
    if   decision == "OVER":  proj_word, proj_icon = "MORE", "✅"
    elif decision == "UNDER": proj_word, proj_icon = "LESS", "❌"
    else:                     proj_word, proj_icon = "NO BET", "⏸️"

    # Confidence grade
    over_p = sim.get("over_prob", 50.0)
    edge   = sim.get("edge", 0)
    confidence = min(99, max(1, int(50 + abs(edge) * 1.4)))
    if   confidence >= 80: conf_chr = "A"
    elif confidence >= 65: conf_chr = "B"
    elif confidence >= 50: conf_chr = "C"
    elif confidence >= 35: conf_chr = "D"
    else:                  conf_chr = "F"

    embed = discord.Embed(
        title=f"{proj_icon}  {info['display_name']}  ·  {line:g} {sim['stat_type']}  ·  {decision}",
        description=(
            f"**Projection:** {proj_word} {line:g}  ·  "
            f"**Confidence:** {conf_chr} ({confidence}/100)\n"
            f"**Game:** Valorant  ·  **Source:** vlr.gg  ·  "
            f"**Series:** {sim['n_series']}  ·  **Maps:** {sim['n_samples']}"
        ),
        color=color,
    )

    # Probability box
    embed.add_field(
        name="📊 Predictive Model — Calibrated OVER probability",
        value=(
            f"**OVER {line:g}:**  {sim['over_prob']:.1f}%\n"
            f"**UNDER {line:g}:** {sim['under_prob']:.1f}%\n"
            f"**Edge vs −110:** {sim['edge']:+.1f}%\n"
            f"**Projected total:** {sim.get('expected_total', 0):.1f} kills  "
            f"(KPR {sim.get('blended_kpr', 0):.2f} × {sim.get('proj_rounds', 0):.0f} rounds)\n"
            f"**Confidence band:** {sim.get('ci_low', 0):.1f} — {sim.get('ci_high', 0):.1f} kills"
        ),
        inline=False,
    )

    # Model signal breakdown — shows WHY we lean the way we do
    z_proj  = sim.get("z_projection", 0)
    z_med   = sim.get("z_median", 0)
    z_trend = sim.get("z_trend", 0)
    z_hit   = sim.get("z_bayes_hit", 0)
    z_opp   = sim.get("z_opponent", 0)
    def _arrow(z):
        if z >=  0.5: return "🟢 OVER"
        if z <= -0.5: return "🔴 UNDER"
        return "⚪ neutral"
    align_mult = sim.get("alignment_mult", 1.0)
    so = sim.get("signals_over", 0); su = sim.get("signals_under", 0)
    align_emoji = "✅" if align_mult >= 0.85 else ("⚖️" if align_mult >= 0.6 else "⚠️")
    embed.add_field(
        name="🧠 Signal Stack (z-scores → consensus)",
        value=(
            f"**Projection:** `{z_proj:+.2f}σ` {_arrow(z_proj)}\n"
            f"**Line vs Median:** `{z_med:+.2f}σ` {_arrow(z_med)}\n"
            f"**Recency Trend:** `{z_trend:+.2f}σ` ({sim.get('trend_pct', 0):+.0f}%)  {_arrow(z_trend)}\n"
            f"**Bayes Hit Rate:** `{z_hit:+.2f}σ` ({sim.get('bayes_hit_rate', 0):.0f}% smoothed)  {_arrow(z_hit)}\n"
            f"**Opp-Tier Shift:** `{z_opp:+.2f}σ`  {_arrow(z_opp)}\n"
            f"**Alignment:** {align_emoji} {so}🟢/{su}🔴  ×{align_mult:.2f} dampener\n"
            f"**Model score:** `{sim.get('model_score', 0):+.2f}` → P(OVER) `{sim['over_prob']:.1f}%`"
        ),
        inline=False,
    )

    # Opponent context block (NEW)
    avg_opp = sim.get("avg_opp_rating")
    if avg_opp:
        opp_filtered = "🛡️ filtered (dropped bottom-25% opp games)" if sim.get("opp_quality_filtered") else "📊 full sample"
        embed.add_field(
            name="🎯 Opponent Context",
            value=(
                f"**Avg opp rating:** {avg_opp}  ·  {opp_filtered}\n"
                f"**KPR split:** {sim.get('opp_split_label', 'n/a')}\n"
                f"**Robust avg (10/10 trim):** {sim.get('trimmed_avg', 0):.1f}  ·  "
                f"**MAD-σ:** {sim.get('sigma_mad', 0):.1f}"
            ),
            inline=False,
        )

    # Recent form
    form_lines = []
    for i, maps in enumerate(series_list[:10], 1):
        total = sum(m["stat_value"] for m in maps)
        per   = " + ".join(f"{m['stat_value']}" for m in maps)
        form_lines.append(f"`{i:2d}.` **{total}** kills  ·  {per}")
    embed.add_field(
        name=f"🎯 Last {len(series_list)} BO3 Series (Maps 1 + 2)",
        value="\n".join(form_lines) or "_no data_",
        inline=False,
    )

    # Stats summary
    embed.add_field(
        name="📈 Historical (M1+M2 per series)",
        value=(
            f"**Avg:** {sim['hist_avg']:.1f}  ·  "
            f"**Median:** {sim['hist_median']:.1f}  ·  "
            f"**Hit-rate vs line:** {sim['hit_rate']:.0f}%\n"
            f"**Ceiling:** {sim['ceiling']}  ·  "
            f"**Floor:** {sim['floor']}  ·  "
            f"**Stability:** {sim.get('stability_label', 'N/A')}"
        ),
        inline=False,
    )

    # ── Confidence score (universal 0-100, sortable across players) ─────────
    decision = sim.get("decision", "PASS")
    edge_signed = (sim.get("over_prob", 50) - 50) / 100  # signed edge: + means OVER bias
    series_totals = []
    series_map: dict = {}
    for m in info["map_stats"]:
        series_map.setdefault(m["match_id"], []).append(m["stat_value"])
    series_totals = [sum(v) for v in series_map.values()]
    s_avg = sum(series_totals) / len(series_totals) if series_totals else 0
    import statistics as _stats
    s_std = _stats.stdev(series_totals) if len(series_totals) > 1 else 0.0
    r_avg, o_avg, trend_pct = _vlr.split_recent_vs_older(info["map_stats"], recent_n_series=3)
    conf = _vlr.confidence_score(
        edge=edge_signed,
        hit_rate=(sim.get("hit_rate", 0) or 0) / 100,
        n_series=len(series_totals),
        stability_std=s_std,
        sample_avg=s_avg,
        trend_pct=trend_pct,
        decision=decision,
    )
    conf_letter, conf_label = _vlr.confidence_grade(conf)
    bar_len = 14
    filled = int(round(conf / 100 * bar_len))
    conf_bar = "█" * filled + "░" * (bar_len - filled)

    # Recency trend label
    if trend_pct >= 8:
        trend_str = f"🔥 Heating up (+{trend_pct:.1f}% recent vs older)"
    elif trend_pct <= -8:
        trend_str = f"🧊 Cooling off ({trend_pct:.1f}% recent vs older)"
    else:
        trend_str = f"➖ Steady ({trend_pct:+.1f}%)"

    embed.add_field(
        name=f"🎯 Confidence — {conf}/100 · Grade {conf_letter}",
        value=f"`{conf_bar}`  {conf_label}\n**Recency:** {trend_str}  ·  Recent3: **{r_avg:.1f}**  ·  Older7: **{o_avg:.1f}**",
        inline=False,
    )

    # ── Player analytics (vlr per-map aggregates) ────────────────────────────
    agg = _vlr.aggregate_stats(info["map_stats"])
    role_str = _vlr.infer_role(agg) if agg else None
    if agg:
        rating_str = f"{agg['rating']:.2f}" if agg.get("rating") is not None else "—"
        kd_str     = f"{agg['kd']:.2f}"      if agg.get("kd")     is not None else "—"
        kpr_str    = f"{agg['kpr']:.2f}"     if agg.get("kpr")    is not None else "—"
        dpr_str    = f"{agg['dpr']:.2f}"     if agg.get("dpr")    is not None else "—"
        apr_str    = f"{agg['apr']:.2f}"     if agg.get("apr")    is not None else "—"
        acs_str    = f"{agg['acs']:.0f}"     if agg.get("acs")    is not None else "—"
        adr_str    = f"{agg['adr']:.0f}"     if agg.get("adr")    is not None else "—"
        kast_str   = f"{agg['kast']:.0f}%"   if agg.get("kast")   is not None else "—"
        hs_str     = f"{agg['hs_pct']:.0f}%" if agg.get("hs_pct") is not None else "—"
        fk_str     = f"{agg['fk_rate']:.3f}" if agg.get("fk_rate") is not None else "—"
        fd_str     = f"{agg['fd_rate']:.3f}" if agg.get("fd_rate") is not None else "—"
        share_str  = f"{agg['fk_share']:.0f}%" if agg.get("fk_share") is not None else "—"

        role_line = f"**Role (inferred):** {role_str}\n" if role_str else ""
        embed.add_field(
            name="🎮 Player Analytics (last 10 series)",
            value=(
                f"{role_line}"
                f"**Rating** {rating_str}  ·  "
                f"**ACS** {acs_str}  ·  "
                f"**K/D** {kd_str}  ·  "
                f"**ADR** {adr_str}  ·  "
                f"**KAST** {kast_str}  ·  "
                f"**HS%** {hs_str}\n"
                f"**KPR** {kpr_str}  ·  "
                f"**DPR** {dpr_str}  ·  "
                f"**APR** {apr_str}  ·  "
                f"**FK/r** {fk_str}  ·  "
                f"**FD/r** {fd_str}  ·  "
                f"**FK win%** {share_str}\n"
                f"_Sample: {agg['n_maps']} maps · {agg['n_rounds']} rounds · "
                f"{agg['total_kills']}K / {agg['total_deaths']}D total_"
            ),
            inline=False,
        )

    # ── Per-map breakdown — spot map-veto edges ─────────────────────────────
    per_map = _vlr.per_map_breakdown(info["map_stats"])
    if per_map:
        # Show top 6 maps by avg, with sample sizes ≥ 1
        rows = []
        for entry in per_map[:6]:
            kpr = f"  ·  KPR `{entry['kpr']:.2f}`" if entry.get("kpr") is not None else ""
            rows.append(
                f"`{entry['map_name']:<10s}` n={entry['n']}  ·  "
                f"avg **{entry['avg']:.1f}**  ·  range {entry['min']}–{entry['max']}{kpr}"
            )
        embed.add_field(
            name="🗺️ Per-Map Breakdown (kills per map)",
            value="\n".join(rows) + "\n_Higher avg/KPR = better map for OVER. Cross-check with the upcoming map veto._",
            inline=False,
        )

    embed.set_footer(text="Valorant · vlr.gg · Last 10 BO3 · Maps 1 & 2 only · 100K sims")
    return embed


# ───────────────────────────────────────────────────────────────────────────
# !vteam — Valorant team-wide grader: ranks all players on a team by
#         best edge × confidence so you can pick the strongest play.
# ───────────────────────────────────────────────────────────────────────────

@bot.command(name="vteam", aliases=["vt"])
async def cmd_vteam(ctx, *, team_arg: str = ""):
    """
    !vteam <Team>  — Grade every player on a Valorant team at their projected
    line and rank by Edge × Confidence. Surfaces the single best OVER/UNDER
    play across the squad.
    """
    name = team_arg.strip()
    if not name:
        await ctx.send(embed=discord.Embed(
            title="❌ Usage",
            description="`!vteam <team name>`\nExample: `!vteam Sentinels`",
            color=0xFF4136,
        ))
        return

    msg = await ctx.send(embed=discord.Embed(
        title=f"🔎 Searching vlr.gg for team `{name}`…",
        color=0x9146FF,
    ))

    team = await asyncio.to_thread(_vlr.search_team, name)
    if not team:
        await msg.edit(embed=discord.Embed(
            title="❌ Team Not Found",
            description=f"Couldn't find Valorant team `{name}` on vlr.gg.",
            color=0xFF4136,
        ))
        return
    team_id, team_slug = team
    roster = await asyncio.to_thread(_vlr.get_team_roster, team_id, team_slug)
    if not roster:
        await msg.edit(embed=discord.Embed(
            title="❌ Empty Roster",
            description=f"vlr.gg team `{team_slug}` returned no players.",
            color=0xFF4136,
        ))
        return

    # ── Pull live PrizePicks lines for this team ──────────────────────────
    try:
        team_lines = await asyncio.to_thread(_ppv.get_valorant_lines_for_team, team_slug)
    except Exception as exc:
        logger.warning(f"[vteam] PP fetch failed: {exc}")
        team_lines = []

    # Filter to MAPS 1-2 Kills (matches our scraper scope) and dedupe by player
    pp_by_player: dict[str, dict] = {}
    for it in team_lines:
        if "kill" not in (it.get("stat_type") or "").lower():
            continue
        if "maps 1-2" not in (it.get("stat_type") or "").lower():
            continue
        if (it.get("line") is None) or it.get("stat_type", "").lower().endswith("(combo)"):
            continue
        nm = (it.get("player_name") or "").lower()
        if nm and nm not in pp_by_player:
            pp_by_player[nm] = it

    using_real_lines = len(pp_by_player) > 0
    line_source_label = (
        f"📊 **Real PrizePicks lines** loaded for {len(pp_by_player)}/{len(roster)} players"
        if using_real_lines
        else "⚠️ No live PrizePicks lines for this team — falling back to synthetic median lines"
    )

    await msg.edit(embed=discord.Embed(
        title=f"🎯 Grading {team_slug.upper()} — {len(roster)} players",
        description=(
            f"{line_source_label}\n\n"
            f"Pulling last-10-BO3 data for each player and grading. "
            f"This usually takes ~20-40s per player."
        ),
        color=0x9146FF,
    ))

    import statistics as _stats
    graded: list[dict] = []
    for p in roster:
        try:
            info = await asyncio.to_thread(
                _vlr.get_player_info, p["display_name"] or p["slug"], 10
            )
        except Exception as exc:
            logger.warning(f"[vteam] {p['slug']} fetch failed: {exc}")
            continue
        if not info or not info.get("map_stats"):
            continue

        # Build series totals
        series_map: dict = {}
        for m in info["map_stats"]:
            series_map.setdefault(m["match_id"], []).append(m["stat_value"])
        totals = sorted(sum(v) for v in series_map.values())
        if len(totals) < 4:
            continue

        # ── Choose the LINE: real PrizePicks line if available, else synthetic
        resolved_name = (info.get("display_name") or p["slug"]).lower()
        pp_match = pp_by_player.get(resolved_name) or pp_by_player.get(p["slug"].lower())
        if pp_match:
            line_f = float(pp_match["line"])
            line_src = "PP"
        else:
            med = totals[len(totals) // 2]
            line_f = float(med) - 0.5
            line_src = "synthetic"

        map_stats = []
        for m in info["map_stats"]:
            mm = dict(m); mm.setdefault("rounds", 24)
            map_stats.append(mm)

        try:
            sim = _vlr.empirical_grade(map_stats, line_f, "Kills")
        except Exception as exc:
            logger.warning(f"[vteam] {p['slug']} grade failed: {exc}")
            continue
        if sim.get("error"):
            continue

        decision = sim.get("decision", "PASS")
        edge_signed = (sim.get("over_prob", 50) - 50) / 100
        s_avg = sum(totals) / len(totals)
        s_std = _stats.stdev(totals) if len(totals) > 1 else 0.0
        _, _, trend_pct = _vlr.split_recent_vs_older(info["map_stats"], 3)
        conf = _vlr.confidence_score(
            edge=edge_signed,
            hit_rate=(sim.get("hit_rate", 0) or 0) / 100,
            n_series=len(totals),
            stability_std=s_std,
            sample_avg=s_avg,
            trend_pct=trend_pct,
            decision=decision,
        )
        agg = _vlr.aggregate_stats(info["map_stats"])
        graded.append({
            "name":      info.get("display_name") or p["slug"],
            "line":      line_f,
            "line_src":  line_src,
            "decision":  decision,
            "edge_pct":  abs(sim.get("over_prob", 50) - 50),
            "conf":      conf,
            "score":     conf * (1 + abs(sim.get("over_prob", 50) - 50) / 100),
            "hit_rate":  sim.get("hit_rate", 0),
            "hist_avg":  sim.get("hist_avg", 0),
            "trend":     trend_pct,
            "role":      _vlr.infer_role(agg),
            "ev_over":   sim.get("ev_over", 0),
            "ev_under":  sim.get("ev_under", 0),
        })

    if not graded:
        await msg.edit(embed=discord.Embed(
            title="❌ No Gradeable Players",
            description=f"Couldn't gather enough data on **{team_slug}**'s roster to grade.",
            color=0xFF4136,
        ))
        return

    # Rank: confidence × (1 + edge), filter PASS to bottom
    actionable = [g for g in graded if g["decision"] in ("OVER", "UNDER")]
    actionable.sort(key=lambda g: g["score"], reverse=True)
    passes = [g for g in graded if g["decision"] == "PASS"]
    passes.sort(key=lambda g: g["conf"], reverse=True)
    ranked = actionable + passes

    # Build leaderboard
    lines: list[str] = []
    for i, g in enumerate(ranked, 1):
        if g["decision"] == "OVER":
            icon, dec_str, ev = "✅", f"OVER {g['line']:.1f}", g["ev_over"]
        elif g["decision"] == "UNDER":
            icon, dec_str, ev = "❌", f"UNDER {g['line']:.1f}", g["ev_under"]
        else:
            icon, dec_str, ev = "⏸️", "PASS", 0
        ev_str = f"{'+' if (ev or 0) >= 0 else ''}{(ev or 0):.3f}u"
        trend_arrow = "🔥" if g["trend"] >= 8 else ("🧊" if g["trend"] <= -8 else "➖")
        line_badge = " 📊PP" if g.get("line_src") == "PP" else " ⚙️syn"
        lines.append(
            f"`#{i}` {icon} **{g['name']}** {g['role']}{line_badge}\n"
            f"   → `{dec_str}`  ·  Conf **{g['conf']}/100**  ·  Edge **{g['edge_pct']:.1f}%**  "
            f"·  Hit `{g['hit_rate']:.0f}%`  ·  Avg `{g['hist_avg']:.1f}`  ·  EV `{ev_str}`  {trend_arrow}"
        )

    best = ranked[0]
    if best["decision"] in ("OVER", "UNDER"):
        headline = (
            f"🏆 **Best Play:** {best['name']} `{best['decision']} {best['line']:.1f}` "
            f"— Confidence **{best['conf']}/100**, Edge **{best['edge_pct']:.1f}%**"
        )
    else:
        headline = "⚖️ No high-edge plays on this team — all legs grade PASS."

    embed = discord.Embed(
        title=f"🎯 Team Grade — {team_slug.upper()} ({len(ranked)} players)",
        description=headline + "\n\n" + "\n\n".join(lines),
        color=0x9146FF,
    )
    embed.set_footer(text=(
        f"Valorant · vlr.gg · Lines = each player's median series total − 0.5  ·  "
        f"Ranked by Confidence × (1+Edge)  ·  100K sims each"
    ))
    await msg.edit(embed=embed)


@bot.command(name="vgrade", aliases=["vg"])
async def cmd_vgrade(ctx, player_name: str = None, line: str = None, *args):
    """Grade a Valorant kills prop. Usage: !vgrade <player> [line]"""
    if not player_name:
        await ctx.send(embed=discord.Embed(
            title="❌ Usage",
            description=(
                "**`!vgrade <player> [line] [odds?]`**\n\n"
                "**Examples:**\n"
                "`!vgrade tenz`         ← auto-fetches live PrizePicks line\n"
                "`!vgrade tenz 32.5`\n"
                "`!vgrade aspas 35.5 -110`\n\n"
                "Pulls last 10 BO3 series (Maps 1 & 2 only) from vlr.gg, "
                "grades empirically off the historical series distribution, returns OVER/UNDER/PASS with confidence."
            ),
            color=0xFF4136,
        ))
        return

    pp_auto = False
    line_f: float | None = None
    if line is not None:
        try:
            line_f = float(line)
        except ValueError:
            # Maybe user passed odds in the line slot — try as odds, fall through to auto-line
            args = (line, *args)
            line_f = None
    if line_f is None:
        # Auto-fetch live PrizePicks line for MAPS 1-2 Kills
        try:
            pp = await asyncio.to_thread(_ppv.get_valorant_player_line, player_name, "MAPS 1-2 Kills")
        except Exception as exc:
            logger.warning(f"[vgrade] PP auto-fetch failed: {exc}")
            pp = None
        if pp and pp.get("line") is not None:
            line_f = float(pp["line"])
            pp_auto = True
        else:
            await ctx.send(embed=discord.Embed(
                title="❌ No Line Provided",
                description=(
                    f"No live PrizePicks `MAPS 1-2 Kills` line found for **{player_name}**.\n"
                    f"Please provide a line: `!vgrade {player_name} 28.5`"
                ),
                color=0xFF4136,
            ))
            return

    # Optional book odds
    book_implied = 0.5238
    for a in args:
        m = re.match(r'^([+-])(\d{2,4})$', a)
        if m:
            v = int(a)
            book_implied = (100 / (v + 100)) if v > 0 else (abs(v) / (abs(v) + 100))
            break

    msg = await ctx.send(f"🔎 Looking up `{player_name}` on vlr.gg…")
    info = _vlr.get_player_info(player_name, n_series=10)
    if not info or not info.get("map_stats"):
        await msg.edit(content=f"❌ No Valorant data found for `{player_name}`.")
        return

    # Inject default rounds (Valorant: first to 13 → ~24 rounds typical)
    map_stats = []
    for m in info["map_stats"]:
        mm = dict(m)
        mm.setdefault("rounds", 24)
        map_stats.append(mm)

    sim = _vlr.empirical_grade(map_stats, line_f, "Kills")
    if sim.get("error"):
        await msg.edit(content=f"❌ Grading error: {sim['error']}")
        return

    embed = _build_valorant_embed(player_name, line_f, info, sim, book_implied)
    if pp_auto:
        existing_footer = embed.footer.text or ""
        embed.set_footer(text=f"📊 Live PrizePicks line auto-fetched · {existing_footer}")
    await msg.edit(content=None, embed=embed)


# ───────────────────────────────────────────────────────────────────────────
# !vpp — grade the entire live Valorant PrizePicks slate
# ───────────────────────────────────────────────────────────────────────────

@bot.command(name="vpp")
async def cmd_vpp(ctx, *, filter_arg: str = ""):
    """
    !vpp                  — list/grade ALL live Valorant PrizePicks props (top 15 by edge)
    !vpp <Team>           — only props for a given team
    !vpp refresh          — bust the cache and pull fresh
    """
    if filter_arg.strip().lower() == "refresh":
        _ppv.invalidate_cache()
        await ctx.send(embed=discord.Embed(
            title="🔄 Cache Cleared",
            description="Next `!vpp` pulls fresh PrizePicks Valorant data.",
            color=0x9146FF,
        ))
        return

    msg = await ctx.send(embed=discord.Embed(
        title="📡 Fetching Valorant PrizePicks Slate…",
        description="Scraping live PrizePicks board through proxy (~10-20s)…",
        color=0x9146FF,
    ))

    try:
        all_props = await asyncio.to_thread(_ppv.get_valorant_lines)
    except Exception as exc:
        await msg.edit(embed=discord.Embed(
            title="❌ Slate Fetch Failed",
            description=f"`{str(exc)[:200]}`",
            color=0xFF4136,
        )); return

    # Keep MAPS 1-2 Kills only (matches our scraper scope)
    props = [
        p for p in all_props
        if "kill" in (p.get("stat_type") or "").lower()
        and "maps 1-2" in (p.get("stat_type") or "").lower()
        and not (p.get("stat_type") or "").lower().endswith("(combo)")
    ]
    if filter_arg.strip():
        nm = filter_arg.strip().lower().replace(" ", "")
        props = [
            p for p in props
            if nm in (p.get("player_team") or "").lower().replace(" ", "")
            or nm in (p.get("player_name") or "").lower()
        ]
    if not props:
        await msg.edit(embed=discord.Embed(
            title="📭 No Props Found",
            description="No live Valorant `MAPS 1-2 Kills` props match." +
                        (f" Filter: `{filter_arg}`" if filter_arg else ""),
            color=0xFFDC00,
        )); return

    await msg.edit(embed=discord.Embed(
        title=f"🎯 Grading {len(props)} Valorant Props",
        description=f"Empirically grading each player off last-10 series. ETA ~{len(props)*15}s.",
        color=0x9146FF,
    ))

    import statistics as _stats
    graded: list[dict] = []
    for prop in props:
        pname = prop["player_name"]
        line_f = float(prop["line"])
        try:
            info = await asyncio.to_thread(_vlr.get_player_info, pname, 10)
        except Exception:
            continue
        if not info or not info.get("map_stats"):
            continue
        series_map: dict = {}
        for m in info["map_stats"]:
            series_map.setdefault(m["match_id"], []).append(m["stat_value"])
        totals = [sum(v) for v in series_map.values()]
        if len(totals) < 4:
            continue
        ms = []
        for m in info["map_stats"]:
            mm = dict(m); mm.setdefault("rounds", 24); ms.append(mm)
        try:
            sim = _vlr.empirical_grade(ms, line_f, "Kills")
        except Exception:
            continue
        if sim.get("error"):
            continue
        decision = sim.get("decision", "PASS")
        edge_signed = (sim.get("over_prob", 50) - 50) / 100
        s_avg = sum(totals)/len(totals)
        s_std = _stats.stdev(totals) if len(totals) > 1 else 0.0
        _, _, trend_pct = _vlr.split_recent_vs_older(info["map_stats"], 3)
        conf = _vlr.confidence_score(
            edge=edge_signed, hit_rate=(sim.get("hit_rate", 0) or 0)/100,
            n_series=len(totals), stability_std=s_std, sample_avg=s_avg,
            trend_pct=trend_pct, decision=decision)
        graded.append({
            "name":     info.get("display_name") or pname,
            "team":     prop.get("player_team", ""),
            "line":     line_f,
            "decision": decision,
            "edge_pct": abs(sim.get("over_prob", 50) - 50),
            "conf":     conf,
            "score":    conf * (1 + abs(sim.get("over_prob", 50) - 50)/100),
            "hit_rate": sim.get("hit_rate", 0),
            "hist_avg": sim.get("hist_avg", 0),
            "ev_over":  sim.get("ev_over", 0),
            "ev_under": sim.get("ev_under", 0),
            "trend":    trend_pct,
        })

    if not graded:
        await msg.edit(embed=discord.Embed(
            title="❌ No Gradeable Props",
            description="None of the props could be graded (vlr.gg lookups failed).",
            color=0xFF4136,
        )); return

    actionable = [g for g in graded if g["decision"] in ("OVER", "UNDER")]
    actionable.sort(key=lambda g: g["score"], reverse=True)
    top = actionable[:15]

    if not top:
        await msg.edit(embed=discord.Embed(
            title="⏸️ No Actionable Plays",
            description=f"All {len(graded)} props graded PASS. Sit this slate out.",
            color=0xFFDC00,
        )); return

    rows = []
    for i, g in enumerate(top, 1):
        if g["decision"] == "OVER":
            icon, dec_str, ev = "✅", f"OVER {g['line']:.1f}", g["ev_over"]
        else:
            icon, dec_str, ev = "❌", f"UNDER {g['line']:.1f}", g["ev_under"]
        ev_str = f"{'+' if (ev or 0) >= 0 else ''}{(ev or 0):.3f}u"
        trend = "🔥" if g["trend"] >= 8 else ("🧊" if g["trend"] <= -8 else "➖")
        rows.append(
            f"`#{i}` {icon} **{g['name']}** ({g['team']})\n"
            f"   → `{dec_str}`  Conf **{g['conf']}/100**  Edge **{g['edge_pct']:.1f}%**  "
            f"Hit `{g['hit_rate']:.0f}%`  Avg `{g['hist_avg']:.1f}`  EV `{ev_str}` {trend}"
        )

    embed = discord.Embed(
        title=f"🎯 Valorant PrizePicks Slate — Top {len(top)} Plays",
        description=(
            f"📊 Graded **{len(graded)}** live props · "
            f"🏆 **{len(actionable)} actionable** (OVER/UNDER) · "
            f"⏸️ {len(graded)-len(actionable)} PASS\n\n"
            + "\n\n".join(rows)
        ),
        color=0x9146FF,
    )
    embed.set_footer(text="Valorant · Live PrizePicks lines · Ranked by Conf × (1+Edge) · empirical grading")
    await msg.edit(embed=embed)


@bot.command(name="vscout", aliases=["vs"])
async def cmd_vscout(ctx, *, player_arg: str = ""):
    """Show a Valorant player's last 10 BO3 series without grading."""
    name = player_arg.strip()
    if not name:
        await ctx.send("Usage: `!vscout <player>`")
        return
    msg = await ctx.send(f"🔎 Scouting `{name}` on vlr.gg…")
    info = _vlr.get_player_info(name, n_series=10)
    if not info or not info.get("map_stats"):
        await msg.edit(content=f"❌ No Valorant data found for `{name}`.")
        return
    series: dict = {}
    for m in info["map_stats"]:
        series.setdefault(m["match_id"], []).append(m)
    totals = [sum(mm["stat_value"] for mm in maps) for maps in series.values()]
    lines = []
    for i, maps in enumerate(series.values(), 1):
        total = sum(mm["stat_value"] for mm in maps)
        per = " + ".join(f"{mm['stat_value']} ({mm['map_name']})" for mm in maps)
        lines.append(f"`{i:2d}.` **{total}**  ·  {per}")
    avg    = sum(totals) / len(totals)
    median = sorted(totals)[len(totals) // 2]
    embed = discord.Embed(
        title=f"🎯 {info['display_name']} — Valorant Scout",
        description=(
            f"**Last {len(totals)} BO3 series (Maps 1 + 2)**\n"
            f"Avg: **{avg:.1f}**  ·  Median: **{median}**  ·  "
            f"High: **{max(totals)}**  ·  Low: **{min(totals)}**"
        ),
        color=0x9146FF,
    )
    embed.add_field(name="Series", value="\n".join(lines), inline=False)
    embed.set_footer(text="Valorant · vlr.gg · Last 10 BO3 · Maps 1 & 2 only")
    await msg.edit(content=None, embed=embed)


@bot.command(name="vlines", aliases=["vl"])
async def cmd_vlines(ctx, *, player_arg: str = ""):
    """Suggest a few candidate kill lines for a Valorant player based on history."""
    name = player_arg.strip()
    if not name:
        await ctx.send("Usage: `!vlines <player>`")
        return
    msg = await ctx.send(f"🔎 Computing lines for `{name}`…")
    info = _vlr.get_player_info(name, n_series=10)
    if not info or not info.get("map_stats"):
        await msg.edit(content=f"❌ No Valorant data found for `{name}`.")
        return
    series: dict = {}
    for m in info["map_stats"]:
        series.setdefault(m["match_id"], []).append(m)
    totals = sorted(sum(mm["stat_value"] for mm in s) for s in series.values())
    n = len(totals)
    if n == 0:
        await msg.edit(content=f"❌ No data.")
        return
    p25 = totals[n // 4]
    p50 = totals[n // 2]
    p75 = totals[(3 * n) // 4]
    embed = discord.Embed(
        title=f"📐 {info['display_name']} — Suggested Lines (Valorant)",
        description=(
            f"Based on last {n} BO3 series (Maps 1 + 2):\n\n"
            f"**Conservative line (p25):** `{p25 - 0.5:.1f}`\n"
            f"**Median line (p50):**       `{p50 - 0.5:.1f}`\n"
            f"**Aggressive line (p75):**   `{p75 - 0.5:.1f}`\n"
        ),
        color=0x9146FF,
    )
    embed.set_footer(text="Valorant · vlr.gg · Suggested PrizePicks-style lines")
    await msg.edit(content=None, embed=embed)


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
