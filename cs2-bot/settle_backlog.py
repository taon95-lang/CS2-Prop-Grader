"""
Backfill-settle every unsettled grade in grades_history.json by fetching the
actual Maps 1+2 kill total from HLTV via the existing scraper pipeline.

Usage:
    python3 cs2-bot/settle_backlog.py            # settle everything
    python3 cs2-bot/settle_backlog.py --kills    # only kill props
    python3 cs2-bot/settle_backlog.py --dry-run  # don't write, just report
"""
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("settle_backlog")

import grades_db
from scraper import get_actual_result

DRY_RUN = "--dry-run" in sys.argv
KILLS_ONLY = "--kills" in sys.argv

def _arg_int(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return int(a.split("=", 1)[1])
    return default

LIMIT = _arg_int("limit", 999)
OFFSET = _arg_int("offset", 0)

db = grades_db._load()
entries = db["entries"]

unsettled = [e for e in entries if e.get("actual") is None]
if KILLS_ONLY:
    unsettled = [e for e in unsettled if e.get("stat") == "Kills"]

unsettled = sorted(unsettled, key=lambda x: x["ts"])
total = len(unsettled)
unsettled = unsettled[OFFSET:OFFSET + LIMIT]

logger.info(f"Found {total} unsettled grades; processing {len(unsettled)} (offset={OFFSET}, limit={LIMIT})")
logger.info(f"DRY_RUN={DRY_RUN}, KILLS_ONLY={KILLS_ONLY}")

stats = {
    "settled": 0,
    "still_pending": 0,
    "errors": 0,
    "skipped_hs": 0,
    "wins": 0,
    "losses": 0,
    "passes_w_outcome": 0,
}

for i, entry in enumerate(unsettled, 1):
    player = entry["player"]
    line = entry["line"]
    stat = entry["stat"]
    opponent = entry.get("opponent") or ""
    grade_ts = entry["ts"]
    rec = (entry.get("recommendation") or "").upper()
    grade = entry.get("grade") or "N/A"

    logger.info(
        f"\n[{i}/{len(unsettled)}] {entry['display']} {line} {stat} "
        f"({entry['date']}, rec={rec})"
    )

    # HS props need a different parser path; skip for now and report.
    if stat != "Kills":
        logger.info(f"  ⏭️  HS prop — needs separate parser, skipping")
        stats["skipped_hs"] += 1
        continue

    try:
        result = get_actual_result(
            player_name=player,
            opponent=opponent,
            grade_ts=grade_ts,
            line=line,
            baseline_match_id=entry.get("baseline_match_id"),
        )
    except Exception as e:
        logger.warning(f"  ❌ Scraper error: {type(e).__name__}: {e}")
        stats["errors"] += 1
        continue

    if not result:
        logger.info(f"  ⏳ No new completed BO3 found yet")
        stats["still_pending"] += 1
        continue

    actual = result["actual"]
    outcome = result["outcome"]

    if not DRY_RUN:
        # Write straight into the entry to avoid record_result's "most recent"
        # behaviour overriding the wrong duplicate.
        entry["actual"] = actual
        entry["outcome"] = outcome
        entry["baseline_match_id"] = result.get("match_id")
        # Save after every settle so timeouts don't lose progress
        grades_db._save(db)

    stats["settled"] += 1

    # Tally win/loss for directional calls
    if rec == "OVER":
        if outcome == "over":
            stats["wins"] += 1
            verdict = "✅ WIN"
        else:
            stats["losses"] += 1
            verdict = "❌ LOSS"
    elif rec == "UNDER":
        if outcome == "under":
            stats["wins"] += 1
            verdict = "✅ WIN"
        else:
            stats["losses"] += 1
            verdict = "❌ LOSS"
    else:
        stats["passes_w_outcome"] += 1
        verdict = f"⏸️  PASS (would have hit {outcome.upper()})"

    logger.info(f"  → actual={actual} ({outcome}) | bot said {rec} {grade}  {verdict}")

    # Throttle so we don't hammer HLTV (each scrape already takes 1-2s)
    time.sleep(0.5)

if not DRY_RUN:
    grades_db._save(db)

logger.info("\n" + "=" * 70)
logger.info("BACKFILL SUMMARY")
logger.info("=" * 70)
for k, v in stats.items():
    logger.info(f"  {k:20s}: {v}")
total_calls = stats["wins"] + stats["losses"]
if total_calls:
    logger.info(
        f"\n  Hit rate (W/(W+L)): {stats['wins']}/{total_calls} "
        f"= {stats['wins']/total_calls*100:.1f}%"
    )
