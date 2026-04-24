# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/              # Deployable applications
│   └── api-server/         # Express API server
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts, run via `pnpm --filter @workspace/scripts run <script>`
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references. This means:

- **Always typecheck from the root** — run `pnpm run typecheck` (which runs `tsc --build --emitDeclarationOnly`). This builds the full dependency graph so that cross-package imports resolve correctly. Running `tsc` inside a single package will fail if its dependencies haven't been built yet.
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck; actual JS bundling is handled by esbuild/tsx/vite...etc, not `tsc`.
- **Project references** — when package A depends on package B, A's `tsconfig.json` must list B in its `references` array. `tsc --build` uses this to determine build order and skip up-to-date packages.

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Discord Bot (cs2-bot/)

Python-based Discord bot for grading CS2 player props from HLTV data exclusively.

### Core Files
- **Entry:** `cs2-bot/bot.py` — main bot, all command handlers, embed design
- **Scraper:** `cs2-bot/scraper.py` — CloudScraper + BeautifulSoup HLTV scraper (chrome116 profile)
- **Simulator:** `cs2-bot/simulator.py` — Negative Binomial + 10k Monte Carlo engine with recency weighting
- **Deep Analysis:** `cs2-bot/deep_analysis.py` — Opponent scouting (defense, H2H, economy, map pool)
- **Grade Engine:** `cs2-bot/grade_engine.py` — Confidence scoring (0-100), edge calc, form streaks, variance tiers, map intel, risk flags, verdict reasoning, multi-line tables
- **Keep-alive:** `cs2-bot/keep_alive.py` — Flask server on port 5000 for uptime

### Commands
- `!grade [Player] [Line] [Kills/HS] [Opponent?]` — Full prop grade with confidence, edge, form, simulation
- `!scout [Player]` — Player scouting card (role, map pool, HLTV 90-day stats, variance)
- `!lines [Player] [Kills/HS]` — Multi-line probability table for line shopping (base ± 3 lines)

### Embed Design (Redesigned)
The `!grade` embed has a compact professional layout:
1. **Verdict banner** — Decision (OVER/UNDER/PASS) + Confidence/100 + Edge% prominently at top
2. **Historical** — Avg, Median, Hit Rate, Variance tier, Form streak, Last 4
3. **Simulation** — Over/Under%, probability bar, Fair line
4. **HLTV 90d** — KPR, Rating, KAST, ADR, K/D, HS%
5. **Map Intel** — Expected maps, projected avg vs line, best/worst map
6. **vs Opponent** — Combined %, defense label, H2H, component breakdown
7. **Risk Flags** — Only shown when active
8. **Series Breakdown** — Per-series hit/miss with map labels
9. **Verdict** — Play directive, one-line reason, unit size

### Grade Engine (grade_engine.py)
- `compute_confidence_score()` — 0-100 weighted from 10+ signals
- `compute_edge_pct()` — Edge vs -110 vig (52.38% implied)
- `compute_form_streak()` — Consecutive hit/miss streak from series data
- `compute_variance_tier()` — CV-based LOW/MEDIUM/HIGH/VERY_HIGH tiers
- `compute_map_intel()` — Per-map kill averages with projected overlay
- `compute_risk_flags()` — Active risk flags (stomp, cold streak, thin sample, etc.)
- `build_verdict_reason()` — One-line reasoning for the call
- `run_lines_table()` — Multi-line probability table for ±3 lines

### Recommendation taxonomy (April 2026)
- `OVER` / `UNDER` — directional bet, counts in W/L tally and calibration
- `PASS` — soft skip (insufficient signal)
- `NO_BET` — hard skip from AUTO NO BET cap or asymmetric score gate; non-directional
- Reporting helpers in `bot.py`: `_is_skip_rec()` and `_is_directional_rec()` — used by `!result`, `!results`, `!calibration`, `!fetchresults` so NO_BET / PASS are excluded from win/loss math instead of being mis-counted as OVER losses
- `settle_backlog.py` already buckets non-directional recs into `passes_w_outcome`

### Workflow
- **Name:** `Elite CS2 Prop Grader Bot`
- **Command:** `cd cs2-bot && python3 bot.py`
- **Token:** Set as `DISCORD_TOKEN` in Replit Secrets

## Packages

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server. Routes live in `src/routes/` and use `@workspace/api-zod` for request and response validation and `@workspace/db` for persistence.

- Entry: `src/index.ts` — reads `PORT`, starts Express
- App setup: `src/app.ts` — mounts CORS, JSON/urlencoded parsing, routes at `/api`
- Routes: `src/routes/index.ts` mounts sub-routers; `src/routes/health.ts` exposes `GET /health` (full path: `/api/health`)
- Depends on: `@workspace/db`, `@workspace/api-zod`
- `pnpm --filter @workspace/api-server run dev` — run the dev server
- `pnpm --filter @workspace/api-server run build` — production esbuild bundle (`dist/index.cjs`)
- Build bundles an allowlist of deps (express, cors, pg, drizzle-orm, zod, etc.) and externalizes the rest

### `lib/db` (`@workspace/db`)

Database layer using Drizzle ORM with PostgreSQL. Exports a Drizzle client instance and schema models.

- `src/index.ts` — creates a `Pool` + Drizzle instance, exports schema
- `src/schema/index.ts` — barrel re-export of all models
- `src/schema/<modelname>.ts` — table definitions with `drizzle-zod` insert schemas (no models definitions exist right now)
- `drizzle.config.ts` — Drizzle Kit config (requires `DATABASE_URL`, automatically provided by Replit)
- Exports: `.` (pool, db, schema), `./schema` (schema only)

Production migrations are handled by Replit when publishing. In development, we just use `pnpm --filter @workspace/db run push`, and we fallback to `pnpm --filter @workspace/db run push-force`.

### `lib/api-spec` (`@workspace/api-spec`)

Owns the OpenAPI 3.1 spec (`openapi.yaml`) and the Orval config (`orval.config.ts`). Running codegen produces output into two sibling packages:

1. `lib/api-client-react/src/generated/` — React Query hooks + fetch client
2. `lib/api-zod/src/generated/` — Zod schemas

Run codegen: `pnpm --filter @workspace/api-spec run codegen`

### `lib/api-zod` (`@workspace/api-zod`)

Generated Zod schemas from the OpenAPI spec (e.g. `HealthCheckResponse`). Used by `api-server` for response validation.

### `lib/api-client-react` (`@workspace/api-client-react`)

Generated React Query hooks and fetch client from the OpenAPI spec (e.g. `useHealthCheck`, `healthCheck`).

### `scripts` (`@workspace/scripts`)

Utility scripts package. Each script is a `.ts` file in `src/` with a corresponding npm script in `package.json`. Run scripts via `pnpm --filter @workspace/scripts run <script>`. Scripts can import any workspace package (e.g., `@workspace/db`) by adding it as a dependency in `scripts/package.json`.
