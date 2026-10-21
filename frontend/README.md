# RescueVision Frontend Platform

Monorepo for everything the user touches: the **Expo field app** (emergency, SOS,
tracking, missions), the **React web dashboard** (statistics, buildings, cases,
reports, live map), and the **shared typed API gateway** both consume.

```
frontend/
├── apps/
│   ├── mobile/     # Expo (React Native) — responder field app
│   └── web/        # React + Vite SPA — ops dashboard
├── packages/
│   ├── api/        # ★ THE GATEWAY: generated typed SDK + WS client + query hooks
│   ├── contracts/  # shared domain language (damage classes, urgency, roles, events)
│   ├── config/     # shared tsconfig / eslint / tailwind presets
│   └── utils/      # pure helpers (geo math, formatting)
└── tooling/        # generate-api script, docker configs
```

**Read [`docs/FRONTEND_ARCHITECTURE.md`](../docs/FRONTEND_ARCHITECTURE.md) before
writing code** — it defines the dependency law, folder responsibilities, data flows,
and the decisions behind them.

## Commands (run from here, always)

```bash
pnpm install          # once
pnpm generate:api     # after any backend schema change (needs FastAPI running)
pnpm dev:web          # web dashboard only
pnpm dev:mobile       # Expo dev server
pnpm build            # all packages + apps (topological order, cached)
pnpm typecheck        # both apps + all packages
pnpm lint
```

## The one rule

Apps import packages. Packages never import apps. All HTTP/WebSocket traffic goes
through `packages/api` — no `fetch`/`axios` in feature code, ever.
