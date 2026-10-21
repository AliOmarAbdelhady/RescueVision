# apps/web — RescueVision Ops Dashboard (React + Vite SPA)

The command center: **dashboard & statistics, buildings explorer, cases, reports,
analytics, emergency routing, live map of responders**. Authenticated internal tool —
no SSR/SEO, so a static SPA served beside the FastAPI backend.

## Law of this app

> `routes/` + `pages/` are thin (URL → feature composition). `layouts/` holds the
> AppShell chrome. All logic lives in `src/features/*`.

```
src/
  routes/               # React Router route tree + guards (role-based)
  layouts/              # AppShell (sidebar/topbar), AuthLayout
  pages/                # one folder per route — composition only
    dashboard/          #   overview stats & urgency
    live-ops/           #   live map, responder positions, SOS feed
    cases/              #   case list + detail
    assessments/        #   new assessment wizard (pre/post upload)
    buildings/          #   per-building damage explorer
    reports/            #   report viewer + PDF download
    routing/            #   rescue route planner (avoid blocked roads)
    analytics/          #   trends, loss estimation views
    settings/           #   admin, users, assumptions
  features/             # vertical slices — see each folder's README
    assessment/  cases/  buildings/  analytics/  live-ops/  routing/  reports/
  components/           # cross-feature UI (table, stats card, map panel)
  stores/               # Zustand client state (ui prefs, map layers)
  hooks/
  lib/                  # bootstrap (injects env + token store → packages/api)
```

## Rules

- No `fetch`/`axios` anywhere — data flows through `@rescuevision/api` hooks.
- Server state only in TanStack Query; Zustand only for UI/session state.
- Maps render with MapLibre GL JS (same style spec as the mobile app).
- Dev proxy in `vite.config.ts` forwards `/api` + `/ws` to FastAPI `:8000`.
