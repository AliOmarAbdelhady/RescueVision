# packages/api — THE GATEWAY ★

The only code in the frontend that speaks HTTP or WebSocket. Both apps (`web`,
`mobile`) and any future client consume this package — never the network directly.

```
src/
  generated/            # openapi-typescript output — GENERATED, COMMITTED
                        #   (regenerate with: pnpm generate:api)
  client.ts             # openapi-fetch instance: baseURL, auth token injection,
                        #   refresh-once-on-401, normalized error shape
  endpoints/            # typed SDK per domain
    predict.ts          #   run assessment (multipart, job-based)
    cases.ts            #   list/detail/artifacts
    reports.ts          #   view + download PDF
    routing.ts          #   emergency route planning
    tracking.ts         #   responder positions, SOS
    auth.ts             #   login/refresh/me
  realtime/             # WebSocket client: auto-reconnect, typed event bus
                        #   events: position · sos · job-progress · victim
  query/                # TanStack Query hooks + query-key factories
                        #   (useCase, useRunPrediction, useLivePositions...)
  index.ts              # the ONLY public entry — keep the surface explicit
```

## Rules

- Types are **generated from FastAPI's `/openapi.json`**, never hand-written.
- **No platform code here**: no `localStorage`, no `expo-*`. Apps inject
  `{ baseUrl, wsUrl, tokenStore }` at bootstrap.
- React appears only in `query/` (hooks shared by both apps).
- Realtime payloads are validated with zod schemas from `@rescuevision/contracts`
  at the socket boundary.

## After any backend schema change

```bash
# FastAPI running locally
pnpm generate:api        # snapshot spec → regenerate types
pnpm typecheck           # compiler shows every break in BOTH apps
```
