# feature: tracking (mobile) — live location

Background location broadcast to the ops dashboard. Scope:

- `expo-location` background task (via `services/LocationService`)
- batching + throttling positions (battery vs freshness tradeoff)
- publishes `position` events over `@rescuevision/api/realtime`
- tracking session lifecycle (start/pause/stop, visible indicator)
- accuracy filtering (drop bad GPS fixes before they hit the wire)

Pairs with `apps/web/src/features/live-ops` which consumes the same events.
