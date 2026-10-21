# feature: live-ops (web)

The real-time command view. Scope:

- live MapLibre map with responder positions (subscribes `position` events,
  updates the map source directly — never re-renders React per fix)
- SOS alert feed (subscribe `sos`, acknowledge/resolve actions, locate person)
- victim-detection geolocation layer (planned)
- responder list + status (online/offline/last-seen)

This is the consumer side of `apps/mobile/src/features/tracking` and
`features/emergency` — the mobile→web bridge made visible.
