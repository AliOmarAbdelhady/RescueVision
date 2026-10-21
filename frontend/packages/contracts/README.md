# packages/contracts — shared domain language

Pure TypeScript: types, enums, constants, zod schemas. **No I/O, no platform APIs,
no imports from apps or other packages** (except nothing — this is the bottom of the
stack alongside `utils`).

Every surface reads the same definitions from here — web charts, mobile map legends,
urgency badges, role guards — so the product speaks one vocabulary everywhere.

```
src/
  damage.ts      # DAMAGE_CLASSES: id, label, color (green→yellow→orange→red scale)
  urgency.ts     # urgency levels + thresholds
  geo.ts         # GeoJSON feature types, coordinate helpers, distance units
  user.ts        # roles (COMMANDER / ANALYST / RESPONDER) + permissions matrix
  events.ts      # zod schemas for every WebSocket event (position, sos,
                 #   job-progress, victim) — validated at the socket boundary
```

## Rules

- If two surfaces would otherwise redefine the same domain concept, the concept
  belongs here.
- Adding an event type = add zod schema + regenerate nothing else; both apps get
  compile-time knowledge of it through `@rescuevision/api/realtime`.
