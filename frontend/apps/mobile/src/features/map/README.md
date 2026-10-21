# feature: map (mobile)

MapLibre React Native view + offline strategy. Scope:

- base map + satellite raster tiles (same style spec as web)
- damage/urgency layer rendering from case GeoJSON artifacts
- offline pack download & management for assigned areas
- responder position + route line overlays

View-model layer only — positioning logic lives in `features/tracking`.
