# apps/mobile — RescueVision Field App (Expo / React Native)

The responder's tool: **emergency/SOS, live location tracking, missions, imagery
capture, offline maps**. Built with Expo + Expo Router.

## Law of this app

> `app/` contains ROUTES ONLY. Every screen stays thin (compose feature components +
> read route params). All logic lives in `src/features/*`.

```
app/                    # navigation tree — file routes
  (auth)/login          # gate: no session token → redirect here
  (tabs)/               # home(missions) · map · sos · profile
  case/[id]/            # assigned case detail
  tracking/[id]/        # active tracking session screen
src/
  features/             # vertical slices — see each folder's README
    auth/  emergency/  tracking/  missions/  map/  imagery/  notifications/
  services/             # native modules behind interfaces:
                        #   LocationService, PermissionsService, StorageService,
                        #   CameraService, BackgroundTaskService
  components/           # cross-feature RN components
  stores/               # Zustand client state (session, settings, tracking flags)
  hooks/
  lib/                  # providers composition, app bootstrap (injects env → packages/api)
app.config.ts           # env-driven Expo config — never hardcode URLs in app.json
eas.json                # EAS build profiles (development/preview/production)
metro.config.js         # monorepo watchFolders + nodeModulesPaths
```

## Rules

- No `fetch`/`axios` anywhere — data flows through `@rescuevision/api` hooks.
- Native modules are called only inside `src/services/` — features import interfaces.
- Env via `EXPO_PUBLIC_*` vars, injected once in `src/lib/` bootstrap.
- Deep links (`case/{id}`, SOS alerts) resolve through typed Expo Router routes.
