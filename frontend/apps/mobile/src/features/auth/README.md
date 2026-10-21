# feature: auth (mobile)

Login flow, session lifecycle, token persistence (via `expo-secure-store` through
`services/StorageService`), role resolution on boot.

Scope: login screen logic, session store wiring, auth-state gate used by `(auth)`
route group. JWT refresh itself lives in `@rescuevision/api` — this feature only
supplies the platform token store.
