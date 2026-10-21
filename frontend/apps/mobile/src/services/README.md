# services (mobile)

Every native module wrapped behind an interface, imported by features as
`LocationService.start()` etc. — never `expo-location` directly.

- `LocationService` — foreground/background location, accuracy settings
- `PermissionsService` — unified permission prompts & status
- `StorageService` — secure token + settings persistence
- `CameraService` — capture, compression
- `BackgroundTaskService` — background fetch / task registration

This isolation keeps features unit-testable with mocked services and confines any
native-module swap to exactly one file.
