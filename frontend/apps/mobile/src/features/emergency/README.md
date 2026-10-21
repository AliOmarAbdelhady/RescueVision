# feature: emergency (mobile) — SOS

The one-tap lifeline. Scope:

- SOS trigger flow (hold-to-confirm), note + photo attachment
- emits `sos` WebSocket event via `@rescuevision/api/realtime`
- escalation state machine (triggered → acknowledged by command → resolved)
- "share my location" request handling when command asks for a fix

Must remain usable with one hand, poor connectivity (queue + retry), and no
attention. Everything here is designed worst-case-first.
