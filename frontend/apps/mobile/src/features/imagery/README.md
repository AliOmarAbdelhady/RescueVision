# feature: imagery (mobile)

Pre/post imagery capture for assessment runs. Scope: camera capture flow
(`services/CameraService`), pairing pre+post shots, compression, resumable upload
to `/predict` job endpoint with progress events. Designed for degraded networks
(pause/resume, background upload).
