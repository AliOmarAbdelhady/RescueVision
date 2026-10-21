# tooling — workspace scripts & infra

```
scripts/
  generate-api.ts   # 1. fetch FastAPI /openapi.json → snapshot to openapi.json
                    # 2. openapi-typescript → packages/api/src/generated/
                    # 3. both committed for reviewable diffs
docker/
  nginx.conf        # static web build + /api + /ws proxy to FastAPI
  Dockerfile.web    # build stage (pnpm --filter web build) + nginx serve stage
```

`generate-api` needs the backend running locally (`uvicorn api.main:app`) — CI
typechecks against the committed snapshot, so CI never needs a live backend.
