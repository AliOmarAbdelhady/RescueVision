# packages/config — shared tooling presets

No runtime code. One formatting/linting/typecheck truth for every workspace member.

```
tsconfig/    # base.json — every app/package extends this (strict, bundler
             #   moduleResolution, path mapping to @rescuevision/*)
eslint/      # shared eslint flat config (import boundaries enforcement lives here:
             #   packages must never import from apps)
tailwind/    # shared preset — added when UI work starts (deliberately deferred)
```

Enforcing the dependency law with lint rules (not just the README) is what keeps
the architecture real as the team grows.
