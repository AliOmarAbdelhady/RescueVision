# packages/utils — pure helpers

Zero-dependency, fully unit-testable functions only. No React, no React Native,
no network, no storage.

```
src/
  format.ts      # numbers, percents, file sizes, damage labels
  geo.ts         # haversine distance, bbox math, tile helpers, coordinate formatting
  time.ts        # timestamps, relative time, duration formatting
```

If a helper needs an import from `apps/*` or does I/O, it does not belong here.
