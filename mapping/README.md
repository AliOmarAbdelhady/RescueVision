# Mapping (satellite city map) — Kahramanmaras, Türkiye earthquake 2023

Imported from the standalone `GradProject` prototype. Builds browsable 2D maps
from real Maxar Open Data imagery of Kahramanmaras city center (~0.8 × 0.8 km,
0.5 m/px), pre-event (26 Jul 2022) and post-event (28 Feb 2023).

## Scripts

| Script | Output (in `output/`) | What it is |
|---|---|---|
| `build_satellite_map.py` | `satellite_city_map.html` | Single Folium map, BEFORE/AFTER toggle, legend for damage classes + entry/exit routes (placeholders) |
| `build_compare_app.py` | `compare.html`, `dataset.html` | Synced side-by-side before/after with a "+ GENERATED" change overlay; browser of every Maxar scene covering the area |
| `build_process_view.py` | `process.html` | Animated "collect tiles → assemble mosaic → final map" view |
| `start_map_server.sh` | — | Serves `output/` on port 8000 (open from a phone on the same Wi-Fi) |

## Setup

```bash
pip install -r mapping/requirements.txt   # plus GDAL CLI (brew install gdal) for thumbnails
python mapping/build_satellite_map.py
python mapping/build_compare_app.py
python mapping/build_process_view.py
bash mapping/start_map_server.sh          # http://localhost:8000/satellite_city_map.html
```

## Data

- `data/turkey-earthquake.parquet` — Maxar event tile catalog (committed, small).
- `data/city_{before,after}[_utm].tif` and `output/city_{before,after}.png` — large
  rasters, **git-ignored**. Keep them locally; regenerate from the Maxar bucket if lost.
- `output/damage_overlay.png` — current "GENERATED" layer: a pixel-change baseline,
  to be replaced by RescueVision's trained damage/change models.

## Next steps (routing)

- Replace `damage_overlay.png` with output from `src/rescuevision` inference.
- Extract roads (OSM / FloodNet SegFormer), mark segments blocked by damage,
  compute entry/exit routes (networkx), and feed them to `dashboard/src/components/RouteMap.tsx`.
