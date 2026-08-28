# RescueVision → Full Disaster-Response Graduation Project

## Context

RescueVision today is a **clean but untrained scaffold**: PyTorch + FastAPI + React + Docker, ~8.5k lines, six real model implementations (U-Net, Siamese U-Net, ChangeFormer, DINOv2 & EfficientNet classifiers, SegFormer). Critical facts about the starting point:

- **Zero trained weights, zero data, zero benchmarks.** All checkpoint/FAISS/data dirs are empty; the two "demo" outputs were produced with no models loaded.
- **Before/after is hard-wired everywhere** — CLI, FastAPI `/predict`, React uploader, and `DisasterPredictor.predict(pre, post)` all *require* a pre-disaster image. There is **no code path** for single-image (no-before) assessment.
- **"Multimodal" is marketing** — only xBD satellite RGB + FloodNet drone RGB. No SAR/thermal/DEM/GPS. Text is output-only (a VLM report that silently falls back to a template).
- **Geospatial is scaffolding only** — [routing/geo_utils.py](src/rescuevision/routing/geo_utils.py) has `pixel_to_geo`/`haversine_distance`, but nothing reads EXIF, no CRS/pyproj, no GeoTIFF, no map library. The route planner runs in **pixel space**.
- **Zero 3D or simulation.** Git is broken (empty `.git/`) — no commit history.

**Goal:** Expand this into a unique, defensible graduation project covering 8 capabilities — (1) single-image damage assessment, (2) AI reconstruction-cost estimation, (3) victim detection (SARD-style), (4) UAV pixel→GPS localization, (5) 2D+3D area reconstruction, (6) people-in-danger counting, (7) AI rescue/evacuation routing, (8) Gazebo/ROS simulation — with **honest feasibility and limitations** baked in.

**Constraints:** ~8–10 months, team of 4+, a real DJI drone + GPU compute (no thermal/RTK budget committed → those become stretch). Priority: a coherent system + one novel spark + honest real-world framing.

This plan was built from a multi-agent research sweep (9 feature researchers + 8 adversarial verifiers, web-sourced). Every accuracy number below is literature-cited and verification-corrected.

---

## The honest verdict (read first)

**Yes, this is achievable — but NOT as "8 production systems."** Each feature is a thesis on its own; end-to-end disaster-response UAV AI is a multi-year, multi-million-euro effort (EU INACHUS; DARPA SubT, won by a large consortium over 3 years). What *is* realistic and impressive for a graduation project is a **scoped, georeferenced core of 4–5 modules** that actually runs end-to-end, plus 2–3 stretch modules demonstrated at concept level. The existing scaffold already has the spine; the new work bolts on as georeferenced I/O.

**The single binding constraint is not model quality — it is georeferencing consistency** between two incompatible imagery regimes: xBD satellite tiles (~0.3–3 m GSD) vs. drone nadir (1–2 cm GSD). Everything good or bad about this project flows from how cleanly you make one scene graph hold both.

---

## Feasibility matrix

| # | Feature | Verdict | Headline limitation | Scoped recommendation | Effort |
|---|---------|---------|---------------------|------------------------|--------|
| 1 | Single-image damage (no before) | Feasible w/ significant effort | Cannot reliably get 4-level severity, nor distinguish "destroyed" vs "under-construction/empty lot" w/o a before image | **Binary standing-vs-collapsed triage** (F1 ~0.84–0.91) + advisory multi-class; footprint-conditioned DINOv2/DINOv3 | M |
| 2 | AI reconstruction cost | Feasible w/ significant effort | Imagery can't see interior/contents/foundation; only external severity | Rebrand as **"indicative cost INDEX"** = damage-class × footprint area × regional unit-cost; validate vs ONE published GRADE total | M |
| 3 | Victim detection (SARD) | **Only partially feasible** | **RGB cannot see through rubble.** SARD finds surface/partially-exposed people only | YOLOv8 fine-tuned on SARD as a **visible-casualty detector**; thermal as an *anomaly flagger* (no public RGB-T SAR dataset to train a fusion head) | M |
| 4 | UAV pixel→GPS | Feasible w/ significant effort | Sub-meter needs RTK/GCP; consumer DJI GPS ≈ 2–5 m, +1° attitude error ≈ +1.75 m at 100 m alt | Nadir pinhole projector from EXIF+XMP; **always emit an uncertainty radius**; report meter-level error validated against sim ground truth | M |
| 5 | 2D + 3D reconstruction | Feasible w/ significant effort | Aerial-only has no facade detail; rubble reconstructs as blobs; cm-accuracy needs RTK/GCP | **WebODM orthomosaic + DSM + mesh as core**; ONE 3D Gaussian Splatting scene as the "wow" demo | M–L |
| 6 | People-in-danger counting | Feasible w/ significant effort | Can only count **visible** people; buried/indoor are invisible → it's a **lower bound** | YOLOv8 + ByteTrack + danger-zone geofence on drone video; refuse if GSD > ~15 cm/px | M |
| 7 | AI rescue/evacuation routing | Feasible w/ significant effort | 2D overhead can't see street-level killers (power lines, gas, fire); it's a planning *aid* not ground truth | Raster hazard cost-map (damage + DEM slope + flood + building-proximity) → A*/RRT* + OSMnx graph; overlay on orthomosaic | M |
| 8 | Gazebo/ROS simulation | Feasible w/ significant effort | Steep 3–6 week learning curve; sim-to-real gap measured at **16–32 mAP points lost** for synthetic-only training | ROS 2 Jazzy + **Gazebo Harmonic** + PX4 SITL; use as (a) end-to-end demo harness and (b) synthetic ground-truth oracle — never sole training set | L |

Effort assumes one team owner per module.

---

## The 5 killer limitations (and how to handle them honestly)

1. **Optical cameras cannot see through rubble.** SARD/HERIDAL detect *surface/partially-exposed* people. Buried/indoor victims are invisible. → Frame feature 3 as "visible-casualty detector" + optional thermal *flagger*; never claim x-ray. Put this in the README and every examiner-facing slide.
2. **Consumer GPS is meter-level, not centimeter.** A victim localized from a single DJI photo is ±2–5 m (often worse over rubble/slopes due to flat-earth assumption). → Always emit a localization *uncertainty radius*; sub-meter is a stretch goal requiring an RTK drone or ground control points.
3. **Two imagery regimes must be reconciled.** xBD satellite (~0.3–3 m GSD, corner-coord JSON sidecar, no embedded geotransform) vs. drone nadir (1–2 cm). xBD tiles and drone mosaics of *different scenes* don't align. → Make **GeoTIFF + per-frame GeoJSON in EPSG:4326** the canonical scene representation; build a shared `geo/` module; for demos, fly the drone over a *single* self-captured AOI so damage map + drone video + 3D are the same place.
4. **Data acquisition is the gating risk.** There is **no public dataset** of raw, overlapping, geotagged, photogrammetry-grade post-disaster drone imagery. OpenAerialMap distributes *processed orthomosaics*, not raw photos; LADI v2 is annotated tiles. → You have a DJI — **fly your own AOI** (e.g., a quarry, demolition site, or constructed rubble pile with permission) for the 2D/3D + GPS + victim demo. Use public data only for ML *training*.
5. **Sim-to-real gap is real and measured.** DLR (2023): a detector trained *only* on synthetic imagery lost 16.8–32.1 mAP points and collapsed near zero on the hardest split. → Treat Gazebo imagery as **pre-training/augmentation + a validation oracle**, never the sole training set; always fine-tune on real (SARD/HERIDAL/VisDrone/your own flights).

---

## Recommended architecture: georeferenced single-source-of-truth

The engineering novelty that makes this project coherent and defensible:

```
                ┌─────────────────────────────────────────────────────┐
                │        CANONICAL SCENE  (one AOI)                    │
                │  GeoTIFF orthomosaic + DSM (EPSG:4326/3857)          │
                │  + scene.geojson: footprints, damage, victims,       │
                │    routes, hazard zones, counts, cost — all in lat/lon│
                └─────────────▲───────────────────────────▲────────────┘
                              │ produces                   │ consumes
   ┌────────────┐   ┌──────────┴─────────┐   ┌─────────────┴──────────┐
   │ Drone flight│ → │ reconstruction/    │   │ routing/ (geo-space)    │
   │ (DJI) or    │   │  WebODM client     │   │  hazard cost-map +      │
   │ Gazebo sim  │   │  ortho+DSM+mesh    │   │  A*/RRT*/OSMnx          │
   └────────────┘   └────────────────────┘   └─────────────────────────┘
        │  per-frame                              ▲
        ▼                                         │
   ┌──────────────┐   ┌─────────────────┐   ┌─────┴────────┐
   │ geo/         │   │ post_only/      │   │ cost/        │
   │ EXIF/XMP +   │   │  footprint-     │   │ damage×area× │
   │ collinearity │   │  cond. DINOv2   │   │ unit-cost    │
   │ → lat/lon+σ  │   │  triage         │   │ INDEX        │
   └──────▲───────┘   └────────▲────────┘   └──────▲───────┘
          │                    │                   │
   ┌──────┴──────┐   ┌─────────┴──────┐    ┌───────┴──────┐
   │ victims/    │   │ (existing      │    │ counting/    │
   │ YOLOv8-SARD │   │  paired xBD    │    │ YOLO+ByteTrack│
   │ + thermal   │   │  damage branch)│    │ + geofence   │
   │ flagger     │   │  stays as-is   │    │              │
   └─────────────┘   └────────────────┘    └──────────────┘
```

**New top-level modules** (all under `src/rescuevision/`, following existing package conventions):
- `geo/` — EXIF/XMP/telemetry reader + collinearity projector + co-registration helpers. **Extends** [routing/geo_utils.py](src/rescuevision/routing/geo_utils.py) (which already has `pixel_to_geo`, `compute_pixel_scale`, `haversine_distance`).
- `post_only/` — single-image damage triage (sits *alongside* the existing paired branch; invoked only when no pre-image is given).
- `reconstruction/` — thin client over NodeODM REST API + an optional nerfstudio/gsplat wrapper.
- `victims/` — Ultralytics YOLOv8 wrapper fine-tuned on SARD + thermal-anomaly flagger.
- `counting/` — detector + ByteTrack + danger-zone geofence.
- `cost/` — deterministic post-processing module (damage class × footprint area × unit-cost table).
- `routing/` — **extend** the existing pixel-space [route_planner.py](src/rescuevision/routing/route_planner.py) to geo-space; wire it into the predictor (it is currently standalone/orphaned).
- `sim/` (top-level, outside the Python package) — Gazebo Harmonic worlds + PX4 SITL + ROS 2 bridge + synthetic-capture scripts.

---

## Recommended tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| ML framework | **PyTorch** (already) + `ultralytics` (YOLOv8/11) + existing `segmentation_models_pytorch`, `transformers`, `timm` | Consistent with current repo; YOLOv8 has built-in ByteTrack/BoT-SORT |
| Victim detector | **YOLOv8 fine-tuned on SARD** (single class "person"); cross-validate on HERIDAL + DLR set | SARD is the canonical SAR dataset; ~0.9 mAP best-case on curated test |
| No-before damage | **DINOv2/DINOv3 ViT footprint-conditioned** binary triage | Follows Damage-TriageFormer; binary F1 ~0.84–0.91 |
| Geospatial | `rasterio`, `geopandas`, `shapely` (already pinned) **+ add `pyproj`, `osmnx`, `piexif`/`exifread`, `ExifTool`** | pyproj for CRS; OSMnx for road graphs + footprints; ExifTool reads DJI XMP gimbal/altitude |
| Reconstruction | **WebODM/NodeODM** (core) + nerfstudio `gsplat`/Splatfacto (one demo) | Free, production-grade; Apache-2.0 splatting keeps you license-clean |
| Routing | `scikit-image` (`route_through_array`) or `pyastar2d` + `networkx`+`osmnx` | Mature raster + graph planners |
| Maps UI | **kepler.gl** or **folium** (backend) + **MapLibre/Leaflet** (React) | GeoJSON-native, free, no API key |
| Simulation | **ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL** | Gazebo Classic is EOL (Jan 2025); this is the only forward-looking 2026 choice |
| Serving | FastAPI (already) + `docker-compose` adding `nodeodm` + (optional) `gazebo` services | One `docker compose up` for the demo |

**Fix-first housekeeping (Phase 0):** re-init git (`git init` + first commit); fix the `str / Path` bug in [scripts/train_damage_seg.py](scripts/train_damage_seg.py):110 and [train_building_seg.py](scripts/train_building_seg.py):98; add `weasyprint` to requirements (used by [reporting/pdf_export.py](src/rescuevision/reporting/pdf_export.py) but not declared).

---

## Core vs. Stretch (team of 4+, parallelized)

**CORE — must demo end-to-end (the thesis spine):**
1. **xBD damage assessment** (existing paired branch) — polish + an ablation (U-Net vs FPN vs ChangeFormer). Reproduce ~0.74 combined F1.
2. **Gazebo + PX4 sim** generating georeferenced frames **with ground truth** (known victim GPS, damage, building areas).
3. **Pixel→GPS localization** validated against that sim truth → **report error in metres** (the credibility anchor).
4. **Visible-casualty detector** (YOLOv8-SARD) + **orthomosaic/DSM** (WebODM) from a self-flown AOI.
5. **Geo-referenced rescue routing** (hazard cost-map + A*/RRT*) overlaid on the orthomosaic → the "map handed to a rescuer."

**STRETCH — demonstrate at concept level if time allows:**
6. Single-image (no-before) damage triage.
7. People-in-danger counter (visible lower bound).
8. Indicative reconstruction-cost INDEX.
8b. One 3D Gaussian Splatting scene as a visual showpiece.
8c. Thermal-anomaly flagger (only if you buy a ~$200–400 thermal payload).

**Suggested ownership (4+ people):**
- **A — ML/CV:** damage models + ablation, post-only triage, victim detector, counting.
- **B — Geospatial & 3D:** `geo/` module, WebODM integration, pixel→GPS, DSM back-projection.
- **C — Routing & maps:** hazard cost-map, A*/RRT*/OSMnx, kepler.gl/folium + React map UI.
- **D — Simulation & integration:** Gazebo/PX4/ROS 2, synthetic capture, dashboard wiring, Docker compose, the end-to-end demo.

---

## Phased roadmap (~8–10 months)

### Phase 0 — Foundation (Weeks 1–3)
- Re-init git, fix the known bugs, add missing deps. Get the existing paired pipeline to actually **train** on xBD (download xBD, run `scripts/prepare_xbd.py` → train U-Net building + damage seg on GPU). Establish the benchmark you'll defend (~0.74 combined F1 target).
- Define the **canonical scene representation** (GeoTIFF + `scene.geojson` schema) and the `geo/` module interface.
- Stand up the Gazebo Harmonic + ROS 2 Jazzy + PX4 SITL "happy path": a quadcopter (`gz_x500`) flying over an empty world, publishing `/camera/image_raw`. Validate the install before any disaster content.

### Phase 1 — Core CV + Simulation truth (Weeks 4–12)
- Train/validate xBD damage branch; produce the U-Net/FPN/ChangeFormer ablation.
- Build the **Gazebo disaster world**: import rubble/building models (3DGEMS, osrf/gazebo_models, RoboCup Rescue arenas) → SDF; add simulated victims (visible + partially-exposed); capture synthetic geo-tagged frames **with ground-truth annotations** (pose, victim GPS, damage labels).
- **Validate pixel→GPS** against sim truth → measure and report localization error in metres. *This is the credibility anchor for the whole project.*
- Train YOLOv8 on SARD; cross-validate on HERIDAL/DLR; **measure the sim-to-real gap** explicitly (synthetic-only vs. fine-tuned).

### Phase 2 — Geospatial + 3D + Routing (Weeks 13–24)
- Implement the `geo/` EXIF/XMP + collinearity projector (real DJI photos) + uncertainty radius.
- Integrate **WebODM/NodeODM** as a Docker sidecar; produce orthomosaic + DSM + mesh from a self-flown AOI.
- Wire damage masks back onto the DSM for debris-height/volume estimates (a real, novel, defensible contribution).
- Build the **hazard cost-map** (damage class + DEM slope + flood + building-proximity) and the geo-space route planner (A*/RRT* raster + OSMnx graph); overlay routes on the orthomosaic in kepler.gl/React.
- Wire the (currently orphaned) route planner into the predictor; make the React dashboard's `RouteMap` component actually render a live geo map.

### Phase 3 — Stretch + Integration + Demo (Weeks 25–40)
- Add post-only damage triage; people counter; indicative cost INDEX.
- One 3D Gaussian Splatting scene for the "wow."
- **End-to-end demo:** drone (real or sim) → orthomosaic + DSM → damage map → victims localized (lat/lon + σ) → hazard map → rescue route → report. Plan a **canned-scenario playback** with 1–2 *live* modules (live demos over ROS/Gazebo are fragile).
- Honest limitations doc, model cards, accuracy tables for the defense.

---

## Per-feature build specs (concise)

**1. Single-image damage (no-before)** — new `post_only/` module. Footprint-conditioned DINOv2/DINOv3 ViT; binary standing-vs-collapsed head (F1 ~0.84–0.91) + advisory 4-class. Footprints from OSM (osmnx) or Microsoft Global ML Building Footprints; fall back to SAM2 proposals. Train on xBD-post-half + Maxar Open Data + LADI v2 (DamageTriage-Bench is *not public* — don't depend on it). **Do not** claim it distinguishes destroyed-vs-under-construction.

**2. Cost INDEX** — new `cost/` module, deterministic post-processing. `cost = repair_ratio[damage_class] × footprint_area × unit_cost_USD_per_m²[hazard, region]`. Repair ratios from EMS-98 (seismic) or Hazus (flood). Pick **one hazard + one event with a published GRADE total** (e.g., Nepal 2015 earthquake, Türkiye 2023) to validate against. Report as a prioritization index, never a true cost.

**3. Victim detection** — new `victims/`. `victim_detector.py` wraps `ultralytics.YOLOv8`, fine-tuned on SARD (`SARD_YOLO` Roboflow split), cross-validated on HERIDAL + DLR set. Thermal branch = **anomaly flagger** (surface heat candidates requiring ground confirmation) — there is **no public paired RGB-T SAR dataset** to train a fusion head. Ingest only UAV tiles; gate on GSD < ~6 cm/px. Lead every output with "visible/partially-exposed only."

**4. Pixel→GPS** — new `geo/`. EXIF + DJI XMP reader (gimbal pitch/yaw, `RelativeAltitude` — these live in the `drone-dji` XMP namespace, *not* standard EXIF; use ExifTool, not just piexif). Compute GSD; nadir pinhole back-projection (flat-earth) → lat/lon **+ uncertainty radius** (propagate attitude + GPS errors). Off-nadir >15° → discard or heavily widen σ.

**5. 2D/3D reconstruction** — new `reconstruction/`. NodeODM REST client (`opendronemap/nodeodm` Docker) → GeoTIFF orthomosaic + DSM + textured mesh. Optional nerfstudio `gsplat`/Splatfacto scene (one, as showpiece; note 3DGS/NeRF are *not* natively georeferenced). Reuse existing `utils/geometry.py` rasterio rasterize for mask→DSM back-projection.

**6. People counting** — new `counting/`. YOLOv8 (VisDrone-pretrained) + built-in ByteTrack over drone video + danger-zone geofence (count only detections inside the damage polygon, geo-registered). Refuse if GSD > ~15 cm/px. Output labelled **"visible persons within danger zone (lower bound)."**

**7. Routing** — **extend** `routing/route_planner.py` from pixel→geo space. `build_costmap.py`: damage_map (5-class) + DEM slope (scipy.ndimage.sobel) + flood_map + building-proximity → cost grid. Two planners: raster (`skimage.graph.route_through_array` or `pyastar2d`) + graph (`osmnx`+`networkx`, edge weight = length×risk). Deliver route overlay on the orthomosaic + the existing project safety disclaimer.

**8. Gazebo sim** — new top-level `sim/`. ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL (`make px4_sitl gz_x500_depth`). Disaster worlds from 3DGEMS / osrf models / RoboCup Rescue arenas. `ros_gz_bridge` → `/camera/image_raw`, optional `/thermal`, `/lidar/points`, world pose. Synthetic-capture scripts emit **ground-truth-annotated** frames. Use as integration harness + validation oracle + pre-training augmentation only.

---

## The unique angle (for examiners)

> **"RescueVision: a georeferenced UAV disaster-response co-pilot with simulation-validated localization."**

Three defensible novelty pillars:

1. **Georeferenced single-source-of-truth.** Every output — damage, victims (lat/lon + σ), routes, counts, cost — lives in one GeoJSON scene graph over a GeoTIFF orthomosaic+DSM (EPSG:4326). The map you hand a rescuer is *internally consistent* (victim GPS, route, and hazard agree). Most student projects are pixel-space silos; this coherence is the engineering contribution.
2. **Simulation-validated accuracy.** Gazebo+PX4 generates synthetic disaster scenes with **known ground truth** (true victim GPS, true damage, true building areas). Every accuracy claim is measured against this oracle and reported as a real error in metres / F1 / USD — examiners can't dismiss the numbers as unvalidated, because the one thing student disaster projects usually lack (ground truth) is exactly what you generated.
3. **Single-image (no-before) damage triage** as the ML research spark — footprint-conditioned DINOv2/DINOv3, honest binary + advisory multi-class, addressing the real-world case where no pre-disaster image exists.

This framing is ambitious, coherent, and *credible* — it stands out precisely because it doesn't over-promise.

---

## Claims to AVOID (honesty guardrails)

- **"Detects people under rubble."** → "Detects visible/partially-exposed casualties; buried victims require thermal/acoustic/GPR, out of scope."
- **"AI predicts reconstruction cost."** → "An imagery-derived *indicative cost index* for prioritization."
- **"Centimeter-accurate victim localization."** → "Meter-level localization from consumer drone; sub-meter requires RTK/GCP."
- **"Real-time full pipeline."** → Several seconds per 1024² tile on CPU; quantify and offer tiling/quantization if you need speed.
- **"Infinite free labeled training data from Gazebo."** → Synthetic is for pre-training/augmentation + a validation oracle; sim-to-real gap is ~16–32 mAP points without real fine-tuning.
- **"xView2 F1 ~0.86."** → The correct figure is **~0.74 combined** (30% loc + 70% damage); ~0.84 localization, ~0.71–0.78 damage.
- **"Counts everyone in danger."** → "Counts *visible* persons in the danger zone — a lower bound."

---

## Datasets & tools master list

**Datasets:** xBD / xView2 (paired, training) · Maxar Open Data · LADI v2 (low-altitude disaster) · NOAA Emergency Response Imagery · SARD (Search & Rescue, ~2k images / ~35.6k person boxes) · HERIDAL · DLR person-detection set · SeaDronesSee · VisDrone2019 · FloodNet · Google Open Buildings V3 / Microsoft Global ML Building Footprints · OSM (osmnx) · EMS-98 / FEMA Hazus repair-ratio tables · World Bank GRADE / PDNA reports (cost validation) · BRIGHT (IEEE GRSS DFC 2025).

**Tools:** PyTorch, `ultralytics` (YOLOv8/11 + ByteTrack), `segmentation_models_pytorch`, DINOv2/DINOv3, SAM2 · OpenDroneMap/WebODM/NodeODM, nerfstudio `gsplat`/Splatfacto · `rasterio`, `geopandas`, `shapely`, `pyproj`, `osmnx`, `networkx`, `scikit-image`, `pyastar2d` · `piexif`/`exifread` + ExifTool · kepler.gl / folium / MapLibre · ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL, `ros_gz_bridge` · 3DGEMS / osrf gazebo_models / RoboCup Rescue arenas.

---

## Testing & validation strategy

- **Unit tests** (extend [tests/](tests/)): geo projections, cost-map math, EXIF parsing, collinearity projector with synthetic known-pose fixtures, geofence counting.
- **The sim oracle:** Gazebo scenes with known victim GPS/pose/damage → measure pixel→GPS error in metres (the headline accuracy number), detector precision/recall vs. ground-truth boxes, route validity (does the path avoid marked hazards?).
- **ML benchmarks:** xBD damage F1 (target ~0.74 combined); SARD mAP@0.5 (expect below the curated ~0.9 on real-style rubble — report the honest number); sim-to-real gap table (synthetic-only vs. fine-tuned).
- **Co-registration sanity:** assert every layer in `scene.geojson` shares the same AOI/CRS and overlays cleanly on the orthomosaic.
- **End-to-end smoke:** `docker compose up` → run the canned scenario → all 5 core modules produce artifacts for one AOI.

---

## Verification (how to confirm it works end-to-end)

1. **Foundation:** `git log` shows history; `pytest -q` passes (after bug fixes); `scripts/prepare_xbd.py` then `train_damage_seg.py` produces a non-empty checkpoint; `run_inference.py --pre ... --post ...` yields a real damage map (no "No damage map available").
2. **Sim + localization:** launch the Gazebo disaster world, capture a frame with a known victim pose, run `geo/` projector, assert reported lat/lon is within the measured σ of ground truth.
3. **3D + routing:** POST a self-flown set of DJI photos to the NodeODM sidecar → orthomosaic + DSM produced; run the hazard cost-map + planner → route overlay renders on the kepler.gl map avoiding "destroyed" zones.
4. **Victims + counting:** run YOLOv8-SARD on a drone clip → detections geo-fenced to the danger polygon, count labelled as lower bound.
5. **Demo:** one canned AOI where the dashboard shows orthomosaic + damage + victim markers (lat/lon + σ) + rescue route + report, all consistent.
