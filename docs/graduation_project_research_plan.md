# RescueVision Graduation Project Research Plan

Research date: 2026-07-03

## Executive Decision

This project can become a strong graduation project, but it should be framed as a **decision-support and simulation system**, not as a fully autonomous lifesaving system. The realistic version is:

> RescueVision-X: a geospatial AI system that analyzes post-disaster UAV/satellite imagery, maps damaged buildings and blocked roads, estimates approximate building loss, detects visible victims, localizes detections on a map, plans safer rescue routes, and validates the pipeline in Gazebo/PX4 simulation.

The critical correction is this: **SARD-style person detection can detect visible or partly visible people in UAV imagery. It cannot reliably detect people buried under collapsed buildings from RGB images.** Under-rubble detection needs other sensing modalities such as radar, acoustic/seismic sensors, CO2 sensors, thermal cameras in favorable cases, or UGV-mounted sensors. That can be included as a research extension or simulated module, but it should not be promised as an operational UAV-photo feature.

## What RescueVision Already Has

The current project already includes a good base:

- xBD/xView2 building localization and damage classification pipeline.
- Pre/post image change detection.
- Building and damage segmentation models.
- FloodNet semantic segmentation path.
- FastAPI backend, React dashboard, report generation, PDF export.
- A route-planning module using masks and graph search.
- Unit tests for masks, routing, metrics, report schema, and xBD parsing.

This means the best upgrade path is not to restart. The best path is to add **post-only UAV scene understanding**, **geospatial fusion**, **visible-person detection**, **2D/3D reconstruction**, and **simulation validation** around the existing pipeline.

## Feature Feasibility Matrix

| Feature | Feasibility | Best graduation-project scope | Main limitation |
|---|---:|---|---|
| Analyze destroyed buildings with pre/post images | High | Keep existing xBD pipeline and improve evaluation/reporting | Requires aligned before/after imagery |
| Analyze buildings/streets without before image | Medium-high | Train post-only semantic segmentation on RescueNet/FloodNet-style data | Lower confidence than pre/post change detection |
| Analyze "everything" in the scene | Medium | Segment buildings, roads, water, trees, vehicles, debris/blockage | Open-world disaster scenes are not fully labelable |
| Estimate cost of destroyed buildings | Medium | Approximate loss estimate with confidence interval | True cost needs local unit costs, occupancy, material, floors, contents, structural inspection |
| Detect people using SARD | Medium-high for visible people | Fine-tune YOLO/RT-DETR on SARD and evaluate recall | SARD is not an under-rubble dataset |
| Detect people under buildings | Low from UAV RGB images | Present as sensor-fusion research extension only | Buried victims are occluded; visual AI cannot see through concrete/rubble |
| Get detected-person GPS from UAV photo | Medium | Geolocate detections using EXIF, camera calibration, gimbal pose, DEM/DSM, or orthomosaic | EXIF GPS is camera position, not target position |
| Make a rescue map and safe routes | High for prototype | Use road-clear/road-blocked segmentation plus OSM/network graph and risk weights | Roads can change after imagery capture; route must be verified |
| Build 2D map | High | Orthomosaic/GeoTIFF from OpenDroneMap or COLMAP/OpenSfM | Needs overlap, geotags, and preferably GCP/RTK |
| Build 3D map | Medium-high | Point cloud/mesh/DSM from photogrammetry | Dense rubble geometry needs many images and good coverage |
| Count people still in danger | Medium for visible people, low for buried people | Count visible detections and estimate exposed risk zones | Occupancy behind walls/rubble is not observable from imagery |
| Gazebo simulation | High for integration | Simulate UAV mission, camera feed, GPS, gimbal, hazards, routing loop | Simulation does not prove real-world AI accuracy by itself |

## Recommended System Architecture

### 1. Data Ingestion Layer

Inputs:

- UAV RGB images.
- Optional thermal images.
- Optional pre-disaster satellite/UAV images.
- Image metadata: GPS, altitude, timestamp, camera intrinsics, gimbal pitch/yaw/roll, drone attitude.
- Optional OpenStreetMap roads/buildings.
- Optional ground control points (GCPs) or RTK/PPK metadata.

Outputs:

- A normalized project folder per mission.
- A metadata table per image.
- GeoJSON/COG/GeoTIFF artifacts for dashboard and GIS tools.

Implementation direction:

- Add a metadata extractor using `exiftool` or Python EXIF/XMP parsing.
- Store image metadata in a mission-level JSON or SQLite/PostGIS database.
- Keep all uncertainty fields: GPS horizontal accuracy, altitude source, gimbal availability, GSD estimate.

### 2. Image Analysis Layer

Keep the current pre/post path:

- Building segmentation.
- Change detection.
- Damage segmentation/classification.
- xBD-style emergency summary.

Add a post-only path:

- Train a semantic segmentation model on RescueNet and FloodNet.
- Classes should initially be practical, not huge:
  - background
  - water/flood
  - building no damage
  - building minor damage
  - building major damage
  - building total destruction
  - road clear
  - road blocked
  - vehicle
  - tree/debris or obstacle

Recommended model choices:

- SegFormer-B2/B3 or Mask2Former for segmentation quality.
- YOLOv8/YOLOv10/RT-DETR for visible-person detection.
- DINOv2/remote-sensing foundation features for retrieval or low-label adaptation.

### 3. Person Detection and Localization Layer

SARD should be used only for **visible-person detection from UAV imagery**.

Pipeline:

1. Run person detector on UAV frame or orthomosaic tiles.
2. Keep detection confidence and bounding box.
3. Convert pixel location to map location:
   - Preferred: detect on orthomosaic tiles that are already georeferenced.
   - Alternative: ray-cast from camera through pixel using intrinsics, drone pose, gimbal pose, and DEM/DSM.
   - Better: use multi-view triangulation if the same person appears in multiple frames.
4. Export detections as GeoJSON points with error radius.

Important limitation:

- Image EXIF GPS gives the **camera/drone position**, not the exact location of the person in the image.
- Accurate localization needs camera calibration, pose, altitude/terrain model, and error modeling.
- For oblique images, localization error can grow quickly if gimbal angle, altitude, or terrain height is wrong.

### 4. Under-Rubble Victim Module

Do not claim RGB UAV AI can detect buried people. Use one of these realistic scopes:

- **Graduation-project-safe scope:** "visible victim detection and geolocation."
- **Research-extension scope:** "under-rubble survivor likelihood layer from external sensors."
- **Simulation-only scope:** simulate radar/CO2/acoustic detections in Gazebo or a custom environment and fuse them with the map.

Possible sensors:

- UWB/low-frequency radar for motion/breathing behind debris.
- Ground penetrating radar for void detection.
- Microphones/acoustic listening devices.
- CO2 sensors.
- Thermal cameras when heat signature is exposed or lightly occluded.
- UGV/crawler robot cameras in voids.

System output should be:

- `visible_person_detected`
- `possible_survivor_signal`
- `sensor_type`
- `confidence`
- `location_error_m`
- `needs_human_verification`

### 5. Cost Estimation Layer

The cost feature is possible as **approximate loss estimation**, not exact construction cost.

Recommended formula:

```text
estimated_loss =
  gross_floor_area_m2
  * local_replacement_cost_per_m2
  * damage_ratio
  * regional_factor
  * uncertainty_factor
```

Inputs:

- Building footprint area from segmentation or OSM.
- Floors/height from 3D reconstruction, DSM, LiDAR, OSM tags, or user input.
- Occupancy type: residential, commercial, school, hospital, industrial.
- Structural type/material if available.
- Damage class from AI.
- Local replacement-cost table.

Output:

- Low/medium/high loss estimate.
- Confidence interval.
- Missing-data warnings.
- A note that this is not an engineering cost estimate.

Best prototype behavior:

- If floors/material/occupancy are unknown, ask the user or use default assumptions and show them.
- Use broad ranges instead of one exact number.
- Provide per-building and total-area loss estimates.

### 6. Route Planning Layer

Current route planning can be upgraded into a geospatial route planner.

Inputs:

- Road network from OpenStreetMap or detected road mask.
- Road-clear/road-blocked segmentation from RescueNet/FloodNet model.
- Hazard polygons: destroyed buildings, debris, floodwater, fire/smoke if available.
- Rescue target points: visible people, possible survivor signals, high-risk buildings.
- Safe zones: medical point, staging area, exit points.

Graph cost:

```text
edge_cost =
  distance
  + blocked_penalty
  + nearby_destroyed_building_penalty
  + flood_penalty
  + uncertainty_penalty
  + slope_or_accessibility_penalty
```

Algorithms:

- A* or Dijkstra for baseline routing.
- D* Lite or repeated A* for dynamic updates.
- Multi-target route ordering for multiple victims.
- Service-area analysis: "what can responders reach in 5, 10, 15 minutes?"

Output:

- Route GeoJSON.
- Blocked-road layer.
- Danger heatmap.
- Alternative route list.
- "Needs verification" markers for uncertain roads.

### 7. 2D and 3D Mapping Layer

Use photogrammetry instead of building from scratch.

Recommended tools:

- OpenDroneMap/WebODM for orthomosaic, DSM/DTM, point cloud, textured mesh.
- COLMAP/OpenMVS as a lower-level alternative.
- CesiumJS or Potree for 3D visualization.
- Leaflet or MapLibre for 2D maps.

Inputs:

- 70-85% frontlap and sidelap image capture for mapping.
- Nadir plus oblique images for buildings and rubble.
- GCPs or RTK/PPK if absolute accuracy matters.

Outputs:

- Orthomosaic GeoTIFF.
- DSM/DTM.
- Point cloud.
- Mesh/3D Tiles.
- Damage and route layers overlaid on the map.

Accuracy expectation:

- Without GCP/RTK, the map can look geometrically good but be shifted by meters.
- With GCP/RTK and good flight planning, absolute accuracy can improve significantly.
- Map accuracy must be measured and reported, not assumed.

### 8. Gazebo/PX4 Simulation Layer

Use Gazebo for robotics integration, not as the only model validation source.

Recommended stack:

- ROS 2 Humble/Jazzy.
- Gazebo Harmonic or Garden.
- PX4 SITL or ArduPilot SITL.
- `ros_gz_bridge` for camera/depth/LiDAR topics.
- QGroundControl for mission monitoring.

Simulated environment:

- Collapsed buildings and debris meshes.
- Blocked roads.
- Safe zones and rescue targets.
- Simulated UAV camera with GPS, altitude, and gimbal orientation.
- Optional simulated thermal/depth/radar detections as separate topics.

What simulation can validate:

- End-to-end data flow.
- UAV mission execution.
- Metadata and geolocation math.
- Routing behavior.
- Dashboard updates.
- Failure handling.

What simulation cannot validate alone:

- Real disaster image generalization.
- True survivor detection reliability.
- Real photogrammetry accuracy under smoke/dust/wind.
- True cost-estimation accuracy.

## Recommended Graduation Project Scope

Do not try to implement every idea at full operational quality. The strongest feasible scope is:

1. **Post-only disaster scene understanding**
   - Train/evaluate semantic segmentation for damaged buildings, clear roads, blocked roads, water/debris.
   - Use RescueNet/FloodNet.

2. **Geospatial 2D/3D map generation**
   - Use OpenDroneMap or COLMAP to create orthomosaic, DSM, and 3D mesh.
   - Overlay AI outputs as GeoJSON.

3. **Visible-person detection and geolocation**
   - Train/evaluate SARD-based visible person detector.
   - Convert detections to map coordinates with uncertainty radius.

4. **Risk-aware rescue routing**
   - Generate routes avoiding blocked/danger zones.
   - Show alternative routes and reachable areas.

5. **Approximate building-loss estimation**
   - Estimate per-building loss using footprint, assumed floors/material/occupancy, damage class, and local cost table.
   - Always output confidence intervals and warnings.

6. **Gazebo/PX4 simulation demo**
   - Simulate UAV mission and run the perception-to-routing pipeline on captured/simulated imagery.

This is already large and unique enough for a graduation project.

## Suggested Project Title Options

- RescueVision-X: Geospatial AI for Post-Disaster Damage Mapping and Rescue Route Planning
- RescueVision 3D: UAV-Based Disaster Scene Understanding, Victim Localization, and Safe Routing
- RescueVision-GIS: A Simulation-Tested AI System for Post-Disaster Mapping and Response Prioritization
- RescueVision SAR: UAV Disaster Intelligence for Damage, Road Blockage, Visible Victim Detection, and Rescue Routing

## Evaluation Plan

### Model-Level Metrics

Building/damage segmentation:

- mIoU
- class IoU
- Dice/F1
- per-class precision/recall
- confusion matrix

Road blockage:

- road-clear IoU
- road-blocked IoU
- blocked-road recall
- false-safe rate: blocked road predicted as safe

Person detection:

- mAP@0.5
- mAP@0.5:0.95
- recall at low confidence thresholds
- false positives per image/km2
- inference FPS

Geolocation:

- mean error in meters
- median error
- 90th/95th percentile error
- error radius calibration: does the true point fall inside the predicted uncertainty circle?

Routing:

- route success rate
- length/time compared to shortest path
- hazard intersections count
- blocked-road avoidance rate
- recomputation time

Cost estimation:

- mean absolute error if ground-truth cost exists
- mean absolute percentage error if reliable cost labels exist
- uncertainty coverage
- missing-data rate

### System-Level Tests

- Upload UAV image set with metadata and generate map layers.
- Run post-only segmentation and route planning.
- Run person detection and geolocation.
- Generate emergency report with maps, route, costs, warnings, and confidence.
- Run a simulated Gazebo mission and process captured frames.

### Field-Test-Like Demo Without Real Disaster

Use a safe controlled area:

- Place mannequins or printed human targets.
- Place fake road obstacles.
- Use visible GCP targets.
- Fly or simulate a grid mission.
- Measure true target positions using phone GNSS, RTK, or surveyed points.
- Compare predicted location to ground truth.

## Main Limitations To State Clearly

1. **No-before-image damage analysis is less reliable.**
   Without a pre-disaster reference, the model must infer damage from appearance only. It can confuse old ruins, construction sites, poor roofing, shadows, debris, and normal irregular structures.

2. **Satellite and UAV models do not generalize automatically.**
   xBD, FloodNet, RescueNet, and SARD differ in altitude, camera angle, disaster type, country, lighting, and labels. A model that works on one dataset can fail elsewhere.

3. **Destroyed-building cost cannot be exact from imagery alone.**
   Imagery does not reveal full structural system, foundation damage, interior damage, contents, code requirements, labor cost, or market conditions.

4. **SARD does not solve under-rubble rescue.**
   It supports visible-person detection from drone views. Buried victims require sensor fusion and human rescue procedures.

5. **GPS from a UAV image is not enough.**
   The drone location is not the target location. Accurate target geolocation requires camera model, orientation, altitude/terrain, and calibration.

6. **Routes are recommendations, not orders.**
   The map may be outdated minutes after capture. Rescuers must verify hazards, structural collapse risk, fire, gas, electricity, crowding, and access constraints.

7. **3D reconstruction has a capture-quality dependency.**
   Bad overlap, rolling shutter, wind, smoke, dust, reflective surfaces, water, or repetitive textures can break reconstruction.

8. **Gazebo has a sim-to-real gap.**
   It is excellent for integration and repeatable tests, but synthetic scenes do not prove real-world perception accuracy.

9. **Ethics and safety are central.**
   False negatives can cost lives. False positives can waste time. All outputs must be human-reviewed and logged with uncertainty.

## Build Roadmap

### Phase 1: Baseline Reliability

- Train or validate current xBD/FloodNet models.
- Produce real metrics, not only screenshots.
- Improve `docs/experiments.md` with completed runs.
- Ensure the API and dashboard run end-to-end.

### Phase 2: Post-Only Disaster Understanding

- Add RescueNet dataset preparation.
- Train SegFormer/Mask2Former for post-only classes.
- Add inference output for building damage, road clear, road blocked, water/debris.
- Add uncertainty and warnings.

### Phase 3: Geospatial Mapping

- Add mission metadata parser.
- Integrate OpenDroneMap outputs as optional input.
- Convert masks to GeoJSON polygons.
- Display layers in Leaflet/MapLibre.

### Phase 4: Visible Person Detection

- Prepare SARD in YOLO format.
- Train/evaluate detector.
- Add person-detection API endpoint.
- Add geolocation by orthomosaic tile first, then optional ray-casting from single frame.

### Phase 5: Risk-Aware Routing

- Build route graph from OSM or detected road mask.
- Add blocked-road and hazard penalties.
- Return best route plus alternatives.
- Show error/confidence on map.

### Phase 6: Loss Estimation

- Add cost table schema.
- Estimate per-building floor area and loss range.
- Add user-editable assumptions in report/dashboard.
- Avoid single exact numbers unless required.

### Phase 7: Gazebo/PX4 Simulation

- Build a collapsed-area world.
- Add UAV camera and GPS/gimbal metadata.
- Capture frames through ROS 2.
- Run end-to-end pipeline on simulation output.
- Measure route/planning/geolocation behavior against known world coordinates.

## Strong Demo Scenario

The final demo should tell one coherent story:

1. A UAV surveys a simulated damaged area in Gazebo.
2. The system builds or loads a georeferenced 2D/3D map.
3. The AI segments destroyed buildings, blocked roads, water/debris, and visible people.
4. Detected people are placed on the map with uncertainty circles.
5. The system estimates approximate building losses.
6. The route planner gives a safe route from staging area to target and an exit route.
7. The dashboard exports an emergency report.
8. The report clearly states limitations and required human verification.

## Source Notes

- xBD provides pre/post satellite imagery, building polygons, damage labels, and metadata for disaster damage assessment: https://arxiv.org/abs/1911.09296
- SARD research evaluates UAV visible-person detection in SAR scenes and selects YOLOv4 for speed/accuracy in that dataset context: https://scispace.com/pdf/automatic-person-detection-in-search-and-rescue-operations-55nr5l6i18.pdf
- RescueNet provides post-disaster UAV semantic segmentation classes including building damage and road-clear/road-blocked labels: https://www.nature.com/articles/s41597-023-02799-4
- RescueNet dataset repository lists 4,494 UAV images and segmentation classes: https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation
- FloodNet provides high-resolution UAV post-flood imagery for segmentation/VQA and includes flooded roads/buildings challenges: https://arxiv.org/abs/2012.02951
- OpenDroneMap documents map accuracy, GCPs, RTK/GPS effects, and geolocation file formats: https://docs.opendronemap.org/map-accuracy/
- Structure-from-Motion photogrammetry can create georectified mosaics and point clouds from UAV imagery: https://www.mdpi.com/2072-4292/4/5/1392
- UAV target geolocation from image pixels requires camera parameters, GNSS/INS pose, terrain/altitude assumptions, and benefits from multi-view optimization: https://www.mdpi.com/2504-446X/8/5/177
- FEMA Hazus provides standardized tools/data for estimating earthquake, flood, tsunami, and hurricane risk/losses: https://www.fema.gov/flood-maps/products-tools/hazus
- Hazus inventory documentation uses structure replacement cost models derived from industry cost-estimation sources such as RSMeans: https://www.fema.gov/sites/default/files/documents/fema_hazus-6-inventory-technical-manual.pdf
- MIT Lincoln Laboratory describes radar for detecting breathing motion under rubble/debris, illustrating why special sensors are needed for buried survivors: https://www.ll.mit.edu/r-d/projects/motion-under-rubble-measured-using-radar
- Ground-penetrating radar research targets void detection and localization in rubble for SAR: https://par.nsf.gov/servlets/purl/10132677
- PX4 supports Gazebo SITL, custom worlds, camera models, and video streaming: https://docs.px4.io/main/en/sim_gazebo_gz/index
- ArduPilot supports SITL with Gazebo for Copter/Plane/Rover: https://ardupilot.org/dev/docs/sitl-with-gazebo.html
- AirSim is useful background but the original Microsoft research project is archived with no further updates: https://www.microsoft.com/en-us/research/project/aerial-informatics-robotics-platform/
- HOT shows disaster response use of drone/satellite imagery and open geospatial mapping for responders: https://www.hotosm.org/en/impact-areas/disaster-response/
- QGIS network analysis documents shortest/fastest path and service-area workflows relevant to route validation: https://docs.qgis.org/latest/en/docs/training_manual/vector_analysis/network_analysis.html
