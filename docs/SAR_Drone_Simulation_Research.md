# Drone Search & Rescue (SAR) Simulation — Complete Research Findings

## Your Situation
- Graduation project about multi-model drone SAR after disasters
- Want: scan area → assess damage → map destroyed city → detect bodies/survivors → fire detection → plan entry/exit paths
- Input images from drones or satellites
- **Constraints:** No real drone, no dataset, no ready simulator, don't want to build a sim from scratch

---

## 1. READY-MADE SIMULATORS (No Hardware Needed)

### TIER 1: Best Options for Your Project

### A) AirSim (Microsoft) — **TOP RECOMMENDATION**
- **What:** Open-source drone/car simulator built on Unreal Engine
- **GitHub:** https://github.com/microsoft/airsim
- **Website:** https://microsoft.github.io/AirSim/
- **Why it's perfect for you:**
  - Flies a virtual drone in realistic 3D cities
  - Captures RGB images, depth, segmentation from cameras
  - Python API (`airsim` package) to control the drone programmatically
  - Supports multiple drones (swarms)
  - Weather effects (rain, snow, fog, dust)
  - Works with PX4 SITL (software-in-the-loop) — real flight controller code
  - Can load Unreal Engine city environments (urban maps with buildings)
- **Can it do disaster scenes?** Not out of the box, but:
  - You can import destroyed-building 3D assets from Unreal Marketplace (many free)
  - UE5 has "Chaos Destruction" built in — you can destroy buildings procedurally
  - UE5 has "Procedural Content Generation" (PCG) plugin for city generation
- **Output:** Captures thousands of images automatically via Python script
- **Difficulty:** Medium (need to install UE + AirSim plugin)
- **Note:** Microsoft discontinued commercial "Project AirSim" but the open-source version is fully available and widely used in research

### B) PX4 SITL + Gazebo + ROS 2 — **MOST REALISTIC FLIGHT DYNAMICS**
- **What:** Real PX4 autopilot firmware running in simulation (Software-In-The-Loop) + Gazebo physics world + ROS 2 middleware
- **Docs:** https://docs.px4.io/main/en/simulation/
- **Why it's useful:**
  - The EXACT same code that flies real drones — extremely realistic
  - Gazebo simulates physics, wind, collisions, sensors
  - You can attach virtual cameras, LiDARs, thermal sensors to the drone
  - ROS 2 connects everything — computer vision, path planning, control
  - QGroundControl provides a GUI to plan missions (waypoints, survey patterns)
  - Huge community, tons of tutorials
- **Can it do disaster environments?**
  - You can build/import Gazebo worlds with destroyed buildings
  - Search "Gazebo disaster world" or "USARSim Gazebo plugin"
  - There's a GitHub repo: https://github.com/basfora/SearchandRescue (SAR simulator on Gazebo)
- **Difficulty:** Medium-High (ROS 2 learning curve)
- **Best for:** If you want to show realistic flight control + mission planning

### C) CARLA + CARLA-Air — **BEST FOR REALISTIC CITY VISUALS**
- **What:** Open-source autonomous driving simulator, now extended for drones
- **Website:** https://carla.org/
- **CARLA-Air paper:** https://arxiv.org/html/2603.28032v1
- **Why it's useful:**
  - Hyper-realistic urban environments (cities, streets, buildings)
  - CARLA-Air adds drone simulation with realistic flight dynamics
  - Can attach RGB camera, semantic segmentation camera, depth camera to drones
  - Full control of weather, lighting, time of day
  - Dynamic actors: cars, pedestrians (you can simulate people needing rescue!)
  - Python API for full control
- **Can it do disaster scenes?**
  - Not natively, but the city visuals are photorealistic
  - You can modify maps, remove buildings, add debris
  - Semantic segmentation gives you perfect ground-truth labels for training AI
- **Difficulty:** Medium
- **Best for:** If visual realism and generating labeled training images is your priority

---

### TIER 2: Good Alternatives

### D) CoppeliaSim (formerly V-REP)
- **Website:** https://www.coppeliarobotics.com/
- **Pros:** Built-in path planning (OMPL library), easy drone simulation, good physics
- **Cons:** Less photorealistic visuals than AirSim/CARLA
- **Best for:** Path planning algorithms focus

### E) Webots
- **Pros:** Very easy to use, good documentation, educational
- **Cons:** Simpler graphics, fewer disaster-relevant assets
- **Best for:** Quick prototyping

### F) gym-pybullet-drones (PyBullet)
- **GitHub:** https://github.com/utiasDSL/gym-pybullet-drones
- **What:** Lightweight Python-only drone simulator for reinforcement learning
- **Pros:** Pure Python, super easy install (`pip install`), fast, works with OpenAI Gym
- **Cons:** Very basic graphics (no photorealistic cities)
- **Best for:** If you want to train RL path planning agents quickly

### G) RotorPy
- **GitHub:** https://github.com/spencerfolk/rotorpy
- **What:** Python multirotor simulator with aerodynamic models
- **Best for:** Control research, trajectory tracking

---

### TIER 3: Domain-Specific Simulators

### H) USARSim (Urban Search & Rescue Simulator)
- **What:** The classic SAR simulator used in RoboCup Rescue competitions
- **Download:** https://sourceforge.net/projects/usarsim/
- **Based on:** Unreal Tournament game engine
- **Note:** Older but specifically designed for SAR scenarios (collapsed buildings, rubble, victims)
- **Gazebo version exists:** "USARSim Gazebo Plugin" brings SAR to modern Gazebo

### I) RoboCup Rescue Simulation
- **What:** Multi-agent disaster simulation used in international competitions
- **Includes:** Simulated civilians, fires, collapsed buildings, road blockages, traffic
- **Great for:** The "multi-model" aspect of your project (agents cooperate to assess damage, find victims)

### J) OpenSees (Earthquake Collapse Simulation)
- **Website:** https://opensees.berkeley.edu/
- **What:** Simulates building structural response to earthquakes, including collapse
- **Use:** Generate realistic damage patterns to feed into your city model

---

## 2. DATASETS (You Said You Don't Have One — Here They Are!)

### Building Damage Assessment Datasets

### A) xBD Dataset — **THE #1 DATASET FOR YOUR PROJECT**
- **Paper:** https://arxiv.org/abs/1911.09296 (cited 468+ times)
- **What:** Largest building damage dataset — 850,736 building annotations across 45,362 km²
- **Contains:** Pre-disaster AND post-disaster satellite image pairs from 19 disasters (floods, fires, earthquakes, tsunamis, volcanoes, wind, etc.)
- **Labels:** 4 damage levels per building: no damage, minor, major, destroyed
- **15 countries** represented
- **How to get:** Available via xView2 challenge — search "xBD dataset download"
- **WHY PERFECT:** This is exactly "before and after" damage assessment that you described

### B) Extensible Building Damage (EBD) Dataset — 2025
- **Paper:** https://spj.science.org/doi/10.34133/remotesensing.0733
- **What:** 18,000+ pre/post-disaster image pairs from 12 recent disaster events
- **Newer and more extensible than xBD**

### C) Ida-BD Dataset
- **What:** 87 high-resolution (0.5m/pixel) pre/post-disaster satellite pairs from Hurricane Ida 2021
- **Link:** https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3563

### D) MultiSense Dataset
- **What:** Combines satellite + drone imagery for damage assessment
- **Link:** https://data.mendeley.com/datasets/krkft96n43/2

### E) Incidents1M
- **What:** 1 million+ images of natural disasters, damage, and incidents
- **Great for:** Broad disaster type classification

### F) USOSAT Damage Assessments (Real Data)
- **What:** Real satellite-based damage assessments from actual disasters (Gaza, Ukraine, earthquake zones)
- **Link:** https://unosat.org/ and https://data.humdata.org/organization/unosat
- **Example:** Gaza damage assessment — 81% of structures damaged (Oct 2025)

### G) Copernicus EMS
- **What:** EU's rapid mapping service — free damage assessment maps from real disasters
- **Great for:** Validation data

### H) Maxar Open Data
- **What:** Free pre/post-disaster satellite imagery for major events
- **Link:** https://www.maxar.com/open-data

---

### Person/Victim Detection Datasets

### A) Person Detection in Disaster Response (Bahmanyar et al., 2023)
- **What:** 311 annotated aerial + drone images from multiple flying vehicles
- **Paper:** https://elib.dlr.de/197394/1/PersonDetection.pdf
- **Specifically designed for SAR**

### B) Thermal Human Detection from UAV (Roboflow)
- **Link:** https://universe.roboflow.com/thermal-disasters-project/thermal-human-detection-from-uav
- **What:** 654 thermal images for detecting humans from drones

### C) POP Dataset (Partially Occluded Person) — 2025
- **What:** UAV-based infrared thermal imaging for partially occluded person detection
- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11840078/

### D) Human Detection in UAV Thermal Imagery (2025)
- **What:** Thermal images from DJI M3T drone in mountainous environments
- **Paper:** https://www.mdpi.com/2313-433X/11/12/436

### E) ArcGIS Human Detection Model
- **Link:** https://hub.arcgis.com/content/42bfd5392d834c83aa21193450888a9e
- **What:** Deep learning model to detect humans in drone imagery for SAR

---

### Fire Detection Datasets

### A) D-Fire Dataset — **BEST FOR FIRE**
- **GitHub:** https://github.com/AlimTleuliyev/wildfire-detection
- **What:** 21,000 labeled images annotated in YOLO format (fire + smoke)
- **Ready to train YOLOv8 out of the box**

### B) FLAME Dataset
- **Link:** https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
- **What:** Aerial drone imagery of forest fires (RGB + thermal)
- **Captured by drones during prescribed burns**

### C) FLAME 2 (RGB + IR)
- **What:** Dual-feed RGB + thermal videos of fires from drones
- **Paper:** Cited 257 times

### D) Aerial Wildfire Detection (Roboflow)
- **Link:** https://universe.roboflow.com/use-cases/wildfire-detection-from-aerial
- **What:** Drone/satellite fire + smoke segmentation datasets

---

## 3. BUILDING DAMAGE ASSESSMENT — AI Models & Methods

### Key Models/Papers:

### A) ChangeOS (Object-based Change Detection)
- **What:** Deep learning framework for building damage assessment
- **Paper:** Cited 616 times
- **Approach:** Segments buildings first, then detects damage between pre/post images

### B) U-Net with Attention for Building Damage
- **Repo:** https://github.com/mmuneeburahman/Natural-Disaster-Damage-Assessment-Deep-Learning
- **What:** U-Net with attention mechanism for damage detection from satellite images
- **Links to datasets and papers too**

### C) Siamese Networks for Change Detection
- **What:** Twin network that compares pre/post images to find changes
- **Very common approach for damage assessment**

### D) YOLOv8 for Building Detection
- **Paper:** Uses YOLOv8 Lite to detect buildings in pre/post images, then counts damaged ones
- **Link:** https://www.mdpi.com/2072-4292/16/22/4263

### E) SAM (Segment Anything Model) for Buildings
- **What:** Meta's foundation model, adapted for satellite imagery building segmentation
- **Library:** samgeo (https://samgeo.gishub.org/)
- **Zero-shot building footprint extraction**

### F) RasterVision
- **What:** Deep learning pipeline for satellite imagery (object detection, segmentation, change detection)
- **Open source**

### G) TorchGeo
- **What:** PyTorch library for geospatial data — loads satellite datasets easily
- **Great for training damage models**

### Recommended Damage Assessment Pipeline:
```
1. Input: Pre-disaster image + Post-disaster image (xBD dataset)
2. Step 1: Extract building footprints (SAM / U-Net)
3. Step 2: Compare buildings pre/post (Siamese network / ChangeOS)
4. Step 3: Classify damage level (no/minor/major/destroyed)
5. Step 4: Output damage map with colored overlays
6. Step 5: Calculate % of city destroyed, estimate economic loss
```

---

## 4. PATH PLANNING — Entry/Exit Routes

### Algorithms:
- **A* (A-Star):** Classic grid-based pathfinding, very common
- **RRT / RRT* / Informed RRT*:** Good for continuous 3D spaces (drone flight)
- **Dijkstra:** Shortest path on road networks
- **Coverage Path Planning:** Lawnmower/zigzag patterns for scanning areas

### Tools:
- **OMPL Library:** Built into CoppeliaSim, available standalone for Python/C++
- **NetworkX + OSMnx:** Python libraries for road network routing
  - Load real city road networks from OpenStreetMap
  - Mark damaged/blocked roads
  - Find shortest safe path for rescue teams
- **pgRouting:** PostgreSQL extension for road network routing
- **Open Motion Planning Library (OMPL):** https://ompl.kavrakilab.org/

### Path Planning Pipeline:
```
1. Input: City map + damage map (from step 3)
2. Step 1: Build road network graph (OSMnx)
3. Step 2: Remove blocked roads (damaged/destroyed)
4. Step 3: A* or Dijkstra to find shortest safe entry route
5. Step 4: Find alternate exit routes
6. Step 5: Optimize for rescue vehicle constraints
7. Output: Optimal entry/exit paths on the damage map
```

---

## 5. LOSS ESTIMATION

### Tools & Methods:

### A) FEMA HAZUS
- **What:** Government-standard loss estimation software (free)
- **Estimates:** Building damage, economic losses, casualties for earthquakes, floods, hurricanes, tsunamis
- **The industry standard for this**

### B) PAGER (USGS)
- **What:** Rapid earthquake impact estimation
- **Estimates:** Population exposure, casualties, economic loss

### C) Open-Source GIS Loss Estimation
- **Paper:** "An Open Source GIS-based Tool for Economic Loss Estimation"
- **Uses:** Building polygons × damage level × unit cost = economic loss

### D) Custom Loss Model (What you can build):
```
Total Loss = Σ(building_area_i × damage_factor_i × cost_per_m²_i)
where damage_factor: no damage=0, minor=0.2, major=0.6, destroyed=1.0
```

### Datasets for Loss Validation:
- **EM-DAT:** International disaster database (mortality + economic damage)
- **DesInventar:** Historical disaster loss database
- **INFORM Risk Index:** Global disaster vulnerability data

---

## 6. CITY MAPPING & 3D RECONSTRUCTION

### A) OpenDroneMap / WebODM — **OPEN SOURCE DRONE MAPPING**
- **Website:** https://www.opendronemap.org/
- **What:** Converts drone images into orthophoto maps, 3D point clouds, DEMs
- **WebODM:** Web interface version, runs in Docker
- **Use:** If your "drone" images are from a simulator, feed them here to build maps

### B) Potree — **3D POINT CLOUD WEB VIEWER**
- **GitHub:** https://github.com/potree/potree
- **What:** Free WebGL point cloud viewer for billions of points
- **Use:** Visualize the 3D reconstructed city in a browser

### C) CloudCompare — **POINT CLOUD PROCESSING**
- **What:** Free 3D point cloud editing/comparison tool
- **Use:** Compare pre/post disaster point clouds to measure volume of debris

### D) COLMAP / OpenSfM — **STRUCTURE FROM MOTION**
- **What:** Reconstruct 3D from 2D images
- **Use:** Build 3D models from your simulated drone photos

### E) Cesium / CesiumJS — **3D GLOBE VISUALIZATION**
- **What:** 3D globe + building models in browser
- **Cesium OSM Buildings:** 3D building models for entire world from OpenStreetMap
- **Use:** Display your damage assessment on a 3D globe

### F) Google Earth Engine
- **What:** Cloud platform for satellite imagery analysis
- **Free for academic/research use**
- **Use:** Access satellite data, run change detection at scale

---

## 7. PROGRAMMATIC CITY GENERATION (If You Need to Build Disaster Environments)

### A) Unreal Engine 5 (for AirSim)
- **Procedural Content Generation (PCG) plugin:** Generate cities automatically
- **Chaos Destruction:** Physically destroy buildings (realistic rubble)
- **Building Destruction Generator (BDG):** Blueprint for quick destroyed buildings
- **CityBLD plugin:** Create urban environments from real-world data
- **Quixel Megascans:** Free photorealistic assets (destroyed buildings, debris)

### B) Blender / BlenderProc
- **BlenderProc:** Python framework for synthetic data generation
- **Generate procedural cities + destroy them + render from drone perspective**
- **Output:** Perfectly labeled training images

### C) OpenStreetMap → 3D Cities
- **Cesium OSM Buildings:** World coverage
- **3D City Reconstruction from OSM:** (Kaszuba & Pellacini paper)
- **D5 Render, CityEngine:** Generate 3D from OSM footprints

---

## 8. COMPUTER VISION MODELS (Ready to Use)

### Person/Victim Detection:
- **YOLOv8 / YOLOv11:** State-of-the-art real-time object detection
- **Pre-trained on COCO** (includes "person" class) — works immediately
- Fine-tune on SAR datasets for better performance

### Fire Detection:
- **D-Fire dataset + YOLOv8** = ready pipeline
- Pre-trained fire detection models on Roboflow Hub

### Building Segmentation:
- **SAM (Segment Anything)** — zero-shot segmentation
- **samgeo** — SAM adapted for geospatial/satellite images

### Change Detection (Pre/Post Disaster):
- **Siamese U-Net** networks
- **ChangeOS** framework

### Zero-Shot Analysis (No Training Needed!):
- **CLIP:** Zero-shot classification ("fire", "rubble", "collapsed building")
- **GPT-4o / GPT-4V Vision:** Send drone image → ask "what damage do you see?"
- **LLaVA, InternVL:** Open-source vision-language models for image understanding
- **GeoChat:** Conversational AI specifically for geospatial/satellite images

---

## 9. MY RECOMMENDED ARCHITECTURE FOR YOUR GRAD PROJECT

### Phase 1: Simulation Environment
```
Choose ONE simulator:
├── AirSim + Unreal Engine 5  (best visuals, realistic drone)
├── PX4 SITL + Gazebo + ROS 2 (best flight realism)
└── CARLA-Air (best city environments + pedestrians)
```

### Phase 2: Generate/Get Data
```
Option A: Use AirSim to fly drone over UE5 city → capture images
Option B: Use existing datasets (xBD for damage, D-Fire for fire, FLAME for aerial fire)
Option C: Generate synthetic with BlenderProc (procedural destroyed city)
```

### Phase 3: AI Models (The "Multi-Model" Part)
```
Module 1: Building Damage Assessment
├── Input: pre/post images (xBD)
├── Model: Siamese U-Net or ChangeOS
└── Output: Damage classification map

Module 2: Victim Detection
├── Input: high-res drone images
├── Model: YOLOv8 (person class) or thermal detection
└── Output: Bounding boxes of people/bodies

Module 3: Fire Detection
├── Input: RGB or thermal drone images
├── Model: YOLOv8 trained on D-Fire dataset
└── Output: Fire/smoke bounding boxes

Module 4: Loss Estimation
├── Input: Damage map from Module 1
├── Model: Area × damage factor × cost
└── Output: Economic loss estimate

Module 5: Path Planning
├── Input: Damage map + road network (OSM)
├── Algorithm: A* / Dijkstra on NetworkX
└── Output: Safe entry/exit routes

Module 6: City Mapping
├── Input: Drone images
├── Tool: OpenDroneMap
└── Output: Orthophoto map / 3D model of destroyed city
```

### Phase 4: Dashboard/Visualization
```
├── Web app: Streamlit or Gradio (Python, super easy)
├── Map: Leaflet / Folium / CesiumJS
├── 3D: Potree for point clouds
└── Show: Damage map, victim locations, fire zones, safe routes, loss stats
```

### Phase 5: Report Generation (Bonus)
```
├── Use LLM (GPT-4o) to analyze images and generate text reports
├── Auto-generate: "X buildings destroyed, Y people detected, Z% of city damaged"
└── RAG pipeline: disaster response guidelines database
```

---

## 10. KEY LINKS SUMMARY

### Simulators:
| Simulator | Link | Difficulty | Graphics | Flight Realism |
|-----------|------|-----------|----------|----------------|
| **AirSim** | github.com/microsoft/airsim | Medium | Excellent | High |
| **PX4+Gazebo+ROS2** | docs.px4.io | Hard | Good | Very High |
| **CARLA-Air** | carla.org | Medium | Excellent | Medium |
| **CoppeliaSim** | coppeliarobotics.com | Easy | Fair | Good |
| **gym-pybullet-drones** | github.com/utiasDSL/gym-pybullet-drones | Easy | Basic | Fair |
| **Webots** | cyberbotics.com | Easy | Fair | Good |

### Datasets:
| Dataset | Task | Size | Link |
|---------|------|------|------|
| **xBD** | Building damage | 850K buildings | arxiv.org/abs/1911.09296 |
| **D-Fire** | Fire/smoke detection | 21K images | github.com/AlimTleuliyev/wildfire-detection |
| **FLAME** | Aerial fire (drone) | RGB+thermal | ieee-dataport.org |
| **Person Detection SAR** | Victim detection | 311 images | elib.dlr.de/197394/1/PersonDetection.pdf |
| **Thermal Human UAV** | Thermal person detection | 654 images | roboflow.com |
| **USOSAT** | Real damage maps | Real events | unosat.org |
| **Maxar Open Data** | Pre/post satellite | Major events | maxar.com/open-data |

### Tools:
| Tool | Purpose | Link |
|------|---------|------|
| **YOLOv8** | Object detection | github.com/ultralytics/ultralytics |
| **SAM / samgeo** | Building segmentation | samgeo.gishub.org |
| **OpenDroneMap** | Drone mapping | opendronemap.org |
| **NetworkX/OSMnx** | Path planning | osmnx |
| **Streamlit/Gradio** | Web dashboard | streamlit.io |
| **FEMA HAZUS** | Loss estimation | fema.gov |
| **CesiumJS** | 3D globe | cesium.com |

---

## 11. SUGGESTED TECH STACK (Minimum Viable Project)

```
Language:        Python
Simulator:       AirSim (UE5) OR just use xBD dataset directly
CV Framework:   PyTorch + Ultralytics (YOLOv8)
Segmentation:   SAM-Geo
Mapping:        OpenDroneMap + Folium
Path Planning:  NetworkX + OSMnx
Dashboard:      Streamlit
Loss Estimation: Custom Python model based on xBD damage levels
```

### Quick Start Path:
1. Download xBD dataset (pre/post disaster satellite images)
2. Train damage classification model (U-Net or use pre-trained)
3. Download D-Fire dataset → train YOLOv8 fire detector
4. Use YOLOv8 (COCO pretrained) for person detection
5. Get OSM road network → implement A* path planning
6. Build Streamlit dashboard showing: damage map + detections + routes + loss
7. Optionally: use AirSim for drone simulation video

---

## 12. PAPERS TO CITE (Key References)

1. **xBD:** Gupta et al., "xBD: A Dataset for Assessing Building Damage from Satellite Imagery" (CVPR 2019) — 468 citations
2. **ChangeOS:** Zheng et al., "Building damage assessment for rapid disaster response" (2021) — 616 citations
3. **AiRobSim:** Chen et al., "Simulating a Multisensor Aerial Robot for Urban SAR" (Sensors 2020) — 34 citations
4. **USARSim:** Carpin et al., "USARSim: a robot simulator for research and education" (ICRA 2007) — 668 citations
5. **AirSim:** Microsoft Research — https://microsoft.github.io/AirSim/
6. **Survey of UAV Simulators:** Chen et al., "A Survey on Open-Source Simulation Platforms for Multi-Robot Systems" (2023) — 42 citations
7. **SAM for Remote Sensing:** Osco et al. (2023) — 540 citations
8. **Multi-UAV Cooperative Coverage:** Zeng et al. (2024) — 14 citations
9. **FLAME fire dataset:** Chen et al. (2022) — 257 citations
10. **D-Fire dataset:** Tleuliyev et al. — GitHub
