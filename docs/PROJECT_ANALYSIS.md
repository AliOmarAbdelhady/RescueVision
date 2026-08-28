# RescueVision — Complete Project Analysis

## Context
This document provides a comprehensive analysis of the RescueVision project, covering its purpose, architecture, pipeline, models, and an evaluation of its suitability as a CS graduation project. The analysis is based on a thorough exploration of the entire codebase (~8,500 lines of Python + ~560 lines of TypeScript).

---

## 1. What is RescueVision?

RescueVision is a **multimodal deep learning system for automated post-disaster damage assessment** using satellite and UAV (drone) imagery. It takes **before** and **after** disaster images and automatically:

1. Detects buildings in the area
2. Identifies which buildings have changed
3. Classifies the damage level of each building
4. Generates pixel-level damage maps
5. Produces emergency assessment reports with urgency scores
6. Supports emergency route planning avoiding blocked roads

**Real-world use case**: When an earthquake, hurricane, or flood hits a region, first responders need to quickly know which buildings are damaged and how severely. Manual assessment takes days. RescueVision automates this in seconds using satellite imagery.

---

## 2. Project Structure

```
RescueVision/
├── configs/                 # 11 YAML experiment configurations
├── data/                    # Raw and processed datasets (xBD, FloodNet)
├── notebooks/               # Kaggle-ready training notebooks (GPU)
├── scripts/                 # 9 CLI entry points
│   ├── train_building_seg.py
│   ├── train_damage_seg.py
│   ├── train_change_detection.py
│   ├── train_damage_classifier.py
│   ├── train_floodnet_segmentation.py
│   ├── run_inference.py
│   ├── export_report.py
│   ├── prepare_xbd.py
│   └── build_retrieval_index.py
├── src/rescuevision/        # Core Python package (~6,080 lines)
│   ├── data/                # Dataset parsing, preprocessing, transforms
│   ├── models/              # Model architectures
│   ├── training/            # Training loops, optimizers, callbacks
│   ├── evaluation/          # Segmentation & classification metrics
│   ├── inference/           # End-to-end prediction pipeline
│   ├── reporting/           # Report generation (Markdown, HTML, PDF)
│   ├── routing/             # Emergency route planning (A*/Dijkstra)
│   ├── retrieval/           # FAISS-based similar case search
│   └── utils/               # Config, I/O, image, geometry, seed
├── api/                     # FastAPI REST backend (~515 lines)
│   ├── main.py              # 7 API endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── services/            # Business logic layer
├── dashboard/               # React + TypeScript frontend (~560 lines)
│   └── src/components/      # 7 React components
├── tests/                   # Unit tests (~451 lines)
├── Dockerfile               # Multi-stage build (backend + dashboard)
├── docker-compose.yml       # Container orchestration
└── outputs/                 # Checkpoints, predictions, reports, FAISS index
```

### Code Statistics

| Module | Lines of Code | Files |
|--------|--------------|-------|
| Core Package (`src/`) | ~6,080 | ~60 Python files |
| API (`api/`) | ~515 | ~8 Python files |
| Scripts (`scripts/`) | ~871 | ~9 Python files |
| Tests (`tests/`) | ~451 | ~5 Python files |
| Dashboard (`dashboard/`) | ~560 | ~11 TS/TSX files |
| Configs (`configs/`) | — | 11 YAML files |
| **Total** | **~8,500+** | **~100+ files** |

---

## 3. The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER / DASHBOARD                             │
│         Uploads pre-disaster and post-disaster images           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (/predict)                         │
│         Saves images to temp, calls inference service           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           DisasterPredictor (Orchestrator)                      │
│                                                                 │
│  Step 1: Building Segmentation                                  │
│     Input:  Pre-disaster RGB image (3 channels, 1024x1024)     │
│     Model:  U-Net + ResNet34 encoder                           │
│     Output: Binary building mask                                │
│                                                                 │
│  Step 2: Change Detection                                       │
│     Input:  Pre + Post image pair                               │
│     Model:  Siamese U-Net + ResNet34 encoder                   │
│     Output: Binary change mask                                  │
│                                                                 │
│  Step 3: Damage Segmentation                                    │
│     Input:  Pre RGB + Post RGB concatenated (6 channels)       │
│     Model:  U-Net + EfficientNet-B3 encoder                    │
│     Output: 5-class pixel map (background, no-damage,          │
│              minor, major, destroyed)                           │
│                                                                 │
│  Step 4: Building-Level Classification                          │
│     Input:  Individual building crops (pre + post, 224x224)    │
│     Model:  DINOv2 + MLP classifier                            │
│     Output: 4-class damage label per building                   │
│                                                                 │
│  Step 5: Post-Processing                                        │
│     - Connected component analysis                              │
│     - Per-building damage map assembly                          │
│     - Urgency score computation                                 │
│     - Summary statistics (JSON)                                 │
│                                                                 │
│  Step 6: Visualization                                          │
│     - Color-coded damage overlay                                │
│     - Overview composite image                                  │
│                                                                 │
│  Step 7: Report Generation                                      │
│     - Markdown emergency report                                 │
│     - HTML report                                               │
│     - PDF export (reportlab + jinja2 templates)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT ARTIFACTS                              │
│                                                                 │
│  outputs/{case_id}/                                             │
│  ├── building_mask.png      # Binary building detection         │
│  ├── change_mask.png        # Change detection result           │
│  ├── damage_map.png         # Color-coded pixel damage          │
│  ├── overlay_damage.png     # Damage overlaid on post image     │
│  ├── overview.png           # Composite visualization           │
│  ├── emergency_summary.json # Statistics + urgency              │
│  ├── case_meta.json         # Full case metadata                │
│  ├── report.md              # Markdown report                   │
│  ├── report.html            # HTML report                       │
│  └── report.pdf             # PDF report for download           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Models Used and Why

### 4.1 U-Net (Building Segmentation + Damage Segmentation)

**Architecture**: Encoder-decoder with skip connections.

| Task | Encoder | Input Channels | Output Classes |
|------|---------|---------------|----------------|
| Building Segmentation | ResNet34 | 3 (RGB) | 1 (binary) |
| Damage Segmentation | EfficientNet-B3 | 6 (pre+post) | 5 (damage levels) |

**Why U-Net?**
- The gold standard for medical and satellite image segmentation
- Skip connections preserve fine-grained spatial detail
- ResNet34 encoder: fast training, good feature extraction for buildings
- EfficientNet-B3 encoder: better accuracy for the harder damage task, efficient parameter usage
- 6-channel input for damage seg: allows the model to directly compare pre/post imagery

### 4.2 Siamese U-Net (Change Detection)

**Architecture**: Two shared-weight encoders (one for pre, one for post) with feature fusion decoder.

- Encoder: ResNet34 (shared weights)
- Fusion modes: concatenation, absolute difference, or both combined
- Output: Binary change mask

**Why Siamese?**
- Shared encoder learns disaster-agnostic features
- Comparing feature representations detects meaningful changes
- More robust than simple pixel differencing (handles lighting, seasonal changes)

### 4.3 DINOv2 + MLP (Building-Level Classification)

**Architecture**: DINOv2 (self-supervised vision transformer) extracts CLS embeddings from pre/post crops, concatenated, then fed into MLP classifier head.

- Two-stage training:
  1. **Stage 1**: Frozen DINOv2 backbone, train MLP only
  2. **Stage 2**: Unfreeze last N transformer blocks, fine-tune end-to-end

**Why DINOv2?**
- State-of-the-art self-supervised features (no labeled pre-training needed for the backbone)
- Excellent transfer learning to domain-specific tasks
- CLS token captures global image context — ideal for building-level classification
- More powerful than CNN features for this type of comparison task

### 4.4 SegFormer (Flood/Road Segmentation — Optional)

**Architecture**: Transformer-based semantic segmentation (hierarchical).

- Used for FloodNet dataset: 10-class flood scene understanding
- Alternative architecture option for damage/building tasks

**Why SegFormer?**
- Lightweight transformer alternative to U-Net
- Better at capturing long-range context (useful for flood boundaries)
- Efficient design suitable for deployment

### 4.5 Alternative Models Available in Configs

The project also provides configs for:
- `ChangeFormer` — Transformer-based change detection
- `EfficientNet` classifier — CNN alternative to DINOv2

### Model Summary Table

| Model | Task | Framework | Parameters | Why This Choice |
|-------|------|-----------|------------|-----------------|
| U-Net + ResNet34 | Building Segmentation | PyTorch + SMP | ~24M | Fast, proven, good features |
| U-Net + EfficientNet-B3 | Damage Segmentation | PyTorch + SMP | ~12M | Higher accuracy, efficient |
| Siamese U-Net + ResNet34 | Change Detection | PyTorch (custom) | ~24M | Shared features, comparison |
| DINOv2 + MLP | Building Classification | PyTorch + Transformers | ~86M | SOTA features, two-stage fine-tuning |
| SegFormer | Flood Segmentation | PyTorch + Transformers | ~4-14M | Lightweight transformer |

---

## 5. Datasets

### 5.1 xBD/xView2 (Primary)

- **Source**: Defense Innovation Unit + NASA, distributed via Kaggle
- **Content**: Satellite imagery from 19 disasters worldwide (earthquakes, fires, floods, tsunamis, etc.)
- **Splits**: train (~9,000 images), hold, test
- **Labels**: Polygon building footprints + damage grades (4 classes)
- **Why this dataset?**: It is the standard benchmark for disaster damage assessment. Used in the xView2 Challenge. Well-labeled, diverse, and publicly available.

### 5.2 FloodNet (Optional)

- **Source**: UAV imagery from Hurricane Florence
- **Content**: 10-class semantic segmentation (roads, buildings, water, pools, vehicles, etc.)
- **Why?**: Adds flood-specific analysis capability

---

## 6. Training Details

### Loss Functions
- **Building/Change Segmentation**: BCEDiceLoss — combines binary cross-entropy + Dice coefficient for balanced segmentation
- **Damage Segmentation**: Weighted CE-DiceLoss — class weights handle the severe imbalance (most pixels are background or no-damage)
- **Classification**: FocalLoss (gamma=2.0) — specifically designed for class imbalance

### Optimizers & Schedules
- Adam optimizer with weight decay
- Cosine annealing learning rate schedule
- ReduceLROnPlateau fallback
- Gradient accumulation support
- Mixed precision training (AMP) for faster GPU utilization

### Data Augmentation
- Horizontal/vertical flips
- Random rotation
- Brightness/contrast jitter
- Normalization using ImageNet statistics

### Metrics
- **Segmentation**: IoU, Dice, Precision, Recall, F1, per-class IoU, Mean IoU
- **Classification**: Accuracy, per-class precision/recall/F1, confusion matrix, macro/micro averages

---

## 7. Key Software Components

### 7.1 Inference Orchestrator (`DisasterPredictor`)
- Loads all models from checkpoints
- Runs sequential pipeline with GPU acceleration
- Handles post-processing (connected components, damage map assembly)
- Generates urgency scoring (based on damage distribution)
- Produces visualizations and JSON summaries

### 7.2 Report Generation
- Template-based reports using Jinja2
- PDF export via reportlab
- Markdown and HTML formats
- Includes damage statistics, urgency level, and map visualization

### 7.3 Emergency Route Planning
- Converts road/flood masks to navigable graphs (NetworkX)
- A*/Dijkstra shortest-path routing
- Avoids flooded/blocked regions
- Pixel-to-geographic coordinate conversion

### 7.4 Retrieval System (FAISS)
- DINOv2-based building embeddings
- FAISS vector index for similarity search
- Retrieves similar historical disaster cases

### 7.5 FastAPI Backend
- 7 REST endpoints (health, predict, cases, reports, download)
- File upload handling for image pairs
- CORS middleware for frontend communication
- Docker containerized deployment

### 7.6 React Dashboard
- Upload panel (drag-and-drop)
- Damage summary cards (building counts, urgency)
- Image overlay viewer (multiple visualization layers)
- Report panel with preview and download
- Similar cases display
- Route map component

---

## 8. Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, segmentation_models_pytorch, Transformers (HuggingFace) |
| Vision Models | U-Net, Siamese U-Net, DINOv2, SegFormer, EfficientNet |
| Image Processing | OpenCV, Albumentations, PIL, scikit-image |
| Vector Search | FAISS |
| Graph Routing | NetworkX |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Reporting | Jinja2, reportlab, Markdown |
| Data | NumPy, Pandas, GeoPandas |
| Deployment | Docker, Docker Compose |
| Training | Mixed Precision (AMP), Kaggle GPU notebooks |
| Configuration | YAML-based experiment configs |

---

## 9. Can This Be a CS Graduation Project?

### Short Answer: YES — it is a strong graduation project.

### Why It Qualifies

**1. Technical Depth**
- Implements **5 different deep learning architectures** (U-Net, Siamese U-Net, DINOv2, SegFormer, EfficientNet)
- Covers **multiple ML paradigms**: binary segmentation, multiclass segmentation, change detection, transfer learning, self-supervised learning, metric learning
- Custom loss functions (FocalLoss, BCEDiceLoss, Weighted CE-Dice)
- Two-stage training with backbone unfreezing
- Mixed precision training

**2. Full-Stack Engineering**
- Complete ML pipeline: data, preprocessing, training, inference, deployment
- REST API backend (FastAPI) with proper schemas (Pydantic)
- Modern frontend (React + TypeScript + Tailwind)
- Docker containerization with multi-service orchestration
- FAISS vector retrieval system
- Automated PDF report generation

**3. Real-World Impact**
- Addresses a genuine humanitarian problem (disaster response)
- Uses real-world datasets from actual disasters
- Produces actionable outputs (damage maps, urgency scores, emergency reports)

**4. Research Novelty**
- Multimodal pipeline combining 4 complementary models
- Siamese architecture for change detection
- DINOv2 (SOTA) for building classification
- End-to-end orchestrator with post-processing

**5. Code Quality**
- ~8,500+ lines of well-structured, modular code
- Proper package structure (`src/rescuevision/`)
- Configuration-driven experiments (YAML)
- Unit tests
- Documentation and Kaggle-ready notebooks

**6. Completeness**
- Data preparation, Training, Inference, Reporting, Deployment, Frontend
- This is a complete product, not a prototype or notebook

### What Makes It Stand Out for a Graduation Project

| Criterion | Assessment |
|-----------|-----------|
| ML/AI Complexity | **Excellent** — 5 architectures, custom losses, multi-task pipeline |
| Software Engineering | **Very Good** — modular, tested, Dockerized, API + Frontend |
| Real-World Relevance | **Excellent** — humanitarian use case with real disaster data |
| Research Contribution | **Good** — strong applied ML, combines SOTA models innovatively |
| Documentation | **Good** — README, configs, code comments |
| Novelty | **Good** — the pipeline orchestration and DINOv2 integration |

### What Could Be Improved to Make It Even Stronger

If you want to maximize the graduation project grade or aim for distinction:

**1. Add an Ablation Study / Comparative Analysis**
- Train and compare multiple architectures for each task
- Create a table showing U-Net vs FPN vs DeepLabV3+ vs SegFormer performance
- This demonstrates scientific rigor and understanding of model selection

**2. Add a Thesis-Worthy Document**
- Write a proper thesis/report (40-60 pages) covering:
  - Literature review of disaster assessment methods
  - Detailed methodology and justification for each model choice
  - Experimental results with tables and graphs
  - Limitations and future work

**3. Implement a Novel Contribution**
- Examples:
  - A custom attention mechanism for damage segmentation
  - Temporal analysis (tracking damage over multiple time steps)
  - Uncertainty quantification (how confident is the model?)
  - A new fusion strategy for combining the 4 models' outputs

**4. Add More Evaluation**
- Cross-validation results
- Comparison with published baselines on xBD
- Error analysis: when does the model fail and why?
- Visual examples of failure cases

**5. Real-Time or Near-Real-Time Processing**
- Optimize inference speed (ONNX export, TensorRT)
- Benchmark latency for the full pipeline
- Add a progress bar or streaming response for the dashboard

**6. Add Explainability**
- Grad-CAM or saliency maps showing what the model focuses on
- This is increasingly expected in ML graduation projects

**7. Testing and CI/CD**
- Expand test coverage (currently 5 test files)
- Add integration tests for the full pipeline
- Set up GitHub Actions for automated testing

---

## 10. Strengths and Weaknesses Summary

### Strengths
- Well-architected modular codebase
- Multiple SOTA model architectures
- Complete end-to-end pipeline
- Real-world humanitarian application
- Docker-ready deployment
- Professional report generation
- FAISS retrieval system
- Route planning capability
- Clean separation of concerns

### Weaknesses / Areas for Growth
- No CI/CD pipeline
- Limited test coverage (5 test files)
- No quantitative results/benchmarks in the repository (need to train models first)
- No real-time video/camera integration
- No user authentication or multi-user support
- No model versioning or experiment tracking (e.g., MLflow)
- Safety disclaimer indicates it is still a research prototype

---

## 11. Final Verdict

**This is absolutely a viable CS graduation project.** It demonstrates:

- Deep understanding of computer vision and deep learning
- Ability to build a complete software system (not just a notebook)
- Awareness of real-world constraints and deployment
- Integration of multiple modern AI techniques into a cohesive product

The project sits comfortably above the average graduation project in both scope and technical depth. With the additions suggested in Section 9 (especially an ablation study, thesis document, and expanded evaluation), it could earn a distinction-level grade.
