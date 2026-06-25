# Tostal Sci-data Platform Architecture Plan

Cloud-hosted geoscience data platform with per-customer Icechunk stores, Jupyter notebook frontend, and Murmurative task models for classification, kriging, and segmentation.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     gateway.{domain}.com                      │
│                    NextJS + BetterAuth                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ OAuth2      │  │ Stripe Check  │  │ Proxy to API      │  │
│  │ Provider    │  │ Middleware    │  │ Backend           │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└──────────────────────────┬───────────────────────────────────┘
                           │ JWT-authenticated
┌──────────────────────────┴───────────────────────────────────┐
│                     api.{domain}.com                          │
│                   FastAPI (OpenAPI 3.1)                       │
│                                                               │
│  /v1/ingest      /v1/classifiers  /v1/krigging  /v1/segmentor│
│  ┌──────────┐    ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │File      │    │Task: facies  │  │Spatial   │  │Task:    │ │
│  │Classifier│    │→ Murmurative │  │interp    │  │litho    │ │
│  │(rule)    │    │  classifier  │  │→ kriging │  │→ segm.  │ │
│  └────┬─────┘    └──────┬───────┘  └────┬─────┘  └────┬────┘ │
│       │                 │               │              │      │
│  ┌────┴─────────────────┴───────────────┴──────────────┴───┐ │
│  │              Icechunk Storage Manager                    │ │
│  │  customer-1 → container-1                                │ │
│  │  customer-2 → container-2                                │ │
│  └──────────────────────────┬──────────────────────────────┘ │
└─────────────────────────────┼────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Neon Postgres│  │ Azure Blob       │  │ Temporal Cloud   │
│  (serverless) │  │ Storage          │  │ (job orch.)      │
│               │  │                  │  │                  │
│ customers     │  │ container-c1/    │  │ ClassifyWorkflow │
│ files         │  │ container-c2/    │  │ KrigeWorkflow    │
│ jobs          │  │ container-c3/    │  │ SegmentWorkflow  │
│ notebooks     │  │ container-models/│  │                  │
│ subscriptions │  │                  │  │                  │
└───────────────┘  └──────────────────┘  └──────────────────┘
                              │
┌─────────────────────────────┴────────────────────────────────┐
│                    app.{domain}.com                           │
│               Custom Jupyter Notebook UI                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  [Cell] /ingest ▸ file picker → upload → render array  │  │
│  │  [Cell] /classify --task facies-map ▸ .las file → map  │  │
│  │  [Cell] /krig ▸ coordinates → interpolated grid        │  │
│  │  [Cell] /segment --task litho ▸ photo → overlay map    │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  jupyter-data-cubes widget (3D array viewer)           │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Auth Gateway** | NextJS 14 + BetterAuth | OAuth2 out of the box, server-side Stripe checks, edge middleware |
| **API Backend** | FastAPI (Python 3.12) | OpenAPI auto-generation, async, Pydantic validation, native Xarray/Icechunk |
| **Notebook Frontend** | datalayer/jupyter-ui + custom ipywidgets | React-based Jupyter UI, extensible widget system |
| **Array Viewer** | jupyter-data-cubes (custom widget) | Three.js/Deck.gl 3D rendering of Zarr arrays in notebook |
| **Storage** | Icechunk on Azure Blob Storage | Cloud-native Zarr, transactional, per-container isolation |
| **Database** | Neon Postgres (serverless) + SQLAlchemy + asyncpg | UUID PKs, metadata only (Icechunk URI refs), built-in query viewer |
| **Data Processing** | Xarray + Dask | Universal format conversion, lazy chunked I/O |
| **ML Models** | PyTorch + Murmurative CUDA ops | Task-specific classifiers/segmenters/krigers |
| **Containerization** | Docker + Azure Container Apps (T4 GPU) | Per-service isolation, scale-to-zero, T4 GPU for model inference |
| **Job Orchestration** | Temporal Cloud | Durable async workflows for classify/krige/segment jobs, retries, observability |
| **Payments** | Stripe API (`price_1TlexE8k0ubC0hdJrGUVKQI9`) | Subscription gating |
| **Infrastructure** | Terraform (Azure provider) | Reproducible container provisioning |

## Auth Flow

```
1. User visits app.{domain}.com
2. Redirected to gateway.{domain}.com/login
3. BetterAuth OAuth2 flow (Google, GitHub, email)
4. Gateway checks Stripe subscription status:
   - Active subscription → issue JWT with customer_id, allowed routes
   - No subscription → redirect to Stripe checkout (price_1TlexE...)
5. JWT attached to all API requests as Bearer token
6. API validates JWT, extracts customer_id, resolves Icechunk container
7. Gateway middleware caches subscription status (5 min TTL)
```

## Storage Design

```
Azure Blob Storage Account: murmurativeprod
│
├── container-customer-{uuid}/
│   └── icechunk store root
│       ├── geology/
│       │   ├── segy/
│       │   │   ├── survey_2024/       ← ingested .sgy converted to Zarr
│       │   │   └── survey_2023/
│       │   ├── las/
│       │   │   ├── well_b17/           ← ingested .las converted to Zarr
│       │   │   └── well_c03/
│       │   ├── dlis/
│       │   ├── raster/
│       │   └── results/
│       │       ├── facies_map_2024.zarr
│       │       └── krige_grid_2024.zarr
│       ├── materials/
│       │   └── hdf5/
│       ├── climate/
│       │   ├── netcdf/
│       │   └── grib/
│       ├── bioimaging/
│       │   └── raster/
│       └── metadata/
│           └── {dataset}.json
│
├── container-customer-{uuid}/
│   └── ...
│
└── container-models/                   ← shared, read-only
    ├── facies-classifier/
    │   └── model.pt                    ← Murmurative facies model weights
    ├── litho-segmentor/
    │   └── model.pt
    └── neural-kriging/
        └── model.pt
```

- **One Azure Blob Storage container per customer** for maximum isolation and per-customer storage billing.
- **Shared read-only container** for ML model weights.
- All data stored as Zarr arrays managed by Icechunk for transactional writes, versioning, and parallel I/O.

## Database Design (Neon Postgres)

Serverless PostgreSQL storing metadata only — customer records, file/index references, job tracking, notebook metadata, and Stripe webhook events. All array data lives in per-customer Icechunk stores on Azure Blob Storage. Neon's built-in SQL Editor provides zero-config query access for debugging and customer support.

### Enum Types

| Type | Values |
|---|---|
| `job_type` | `classify`, `krige`, `segment` |
| `job_status` | `pending`, `processing`, `completed`, `failed`, `canceled` |
| `subscription_status` | `active`, `past_due`, `canceled`, `incomplete`, `incomplete_expired`, `trialing`, `unpaid` |

### Table Schemas

#### 1. customers

```sql
CREATE TABLE customers (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stripe_customer_id    TEXT UNIQUE,
    email                 TEXT NOT NULL,
    name                  TEXT,
    azure_container_name  TEXT NOT NULL UNIQUE,
    icechunk_store_uri    TEXT NOT NULL,
    subscription_status   subscription_status NOT NULL DEFAULT 'incomplete',
    current_period_end    TIMESTAMPTZ,
    storage_quota_bytes   BIGINT DEFAULT 107374182400,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

#### 2. files

```sql
CREATE TABLE files (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id       UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    display_id        TEXT NOT NULL UNIQUE,
    filename          TEXT NOT NULL,
    file_format       TEXT NOT NULL,
    domain            TEXT NOT NULL,
    subdirectory      TEXT NOT NULL,
    icechunk_uri      TEXT NOT NULL,
    shape             JSONB,
    dtype             TEXT,
    chunk_size        JSONB,
    size_bytes        BIGINT,
    description       TEXT,
    tags              JSONB,
    ingestion_status  job_status NOT NULL DEFAULT 'pending',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_files_customer ON files(customer_id);
CREATE INDEX idx_files_display ON files(display_id);
```

#### 3. notebooks

```sql
CREATE TABLE notebooks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    display_id          TEXT NOT NULL UNIQUE,
    name                TEXT NOT NULL,
    description         TEXT,
    icechunk_state_uri  TEXT,
    cell_count          INT DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'draft',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_notebooks_customer ON notebooks(customer_id);
```

#### 4. jobs

```sql
CREATE TABLE jobs (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id          UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    notebook_id          UUID REFERENCES notebooks(id) ON DELETE SET NULL,
    display_id           TEXT NOT NULL UNIQUE,
    job_type             job_type NOT NULL,
    task                 TEXT NOT NULL,
    temporal_workflow_id TEXT,
    status               job_status NOT NULL DEFAULT 'pending',
    progress             SMALLINT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    input_file_ids       JSONB NOT NULL DEFAULT '[]',
    parameters           JSONB,
    result_icechunk_uri  TEXT,
    result_shape         JSONB,
    classes              JSONB,
    confidence           REAL CHECK (confidence >= 0 AND confidence <= 1),
    error_message        TEXT,
    estimated_completion TIMESTAMPTZ,
    started_at           TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_jobs_customer ON jobs(customer_id);
CREATE INDEX idx_jobs_notebook ON jobs(notebook_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(job_type);
```

#### 5. subscription_events

```sql
CREATE TABLE subscription_events (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID REFERENCES customers(id) ON DELETE SET NULL,
    stripe_event_id         TEXT NOT NULL UNIQUE,
    stripe_event_type       TEXT NOT NULL,
    stripe_customer_id      TEXT,
    stripe_subscription_id  TEXT,
    raw_payload             JSONB NOT NULL,
    processed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sub_events_customer ON subscription_events(customer_id);
CREATE INDEX idx_sub_events_stripe_customer ON subscription_events(stripe_customer_id);
CREATE INDEX idx_sub_events_type ON subscription_events(stripe_event_type);
```

### Entity Relationships

```
customers 1 ──── * files               (customer_id FK)
customers 1 ──── * notebooks           (customer_id FK)
customers 1 ──── * jobs                (customer_id FK)
customers 1 ──── * subscription_events (customer_id FK, nullable for pre-provisioning events)
notebooks 1 ──── * jobs                (notebook_id FK, nullable — jobs may be ad-hoc via API)
```

- **files → jobs**: `jobs.input_file_ids` is a JSONB array of file UUIDs. Application-layer integrity — PostgreSQL doesn't support array foreign keys natively.
- **Notebook state** (cells, outputs, widget state) is stored as JSON in `notebooks/{id}/state.json` inside the customer's Icechunk container. The `notebooks.icechunk_state_uri` column stores the path reference.
- **Stripe processing**: `subscription_events.raw_payload` stores the full webhook body for audit/replay. `customer_id` is nullable because webhook events may arrive before the corresponding customer row is provisioned. Application logic upserts `customers.subscription_status` and `customers.current_period_end` from processed events.
- **Display IDs**: `files.display_id` / `jobs.display_id` / `notebooks.display_id` are application-generated human-readable IDs (e.g. `ing_abc123`, `cls_def456`, `nb_abc123`). Actual entity relationships use UUIDs.

## API Design (OpenAPI 3.1)

### POST /v1/ingest

Upload any array data file; auto-classifies format and routes to correct domain folder.

```
Request:
  multipart/form-data
  file: binary (.sgy, .las, .dlis, .h5, .nc, .tif, .json, .zarr)
  metadata: {name, description, tags} (optional)

Response:
  {
    "id": "ing_abc123",
    "path": "/geology/segy/survey_2024.sgy",
    "format": "segy",
    "domain": "geology",
    "shape": [1024, 512, 256],
    "dtype": "float32",
    "chunk_size": [128, 256, 256],
    "icechunk_uri": "icechunk://azure://acct/customer-xyz/geology/segy/survey_2024",
    "size_bytes": 536870912,
    "created_at": "2026-06-23T..."
  }
```

**File auto-classification rules (rule-based, extension + magic bytes):**

| Extension(s) | Domain | Subdirectory |
|---|---|---|
| `.sgy`, `.segy` | geology | `segy/` |
| `.las` | geology | `las/` |
| `.dlis` | geology | `dlis/` |
| `.h5`, `.hdf5` | materials | `hdf5/` |
| `.nc`, `.nc4` | climate | `netcdf/` |
| `.grib`, `.grb` | climate | `grib/` |
| `.tif`, `.tiff` | geology | `raster/` |
| `.jpg`, `.jpeg`, `.png` | bioimaging | `raster/` |
| `.zarr` | auto | write directly |
| `.json` | metadata | `/` |
| all others | other | `/` |

### POST /v1/classifiers

Run classification tasks on ingested files.

```
Request:
  {
    "task": "facies-map",               // "facies-map" | "lithology" | "fault-detect"
    "file_ids": ["ing_abc123"],         // previously ingested .las or .sgy files
    "parameters": {
      "depth_range": [1200, 3500],
      "output_resolution": 0.5
    }
  }

Response:
  {
    "job_id": "cls_def456",
    "status": "processing",
    "estimated_completion": "2026-06-23T..."
  }

GET /v1/classifiers/{job_id}/result
  {
    "job_id": "cls_def456",
    "status": "completed",
    "result_path": "/geology/results/facies_map_2024.zarr",
    "result_shape": [4600, 1],
    "classes": ["sandstone", "shale", "limestone", "dolomite"],
    "confidence": 0.89,
    "rendering_url": "/v1/render/cls_def456"
  }
```

### POST /v1/krigging

Spatial interpolation from scattered observations to a regular grid.

```
Request:
  {
    "observations": {
      "file_ids": ["ing_abc123", "ing_xyz789"],  // .las files with spatial coordinates
      "variables": ["porosity", "permeability"]
    },
    "grid": {
      "x_range": [542000, 546000],
      "y_range": [6420000, 6424000],
      "z_range": [1800, 2200],
      "resolution": [50, 50, 1]
    },
    "method": "murmurative",        // "murmurative" | "ordinary" | "universal"
    "variogram_model": "auto"       // "auto" | "exponential" | "spherical" | "matern"
  }

Response:
  {
    "job_id": "krg_ghi789",
    "status": "processing",
    "result_path": "/geology/results/krige_grid_2024.zarr",
    "result_shape": [80, 80, 400]
  }
```

### POST /v1/segmentor

Image segmentation for lithology mapping from photos.

```
Request:
  {
    "task": "litho",                   // "litho" | "outcrop" | "core"
    "file_ids": ["ing_photo001"],      // ingested image files
    "parameters": {
      "classes": ["sandstone", "shale", "conglomerate", "fault_gouge"],
      "output_resolution": "full"
    }
  }

Response:
  {
    "job_id": "seg_jkl012",
    "status": "processing",
    "result_path": "/geology/results/litho_seg_2024.zarr",
    "result_shape": [4096, 6144],
    "classes": ["sandstone", "shale", "conglomerate", "fault_gouge"],
    "rendering_url": "/v1/render/seg_jkl012"
  }
```

### GET /v1/render/{job_id}

Streams Zarr chunks for progressive rendering in the notebook array viewer.

## Frontend Design

### Notebook UI

Customized datalayer/jupyter-ui with restricted command interface. The notebook supports `/` commands typed in cells:

| Command | Behavior |
|---|---|
| `/ingest` | Opens file picker widget → uploads to `/v1/ingest` → renders uploaded array in jupyter-data-cubes |
| `/classify --task {task}` | Shows task dropdown + file selector → calls `/v1/classifiers` → renders classification map |
| `/krig` | Shows coordinate grid config + variable selector → calls `/v1/krigging` → renders interpolated grid |
| `/segment --task {task}` | Shows image file picker → calls `/v1/segmentor` → renders segmentation overlay |
| `/view {file_id}` | Loads Zarr array from Icechunk → renders in jupyter-data-cubes widget |
| `/export {file_id} --format {fmt}` | Downloads result in specified format (.zarr, .nc, .tif, .csv) |

### Notebook Cell State Machine

```
[Empty cell] → type / → [Command suggestions dropdown]
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        /ingest      /classify      /krig
           │             │             │
    ┌──────┴──────┐ ┌───┴────┐  ┌────┴─────┐
    │File picker  │ │Task    │  │Grid      │
    │↓            │ │picker  │  │config    │
    │Upload       │ │↓       │  │↓         │
    │↓            │ │Run     │  │Run       │
    │Render array │ │↓       │  │↓         │
    └─────────────┘ │Render  │  │Render    │
                    │map     │  │grid      │
                    └────────┘  └──────────┘
```

### jupyter-data-cubes Widget

Custom ipywidgets-based 3D array viewer, inspired by [napari](https://napari.org/) and the [xarray-napari collaboration](https://xarray.dev/blog/xarray-napari-plan):

- Renders Zarr chunks progressively via lazy loading
- **Named dimension display**: reads xarray dimension labels from Icechunk/Zarr metadata (e.g. `time × depth × inline × crossline`) instead of raw axis indices
- **Spatial alignment between layers**: segmentation masks, kriged grids, and overlays are positioned using Zarr coordinate arrays so layers with different shapes or resolutions stay spatially registered
- Scientific colormaps (viridis, RdBu, seismic, etc.)
- Overlay toggles for segmentation masks on source images (aligned via physical coordinates, not pixel indices)
- Built on Three.js / Deck.gl for WebGL rendering
- Communicates with backend via Jupyter comms to fetch Zarr chunks lazily
- Handles up to 4D arrays (e.g., time × depth × inline × crossline)

### Notebook Persistence & Project Model

Notebook cells and outputs persist across sessions. Each customer can create multiple notebooks/projects. Notebook metadata (name, description, timestamps, Icechunk URI) is stored in the `notebooks` table in Neon Postgres. Notebook state (cells, outputs, widget state) is stored as JSON in the customer's Icechunk container under a `notebooks/` prefix.

**Notebook schema (stored as JSON in Icechunk under `notebooks/{notebook_id}/state.json`):**

```
notebooks/
├── {notebook_id}/
│   ├── state.json         ← cells, outputs, widget state, metadata
│   └── outputs/           ← rendered Zarr outputs linked to cells
│       ├── cell_001.zarr
│       └── cell_003.zarr
├── {notebook_id}/
│   └── ...
```

**`state.json` schema:**

```json
{
  "notebook_id": "nb_abc123",
  "name": "B-17 Basin Analysis",
  "description": "Facies mapping and kriging for the B-17 well field",
  "created_at": "2026-06-23T...",
  "updated_at": "2026-06-23T...",
  "cells": [
    {
      "cell_id": "cell_001",
      "type": "ingest",
      "command": "/ingest",
      "status": "completed",
      "result": {
        "file_id": "ing_xyz",
        "icechunk_uri": "icechunk://azure://...",
        "shape": [1024, 512, 256]
      },
      "widget_state": {"colormap": "seismic", "dim_display": ["inline", "crossline", "depth"], "current_slice": {"depth": 128}},
      "output_ref": "outputs/cell_001.zarr"
    },
    {
      "cell_id": "cell_002",
      "type": "classify",
      "command": "/classify --task facies-map",
      "status": "completed",
      "input_files": ["cell_001"],
      "result": {
        "job_id": "cls_def456",
        "classes": ["sandstone", "shale", "limestone"]
      },
      "widget_state": {"overlay_opacity": 0.7, "spatial_alignment": {"source_cell": "cell_001"}},
      "output_ref": "outputs/cell_002.zarr"
    }
  ],
  "cell_order": ["cell_001", "cell_002"],
  "variables": {}  // shared namespace for cross-cell references
}
```

**Frontend project/notebook UI:**

```
┌─────────────────────────────────────────────────────────┐
│  Projects                              [+ New Project]  │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 📓 B-17 Basin Analysis           Updated 2h ago    ││
│  │ 📓 C-03 Well Correlation          Updated 1d ago    ││
│  │ 📓 Regional Seismic Survey        Updated 3d ago    ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Current: B-17 Basin Analysis                       ││
│  │                                                     ││
│  │  [Cell 1] /ingest ▸ survey_2024.sgy ✓              ││
│  │  ┌──────────────────────────────────────┐          ││
│  │  │         (rendered array)             │          ││
│  │  └──────────────────────────────────────┘          ││
│  │                                                     ││
│  │  [Cell 2] /classify --task facies-map ✓            ││
│  │  ┌──────────────────────────────────────┐          ││
│  │  │       (facies classification map)    │          ││
│  │  └──────────────────────────────────────┘          ││
│  │                                                     ││
│  │  [Cell 3] /krig                             [+ Cell]││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Murmurative Task Models

Three task-specific models, each trained offline and deployed to the shared container-models store. Training compute costs are negligible due to small model sizes and modest publicly available labeled data.

### Facies Classifier

| Parameter | Value |
|---|---|
| Architecture | Murmurative sequence classifier |
| Parameters | ~30M |
| Input | Well log curves (.las): GR, resistivity, density, sonic, neutron porosity |
| Output | Lithology/facies class per depth point |
| Training data | Public datasets: FORCE 2020, Taranaki Basin, Volve |
| Sequence length | 5K–50K depth points per well |
| Murmurative config | M=256 slots, R=3 rounds, k=7, chunk_size=256 |
| Training cost | ~$1 compute (15 min on 1× H100) |

### Lithology Segmenter

| Parameter | Value |
|---|---|
| Architecture | Hybrid: convolutional encoder → Murmurative attention bottleneck → convolutional decoder |
| Parameters | ~50M |
| Input | Outcrop photos, core photos, drone imagery (.jpg/.png/.tif) |
| Output | Pixel-level lithology class map |
| Training data | Public outcrop datasets, core photo collections |
| Murmurative config | M=256 slots, R=3 rounds, k=7; conv encoder extracts local features, Murmurative attends globally over feature map patches, conv decoder upsamples to pixel resolution |
| Training cost | ~$5 compute (20 min on 4× H100) |

### Neural Kriging Model

| Parameter | Value |
|---|---|
| Architecture | Murmurative neural kriging — slot-based attention over scattered spatial observations |
| Parameters | ~20M |
| Input | Scattered observations with (x, y, z) coordinates + measured values |
| Output | Interpolated values on a regular grid |
| Training | Self-supervised: randomly mask observations, predict masked values |
| Training data | Any geoscience dataset with spatial coordinates (no labels needed) |
| Murmurative advantage | O(1) memory per query regardless of observation count; slots aggregate nearby measurements naturally |
| Training cost | ~$8 compute (30 min on 4× H100) |

### Total Task Model Training Budget

| Item | Cost |
|---|---|
| Facies classifier | ~$1 |
| Lithology segmenter | ~$5 |
| Neural kriging | ~$8 |
| Experimentation + hyperparameter tuning (5×) | ~$70 |
| **Total** | **~$84** |

Training data is publicly available and free. The bottleneck is model architecture development, not GPU hours.

## Directory Structure

```
murmurative-platform/
├── gateway/                          # NextJS OAuth gateway
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx             # Landing / status page
│   │   │   ├── login/
│   │   │   ├── api/
│   │   │   │   └── auth/
│   │   │   │       └── [...all]     # BetterAuth route handler
│   │   │   └── subscribe/           # Stripe checkout redirect
│   │   ├── lib/
│   │   │   ├── auth.ts              # BetterAuth config (OAuth providers)
│   │   │   ├── stripe.ts            # Stripe client + subscription check
│   │   │   └── api-proxy.ts         # Proxy requests to FastAPI backend
│   │   ├── middleware.ts            # Auth + subscription gating
│   │   └── types.ts
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   └── Dockerfile
│
├── api/                              # FastAPI backend
│   ├── app/
│   │   ├── main.py                   # FastAPI app, CORS, middleware
│   │   ├── config.py                 # Settings from env vars
│   │   ├── routes/
│   │   │   ├── ingest.py             # POST /v1/ingest
│   │   │   ├── classifiers.py        # POST/GET /v1/classifiers
│   │   │   ├── krigging.py           # POST/GET /v1/krigging
│   │   │   ├── segmentor.py          # POST/GET /v1/segmentor
│   │   │   ├── render.py             # GET /v1/render/{job_id}
│   │   │   └── notebooks.py          # GET/POST/PUT/DELETE /v1/notebooks
│   │   ├── services/
│   │   │   ├── storage.py            # Icechunk store management per customer
│   │   │   ├── db.py                 # SQLAlchemy engine, session, asyncpg config
│   │   │   ├── file_classifier.py    # Rule-based format → domain/path routing
│   │   │   ├── converters.py         # Xarray-based format converters
│   │   │   ├── classifier_model.py   # Murmurative classifier inference
│   │   │   ├── kriging_model.py      # Neural kriging inference
│   │   │   ├── segmentor_model.py    # Murmurative segmenter inference
│   │   │   ├── jobs.py               # Temporal workflow client (classify/krige/segment)
│   │   │   ├── notebook_store.py     # Notebook state CRUD via Icechunk
│   │   │   └── customer.py           # Customer provisioning + Stripe webhooks
│   │   ├── models/
│   │   │   └── schemas.py            # Pydantic models → OpenAPI schema
│   │   └── middleware/
│   │       └── auth.py               # JWT validation + customer extraction
│   ├── tests/
│   │   ├── test_ingest.py
│   │   ├── test_classifiers.py
│   │   ├── test_krigging.py
│   │   ├── test_segmentor.py
│   │   └── test_db.py
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/
│   ├── alembic.ini
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                         # Jupyter Notebook UI
│   ├── packages/
│   │   ├── jupyter-ui/               # datalayer/jupyter-ui (forked)
│   │   │   ├── src/
│   │   │   ├── cells/
│   │   │   │   │   ├── CommandCell.tsx    # / command handler + suggestions
│   │   │   │   │   ├── IngestCell.tsx     # File upload + array render
│   │   │   │   │   ├── ClassifyCell.tsx   # Task selector + file → result
│   │   │   │   │   ├── KrigCell.tsx       # Grid config → result
│   │   │   │   │   └── SegmentCell.tsx    # Photo → overlay
│   │   │   │   ├── projects/         # Multi-notebook project management
│   │   │   │   │   ├── ProjectList.tsx    # Project sidebar with list
│   │   │   │   │   └── ProjectEditor.tsx  # Notebook container + cell renderer
│   │   │   │   ├── widgets/          # Widget bindings for ipywidgets
│   │   │   │   ├── api/              # OpenAPI-generated API client
│   │   │   │   └── auth/             # OAuth flow + token management
│   │   │   └── package.json
│   │   └── jupyter-data-cubes/       # Custom ipywidgets array viewer
│   │       ├── src/
│   │       │   ├── widget.ts         # Widget model (Python side)
│   │       │   ├── renderer.ts       # WebGL renderer (TypeScript)
│   │       │   ├── chunk_loader.ts   # Lazy Zarr chunk fetcher
│   │       │   └── colormaps.ts      # Scientific colormap definitions
│   │       ├── jupyter-data-cubes/
│   │       │   └── __init__.py       # Python widget registration
│   │       └── package.json
│   ├── jupyter_config.py             # Jupyter server config
│   └── Dockerfile
│
├── models/                           # ML model training (offline)
│   ├── facies-classifier/
│   │   ├── train.py
│   │   ├── model.py                  # Murmurative sequence classifier
│   │   ├── data.py                   # LAS → training examples
│   │   └── config.yaml
│   ├── litho-segmentor/
│   │   ├── train.py
│   │   ├── model.py                  # Murmurative spatial segmenter
│   │   ├── data.py
│   │   └── config.yaml
│   └── neural-kriging/
│       ├── train.py
│       ├── model.py                  # Murmurative neural kriging
│       ├── data.py
│       └── config.yaml
│
├── workflows/                        # Temporal workflows + activities
│   ├── worker.py                     # Temporal worker entrypoint
│   ├── workflows/
│   │   ├── classify.py               # Classification workflow (ingest → classify → store)
│   │   ├── krige.py                  # Kriging workflow (load observations → interpolate → store)
│   │   └── segment.py                # Segmentation workflow (load image → segment → store)
│   └── activities/
│       ├── model_inference.py        # GPU model inference activity (T4)
│       ├── file_convert.py           # Xarray format conversion activity
│       └── storage_ops.py            # Icechunk read/write activity
│
├── storage/                          # Infrastructure as code
│   ├── terraform/
│   │   ├── main.tf                   # Azure provider, resource groups
│   │   ├── storage.tf                # Blob Storage accounts, containers
│   │   ├── containerapps.tf          # Azure Container Apps
│   │   └── variables.tf
│   └── scripts/
│       ├── init-customer.sh          # Provision container + Icechunk store
│       └── seed-data.py              # Seed test data for development
│
├── docker-compose.yml                # Local development
├── Makefile
├── .env.example
└── README.md
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1–3)

- Terraform Azure infrastructure: Blob Storage, Container Apps (T4 GPU), DNS
- Provision Neon Postgres project + branch; run initial Alembic migration (customers table)
- NextJS gateway: BetterAuth OAuth (Google + GitHub), Stripe subscription middleware
- FastAPI skeleton: health check, auth middleware, SQLAlchemy + asyncpg setup, OpenAPI docs at `/docs`
- Icechunk storage manager: provision per-customer containers
- Customer provisioning script (creates Postgres row + Azure storage container)
- Set up Temporal Cloud namespace + deploy worker skeleton

### Phase 2: Ingest Pipeline (Weeks 3–5)

- `POST /v1/ingest`: file upload → rule-based classifier → converter → Icechunk write
- Xarray-based format converters: SEG-Y, LAS, DLIS, NetCDF, HDF5, GeoTIFF, JSON
- File classifier: extension + magic bytes → domain/path routing table
- `/v1/render/{file_id}`: basic Zarr array metadata + chunk preview
- Temporal workflow: file ingest + convert activity (retry logic for large files)
- Write file metadata to `files` table (display_id, icechunk_uri, shape, dtype, size_bytes)
- Alembic migration: `files` table

### Phase 3: Notebook Frontend (Weeks 5–8)

- Fork and customize datalayer/jupyter-ui
- Build CommandCell: `/` command parsing with suggestions dropdown
- Build IngestCell: file picker → upload progress → array render
- Build jupyter-data-cubes widget: WebGL 3D viewer with lazy chunk loading
- Build project/notebook management UI: project list, notebook CRUD, cell persistence
- Build `/v1/notebooks` endpoints: create, read, update, delete notebooks + cells
- Implement notebook state persistence to Icechunk (`notebooks/{id}/state.json`)
- Notebook metadata stored in `notebooks` table (name, description, icechunk_state_uri)
- Alembic migration: `notebooks` table
- OAuth integration: gateway redirect → token storage → authenticated API calls
- Stripe subscription gate on frontend (block cells, show upgrade prompt)

### Phase 4: ML Models (Weeks 6–9, parallel with Phase 3)

- Train facies classifier (Murmurative, ~30M params)
- Train lithology segmenter (Murmurative, ~50M params)
- Train neural kriging model (Murmurative, ~20M params)
- Export trained weights to container-models Icechunk store
- Model inference service: load model on demand, run prediction, save Zarr result

### Phase 5: Task Endpoints + Task Cells (Weeks 9–11)

- Build Temporal workflows: ClassifyWorkflow, KrigeWorkflow, SegmentWorkflow
- Build Temporal activities: model inference (T4 GPU), file conversion, storage ops
- `POST /v1/classifiers`: API handler → write job to `jobs` table → Temporal workflow start → return job_id
- `POST /v1/krigging`: API handler → write job to `jobs` table → Temporal workflow start → return job_id
- `POST /v1/segmentor`: API handler → write job to `jobs` table → Temporal workflow start → return job_id
- Temporal workflow stores result Zarr to customer container, updates job record with result_icechunk_uri + status
- Alembic migration: `jobs` table, `subscription_events` table
- ClassifyCell, KrigCell, SegmentCell in frontend with async polling for job completion
- Progressive result rendering in jupyter-data-cubes powered by Temporal signals

### Phase 6: Polish + Production Deploy (Weeks 11–12)

- End-to-end integration testing across all endpoints and cell types
- Rate limiting, file size limits (configurable per subscription tier), job queue limits
- Error handling with user-facing messages, retry logic for transient failures
- OpenAPI documentation with request/response examples
- Stripe webhook handler: subscription created/updated/canceled → upsert `customers` row + log `subscription_events`
- Production deployment to Azure Container Apps

## Decisions

1. **Job orchestration: Temporal Cloud.** Durable async workflows for classify/krige/segment jobs with built-in retries, timeouts, and observability. Temporal workers run on Azure Container Apps and dispatch GPU inference to T4 instances. Lightweight and faster to integrate than self-hosted Redis/Celery.

2. **GPU instances: Azure Container Apps with T4.** T4 GPUs are sufficient for small Murmurative model inference (30M–50M params). Container Apps provide scale-to-zero for cost control and per-job GPU allocation via Temporal activity heartbeating.

3. **Segmenter architecture: Hybrid conv-Murmurative.** Convolutional encoder extracts local features from image patches → Murmurative attention bottleneck attends globally over the feature map → convolutional decoder upsamples to pixel resolution. This keeps Murmurative's O(N) attention advantage while leveraging conv layers for spatial locality.

4. **Notebook persistence: Multi-notebook per customer, stored in Icechunk.** Customers can create multiple notebooks/projects. Each notebook's cells, outputs, and widget state persist in `notebooks/{notebook_id}/state.json` in the customer's Icechunk container. Cell outputs reference Zarr arrays stored in `notebooks/{notebook_id}/outputs/`.

5. **Multi-user: Single user per customer initially.** One Stripe subscription = one customer = one user. Enforced via `customer_id` claim in JWT. Architecture supports future multi-user by keeping `customer_id` as the isolation boundary — when multi-user is added, multiple users map to the same `customer_id` and share the Icechunk container.

6. **Subscription tiers: Single tier initially.** One plan (`price_1TlexE8k0ubC0hdJrGUVKQI9`). Future tiers can gate on storage quota per container, concurrent job limits, and available task models.

7. **Database: Neon Postgres (serverless).** Metadata-only persistence — array data stays in Icechunk. UUID primary keys on all tables. Neon's branching model makes schema migrations safe (branch → migrate → merge) and the built-in query viewer eliminates the need for a separate admin tool. No array data is stored in Postgres; only Icechunk URI references, display IDs, job status, and subscription audit logs.