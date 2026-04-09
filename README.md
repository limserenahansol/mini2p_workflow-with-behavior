# Mini-2P Calcium Imaging Workflow with Behavior

Complete MATLAB pipeline for **2-photon calcium imaging** data processing and **behavioral video scoring**, from raw acquisition to a single synchronized neuron+behavior output file ready for downstream analysis.

---

## Repository Structure

```
mini2p_workflow-with-behavior/
│
├── extract_pipeline/                         # General-purpose EXTRACT pipelines
│   ├── speed_dff_extract_HS_tiff.m           # v1 — single-file pipeline
│   └── speed_dff_extract_HS_tiff_integrated_v2.m  # v2 — batch + advanced features
│
├── pain_2plane_pipeline/                     # Pain experiment: 2-plane imaging + behavior
│   ├── pain_2plane_master.m                  # Master runner (set paths, press F5)
│   ├── pain_2plane_step1_merge_planes.m      # Load interleaved TIFs → max-project
│   ├── pain_2plane_step2_extract.m           # EXTRACT wrapper (MC + cell detection)
│   ├── pain_2plane_step3_parse_timestamps.m  # Parse TDMS timestamp files
│   ├── pain_2plane_step4_behavior_scoring.m  # Interactive video scoring GUI (4 stim types)
│   ├── pain_2plane_step5_merge.m             # Merge neuron traces + behavior events
│   ├── PIPELINE.md                           # Detailed pipeline documentation
│   ├── validate_pipeline.py                  # Python validation/QC script
│   │
│   ├── downstream_master.m                   # Downstream analysis runner (press F5)
│   ├── downstream_step1_sync_video.m         # DS-1: Synchronized 3-view video clip
│   ├── downstream_step2_peri_event_traces.m  # DS-2: Peri-event dF/F traces
│   ├── downstream_step3_auc_stats.m          # DS-3: AUC pre/post statistics
│   ├── downstream_step4_baseline_vs_late.m   # DS-4: Baseline vs late comparison
│   └── DOWNSTREAM_README.md                  # Downstream pipeline documentation
│
└── README.md                                 # ← You are here
```

---

## Pipelines

### 1. EXTRACT Pipeline (`extract_pipeline/`)

General-purpose calcium imaging cell extraction pipeline. Two versions:

| Version | File | Description |
|---------|------|-------------|
| **v1** | `speed_dff_extract_HS_tiff.m` | Single-file processing with permissive cell detection, morphology cleanup, per-cell trace plots, QC report |
| **v2** | `speed_dff_extract_HS_tiff_integrated_v2.m` | All v1 features + batch mode, custom MC template from stable block, z-score before EXTRACT, temporal binning, dual-channel support, SignalSorter integration |

**Both versions produce:**
- Motion-corrected HDF5 movie
- EXTRACT spatial/temporal weights
- ΔF/F heatmap, cell overlay, per-cell trace PNGs
- QC report (text + figure + MAT)
- Quality-check videos at each processing stage

### 2. Pain 2-Plane Pipeline (`pain_2plane_pipeline/`)

End-to-end pipeline for **2-plane interleaved calcium imaging** with synchronized behavioral video scoring. Designed for pain experiments with ETL z-stack acquisition.

### 3. Downstream Analysis Pipeline (`pain_2plane_pipeline/downstream_*`)

Post-processing analysis that runs after the main pipeline produces `final_neuron_behavior.mat`:

| Step | Script | Description |
|------|--------|-------------|
| **DS-1** | `downstream_step1_sync_video.m` | Synchronized 3-panel video: cam1 (stimuli) + cam2 (reaction) + dF/F heatmap |
| **DS-2** | `downstream_step2_peri_event_traces.m` | Peri-event (time-locked) dF/F traces per stimulus type (mean ± SEM) |
| **DS-3** | `downstream_step3_auc_stats.m` | AUC pre/post event bar graphs with Wilcoxon signed-rank tests |
| **DS-4** | `downstream_step4_baseline_vs_late.m` | Baseline (0–5 min) vs late (23–28 min) activity comparison |

**Event types (Camera 1):** Soft Touch, Strong Touch, Mechanic Pain, Thermo Pain  
**Event types (Camera 2):** Mouse Reaction, Reaction Offset

**Run:** Open `downstream_master.m`, verify paths, press F5. See [`DOWNSTREAM_README.md`](pain_2plane_pipeline/DOWNSTREAM_README.md) for full documentation.

---

## Pipeline Schematic (Pain 2-Plane)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA ACQUISITION                         │
│                                                                     │
│  CellVideo1/CellVideo/         MiceVideo1/MiceVideo/                │
│  ┌──────────────────┐          ┌──────────────────┐                 │
│  │ 9 TIF stacks     │          │ 4 AVI files      │                 │
│  │ 2000 frames each │          │ ~30 Hz, 1600x1200│                 │
│  │ interleaved A/B  │          │ behavioral camera │                 │
│  └────────┬─────────┘          └────────┬─────────┘                 │
│           │                             │                           │
│  CellVideo_CHA_Info.tdms       MiceVideo...-reference.tdms          │
│  (18000 frame timestamps)      (58640 frame timestamps)             │
└───────────┼─────────────────────────────┼───────────────────────────┘
            │                             │
            ▼                             ▼
┌───────────────────────┐    ┌────────────────────────────┐
│  STEP 1: Merge Planes │    │  STEP 3: Parse Timestamps  │
│                       │    │                            │
│  odd frames = plane A │    │  Neuron: 18000 → 9000 ts  │
│  even frames = plane B│    │  (take 1st of each pair)   │
│  max(A, B) per pair   │    │  Behavior: 58640 ts @ 30Hz│
│  18000 → 9000 frames  │    │                            │
│                       │    │  Output: timestamps.mat    │
│  Output:              │    └─────────────┬──────────────┘
│  combined_maxproj.tif │                  │
└───────────┬───────────┘                  │
            │                              │
            ▼                              │
┌───────────────────────┐                  │
│  STEP 2: EXTRACT      │                  │
│                       │                  │
│  Motion correction    │                  │
│  (NoRMCorre + custom  │                  │
│   MC template)        │    ┌─────────────────────────────┐
│  Spatial denoising    │    │  STEP 4: Behavior Scoring   │
│  EXTRACT cell detect  │    │  (Interactive MATLAB GUI)   │
│  Morphology cleanup   │    │                             │
│  ΔF/F computation     │    │  ┌─────────────────────┐   │
│  Per-cell traces      │    │  │ Video Player        │   │
│                       │    │  │ ┌─────────────────┐ │   │
│  Output:              │    │  │ │                 │ │   │
│  final_analysis_      │    │  │ │  behavioral     │ │   │
│  results.mat          │    │  │ │  video frame    │ │   │
│  (deltaF_over_F,      │    │  │ │                 │ │   │
│   spatial_weights,    │    │  │ └─────────────────┘ │   │
│   check videos,       │    │  │ Frame: 1234/58640   │   │
│   QC report)          │    │  │ Time: 17:49:52.301  │   │
└───────────┬───────────┘    │  │ Events: 12          │   │
            │                │  │                     │   │
            │                │  │ [A]=mark [D]=undo   │   │
            │                │  │ [Space]=play/pause  │   │
            │                │  └─────────────────────┘   │
            │                │                             │
            │                │  Output:                    │
            │                │  behavior_events_cam1.mat   │
            │                └──────────────┬──────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│              STEP 5: Merge & Align                      │
│                                                         │
│  Neuron traces (9000 frames × N cells @ 4.6 Hz)        │
│  + Behavior events (timestamps from 30 Hz scoring)      │
│  → Align by TDMS timestamps                             │
│  → Binary event vector at neuron frame rate              │
│                                                         │
│  Output: final_neuron_behavior.mat                      │
│  ┌───────────────────────────────────────────┐          │
│  │ deltaF_over_F    (9000 × N_cells)         │          │
│  │ neuron_timestamps (9000 × 1)              │          │
│  │ event_vector      (9000 × 1, binary)      │          │
│  │ spatial_weights   (H × W × N_cells)       │          │
│  │ metadata          (rates, paths, info)     │          │
│  └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| **MATLAB** | R2022a or later | Main pipeline (needs `tdmsread` built-in) |
| **Image Processing Toolbox** | — | `imgaussfilt`, `regionprops`, `bwboundaries` |
| **Computer Vision Toolbox** | — | `VideoReader`, `insertText`, `VideoWriter` |
| **Python** | 3.9+ | Validation script only (optional) |

### Step 1: Clone this repository

```bash
git clone https://github.com/limserenahansol/mini2p_workflow-with-behavior.git
cd mini2p_workflow-with-behavior
```

### Step 2: Install EXTRACT and NoRMCorre

```bash
# Clone EXTRACT (cell detection)
git clone https://github.com/schnitzer-lab/EXTRACT-public.git

# Clone NoRMCorre (motion correction)
git clone https://github.com/flatironinstitute/NoRMCorre.git
```

Place these anywhere on your system. You'll point to them in the master script.

### Step 3: Add paths in MATLAB

In the master script (`pain_2plane_master.m`), update these paths:

```matlab
% ---- Edit these to match your system ----
EXTRACT_path   = 'C:\path\to\EXTRACT-public';
NoRMCorre_path = 'C:\path\to\NoRMCorre';
```

### Step 4 (optional): Python validation dependencies

```bash
pip install npTDMS Pillow opencv-python scipy h5py numpy
```

---

## Quick Start (Pain 2-Plane Pipeline)

### 1. Edit paths in the master script

Open `pain_2plane_pipeline/pain_2plane_master.m` and set:

```matlab
% Session folder containing CellVideo1/, MiceVideo1/, etc.
session_dir = 'E:\20260409_Pain_2plane_...\';

% Where all outputs go
output_folder = fullfile(session_dir, 'output');
```

### 2. Run the full pipeline

```matlab
>> cd pain_2plane_pipeline
>> pain_2plane_master
```

This runs Steps 1–5 in order. Step 4 (behavioral scoring) opens an interactive GUI — the pipeline pauses until you finish scoring and close the GUI.

### 3. Or run steps individually

```matlab
% Step 1: Merge 2-plane TIFs → combined_maxproj.tif
pain_2plane_step1_merge_planes(session_dir, output_folder)

% Step 2: Run EXTRACT on the combined movie
pain_2plane_step2_extract(combined_tif_path, output_folder)

% Step 3: Parse TDMS timestamps
pain_2plane_step3_parse_timestamps(session_dir, output_folder)

% Step 4: Interactive behavioral scoring GUI
pain_2plane_step4_behavior_scoring(session_dir, 1, output_folder, timestamps_mat)

% Step 5: Merge neuron + behavior → final file
pain_2plane_step5_merge(output_folder)
```

### 4. Skip completed steps

Set skip flags in `pain_2plane_master.m`:

```matlab
skip_step1 = true;   % Already merged planes
skip_step2 = false;  % Re-run EXTRACT with new params
skip_step3 = true;   % Timestamps already parsed
skip_step4 = true;   % Scoring already done
skip_step5 = false;  % Re-merge with updated scoring
```

---

## Quick Start (General EXTRACT Pipeline)

For standard single-plane recordings (not 2-plane interleaved):

### v1 (simple single-file)

```matlab
% Open extract_pipeline/speed_dff_extract_HS_tiff.m
% Edit paths at the top, then press F5
```

### v2 (batch + advanced)

```matlab
% Open extract_pipeline/speed_dff_extract_HS_tiff_integrated_v2.m
% Set run_mode = 'single' or 'batch'
% Edit paths, then press F5
```

Key v2 options:

```matlab
run_mode = 'batch';           % Process all TIFs in input_dir
extract_preset = 'permissive'; % 'permissive' for dim cells, 'stricter' for high-SNR only
extract_bin_time = 4;          % Temporal binning before EXTRACT (speed vs resolution)
dff_trace_source = 'extract';  % 'extract' = all cells, 'signal_sorter' = curated subset
```

---

## Expected Data Structure

### Pain 2-Plane Experiment

```
<session_folder>/
├── CellVideo1/
│   ├── CellVideo/
│   │   ├── CellVideo 1.tif        # 2000 frames, 512×440, 16-bit
│   │   ├── CellVideo 2.tif        # interleaved: frame1=planeA, frame2=planeB, ...
│   │   └── ... (up to 9 files)
│   ├── CellVideo_CHA_Info.tdms    # Per-frame timestamps (18000 entries)
│   └── Information-CHA.txt        # Acquisition metadata
├── MiceVideo1/
│   ├── MiceVideo/
│   │   ├── CHA...174811.avi       # Behavioral camera segments
│   │   ├── CHA...175812.avi       # 4 files, ~30 Hz, 1600×1200
│   │   ├── CHA...180812.avi
│   │   └── CHA...181812.avi
│   └── MiceVideo...-reference.tdms  # Per-frame timestamps (58640 entries)
├── MiceVideo2/                     # Second camera (same structure)
├── SyncInformation/
│   └── SignalSync_1..4.tdms        # Hardware sync signals
└── etl1data.csv                    # ETL z-stack parameters
```

### Standard Single-Plane Recording

```
<data_folder>/
├── recording_001.tif              # Single TIF stack
└── ...
```

---

## Pipeline Outputs

### Pain 2-Plane Pipeline

| Step | Output File | Contents |
|------|-------------|----------|
| 1 | `combined_maxproj.tif` | 9000-frame max-projected movie (uint16) |
| 1 | `combined_maxproj.mat` | Same movie in MAT format (faster reload) |
| 2 | `*_MC_extractout.mat` | EXTRACT raw output (spatial + temporal weights) |
| 2 | `final_analysis_results.mat` | ΔF/F traces, cell info, peak indices |
| 2 | `02_check_after_motion_correction.avi` | QC video: motion-corrected movie |
| 2 | `04_check_EXTRACT_overlay.avi` | QC video: detected cells on movie |
| 2 | `cell_overlay_full_FOV.png` | All detected cells on mean image |
| 2 | `deltaF_over_F_heatmap.png` | ΔF/F heatmap (time × cells) |
| 2 | `trace_trajectories_per_cell/` | One PNG per cell (full-length raw trace) |
| 2 | `QC_report.txt` / `.mat` / `.png` | Quality control metrics |
| 3 | `timestamps.mat` | Parsed neuron + behavior timestamps |
| 4 | `behavior_events_cam1.mat` | Scored events with frame indices + timestamps |
| 5 | **`final_neuron_behavior.mat`** | **Synchronized neuron traces + behavior events** |

### Final Output Schema (`final_neuron_behavior.mat`)

```matlab
>> load('final_neuron_behavior.mat')

deltaF_over_F          % [9000 × N_cells] double — ΔF/F traces at ~4.6 Hz
neuron_timestamps      % [9000 × 1] datetime     — absolute timestamp per frame
neuron_seconds         % [9000 × 1] double        — seconds since first frame
event_vector           % [9000 × 1] logical       — 1 = stimulus event in this frame
event_timestamps_original  % [M × 1] datetime     — original event times at 30 Hz
spatial_weights        % [H × W × N_cells]        — EXTRACT spatial footprints
metadata               % struct with rates, paths, session info
```

---

## Behavioral Scoring GUI (Step 4)

### Controls

| Key | Action |
|-----|--------|
| **Space** | Play / Pause |
| **Right Arrow** | Step forward 1 frame |
| **Left Arrow** | Step backward 1 frame |
| **A** | Mark stimulus event at current frame |
| **D** | Delete last event |
| **S** | Save events to file |
| **Q** / **Escape** | Save and close |
| **+** / **-** | Speed up / slow down playback |

### Usage

```matlab
% Score camera 1 videos
pain_2plane_step4_behavior_scoring(session_dir, 1, output_folder, 'timestamps.mat')

% Score camera 2 videos
pain_2plane_step4_behavior_scoring(session_dir, 2, output_folder, 'timestamps.mat')
```

---

## EXTRACT Parameters

Both pipelines use EXTRACT with these key tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `avg_cell_radius` | 8 | Expected cell radius in pixels |
| `extract_preset` | `'permissive'` | `'permissive'` = dim cells; `'stricter'` = high SNR |
| `extract_bin_time` | 4 | Temporal binning before EXTRACT (1 = full rate) |
| `use_custom_mc_template` | `true` | Build MC template from stable block |
| `mc_max_shift` | 15 | Max motion correction shift (pixels) |
| `denoise_gauss_sigma` | 1.0 | Spatial Gaussian smoothing sigma |
| `ca_frame_rate_hz` | varies | Frame rate for time axis (4.605 for 2-plane) |
| `dff_trace_source` | `'extract'` | `'extract'` = all cells; `'signal_sorter'` = curated |

---

## Validation (Python)

Run the Python validation script to verify data integrity before running the MATLAB pipeline:

```bash
cd pain_2plane_pipeline
python validate_pipeline.py "E:\path\to\session_folder"
```

This checks:
- TIF file counts, dimensions, and interleaving
- TDMS timestamp parsing and frame rate consistency
- AVI file integrity and frame counts
- Temporal alignment sanity (rate ratios, duration matching)
- Output file structure (if pipeline has been run)

---

## Differences Between v1 and v2 (EXTRACT Pipeline)

| Feature | v1 | v2 |
|---------|----|----|
| Batch processing | — | Multiple TIFs in one run |
| MC template | Default NoRMCorre | Custom template from stable block |
| Z-score before EXTRACT | — | Optional per-pixel z-score |
| Temporal binning | — | Configurable bin factor |
| Dual channel | — | Green/red deinterleaving |
| SignalSorter | — | Automated cell curation |
| Trace source control | — | `dff_trace_source` option |
| Per-cell trace PNGs | ✓ | ✓ |
| QC report | ✓ | ✓ |
| Check videos | 4 stages | 4 stages (configurable FPS) |

---

## Troubleshooting

### "No TIF files found"
Check that the session folder path is correct and contains `CellVideo1/CellVideo/*.tif`.

### EXTRACT finds too few / too many cells
- Switch between `extract_preset = 'permissive'` and `'stricter'`
- Adjust `avg_cell_radius` to match your cell size
- Try `extract_mode = 'step1'` to visualize cell candidates before running full extraction

### Motion correction fails or is slow
- Increase `mc_max_shift` for large drift (up to 50)
- Set `use_custom_mc_template = false` if the template block search is too slow
- Enable GPU: requires MATLAB Parallel Computing Toolbox + CUDA GPU

### SignalSorter keeps too few cells
- Set `dff_trace_source = 'extract'` to use all EXTRACT cells (default)
- The 'signal_sorter' option uses automated curation which can be aggressive

### TDMS reading fails
- Requires MATLAB R2022a+ for built-in `tdmsread()`
- For older MATLAB: install [TDMS reader from File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/44206-tdms-reader)

---

## Citation

If you use this pipeline, please cite:
- **EXTRACT:** Inan et al., "Fast and statistically robust cell extraction from large-scale neural calcium imaging datasets" (2021)
- **NoRMCorre:** Pnevmatikakis & Giovannucci, "NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data" (2017)

---

## License

This project is provided as-is for academic research use.
