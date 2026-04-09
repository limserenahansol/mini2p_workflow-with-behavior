# Pain 2-Plane Calcium Imaging + Behavior Pipeline

End-to-end MATLAB pipeline for **2-plane interleaved calcium imaging** with synchronized behavioral video scoring and neuron-behavior merging.

## Data Structure

```
<session_folder>/
  CellVideo1/
    CellVideo/              Multi-part TIF stacks (e.g. 9 files, 2000 frames each)
                            512x440 px, 16-bit, interleaved plane A / plane B
    CellVideo_CHA_Info.tdms Per-frame timestamps (channel A)
    Information-CHA.txt     Acquisition metadata (frame rate, FOV, ETL, etc.)
  CellVideo2/
    Information-CHB.txt     Channel B metadata (same acquisition)
  MiceVideo1/
    MiceVideo/              Behavioral camera AVIs (channel A, ~30 Hz)
    MiceVideo...-reference.tdms   Per-frame timestamps
  MiceVideo2/
    MiceVideo/              Behavioral camera AVIs (channel B, ~30 Hz)
    MiceVideo...-reference.tdms   Per-frame timestamps
  SyncInformation/
    SignalSync_1..N.tdms    Hardware sync signals
  etl1data.csv              ETL z-stack parameters
```

**Key numbers (example session):**
- 18000 raw neuron frames at 9.21 Hz total -> 9000 max-projected frames (~4.6 Hz per plane)
- Behavioral cameras at 30 Hz, 4 AVI segments per camera
- Session duration ~28 min

---

## Pipeline Steps

### Step 1 — Load and Merge Planes

**Script:** `pain_2plane_step1_merge_planes.m`

1. Load all TIF files from `CellVideo1/CellVideo/` in order.
2. Deinterleave: odd frames = plane A, even frames = plane B.
3. For each consecutive pair, compute pixel-wise `max(planeA_frame, planeB_frame)`.
4. Result: N/2 combined frames (e.g. 9000 from 18000).
5. Save as `combined_maxproj.tif` in the output folder.

**Why max-projection?** Collapses two focal planes into one image per time point, preserving the brightest signal from either plane. This gives a single-plane movie suitable for EXTRACT.

---

### Step 2 — Run EXTRACT

**Script:** `pain_2plane_step2_extract.m`

Wrapper around `speed_dff_extract_HS_tiff_integrated_v2.m` with settings tuned for this data:

- **Input:** `combined_maxproj.tif` from Step 1.
- **Frame rate:** `ca_frame_rate_hz = 4.605` (9.21 / 2 planes).
- **Check videos:** Only after motion correction (`02_check_after_MC.avi`) and after EXTRACT (`04_check_EXTRACT_overlay.avi`). Videos 01 and 03 are skipped for speed.
- **EXTRACT preset:** `permissive` (default; tune if too many/few cells).
- **All other parameters:** Same as v2 defaults (custom MC template, morphology cleanup, per-cell trace plots, QC report, etc.).

**Outputs:** `*_MC_extractout.mat`, `deltaF_over_F`, per-cell trace PNGs, heatmap, overlay, `final_analysis_results.mat`.

---

### Step 3 — Parse Timestamps

**Script:** `pain_2plane_step3_parse_timestamps.m`

1. Read `CellVideo1/CellVideo_CHA_Info.tdms` for per-frame neuron timestamps.
2. After max-projection pairing, take the timestamp of each pair's first frame -> 9000 neuron timestamps.
3. Read `MiceVideo1/MiceVideo...-reference.tdms` for camera 1 frame timestamps.
4. Read `MiceVideo2/MiceVideo...-reference.tdms` for camera 2 frame timestamps.
5. Optionally read `SyncInformation/SignalSync_*.tdms` for hardware sync pulses.
6. Save `timestamps.mat` with `neuron_timestamps`, `behav1_timestamps`, `behav2_timestamps`.

**TDMS reading:** Uses MATLAB `tdmsread()` (R2022a+). Falls back to community `TDMS_readTDMSFile` if unavailable.

---

### Step 4 — Behavioral Video Scoring GUI

**Script:** `pain_2plane_step4_behavior_scoring.m`

MATLAB figure-based video player for manual event annotation:

- Concatenates all AVI segments from one camera into a continuous timeline.
- Playback: play/pause (Space), step forward/backward (arrow keys), speed control.
- **Event marking:** Press **A** to mark stimulus touch at the current frame.
- Press **D** to delete the last event. Press **S** to save.
- Displays: frame number, elapsed time, timestamp (from TDMS), event count.
- Saves `behavior_events_cam1.mat` (or cam2) with:
  - `event_frames` — frame indices in the concatenated video
  - `event_timestamps` — absolute timestamps from TDMS
  - `event_labels` — cell array of event types (all `'A'` for now)

---

### Step 5 — Merge Neuron Traces + Behavior

**Script:** `pain_2plane_step5_merge.m`

1. Load `final_analysis_results.mat` from Step 2 (`deltaF_over_F`: 9000 x N_cells).
2. Load `timestamps.mat` from Step 3 (neuron + behavior timestamps).
3. Load `behavior_events_cam1.mat` from Step 4 (scored events).
4. **Temporal alignment:**
   - For each neuron frame, define a time window [t_neuron - dt/2, t_neuron + dt/2] where dt = 1/4.605 s.
   - Mark `event_vector(i) = 1` if any scored behavior event falls within that window.
5. **Final output:** `final_neuron_behavior.mat` containing:

| Field | Size | Description |
|-------|------|-------------|
| `deltaF_over_F` | 9000 x N_cells | Neuron traces at ~4.6 Hz |
| `neuron_timestamps` | 9000 x 1 | Absolute time per neuron frame |
| `event_vector` | 9000 x 1 | Binary: 1 = stimulus event in this frame |
| `event_timestamps_original` | M x 1 | Original event times at 30 Hz resolution |
| `spatial_weights` | H x W x N_cells | EXTRACT spatial footprints |
| `metadata` | struct | Frame rates, paths, session info |

---

## Running the Pipeline

### Quick Start

```matlab
% Set session folder and run everything
pain_2plane_master
```

### Step-by-Step

```matlab
% Edit paths in pain_2plane_master.m, then:
% 1. Merge planes (run once)
pain_2plane_step1_merge_planes

% 2. Run EXTRACT (re-run with skip_steps_1_2=true to tune EXTRACT only)
pain_2plane_step2_extract

% 3. Parse timestamps
pain_2plane_step3_parse_timestamps

% 4. Score behavior (interactive GUI — do for each camera)
pain_2plane_step4_behavior_scoring

% 5. Merge everything
pain_2plane_step5_merge
```

### Re-running Steps

Each step checks for existing outputs. Use `skip_*` flags in `pain_2plane_master.m` to control which steps run:

```matlab
skip_step1 = true;   % plane merge already done
skip_step2 = false;  % re-run EXTRACT with new params
skip_step3 = true;   % timestamps already parsed
skip_step4 = true;   % scoring already done
skip_step5 = false;  % re-merge with new scoring
```

---

## Requirements

- **MATLAB** R2022a+ (for `tdmsread`; or install community TDMS reader)
- **EXTRACT:** [schnitzer-lab/EXTRACT-public](https://github.com/schnitzer-lab/EXTRACT-public)
- **NoRMCorre:** for motion correction
- **Image Processing Toolbox** (for `imgaussfilt`)
- **Computer Vision Toolbox** (for `VideoReader`, `insertText`)

---

## Output Files Summary

| Step | Output | Location |
|------|--------|----------|
| 1 | `combined_maxproj.tif` | output_folder |
| 2 | `*_MC_extractout.mat`, figures, check videos, `final_analysis_results.mat` | output_folder |
| 3 | `timestamps.mat` | output_folder |
| 4 | `behavior_events_cam1.mat` | output_folder |
| 5 | `final_neuron_behavior.mat` | output_folder |
