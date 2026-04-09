# Downstream Analysis Pipeline

Post-processing pipeline for the Pain 2-Plane calcium imaging experiment.  
Runs **after** the main pipeline (Steps 1–5) has produced `final_neuron_behavior.mat`.

---

## Workflow Overview

```
final_neuron_behavior.mat   (from main pipeline Step 5)
        │
        ├──► DS-1: Sync 3-View Video ──► sync_3view_clip.avi
        │
        ├──► DS-2: Peri-Event Traces ──► peri_event_results.mat + figures/
        │         │
        │         └──► DS-3: AUC Statistics ──► auc_statistics.mat + figures/
        │
        └──► DS-4: Baseline vs Late ──► baseline_vs_late_stats.mat + figures/
```

---

## Steps

### DS-1: Synchronized 3-View Video Clip

**Script:** `downstream_step1_sync_video.m`

Generates a composite AVI video with three synchronized panels:

| Panel | Content |
|-------|---------|
| Top-left | Camera 1 (stimuli) with color-coded event labels |
| Top-right | Camera 2 (mouse reaction) with event labels |
| Bottom | dF/F heatmap (cells × time) with playhead & event ticks |

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_neuron_frame` | 4400 | First neuron frame (~16 min into session) |
| `clip_duration_sec` | 180 | Real-time duration (3 min) |
| `speed_multiplier` | 5 | Playback speed |
| `video_width × height` | 1280×720 | Output resolution |

**Output:** `output/sync_3view_clip.avi`

---

### DS-2: Peri-Event (Time-Locked) dF/F Traces

**Script:** `downstream_step2_peri_event_traces.m`

For each stimulus type, extracts a time window around every event onset and computes trial-averaged dF/F traces.

**Event types analyzed:**
- Soft Touch (green)
- Strong Touch (orange)
- Mechanic Pain (red)
- Thermo Pain (purple)
- Mouse Reaction (blue)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_sec` | 3 | Seconds before event onset |
| `post_sec` | 5 | Seconds after event onset |

**Outputs:**
- `output/peri_event_results.mat` — all peri-event data (trials × time × cells)
- `output/figures/peri_event_pop_*.png` — population-average traces
- `output/figures/peri_event_cells_*.png` — per-cell trace panels

---

### DS-3: AUC Pre/Post Event Statistics

**Script:** `downstream_step3_auc_stats.m`

Computes the **Area Under Curve** (AUC) of dF/F in a pre-event window vs a post-event window, then runs paired statistical tests.

**Method:**
1. For each event, compute `AUC_pre = ∫ dF/F dt` over [−2, 0] s
2. Compute `AUC_post = ∫ dF/F dt` over [0, +2] s
3. Wilcoxon signed-rank test (or paired t-test if < 5 trials)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_window_sec` | 2 | Pre-event AUC window (seconds) |
| `post_window_sec` | 2 | Post-event AUC window (seconds) |
| `alpha` | 0.05 | Significance threshold |

**Outputs:**
- `output/auc_statistics.mat` — AUC values, p-values per cell & population
- `output/figures/auc_population_*.png` — population bar graph (pre vs post)
- `output/figures/auc_per_cell_*.png` — per-cell bar graph with significance markers

---

### DS-4: Baseline vs Late Activity Comparison

**Script:** `downstream_step4_baseline_vs_late.m`

Compares neuronal activity between an early **baseline** period and a **late** period to assess sensitization/habituation over the session.

**Default time windows:**
- Baseline: **0–5 min** (0–300 s)
- Late: **23–28 min** (1380–1680 s)

**Metrics compared (per cell):**

| Metric | Description |
|--------|-------------|
| Mean dF/F | Average fluorescence change |
| Peak dF/F | Maximum fluorescence change |
| Transient rate | Calcium events per minute (threshold = median + 2×MAD) |

**Outputs:**
- `output/baseline_vs_late_stats.mat` — all metrics and p-values
- `output/figures/baseline_vs_late_mean_dff.png` — paired bar graph
- `output/figures/baseline_vs_late_peak_dff.png` — paired bar graph
- `output/figures/baseline_vs_late_event_rate.png` — paired bar graph
- `output/figures/baseline_vs_late_scatter.png` — per-cell scatter (baseline vs late)
- `output/figures/baseline_vs_late_session_trace.png` — full session trace with shaded windows

---

## Quick Start

1. Open `downstream_master.m` in MATLAB
2. Verify `session_dir` and `output_folder` paths
3. Adjust skip flags and parameters as needed
4. Press **F5** to run

```matlab
% Example: Run only the statistics steps (skip video generation)
skip_ds1 = true;
skip_ds2 = false;
skip_ds3 = false;
skip_ds4 = false;
```

---

## Output File Summary

| File | Step | Description |
|------|------|-------------|
| `sync_3view_clip.avi` | DS-1 | Composite synchronized video |
| `peri_event_results.mat` | DS-2 | Peri-event traces (trials × time × cells) |
| `auc_statistics.mat` | DS-3 | AUC values and paired test results |
| `baseline_vs_late_stats.mat` | DS-4 | Baseline vs late metrics and statistics |
| `figures/*.png` | DS-2,3,4 | All publication-ready figures |

---

## Requirements

- MATLAB R2020b or later
- Image Processing Toolbox (for `insertText` in DS-1)
- Statistics and Machine Learning Toolbox (for `signrank`)
- Completed main pipeline output (`final_neuron_behavior.mat`)
- Camera AVI folders accessible (for DS-1 only)

---

## Adjusting Parameters

All parameters are set in `downstream_master.m` via `opts` structs. Common adjustments:

```matlab
% Change the video clip window
vid_opts.start_neuron_frame = 2000;   % start earlier
vid_opts.clip_duration_sec  = 300;    % 5 min instead of 3

% Wider peri-event window
peri_opts.pre_sec  = 5;
peri_opts.post_sec = 10;

% Stricter significance
auc_opts.alpha = 0.01;

% Different baseline/late windows
bl_opts.baseline_end_sec = 600;       % first 10 min
bl_opts.late_start_sec   = 1200;      % 20 min
bl_opts.late_end_sec     = 1500;      % 25 min
```
