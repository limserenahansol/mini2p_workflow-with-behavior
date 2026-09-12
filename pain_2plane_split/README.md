# pain_2plane_split — two-plane CEA-Ntsr1 imaging, depths kept separate

Companion to `pain_2plane_pipeline/`. That pipeline max-projects the two ETL
depths into one movie; this one keeps them apart, because on the 2026-09-11
sessions the projection was measured to merge neurons:

- 5 of 17 cells at one depth sit within a cell radius of a cell at the other,
  so ~29 % get merged;
- `max()` is nonlinear and neither depth dominates, so the projected value
  flips between the two neurons in 50 % of consecutive frames.

Nothing here writes into `<session>\output\`. All results go to
`<session>\output_split\`, so both pipelines can run on the same data.

Full measured QC report, including every number quoted below:
[`EXTRACT_QC_20260911.md`](EXTRACT_QC_20260911.md).

---

## Order of work

| # | step | script | writes to |
|---|------|--------|-----------|
| 1 | split the two depths | `pain_2plane_step1_split_planes.m` (via `split_master.m`) | `output_split\plane_A.tif`, `plane_B.tif` |
| 2 | EXTRACT per plane | `pain_2plane_step2_extract.m` (via `split_master.m`) | `output_split\plane_A\`, `plane_B\` |
| 3 | is each footprint a cell? | `extract_qc_cells.py` | `<plane>\qc_cells\` |
| 4 | build the curation sheets | `cell_curation_sheet.py`, `make_curation_page.py` | `output_split\curation\` |
| 5 | apply the decisions | `apply_curation.py` then `apply_curation.m` | `<plane>\curated\` |
| 6 | prove the curation is what was decided | `verify_curation.py` | stdout, exits non-zero on any mismatch |
| 7 | per-cell figures | `plot_curation_on_fov.py`, `plot_cell_by_cell.py`, `plot_cellmap_traces.py` | `<plane>\curated\` |
| 8 | one time base for imaging and cameras | `split_timestamps.py` | `output_split\timestamps\` |
| 9 | open-field tracking | `openfield_track.py` | `output_split\tracking\` |
| 10 | centre versus corner cells | `openfield_place_cells.py` | `output_split\place\` |
| 11 | match cells across sessions | `match_sessions.py`, `linked_cells_report.py` | `<pain session>\output_split\match\` |
| 12 | raw Ca / dF/F / z per cell | `pain_cell_traces.py` | `<plane>\curated\pain_traces.*` |
| 13 | the deck | `make_pipeline_ppt.py` | `CEANTSR1_pipeline_*.pptx` |

`compare_curated_traces.py` and `sync_curation_record.py` are checks, not
steps.

---

## Five things that will bite anyone running this

### 1. ActSort silently replaces EXTRACT's code

ActSort ships its own fork of 20 EXTRACT helper files, 19 of which differ,
and it is in the **saved** MATLAB path on this machine (15 folders). Because
`addpath` prepends, ActSort wins every name. EXTRACT then dies with

```
Not enough input arguments.
    get_circularity_metrics line 12
```

because ActSort's version takes a 4th argument `parallel` that EXTRACT never
passes, and line 12 is `if parallel`. Worse than the crash: cell finding had
already run on ActSort's noise and SNR estimators, so the cell counts were
wrong before anything errored.

`mini2p_toolbox_paths.m` now `rmpath`s every ActSort folder and asserts that
all 141 EXTRACT functions resolve inside the EXTRACT tree. Load ActSort only
after extraction, in a fresh session.

### 2. numpy writes row-major, MATLAB reshapes column-major

`apply_curation.py` hands footprints to MATLAB as a `(h*w, k)` matrix. Written
with numpy's default C-order ravel, MATLAB's `reshape` scattered one 212 px
soma into 17 stripes of 16 px spanning all 440 rows — and nothing errored, so
every trace solved against it was wrong. With 440 ≠ 512 it is not even a
transpose.

Footprints are now re-raveled in Fortran order once, at the file boundary, and
`pain_2plane_step2_extract.m` asserts each footprint is a single compact blob
before using it.

### 3. Two clocks, about 37 s apart

The imaging side (`CHA/Time`, `SignalSync_N/Time`) runs ~37 s behind the
behaviour side (`MiceVideoN/Ref Time`). The recordings are simultaneous — the
spans agree to within 0.05 s — so this is a clock offset, not a delay.
Comparing absolute timestamps across the two clocks puts the data 37 s out.

### 4. Behaviour frame index ÷ 25 is not time

The cameras dropped frames: 6 in the open field, 11 and 12 in the pain
session, clustered, and in the pain session **both cameras drop at the same
indices** (a system stall, not a camera fault). The error is cumulative and
reaches 0.24–0.48 s. Use the per-frame `Ref Time`. `split_timestamps.py`
emits the mapping.

### 5. EXTRACT cannot hold a fixed footprint set

With cell finding off and every quality gate opened, EXTRACT's S-step still
moves the curated footprints (0.8–8.1 px here) and drops some outright —
`is_S_tiny` contains an unconditional `S_smooth_area_1 == 0` clause that no
threshold opens, and it removes exactly the footprints curation added.

So traces for a curated set come from a joint least-squares solve,
`(SᵀS)⁻¹SᵀM`, which is the T-step's core: it demixes overlapping footprints
and by construction can neither move nor drop a cell. EXTRACT's T-step traces
are still saved when its footprints stayed within one radius, and
`compare_curated_traces.py` measures the agreement (detrended r: median
0.94–0.96, worst cell 0.72).

---

## How "is this a cell" is decided

Two separate questions, because *no signal* and *not a cell* are different
claims.

**Does it have calcium signal?** At 4.6 Hz with a measured single-frame SNR of
0.75, a real cell's trace also looks like noise by eye, so the null is
measured rather than assumed: 200 control ROIs are cut from the same
motion-corrected movie, with the same footprint shapes moved to places
containing no detected cell, and their traces taken by the identical method.
Skewness and lag-1 autocorrelation of the detrended dF/F are combined with
Fisher's method and calibrated against the same statistic over those 200
controls, so the cut sits at a 5 % false-positive rate on background.

Detrending (20 s running median) is not optional: background ROIs in the pain
session carry a raw lag-1 autocorrelation of 0.40, so on raw traces the test
measures drift, not calcium.

**Is there a soma there?** The footprint's brightness over the ring around it,
in the session mean image. Background median ≈ 0.000, p95 0.006–0.010.

| verdict | signal | anatomical contrast | meaning |
|---|---|---|---|
| `active` | yes | — | fired during the session |
| `silent` | no | above the background p95 | a real soma that did not fire |
| `SUSPECT` | no | not above it | neither; drop it |

The active/silent line is a threshold, not a biological state. In pain plane B
cell #7 passed at p = 0.0498 and cell #10 failed at p = 0.0547, with
practically identical skew and autocorrelation. Treat the labels as a ranking.
For stimulus-locked analysis, test every cell that sits on a real soma and
control the false discovery rate across them, rather than pre-filtering by
this label — the stimulus times make that a far more powerful test.

---

## Results so far, 2026-09-11 sessions

| | OF A | OF B | PA A | PA B |
|---|---|---|---|---|
| detected | 14 | 14 | 12 | 15 |
| curated | 12 | 16 | 11 | 16 |
| active / silent / SUSPECT | 5 / 7 / 0 | 5 / 9 / 2 | 6 / 4 / 1 | 6 / 10 / 0 |

**Open field.** Floor is 454 × 516 px = 11.9 % of the FOV; everything outside
it is ignored, which matters because "largest dark blob in the frame" tracked
the dark background for 1632 of 1632 sampled frames. Detection 99.96 % of
8159 frames. Occupancy: centre 5.6 %, corner 41.4 %, edge 53.0 %.

**Centre versus corner.** 9 usable cells; 2 prefer the centre (A#12 contrast
+0.900, A#3 +0.520, both q = 0.0022), none prefer the corners. Tested against
a circular-shift null (2000 surrogates), which keeps the trace's
autocorrelation and the animal's occupancy — a t-test across frames would
treat each frame as independent and is not valid for calcium. 0 of 9 are
speed-correlated, so the zone effects are not speed in disguise. Centre
occupancy is only 5.6 %, so absence of corner cells is weak evidence.

**Same neuron across sessions.** Both sessions image the same two depths, and
each depth matches itself across the two sessions at image correlation
r ≈ 0.87 against 0.33–0.44 for the wrong depth. Registration is a direct
correlation search over shifts; phase correlation failed here, returning
contradictory shifts between depths and no pairs at all. Pairs must be within
one cell radius **and** agree in footprint shape (r ≥ 0.5), because position
alone pairs neighbours in a dense plane: 13 of 21 position pairs pass both.
Both centre-preferring cells could be followed into the pain session.

**Ready for event locking.** Every trace carries a time vector on the same
base as the behaviour cameras, so a scored stimulus time indexes straight into
it. `lock_events()` in `pain_cell_traces.py` returns
(cells × events × lag) windows.

---

## Requirements

MATLAB R2024b with the Image Processing Toolbox, EXTRACT-public, NoRMCorre.
Python 3.13 with numpy, scipy, pandas, opencv-python, h5py, matplotlib,
npTDMS, python-pptx, Pillow.

Session paths are set at the top of each script.
