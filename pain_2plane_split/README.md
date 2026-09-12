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
| 10 | match cells across sessions | `match_sessions.py`, audited by `match_audit.py` | `<pain session>\output_split\match\` |
| 11 | **the 39-cell union — every later step reads this** | `transfer_footprints.py`, loaded by `union_data.py` | `<each session>\output_split\union\plane_*\union_traces.mat` |
| 12 | centre versus corner cells | `openfield_place_cells.py` | `output_split\place\` |
| 13 | one row per neuron, both sessions | `linked_cells_report.py` | `<pain session>\output_split\match\` |
| 14 | raw Ca / dF/F / z per cell | `pain_cell_traces.py` | `<plane>\curated\pain_traces.*` |
| 15 | the deck | `make_pipeline_ppt.py` | `CEANTSR1_pipeline_*.pptx` |

`compare_curated_traces.py` and `sync_curation_record.py` are checks, not
steps.

## Figures for confirmation

Four figure sets, each opening with the whole field of view and then going
to detail. Methods and statistics are written into each figure's own legend,
so they stand alone. Examples are in [`figures/`](figures).

| script | what it argues | output |
|---|---|---|
| `fig1_processing.py` | each processing step improved the movie, measured | `<plane>\curated\fig1_processing.png` |
| `fig2_detection.py` | the detection is real, shown rather than asserted | `<plane>\curated\fig2_detection.png` |
| `fig3_place.py` | how a centre or corner cell is decided | `<open field>\place\fig3_place.png` |
| `fig4_matched.py` | the same neuron in both sessions, one row per cell | `<pain>\match\fig4_matched_{z,dff,rawF}.png` |
| `make_visual_ppt.py` | the four sets as one deck, tall figures cut into slide-shaped bands | `CEANTSR1_visualisation_*.pptx` |
| `make_pipeline_ppt.py` | the narrative report: data, pipeline, QC, results | `CEANTSR1_pipeline_*.pptx` |

Three design rules these follow, because each one caught a mistake:

**One metric judges one stage.** Plotting a single number across every
processing step and calling the rise an improvement is easy and wrong - a
spatial blur raises frame-to-frame correlation for free. `fig1_processing.py`
gives each stage the metric that is supposed to fix it and marks which with a
grey band. Two metrics also became invalid after the band-pass, which removes
the mean: mean/noise SNR read 1.48 and a ratio-form contrast read +15.5, both
from dividing by ~0. They are now marked n/a or normalised by the image SD.

**Show the comparison, not the p-value.** `fig2_detection.py` puts each cell
next to its own footprint shape moved to the nearest clear spot in the same
movie. Same tissue, same noise, same extraction, no soma. On pain plane A the
cell beats its own control on lag-1 autocorrelation in 11 of 11 cases and on
skew in only 6 of 11 - so one of the two metrics is doing the work here, which
the figure shows rather than hides.

**Never let smoothing invent data.** The first place-map version blurred the
occupancy-normalised rate maps with a permissive weight cut, which painted
colour into bins the mouse never entered. The occupancy mask is now
re-applied after smoothing.

---

## The 39-cell union is the unit of analysis

Every cross-session question used to be asked of the neurons EXTRACT happened
to detect *twice*. That is a detection limit, not an anatomical one:
`match_audit.py` found a soma at the location of **25 of 27** pain cells and
**20 of 28** open-field cells in the other session's mean image.

`transfer_footprints.py` therefore registers the union of both sessions'
footprints and solves all of them jointly on **both** movies, so detection has
to succeed once per neuron instead of twice. Result: **39 neurons** (plane A
18, plane B 21), each with a trace in both sessions under one shared id, so
`A1` in the pain session and `A1` in the open field are the same neuron by
construction. 34 of the 39 sit on a soma in both sessions; 16 were detected
independently twice; 25 are active in at least one session, 10 in both.

`union_data.py` is the only loader. `usable()` filters on `anat >= 1` image SD
— a transferred footprint is a hypothesis, and below 1 SD it landed on nothing
and its trace is background whatever it looks like. It does **not** filter on
activity: a neuron that was quiet is a valid row, that *is* the measurement.

This changed the answers, which is the point: place coding went from 9 cells
tested to 36, and from 2 centre / 0 corner to **6 centre / 2 corner**.

Two bugs surfaced on the way, both worth knowing:

- **The audit's own sign error.** `match_audit.py` passed the pain session as
  `A` and then added the returned shift to the open-field centroids, and so
  reported 0 pairs where the matcher found 20. `grid_shift(A, B)` compares
  `A(y)` against `B(y + dy)`, so the shift maps A's coordinates into B's. The
  matcher was right; the audit was wrong. The convention is now written at the
  call site.
- **A shared module RNG made results depend on call order.** Two scripts
  computing the same circular-shift null disagreed — 2 corner cells in one, 0
  in the other — because borderline cells at q ≈ 0.046 flipped with the order
  the surrogates were drawn in. Seeds now come per cell from
  `hashlib.md5(plane|label|what)`, not `hash()` (which is salted per process),
  and the two scripts agree exactly.

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

Union of the two sessions, which is what every cross-session analysis uses:

| | plane A | plane B | both |
|---|---|---|---|
| union neurons | 18 | 21 | **39** |
| on a soma in both sessions | 15 | 19 | 34 |
| detected independently twice | 5 | 11 | 16 |
| active in the pain session | 9 | 9 | 18 |
| active in the open field | 6 | 11 | 17 |

**Open field.** Floor is 454 × 516 px = 11.9 % of the FOV; everything outside
it is ignored, which matters because "largest dark blob in the frame" tracked
the dark background for 1632 of 1632 sampled frames. Detection 99.96 % of
8159 frames. Occupancy: centre 5.6 %, corner 41.4 %, edge 53.0 %.

**Centre versus corner**, on the 39-cell union: 36 usable cells; **6 prefer
the centre** (A3 +0.896, A6 +0.679, B16 +0.543, A17 +0.539, A7 +0.521,
B17 +0.371) and **2 prefer the corners** (A16 −0.632, B20 −0.489), all at
q ≤ 0.045; 28 are not selective. The per-session curated set gave 2 and 0 from
9 cells, so most of this was a detection limit, not a biological absence.
Tested against a circular-shift null (2000 surrogates), which
keeps the trace's autocorrelation and the animal's occupancy — a t-test across
frames would treat each frame as independent and is not valid for calcium.
2 of 36 carry significant spatial information; 1 of 36 is speed-correlated and
no cell is both, so the zone effects are not speed in disguise. Centre
occupancy is 5.6 %, so the centre mean rests on 83 imaging frames against 623
in the corners — a centre effect is the harder one to detect, and few corner
cells is weak evidence.

**Same neuron across sessions.** Both sessions image the same two depths, and
each depth matches itself across the two sessions at image correlation
r ≈ 0.87 against 0.33–0.44 for the wrong depth. Registration is a direct
correlation search over shifts; phase correlation failed here, returning
contradictory shifts between depths and no pairs at all. For *detection*
pairs, position alone is not enough in a dense plane, so a pair must also
agree in footprint shape (r ≥ 0.5): 16 of 22 position pairs pass both, against
~3 expected by chance. Those 16 plus the 23 cells detected in only one session
are the 39-neuron union above.

**Zone-selective cells followable into the pain session:** A3 (centre, pain
active), A6 (centre, active), A17 (centre, active), A16 (corner, active),
A7 (centre, quiet), B16 (centre, quiet), B20 (corner, quiet). B17 is
centre-preferring in the open field but its pain footprint sits at 0.23 SD —
it landed on nothing there, so its pain trace means nothing and it is excluded.

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
