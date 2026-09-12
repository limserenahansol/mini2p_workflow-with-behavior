# EXTRACT QC — CEA-Ntsr1, 2026-09-11 pain session

Everything below was measured from `CellVideo1/CellVideo/CellVideo 1.tif`
(2000 frames read, 1000 per depth) rather than taken from the folder names or
the existing docs. Pixel size 0.898 µm/px and frame rate 9.21 Hz from
`Information-CHA.txt`.

**Nothing has been run or changed yet.** Three findings have to be settled
first because each one changes what EXTRACT is given.

---

## What the data actually is

| | |
|---|---|
| `CellVideo1` (CHA) | 6000 frames, 512 × 440, 16-bit, 3 TIFs of 2000 |
| `CellVideo2` (CHB) | same shape and count, **same field of view, much dimmer** |
| `etl1data.csv` | two ETL positions, **65 µm and 15 µm**, Manual, Descending |
| `Information-CHA/CHB` | `ImagingMode ZStack_ETL`, 9.21 Hz, `CHA,CHB not in time division` |

**The two depths alternate frame by frame inside CellVideo1.** Mean of even
frames and mean of odd frames show *different somata* — the bright cell at
left-centre in one is absent from the other. So Step 1's deinterleave is the
right operation. `CellVideo2` is a second detection channel of the same
field, not the second depth; the pipeline never reads it, which is defensible
but should be a deliberate choice (see open question below).

**Single frames carry almost no structure.** A single exposure is scattered
photon events on black; anatomy only appears after averaging ~100 frames and
is clear at ~1000. This is why a naive frame-to-frame correlation returns
~0.01 and why every parameter below has to respect the photon budget.

---

## Finding 1 — BLOCKER. Max-projecting the two depths mixes neurons

Step 1 collapses the depth pair with `max(planeA, planeB)`. Measured
consequences:

| | |
|---|---|
| somata at depth A | 17 |
| somata at depth B | 14 |
| A-cells landing within one cell radius (8 px) of a B-cell | **5 of 17 = 29 %** |
| pixel-frames inside cells where the winning depth **changes** between consecutive output frames | **50 %** |

The first number means roughly a third of the cells cannot be separated after
projection: EXTRACT sees one blob where there are two neurons 50 µm apart and
returns one mixed trace.

The second number is worse. `max` is nonlinear and neither depth dominates
(depth A wins 49 % of the time), so the projected value flips between the two
neurons about every other frame. **Half of every projected trace is not one
cell's activity.** No EXTRACT parameter can repair that; it is baked into the
input movie.

**Fix:** run the two depths as two movies.

```
even frames -> plane_A.tif   4.605 Hz  -> EXTRACT -> cells tagged plane A
odd  frames -> plane_B.tif   4.605 Hz  -> EXTRACT -> cells tagged plane B
```

4.605 Hz is already the rate the pipeline assumes, so nothing downstream needs
a new frame rate — only a plane label per cell. Expected yield ~31 cells
instead of ~26 merged ones, and every trace belongs to one neuron.

---

## Finding 2 — BLOCKER. The traces are not ΔF/F

`ceantsr1_20260911_master.m` sets

```matlab
extract_opts.use_zscore_before_extract = true
```

and `pain_2plane_step2_extract.m` sets

```matlab
config.skip_dff = 1;
config.F_per_pixel = ones(...);
```

So EXTRACT receives a **z-scored** movie and is told not to compute ΔF/F. Its
output, stored as `deltaF_over_F`, is in z-units with a mean near zero and
genuinely negative values.

`downstream_step2_peri_event_traces.m` then computes

```matlab
peri_baselined = (trace - f0) / abs(f0);
```

with `f0` = mean of a 2 s pre-event window. Dividing a zero-mean quantity by
the absolute value of a short, noisy window of itself is not ΔF/F — the
denominator is near zero by construction, so the result explodes on some
trials and not others. The `abs()` is already a patch around a sign flip,
which suggests the instability was noticed.

**Pick one and be explicit:**

| Option | What to change | Output |
|---|---|---|
| **A — real ΔF/F** (recommended) | `use_zscore_before_extract = false`, let EXTRACT see raw F | `(F − F0)/F0` is meaningful; F0 > 0 |
| B — z-score throughout | keep z-scoring, and in downstream use `trace − f0` only, never divide | event-locked **z** change, not ΔF/F |

Option A is what the user's request ("event-locked trace dF/F each cell")
needs. Option B is defensible but must not be labelled ΔF/F anywhere.

---

## Finding 3 — BLOCKER for the event analysis. Binning leaves 2 samples per window

`extract_bin_time = 4` on top of the depth split:

```
9.21 Hz  ->  4.605 Hz (one depth)  ->  1.151 Hz (bin 4)  =  0.87 s per sample
```

Against the windows `downstream_step2` uses:

| Window | bin 4 | bin 2 | unbinned |
|---|---|---|---|
| F0 `[-3, -1]` s | **2 samples** | 5 | 9 |
| response `[1, 3]` s | **2 samples** | 5 | 9 |

Two samples cannot define a baseline or a response. But binning is not
pointless — it is what makes the cells findable at all:

| | temporal SD, background | soma − background | SNR |
|---|---|---|---|
| unbinned | 312 | 234 | **0.75** |
| bin 2 | 221 | 234 | **1.06** |
| bin 4 | 156 | 234 | **1.50** |

**Fix — the standard split:** bin heavily to FIND the cells, then apply those
spatial footprints to the UNBINNED movie to get the traces.

```
bin 4  ->  EXTRACT  ->  keep spatial filters S only
S  applied to the 4.605 Hz movie  ->  traces at 4.605 Hz
```

If that is too invasive for now, `extract_bin_time = 2` is the compromise: SNR
1.06 instead of 1.50, and 5 samples per window instead of 2.

---

## Finding 4 — OK. Cell radius is right

| | |
|---|---|
| measured median equivalent radius | **7.9 px = 7.1 µm** |
| depth A | 6.8 px (6.1 µm), range 3.1–14.7 |
| depth B | 8.5 px (7.6 µm), range 4.5–13.4 |
| `avg_cell_radius` in config | **8 px = 7.2 µm** |

No change needed. The wide range (3–15 px) argues for keeping
`size_lower_limit` and `size_upper_limit` generous, as the `low_snr` preset
already does.

---

## Finding 5 — WATCH. One region will generate false cells

The lower-right quadrant holds a bright, high-contrast **fibrous** structure —
processes and what looks like a vessel edge, not somata. It is brighter than
most real cells. With `cellfind_min_snr` at 0.5 (the `permissive` preset) this
will produce ROIs. The `low_snr` preset's `cellfind_numpix_threshold = 25` and
`eccent_thresh = 7` help, but eccentricity 7 is loose for rejecting elongated
processes.

**Check after the first run:** overlay the footprints on the mean image and
confirm nothing in that quadrant was accepted. If it was, raise
`eccent_thresh` toward 3–4 rather than tightening SNR, so real dim somata are
not lost along with it.

---

## Finding 6 — documentation bug

`PIPELINE.md` states the EXTRACT preset is `permissive`;
`ceantsr1_20260911_master.m` uses `low_snr`. `low_snr` is the correct choice
for this recording (SNR 0.75 unbinned). `PIPELINE.md` should be corrected so
nobody reruns with `permissive` on the strength of the docs.

---

## Open question for Hansol

`CellVideo2` (CHB) covers the same field at roughly 2/3 the brightness and is
never read. Was CHB meant to be

- a second indicator / anatomical channel (then it should stay out of EXTRACT
  but may be worth keeping as a reference image), or
- a redundant PMT channel (then ignoring it is right), or
- intended to be **combined** with CHA to improve SNR (possible — averaging
  the two channels would raise SNR by up to √2 if their noise is independent,
  which is worth testing before discarding)?

The answer changes whether the EXTRACT input should be CHA alone or
mean(CHA, CHB).

---

## Finding 7 — BLOCKER, now fixed. ActSort silently replaces EXTRACT's internals

EXTRACT died on both planes with

```
Not enough input arguments.
    get_circularity_metrics line 12
    remove_redundant line 22
    run_extract line 471
    extractor line 408
```

after already finding 73 cells (plane A) and 60 (plane B). `regionprops` was
not the cause: MATLAB reports a missing argument at the line where that
argument is first *used*, and there is no `regionprops` frame in the stack.

**ActSort ships its own fork of 20 EXTRACT helper files, 19 of which differ.**
`mini2p_toolbox_paths.m` added ActSort *after* EXTRACT, and `addpath`
prepends, so ActSort won every name:

| file | EXTRACT | ActSort |
|---|---|---|
| `get_circularity_metrics.m` | 25 lines, `(S, fov_size, threshold)` | 46 lines, `(S, fov_size, parallel, threshold)` |
| `get_trace_noise.m` | 49 lines | 6 lines |
| `kappa_of_epsilon.m` | 22 lines | 3 lines |
| `estimate_noise_std.m`, `filter_images.m`, `smooth_images.m`, `get_trace_snr.m`, `temporal_corruption.m`, `maybe_gpu.m`, `find_spurious_cells.m`, `downsample_time.m`, `get_active_frames.m`, `normalize_to_one.m`, `eps_func.m`, `select_indices.m`, `ndSparse.m`, `get_free_mem.m`, `plot_cells_overlay.m`, `brewermap.m` | — | all differ |
| `peakseek.m` | — | identical, harmless |

ActSort's `get_circularity_metrics` takes a 3rd argument `parallel` that
EXTRACT never passes, and line 12 is `if parallel`. Hence the error.

The damage is worse than a crash: the 73 and 60 candidate cells were found
using **ActSort's** noise and SNR estimators, not EXTRACT's. It corrupts the
numbers before it stops.

ActSort is in the **saved** MATLAB path (15 folders), so this affects every
MATLAB session on this machine, not just this script.

Fix in `mini2p_toolbox_paths.m`: `rmpath` any ActSort folder, add NoRMCorre
then EXTRACT, and then assert that all 141 EXTRACT functions resolve inside
the EXTRACT tree. Verified: with ActSort deliberately pre-loaded, the helper
removes 15 folders, the guard passes, and the 2-argument call works.
NoRMCorre has zero filename collisions with EXTRACT, so its order is
irrelevant. Load ActSort only after extraction, in a fresh session.

---

## Finding 8 — the traces carry drift that every ROI shares

Measured on background ROIs (see Finding 9 for how they are built), median
lag-1 autocorrelation of raw ΔF/F:

| | plane A | plane B |
|---|---|---|
| raw | 0.397 | 0.341 |
| after a 20 s running-median detrend | 0.120 | 0.098 |

At 0.4, background is as temporally correlated as a cell, so autocorrelation
and skew computed on raw traces measure drift (residual motion, illumination,
neuropil) rather than calcium, and cells stop separating from background. All
QC metrics are therefore computed on detrended ΔF/F. A 20 s window is ~13×
the slowest GCaMP decay being looked for, so transients pass through intact.

This also caught a reporting bug of my own: a τ that never falls to 1/e
within the window was printed as `0.00 s`, i.e. the slowest trace in the set
was reported as the fastest. It is now `NaN`, which reads correctly as "too
slow to be an indicator transient".

---

## Finding 9 — result. 9 of 27 cells carry calcium signal; 24 of 27 are real somata

`extract_qc_cells.py`. The question is not "does this look like a cell" — at
4.6 Hz and single-frame SNR 0.75 a real cell's trace also looks like noise by
eye. The question is whether it is **distinguishable from background in the
same movie**, so the null is measured: 200 control ROIs cut from the same
motion-corrected movie, with the same footprint shapes translated to
locations ≥12 px from any detected cell, traces pulled by the identical
weighted average.

Verdict per cell, from a Fisher combination of the empirical p-values for
skew and lag-1 autocorrelation, calibrated against the same statistic over
the 200 controls (kept at `p_joint ≤ 0.05`, so 5 % of background would pass):

| | plane A | plane B | total |
|---|---|---|---|
| detected | 12 | 15 | 27 |
| **active** — calcium signal separable from background | 5 | 4 | **9** |
| **silent** — no signal, but footprint brighter than its surround above the background p95 | 4 | 11 | **15** |
| **SUSPECT** — neither | 3 | 0 | **3** |

The anatomical contrast measure is what separates "no signal" from "not a
cell", and its null is tight: background median ≈ 0.000, p95 = 0.010
(plane A) / 0.006 (plane B), max 0.019 / 0.010; cells run 0.001–0.078.

**So EXTRACT is not detecting background.** 24 of 27 footprints sit on
genuine somata. Most of those neurons simply did not fire enough in 651 s to
produce a detectable transient — expected for a sparse Cre line. Only 3
footprints (plane A #2, #9, #11) have neither signal nor anatomical contrast.

τ separates the two groups cleanly: active cells 0.20–1.37 s, everything else
0.14–0.18 s ≈ one frame, i.e. white noise. Plane B #2 is a textbook GCaMP
trace — skew 2.21, ac1 0.75, τ 1.37 s, 27 events.

EXTRACT's own trace agrees with a plain footprint-weighted average of the raw
movie at r = 0.78–0.96 for every cell except plane A #9 (r = 0.64).

### Morphology flags

- **plane A #9**: 2777 px, 4× the median (700 px), and the lowest
  trace-vs-pixels agreement (0.64). Consistent with a vessel or a multi-cell
  merge. Already SUSPECT; drop it.
- **plane A #8, plane B #11**: footprint contains 2 separate blobs above 30 %
  of its own peak — two neurons mixed into one trace. Worse than a missed
  cell, because the trace is a blend. Exclude, or re-run those with a smaller
  radius.
- **plane B #12**: 1064 px, the largest in plane B, two lobes that merge
  below the 30 % threshold. Treat as a merge candidate.

### Consequence for step 4

9 active cells across both depths is a small sample for classifying pin-prick
versus heat versus behaviour-responsive neurons. The classification is still
worth doing, but per-category counts will be single digits and no
between-category comparison will be powered. Worth knowing before the manual
scoring rather than after.

---

## Finding 10 — open field. The FOV is 8× the arena, and the old mask tracked the wall

`openfield_track.py`. The camera sees 1600×1200; the white floor is
454×516 px = **11.9 % of the FOV**. Also in frame: the transparent box, the
bench on the left, a dark background filling the right third, white tape on
the box walls, and dark tape on the box lip.

"Largest dark blob in the frame" locked onto the dark right-hand background:
measured over the whole session it returned a body of 923,038–955,852 px
(half the frame) sitting still at x ≈ 1060, for 1632 of 1632 sampled frames.

The floor is the saturated white square and nothing else. Threshold sweep on
the session median image (largest bright component, holes filled):

| threshold | bbox | bbox fill | what it is |
|---|---|---|---|
| >180 | 640×680 | 0.68 | climbs into the white tape strips on the wall |
| >200 | 623×669 | 0.66 | same, plus bleed to the right |
| >220 | 571×625 | 0.66 | ragged right edge, tape spikes at the top |
| >235 | 566×531 | 0.75 | close, right edge still ragged |
| **>245** | **469×527** | **0.89** | **the floor, clean** |
| >250 | 453×527 | 0.91 | — |

`ARENA_THR` was 180 and is now 245. The mask is the fitted `minAreaRect`
(525×463 at 89.4°, i.e. axis-aligned to within 0.6°) rather than the ragged
threshold blob, eroded 8 px so the dark box wall cannot leak in as a ring.
Zones are the standard 3×3 grid of the floor rectangle, with width and height
fractions taken separately because the floor is not square.

Full pass, 8159 frames (5.44 min at 25 fps): detection 99.96 %, body area
median 6411 px (range 827–10875), distance 16476 px, median speed 17.6 px/s,
46.0 % of time moving. Occupancy: **centre 5.6 %, corner 41.5 %, edge
52.9 %** — strongly thigmotactic, as expected for a tethered mouse in a novel
box.

Known limitation, recorded rather than hidden: positions are floor-only by
instruction, so when the animal presses against a wall the part of its body
over the box lip is cut off and the centroid is pulled inward. Body area
drops from 6–9k px to ~2.5–3.3k px in those frames. 45.1 % of frames touch
the arena boundary and are flagged `clipped` in the CSV.

Distance is in pixels; pass `--arena-cm <floor side in cm>` to convert.

---

## Finding 11 — the deinterleave is now verified, not inferred, and the planes have depths

`CellVideo1\CellVideo_CHA_Info.tdms` has a **`Slice` channel**: the
acquisition software recorded which depth every raw frame belongs to. So the
split does not have to be assumed. Measured on both sessions:

| | open field | pain |
|---|---|---|
| raw frames | 3000 | 6000 |
| `Slice` alternates 1,2,1,2,… strictly | yes | yes |
| odd raw frame → | Slice 1 | Slice 1 |
| counts | 1500 / 1500 | 3000 / 3000 |
| `FrameLost` nonzero | 0 | 0 |
| per-plane rate | 4.6054 Hz | 4.6054 Hz |
| Slice 2 − Slice 1 | +109 ms | +109 ms |

`pain_2plane_step1_split_planes.m` assigns odd `raw_idx` to plane A, which
matches `Slice 1` exactly. With `etl1data.csv` (65.0/139.6 µm listed first,
15.0/89.6 µm second, "Stack Direction Descending") that fixes the physical
identity:

- **plane A = Slice 1 = ETL 65 µm relative (139.6 µm absolute)**
- **plane B = Slice 2 = ETL 15 µm relative (89.6 µm absolute), 109 ms later**

No frames were lost in either session, so the split is exact.

The 109 ms is one raw-frame interval at 9.21 Hz — equivalently half a
per-plane period. It is also why the existing timestamp step is wrong for
split planes: `pain_2plane_step3_parse_timestamps.m` states *"we keep the
timestamp of each pair's first frame (odd-indexed raw frames)"*, which is
right for a max-projected movie but gives plane B a fixed 109 ms bias. The
split version must emit two timestamp vectors. The frame counts in that
file's header (18000 raw / 9000 paired) match neither session here and
should not be used as a check.

To do before step 8: a split-aware timestamp step that reads the `Slice`
channel directly — parity works, but reading the recorded value is free and
cannot drift.

---

## Finding 12 — two clocks, ~37 s apart, and dropped behaviour frames

Two separate problems, both measured.

### The clocks

| stream | clock | start − session start |
|---|---|---|
| imaging `CHA/Time` | imaging | −36.64 s (OF) / −36.21 s (pain) |
| `SignalSync_N/Time` | imaging | −37.41 s (OF) / −37.07, −36.98 s (pain) |
| `MiceVideoN/Ref Time` | behaviour | +0.21 s (OF) / +0.56, +0.65 s (pain) |

The imaging-side clock runs ~37 s behind the behaviour-side one, in both
sessions, on every channel. The behaviour clock is the one that agrees with
the session folder name and `Information-CHA.txt`.

**But this is not a 37 s misalignment of the data.** The spans agree to
within 0.05 s (open field: imaging 325.59 s, sync 326.60 s, video 326.56 s),
so the recordings really are simultaneous — only the timestamp clocks differ.
Aligning by *absolute* timestamps across the two clocks would put the data
37 s out. Aligning by *relative* seconds from each stream's own first sample
— which is what `timestamps.mat` stores — is off by only **−0.58 s**
(behaviour starts before imaging), with ≤0.28 s residual uncertainty from the
unknown sync-pulse-to-camera-frame index offset.

The sync line cannot resolve that last 0.28 s by jitter matching: its
intervals are 40.000 ms with sd 0.022 ms and only two distinct values, i.e.
a generated clock with no jitter signature, so cross-correlating it against
the camera's intervals returns r ≈ 0 at every lag. Do not read a lag out of
that correlation.

### The dropped frames

| stream | saved | gap events | frames missing | sync pulses |
|---|---|---|---|---|
| OF `MiceVideo2` | 8159 | 5 | 6 | 8166 |
| pain `MiceVideo1` | 16299 | 8 | 11 | 16311 |
| pain `MiceVideo2` | 16296 | 10 | 12 | 16310 |

`saved + missing` lands within 1–2 of the sync pulse count, so the sync line
pulsed for frames the camera did not save. The drops are clustered, and in
the pain session **both cameras drop at the same frame indices**
(≈2103–2107, 8799–8801, 13823–13830) — a system-wide stall, not a camera
fault.

Consequence: **behaviour frame index ÷ 25 is not time.** The error is
cumulative and reaches 6–12 frames (0.24–0.48 s) by the end of a session.
Use `Ref Time` per frame. The scoring and open-field tracking outputs are
indexed by frame, so they need this mapping applied before any
neuron–behaviour comparison.

---

## Finding 13 — open-field EXTRACT. Cleaner movie, but many merged footprints

Split: 3000 → 1500 frames per plane, `corr(mean A, mean B) = 0.564` (pain
session 0.592), so the depths differ here too. EXTRACT: plane A 123 s,
plane B 112 s.

| | open field A | open field B | pain A | pain B |
|---|---|---|---|---|
| frames | 1500 | 1500 | 3000 | 3000 |
| noise std | 14.19 | 14.33 | 20.75 | 21.30 |
| candidates found | 38 | 58 | 73 | 60 |
| after quality + morphology | 14 | 14 | 12 | 15 |
| **active** | 5 | 7 | 5 | 4 |
| silent | 8 | 6 | 4 | 11 |
| SUSPECT | 1 | 1 | 3 | 0 |
| **usable** (active, single blob, not oversized) | **4** | **7** | **5** | **4** |

The open-field recording is substantially cleaner: noise std 14.2 against
20.8, and background raw lag-1 autocorrelation 0.067 / 0.044 against 0.397 /
0.341. The pain session carries far more drift and residual motion, which is
unsurprising given the stimulation.

**The cost of half the frames shows up as merged footprints.** EXTRACT
separates nearby cells using temporal decorrelation, and with 1500 mostly
quiet samples it has less to work with:

| | footprints with >1 blob |
|---|---|
| open field plane A | 5 of 14 (#4 ×2, #6 ×2, #7 ×3, #10 ×3, #13 ×5) |
| open field plane B | 3 of 14 (#5 ×2, #9 ×4, #14 ×2) |
| pain plane A | 1 of 12 (#8 ×2) |
| pain plane B | 1 of 15 (#11 ×2) |

Only one of these is also active — **open-field plane A #10, a 3-way merge**
— so its trace is a blend of three neurons and it is excluded from the usable
set. Open-field plane A #13 (2947 px, 5 blobs, r_extract 0.612) and plane B
#5 (2533 px, 2 blobs, r_extract 0.511) are the two oversized ones; both have
the lowest trace-versus-pixels agreement in their plane, as expected for a
footprint spanning several sources.

### Usable cells for downstream analysis

| session | plane A | plane B | total |
|---|---|---|---|
| open field | #1, #2, #3, #12 | #1, #6, #7, #8, #10, #12, #13 | **11** |
| pain | #1, #3, #4, #5, #6 | #1, #2, #3, #14 | **9** |

Since both sessions image the same two depths (Finding 11), these two sets
can in principle be matched cell by cell, which is what would let a
pin-prick-responsive neuron also be tested for centre versus corner coding.
Matching has not been attempted yet — it needs footprint registration
between the two sessions, and the FOV may have shifted between 15:27 and
15:56.

If more cells are needed, the lever is re-running the open field with a
smaller `avg_cell_radius` for the merged ones specifically, not loosening
`cellfind_min_snr`: the problem there is merging, not sensitivity.

---

## Order of work

1. ~~Split the depths; write `plane_A.tif` and `plane_B.tif` per session~~ —
   done, 6000 → 3000 frames per plane, 27 s, corr(mean A, mean B) = 0.592
2. ~~Decide ΔF/F versus z~~ — `use_zscore_before_extract = false`
3. ~~EXTRACT per plane with `low_snr`, radius 8, find-on-binned /
   trace-on-unbinned~~ — done after Finding 7; plane A 234 s, plane B 209 s
4. ~~Footprint overlay QC~~ — done, Finding 9. Footprints in plane A and
   plane B are at different positions, confirming the depths are separate
   populations and the max-projection of Finding 1 really was merging them
5. ~~Open-field tracking~~ — done, Finding 10
6. ~~Open field: split into planes and run EXTRACT~~ — done, Finding 13;
   14 + 14 cells, 11 usable
7. Manual scoring of the pain video (key 1 = pin prick, key 2 = heat)
8. TDMS timestamp parsing and neuron–behaviour sync — must handle the two
   clocks and the dropped behaviour frames (Finding 12) and emit one
   timestamp vector per plane (Finding 11)
9. Responsive-cell classification on the 9 active cells: pin / heat /
   non-responsive / behaviour-responsive, with raw Ca, z-score and
   event-locked ΔF/F per cell
10. Open-field centre-versus-corner neuron clustering, occupancy-normalised

Hansol Lim — CEA-Ntsr1 mini2p
