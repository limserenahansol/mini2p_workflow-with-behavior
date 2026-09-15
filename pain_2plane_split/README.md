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
| 11 | **the 31-cell union — every later step reads this** | `transfer_footprints.py`, loaded by `union_data.py` | `<each session>\output_split\union\plane_*\union_traces.mat` |
| 12 | centre versus corner cells | `openfield_place_cells.py` | `output_split\place\` |
| 13 | one row per neuron, both sessions | `linked_cells_report.py` | `<pain session>\output_split\match\` |
| 14 | raw Ca / dF/F / z per cell | `pain_cell_traces.py` | `<plane>\curated\pain_traces.*` |
| 15 | the deck | `make_pipeline_ppt.py` | `CEANTSR1_pipeline_*.pptx` |
| 16 | human key-press delay, measured | `tap_delay.py` | `<pain>\output_split\events\tap_delay.*` |
| 17 | mouse movement and freezing from both cameras | `pain_motion.py` | `<pain>\output_split\events\fig9_movement.png` |
| 18 | per-delivery responses — **withdrawn**, kept for the record | `pain_responsive.py` | `events\responsive_cells.csv` |
| 19 | 5 windows × 3 statistics × 2 stimuli, plus a joint model | `pain_search.py` | `events\search_grid.csv` |
| 20 | the same tests at every threshold | `loosen.py` | `events\loosen_ladder.csv` |
| 21 | **the nine session measurements the argument rests on** | `pain_audit.py` | `events\pain_audit.{txt,csv}` |
| 22 | **THE final pain classification** | `pain_final.py` | `events\pain_final.{csv,txt}`, `fig29_pain_final.png` |
| 23 | centre/corner by rate, not count | `of_zone_rates.py`, `of_both_methods.py`, `of_cell_rows.py` | `<open field>\atlas\` |
| 24 | one row per neuron, both sessions | `cell_atlas.py --level final` | `<session>\output_split\atlas\` |
| 25 | the result figures | `results_figures.py --level final`, `summary_figures.py --level final` | `<pain>\output_split\results\` |
| 26 | the two decks | `make_processing_ppt.py`, `make_results_ppt.py` | `CEANTSR1_{processing,results}_*.pptx` |

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
| `fig5_separability.py` | why 8 footprints were dropped from the union, measured both ways | `<pain>\match\fig5_separability.png` |
| `make_visual_ppt.py` | the four sets as one deck, tall figures cut into slide-shaped bands | `CEANTSR1_visualisation_*.pptx` |
| `make_pipeline_ppt.py` | the narrative report: data, pipeline, QC, results | `CEANTSR1_pipeline_*.pptx` |

Four design rules these follow, because each one caught a mistake:

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

**Measure the quantity that decides the outcome.** Two footprints being
"different cells" was checked by the distance between their centres, when what
actually decides whether they can each have a trace is their correlation as
regressors. Most of the duplicate pairs were 1.5–7 px apart and passed every
position check. See the separability section below.

**Never let smoothing invent data.** The first place-map version blurred the
occupancy-normalised rate maps with a permissive weight cut, which painted
colour into bins the mouse never entered. The occupancy mask is now
re-applied after smoothing.

---

## The 31-cell union is the unit of analysis

Every cross-session question used to be asked of the neurons EXTRACT happened
to detect *twice*. That is a detection limit, not an anatomical one:
`match_audit.py` found a soma at the location of **25 of 27** pain cells and
**20 of 28** open-field cells in the other session's mean image.

`transfer_footprints.py` therefore registers the union of both sessions'
footprints and solves all of them jointly on **both** movies, so detection has
to succeed once per neuron instead of twice. Result: **31 neurons** (plane A
13, plane B 18), each with a trace in both sessions under one shared id, so
`A1` in the pain session and `A1` in the open field are the same neuron by
construction. 26 of the 31 sit on a soma in both sessions; 17 were detected
independently twice; 21 are active in at least one session, 8 in both.

`union_data.py` is the only loader. `usable()` filters on `anat >= 1` image SD
— a transferred footprint is a hypothesis, and below 1 SD it landed on nothing
and its trace is background whatever it looks like. It does **not** filter on
activity: a neuron that was quiet is a valid row, that *is* the measurement.

This changed the answers, which is the point: place coding went from 9 cells
tested to 28.

### Two footprints on one soma cannot both have a trace

The union is not both sets stacked. The first version added every open-field
footprint that was not a shape-confirmed match — and an open-field footprint
can sit *on top of* a pain footprint and still fail the shape test, so the
same soma entered the design matrix twice, 8 times across the two planes.

Traces come from one joint least-squares solve, which has no way to choose
between two near-identical regressors: it pays for the soma's transient with
a positive trace on one copy and a negative trace on the other. Re-running the
solve both ways measures it — 7 of the 8 pairs came out anti-correlated at
r = −0.25 to −0.83, one cell's raw F was negative in *every* frame of both
sessions, and cond(SᵀS) was 49 instead of 5.

It also changed a result. An earlier run of this pipeline reported 6
centre-preferring and 2 corner-preferring cells. Both "corner" cells were the
negative halves of centre cells (r = −0.82 and −0.84) — centre minus corner is
negative for a sign-flipped centre cell by construction. The corrected answer
is **2 centre, 0 corner**, and they are the same two cells with the same
contrasts (+0.899, +0.521) that the per-session analysis found before the
union existed.

Overlap is now measured between footprints **as regressors** (`gram_matrix`),
because that is what decides whether least squares can separate them — not the
distance between their centres, which was 1.5–7 px for most of these pairs and
so passed every position check. The threshold is calibrated, not chosen: over
361 footprint pairs *within* a curated session — neurons EXTRACT and curation
kept as distinct — the largest correlation is **0.167**, so the cut at 0.30 is
far above anything this data calls two cells and far below every pair dropped.
`transfer_footprints.py` refuses to save a union that still contains an
inseparable pair, and `fig5_separability.py` shows the whole thing.

Two further bugs surfaced on the way:

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
| union neurons | 13 | 18 | **31** |
| on a soma in both sessions | 10 | 16 | 26 |
| detected independently twice | 6 | 11 | 17 |
| active in the pain session | 6 | 8 | 14 |
| active in the open field | 7 | 8 | 15 |
| dropped as inseparable duplicates | 5 | 3 | 8 |

**Open field.** Floor is 454 × 516 px = 11.9 % of the FOV; everything outside
it is ignored, which matters because "largest dark blob in the frame" tracked
the dark background for 1632 of 1632 sampled frames. Detection 99.96 % of
8159 frames. Occupancy: centre 5.6 %, corner 41.4 %, edge 53.0 %.

**Centre versus corner**, on the 31-cell union: 28 usable cells; **2 prefer
the centre** (A3 +0.899, A7 +0.521, both q = 0.0070) and **none prefer the
corners**; 26 are not selective. Tested against a circular-shift null (2000
surrogates), which keeps the trace's autocorrelation and the animal's
occupancy — a t-test across frames would treat each frame as independent and
is not valid for calcium. 0 of 28 carry significant spatial information; 1 of
28 is speed-correlated and no cell is both, so the zone effect is not speed in
disguise. Centre occupancy is 5.6 %, so the centre mean rests on 83 imaging
frames against 623 in the corners — a centre effect is the harder one to
detect, and the absence of corner cells is weak evidence.

These are the same two cells, with the same contrasts, that the per-session
analysis found from 9 usable cells. Tripling the number tested did not add
any: see the separability section above for the run that appeared to, and why
it was wrong.

**Same neuron across sessions.** Both sessions image the same two depths, and
each depth matches itself across the two sessions at image correlation
r ≈ 0.87 against 0.33–0.44 for the wrong depth. Registration is a direct
correlation search over shifts; phase correlation failed here, returning
contradictory shifts between depths and no pairs at all. For *detection*
pairs, position alone is not enough in a dense plane, so a pair must also
agree in footprint shape (r ≥ 0.5): 16 of 22 position pairs pass both, against
~3 expected by chance. One more pair sits 12.2 px apart — outside the 8 px
radius — but agrees in shape at r = 0.86 with 96 % shared support, and the
separability check picks it up, giving 17 neurons detected twice. Those 17
plus the 14 detected in only one session are the 31-neuron union above.

**Zone-selective cells followable into the pain session:** A3 (centre, pain
active) and A7 (centre, pain quiet) — both of them, with footprints on a soma
in both sessions (anat 6.0 and 4.3 image SDs in the pain session).

**Event locking.** Every trace carries a time vector on the same base as the
behaviour cameras, so a scored stimulus time indexes straight into it. Frame
index ÷ 25 is *not* that time (see pitfall 4); the per-frame `Ref Time`
mapping is used, which differs by up to 0.78 s by the end of the pain session.

---

## Pain assay — what the session can and cannot answer

Behaviour was scored by hand on 2026-09-14: 53 pin pricks, 34 heat, 76 paw
withdrawals, 50 flinches, 5 escapes. Licking/biting and guarding were never
scored, paw attending once.

**Four earlier answers were withdrawn**, and the reason is the session design
rather than the neurons. `pain_audit.py` recomputes all nine numbers below
into `pain_audit.txt`, and every slide and figure reads them from
`pain_audit.csv` — they were typed in from memory once and five of them had
drifted.

| # | measurement | value | what it rules out |
|---|---|---|---|
| 1 | median gap between pin pricks | **3.16 s** | no clean pre-window exists |
| 1 | pin trials with another delivery inside [–2,0] s | **28 %** | — and 47 % for [–4,–2], 45 % for [–6,–4], 30 % for [–8,–6] |
| 2 | lag of the population peak | **–1 s**, *before* the key press | a pre-window baseline starts at the top of the response |
| 3 | pin effect with a [–2,0] s baseline | –0.047 z, p = 0.0044 — **suppression** | … |
| 3 | the same traces with [–6,–4] s | +0.120 z, p = 0.00088 — **excitation** | the sign was a property of the window |
| 4 | jittered events landing within 1 s of a real one | **31 %** | an event-jitter null cannot be built |
| 5 | pin deliveries inside the heat block | **0** (pin 13–204 s and 519–586 s, heat 269–518 s) | "pin vs heat" is also "early vs middle" |
| 6 | Jaccard overlap of the withdrawal and flinch UP lists | **0.90** (pin–withdrawal 0.70) | pin, withdrawal and flinch are one finding |
| 7 | a global stimulus-free reference | 41 % of the session but **+97 s late**, covering **11 %** of the pin block | the reference must be local |
| 8 | pin trials with 6 s of clear time | **12 of 53**, resolving 0.43 z — larger than the biggest effect present (0.41 z) | the clean subset cannot substitute |
| 9 | escape episodes | 5, 3.9 s = **18 imaging frames** | only escape is testable, and only as a hint |

### The final classification

`pain_final.py`, exploratory FDR q ≤ 0.20 across the 28 interpretable cells,
circular-shift null (2000 shifts, exact by FFT).

**TIER 1 — stimulation-modulated, robust.** Is the cell different during the
10 s after a delivery than during the quiet time, referenced to the
stimulus-free frames **within ±60 s of that same delivery** so a
session-long drift cancels? **2 UP: A1 (+0.41 z, q = 0.007) and A4 (+0.32 z,
q = 0.007); 0 DOWN.** The next cell sits at q = 0.30, so these two are not a
cut through a continuum. It does *not* mean pin-specific — handling, arousal
and movement are all inside "stimulation going on".

**TIER 2 — modality-selective: WITHDRAWN.** Not determinable here
(measurement 5). The columns stay in the CSV as `t2_*`, marked invalid; the
full-session version flagged 14 of 28 cells at 0.05–0.13 z, which is what a
slow drift looks like.

**TIER 2b — behaviour-linked, weak.** Escape inside versus outside: **6 UP
(A12, B7, B13, B14, B15, B16)** on 18 imaging frames. Reported with its n so
it reads as a hint.

**Not claimed:** per-delivery phasic responses to pin or heat, any sentence of
the form "N cells respond to pin", and pin-versus-heat selectivity. The first
is set by the window and not by the neurons — 0, 6, 7 and 8 cells across four
defensible versions, sharing only A9 between the two best.

**What does hold without any window choice:** 26 of 28 cells raise their mean
dF/F from the 30 s before the first delivery to the 30 s after the last
(Wilcoxon signed-rank, p < 0.0001). Sensitisation, arousal and slow imaging
drift would all look like this; separating them needs a no-stimulus control
session.

**Across the two sessions:** no overlap. Neither centre cell (A3, A7) has a
pain verdict and neither TIER 1 cell is zone-selective. With 2 centre cells
and 8 cells carrying a pain verdict out of 28, chance gives no overlap too —
this describes these cells, not a population.

### Two mistakes worth keeping visible

**One figure, two references.** `fig21_group_means.png` subtracted a [–2,0] s
pre-window on its five event panels and the stimulus-free time on its two
epoch panels. Because the population peaks at –1 s, the five left panels read
"pin: no response (n = 28)" while the two right panels — the correct ones —
already showed A1 and A4 UP. The headline was read off the wrong panels. Every
panel now uses the local stimulus-free reference, and the groups are read from
`pain_final.csv` so a figure cannot disagree with the table.

**A panel that re-smuggled a withdrawn claim.** `fig29_pain_final.png` had a
middle panel "TIER 1, pin deliveries only" with its own significance
colouring, which marked A11 — while the verdict list on the same figure called
A11 not modulated. It now shows pin-only and heat-only side by side with
*nothing* marked significant, and says so.

### What the next session needs

1. **Space the pin pricks 15–20 s apart.** 20 well-spaced pricks carry more
   information than the 53 here, because all 20 would have a clean pre-window
   and resolve about 0.25 z.
2. **Interleave pin and heat instead of blocking them.** That makes
   "pin versus heat" a within-time comparison, which is the test TIER 2 wanted.
3. A no-stimulus control session of the same length, and a 60–90 s
   stimulus-free block at each end so the reference does not have to be carved
   out of the gaps.

---

## Requirements

MATLAB R2024b with the Image Processing Toolbox, EXTRACT-public, NoRMCorre.
Python 3.13 with numpy, scipy, pandas, opencv-python, h5py, matplotlib,
npTDMS, python-pptx, Pillow.

Session paths are set at the top of each script.
