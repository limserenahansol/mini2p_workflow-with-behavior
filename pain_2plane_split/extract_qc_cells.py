"""extract_qc_cells.py  -  is each EXTRACT "cell" a cell, or is it background?

THE QUESTION THIS ANSWERS
  EXTRACT's parameters are sensitive and it will happily return background as
  cells. Looking at footprints on the mean image is not enough: a patch of
  neuropil or a vessel also looks like a blob. And looking at the traces is
  not enough either, because at 4.605 Hz with a measured single-frame SNR of
  0.75 a real cell's trace ALSO looks like noise by eye.

  So the test is not "does this look like a cell" but "is this
  distinguishable from background in the same movie". The null distribution
  is measured, not assumed: N control ROIs are cut from the SAME
  motion-corrected movie, with the SAME footprint shapes as the real cells,
  translated to locations that contain no detected cell. Their traces are
  extracted by exactly the same weighted average. Any metric a real cell
  scores on, the controls score on too - the only difference that counts is
  the gap between them.

WHAT IS MEASURED, AND WHY THAT METRIC
  skew    Calcium fluorescence rises fast and decays slowly, so a real trace
          is positively skewed. Shot noise is symmetric, skew ~ 0.
  ac1     Lag-1 autocorrelation. GCaMP decays over ~0.4-1.5 s = 2-7 frames
          here, so consecutive samples of a real trace are correlated. Shot
          noise is white, ac1 ~ 0. This is the most diagnostic single number.
  tau     Lag at which the autocorrelation falls to 1/e, in seconds. Should
          land in the indicator's decay range, not at zero and not at tens of
          seconds (which would be drift or haemodynamics, not spikes).
  snr     (p99 - median) / robust sigma. Robust sigma = 1.4826 * MAD, so a
          few large transients cannot inflate the noise estimate the way an
          SD would.
  events  Peaks in the smoothed trace above median + 3 sigma, with a 1.1 s
          refractory period.
  r_x     Correlation between EXTRACT's own trace for the cell and the plain
          footprint-weighted average of the raw movie. A genuine cell agrees
          with its own pixels; a footprint that EXTRACT fitted to noise does
          not.

  A cell is flagged SUSPECT when its skew AND its ac1 both fall below the
  95th percentile of the control ROIs - i.e. on the two metrics that separate
  signal from shot noise, it is inside the background distribution.

OUTPUTS  ->  <plane folder>\qc_cells\
  qc_cells.csv          the per-cell table
  qc_cells_report.txt   the same, plus the control null and the verdict
  qc_cells_null.png     each metric: cells against the control distribution
  qc_cells_traces.png   per-cell footprint, trace, and autocorrelation

USAGE
  python extract_qc_cells.py                      # both planes
  python extract_qc_cells.py --plane plane_A
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import scipy.sparse as sp
from scipy import stats

SESSION = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
           "15-56-48")
FS = 4.605          # Hz, full (unbinned) frame rate per plane
N_CTRL = 200        # control ROIs per plane
CTRL_MARGIN = 12    # px a control ROI must keep from any detected cell
CHUNK = 200         # frames per read; keeps peak RAM near 180 MB
F0_PCT = 20         # percentile used as F0, for cells and controls alike
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
def load_plane(folder):
    """Footprints (h*w, k) and EXTRACT's dF/F (k, T) for one plane.

    HDF5 reverses MATLAB's dimension order, so what MATLAB saved as
    (h, w, k) = (440, 512, 12) arrives as (12, 512, 440) and what it saved
    as (T, k) arrives as (k, T). Verified by counting nonzeros per slice:
    only axis 0 of spatial_weights has one blob per slice, so axis 0 is the
    cell axis in both arrays. Getting this wrong silently transposes every
    footprint, so it is asserted below rather than assumed.
    """
    mat = os.path.join(folder, "final_analysis_results.mat")
    with h5py.File(mat, "r") as f:
        dff = np.array(f["deltaF_over_F"])            # (k, T)
        S3 = np.array(f["output"]["spatial_weights"])  # (k, w, h)
    if S3.ndim != 3:
        raise SystemExit(f"unexpected spatial_weights shape {S3.shape}")
    k, w, h = S3.shape
    if dff.shape[0] != k:
        raise SystemExit(f"dF/F has {dff.shape[0]} rows but "
                         f"spatial_weights has {k} cells")
    # (k, w, h) -> (k, h, w) -> (h*w, k), C-order ravel of an (h, w) image
    S = S3.transpose(0, 2, 1).reshape(k, h * w).T.astype(np.float32)
    if not np.all((S > 0).sum(0) > 0):
        raise SystemExit("some footprint is empty after reshaping")
    return S, dff, (h, w)


def movie_path(folder, plane):
    p = os.path.join(folder, f"{plane}_1.h5")
    if not os.path.exists(p):
        raise SystemExit(f"motion-corrected movie not found: {p}")
    return p


# ---------------------------------------------------------- control ROIs
def make_controls(S, shape, n_ctrl):
    """Same footprint shapes, moved to places with no detected cell.

    Shape-matched on purpose: area and compactness both affect how much
    shot noise averages out, so a control must be the same size as the cell
    it is standing in for, or the comparison is rigged.
    """
    h, w = shape
    occupied = (S.sum(1).reshape(h, w) > 0).astype(np.uint8)
    import cv2
    keep_out = cv2.dilate(occupied, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * CTRL_MARGIN + 1,) * 2)) > 0

    cols, k = [], S.shape[1]
    tries = 0
    while len(cols) < n_ctrl and tries < n_ctrl * 200:
        tries += 1
        c = int(RNG.integers(k))
        img = S[:, c].reshape(h, w)
        ys, xs = np.nonzero(img)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        patch = img[y0:y1 + 1, x0:x1 + 1]
        ph, pw = patch.shape
        ny = int(RNG.integers(0, h - ph))
        nx = int(RNG.integers(0, w - pw))
        sub = keep_out[ny:ny + ph, nx:nx + pw]
        if sub[patch > 0].any():
            continue
        blank = np.zeros((h, w), np.float32)
        blank[ny:ny + ph, nx:nx + pw] = patch
        cols.append(blank.ravel())
    if len(cols) < n_ctrl:
        print(f"  only {len(cols)} control ROIs fitted (wanted {n_ctrl})")
    return np.array(cols).T if cols else np.zeros((h * w, 0), np.float32)


# ------------------------------------------------------------- extraction
def weighted_traces(mov_h5, W, shape):
    """Footprint-weighted mean of the raw movie for every column of W."""
    h, w = shape
    Wn = W / np.maximum(W.sum(0, keepdims=True), 1e-12)
    Wsp = sp.csr_matrix(Wn.T)                          # (n_roi, h*w)
    out = []
    acc = np.zeros(h * w, np.float64)
    with h5py.File(mov_h5, "r") as f:
        mov = f["mov"]
        T = mov.shape[0]
        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            blk = np.asarray(mov[t0:t1])               # (nt, w, h)
            blk = blk.transpose(0, 2, 1).reshape(t1 - t0, h * w)
            out.append((Wsp @ blk.T).T)                # (nt, n_roi)
            acc += blk.sum(0)
    return np.concatenate(out, 0).T, (acc / T).reshape(h, w)


def anat_contrast(col, mean_img, inner=3, outer=12):
    """How much brighter the footprint is than the ring around it.

    This is the measurement that separates the two reasons a cell can be
    SUSPECT. A footprint sitting on a genuine soma is brighter than its
    surround in the session mean image whether or not the neuron ever fired;
    a footprint EXTRACT fitted to neuropil is not. Without this the report
    can only say "no signal", which would wrongly read as "not a cell".
    """
    import cv2
    m = (col.reshape(mean_img.shape) > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.nan
    d_in = cv2.dilate(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * inner + 1,) * 2))
    d_out = cv2.dilate(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * outer + 1,) * 2))
    ring = (d_out > 0) & (d_in == 0)
    if ring.sum() == 0:
        return np.nan
    a, b = mean_img[m > 0].mean(), mean_img[ring].mean()
    return float((a - b) / abs(b)) if b != 0 else np.nan


def to_dff(F):
    f0 = np.percentile(F, F0_PCT, axis=1, keepdims=True)
    f0 = np.where(np.abs(f0) < 1e-6, 1e-6, f0)
    return (F - f0) / f0


# ---------------------------------------------------------------- metrics
def n_blobs(col, shape, rel=0.3, min_px=20):
    """How many separate blobs are in one footprint.

    Added because the area test missed real merges: plane B cells 11 and 12
    are visibly two somata each in the footprint figure, yet neither exceeds
    3x the median area, so "big footprint" said nothing. Counting connected
    components above 30 % of the footprint's own peak catches a merged pair
    regardless of how big the pair happens to be. A merged footprint mixes
    two neurons' calcium into one trace, which is worse than a missed cell.
    """
    import cv2
    img = col.reshape(shape)
    if img.max() <= 0:
        return 0
    bw = (img > rel * img.max()).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_px))


def autocorr(x, maxlag=30):
    x = x - x.mean()
    n = len(x)
    v = np.dot(x, x)
    if v <= 0:
        return np.zeros(maxlag + 1)
    return np.array([np.dot(x[:n - L], x[L:]) / v for L in range(maxlag + 1)])


def tau_1e(ac, fs):
    """Lag where the autocorrelation first drops below 1/e, interpolated.

    Returns NaN when it never crosses inside the window. The first version
    returned 0.0 both for "decays instantly" and for "never decays", which
    inverts the meaning: cell 1 of plane A came out as tau 0.00 s while its
    lag-1 autocorrelation was 0.81, i.e. the slowest trace in the set was
    reported as the fastest. A NaN here means the timescale is longer than
    the window, which is drift, not an indicator transient.
    """
    thr = 1 / np.e
    below = np.nonzero(ac < thr)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    a, b = ac[i - 1], ac[i]
    frac = (a - thr) / max(a - b, 1e-12)
    return (i - 1 + frac) / fs


def detrend(X, fs=FS, win_s=20.0):
    """Remove slow drift with a running median, keep calcium transients.

    Necessary, not optional: measured on this session the BACKGROUND ROIs
    have a median lag-1 autocorrelation of 0.397, so the movie carries slow
    structure (residual motion, illumination drift, neuropil) that every ROI
    shares. Autocorrelation and skew computed on the raw trace then measure
    that drift instead of calcium, and cells stop separating from background.

    A 20 s window is ~13x longer than the slowest GCaMP decay being looked
    for (0.4-1.5 s), so transients pass through essentially untouched while
    anything slower than ~0.05 Hz is removed.
    """
    from scipy.ndimage import median_filter
    n = int(round(win_s * fs)) | 1          # odd window
    base = median_filter(X, size=(1, n), mode="nearest")
    return X - base, base


def metrics(x, fs=FS):
    med = np.median(x)
    sig = 1.4826 * np.median(np.abs(x - med))
    sig = sig if sig > 0 else 1e-12
    ac = autocorr(x)
    sm = np.convolve(x, np.ones(3) / 3, mode="same")
    thr = med + 3 * sig
    above = sm > thr
    starts = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(np.int8),
                                                    [0]))) == 1)
    refr = max(int(round(1.1 * fs)), 1)
    ev, last = 0, -10 ** 9
    for s in starts:
        if s - last >= refr:
            ev += 1
            last = s
    return dict(skew=float(stats.skew(x)),
                ac1=float(ac[1]),
                tau_s=float(tau_1e(ac, fs)),
                snr=float((np.percentile(x, 99) - med) / sig),
                events=int(ev)), ac


# ------------------------------------------------------------------- main
def run_plane(plane, session, outroot=None, curated=False):
    folder = os.path.join(session, "output_split", plane)
    if curated:
        folder = os.path.join(folder, "curated")
    print(f"\n{'=' * 68}\n{plane}{' (curated)' if curated else ''}\n"
          f"{'=' * 68}")
    S, dff_extract, shape = load_plane(folder)
    h, w = shape
    k = S.shape[1]
    print(f"  FOV {h}x{w}, {k} cells, dF/F {dff_extract.shape}")
    # orientation check: these must match the Row/Col column of QC_report.txt,
    # which MATLAB wrote from the same footprints
    for i in range(min(k, 3)):
        ys, xs = np.nonzero(S[:, i].reshape(h, w))
        wt = S[:, i].reshape(h, w)[ys, xs]
        print(f"    cell {i + 1} centroid row {np.average(ys, weights=wt):.1f}"
              f"  col {np.average(xs, weights=wt):.1f}"
              f"   (compare QC_report.txt)")

    C = make_controls(S, shape, N_CTRL)
    print(f"  {C.shape[1]} control ROIs, >= {CTRL_MARGIN} px from any cell")

    W = np.concatenate([S, C], 1).astype(np.float32)
    print(f"  pulling {W.shape[1]} traces from the movie "
          f"({CHUNK}-frame chunks)...")
    F, mean_img = weighted_traces(movie_path(folder, plane), W, shape)
    dff_all = to_dff(F)
    dff_det, base = detrend(dff_all)
    cell_raw, ctrl_raw = dff_all[:k], dff_all[k:]
    cell_det, ctrl_det = dff_det[:k], dff_det[k:]
    drift = np.ptp(base, axis=1)

    rows, acs = [], []
    for i in range(k):
        m, ac = metrics(cell_det[i])
        m["cell"] = i + 1
        m["r_extract"] = float(np.corrcoef(cell_raw[i],
                                           dff_extract[i])[0, 1])
        m["area_px"] = int((S[:, i] > 0).sum())
        m["drift"] = float(drift[i])
        m["ac1_raw"] = float(autocorr(cell_raw[i], 1)[1])
        m["n_blobs"] = n_blobs(S[:, i], shape)
        m["anat"] = anat_contrast(S[:, i], mean_img)
        rows.append(m)
        acs.append(ac)
    MK = ("skew", "ac1", "tau_s", "snr", "events")
    cm = {kk: np.array([r[kk] for r in rows]) for kk in MK}

    nullm = {kk: [] for kk in MK}
    for j in range(ctrl_det.shape[0]):
        m, _ = metrics(ctrl_det[j])
        for kk in nullm:
            nullm[kk].append(m[kk])
    nullm = {kk: np.array(v) for kk, v in nullm.items()}
    null_ac1_raw = np.array([autocorr(ctrl_raw[j], 1)[1]
                             for j in range(ctrl_raw.shape[0])])
    null_anat = np.array([anat_contrast(C[:, j], mean_img)
                          for j in range(C.shape[1])])
    anat_cut = float(np.nanpercentile(null_anat, 95))

    p95 = {kk: float(np.nanpercentile(nullm[kk], 95)) for kk in nullm}

    # Joint test, calibrated on the controls themselves.
    #
    # Requiring skew AND ac1 to each clear the null's 95th percentile throws
    # power away: plane A cell 2 sat at the 84th percentile on skew and the
    # 94th on ac1 and was called SUSPECT, although being that high on both
    # at once is far less likely than either alone. So the two one-sided
    # empirical p-values are combined with Fisher's method and the combined
    # statistic is compared against the SAME statistic computed for every
    # control ROI. That calibration is what makes it honest: whatever
    # correlation exists between skew and ac1 is present in the null too,
    # and the cut lands at a measured 5 % false-positive rate on background.
    def emp_p(null, x):
        return (1.0 + np.sum(null >= x)) / (len(null) + 1.0)

    def fisher(sk, a1):
        return -2.0 * (np.log(emp_p(nullm["skew"], sk))
                       + np.log(emp_p(nullm["ac1"], a1)))

    null_X = np.array([fisher(s, a) for s, a
                       in zip(nullm["skew"], nullm["ac1"])])
    X_cut = float(np.percentile(null_X, 95))
    cell_X = np.array([fisher(s, a) for s, a in zip(cm["skew"], cm["ac1"])])
    # The verdict is taken from p_joint, not from X against X_cut: the two
    # disagree by one control ROI right at the boundary (plane B cell 7 came
    # out at p_joint 0.055 yet on the "cell" side of X_cut), and a table whose
    # verdict column contradicts its own p column is worse than either rule.
    p_joint = (1.0 + np.array([np.sum(null_X >= X) for X in cell_X])) \
        / (len(null_X) + 1.0)
    suspect = p_joint > 0.05

    med_area = float(np.median([r["area_px"] for r in rows]))
    for r, s, X, pj in zip(rows, suspect, cell_X, p_joint):
        # Three outcomes, not two. "No calcium signal" and "not a neuron"
        # are different claims and the anatomy decides between them.
        if not s:
            r["verdict"] = "active"
        elif np.isfinite(r["anat"]) and r["anat"] > anat_cut:
            r["verdict"] = "silent"
        else:
            r["verdict"] = "SUSPECT"
        r["fisher"] = float(X)
        r["p_joint"] = float(pj)
        r["pct_anat"] = float(100 * np.nanmean(null_anat < r["anat"]))
        r["pct_skew"] = float(100 * (nullm["skew"] < r["skew"]).mean())
        r["pct_ac1"] = float(100 * (nullm["ac1"] < r["ac1"]).mean())
        # a footprint several times the typical size is usually two merged
        # cells or a vessel, whatever its trace looks like
        r["big_footprint"] = int(r["area_px"] > 3 * med_area)

    # ---- report ----
    L = [f"===== EXTRACT cell QC: {plane} =====",
         f"FOV {h}x{w}   cells {k}   frames {dff_extract.shape[1]} "
         f"({dff_extract.shape[1] / FS:.0f} s at {FS} Hz)",
         f"control ROIs {C.shape[1]}, shape-matched, >= {CTRL_MARGIN} px "
         f"from any cell",
         "",
         "All metrics are computed on DETRENDED dF/F (running median, 20 s "
         "window),",
         "because the raw traces carry drift that every ROI shares:",
         f"  background lag-1 autocorrelation, raw       "
         f"{np.median(null_ac1_raw):.3f} (median)",
         f"  background lag-1 autocorrelation, detrended "
         f"{np.median(nullm['ac1']):.3f} (median)",
         "Measured on the raw trace, background looks as correlated as a "
         "cell, so",
         "the comparison below would have no power.",
         "",
         "control null (background ROIs from the same movie, detrended)",
         f"  {'metric':8s} {'median':>8s} {'p95':>8s} {'max':>8s}"]
    for kk in MK:
        L.append(f"  {kk:8s} {np.nanmedian(nullm[kk]):8.3f} {p95[kk]:8.3f} "
                 f"{np.nanmax(nullm[kk]):8.3f}")
    L.append(f"  {'anat':8s} {np.nanmedian(null_anat):8.3f} "
             f"{anat_cut:8.3f} {np.nanmax(null_anat):8.3f}   "
             f"(brightness of the footprint over its surrounding ring, "
             f"session mean image)")
    L += ["",
          "verdicts",
          "  active   calcium signal separable from background "
          "(p_joint <= 0.05)",
          "  silent   no calcium signal, but the footprint is brighter than "
          "its",
          "           surround above the background p95 - a real soma that "
          "did not",
          "           fire enough in 651 s to be detected",
          "  SUSPECT  neither: no signal and no anatomical contrast",
          "",
          "per cell   (pct = percentile within the control null)",
          "  tau_s = NaN means the autocorrelation never fell to 1/e within "
          "6.5 s,",
          "  i.e. a timescale too slow to be an indicator transient.",
          f"  {'ID':>3s} {'area':>5s} {'skew':>6s} {'pct':>4s} {'ac1':>6s} "
          f"{'pct':>4s} {'tau_s':>6s} {'snr':>5s} {'ev':>3s} "
          f"{'drift':>6s} {'r_extr':>7s} {'anat':>6s} {'pct':>4s} "
          f"{'p_joint':>8s}  verdict"]
    for r in rows:
        L.append(f"  {r['cell']:3d} {r['area_px']:5d} {r['skew']:6.2f} "
                 f"{r['pct_skew']:4.0f} {r['ac1']:6.3f} {r['pct_ac1']:4.0f} "
                 f"{r['tau_s']:6.2f} {r['snr']:5.1f} {r['events']:3d} "
                 f"{r['drift']:6.2f} {r['r_extract']:7.3f} "
                 f"{r['anat']:6.3f} {r['pct_anat']:4.0f} "
                 f"{r['p_joint']:8.4f}  {r['verdict']}"
                 + ("   BIG FOOTPRINT" if r["big_footprint"] else "")
                 + (f"   {r['n_blobs']} BLOBS" if r["n_blobs"] > 1 else ""))
    # The number that actually matters downstream is the intersection, not
    # each flag on its own: a cell can be "active" and still be a merge of
    # three somata, in which case its trace is a blend of three neurons and
    # using it is worse than dropping it. Open-field plane A #10 is exactly
    # that case, so the cross-tabulation is printed rather than left to the
    # reader to do by eye.
    for r in rows:
        r["usable"] = int(r["verdict"] == "active"
                          and r["n_blobs"] <= 1
                          and not r["big_footprint"])
    usable = [r["cell"] for r in rows if r["usable"]]
    dropped_active = [(r["cell"],
                       "merge x%d" % r["n_blobs"] if r["n_blobs"] > 1
                       else "big footprint")
                      for r in rows if r["verdict"] == "active"
                      and not r["usable"]]
    n_act = sum(r["verdict"] == "active" for r in rows)
    n_sil = sum(r["verdict"] == "silent" for r in rows)
    n_bad = sum(r["verdict"] == "SUSPECT" for r in rows)
    big = [r["cell"] for r in rows if r["big_footprint"]]
    merged = [r["cell"] for r in rows if r["n_blobs"] > 1]
    L += ["",
          f"Joint test: Fisher combination of the two empirical p-values,",
          f"calibrated against the same statistic computed for each of the "
          f"{len(null_X)} control",
          f"ROIs (95th percentile of the null = {X_cut:.2f}). A cell is kept "
          f"at p_joint <= 0.05,",
          f"so by construction 5 % of background ROIs would pass.",
          "",
          f"VERDICT: {n_act} active, {n_sil} silent, {n_bad} SUSPECT, "
          f"of {k} detected.",
          f"         The {n_sil} silent ones are real somata with nothing to "
          f"classify; the",
          f"         {n_bad} SUSPECT should be dropped.",
          "",
          f"USABLE for response classification: {len(usable)} cells "
          f"{usable}",
          f"         (active AND a single-blob footprint AND not oversized)",
          (f"         active but excluded: "
           + ", ".join(f"#{c} ({why})" for c, why in dropped_active)
           + " - the trace is a blend of more than one neuron"
           if dropped_active else
           "         every active cell has a clean single-blob footprint"),
          (f"         footprint larger than 3x the median ({med_area:.0f} px)"
           f": cells {big} - check for a vessel."
           if big else
           f"         no footprint exceeds 3x the median area "
           f"({med_area:.0f} px)."),
          (f"         footprint contains more than one blob: cells {merged}"
           f" - two neurons mixed into one trace, exclude or re-run with a"
           f" smaller radius."
           if merged else
           "         every footprint is a single blob."),
          "",
          "The silent/SUSPECT split is what keeps this from being read as",
          "'EXTRACT detected background'. A neuron that never fired in 651 s",
          "has no calcium signal for any temporal metric to find, so an",
          "absence of signal on its own says nothing about whether the",
          "footprint is a cell. Still read the footprint figure alongside",
          "this table."]
    txt = "\n".join(L)
    print(txt)

    outdir = os.path.join(outroot or folder, "qc_cells")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qc_cells_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "qc_cells.csv"),
                              index=False)

    plot_null(cm, nullm, p95, suspect, plane, outdir)
    plot_cells(S, shape, cell_raw, cell_det, base[:k], acs, rows, plane,
               outdir)
    print(f"\n  wrote {outdir}")
    return rows, nullm


def plot_null(cm, nullm, p95, suspect, plane, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ("skew", "ac1", "tau_s", "snr", "events")
    fig, ax = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.4))
    for a, kk in zip(ax, keys):
        nn = nullm[kk][np.isfinite(nullm[kk])]
        a.hist(nn, bins=30, color="#B0B0B0", label="background")
        a.axvline(p95[kk], color="k", ls="--", lw=1, label="null p95")
        for v, s in zip(cm[kk], suspect):
            if np.isfinite(v):
                a.axvline(v, color="#C1272D" if s else "#1C6E8C", lw=1.4)
        nan_n = int(np.sum(~np.isfinite(nullm[kk])))
        a.set_title(kk + (f"\n({nan_n} null NaN)" if nan_n else ""))
        a.set_xlabel(kk)
    ax[0].set_ylabel("control ROIs")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{plane}: cells (blue) vs background ROIs (grey); "
                 f"red = SUSPECT", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "qc_cells_null.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def plot_cells(S, shape, raw, det, base, acs, rows, plane, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = shape
    k = len(rows)
    fig, ax = plt.subplots(k, 3, figsize=(13, 1.45 * k),
                           gridspec_kw=dict(width_ratios=[1, 5, 1.4]))
    ax = np.atleast_2d(ax)
    t = np.arange(raw.shape[1]) / FS
    for i, r in enumerate(rows):
        img = S[:, i].reshape(h, w)
        ys, xs = np.nonzero(img)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = max(cy - 30, 0), min(cy + 30, h)
        x0, x1 = max(cx - 30, 0), min(cx + 30, w)
        ax[i, 0].imshow(img[y0:y1, x0:x1], cmap="inferno")
        ax[i, 0].set_xticks([]), ax[i, 0].set_yticks([])
        ax[i, 0].set_ylabel(f"#{r['cell']}", fontsize=8)
        col = {"active": "#1C6E8C", "silent": "#B8860B",
               "SUSPECT": "#C1272D"}[r["verdict"]]
        ax[i, 1].plot(t, raw[i], lw=.3, color="#C8C8C8")
        ax[i, 1].plot(t, base[i], lw=.9, color="#E08214")
        ax[i, 1].plot(t, det[i] + np.median(raw[i]), lw=.35, color=col)
        ax[i, 1].set_xlim(0, t[-1])
        ax[i, 1].set_ylabel("dF/F", fontsize=7)
        tau = "  tau n/a" if not np.isfinite(r["tau_s"]) \
            else f"  tau {r['tau_s']:.2f}s"
        ax[i, 1].text(.005, .96,
                      f"skew {r['skew']:.2f} (p{r['pct_skew']:.0f})  "
                      f"ac1 {r['ac1']:.2f} (p{r['pct_ac1']:.0f})" + tau +
                      f"  snr {r['snr']:.1f}  ev {r['events']}  "
                      f"drift {r['drift']:.2f}  "
                      f"r_extract {r['r_extract']:.2f}  {r['verdict']}",
                      transform=ax[i, 1].transAxes, va="top", fontsize=7,
                      color=col)
        ax[i, 2].plot(np.arange(len(acs[i])) / FS, acs[i], lw=1, color=col)
        ax[i, 2].axhline(1 / np.e, color="k", ls=":", lw=.8)
        ax[i, 2].axhline(0, color="k", lw=.5)
        ax[i, 2].set_ylim(-.2, 1)
        if i < k - 1:
            ax[i, 1].set_xticks([]), ax[i, 2].set_xticks([])
    ax[-1, 1].set_xlabel("time (s)")
    ax[-1, 2].set_xlabel("lag (s)")
    fig.suptitle(f"{plane}: per-cell dF/F - grey = raw, orange = 20 s "
                 f"running median, colour = detrended (metrics use this); "
                 f"right = autocorrelation of the detrended trace",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .99])
    fig.savefig(os.path.join(outdir, "qc_cells_traces.png"), dpi=135,
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=SESSION)
    ap.add_argument("--plane", action="append",
                    help="plane_A / plane_B (default: both)")
    ap.add_argument("--curated", action="store_true",
                    help="run on <plane>\\curated\\ instead of the raw "
                         "detection")
    a = ap.parse_args()
    for p in (a.plane or ["plane_A", "plane_B"]):
        run_plane(p, a.session, curated=a.curated)


if __name__ == "__main__":
    main()
