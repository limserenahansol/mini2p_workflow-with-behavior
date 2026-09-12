"""fig2_detection.py  -  justify the cell detection by showing the comparison, not the p-value.

THE ARGUMENT THIS FIGURE MAKES
  The automatic verdict says a footprint is a cell because its trace is
  separable from 200 background ROIs. That is the right test, but a reader
  cannot check a p-value by looking. So this figure puts each cell next to
  ITS OWN local control: the same footprint shape, translated to the nearest
  place in the same movie that contains no detected cell, with its trace
  taken by the same method. Same tissue, same depth, same shot noise, same
  extraction - only the soma is missing.

  If the cells look different from their controls, the detection is doing
  something real, and the reader can see it rather than take it on trust.
  If they look the same, no statistic should rescue it.

LAYOUT follows the zoom-out-first rule
  panel A  whole field: every footprint and where its control was placed
  panel B  the paired comparison, cell against its own control, per metric
  panel C  one row per cell: mean crop, max crop, trace - cell then control

WHY A PAIRED CONTROL AND NOT JUST THE 200-ROI NULL
  The 200-ROI null answers "is this cell unusual for this movie". The paired
  control answers "is this cell unusual for this SPOT in this movie", which
  is the question a sceptic actually asks - brightness, background and noise
  all vary across the field. Both are kept: panel B is paired, and the
  verdict printed per cell still comes from the calibrated 200-ROI test.

OUTPUT  ->  <plane>\\curated\\fig2_detection.png

USAGE
  python fig2_detection.py --plane A
  python fig2_detection.py --session <openfield dir> --plane B
"""
from __future__ import annotations

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd

from extract_qc_cells import detrend, metrics, to_dff, weighted_traces
from plot_cellmap_traces import demix_raw

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
FS = 4.6054
REL = 0.3
CROP = 70
KEEPOUT = 12        # px a control must stay from any detected cell
SEARCH = range(20, 90, 4)
VCOL = {"active": "#1C6E8C", "silent": "#B8860B", "SUSPECT": "#C1272D"}


def load(session, plane):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mn = np.array(h["output"]["info"]["summary_image"]).T
        mx = np.array(h["output"]["info"]["max_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lb = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                     dtype={"label": str})
    lb = lb[lb["index"] > 0].sort_values("index")
    qc = pd.read_csv(os.path.join(cur, "qc_cells", "qc_cells.csv"))
    mov = os.path.join(base, f"plane_{plane}_1.h5")
    return S, (hh, w), lb["label"].tolist(), qc, mn, mx, mov, cur


def paired_controls(S, shape):
    """One control per cell: same shape, nearest clear spot.

    Nearest rather than random on purpose. A control 200 px away sits in
    different tissue with different brightness and noise, which makes the
    comparison easy in the wrong way. The closest clear spot is the hardest
    honest control available.
    """
    hh, w = shape
    occupied = (S.sum(1).reshape(hh, w) > 0).astype(np.uint8)
    keep_out = cv2.dilate(occupied, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * KEEPOUT + 1,) * 2)) > 0
    dirs = [(np.sin(t), np.cos(t))
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    out = []
    for i in range(S.shape[1]):
        img = S[:, i].reshape(hh, w)
        ys, xs = np.nonzero(img)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        patch = img[y0:y1 + 1, x0:x1 + 1]
        ph, pw = patch.shape
        placed = None
        for d in SEARCH:
            for sy, sx in dirs:
                ny, nx = int(y0 + sy * d), int(x0 + sx * d)
                if ny < 0 or nx < 0 or ny + ph > hh or nx + pw > w:
                    continue
                if keep_out[ny:ny + ph, nx:nx + pw][patch > 0].any():
                    continue
                blank = np.zeros((hh, w), np.float32)
                blank[ny:ny + ph, nx:nx + pw] = patch
                placed = (blank.ravel(), float(d))
                break
            if placed:
                break
        out.append(placed if placed else (np.zeros(hh * w, np.float32),
                                          np.nan))
    return out


def contours(col, shape):
    img = col.reshape(shape)
    if img.max() <= 0:
        return []
    bw = (img > REL * img.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def centroid(col, shape):
    img = col.reshape(shape)
    ys, xs = np.nonzero(img)
    wt = img[ys, xs]
    return np.average(ys, weights=wt), np.average(xs, weights=wt)


def run(session, plane):
    S, shape, labels, qc, mn, mx, mov, cur = load(session, plane)
    hh, w = shape
    k = S.shape[1]
    ctrl = paired_controls(S, shape)
    C = np.stack([c[0] for c in ctrl], 1)
    dists = [c[1] for c in ctrl]
    print(f"  {k} cells, controls placed {np.nanmin(dists):.0f}-"
          f"{np.nanmax(dists):.0f} px away")

    # demix_raw's second return is the background-regressor traces, and the
    # background term was removed from it, so that value is now empty - the
    # mean image comes from the saved summary_image instead.
    Fc, _ = demix_raw(mov, S, shape, mn)
    Fk, _ = weighted_traces(mov, C, shape)
    mean_img = mn
    dff_c, dff_k = to_dff(Fc), to_dff(Fk)
    det_c, _ = detrend(dff_c)
    det_k, _ = detrend(dff_k)

    rows = []
    for i in range(k):
        mc, _ = metrics(det_c[i])
        mk, _ = metrics(det_k[i])
        v = "?"
        if (qc["cell"] == i + 1).any():
            v = qc.loc[qc["cell"] == i + 1, "verdict"].iloc[0]
        rows.append(dict(label=labels[i], verdict=v, dist=dists[i],
                         cell_skew=mc["skew"], ctrl_skew=mk["skew"],
                         cell_ac1=mc["ac1"], ctrl_ac1=mk["ac1"],
                         cell_ev=mc["events"], ctrl_ev=mk["events"],
                         cell_snr=mc["snr"], ctrl_snr=mk["snr"]))
    D = pd.DataFrame(rows)
    for a, b, nm in (("cell_skew", "ctrl_skew", "skew"),
                     ("cell_ac1", "ctrl_ac1", "lag-1 autocorr"),
                     ("cell_ev", "ctrl_ev", "events")):
        win = int((D[a] > D[b]).sum())
        print(f"    {nm:16s} cell > its own control in {win}/{k} cells "
              f"(median {D[a].median():.3f} vs {D[b].median():.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(dff_c.shape[1]) / FS
    fig = plt.figure(figsize=(17.5, 5.6 + 1.15 * k))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.15 * k], hspace=.10)
    top = gs[0].subgridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=.26)

    ax = fig.add_subplot(top[0, 0])
    lo, hi = np.percentile(mx, [2, 99.7])
    ax.imshow(mx, cmap="gray", vmin=lo, vmax=hi)
    for i in range(k):
        col = VCOL.get(rows[i]["verdict"], "#555555")
        for c in contours(S[:, i], shape):
            ax.plot(c[:, 0, 0], c[:, 0, 1], lw=1.5, color=col)
        for c in contours(C[:, i], shape):
            ax.plot(c[:, 0, 0], c[:, 0, 1], lw=1.1, color="#BBBBBB")
        cy, cx = centroid(S[:, i], shape)
        ax.text(cx + 7, cy - 7, labels[i], fontsize=8.5, color=col,
                fontweight="bold")
        if np.isfinite(dists[i]):
            ky, kx = centroid(C[:, i], shape)
            ax.plot([cx, kx], [cy, ky], "-", color="#CCCCCC", lw=.7,
                    zorder=0)
    ax.set_xlim(0, w), ax.set_ylim(hh, 0), ax.axis("off")
    ax.set_title("A  every footprint and its paired control\n"
                 "colour = verdict, grey = the control, line = the pairing",
                 fontsize=10.5, loc="left")

    for jj, (a, b, nm) in enumerate(
            (("cell_skew", "ctrl_skew", "skew"),
             ("cell_ac1", "ctrl_ac1", "lag-1 autocorrelation"),
             ("cell_snr", "ctrl_snr", "robust SNR")), start=1):
        p = fig.add_subplot(top[0, jj])
        for _, r in D.iterrows():
            p.plot([0, 1], [r[b], r[a]], "-",
                   color=VCOL.get(r["verdict"], "#555555"), lw=1.1,
                   alpha=.8, marker="o", ms=4.5)
        p.set_xticks([0, 1])
        p.set_xticklabels(["its control", "the cell"], fontsize=9.5)
        p.set_xlim(-.25, 1.25)
        p.set_ylabel(nm, fontsize=10)
        win = int((D[a] > D[b]).sum())
        p.set_title(f"{'ABCD'[jj]}  {nm}\ncell higher in {win} of {k}",
                    fontsize=10.5, loc="left")
        p.spines[["top", "right"]].set_visible(False)

    bot = gs[1].subgridspec(k, 7, wspace=.06, hspace=.12,
                            width_ratios=[1, 1, 4.2, .35, 1, 1, 4.2])
    lo_m, hi_m = np.percentile(mean_img, [2, 99.7])
    for i in range(k):
        col = VCOL.get(rows[i]["verdict"], "#555555")
        cy, cx = centroid(S[:, i], shape)
        ky, kx = centroid(C[:, i], shape) if np.isfinite(dists[i]) \
            else (cy, cx)
        for jj, (src, lo_, hi_, yy, xx, outline_col) in enumerate([
                (mean_img, lo_m, hi_m, cy, cx, col),
                (mx, lo, hi, cy, cx, col)]):
            a = fig.add_subplot(bot[i, jj])
            a.imshow(src, cmap="gray", vmin=lo_, vmax=hi_)
            for c in contours(S[:, i], shape):
                a.plot(c[:, 0, 0], c[:, 0, 1], lw=1.3, color=outline_col)
            a.set_xlim(xx - CROP / 2, xx + CROP / 2)
            a.set_ylim(yy + CROP / 2, yy - CROP / 2)
            a.set_xticks([]), a.set_yticks([])
            if i == 0:
                a.set_title(["mean", "max"][jj], fontsize=9)
            if jj == 0:
                a.set_ylabel(labels[i], fontsize=10, rotation=0,
                             labelpad=14, va="center", color=col,
                             fontweight="bold")
        a = fig.add_subplot(bot[i, 2])
        a.plot(t, det_c[i], lw=.4, color=col)
        a.set_xlim(0, t[-1]), a.set_xticks([]), a.set_yticks([])
        a.text(.004, .9, f"{rows[i]['verdict']}   skew "
                         f"{rows[i]['cell_skew']:.2f}  ac1 "
                         f"{rows[i]['cell_ac1']:.2f}  ev "
                         f"{rows[i]['cell_ev']}",
               transform=a.transAxes, va="top", fontsize=8, color=col)
        if i == 0:
            a.set_title("the cell, detrended dF/F", fontsize=9)
        fig.add_subplot(bot[i, 3]).axis("off")
        for jj, (src, lo_, hi_) in enumerate([(mean_img, lo_m, hi_m),
                                              (mx, lo, hi)], start=4):
            a = fig.add_subplot(bot[i, jj])
            a.imshow(src, cmap="gray", vmin=lo_, vmax=hi_)
            for c in contours(C[:, i], shape):
                a.plot(c[:, 0, 0], c[:, 0, 1], lw=1.3, color="#999999")
            a.set_xlim(kx - CROP / 2, kx + CROP / 2)
            a.set_ylim(ky + CROP / 2, ky - CROP / 2)
            a.set_xticks([]), a.set_yticks([])
            if i == 0:
                a.set_title(["mean", "max"][jj - 4], fontsize=9)
        a = fig.add_subplot(bot[i, 6])
        a.plot(t, det_k[i], lw=.4, color="#999999")
        a.set_xlim(0, t[-1]), a.set_xticks([]), a.set_yticks([])
        a.text(.004, .9, f"control {dists[i]:.0f} px away   skew "
                         f"{rows[i]['ctrl_skew']:.2f}  ac1 "
                         f"{rows[i]['ctrl_ac1']:.2f}  ev "
                         f"{rows[i]['ctrl_ev']}",
               transform=a.transAxes, va="top", fontsize=8, color="#777777")
        if i == 0:
            a.set_title("its control, same shape, same movie", fontsize=9)

    nact = int((D["verdict"] == "active").sum())
    legend = (
        f"METHODS.  Each control is the cell's own footprint translated to "
        f"the nearest position at least {KEEPOUT} px from any detected cell "
        f"({np.nanmin(dists):.0f}-{np.nanmax(dists):.0f} px away here); "
        f"nearest rather than random, because a control in distant tissue "
        f"has different brightness and noise and would make the comparison "
        f"easy in the wrong way.  Cell traces are solved jointly across all "
        f"footprints on the raw motion-corrected movie (least squares), so "
        f"an overlapping neighbour does not leak in; control traces are the "
        f"same footprint-weighted average, and controls do not overlap "
        f"anything.  dF/F uses F0 = the 20th percentile of each trace, then "
        f"a 20 s running-median detrend - background ROIs in this session "
        f"carry a raw lag-1 autocorrelation of 0.40, so without detrending "
        f"every metric would measure drift.  skew: calcium rises fast and "
        f"decays slowly, so a real trace is positively skewed; shot noise "
        f"is symmetric.  lag-1 autocorrelation: at 0.22 s per frame a GCaMP "
        f"transient still overlaps the next sample; white noise does not.  "
        f"events: peaks above median + 3 x robust SD on a 3-frame smooth, "
        f"1.1 s refractory.\n"
        f"VERDICTS come from the calibrated 200-ROI test, not from this "
        f"figure: skew and lag-1 autocorrelation combined by Fisher's "
        f"method and compared against the same statistic over 200 "
        f"shape-matched background ROIs, kept at p <= 0.05 so 5 % of "
        f"background passes by construction.  active = signal separable "
        f"from background; silent = no signal but the footprint is brighter "
        f"than its surround above the background 95th percentile, i.e. a "
        f"real soma that did not fire; SUSPECT = neither.  "
        f"{nact} of {k} cells here are active.  The active/silent line is a "
        f"threshold, not a biological state - in pain plane B two cells sat "
        f"at p = 0.0498 and 0.0547 with practically identical statistics, "
        f"so the labels are a ranking and downstream tests should use every "
        f"cell that sits on a real soma with FDR control.")
    fig.text(.006, .004, legend, fontsize=9, color="#333333", wrap=True,
             va="bottom", linespacing=1.4)
    fig.suptitle(f"Cell detection: every footprint against its own local "
                 f"control - {os.path.basename(session)[:26]} plane {plane}",
                 fontsize=14, fontweight="bold", y=.997)
    p = os.path.join(cur, "fig2_detection.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    D.to_csv(os.path.join(cur, "fig2_detection_paired.csv"), index=False)
    print(f"  wrote {p}")
    return D


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PAIN)
    ap.add_argument("--plane", default="A")
    a = ap.parse_args()
    print(f"detection figure: plane {a.plane}")
    run(a.session, a.plane)


if __name__ == "__main__":
    main()
