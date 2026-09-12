"""plot_cell_by_cell.py  -  two figures per plane, cell by cell.

FIGURE 2  cell_by_cell_location_trace.png
  One row per cell: where it is in the whole field, a zoom on it, and its
  full dF/F trace. Answers "which cell is this and what does it do".

FIGURE 3  cell_by_cell_evaluation.png
  One row per cell: the trace with detected events marked, its
  autocorrelation against the indicator's plausible decay range, and the
  verdict with the numbers behind it. Answers "is this a neuronal calcium
  trace or not".

HOW THE VERDICT IS REACHED - it is not an opinion
  A real calcium transient rises fast and decays slowly, so the trace is
  positively skewed and consecutive samples are correlated. Shot noise is
  symmetric and white. But at 4.6 Hz with a measured single-frame SNR of
  0.75, a real cell's trace also looks like noise by eye, so "looks like
  calcium" is settled against a measured null: 200 control ROIs cut from the
  same movie, with the same footprint shapes moved to places containing no
  detected cell, traces taken by the identical method. Skew and lag-1
  autocorrelation are combined with Fisher's method and calibrated on those
  controls, so the cut sits at a 5 % false-positive rate on background.

  Three outcomes, because "no signal" and "not a cell" are different claims:
    active   calcium signal separable from background
    silent   no signal, but the footprint is brighter than its surround
             above the background p95 - a real soma that did not fire
    SUSPECT  neither

  All metrics are on DETRENDED dF/F: background ROIs in the pain session have
  a raw lag-1 autocorrelation of 0.40, so on raw traces the test would
  measure drift rather than calcium.

  tau is the most legible single number here: active cells come out at
  0.20-1.37 s, which is the GCaMP range, while everything else sits at
  0.14-0.18 s, i.e. one frame - white noise.

OUTPUT  ->  <plane>\\curated\\cell_by_cell_location_trace.png
            <plane>\\curated\\cell_by_cell_evaluation.png

USAGE
  python plot_cell_by_cell.py            # all four planes
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd

from extract_qc_cells import FS, autocorr, detrend, metrics, to_dff
from plot_cellmap_traces import demix_raw

SESSIONS = [
    ("OF", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52"),
    ("PA", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
           "2026-09-11_15-56-48"),
]
REL = 0.3
ZOOM = 45          # half-window of the per-cell crop, px
GCAMP_LO, GCAMP_HI = 0.2, 2.0   # plausible indicator decay range, s
VCOL = {"active": "#1C6E8C", "silent": "#B8860B", "SUSPECT": "#C1272D"}


def load(session, plane):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mx = np.array(h["output"]["info"]["max_image"]).T
        mn = np.array(h["output"]["info"]["summary_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lab = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                      dtype={"label": str})
    lab = lab[lab["index"] > 0].sort_values("index")
    qcf = os.path.join(cur, "qc_cells", "qc_cells.csv")
    qc = pd.read_csv(qcf) if os.path.exists(qcf) else None
    mov = os.path.join(base, f"plane_{plane}_1.h5")
    return S, (hh, w), lab, qc, mn, mx, mov, cur


def contours(col, shape):
    img = col.reshape(shape)
    bw = (img > REL * img.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def centroid(col, shape):
    img = col.reshape(shape)
    ys, xs = np.nonzero(img)
    wt = img[ys, xs]
    return np.average(ys, weights=wt), np.average(xs, weights=wt)


def run(tag, session, plane):
    S, shape, lab, qc, mn, mx, mov, cur = load(session, plane)
    hh, w = shape
    k = S.shape[1]
    labels = lab["label"].tolist()
    F, mean_img = demix_raw(mov, S, shape, mn)
    dff = to_dff(F)
    det, _ = detrend(dff)
    t = np.arange(dff.shape[1]) / FS

    verdict, ev = [], []
    for i in range(k):
        m, ac = metrics(det[i])
        ev.append((m, ac))
        if qc is not None and (qc["cell"] == i + 1).any():
            verdict.append(qc.loc[qc["cell"] == i + 1, "verdict"].iloc[0])
        else:
            verdict.append("?")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- figure 2: location and trace ----------------
    fig, ax = plt.subplots(k, 3, figsize=(15.5, 1.55 * k),
                           gridspec_kw=dict(width_ratios=[1, 1, 6.2]),
                           squeeze=False)
    lo, hi = np.percentile(mx, [2, 99.7])
    for i in range(k):
        cy, cx = centroid(S[:, i], shape)
        a0, a1, a2 = ax[i]
        a0.imshow(mx, cmap="gray", vmin=lo, vmax=hi)
        a0.plot(cx, cy, "o", ms=9, mfc="none", mec="#E0A020", mew=1.8)
        a0.add_patch(plt.Rectangle((cx - ZOOM, cy - ZOOM), 2 * ZOOM,
                                   2 * ZOOM, fill=False, ec="#E0A020",
                                   lw=.8))
        a0.set_xlim(0, w), a0.set_ylim(hh, 0), a0.axis("off")
        a1.imshow(mx, cmap="gray", vmin=lo, vmax=hi)
        for c in contours(S[:, i], shape):
            a1.plot(c[:, 0, 0], c[:, 0, 1], lw=1.6, color="#E0A020")
        a1.set_xlim(cx - ZOOM, cx + ZOOM)
        a1.set_ylim(cy + ZOOM, cy - ZOOM)
        a1.axis("off")
        a2.plot(t, dff[i], lw=.45, color=VCOL.get(verdict[i], "#555555"))
        a2.set_xlim(0, t[-1])
        a2.set_ylabel(labels[i], fontsize=11, fontweight="bold", rotation=0,
                      labelpad=18, va="center",
                      color=VCOL.get(verdict[i], "#555555"))
        a2.text(.004, .93, f"row {cy:.0f}, col {cx:.0f}   peak dF/F "
                           f"{np.percentile(dff[i], 99.5):.3f}",
                transform=a2.transAxes, va="top", fontsize=9, color="#555555")
        if i < k - 1:
            a2.set_xticks([])
        a2.spines[["top", "right"]].set_visible(False)
        if i == 0:
            a0.set_title("where in the FOV", fontsize=10)
            a1.set_title(f"zoom {2 * ZOOM} px", fontsize=10)
            a2.set_title("dF/F, full session", fontsize=10)
    ax[-1][2].set_xlabel("time (s)")
    fig.suptitle(f"{tag} plane {plane}: {k} curated cells - location and "
                 f"trace  (MAX image; dF/F from the raw movie, "
                 f"F0 = 20th percentile)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .985])
    p2 = os.path.join(cur, "cell_by_cell_location_trace.png")
    fig.savefig(p2, dpi=135, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figure 3: is it a calcium trace ----------------
    fig, ax = plt.subplots(k, 3, figsize=(15.5, 1.5 * k),
                           gridspec_kw=dict(width_ratios=[5.2, 1.5, 2.3]),
                           squeeze=False)
    for i in range(k):
        m, ac = ev[i]
        col = VCOL.get(verdict[i], "#555555")
        a0, a1, a2 = ax[i]
        a0.plot(t, det[i], lw=.45, color=col)
        med = np.median(det[i])
        sig = 1.4826 * np.median(np.abs(det[i] - med))
        a0.axhline(med + 3 * sig, color="#999999", ls=":", lw=.8)
        sm = np.convolve(det[i], np.ones(3) / 3, mode="same")
        above = sm > med + 3 * sig
        starts = np.flatnonzero(np.diff(np.concatenate(
            ([0], above.astype(np.int8), [0]))) == 1)
        refr = max(int(round(1.1 * FS)), 1)
        last = -10 ** 9
        for s_ in starts:
            if s_ - last >= refr:
                a0.plot(t[s_], det[i][s_], "v", ms=4, color="#C1272D")
                last = s_
        a0.set_xlim(0, t[-1])
        a0.set_ylabel(labels[i], fontsize=11, fontweight="bold", rotation=0,
                      labelpad=18, va="center", color=col)
        a0.spines[["top", "right"]].set_visible(False)
        if i < k - 1:
            a0.set_xticks([])

        lags = np.arange(len(ac)) / FS
        a1.axvspan(GCAMP_LO, GCAMP_HI, color="#DDE8EE")
        a1.plot(lags, ac, lw=1.2, color=col)
        a1.axhline(1 / np.e, color="k", ls=":", lw=.8)
        a1.axhline(0, color="k", lw=.5)
        a1.set_ylim(-.25, 1)
        a1.set_xlim(0, lags[-1])
        if i < k - 1:
            a1.set_xticks([])

        tau = "n/a" if not np.isfinite(m["tau_s"]) else f"{m['tau_s']:.2f} s"
        a2.axis("off")
        a2.text(0, .93, verdict[i], fontsize=12, fontweight="bold",
                color=col, va="top")
        a2.text(0, .60,
                f"tau {tau}    ac1 {m['ac1']:.2f}\n"
                f"skew {m['skew']:.2f}   events {m['events']}\n"
                f"snr {m['snr']:.1f}",
                fontsize=9.5, color="#333333", va="top", family="monospace")
        if i == 0:
            a0.set_title("detrended dF/F, red marks = detected events, "
                         "dotted = median + 3 sigma", fontsize=10)
            a1.set_title("autocorrelation\nshaded = GCaMP range", fontsize=10)
            a2.set_title("verdict", fontsize=10)
    ax[-1][0].set_xlabel("time (s)")
    ax[-1][1].set_xlabel("lag (s)")
    fig.suptitle(f"{tag} plane {plane}: does each trace look like a neuronal "
                 f"calcium trace?  "
                 f"(verdict from a 200-ROI background null, Fisher "
                 f"combination of skew and lag-1 autocorrelation, "
                 f"p <= 0.05)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .985])
    p3 = os.path.join(cur, "cell_by_cell_evaluation.png")
    fig.savefig(p3, dpi=135, bbox_inches="tight")
    plt.close(fig)

    nv = pd.Series(verdict).value_counts().to_dict()
    print(f"  {tag}-{plane}: {k} cells  {nv}")
    return [dict(plane=f"{tag}-{plane}", label=labels[i],
                 verdict=verdict[i],
                 tau_s=ev[i][0]["tau_s"], ac1=ev[i][0]["ac1"],
                 skew=ev[i][0]["skew"], events=ev[i][0]["events"],
                 snr=ev[i][0]["snr"],
                 peak_dff=round(float(np.percentile(dff[i], 99.5)), 4))
            for i in range(k)]


def main():
    rows = []
    for tag, sess in SESSIONS:
        for plane in ("A", "B"):
            rows += run(tag, sess, plane)
    D = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cell_evaluation.csv")
    D.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(D.groupby(["plane", "verdict"]).size().to_string())
    act = D[D["verdict"] == "active"]
    if len(act):
        print(f"\nactive cells: tau {act['tau_s'].min():.2f}-"
              f"{act['tau_s'].max():.2f} s, peak dF/F "
              f"{act['peak_dff'].min():.3f}-{act['peak_dff'].max():.3f}")
    oth = D[D["verdict"] != "active"]
    if len(oth):
        print(f"everything else: tau {np.nanmin(oth['tau_s']):.2f}-"
              f"{np.nanmax(oth['tau_s']):.2f} s "
              f"(one frame = {1 / FS:.2f} s)")


if __name__ == "__main__":
    main()
