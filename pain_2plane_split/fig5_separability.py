"""fig5_separability.py  -  why 8 footprints were dropped from the union.

THE MISTAKE THIS FIGURE DOCUMENTS
  The first union was "every pain footprint plus every open-field footprint
  that is not a shape-confirmed match". That is wrong in a way nothing else
  in the pipeline would have caught: an open-field footprint can sit on top
  of a pain footprint, fail the shape test, and be added anyway - so the
  same soma enters the design matrix twice.

  Least squares then has two near-identical regressors and no way to choose,
  so it pays for the soma's transient with a positive trace on one copy and
  a negative trace on the other. Both traces look like calcium. One of them
  is upside down.

  Consequence in the results: the two "corner-preferring" cells reported in
  an earlier run were the negative halves of centre-preferring cells
  (trace r = -0.82 and -0.84). Centre minus corner is negative for a
  sign-flipped centre cell by construction.

WHAT THE FIGURE SHOWS, ZOOMED OUT FIRST
  1  the whole field, both planes: the union kept (green) and the
     footprints dropped (red), so the scale of the problem is visible
  2  one pair close up, footprint on footprint
  3  the two traces the degenerate solve produces for that pair, measured
     by re-running it, against the single trace the clean union gives
  4  the calibration: footprint correlation for 361 pairs WITHIN a curated
     session (neurons EXTRACT and curation kept as distinct) against the 8
     cross-session pairs that were dropped
  5  the numbers before and after

OUTPUT  ->  <pain session>\\output_split\\match\\fig5_separability.png

USAGE
  python fig5_separability.py
"""
from __future__ import annotations

import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import transfer_footprints as T  # noqa: E402
from extract_qc_cells import detrend, to_dff  # noqa: E402
from match_audit import shape_r  # noqa: E402

KEEP, DROP = "#2E7D5B", "#C1272D"
INK, DIM = "#1A1A1A", "#666666"


def bpnorm(img):
    v = img.astype(np.float32)
    lo, hi = np.percentile(v, [2, 99.5])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def outline(ax, col, shape, color, lw=1.1, ls="-"):
    m = (col.reshape(shape) > 0.2 * col.max()).astype(np.uint8)
    for c in cv2.findContours(m, cv2.RETR_LIST,
                              cv2.CHAIN_APPROX_NONE)[0]:
        if len(c) > 4:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=color, lw=lw, ls=ls)


def build(plane):
    """Redo the union for this plane, keeping both the kept and dropped sets."""
    o, p = T.load(T.OF, plane), T.load(T.PA, plane)
    hh, w = p["shape"]
    L = []
    M, _ = T.best_transform(o, p, L)
    pr = T.assign(T.warp_pts(o["cent"], M), p["cent"], T.RADIUS)
    keep_pairs = []
    for i, j, d in pr:
        wi = cv2.warpAffine(o["S"][:, i].reshape(hh, w), M, (w, hh))
        if shape_r(wi.ravel(), p["S"][:, j], (hh, w)) >= T.SHAPE_MIN:
            keep_pairs.append(i)
    Spa = [p["S"][:, j] for j in range(p["S"].shape[1])]
    lab = [f"{plane}{j + 1}" for j in range(len(Spa))]
    dropped = []
    for i in range(o["S"].shape[1]):
        if i in keep_pairs:
            continue
        wc = cv2.warpAffine(o["S"][:, i].reshape(hh, w), M, (w, hh)).ravel()
        r_sep, j_sep = T.gram_max(wc, np.stack(Spa, 1))
        if r_sep >= T.SEP_MAX:
            dropped.append(dict(of=o["lab"][i], col=wc, r=r_sep, into=j_sep,
                                shape=shape_r(wc, Spa[j_sep], (hh, w))))
            continue
        Spa.append(wc)
        lab.append(f"{plane}{len(Spa)}")
    return dict(plane=plane, shape=(hh, w), S=np.stack(Spa, 1), lab=lab,
                dropped=dropped, mean_of=o["mean"], mean_pa=p["mean"],
                mov_of=o["mov"])


def solve_both(b):
    """Traces with and without the dropped footprints, same movie."""
    S, sh = b["S"], b["shape"]
    clean = T.solve(b["mov_of"], S, sh)
    D = np.stack([d["col"] for d in b["dropped"]], 1)
    deg = T.solve(b["mov_of"], np.hstack([S, D]), sh)
    return clean, deg


def gram_within():
    """Footprint correlation for every pair within each curated session."""
    v = []
    for root in (T.PA, T.OF):
        for plane in ("A", "B"):
            S = T.load(root, plane)["S"]
            R = T.gram_matrix(S)
            v.append(R[np.triu_indices(R.shape[0], 1)])
    return np.concatenate(v)


def main():
    B = {p: build(p) for p in ("A", "B")}
    sol = {}
    for p in ("A", "B"):
        print(f"  plane {p}: solving with and without the "
              f"{len(B[p]['dropped'])} dropped footprints ...")
        sol[p] = solve_both(B[p])
    within = gram_within()

    # the worst anti-correlated pair the degenerate solve produces
    worst = None
    rows = []
    for p in ("A", "B"):
        b, (clean, deg) = B[p], sol[p]
        n = b["S"].shape[1]
        dz, _ = detrend(to_dff(deg))
        cz, _ = detrend(to_dff(clean))
        for k, d in enumerate(b["dropped"]):
            j = d["into"]
            r = float(np.corrcoef(dz[j], dz[n + k])[0, 1])
            rows.append(dict(plane=p, of=d["of"], into=b["lab"][j],
                             foot_r=d["r"], shape_r=d["shape"], trace_r=r))
            if worst is None or r < worst[-1]:
                worst = (p, k, j, n, r)
        Gc = T.gram_raw(b["S"])
        Gd = T.gram_raw(np.hstack([b["S"],
                                   np.stack([d["col"]
                                             for d in b["dropped"]], 1)]))
        rows[-1]["_cond_clean"] = np.linalg.cond(Gc)
        rows[-1]["_cond_deg"] = np.linalg.cond(Gd)
    R = pd.DataFrame(rows)

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.55, .95, .95, .46, .26],
                          hspace=.34, wspace=.24)

    # ---- row 1: the whole field, both planes -------------------------
    for c, p in enumerate(("A", "B")):
        b = B[p]
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
        ax.imshow(bpnorm(b["mean_of"]), cmap="gray")
        for i in range(b["S"].shape[1]):
            outline(ax, b["S"][:, i], b["shape"], KEEP)
        for d in b["dropped"]:
            outline(ax, d["col"], b["shape"], DROP, lw=1.5, ls="--")
            yy, xx = np.nonzero(d["col"].reshape(b["shape"]))
            ax.text(xx.mean(), yy.min() - 8, f"OF {d['of']}", color=DROP,
                    fontsize=8.5, ha="center", va="bottom",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])
        ax.set_title(f"plane {p}: {b['S'].shape[1]} union footprints kept "
                     f"(green), {len(b['dropped'])} dropped (red, dashed)",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- row 2 left: the worst pair, close up ------------------------
    pw, kw, jw, nw, rw = worst
    b = B[pw]
    d = b["dropped"][kw]
    hh, w = b["shape"]
    m1 = b["S"][:, jw].reshape(hh, w)
    m2 = d["col"].reshape(hh, w)
    ys, xs = np.nonzero((m1 > 0) | (m2 > 0))
    pad = 26
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, hh)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(bpnorm(b["mean_of"])[y0:y1, x0:x1], cmap="gray")
    for col, cc, ls in ((b["S"][:, jw], KEEP, "-"), (d["col"], DROP, "--")):
        m = (col.reshape(hh, w) > 0.2 * col.max()).astype(np.uint8)[y0:y1,
                                                                    x0:x1]
        for c_ in cv2.findContours(m, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)[0]:
            if len(c_) > 4:
                ax.plot(c_[:, 0, 0], c_[:, 0, 1], color=cc, lw=2, ls=ls)
    ax.set_title(f"one soma, two footprints\nunion {b['lab'][jw]} (green) "
                 f"and OF {d['of']} (red)\nfootprint r = {d['r']:.3f}, "
                 f"shape r = {d['shape']:+.3f}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- row 2 right: what the degenerate solve does to the traces ---
    clean, deg = sol[pw]
    dz, _ = detrend(to_dff(deg))
    cz, _ = detrend(to_dff(clean))
    t = np.arange(dz.shape[1]) / 4.6083
    # Both trace panels share one y scale. They have to: the point is partly
    # that the degenerate solve inflates both copies' amplitude (variance
    # inflation 1 / (1 - r^2)), and autoscaled axes would hide exactly that.
    a1 = dz[jw] - np.median(dz[jw])
    a2 = dz[nw + kw] - np.median(dz[nw + kw])
    a3 = cz[jw] - np.median(cz[jw])
    ylim = 1.05 * max(np.abs(np.concatenate([a1, a2, a3])).max(), 1e-9)

    ax = fig.add_subplot(gs[1, 1:])
    ax.plot(t, a1, lw=.7, color=KEEP,
            label=f"{b['lab'][jw]}, both footprints in the solve")
    ax.plot(t, a2, lw=.7, color=DROP,
            label=f"OF {d['of']}, the second copy   (r = {rw:+.3f})")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("detrended dF/F")
    ax.set_xticklabels([])
    ax.legend(fontsize=9, loc="upper right", frameon=False, ncol=2)
    ax.set_title("keeping both: one soma's signal is split into a positive "
                 "and a negative copy", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[2, 1:])
    ax.plot(t, a3, lw=.7, color=INK,
            label=f"{b['lab'][jw]}, dropped footprint excluded")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("time (s), open-field session")
    ax.set_ylabel("detrended dF/F")
    rk = float(np.corrcoef(cz[jw], dz[jw])[0, 1])
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_title(f"the union actually used: one footprint, one trace - same "
                 f"y scale, {np.std(a1) / max(np.std(a3), 1e-12):.1f}x "
                 f"smaller than the split version (r = {rk:+.3f} with it)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- row 3 left: the calibration ---------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(np.zeros(len(within)) + np.random.default_rng(0).normal(
        0, .045, len(within)), within, s=5, color=DIM, alpha=.5)
    ax.scatter(np.ones(len(R)) + np.random.default_rng(1).normal(
        0, .045, len(R)), R["foot_r"], s=34, color=DROP, zorder=3)
    ax.axhline(T.SEP_MAX, color=INK, ls=":", lw=1.3)
    ax.text(.5, T.SEP_MAX + .018, f"SEP_MAX {T.SEP_MAX:.2f}", fontsize=9,
            ha="center", color=INK)
    ax.axhline(within.max(), color=DIM, ls="--", lw=1)
    ax.text(.5, within.max() - .055, f"within-session max "
            f"{within.max():.3f}", fontsize=9, ha="center", color=DIM)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"within one session\n({len(within)} pairs of "
                        f"distinct cells)", f"dropped\n({len(R)} pairs)"],
                       fontsize=9)
    ax.set_ylabel("footprint correlation (as regressors)")
    ax.set_ylim(-.03, 1.0)
    ax.set_title("the threshold is calibrated, not chosen", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- row 4: the table, row 5: how to read it ---------------------
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    L = [f"{'plane':>5s} {'dropped OF':>11s} {'folded into':>12s} "
         f"{'footprint r':>12s} {'shape r':>8s} "
         f"{'trace r if both kept':>21s}"]
    for _, r in R.iterrows():
        L.append(f"{r['plane']:>5s} {r['of']:>11s} {r['into']:>12s} "
                 f"{r['foot_r']:12.3f} {r['shape_r']:+8.3f} "
                 f"{r['trace_r']:+21.3f}")
    cc = R.dropna(subset=["_cond_clean"])
    L.append("")
    L.append("cond(S'S):  " + ",  ".join(
        f"plane {r['plane']} {r['_cond_deg']:.1f} with both -> "
        f"{r['_cond_clean']:.1f} without" for _, r in cc.iterrows()))
    ax.text(0, 1, "\n".join(L), family="Consolas", fontsize=9.5,
            va="top", color=INK)

    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.text(0, 1,
            "HOW TO READ IT.  A pair of footprints with correlation r as "
            "regressors inflates the variance of both traces by 1 / (1 - "
            "r^2): 1.6x at r = 0.6, 4.8x at r = 0.89. Within a curated "
            "session the worst pair of distinct neurons reaches "
            f"{within.max():.3f}, so SEP_MAX = {T.SEP_MAX:.2f} sits far "
            "above anything this data calls two cells and far below every "
            "pair dropped here.  The trace column is measured, not argued: "
            "the solve was re-run with the dropped footprints included, on "
            "the same movie, and those are the correlations it produces.  "
            "Dropping a footprint does not lose the neuron - its soma is "
            "already covered by the union cell it was folded into, which is "
            "recorded per row as merged_of in union_cells.csv.",
            fontsize=9.5, color=DIM, va="top", wrap=True)

    fig.suptitle("Union QC: two footprints on one soma cannot both have a "
                 "trace\n8 open-field footprints dropped for separability; "
                 "the 2 'corner-preferring' cells of the earlier run were "
                 "the negative halves of centre cells", fontsize=13)
    fig.subplots_adjust(left=.05, right=.985, top=.925,
                        bottom=.015)
    out = os.path.join(T.PA, "match", "fig5_separability.png")
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    R.drop(columns=[c for c in R.columns if c.startswith("_")]).to_csv(
        os.path.join(T.PA, "match", "fig5_dropped_footprints.csv"),
        index=False)
    print(f"\n{R.to_string(index=False)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
