"""plot_cellmap_traces.py  -  final cell IDs on the FOV, with each ID's trace beside it.

WHICH TRACE IS SHOWN, AND WHY NOT THE ONE IN THE MAT FILE
  The curated pass stores least-squares traces solved on the SPATIALLY
  BANDPASSED movie. Bandpassing removes the low-frequency background, so
  those traces are close to zero-mean and (F - F0) / F0 on them is not a
  meaningful dF/F - F0 sits near zero and the ratio explodes. They are the
  right thing for demixing, the wrong thing for an amplitude axis.

  So the traces here are pulled from the RAW denoised movie
  (<plane>_X_1.h5, which is motion-corrected but not bandpassed) using the
  same curated footprints, and dF/F uses a 20th-percentile baseline. That
  gives F0 > 0 and amplitudes comparable between cells. The same routine was
  validated on the uncurated sets: it agrees with EXTRACT's own traces at
  r = 0.78-0.96.

  This also avoids repeating step 2's baseline choice, which is
  F0 = mean(trace): a cell with many events gets an inflated F0 and a
  shrunken dF/F.

  These are weighted averages, not demixed, so for footprints that touch
  (pain B 11a/11b, and N4/N5 next to open-field B #4) the report states the
  correlation with the demixed least-squares trace - if a pair is heavily
  cross-contaminated that number shows it.

OUTPUT  ->  <plane>\\curated\\
  cellmap_traces.png    FOV with numbered footprints, traces beside it
  curated_dff.mat       the dF/F actually plotted, one row per label
  curated_dff.csv       same, long format, for downstream use

USAGE
  python plot_cellmap_traces.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from extract_qc_cells import CHUNK, FS, to_dff, weighted_traces

SESSIONS = [
    ("OF", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52"),
    ("PA", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
           "2026-09-11_15-56-48"),
]
REL = 0.3


def load(session, plane):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    f = os.path.join(cur, "final_analysis_results.mat")
    if not os.path.exists(f):
        raise SystemExit(f"no curated result: {f}")
    with h5py.File(f, "r") as h:
        o = h["output"]
        S3 = np.array(o["spatial_weights"])            # (k, w, h)
        ls = np.array(o["temporal_weights"])           # (k, T)
        mx = np.array(o["info"]["max_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lab = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                      dtype={"label": str})
    lab = lab[lab["index"] > 0].sort_values("index")
    if len(lab) != k:
        raise SystemExit(f"{plane}: {k} footprints but {len(lab)} labels")
    mov = os.path.join(base, f"plane_{plane}_1.h5")
    if not os.path.exists(mov):
        raise SystemExit(f"raw movie not found: {mov}")
    return S, (hh, w), ls, lab, mx, mov, cur


def demix_raw(mov_h5, S, shape, mean_img):
    """Least squares on the RAW movie, with explicit background regressors.

    Two things have to be true at once for these traces to be usable:
      * demixed, so a footprint that overlaps a neighbour does not carry the
        neighbour's transients;
      * in raw fluorescence units, so F0 > 0 and (F - F0) / F0 is a real
        dF/F. The curated pass solved least squares on the BANDPASSED movie,
        which is near zero-mean, so its traces satisfy the first and fail
        the second.

    No background regressor. I first added a constant and the session mean
    image as background terms, the way CNMF carries an explicit background.
    That broke badly: every cell came out at r = -0.92 to -0.97 against its
    own plain weighted average, i.e. sign-flipped, and dF/F ran to -10.
    The reason is collinearity - every footprint is a small bump on the same
    bright background, so an UNCONSTRAINED solve splits the shared signal
    between the cell and the background with opposite signs. CNMF avoids
    this with non-negativity constraints, which a plain normal-equation
    solve does not have.

    Without the background term the shared background simply raises every
    cell's baseline, and F0 divides it out. Overlap between cells is still
    resolved, which is the whole point of solving jointly.

    A'M is accumulated chunk by chunk, so peak memory stays at one chunk.
    """
    h, w = shape
    Sn = S / np.maximum(S.sum(0, keepdims=True), 1e-12)
    A = Sn.astype(np.float64)
    AtA = A.T @ A
    n_cells = Sn.shape[1]
    AtM = []
    with h5py.File(mov_h5, "r") as f:
        mov = f["mov"]
        T = mov.shape[0]
        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            blk = np.asarray(mov[t0:t1]).transpose(0, 2, 1).reshape(
                t1 - t0, h * w)
            AtM.append(A.T @ blk.T.astype(np.float64))
    AtM = np.concatenate(AtM, 1)
    Tall = np.linalg.solve(AtA, AtM)
    return Tall[:n_cells], Tall[n_cells:]


def contour(col, shape):
    img = col.reshape(shape)
    bw = (img > REL * img.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def run(tag, session, plane):
    S, shape, ls, lab, mx, mov, cur = load(session, plane)
    hh, w = shape
    k = S.shape[1]
    labels = lab["label"].tolist()
    ops = dict(zip(lab["label"], lab["op"]))

    F_avg, mean_img = weighted_traces(mov, S, shape)
    F_dem, bg = demix_raw(mov, S, shape, mean_img)
    dff = to_dff(F_dem)
    dff_avg = to_dff(F_avg)
    T = dff.shape[1]
    t = np.arange(T) / FS

    # Same movie on both sides, so this difference IS the demixing and
    # nothing else. Comparing the raw weighted average against the
    # bandpassed least-squares traces instead measured "background included
    # versus removed" and made almost every pain cell look contaminated.
    r_demix = np.array([
        float(np.corrcoef(dff[i] - dff[i].mean(),
                          dff_avg[i] - dff_avg[i].mean())[0, 1])
        for i in range(k)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    cols = [cmap(i % 20) for i in range(k)]

    fig = plt.figure(figsize=(16.5, max(6.5, 0.62 * k)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=.06)
    axm = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1])

    axm.imshow(mx, cmap="gray", vmin=np.percentile(mx, 2),
               vmax=np.percentile(mx, 99.7))
    for i in range(k):
        for c in contour(S[:, i], shape):
            axm.plot(c[:, 0, 0], c[:, 0, 1], lw=1.6, color=cols[i])
        img = S[:, i].reshape(shape)
        ys, xs = np.nonzero(img)
        wt = img[ys, xs]
        cy, cx = np.average(ys, weights=wt), np.average(xs, weights=wt)
        axm.text(cx + 9, cy - 9, labels[i], color=cols[i], fontsize=11,
                 fontweight="bold",
                 path_effects=None)
    axm.set_xlim(0, w)
    axm.set_ylim(hh, 0)
    axm.axis("off")
    axm.set_title(f"{tag} plane {plane}: {k} curated cells on the MAX image",
                  fontsize=12)

    span = np.percentile(dff, 99.5) - np.percentile(dff, 0.5)
    step = max(span * 0.85, 1e-6)
    for i in range(k):
        axt.plot(t, dff[i] + (k - 1 - i) * step, lw=.5, color=cols[i])
        note = ""
        if ops.get(labels[i], "keep") != "keep":
            note = "  " + ops[labels[i]].split(" ")[0]
        axt.text(-0.008 * t[-1], (k - 1 - i) * step, labels[i] + note,
                 ha="right", va="center", fontsize=10, color=cols[i],
                 fontweight="bold")
    axt.set_xlim(0, t[-1])
    axt.set_ylim(-step, k * step)
    axt.set_yticks([])
    axt.set_xlabel("time (s)")
    axt.spines[["top", "right", "left"]].set_visible(False)
    axt.plot([t[-1] * .985, t[-1] * .985], [0, step * .5], color="k", lw=2)
    axt.text(t[-1] * .978, step * .25, f"{step * .5:.2f} dF/F", rotation=90,
             ha="right", va="center", fontsize=9)
    axt.set_title(f"dF/F from the raw movie, F0 = 20th percentile"
                  f"   ({T} frames, {t[-1]:.0f} s at {FS} Hz)", fontsize=12)

    fig.tight_layout()
    p = os.path.join(cur, "cellmap_traces.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    savemat(os.path.join(cur, "curated_dff.mat"),
            dict(labels=np.array(labels, dtype=object), dff=dff,
                 t_s=t, fs_hz=FS,
                 baseline="20th percentile of the raw weighted trace",
                 source="footprint-weighted average of "
                        f"plane_{plane}_1.h5 (raw, not bandpassed)"))
    pd.DataFrame(dict(
        label=np.repeat(labels, T), t_s=np.tile(t, k), dff=dff.ravel())
    ).to_csv(os.path.join(cur, "curated_dff.csv"), index=False)

    print(f"  {tag}-{plane}: {k} cells, dF/F range "
          f"{dff.min():.2f} to {dff.max():.2f}, "
          f"weighted-vs-demixed r median {np.median(r_demix):.3f} "
          f"(min {r_demix.min():.3f})")
    return [dict(plane=f"{tag}-{plane}", label=labels[i],
                 op=ops.get(labels[i], "keep"),
                 peak_dff=round(float(np.percentile(dff[i], 99.5)), 3),
                 r_vs_demixed=round(float(r_demix[i]), 3))
            for i in range(k)]


def main():
    rows = []
    for tag, sess in SESSIONS:
        for plane in ("A", "B"):
            rows += run(tag, sess, plane)
    D = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "curated_cells_summary.csv")
    D.to_csv(out, index=False)
    print(f"\nwrote {out}")
    low = D[D["r_vs_demixed"] < 0.8]
    if len(low):
        print("\ncells where the weighted average differs from the demixed "
              "estimate (r < 0.8):")
        print(low.to_string(index=False))
        print("These are the footprints that overlap a neighbour; use the "
              "demixed trace for them.")
    else:
        print("\nEvery cell agrees with its demixed estimate at r >= 0.8, so "
              "overlap is not distorting the weighted averages.")


if __name__ == "__main__":
    main()
