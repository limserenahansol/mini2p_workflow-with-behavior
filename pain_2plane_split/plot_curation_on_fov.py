"""plot_curation_on_fov.py  -  show the curation decisions on the FOV.

cellmap_traces.png draws only the cells that survived, so it cannot answer
"where were the ones I discarded, and what did the trim and the split
actually do". This draws EVERY original candidate on the field with its
fate:

  kept       solid green
  discarded  dashed red
  trimmed    original outline dotted orange, kept part solid orange
  split      the two halves solid magenta, labelled 11a / 11b
  added      solid cyan, labelled N3 / N4 / N5

Background is the MAX image over time, so an added footprint can be judged
against the soma it was meant to capture.

OUTPUT  ->  <plane>\\curated\\curation_on_fov.png   and a zoomed panel
            curation_on_fov_zoom.png around the cells that changed

USAGE
  python plot_curation_on_fov.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd

SESSIONS = [
    ("OF", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52"),
    ("PA", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
           "2026-09-11_15-56-48"),
]
REL = 0.3
STYLE = {
    "keep": dict(color="#2E7D5B", ls="-", lw=1.7),
    "discard": dict(color="#C1272D", ls="--", lw=1.7),
    "trim": dict(color="#E08214", ls="-", lw=2.0),
    "trim_orig": dict(color="#E08214", ls=":", lw=1.3),
    "split": dict(color="#B23AA8", ls="-", lw=2.0),
    "add": dict(color="#1C8CA8", ls="-", lw=2.0),
}


def load_S(path):
    with h5py.File(path, "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mx = np.array(h["output"]["info"]["max_image"]).T
    k, w, hh = S3.shape
    return S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32), \
        (hh, w), mx


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


def kind_of(op):
    if op == "discard":
        return "discard"
    if op.startswith("trim"):
        return "trim"
    if op.startswith("split"):
        return "split"
    if op.startswith("add"):
        return "add"
    return "keep"


def draw(ax, S, shape, lab, cur_S, cur_index):
    """Every original candidate plus the added ones, with its fate."""
    hh, w = shape
    drawn = []
    for _, r in lab.iterrows():
        k = kind_of(str(r["op"]))
        label = str(r["label"])
        if k == "add":
            col = cur_S[:, int(r["index"]) - 1]
        elif r["origin"] > 0:
            col = S[:, int(r["origin"]) - 1]
        else:
            continue
        # original outline
        st = STYLE["trim_orig"] if k == "trim" else STYLE[k]
        for c in contours(col, shape):
            ax.plot(c[:, 0, 0], c[:, 0, 1], **st)
        # the kept part, for trim and split
        if k in ("trim", "split") and r["index"] > 0:
            for c in contours(cur_S[:, int(r["index"]) - 1], shape):
                ax.plot(c[:, 0, 0], c[:, 0, 1], **STYLE[k])
        src = cur_S[:, int(r["index"]) - 1] if r["index"] > 0 else col
        cy, cx = centroid(src, shape)
        ax.text(cx + 8, cy - 8, label, fontsize=10, fontweight="bold",
                color=STYLE[k]["color"])
        drawn.append((label, k, cy, cx))
    return drawn


def run(tag, session, plane):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    S, shape, mx = load_S(os.path.join(base, "final_analysis_results.mat"))
    cur_S, shape2, mx2 = load_S(os.path.join(
        cur, "final_analysis_results.mat"))
    if shape != shape2:
        raise SystemExit(f"{plane}: shape mismatch {shape} vs {shape2}")
    lab = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                      dtype={"label": str})
    hh, w = shape

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_kept = int((lab["index"] > 0).sum())
    n_drop = int((lab["index"] == 0).sum())
    changed = lab[(lab["index"] > 0) & (lab["op"] != "keep")]

    fig, ax = plt.subplots(figsize=(11, 9.4))
    ax.imshow(mx, cmap="gray", vmin=np.percentile(mx, 2),
              vmax=np.percentile(mx, 99.7))
    drawn = draw(ax, S, shape, lab, cur_S, None)
    ax.set_xlim(0, w)
    ax.set_ylim(hh, 0)
    ax.axis("off")
    ax.set_title(f"{tag} plane {plane}: curation on the FOV  -  "
                 f"{len(lab)} candidates, {n_kept} kept, {n_drop} discarded",
                 fontsize=13)
    keys = ["keep", "discard", "trim", "split", "add"]
    names = {"keep": "kept", "discard": "discarded",
             "trim": "trimmed (dotted = original)",
             "split": "split into two", "add": "added (missed soma)"}
    present = {k for _, k, _, _ in drawn}
    ax.legend(handles=[Line2D([], [], **STYLE[k], label=names[k])
                       for k in keys if k in present],
              loc="lower right", fontsize=10, framealpha=.85)
    fig.tight_layout()
    p = os.path.join(cur, "curation_on_fov.png")
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)

    # zoom on whatever actually changed, so trim/split/add can be judged
    interesting = [(l, k, cy, cx) for l, k, cy, cx in drawn if k != "keep"]
    if interesting:
        n = len(interesting)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow),
                                squeeze=False)
        for a, (label, k, cy, cx) in zip(axs.ravel(), interesting):
            R = 55
            y0, y1 = int(max(cy - R, 0)), int(min(cy + R, hh))
            x0, x1 = int(max(cx - R, 0)), int(min(cx + R, w))
            a.imshow(mx, cmap="gray", vmin=np.percentile(mx, 2),
                     vmax=np.percentile(mx, 99.7))
            draw(a, S, shape, lab, cur_S, None)
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)
            a.axis("off")
            a.set_title(f"{label}  ({names[k]})", fontsize=11)
        for a in axs.ravel()[n:]:
            a.axis("off")
        fig.suptitle(f"{tag} plane {plane}: the candidates that changed",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .97])
        pz = os.path.join(cur, "curation_on_fov_zoom.png")
        fig.savefig(pz, dpi=145, bbox_inches="tight")
        plt.close(fig)
    else:
        pz = None
    print(f"  {tag}-{plane}: {len(lab)} candidates -> {n_kept} kept, "
          f"{n_drop} discarded, {len(changed)} changed")
    return p, pz


def main():
    for tag, sess in SESSIONS:
        for plane in ("A", "B"):
            run(tag, sess, plane)
    print("\nwrote curation_on_fov.png (+ _zoom.png where anything changed) "
          "in each <plane>\\curated\\ folder")


if __name__ == "__main__":
    main()
