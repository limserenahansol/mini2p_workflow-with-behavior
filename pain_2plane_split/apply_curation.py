"""apply_curation.py  -  build the curated footprint set from Hansol's decisions.

This only builds the SPATIAL footprints. Traces are re-derived afterwards by
apply_curation.m, which runs EXTRACT's alternating estimation with these
footprints as S_init. That matters for the split and the added cells: a plain
footprint-weighted average of two adjacent somata cross-contaminates them,
whereas EXTRACT's T-step demixes overlapping sources. Nothing here touches
the original final_analysis_results.mat.

OPERATIONS
  discard  drop the footprint entirely
  trim     keep only the largest connected blob of the footprint
  split    make one footprint per connected blob, labelled 11a, 11b, ...
  add      seed a new footprint as a gaussian disk at a given centroid;
           EXTRACT's S-step then shapes it against the movie

DECISIONS, 2026-09-11, from the curation sheets
  OF-A  discard 7, 13
  OF-B  discard 9; trim 14 to its large blob; add N3, N4, N5 - three somata
        EXTRACT missed in the diagonal chain #13 -> N3 -> N4 -> #4 -> N5 -> #2.
        N4 and N5 are the "next to 4" and the clear one between #4 and #2;
        N3 is the third. N3 and N4 are faint (band-pass peak 1.6 and 2.2
        against 3.5 for N5 and 17.2 for #4), so whether they are dim cells or
        neuropil is left for the QC on their re-extracted traces to settle.
  PA-A  discard 9 only - its contour is empty in both the mean and max images
        (anat 0.001, the lowest trace-versus-pixels agreement in the plane at
        0.64). Every other candidate kept, including #6 and #12, which are
        elongated (ecc 0.934, 0.941) and may be fibres rather than somata.
  PA-B  split 11 into two cells; keep everything else

OUTPUTS  ->  <session>\\output_split\\plane_<X>\\curated\\
  curated_S.mat        S (h*w x k), fov_size, labels - input to apply_curation.m
  curated_labels.csv   new index -> label, origin, operation, area

USAGE
  python apply_curation.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

SESSIONS = {
    "openfield": "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
                 "2026-09-11_15-27-52",
    "pain": "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
            "2026-09-11_15-56-48",
}
REL = 0.3          # blob level, as a fraction of the footprint's own peak
MIN_BLOB = 20      # px; smaller pieces are fragments, not a second soma
SEED_RADIUS = 8.0  # measured median cell radius

DECISIONS = {
    ("openfield", "A"): dict(discard=[7, 13]),
    ("openfield", "B"): dict(discard=[9], trim=[14],
                             add=[("N3", 281.6, 314.6),
                                  ("N4", 281.8, 343.1),
                                  ("N5", 296.7, 388.2)]),
    ("pain", "A"): dict(discard=[9]),
    ("pain", "B"): dict(split=[11]),
}


def load_S(folder):
    with h5py.File(os.path.join(folder, "final_analysis_results.mat"),
                   "r") as f:
        S3 = np.array(f["output"]["spatial_weights"])
    k, w, h = S3.shape
    return S3.transpose(0, 2, 1).reshape(k, h * w).T.astype(np.float32), (h, w)


def blobs(col, shape):
    """Partition one footprint into its blobs, largest first.

    A footprint is a weighted profile, not a mask, so each part has to keep
    the original weights - a hard-edged mask would make the least-squares
    fit worse. The first version did that by dilating each thresholded blob
    by 9x9 and copying the weights inside. That let a part reach into the
    low-weight skirt beyond the threshold, and the verification caught it:
    the two halves of pain B #11 together covered 22 px that were not in the
    original footprint at all.

    So the support is partitioned instead of grown: every nonzero pixel of
    the original goes to whichever kept blob is nearest. The parts are then
    disjoint by construction and their union is exactly the original
    support - neither half can claim the other's skirt, and nothing appears
    outside.
    """
    img = col.reshape(shape)
    bw = (img > REL * img.max()).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    order = sorted((int(st[j, cv2.CC_STAT_AREA]), j)
                   for j in range(1, n))[::-1]
    order = [(a, j) for a, j in order if a >= MIN_BLOB]
    if not order:
        return []
    support = img > 0
    dists = []
    for _a, j in order:
        d = cv2.distanceTransform((lab != j).astype(np.uint8),
                                  cv2.DIST_L2, 3)
        dists.append(d)
    owner = np.argmin(np.stack(dists, 0), axis=0)
    out = []
    for idx, (a, _j) in enumerate(order):
        keep = np.where(support & (owner == idx), img, 0).astype(np.float32)
        out.append((int((keep > REL * img.max()).sum()), keep.ravel()))
    return out


def seed(cy, cx, shape, r=SEED_RADIUS):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 1.5) ** 2))
    g[g < 0.05] = 0
    return g.ravel().astype(np.float32)


def run(tag, plane):
    session = SESSIONS[tag]
    folder = os.path.join(session, "output_split", f"plane_{plane}")
    S, shape = load_S(folder)
    k = S.shape[1]
    d = DECISIONS.get((tag, plane), {})
    drop = set(d.get("discard", []))
    trim = set(d.get("trim", []))
    split = set(d.get("split", []))

    cols, rows = [], []
    for i in range(1, k + 1):
        if i in drop:
            rows.append(dict(label=str(i), origin=i, op="discard",
                             index=0, area_px=int((S[:, i - 1] > 0).sum())))
            continue
        if i in split:
            bl = blobs(S[:, i - 1], shape)
            if len(bl) < 2:
                raise SystemExit(f"{tag} plane {plane} #{i}: asked to split "
                                 f"but only {len(bl)} blob(s) found")
            for sfx, (a, col) in zip("abcdefg", bl):
                cols.append(col)
                rows.append(dict(label=f"{i}{sfx}", origin=i, op="split",
                                 index=len(cols), area_px=a))
            continue
        if i in trim:
            bl = blobs(S[:, i - 1], shape)
            if len(bl) < 2:
                raise SystemExit(f"{tag} plane {plane} #{i}: asked to trim "
                                 f"but only {len(bl)} blob(s) found")
            a, col = bl[0]
            cols.append(col)
            rows.append(dict(label=str(i), origin=i, op="trim (largest blob "
                                                        f"of {len(bl)})",
                             index=len(cols), area_px=a))
            continue
        cols.append(S[:, i - 1])
        rows.append(dict(label=str(i), origin=i, op="keep",
                         index=len(cols), area_px=int((S[:, i - 1] > 0).sum())))

    for name, cy, cx in d.get("add", []):
        col = seed(cy, cx, shape)
        cols.append(col)
        rows.append(dict(label=name, origin=-1,
                         op=f"add seed at row {cy:.0f} col {cx:.0f}",
                         index=len(cols), area_px=int((col > 0).sum())))

    Sc = np.stack(cols, 1) if cols else np.zeros((S.shape[0], 0), np.float32)
    # Columns are built as C-order (row-major) ravels of an (h, w) image,
    # which is numpy's default. MATLAB's reshape fills COLUMN-major, so
    # writing them as-is scrambled every footprint: reshape(S, 440, 512)
    # turned one 212 px soma into 17 stripes of 16 px spanning all 440 rows,
    # and every trace solved against it was therefore wrong. Because
    # 440 != 512 it is not even a clean transpose. So each column is
    # re-raveled in Fortran order here, once, at the boundary.
    h, w = shape
    Sc = np.stack([Sc[:, i].reshape(h, w).ravel(order="F")
                   for i in range(Sc.shape[1])], 1).astype(np.float32) \
        if Sc.shape[1] else Sc
    out = os.path.join(folder, "curated")
    os.makedirs(out, exist_ok=True)
    savemat(os.path.join(out, "curated_S.mat"),
            dict(S=Sc, fov_size=np.array(shape, float),
                 order="fortran: reshape(S(:,i), fov_size(1), fov_size(2))",
                 labels=np.array([r["label"] for r in rows if r["index"] > 0],
                                 dtype=object)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "curated_labels.csv"), index=False)
    n_out = int((D["index"] > 0).sum())
    ops = D[D["op"] != "keep"]
    print(f"  {tag} plane {plane}: {k} -> {n_out} footprints")
    for _, r in ops.iterrows():
        print(f"      #{r['origin'] if r['origin'] > 0 else '-'} "
              f"-> {r['label']:4s}  {r['op']}  ({r['area_px']} px)")
    return n_out


def main():
    print("building curated footprint sets")
    tot = 0
    for tag in ("openfield", "pain"):
        for plane in ("A", "B"):
            tot += run(tag, plane)
    print(f"\n{tot} curated footprints across the four planes")
    print("next: apply_curation.m re-derives the traces with EXTRACT")


if __name__ == "__main__":
    main()
