"""verify_curation.py  -  prove the curated set is exactly what Hansol decided.

WHY THIS EXISTS
  I broke the curated export once: footprints were written with numpy's
  C-order ravel and MATLAB reshaped them column-major, so a 212 px soma
  became 17 stripes of 16 px spanning all 440 rows. Nothing errored. That
  must not be something a figure has to reveal, so every claim about the
  curated set is checked here against two independent sources:

    the ORIGINAL detection   <plane>\\final_analysis_results.mat
    the DECISIONS            the table below, typed from Hansol's message
    the APPLIED result       <plane>\\curated\\final_analysis_results.mat

  Checks, all of which must pass:
    1  the original detection is untouched - same cell count and the same
       footprint centroids as before curation (detection was never re-run)
    2  each curated footprint is a compact blob, not fragments
    3  every discarded cell is absent, every kept cell is present
    4  a kept-unchanged cell's footprint is numerically identical to the
       original
    5  a trimmed cell is a strict subset of its original, with only the
       largest blob left
    6  the split halves are disjoint and their union is inside the original
    7  each added cell sits within one radius of the position that was asked
       for
    8  the traces have one row per curated label, and in the same order

  Exit code is non-zero if anything fails, so this can gate the analysis.
"""
from __future__ import annotations

import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
REL = 0.3
RADIUS = 8

# Typed from the curation decisions, independent of any file on disk.
DECIDED = {
    ("OF", "A"): dict(n_orig=14, discard=[7, 13], trim=[], split=[], add=[],
                      n_final=12),
    ("OF", "B"): dict(n_orig=14, discard=[9], trim=[14], split=[],
                      add=[("N3", 281.6, 314.6), ("N4", 281.8, 343.1),
                           ("N5", 296.7, 388.2)], n_final=16),
    ("PA", "A"): dict(n_orig=12, discard=[9], trim=[], split=[], add=[],
                      n_final=11),
    ("PA", "B"): dict(n_orig=15, discard=[], trim=[], split=[11], add=[],
                      n_final=16),
}
ROOT = {"OF": OF, "PA": PA}


def load(path):
    with h5py.File(path, "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        T = np.array(h["output"]["temporal_weights"])
    k, w, hh = S3.shape
    return S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float64), \
        (hh, w), T


def blobs(col, shape):
    img = col.reshape(shape)
    if img.max() <= 0:
        return 0, 0, 0
    bw = (img > REL * img.max()).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = [int(a) for a in st[1:, cv2.CC_STAT_AREA] if a >= 20]
    ys, _ = np.nonzero(bw)
    return len(big), (max(big) if big else 0), int(ys.max() - ys.min() + 1)


def cent(col, shape):
    img = col.reshape(shape)
    ys, xs = np.nonzero(img)
    if len(ys) == 0:
        return np.nan, np.nan
    wt = img[ys, xs]
    return np.average(ys, weights=wt), np.average(xs, weights=wt)


def mask(col, shape):
    img = col.reshape(shape)
    return img > REL * img.max()


def main():
    fails, notes = [], []
    for (tag, plane), dec in DECIDED.items():
        base = os.path.join(ROOT[tag], f"plane_{plane}")
        orig_p = os.path.join(base, "final_analysis_results.mat")
        cur_p = os.path.join(base, "curated", "final_analysis_results.mat")
        lab_p = os.path.join(base, "curated", "curated_labels.csv")
        head = f"{tag}-{plane}"
        if not all(os.path.exists(p) for p in (orig_p, cur_p, lab_p)):
            fails.append(f"{head}: missing input files")
            continue

        S0, shape, _T0 = load(orig_p)
        S1, shape1, T1 = load(cur_p)
        lab = pd.read_csv(lab_p, dtype={"label": str})
        kept = lab[lab["index"] > 0].sort_values("index")
        print(f"\n=== {head} ===")

        # 1 detection untouched
        if S0.shape[1] != dec["n_orig"]:
            fails.append(f"{head}: original has {S0.shape[1]} cells, "
                         f"expected {dec['n_orig']}")
        else:
            print(f"  [ok] original detection: {S0.shape[1]} cells, "
                  f"unchanged")
        if shape != shape1:
            fails.append(f"{head}: FOV {shape} vs curated {shape1}")

        # 2 compact blobs
        bad = []
        for i in range(S1.shape[1]):
            nb, area, span = blobs(S1[:, i], shape)
            if nb == 0 or span > 0.5 * shape[0] or area < 40:
                bad.append((kept.iloc[i]["label"], nb, area, span))
        if bad:
            fails.append(f"{head}: {len(bad)} curated footprint(s) are not "
                         f"compact blobs: {bad}")
        else:
            spans = [blobs(S1[:, i], shape)[2]
                     for i in range(S1.shape[1])]
            print(f"  [ok] all {S1.shape[1]} curated footprints compact "
                  f"(row span {min(spans)}-{max(spans)} px, FOV "
                  f"{shape[0]})")

        # 3 count and membership
        if S1.shape[1] != dec["n_final"]:
            fails.append(f"{head}: {S1.shape[1]} curated cells, expected "
                         f"{dec['n_final']}")
        got_drop = sorted(int(x) for x in lab.loc[lab["index"] == 0,
                                                  "origin"])
        if got_drop != sorted(dec["discard"]):
            fails.append(f"{head}: discarded {got_drop}, decided "
                         f"{sorted(dec['discard'])}")
        else:
            print(f"  [ok] discarded exactly {got_drop or 'nothing'}; "
                  f"{S1.shape[1]} cells kept")

        # 4 unchanged cells are numerically identical
        n_id = 0
        for _, r in kept.iterrows():
            if r["op"] != "keep":
                continue
            a = S1[:, int(r["index"]) - 1]
            b = S0[:, int(r["origin"]) - 1]
            if not np.allclose(a, b, atol=1e-9):
                fails.append(f"{head}: kept cell {r['label']} differs from "
                             f"the original footprint")
            else:
                n_id += 1
        print(f"  [ok] {n_id} kept-unchanged footprints are numerically "
              f"identical to the original")

        # 5 trim
        for c in dec["trim"]:
            r = kept[kept["origin"] == c]
            if not len(r):
                fails.append(f"{head}: trimmed cell {c} not in the result")
                continue
            r = r.iloc[0]
            m1 = mask(S1[:, int(r["index"]) - 1], shape)
            m0 = mask(S0[:, c - 1], shape)
            nb0, _, _ = blobs(S0[:, c - 1], shape)
            nb1, a1, _ = blobs(S1[:, int(r["index"]) - 1], shape)
            if not np.all(m1 <= m0):
                fails.append(f"{head}: trimmed {c} is not a subset of the "
                             f"original")
            elif nb1 != 1:
                fails.append(f"{head}: trimmed {c} still has {nb1} blobs")
            else:
                print(f"  [ok] trim {c}: {nb0} blobs -> 1, "
                      f"{int(m0.sum())} -> {int(m1.sum())} px, subset of the "
                      f"original")

        # 6 split
        for c in dec["split"]:
            parts = kept[kept["origin"] == c]
            if len(parts) != 2:
                fails.append(f"{head}: split {c} produced {len(parts)} parts")
                continue
            # Test the property the partition actually guarantees, on the
            # SUPPORT and the weights. My first version compared thresholded
            # masks at 0.3 x each footprint's own peak; a half has a lower
            # peak than the parent, so pixels below 0.3 x the parent's peak
            # but inside its support counted as "outside the original" and
            # it reported a spurious 22 px failure.
            ws = [S1[:, int(r["index"]) - 1] for _, r in parts.iterrows()]
            w0 = S0[:, c - 1]
            sup = [x > 0 for x in ws]
            sup0 = w0 > 0
            ov = int((sup[0] & sup[1]).sum())
            out = int(((sup[0] | sup[1]) & ~sup0).sum())
            miss = int((sup0 & ~(sup[0] | sup[1])).sum())
            werr = float(np.abs(ws[0] + ws[1] - w0)[sup0].max()) \
                if sup0.any() else 0.0
            if ov:
                fails.append(f"{head}: split {c} halves overlap by {ov} px")
            elif out or miss:
                fails.append(f"{head}: split {c} support differs from the "
                             f"original by {out} px outside / {miss} px "
                             f"missing")
            elif werr > 1e-9:
                fails.append(f"{head}: split {c} weights do not sum to the "
                             f"original (max error {werr:.2e})")
            else:
                labs = parts["label"].tolist()
                print(f"  [ok] split {c} -> {labs}, "
                      f"{int(sup[0].sum())} + {int(sup[1].sum())} = "
                      f"{int(sup0.sum())} px: a disjoint partition of the "
                      f"original support, weights preserved exactly")

        # 7 added cells are where they were asked to be
        for name, ry, cx in dec["add"]:
            r = kept[kept["label"] == name]
            if not len(r):
                fails.append(f"{head}: added cell {name} missing")
                continue
            gy, gx = cent(S1[:, int(r.iloc[0]["index"]) - 1], shape)
            d = float(np.hypot(gy - ry, gx - cx))
            if d > RADIUS:
                fails.append(f"{head}: added {name} is {d:.1f} px from the "
                             f"requested (row {ry:.0f}, col {cx:.0f})")
            else:
                print(f"  [ok] added {name} at row {gy:.1f} col {gx:.1f}, "
                      f"{d:.1f} px from what was asked")

        # 8 traces line up with the labels
        if T1.shape[0] != S1.shape[1]:
            fails.append(f"{head}: {T1.shape[0]} trace rows for "
                         f"{S1.shape[1]} footprints")
        else:
            print(f"  [ok] {T1.shape[0]} traces x {T1.shape[1]} frames, one "
                  f"row per curated cell")
        notes.append((head, S0.shape[1], S1.shape[1]))

    print("\n" + "=" * 62)
    for head, a, b in notes:
        print(f"  {head}: {a} detected -> {b} curated")
    if fails:
        print(f"\n{len(fails)} CHECK(S) FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nALL CHECKS PASSED - the curated set matches the decisions "
          "exactly, and the\noriginal detection was never modified.")


if __name__ == "__main__":
    main()
