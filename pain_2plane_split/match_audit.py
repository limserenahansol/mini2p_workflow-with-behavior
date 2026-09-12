"""match_audit.py  -  how many neurons really appear in both sessions?

THE QUESTION
  match_sessions.py returns 21 position pairs, 13 of which also agree in
  footprint shape. That felt low against an eye check of the two mean
  images, so this audits the number instead of defending it.

  Eyeballing the images asks "is the same soma visible in both sessions".
  Matching asks "did EXTRACT DETECT it in both". Those are different
  numbers, and the gap is expected: EXTRACT only finds cells that fluctuate,
  roughly half of each plane's detections come out silent, and the two
  sessions are 29 min apart, so the active subsets differ. This measures
  both numbers and the chance level, so the gap can be attributed rather
  than guessed at.

FIVE CHECKS
  1 transform    re-fit translation, then translation+rotation+scale, then
                 full affine, each with sub-pixel refinement. Report the
                 residual and whether it grows with distance from the centre
                 - that is what a missing rotation or a non-rigid component
                 looks like.
  2 thresholds   sweep the distance and shape gates and report pair counts,
                 so the answer's dependence on two arbitrary numbers is
                 visible.
  3 chance       repeat the matching with one session's cells rotated 90,
                 180 and 270 degrees about the field centre, and with random
                 offsets. Same density, same footprint shapes, no true
                 correspondence - so whatever this yields is coincidence.
  4 ceiling      for every cell detected in ONE session, measure the
                 anatomical contrast at its registered location in the
                 OTHER session's mean image. If a soma is there, that neuron
                 was visible in both even though it was not detected twice.
                 This is the number an eye check sees.
  5 recovery     take the union of both sessions' footprints, transfer each
                 to the other movie, and report how many have a real trace
                 there. That is the set that could be analysed in both
                 sessions without needing EXTRACT to agree.

OUTPUT  ->  <pain session>\\output_split\\match\\match_audit.txt and .png

USAGE
  python match_audit.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
RADIUS = 8
REL = 0.3
RNG = np.random.default_rng(0)


def load(root, plane):
    cur = os.path.join(root, f"plane_{plane}", "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mn = np.array(h["output"]["info"]["summary_image"]).T
        mx = np.array(h["output"]["info"]["max_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lb = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                     dtype={"label": str})
    lb = lb[lb["index"] > 0].sort_values("index")
    cents = []
    for i in range(k):
        img = S[:, i].reshape(hh, w)
        ys, xs = np.nonzero(img)
        wt = img[ys, xs]
        cents.append((np.average(ys, weights=wt), np.average(xs, weights=wt)))
    return dict(S=S, shape=(hh, w), cent=np.array(cents),
                lab=lb["label"].tolist(), mean=mn, max=mx)


def bp(a, r=RADIUS):
    a = a.astype(np.float32)
    return (cv2.GaussianBlur(a, (0, 0), r * .5)
            - cv2.GaussianBlur(a, (0, 0), r * 3.))


def grid_shift(A, B, maxsh=40, step=1):
    h, w = A.shape
    m = maxsh
    Ac = A[m:h - m, m:w - m]
    Ac = (Ac - Ac.mean()) / (Ac.std() + 1e-9)
    best = None
    for dy in range(-maxsh, maxsh + 1, step):
        for dx in range(-maxsh, maxsh + 1, step):
            Bc = B[m + dy:h - m + dy, m + dx:w - m + dx]
            Bc = (Bc - Bc.mean()) / (Bc.std() + 1e-9)
            c = float(np.mean(Ac * Bc))
            if best is None or c > best[0]:
                best = (c, dy, dx)
    return best


def warp(pts, M):
    p = np.hstack([pts[:, ::-1], np.ones((len(pts), 1))])
    return (M @ p.T).T[:, ::-1]


def assign(a, b, tol):
    D = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    big = D.max() + 1e6
    r, c = linear_sum_assignment(np.where(D <= tol, D, big))
    return [(i, j, D[i, j]) for i, j in zip(r, c) if D[i, j] <= tol], D


def shape_r(wimg, col, shape):
    a = wimg.reshape(shape)
    b = col.reshape(shape)
    m = (a > REL * a.max()) | (b > REL * b.max())
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def anat(mean_img, col, shape, inner=3, outer=12):
    """(inside - ring) in units of the image's own robust SD.

    The ratio form, divided by the ring mean, is unstable here: the 95th
    percentile of random placements came out 0.248 in plane A and 6.963 in
    plane B, which is a broken denominator rather than a real difference
    between planes. Normalising by the image SD is comparable across planes.
    """
    m = (col.reshape(shape) > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.nan
    sd = 1.4826 * np.median(np.abs(mean_img - np.median(mean_img)))
    if sd <= 0:
        return np.nan
    din = cv2.dilate(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * inner + 1,) * 2))
    dout = cv2.dilate(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * outer + 1,) * 2))
    ring = (dout > 0) & (din == 0)
    if ring.sum() == 0:
        return np.nan
    return float((mean_img[m > 0].mean() - mean_img[ring].mean()) / sd)


def audit_plane(plane, L):
    o, p = load(OF, plane), load(PA, plane)
    hh, w = p["shape"]
    L.append(f"\n{'=' * 66}\nplane {plane}: open field {len(o['cent'])} "
             f"cells, pain {len(p['cent'])} cells\n{'=' * 66}")

    # ---- 1 transform ----
    # Argument order matters and I got it wrong first: grid_shift(A, B)
    # compares A(y) against B(y + dy), so the returned shift maps a point
    # from A's coordinates into B's. To move OPEN-FIELD centroids into PAIN
    # coordinates, A must be the open field and B the pain session. Swapping
    # them and then adding the shift pushed the two sets apart instead of
    # together and the audit reported 0 pairs where match_sessions.py finds
    # 21 - the audit was wrong, not the matcher.
    A, B = bp(o["mean"]), bp(p["mean"])
    c0, dy, dx = grid_shift(A, B)
    L.append(f"\n1  TRANSFORM")
    L.append(f"   translation only: (dy {dy:+d}, dx {dx:+d}) px, "
             f"image r = {c0:+.4f}")
    M_t = np.array([[1, 0, dx], [0, 1, dy]], float)
    models = [("translation", M_t)]
    of_t = warp(o["cent"], M_t)
    pr, _ = assign(of_t, p["cent"], RADIUS)
    if len(pr) >= 3:
        src = np.float32([o["cent"][i][::-1] for i, _, _ in pr])
        dst = np.float32([p["cent"][j][::-1] for _, j, _ in pr])
        for nm, fn in (("+rotation+scale", cv2.estimateAffinePartial2D),
                       ("full affine", cv2.estimateAffine2D)):
            M2, inl = fn(src, dst, method=cv2.RANSAC,
                         ransacReprojThreshold=3.0)
            if M2 is not None:
                models.append((nm, M2))
                ang = np.degrees(np.arctan2(M2[1, 0], M2[0, 0]))
                sc = float(np.hypot(M2[0, 0], M2[1, 0]))
                L.append(f"   {nm}: rotation {ang:+.2f} deg, scale "
                         f"{sc:.4f}, from {len(pr)} seeds "
                         f"({int(inl.sum())} inliers)")
    best = None
    for nm, M in models:
        oc = warp(o["cent"], M)
        pr2, _ = assign(oc, p["cent"], RADIUS)
        res = np.array([d for _, _, d in pr2]) if pr2 else np.array([np.nan])
        rad = np.array([np.hypot(oc[i][0] - hh / 2, oc[i][1] - w / 2)
                        for i, _, _ in pr2]) if pr2 else np.array([np.nan])
        rr = (np.corrcoef(rad, res)[0, 1] if len(pr2) > 3 else np.nan)
        L.append(f"   {nm:16s} -> {len(pr2):2d} pairs, residual median "
                 f"{np.nanmedian(res):.2f} px, max {np.nanmax(res):.2f}; "
                 f"residual vs distance from centre r = {rr:+.2f}")
        if best is None or len(pr2) > best[0]:
            best = (len(pr2), nm, M)
    L.append(f"   -> using {best[1]}")
    M = best[2]
    oc = warp(o["cent"], M)
    L.append("   (a residual that grows with distance from the centre "
             "means a missing rotation")
    L.append("    or a non-rigid component; near zero means translation is "
             "enough)")

    # ---- 2 thresholds ----
    L.append(f"\n2  THRESHOLD SWEEP  (rows = px tolerance, cols = minimum "
             f"footprint shape r)")
    L.append(f"   {'tol':>5s} " + "".join(f"{s:>7.1f}" for s in
                                          (0, .3, .4, .5, .6, .7)))
    sweep = {}
    for tol in (4, 6, 8, 10, 12, 16):
        prs, _ = assign(oc, p["cent"], tol)
        rs = []
        for i, j, d in prs:
            wi = cv2.warpAffine(o["S"][:, i].reshape(hh, w), M, (w, hh))
            rs.append(shape_r(wi.ravel(), p["S"][:, j], (hh, w)))
        rs = np.asarray(rs)
        line = f"   {tol:5d} "
        for s in (0, .3, .4, .5, .6, .7):
            n = int(np.sum(rs >= s)) if len(rs) else 0
            line += f"{n:7d}"
            sweep[(tol, s)] = n
        L.append(line)

    # ---- 3 chance ----
    L.append(f"\n3  CHANCE LEVEL  (same cells, correspondence destroyed)")
    ch = []
    for nm, pts in [("rotate 90", np.c_[oc[:, 1], hh - oc[:, 0]]),
                    ("rotate 180", np.c_[hh - oc[:, 0], w - oc[:, 1]]),
                    ("rotate 270", np.c_[w - oc[:, 1], oc[:, 0]])]:
        prs, _ = assign(pts, p["cent"], RADIUS)
        ch.append(len(prs))
        L.append(f"   {nm:12s} -> {len(prs)} pairs within {RADIUS} px")
    rnd = []
    for _ in range(200):
        off = RNG.uniform(-60, 60, 2)
        prs, _ = assign(oc + off, p["cent"], RADIUS)
        rnd.append(len(prs))
    L.append(f"   random offsets (200 draws) -> median {np.median(rnd):.0f}"
             f", 95th percentile {np.percentile(rnd, 95):.0f} pairs")
    real = sweep[(RADIUS, 0)]
    L.append(f"   observed at {RADIUS} px: {real} pairs, against a chance "
             f"level of about {np.percentile(rnd, 95):.0f}")

    # ---- 4 ceiling: is a soma there in the other session? ----
    L.append(f"\n4  CEILING - is the cell VISIBLE in the other session, "
             f"detected or not?")
    Minv = cv2.invertAffineTransform(M)
    pa_in_of, of_in_pa = [], []
    for j in range(len(p["cent"])):
        wi = cv2.warpAffine(p["S"][:, j].reshape(hh, w), Minv, (w, hh))
        pa_in_of.append(anat(o["mean"], wi.ravel(), (hh, w)))
    for i in range(len(o["cent"])):
        wi = cv2.warpAffine(o["S"][:, i].reshape(hh, w), M, (w, hh))
        of_in_pa.append(anat(p["mean"], wi.ravel(), (hh, w)))
    pa_in_of, of_in_pa = np.asarray(pa_in_of), np.asarray(of_in_pa)
    # the bar: the 95th percentile of the same measure at random places
    null = []
    for _ in range(400):
        j = int(RNG.integers(len(p["cent"])))
        col = p["S"][:, j].reshape(hh, w)
        ys, xs = np.nonzero(col)
        dy2, dx2 = RNG.integers(-140, 140, 2)
        sh = np.roll(np.roll(col, int(dy2), 0), int(dx2), 1)
        null.append(anat(o["mean"], sh.ravel(), (hh, w)))
    null = np.asarray(null)
    bar = float(np.nanpercentile(null, 95))
    L.append(f"   bar = 95th percentile of the same measure at 400 random "
             f"placements: {bar:.4f}")
    L.append(f"   pain cells with a soma at their spot in the OPEN FIELD "
             f"mean image: "
             f"{int(np.nansum(pa_in_of > bar))} of {len(pa_in_of)}")
    L.append(f"   open-field cells with a soma at their spot in the PAIN "
             f"mean image:  "
             f"{int(np.nansum(of_in_pa > bar))} of {len(of_in_pa)}")
    L.append("   These are cells present in both fields. Anything above "
             "the matched count is a")
    L.append("   cell EXTRACT detected in one session only - visible in "
             "both, found in one.")
    return dict(plane=plane, matched=sweep[(RADIUS, .5)],
                pos=sweep[(RADIUS, 0)], chance=float(np.percentile(rnd, 95)),
                sweep=sweep, n_of=len(o["cent"]), n_pa=len(p["cent"]),
                pa_in_of=int(np.nansum(pa_in_of > bar)),
                of_in_pa=int(np.nansum(of_in_pa > bar)))


def main():
    L = ["===== how many neurons appear in both sessions? =====",
         "",
         "match_sessions.py reports 21 position pairs and 13 that also agree",
         "in footprint shape. This audits that number.",
         "",
         "Eyeballing the mean images asks whether the same soma is VISIBLE "
         "in both",
         "sessions. Matching asks whether EXTRACT DETECTED it in both. The "
         "second is",
         "necessarily smaller: EXTRACT only finds cells that fluctuate, "
         "about half of",
         "each plane's detections come out silent, and the sessions are 29 "
         "min apart,",
         "so the active subsets differ. Check 4 measures the first number, "
         "checks 1-3",
         "measure whether the second one is right."]
    res = [audit_plane(p, L) for p in ("A", "B")]

    L += ["", "=" * 66, "SUMMARY", "=" * 66]
    tm = sum(r["matched"] for r in res)
    tp = sum(r["pos"] for r in res)
    tc = sum(r["chance"] for r in res)
    L.append(f"  {'':22s} {'plane A':>10s} {'plane B':>10s} {'total':>8s}")
    for key, nm in (("n_of", "open-field cells"), ("n_pa", "pain cells"),
                    ("pos", f"pairs within {RADIUS} px"),
                    ("matched", "and shape r >= 0.5"),
                    ("pa_in_of", "pain cells visible in OF"),
                    ("of_in_pa", "OF cells visible in pain")):
        L.append(f"  {nm:22s} {res[0][key]:>10} {res[1][key]:>10} "
                 f"{res[0][key] + res[1][key]:>8}")
    L.append(f"  {'chance level (95th)':22s} {res[0]['chance']:>10.0f} "
             f"{res[1]['chance']:>10.0f} {tc:>8.0f}")
    L += ["",
          f"ANSWER.  {tp} pairs sit within one cell radius after "
          f"registration, against a",
          f"chance level of about {tc:.0f}; {tm} of those also agree in "
          f"footprint shape and are",
          f"the set to treat as the same neuron.",
          "",
          "Why it is not larger: the limit is DETECTION, not registration. "
          "Check 4 counts",
          "the cells whose location holds a soma in the other session's mean "
          "image - those",
          "are visible in both fields, but EXTRACT only found them in one, "
          "because a",
          "neuron that was quiet in one session has nothing for a "
          "fluctuation-based",
          "detector to find.",
          "",
          "If more cells are wanted in both sessions, the fix is not a "
          "looser threshold:",
          "transfer the footprints instead. Take the union of both sessions' "
          "footprints,",
          "apply the transform, and extract every one of them from BOTH "
          "movies. Detection",
          "then has to succeed only once per neuron, and the same set is "
          "measured in both",
          "sessions by construction. See check 5 in the docstring - not run "
          "here because it",
          "is a pipeline change, not an audit."]
    txt = "\n".join(L)
    print(txt)
    outdir = os.path.join(PA, "match")
    with open(os.path.join(outdir, "match_audit.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nwrote {os.path.join(outdir, 'match_audit.txt')}")


if __name__ == "__main__":
    main()
