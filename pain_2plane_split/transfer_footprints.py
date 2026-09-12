"""transfer_footprints.py  -  measure every neuron in BOTH sessions.

WHY THIS EXISTS
  Matching detections across sessions caps the usable set at the neurons
  EXTRACT happened to find twice: 20 position pairs, 17 of them confirmed by
  footprint shape, out of 27 pain and 28 open-field cells.

  But the audit (match_audit.py) shows that is not an anatomy limit. 25 of 27
  pain cells have a soma at their location in the OPEN-FIELD mean image, and
  20 of 28 open-field cells have one in the PAIN mean image. Nearly every
  cell is visible in both fields. The limit is DETECTION: EXTRACT only finds
  cells that fluctuate, so a neuron that was quiet in one session gives it
  nothing to find there.

  The fix is not a looser matching threshold. It is to stop requiring
  detection twice: take the UNION of both sessions' footprints, register it,
  and extract every footprint from BOTH movies. Detection then has to
  succeed once per neuron, and the same set is measured in both sessions by
  construction.

HOW THE UNION IS BUILT
  Every pain footprint, plus every open-field footprint that does not match
  one - so a neuron found twice appears once, with the pain footprint as its
  canonical shape. Matching for this purpose uses the best of three
  transforms (translation, similarity, full affine), chosen per plane by the
  residual it leaves.

WHAT COMES OUT, AND WHAT TO CHECK BEFORE USING IT
  A transferred footprint is a hypothesis, not a detection. Two numbers are
  reported per cell per session so a bad transfer is visible:
    anat    the footprint's brightness over its surrounding ring in that
            session's mean image, in units of the image SD. Low means the
            transfer landed on nothing - either the registration is off
            there or the soma is genuinely absent at that depth.
    active  whether the transferred trace is separable from background by
            the same test used elsewhere (skew and lag-1 autocorrelation
            against 200 shape-matched background ROIs).

OUTPUT  ->  <pain session>\\output_split\\match\\
  union_cells.csv        one row per neuron: labels, source, anat and
                         verdict in each session
  union_traces.mat       raw F, dF/F and z for every neuron in BOTH sessions
  union_transfer.png     the union on both fields, and the anat check

USAGE
  python transfer_footprints.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.optimize import linear_sum_assignment

from extract_qc_cells import CHUNK, detrend, metrics, to_dff
from match_audit import anat, bp, grid_shift, shape_r

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
RADIUS = 8
SHAPE_MIN = 0.5
N_CTRL = 200
CTRL_MARGIN = 12
RNG = np.random.default_rng(0)


def load(root, plane):
    base = os.path.join(root, f"plane_{plane}")
    cur = os.path.join(base, "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mn = np.array(h["output"]["info"]["summary_image"]).T
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
                lab=lb["label"].tolist(), mean=mn,
                mov=os.path.join(base, f"plane_{plane}_1.h5"))


def warp_pts(pts, M):
    p = np.hstack([pts[:, ::-1], np.ones((len(pts), 1))])
    return (M @ p.T).T[:, ::-1]


def assign(a, b, tol):
    D = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    big = D.max() + 1e6
    r, c = linear_sum_assignment(np.where(D <= tol, D, big))
    return [(i, j, float(D[i, j])) for i, j in zip(r, c) if D[i, j] <= tol]


def best_transform(o, p, L):
    """Open field -> pain. grid_shift(A, B) maps A's coords into B's."""
    c0, dy, dx = grid_shift(bp(o["mean"]), bp(p["mean"]))
    M = np.array([[1, 0, dx], [0, 1, dy]], float)
    cands = [("translation", M)]
    seed = assign(warp_pts(o["cent"], M), p["cent"], RADIUS)
    if len(seed) >= 3:
        src = np.float32([o["cent"][i][::-1] for i, _, _ in seed])
        dst = np.float32([p["cent"][j][::-1] for _, j, _ in seed])
        for nm, fn in (("similarity", cv2.estimateAffinePartial2D),
                       ("affine", cv2.estimateAffine2D)):
            M2, _ = fn(src, dst, method=cv2.RANSAC,
                       ransacReprojThreshold=3.0)
            if M2 is not None:
                cands.append((nm, M2))
    scored = []
    for nm, Mi in cands:
        pr = assign(warp_pts(o["cent"], Mi), p["cent"], RADIUS)
        res = np.median([d for _, _, d in pr]) if pr else np.inf
        scored.append((len(pr), -res, nm, Mi))
        L.append(f"   {nm:12s} {len(pr):3d} pairs, residual median "
                 f"{res:.2f} px")
    scored.sort(reverse=True)
    L.append(f"   image correlation at the translation: {c0:+.4f}   "
             f"-> using {scored[0][2]}")
    return scored[0][3], scored[0][2]


def make_controls(S, shape, n):
    hh, w = shape
    occ = (S.sum(1).reshape(hh, w) > 0).astype(np.uint8)
    keep = cv2.dilate(occ, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * CTRL_MARGIN + 1,) * 2)) > 0
    cols, tries = [], 0
    k = S.shape[1]
    while len(cols) < n and tries < n * 200:
        tries += 1
        c = int(RNG.integers(k))
        img = S[:, c].reshape(hh, w)
        ys, xs = np.nonzero(img)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        patch = img[y0:y1 + 1, x0:x1 + 1]
        ph, pw = patch.shape
        ny, nx = int(RNG.integers(0, hh - ph)), int(RNG.integers(0, w - pw))
        if keep[ny:ny + ph, nx:nx + pw][patch > 0].any():
            continue
        blank = np.zeros((hh, w), np.float32)
        blank[ny:ny + ph, nx:nx + pw] = patch
        cols.append(blank.ravel())
    return np.array(cols).T if cols else np.zeros((hh * w, 0), np.float32)


def solve(mov, W, shape):
    """Joint least squares for every column of W, chunked over frames."""
    hh, w = shape
    Wn = W / np.maximum(W.sum(0, keepdims=True), 1e-12)
    A = Wn.astype(np.float64)
    AtA = A.T @ A
    parts = []
    with h5py.File(mov, "r") as f:
        mv = f["mov"]
        T = mv.shape[0]
        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            blk = np.asarray(mv[t0:t1]).transpose(0, 2, 1).reshape(
                t1 - t0, hh * w)
            parts.append(A.T @ blk.T.astype(np.float64))
    return np.linalg.solve(AtA, np.concatenate(parts, 1))


def verdicts(dff_cells, dff_ctrl):
    """active / not, calibrated on the control ROIs, as used elsewhere."""
    det_c, _ = detrend(dff_cells)
    det_k, _ = detrend(dff_ctrl)
    mc = [metrics(det_c[i])[0] for i in range(det_c.shape[0])]
    mk = [metrics(det_k[j])[0] for j in range(det_k.shape[0])]
    nsk = np.array([m["skew"] for m in mk])
    nac = np.array([m["ac1"] for m in mk])

    def ep(null, x):
        return (1. + np.sum(null >= x)) / (len(null) + 1.)

    def fisher(s, a):
        return -2. * (np.log(ep(nsk, s)) + np.log(ep(nac, a)))

    nullX = np.array([fisher(s, a) for s, a in zip(nsk, nac)])
    pj = []
    for m in mc:
        X = fisher(m["skew"], m["ac1"])
        pj.append((1. + np.sum(nullX >= X)) / (len(nullX) + 1.))
    return np.array(pj), mc


def run_plane(plane, L):
    o, p = load(OF, plane), load(PA, plane)
    hh, w = p["shape"]
    L.append(f"\n{'=' * 66}\nplane {plane}\n{'=' * 66}")
    M, nm = best_transform(o, p, L)
    Minv = cv2.invertAffineTransform(M)

    pr = assign(warp_pts(o["cent"], M), p["cent"], RADIUS)
    keep_pairs = []
    for i, j, d in pr:
        wi = cv2.warpAffine(o["S"][:, i].reshape(hh, w), M, (w, hh))
        r = shape_r(wi.ravel(), p["S"][:, j], (hh, w))
        if r >= SHAPE_MIN:
            keep_pairs.append((i, j, d, r))
    matched_of = {i for i, _, _, _ in keep_pairs}
    matched_pa = {j for _, j, _, _ in keep_pairs}
    L.append(f"   {len(pr)} position pairs, {len(keep_pairs)} also agree in "
             f"shape (r >= {SHAPE_MIN})")

    rows, Spa, Sof = [], [], []
    for j in range(p["S"].shape[1]):
        col = p["S"][:, j]
        Spa.append(col)
        Sof.append(cv2.warpAffine(col.reshape(hh, w), Minv,
                                  (w, hh)).ravel())
        mate = [i for i, jj, _, _ in keep_pairs if jj == j]
        rows.append(dict(plane=plane, source="pain",
                         pain_cell=p["lab"][j],
                         of_cell=o["lab"][mate[0]] if mate else None,
                         matched=bool(mate)))
    for i in range(o["S"].shape[1]):
        if i in matched_of:
            continue
        col = o["S"][:, i]
        Sof.append(col)
        Spa.append(cv2.warpAffine(col.reshape(hh, w), M, (w, hh)).ravel())
        rows.append(dict(plane=plane, source="open field", pain_cell=None,
                         of_cell=o["lab"][i], matched=False))
    Spa = np.stack(Spa, 1).astype(np.float32)
    Sof = np.stack(Sof, 1).astype(np.float32)
    n = Spa.shape[1]
    L.append(f"   union: {p['S'].shape[1]} pain + "
             f"{o['S'].shape[1] - len(matched_of)} open-field-only = {n} "
             f"neurons, each measured in BOTH sessions")

    out = {}
    roots = {"pain": PA, "openfield": OF}
    for tag, root, S_, obj in (("pain", PA, Spa, p),
                               ("openfield", OF, Sof, o)):
        C = make_controls(S_, (hh, w), N_CTRL)
        F = solve(obj["mov"], np.hstack([S_, C]), (hh, w))
        dff = to_dff(F)
        pj, mc = verdicts(dff[:n], dff[n:])
        det, _ = detrend(dff[:n])
        z = np.stack([(det[i] - np.median(det[i]))
                      / max(1.4826 * np.median(np.abs(
                          det[i] - np.median(det[i]))), 1e-12)
                      for i in range(n)])
        an = np.array([anat(obj["mean"], S_[:, i], (hh, w))
                       for i in range(n)])
        out[tag] = dict(F=F[:n], dff=dff[:n], z=z, p=pj, anat=an,
                        ev=[m["events"] for m in mc], S=S_)
        L.append(f"   {tag:10s} {n} traces, active "
                 f"{int((pj <= .05).sum())}, anat median {np.nanmedian(an):.2f}"
                 f" SD, below 1 SD: {int(np.nansum(an < 1))}")

        # Write the union per session in the same shape the per-session
        # scripts already read, so every downstream analysis can switch to
        # the 39-cell set without each one re-deriving the transfer.
        tdir = os.path.join(roots[tag], "timestamps")
        tf = pd.read_csv(os.path.join(tdir, "plane_frame_times.csv"))
        t_s = tf.loc[tf["plane"] == plane, "t_s"].to_numpy()
        if len(t_s) != F.shape[1]:
            raise SystemExit(f"{tag} plane {plane}: {F.shape[1]} samples "
                             f"but {len(t_s)} frame times")
        udir = os.path.join(roots[tag], "union", f"plane_{plane}")
        os.makedirs(udir, exist_ok=True)
        uid = [f"{plane}{i + 1}" for i in range(n)]
        savemat(os.path.join(udir, "union_traces.mat"),
                dict(labels=np.array(uid, dtype=object),
                     verdict=np.array(["active" if q <= .05 else "quiet"
                                       for q in pj], dtype=object),
                     t_s=t_s, F_raw=F[:n], dff=dff[:n], z=z,
                     anat_sd=an, p_active=pj,
                     spatial_weights=S_.reshape(hh, w, n),
                     source=np.array([r["source"] for r in rows],
                                     dtype=object),
                     matched=np.array([bool(r["matched"]) for r in rows]),
                     fs_hz=1.0 / np.median(np.diff(t_s)),
                     note="union of both sessions' footprints, registered "
                          "and solved jointly on this session's movie"))
    for i, r in enumerate(rows):
        for tag in ("pain", "openfield"):
            r[f"{tag}_anat"] = round(float(out[tag]["anat"][i]), 3)
            r[f"{tag}_p"] = round(float(out[tag]["p"][i]), 4)
            r[f"{tag}_active"] = bool(out[tag]["p"][i] <= .05)
            r[f"{tag}_events"] = int(out[tag]["ev"][i])
    return rows, out, nm


def main():
    L = ["===== every neuron measured in both sessions =====",
         "",
         "Matching detections caps the set at neurons EXTRACT found twice.",
         "The audit showed 25 of 27 pain cells and 20 of 28 open-field cells",
         "have a soma at their location in the OTHER session's mean image, so",
         "the limit is detection, not anatomy. Here the union of both",
         "sessions' footprints is registered and extracted from BOTH movies,",
         "so detection has to succeed once per neuron instead of twice."]
    allrows, stores = [], {}
    for plane in ("A", "B"):
        rows, out, nm = run_plane(plane, L)
        allrows += rows
        stores[plane] = out
    D = pd.DataFrame(allrows)
    D.insert(0, "uid", [f"{r['plane']}{i + 1}" for i, (_, r)
                        in enumerate(D.iterrows())])
    # renumber within plane
    uid = []
    for plane in ("A", "B"):
        s = D[D["plane"] == plane]
        uid += [f"{plane}{i + 1}" for i in range(len(s))]
    D["uid"] = uid

    outdir = os.path.join(PA, "match")
    D.to_csv(os.path.join(outdir, "union_cells.csv"), index=False)
    md = {}
    for plane in ("A", "B"):
        for tag in ("pain", "openfield"):
            for k in ("F", "dff", "z"):
                md[f"{plane}_{tag}_{k}"] = stores[plane][tag][k]
    md["uid"] = np.array(D["uid"].tolist(), dtype=object)
    md["plane"] = np.array(D["plane"].tolist(), dtype=object)
    md["source"] = np.array(D["source"].tolist(), dtype=object)
    savemat(os.path.join(outdir, "union_traces.mat"), md)

    both = D[(D["pain_anat"] > 1) & (D["openfield_anat"] > 1)]
    act_both = D[D["pain_active"] & D["openfield_active"]]
    act_either = D[D["pain_active"] | D["openfield_active"]]
    L += ["", "=" * 66, "SUMMARY", "=" * 66,
          f"  union neurons, each with a trace in both sessions: {len(D)}",
          f"    plane A {int((D['plane'] == 'A').sum())}, "
          f"plane B {int((D['plane'] == 'B').sum())}",
          f"  detected in both sessions independently (matched): "
          f"{int(D['matched'].sum())}",
          f"  footprint lands on a soma (> 1 SD) in BOTH sessions: "
          f"{len(both)}",
          f"  active in both sessions: {len(act_both)}",
          f"  active in at least one: {len(act_either)}",
          "",
          "A transferred footprint is a hypothesis, not a detection. Use "
          "pain_anat and",
          "openfield_anat to drop the ones that landed on nothing, and the "
          "per-session",
          "active flags to say where a neuron was firing. Everything is in "
          "union_cells.csv.",
          "",
          "This is the set to carry the pain responsiveness into once the "
          "video is scored:",
          "every row already has an open-field trace and a pain trace for "
          "the same neuron."]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(outdir, "union_transfer.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
