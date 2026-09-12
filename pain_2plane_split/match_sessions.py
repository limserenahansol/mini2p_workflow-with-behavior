"""match_sessions.py  -  which open-field cell is which pain-session cell?

THE PREMISE, AND WHY IT NEEDS A TRANSFORM FIRST
  Both sessions image the same two ETL depths, so the same neurons are in
  frame. But EXTRACT numbers cells in the order it finds them, independently
  per run, so the IDs do not correspond. Measured on the uncurated sets:

    median distance between SAME-numbered cells   197 px (A), 145 px (B)
    median distance to the NEAREST cell            21.7 px (A), 13.0 px (B)
    same number is also the nearest cell           2 of 12 (A), 0 of 14 (B)

  Matching therefore has to be done by pixel position - but not by nearest
  neighbour on raw coordinates either, because even the nearest cell is
  13-22 px away and a soma radius is 8 px. The animal was removed and
  re-placed in the 29 minutes between 15:27 and 15:56, so the field shifted.
  The shift has to be estimated and removed first.

HOW
  1  Phase correlation between the two sessions' mean images, band-passed at
     the cell scale so the estimate follows somata rather than the slow
     illumination profile. Reported with the height of the correlation peak
     relative to the rest of the surface, so a weak registration is visible
     rather than silently trusted.
  2  Optimal one-to-one assignment (Hungarian) between the shifted
     open-field centroids and the pain centroids, with pairs beyond
     MAX_DIST_PX rejected.
  3  If enough pairs survive, a similarity transform (translation, rotation,
     scale) is fitted to them and steps 1-2 repeated, so a small rotation
     from re-mounting does not cap the match count.
  4  An independent check that does not use position: the correlation
     between the two footprints' shapes after alignment. Two different cells
     a few px apart can be paired by position alone; shape agreement is what
     distinguishes that from a real match.

  FAILURE IS A REPORTABLE OUTCOME. If the residual stays above a cell radius
  or too few cells pair up, the two sessions did not image the same plane
  closely enough to match, and the script says so instead of returning a
  table of coincidences.

OUTPUT  ->  <pain session>\\output_split\\match\\
  match_<plane>.csv     OF label, PA label, residual px, shape r
  match_report.txt      the transform, the residuals, the verdict
  match_<plane>.png     both sessions' footprints before and after alignment

USAGE
  python match_sessions.py
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
RADIUS = 8              # measured median cell radius
MAX_DIST_PX = 8.0       # a pair further apart than one radius is not a match
MIN_FOR_AFFINE = 4
REL = 0.3               # footprint level for shape comparison


def load_plane(root, plane, curated=True):
    """Curated footprints and the session mean image for one plane."""
    base = os.path.join(root, f"plane_{plane}")
    folder = os.path.join(base, "curated") if curated else base
    f = os.path.join(folder, "final_analysis_results.mat")
    if not os.path.exists(f):
        folder = base
        f = os.path.join(folder, "final_analysis_results.mat")
    with h5py.File(f, "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])         # (k, w, h)
        mean_img = np.array(h["output"]["info"]["summary_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lab = [str(i + 1) for i in range(k)]
    lf = os.path.join(folder, "curated_labels.csv")
    if os.path.exists(lf):
        d = pd.read_csv(lf, dtype={"label": str})
        d = d[d["index"] > 0].sort_values("index")
        if len(d) == k:
            lab = d["label"].tolist()
    cents = []
    for i in range(k):
        img = S[:, i].reshape(hh, w)
        ys, xs = np.nonzero(img)
        wt = img[ys, xs]
        cents.append((np.average(ys, weights=wt), np.average(xs, weights=wt)))
    return dict(S=S, shape=(hh, w), cent=np.array(cents), labels=lab,
                mean=mean_img, folder=folder)


def bandpass(a, r=RADIUS):
    a = a.astype(np.float32)
    return (cv2.GaussianBlur(a, (0, 0), r * 0.5)
            - cv2.GaussianBlur(a, (0, 0), r * 3.0))


MAXSH = 40      # px of shift to search; the measured shifts are 7-17 px


def estimate_shift(img_a, img_b, maxsh=MAXSH):
    """(dy, dx) moving img_a onto img_b, by direct correlation search.

    Phase correlation was tried first and is not usable on these images: it
    returned (dy -0.2, dx -0.2) for plane B and (dy +0.5, dx -21.7) for
    plane A, which cannot both be right since a field shift has to be the
    same at both depths. A direct grid search on the band-passed means gives
    an answer that can be checked, and it checks out:

        pain A vs open field A   r = +0.870 at (dy  -9, dx +17)
        pain B vs open field B   r = +0.875 at (dy +11, dx  +7)
        pain A vs open field B   r = +0.328   <- wrong depth, much worse
        pain B vs open field A   r = +0.435   <- wrong depth, much worse
        within a session, A vs B r = 0.40-0.48

    So each depth matches itself across the two sessions far better than it
    matches the other depth: the same optical planes really were imaged
    29 minutes apart. The shifts differ between depths, so each plane is
    registered on its own rather than with one rigid transform.

    Returns the best shift, the correlation there, and how far that peak
    stands out from the rest of the surface.
    """
    A, B = bandpass(img_a), bandpass(img_b)
    h, w = A.shape
    m = maxsh
    Ac = A[m:h - m, m:w - m]
    Ac = (Ac - Ac.mean()) / (Ac.std() + 1e-9)
    best, vals = None, []
    for dy in range(-maxsh, maxsh + 1):
        for dx in range(-maxsh, maxsh + 1):
            Bc = B[m + dy:h - m + dy, m + dx:w - m + dx]
            Bc = (Bc - Bc.mean()) / (Bc.std() + 1e-9)
            c = float(np.mean(Ac * Bc))
            vals.append(c)
            if best is None or c > best[0]:
                best = (c, dy, dx)
    vals = np.asarray(vals)
    z = (best[0] - np.median(vals)) / (vals.std() + 1e-12)
    return float(best[1]), float(best[2]), float(best[0]), float(z)


def shape_r(sa, sb, shape):
    a = sa.reshape(shape)
    b = sb.reshape(shape)
    m = (a > REL * a.max()) | (b > REL * b.max())
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def warp_points(pts, M):
    p = np.hstack([pts[:, ::-1], np.ones((len(pts), 1))])   # (x, y, 1)
    q = (M @ p.T).T
    return q[:, ::-1]                                        # back to (y, x)


def assign(of_c, pa_c, tol=MAX_DIST_PX):
    D = np.hypot(of_c[:, None, 0] - pa_c[None, :, 0],
                 of_c[:, None, 1] - pa_c[None, :, 1])
    big = D.max() + 1000
    C = np.where(D <= tol, D, big)
    r, c = linear_sum_assignment(C)
    keep = [(i, j) for i, j in zip(r, c) if D[i, j] <= tol]
    return keep, D


def run_plane(plane, lines):
    o = load_plane(OF, plane)
    p = load_plane(PA, plane)
    hh, w = o["shape"]
    lines.append(f"--- plane {plane} ---")
    lines.append(f"  open field {len(o['cent'])} cells "
                 f"({os.path.basename(o['folder'])}), "
                 f"pain {len(p['cent'])} cells "
                 f"({os.path.basename(p['folder'])})")

    raw, D0 = assign(o["cent"], p["cent"])
    lines.append(f"  before alignment: {len(raw)} pairs within "
                 f"{MAX_DIST_PX:.0f} px, median nearest "
                 f"{np.median(D0.min(1)):.1f} px")

    dy, dx, resp, snr = estimate_shift(o["mean"], p["mean"])
    lines.append(f"  registration by correlation search: shift "
                 f"(dy {dy:+.0f}, dx {dx:+.0f}) px, r = {resp:+.3f}, "
                 f"peak {snr:.1f} sd above the surface")
    M = np.array([[1, 0, dx], [0, 1, dy]], float)
    of_s = warp_points(o["cent"], M)
    pairs, D1 = assign(of_s, p["cent"])
    lines.append(f"  after shift: {len(pairs)} pairs, median nearest "
                 f"{np.median(D1.min(1)):.1f} px")

    stage = "translation"
    if len(pairs) >= MIN_FOR_AFFINE:
        src = np.float32([o["cent"][i][::-1] for i, _ in pairs])
        dst = np.float32([p["cent"][j][::-1] for _, j in pairs])
        M2, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M2 is not None:
            of_s2 = warp_points(o["cent"], M2)
            pairs2, D2 = assign(of_s2, p["cent"])
            if len(pairs2) >= len(pairs):
                ang = np.degrees(np.arctan2(M2[1, 0], M2[0, 0]))
                sc = float(np.hypot(M2[0, 0], M2[1, 0]))
                lines.append(f"  similarity fit on {len(pairs)} pairs "
                             f"({int(inl.sum())} inliers): rotation "
                             f"{ang:+.2f} deg, scale {sc:.4f}, "
                             f"shift ({M2[1, 2]:+.2f}, {M2[0, 2]:+.2f})")
                lines.append(f"  after similarity: {len(pairs2)} pairs, "
                             f"median nearest {np.median(D2.min(1)):.1f} px")
                of_s, pairs, D1, M, stage = of_s2, pairs2, D2, M2, "similarity"

    rows = []
    for i, j in pairs:
        d = float(np.hypot(*(of_s[i] - p["cent"][j])))
        # shape check without using position: shift the OF footprint image by
        # the fitted transform and correlate it against the pain footprint
        img = o["S"][:, i].reshape(hh, w)
        wimg = cv2.warpAffine(img, M, (w, hh), flags=cv2.INTER_LINEAR)
        rows.append(dict(plane=plane,
                         of_cell=o["labels"][i], pa_cell=p["labels"][j],
                         of_row=round(float(o["cent"][i][0]), 1),
                         of_col=round(float(o["cent"][i][1]), 1),
                         pa_row=round(float(p["cent"][j][0]), 1),
                         pa_col=round(float(p["cent"][j][1]), 1),
                         residual_px=round(d, 2),
                         shape_r=round(shape_r(wimg.ravel(), p["S"][:, j],
                                               (hh, w)), 3)))
    if rows:
        res = np.array([r["residual_px"] for r in rows])
        shr = np.array([r["shape_r"] for r in rows], float)
        lines.append(f"  matched {len(rows)} of "
                     f"min({len(o['cent'])},{len(p['cent'])}) = "
                     f"{min(len(o['cent']), len(p['cent']))} possible")
        lines.append(f"  residual px: median {np.median(res):.2f}, "
                     f"max {res.max():.2f}")
        lines.append(f"  footprint shape r: median "
                     f"{np.nanmedian(shr):.3f}, min {np.nanmin(shr):.3f}")
        strong = int(np.sum(shr >= 0.5))
        lines.append(f"  pairs with shape r >= 0.5: {strong} of {len(rows)}")
    else:
        lines.append("  no pairs survived - see the verdict below")
    return rows, lines, (o, p, of_s, pairs, M, stage)


def figure(plane, o, p, of_s, pairs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hh, w = o["shape"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    for a, (oc, ttl) in zip(ax, [(o["cent"], "before alignment"),
                                 (of_s, "after alignment")]):
        a.imshow(p["mean"], cmap="gray",
                 vmin=np.percentile(p["mean"], 2),
                 vmax=np.percentile(p["mean"], 99.5))
        a.scatter(p["cent"][:, 1], p["cent"][:, 0], s=90,
                  facecolors="none", edgecolors="#1C6E8C", lw=1.6,
                  label=f"pain ({len(p['cent'])})")
        a.scatter(oc[:, 1], oc[:, 0], s=90, facecolors="none",
                  edgecolors="#C1272D", lw=1.6,
                  label=f"open field ({len(oc)})")
        if ttl.startswith("after"):
            for i, j in pairs:
                a.plot([of_s[i][1], p["cent"][j][1]],
                       [of_s[i][0], p["cent"][j][0]], "-",
                       color="#E0A020", lw=1.4)
        a.set_title(ttl + (f"   {len(pairs)} pairs"
                           if ttl.startswith("after") else ""))
        a.set_xlim(0, w)
        a.set_ylim(hh, 0)
        a.axis("off")
        a.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"plane {plane}: cross-session cell matching "
                 f"(background = pain session mean image)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"match_{plane}.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = os.path.join(PA, "match")
    os.makedirs(outdir, exist_ok=True)
    lines = ["===== cross-session cell matching =====",
             "",
             "open field 15:27 versus pain 15:56, same animal, same two ETL "
             "depths.",
             f"A pair is accepted only within {MAX_DIST_PX:.0f} px "
             f"(one cell radius) after alignment.",
             ""]
    allrows, figs = [], []
    for plane in ("A", "B"):
        rows, lines, bundle = run_plane(plane, lines)
        allrows += rows
        figs.append((plane, bundle))
        lines.append("")

    D = pd.DataFrame(allrows)
    if len(D):
        D.to_csv(os.path.join(outdir, "match_cells.csv"), index=False)
        good = D[D["shape_r"] >= 0.5]
        lines += [
            f"VERDICT: {len(D)} pairs within one cell radius, "
            f"{len(good)} of them also agreeing in footprint shape "
            f"(r >= 0.5).",
            "",
            "Only the shape-agreeing pairs should be treated as the same "
            "neuron. Position",
            "alone can pair two different cells a few px apart, which is "
            "exactly the",
            "situation here - the planes are dense.",
        ]
        if len(good) == 0:
            lines.append("No pair agrees in shape, so no cell can be "
                         "followed across the two sessions.")
    else:
        lines += ["VERDICT: nothing matched. The two sessions did not image "
                  "the same plane",
                  "closely enough for cell-by-cell matching, so open-field "
                  "and pain results",
                  "cannot be linked per neuron - only compared as "
                  "populations."]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, "match_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    for plane, (o, p, of_s, pairs, M, stage) in figs:
        figure(plane, o, p, of_s, pairs, outdir)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
