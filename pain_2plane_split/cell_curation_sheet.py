"""cell_curation_sheet.py  -  contact sheets for discarding footprints by eye.

WHAT THIS IS FOR
  The automatic QC (extract_qc_cells.py) says whether a footprint's trace is
  separable from background and whether it sits on something brighter than
  its surround. It cannot say "that is a blood vessel" or "that is two cells
  and a process". Those calls are made by looking, so this produces the view
  that makes them possible and then records the decision in one file.

WHAT EACH TILE SHOWS
  Two images per candidate, side by side, cropped around the footprint:
    MEAN  output.info.summary_image - the session average. A real soma is a
          filled blob here whether or not it ever fired.
    MAX   output.info.max_image - the per-pixel maximum over time. An ACTIVE
          soma is brightest here; neuropil and vessels are not. Mean and max
          disagreeing is itself informative.
  The footprint outline (30 % of its own peak) is drawn on both.

  Brightness is stretched using the WHOLE plane's percentiles, not each
  crop's, so a dim candidate looks dim instead of being auto-brightened into
  looking convincing. When a tile looks empty, read the anat number: that is
  the measured contrast against the surrounding ring.

OUTPUTS  ->  <session>\\output_split\\curation\\
  curation_<plane>_tiles.png     the contact sheet, one tile per candidate
  curation_<plane>_overview.png  whole FOV, numbered, for spatial context
  curation_template.csv          one row per candidate, with a DECISION
                                 column pre-filled from the automatic QC

HOW TO USE THE TEMPLATE
  Edit the DECISION column only: `keep` or `discard`. Everything downstream
  reads that column, so the automatic verdict is a starting suggestion and
  the file is the record of what was actually decided and why.

USAGE
  python cell_curation_sheet.py                 # all four planes
  python cell_curation_sheet.py --only pain
"""
from __future__ import annotations

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd

SESSIONS = [
    ("openfield", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
                  "2026-09-11_15-27-52"),
    ("pain", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
             "2026-09-11_15-56-48"),
]
CROP = 110          # px per tile, centred on the footprint
TILE = 210          # px each tile is drawn at
LO, HI = 2.0, 99.7  # percentile stretch, computed per plane not per crop
REL = 0.3           # footprint outline level, fraction of its own peak


def read_morph(folder):
    """Eccentricity and solidity per cell, from step 2's QC_report.txt.

    Worth surfacing here because the automatic QC in extract_qc_cells.py
    measures brightness contrast against the surround and says nothing about
    SHAPE, so a bright elongated fibre or vessel passes it. Looking at the
    pain plane A sheet, #6 and #12 are plainly fibres rather than somata -
    and they are exactly the two highest eccentricities in that plane (0.934
    and 0.941), just under the 0.95 the morphology filter allows. So the
    number that matches what the eye sees already exists; it only needed to
    be put next to the picture.
    """
    p = os.path.join(folder, "QC_report.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        f = line.split()
        if len(f) >= 8 and f[0].isdigit():
            try:
                out[int(f[0])] = (float(f[4]), float(f[5]))   # Ecc, Sol
            except ValueError:
                pass
    return out


def load(folder):
    """Footprints (h*w, k), the mean image and the max image."""
    with h5py.File(os.path.join(folder, "final_analysis_results.mat"),
                   "r") as f:
        S3 = np.array(f["output"]["spatial_weights"])        # (k, w, h)
        mean_img = np.array(f["output"]["info"]["summary_image"]).T
        max_img = np.array(f["output"]["info"]["max_image"]).T
    k, w, h = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, h * w).T
    if mean_img.shape != (h, w):
        raise SystemExit(f"summary_image is {mean_img.shape}, expected "
                         f"{(h, w)}")
    return S, mean_img, max_img, (h, w)


def stretch(img):
    lo, hi = np.percentile(img, [LO, HI])
    v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    return (v * 255).astype(np.uint8)


def outline(col, shape):
    bw = (col.reshape(shape) > REL * col.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def crop_at(img8, cy, cx, shape, cnts):
    h, w = shape
    y0 = int(np.clip(cy - CROP // 2, 0, max(h - CROP, 0)))
    x0 = int(np.clip(cx - CROP // 2, 0, max(w - CROP, 0)))
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, cnts, -1, (0, 240, 255), 1)
    sub = vis[y0:y0 + CROP, x0:x0 + CROP]
    if sub.shape[:2] != (CROP, CROP):
        pad = np.zeros((CROP, CROP, 3), np.uint8)
        pad[:sub.shape[0], :sub.shape[1]] = sub
        sub = pad
    return cv2.resize(sub, (TILE, TILE), interpolation=cv2.INTER_NEAREST)


def run_plane(tag, session, plane, rows_out):
    folder = os.path.join(session, "output_split", f"plane_{plane}")
    S, mean_img, max_img, shape = load(folder)
    h, w = shape
    k = S.shape[1]
    qc = pd.read_csv(os.path.join(folder, "qc_cells", "qc_cells.csv"))
    morph = read_morph(folder)
    m8, x8 = stretch(mean_img), stretch(max_img)

    tiles, labels = [], []
    for i in range(k):
        col = S[:, i]
        img = col.reshape(shape)
        ys, xs = np.nonzero(img)
        wt = img[ys, xs]
        cy, cx = np.average(ys, weights=wt), np.average(xs, weights=wt)
        cnts = outline(col, shape)
        pair = np.hstack([crop_at(m8, cy, cx, shape, cnts),
                          np.full((TILE, 3, 3), 40, np.uint8),
                          crop_at(x8, cy, cx, shape, cnts)])
        r = qc[qc["cell"] == i + 1]
        r = r.iloc[0] if len(r) else None
        ecc, sol = morph.get(i + 1, (np.nan, np.nan))
        elong = np.isfinite(ecc) and ecc >= 0.90
        # The question here is "is this a cell", which is NOT the same as
        # "is this usable for the response analysis". A silent neuron is a
        # cell; it just never fired. So the pre-filled decision discards only
        # what looks like not-a-cell - nothing at the location (SUSPECT),
        # more than one soma in the footprint, an oversized footprint, or an
        # elongated one that is probably a fibre. Activity is reported but
        # does not drive the suggestion.
        notcell = (r is not None
                   and (r["verdict"] == "SUSPECT" or r["n_blobs"] > 1
                        or r["big_footprint"] == 1)) or elong
        flags = []
        if r is not None and r["verdict"] == "SUSPECT":
            flags.append("no signal+no contrast")
        if r is not None and r["n_blobs"] > 1:
            flags.append(f"{int(r['n_blobs'])} blobs")
        if r is not None and r["big_footprint"] == 1:
            flags.append("oversized")
        if elong:
            flags.append("elongated")
        if r is not None:
            l1 = (f"{tag[:2].upper()}-{plane}#{i + 1}  {r['verdict']}"
                  + ("  | " + ", ".join(flags) if flags else ""))
            l2 = (f"area {int(r['area_px'])}  ecc {ecc:.2f}  sol {sol:.2f}"
                  f"  anat {r['anat']:.3f}  ev {int(r['events'])}")
        else:
            l1, l2 = f"{tag[:2].upper()}-{plane}#{i + 1}", ""
        bar = np.full((34, pair.shape[1], 3), 22, np.uint8)
        cv2.putText(bar, l1, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (90, 90, 255) if notcell else (140, 255, 160), 1)
        cv2.putText(bar, l2, (5, 29), cv2.FONT_HERSHEY_SIMPLEX, .37,
                    (200, 200, 200), 1)
        tiles.append(np.vstack([bar, pair]))
        labels.append(l1)
        rows_out.append(dict(
            session=tag, plane=plane, cell=i + 1,
            area_px=int(r["area_px"]) if r is not None else -1,
            ecc=round(ecc, 3), solidity=round(sol, 3),
            n_blobs=int(r["n_blobs"]) if r is not None else -1,
            anat=round(float(r["anat"]), 4) if r is not None else np.nan,
            events=int(r["events"]) if r is not None else -1,
            p_joint=round(float(r["p_joint"]), 4) if r is not None else np.nan,
            auto_verdict=r["verdict"] if r is not None else "?",
            flags="; ".join(flags),
            active=int(r["verdict"] == "active") if r is not None else 0,
            DECISION="discard" if notcell else "keep"))

    outdir = os.path.join(session, "output_split", "curation")
    os.makedirs(outdir, exist_ok=True)

    ncol = 3
    nrow = int(np.ceil(len(tiles) / ncol))
    tw, th = tiles[0].shape[1], tiles[0].shape[0]
    sheet = np.full((nrow * (th + 6), ncol * (tw + 6), 3), 12, np.uint8)
    for i, t in enumerate(tiles):
        r_, c_ = divmod(i, ncol)
        sheet[r_ * (th + 6):r_ * (th + 6) + th,
              c_ * (tw + 6):c_ * (tw + 6) + tw] = t
    hdr = np.full((40, sheet.shape[1], 3), 12, np.uint8)
    cv2.putText(hdr, f"{tag}  plane {plane}   {k} candidates   "
                     f"left = MEAN image, right = MAX over time   "
                     f"red label = flagged by automatic QC",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    p = os.path.join(outdir, f"curation_{tag}_{plane}_tiles.png")
    cv2.imwrite(p, np.vstack([hdr, sheet]))

    ov = cv2.cvtColor(x8, cv2.COLOR_GRAY2BGR)
    for i in range(k):
        col = S[:, i]
        cv2.drawContours(ov, outline(col, shape), -1, (0, 240, 255), 1)
        img = col.reshape(shape)
        ys, xs = np.nonzero(img)
        wt = img[ys, xs]
        cv2.putText(ov, str(i + 1),
                    (int(np.average(xs, weights=wt)) + 6,
                     int(np.average(ys, weights=wt)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 255, 255), 1)
    cv2.putText(ov, f"{tag} plane {plane} - MAX image, {k} candidates",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(outdir,
                             f"curation_{tag}_{plane}_overview.png"),
                cv2.resize(ov, (w * 2, h * 2),
                           interpolation=cv2.INTER_NEAREST))
    print(f"  {tag} plane {plane}: {k} candidates -> {os.path.basename(p)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="openfield | pain")
    a = ap.parse_args()
    rows = []
    for tag, s in SESSIONS:
        if a.only and a.only != tag:
            continue
        for plane in ("A", "B"):
            run_plane(tag, s, plane, rows)
    D = pd.DataFrame(rows)
    for tag, s in SESSIONS:
        if a.only and a.only != tag:
            continue
        outdir = os.path.join(s, "output_split", "curation")
        os.makedirs(outdir, exist_ok=True)
        D.to_csv(os.path.join(outdir, "curation_template.csv"), index=False)
    print(f"\n{len(D)} candidates total; pre-filled keep="
          f"{int((D['DECISION'] == 'keep').sum())}, discard="
          f"{int((D['DECISION'] == 'discard').sum())}")
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
