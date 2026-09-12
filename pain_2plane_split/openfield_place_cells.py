"""openfield_place_cells.py  -  which neurons code centre versus corner?

WHAT IS BEING ASKED
  In an open field a mouse spends most of its time against the walls. Cells
  in the amygdala complex have been reported to fire preferentially when the
  animal is in the exposed centre or, conversely, in the sheltered corners.
  The question here is which of the imaged CEA-Ntsr1 neurons do that.

THE THREE WAYS THIS ANALYSIS COULD LIE, AND WHAT IS DONE ABOUT THEM

  1  Occupancy. Measured on this session the mouse is in the centre 5.6 % of
     the time and in the corners 41.5 %. A plain difference of means compares
     18 s of data against 136 s, so the centre mean is far noisier, and any
     slow drift in the trace lands unevenly across zones. Both the zone
     contrast and the rate maps are therefore occupancy-normalised, and the
     traces are detrended before anything else.

  2  Autocorrelation. Calcium samples are not independent - a single
     transient spans several frames - so a t-test across frames has an
     effective n far below the frame count and will call noise significant.
     The null here is a CIRCULAR SHIFT of the neural trace against the
     behaviour, which preserves the trace's own autocorrelation and the
     animal's occupancy structure exactly and only destroys the pairing.
     That is the standard test for place coding.

  3  Speed. The mouse moves differently in the centre than against a wall, so
     a "centre cell" may just be a speed cell. The correlation with speed is
     reported for every cell, with its own shift test, so a zone effect that
     is really a speed effect is visible rather than hidden.

  Spatial information is the Skaggs measure, on the rectified detrended
  trace as the activity proxy, with the same shift null.

INPUTS
  <session>\\output_split\\plane_{A,B}\\final_analysis_results.mat   traces
  <session>\\output_split\\plane_{A,B}\\qc_cells\\qc_cells.csv       usable flag
  <session>\\output_split\\tracking\\openfield_track.csv             position
  <session>\\output_split\\timestamps\\behav_frame_times.csv         real frame times
  <session>\\output_split\\timestamps\\plane_frame_times.csv         imaging times

  The tracking CSV's own `t` column is IGNORED: it was written as
  frame_index / 25, and the behaviour camera dropped 6 frames, so that column
  drifts by up to 0.24 s. Frame times come from the timestamp step instead.

OUTPUTS  ->  <session>\\output_split\\place\\
  place_cells.csv          per cell: zone means, contrast, p, spatial info, speed r
  place_report.txt         the same plus occupancy and the verdict
  place_ratemaps.png       occupancy-normalised rate map per cell
  place_summary.png        zone contrast against its null, and the k = 2 split

USAGE
  python openfield_place_cells.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import openfield_track as oft
from extract_qc_cells import detrend, load_plane

SESSION = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52")
N_SHIFT = 2000          # circular-shift surrogates
MIN_SHIFT_S = 10.0      # never shift by less than this, or the null leaks signal
N_BIN = 12              # spatial bins per axis for the rate maps
SMOOTH_BINS = 1.2       # gaussian sigma, in bins, for display maps
MIN_OCC_S = 0.5         # a spatial bin needs this much dwell time to be shown
ZONE = {0: "centre", 1: "corner", 2: "edge"}
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ loading
def load_tracking(session):
    """Position per behaviour frame, on the imaging time base."""
    tdir = os.path.join(session, "output_split", "timestamps")
    bt = pd.read_csv(os.path.join(tdir, "behav_frame_times.csv"))
    cams = bt["camera"].unique()
    if len(cams) != 1:
        raise SystemExit(f"expected one behaviour camera, found {list(cams)}")
    bt = bt.sort_values("frame")

    tr = pd.read_csv(os.path.join(session, "output_split", "tracking",
                                  "openfield_track.csv"))
    # openfield_track.py writes `frame` 0-based; behav_frame_times.csv is
    # 1-based. Off by one here would shift everything by 40 ms silently.
    tr = tr.sort_values("frame")
    if tr["frame"].min() != 0:
        raise SystemExit(f"tracking frames start at {tr['frame'].min()}, "
                         "expected 0-based")
    if len(tr) != len(bt):
        raise SystemExit(f"{len(tr)} tracked frames but {len(bt)} frame "
                         f"times - the video and the TDMS disagree")
    tr = tr.assign(t_s=bt["t_s"].to_numpy())
    drift = tr["t_s"].to_numpy() - tr["t"].to_numpy()
    print(f"  tracking: {len(tr)} frames, detection "
          f"{100 * tr['ok'].mean():.2f} %")
    print(f"  correcting frame_index/25 by up to {np.abs(drift).max() * 1000:.0f}"
          f" ms (dropped behaviour frames)")
    return tr


def load_traces(session, plane):
    """The 39-cell UNION set, so every neuron gets tested here.

    Two earlier sources were wrong for this analysis and both were changed:

      1  the pre-curation detection - it reported 14 cells for a plane that
         has 12 after curation, and took its usable list from footprints
         Hansol had discarded;
      2  the per-session CURATED set - correct, but it caps the test at the
         cells EXTRACT detected in THIS session, which was 9 usable cells
         across both planes.

    The union (transfer_footprints.py) registers both sessions' footprints
    and solves all of them on this session's movie, so a neuron detected in
    the pain session is measured here too. 39 neurons, of which 34 land on a
    soma in both sessions.

    The filter is anatomical, not activity: a neuron that was quiet in the
    open field is still a valid row - that IS the measurement, and dropping
    it would bias the place result towards cells that happened to fire.
    Only transfers that landed on nothing are excluded.
    """
    from union_data import load_union, usable as union_usable
    U = load_union()
    ses = "openfield" if "openfield" in os.path.basename(session) else "pain"
    d = U[(ses, plane)]
    dff = np.asarray(d["dff"], float)
    labels = d["labels"]
    usable = union_usable(U, ses, plane) + 1      # 1-based, as used below
    det, _ = detrend(dff)
    z = (det - det.mean(1, keepdims=True)) / det.std(1, keepdims=True)
    n_src = sum(1 for s in d["source"] if s == "pain")
    print(f"  plane {plane}: {len(labels)} union cells "
          f"({n_src} detected in the pain session, "
          f"{len(labels) - n_src} only here), "
          f"{len(usable)} on a soma in this session")
    return z, d["t"], usable, d["S"], d["shape"], labels


# ---------------------------------------------------------------- alignment
def behaviour_on_imaging(tr, t_img, Z):
    """Position, zone and speed sampled at each imaging frame time."""
    ok = tr["ok"].to_numpy() == 1
    tb = tr["t_s"].to_numpy()
    x = np.interp(t_img, tb[ok], tr["x"].to_numpy()[ok])
    y = np.interp(t_img, tb[ok], tr["y"].to_numpy()[ok])
    sp = tr["speed_px_s"].to_numpy()
    good = ok & np.isfinite(sp)
    speed = np.interp(t_img, tb[good], sp[good])
    # imaging frames with no behaviour frame within half a camera period
    near = np.min(np.abs(t_img[:, None] - tb[ok][None, :]), axis=1)
    valid = near <= 0.5 / 25 * 3
    zi = Z[np.clip(np.round(y).astype(int), 0, Z.shape[0] - 1),
           np.clip(np.round(x).astype(int), 0, Z.shape[1] - 1)]
    return x, y, zi, speed, valid


# ---------------------------------------------------------------- statistics
def zone_contrast(z, zi):
    """centre mean minus corner mean, in z units."""
    c = z[zi == 0]
    k = z[zi == 1]
    if len(c) < 5 or len(k) < 5:
        return np.nan
    return float(c.mean() - k.mean())


def skaggs(r, zi_or_bin, n_states):
    """Spatial information in bits per sample, occupancy-weighted."""
    r = np.maximum(r, 0)
    rbar = r.mean()
    if rbar <= 0:
        return 0.0
    info = 0.0
    for s in range(n_states):
        m = zi_or_bin == s
        p = m.mean()
        if p <= 0:
            continue
        ri = r[m].mean()
        if ri > 0:
            info += p * (ri / rbar) * np.log2(ri / rbar)
    return float(info)


def shift_null(fn, z, min_shift, n_shift=N_SHIFT, seed=None):
    """Circular shifts of the trace, never smaller than min_shift samples.

    The seed is derived from the cell, not taken from a shared generator.
    With a module-level RNG the surrogates depended on how many times the
    caller had drawn before, so this script (3 draws per cell) and
    fig3_place.py (1 draw per cell) produced different nulls for the same
    cell and disagreed on the borderline calls: two corner cells at
    q = 0.046 appeared in one and not the other. A per-cell seed makes the
    null reproducible and identical everywhere.
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    n = len(z)
    out = np.empty(n_shift)
    hi = n - min_shift
    for i in range(n_shift):
        s = int(rng.integers(min_shift, hi))
        out[i] = fn(np.roll(z, s))
    return out


def cell_seed(plane, label, what):
    """A stable seed per (cell, statistic), so any script reproduces it.

    hashlib, not hash(): Python salts str hashing per process unless
    PYTHONHASHSEED is set, so hash() would give a different null on every
    run - the opposite of the point.
    """
    import hashlib
    h = hashlib.md5(f"{plane}|{label}|{what}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def emp_p_two(null, obs):
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(np.abs(null - np.median(null))
                             >= abs(obs - np.median(null))))
                 / (len(null) + 1))


def emp_p_one(null, obs):
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= obs)) / (len(null) + 1))


# --------------------------------------------------------------------- main
def run(session):
    print("=" * 70)
    print("open field: centre versus corner")
    print("=" * 70)

    vid = oft.find_video(session)
    import cv2
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    med, arena, bbox, quad = oft.find_arena(cap, fps)
    cap.release()
    Z = oft.zones(bbox, arena.shape)
    print(f"  arena {bbox[2]}x{bbox[3]} px at ({bbox[0]},{bbox[1]})")

    tr = load_tracking(session)

    rows, maps = [], {}
    for plane in ("A", "B"):
        z, t_img, usable, S, shape, labels = load_traces(session, plane)
        x, y, zi, speed, valid = behaviour_on_imaging(tr, t_img, Z)
        occ = {v: float(np.mean(zi[valid] == v)) for v in ZONE}
        print(f"    occupancy at imaging frames: "
              + "  ".join(f"{ZONE[v]} {100 * occ[v]:.1f} %" for v in ZONE)
              + f"   ({valid.sum()}/{len(valid)} frames usable)")

        # spatial bins for the rate maps
        bx = np.clip(((x - bbox[0]) / bbox[2] * N_BIN).astype(int),
                     0, N_BIN - 1)
        by = np.clip(((y - bbox[1]) / bbox[3] * N_BIN).astype(int),
                     0, N_BIN - 1)
        bidx = by * N_BIN + bx
        dt_img = float(np.median(np.diff(t_img)))
        occ_map = np.bincount(bidx[valid], minlength=N_BIN ** 2) * dt_img

        min_shift = int(round(MIN_SHIFT_S / dt_img))
        for c in usable:
            zz = z[c - 1][valid]
            zv, spv, biv = zi[valid], speed[valid], bidx[valid]

            d = zone_contrast(zz, zv)
            sd_ = dict(plane=plane, label=labels[c - 1])
            nd = shift_null(lambda s: zone_contrast(s, zv), zz, min_shift,
                            seed=cell_seed(plane, labels[c - 1],
                                           "contrast"))
            p_d = emp_p_two(nd, d)

            info = skaggs(zz, biv, N_BIN ** 2)
            ni = shift_null(lambda s: skaggs(s, biv, N_BIN ** 2), zz,
                            min_shift,
                            seed=cell_seed(plane, labels[c - 1], "info"))
            p_i = emp_p_one(ni, info)

            r_sp = float(np.corrcoef(zz, spv)[0, 1])
            ns = shift_null(lambda s: float(np.corrcoef(s, spv)[0, 1]), zz,
                            min_shift,
                            seed=cell_seed(plane, labels[c - 1], "speed"))
            p_sp = emp_p_two(ns, r_sp)

            rows.append(dict(
                plane=plane, cell=labels[c - 1],
                m_centre=float(zz[zv == 0].mean()),
                m_corner=float(zz[zv == 1].mean()),
                m_edge=float(zz[zv == 2].mean()),
                contrast=d, p_contrast=p_d,
                null_sd=float(np.nanstd(nd)),
                spatial_info=info, p_info=p_i,
                r_speed=r_sp, p_speed=p_sp,
                n_centre=int(np.sum(zv == 0)),
                n_corner=int(np.sum(zv == 1))))

            m = np.zeros(N_BIN ** 2)
            cnt = np.bincount(biv, minlength=N_BIN ** 2)
            np.add.at(m, biv, zz)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(cnt > 0, m / np.maximum(cnt, 1), np.nan)
            rate[occ_map < MIN_OCC_S] = np.nan
            maps[(plane, labels[c - 1])] = rate.reshape(N_BIN, N_BIN)

    D = pd.DataFrame(rows)
    # Benjamini-Hochberg across the cells tested, per statistic
    for col in ("p_contrast", "p_info", "p_speed"):
        p = D[col].to_numpy()
        o = np.argsort(p)
        q = np.empty_like(p)
        n = len(p)
        q[o] = np.minimum.accumulate(
            (p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
        D["q" + col[1:]] = np.clip(q, 0, 1)

    D["preference"] = np.where(
        D["q_contrast"] > 0.05, "none",
        np.where(D["contrast"] > 0, "centre", "corner"))

    outdir = os.path.join(session, "output_split", "place")
    os.makedirs(outdir, exist_ok=True)
    D.to_csv(os.path.join(outdir, "place_cells.csv"), index=False)
    report(D, occ, outdir, dt_img)
    plot_maps(maps, D, bbox, N_BIN, outdir)
    plot_summary(D, outdir)
    print(f"\nwrote {outdir}")
    return D


def report(D, occ, outdir, dt_img):
    L = ["===== open field: centre versus corner =====",
         f"cells tested: {len(D)} (usable only: active, single-blob "
         f"footprint, not oversized)",
         f"imaging {1 / dt_img:.4f} Hz, {N_SHIFT} circular-shift surrogates, "
         f"minimum shift {MIN_SHIFT_S:.0f} s",
         "",
         "zone occupancy at imaging frames: "
         + "  ".join(f"{ZONE[v]} {100 * occ[v]:.1f} %" for v in ZONE),
         "",
         "Statistics. Zone contrast is the centre mean minus the corner mean "
         "of the",
         "detrended z-scored dF/F. Its p-value is two-sided against a "
         "circular-shift",
         "null, which keeps the trace's autocorrelation and the animal's "
         "occupancy",
         "and destroys only the pairing - a t-test across frames would treat "
         "each",
         "frame as independent and is not valid here. Spatial information is "
         "the",
         "Skaggs measure on the rectified trace, one-sided against the same "
         "null.",
         "q-values are Benjamini-Hochberg across the cells tested.",
         "",
         f"  {'cell':>6s} {'centre':>7s} {'corner':>7s} {'edge':>7s} "
         f"{'contr':>6s} {'nullSD':>6s} {'p':>6s} {'q':>6s} "
         f"{'info':>6s} {'q_inf':>6s} {'r_spd':>6s} {'q_spd':>6s}  pref"]
    for _, r in D.iterrows():
        L.append(f"  {r['plane']}#{str(r['cell']):<4s} {r['m_centre']:7.3f} "
                 f"{r['m_corner']:7.3f} {r['m_edge']:7.3f} "
                 f"{r['contrast']:6.3f} {r['null_sd']:6.3f} "
                 f"{r['p_contrast']:6.4f} {r['q_contrast']:6.4f} "
                 f"{r['spatial_info']:6.3f} {r['q_info']:6.4f} "
                 f"{r['r_speed']:6.3f} {r['q_speed']:6.4f}  "
                 f"{r['preference']}")
    nc = int((D["preference"] == "centre").sum())
    nk = int((D["preference"] == "corner").sum())
    nn = int((D["preference"] == "none").sum())
    ni = int((D["q_info"] <= 0.05).sum())
    ns = int((D["q_speed"] <= 0.05).sum())
    both = D[(D["q_contrast"] <= 0.05) & (D["q_speed"] <= 0.05)]
    L += ["",
          f"VERDICT: {nc} centre-preferring, {nk} corner-preferring, "
          f"{nn} not selective (q <= 0.05).",
          f"         {ni} of {len(D)} carry significant spatial information.",
          f"         {ns} of {len(D)} are significantly speed-correlated.",
          (f"         {len(both)} cell(s) are BOTH zone- and speed-"
           f"significant: "
           + ", ".join(f"{r['plane']}#{r['cell']}"
                       for _, r in both.iterrows())
           + " - the zone effect there cannot be separated from a speed "
             "effect with this design."
           if len(both) else
           "         no cell is both zone- and speed-significant, so the "
           "zone effects are not speed in disguise."),
          "",
          "The k = 2 split the request asked for is the sign of the contrast "
          "among the",
          "cells that pass, which is what is plotted. With this many cells a "
          "k-means",
          "on a one-dimensional index is a threshold, so the threshold is "
          "stated",
          "rather than dressed up as clustering.",
          "",
          "Caveats.",
          f"  Centre occupancy is {100 * occ[0]:.1f} %, so the centre mean "
          f"rests on",
          f"  {int(D['n_centre'].iloc[0])} imaging frames against "
          f"{int(D['n_corner'].iloc[0])} in the corners. A centre effect is",
          "  therefore much harder to detect than a corner effect, and an "
          "absence of",
          "  centre cells here is weak evidence that none exist.",
          "  Positions are floor-only, so a body pressed against a wall has "
          "its",
          "  centroid pulled inward. The centre cell of the 3x3 grid starts "
          "~150 px",
          "  from the wall and the bias is ~20-30 px, so wall frames cannot "
          "leak into",
          "  the centre, but corner/edge boundaries are softer than they "
          "look.",
          "  One session, one animal: this is a description of these cells, "
          "not a",
          "  population result."]
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(outdir, "place_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")


def plot_maps(maps, D, bbox, nbin, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    keys = list(maps)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, k in zip(ax, keys):
        m = maps[k]
        mm = np.where(np.isnan(m), 0, m)
        w = gaussian_filter((~np.isnan(m)).astype(float), SMOOTH_BINS)
        # Re-apply the occupancy mask after smoothing. With a 0.05 weight cut
        # the blur painted colour into bins the mouse never entered, so the
        # map looked full while the occupancy behind it was sparse.
        sm = np.where(w > 0.35, gaussian_filter(mm, SMOOTH_BINS) / w, np.nan)
        sm[np.isnan(m)] = np.nan
        v = np.nanmax(np.abs(sm)) or 1.0
        im = a.imshow(sm, cmap="RdBu_r", vmin=-v, vmax=v, origin="upper")
        r = D[(D["plane"] == k[0]) & (D["cell"] == k[1])].iloc[0]
        a.set_title(f"{k[0]}#{k[1]}  contrast {r['contrast']:+.2f} "
                    f"(q {r['q_contrast']:.3f})\n{r['preference']}",
                    fontsize=9)
        a.set_xticks([]), a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=.046, label="z")
    for a in ax[len(keys):]:
        a.axis("off")
    fig.suptitle("occupancy-normalised mean z-scored dF/F per spatial bin "
                 "(white = under 0.5 s dwell)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(os.path.join(outdir, "place_ratemaps.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def plot_summary(D, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    lbl = [f"{r['plane']}#{r['cell']}" for _, r in D.iterrows()]
    col = {"centre": "#C1272D", "corner": "#1C6E8C", "none": "#B0B0B0"}
    cs = [col[p] for p in D["preference"]]

    o = np.argsort(D["contrast"].to_numpy())
    ax[0].barh(np.arange(len(D)), D["contrast"].to_numpy()[o],
               color=np.array(cs)[o])
    ax[0].errorbar(D["contrast"].to_numpy()[o], np.arange(len(D)),
                   xerr=1.96 * D["null_sd"].to_numpy()[o], fmt="none",
                   ecolor="k", elinewidth=.8, capsize=2)
    ax[0].set_yticks(np.arange(len(D)))
    ax[0].set_yticklabels(np.array(lbl)[o], fontsize=8)
    ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_xlabel("centre mean - corner mean (z)")
    ax[0].set_title("zone contrast\nbars = 95 % of the shift null")

    ax[1].scatter(D["contrast"], -np.log10(D["q_contrast"].clip(1e-4)),
                  c=cs, s=45)
    ax[1].axhline(-np.log10(0.05), color="k", ls="--", lw=.9)
    ax[1].axvline(0, color="k", lw=.8)
    for (_, r), s in zip(D.iterrows(), lbl):
        ax[1].annotate(s, (r["contrast"],
                           -np.log10(max(r["q_contrast"], 1e-4))),
                       fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
    ax[1].set_xlabel("zone contrast (z)")
    ax[1].set_ylabel("-log10 q")
    ax[1].set_title("corner-preferring (blue) vs centre (red)\n"
                    "dashed = q 0.05")

    ax[2].scatter(D["r_speed"], D["contrast"], c=cs, s=45)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].axvline(0, color="k", lw=.8)
    for (_, r), s in zip(D.iterrows(), lbl):
        ax[2].annotate(s, (r["r_speed"], r["contrast"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[2].set_xlabel("correlation with speed")
    ax[2].set_ylabel("zone contrast (z)")
    ax[2].set_title("is the zone effect just speed?")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "place_summary.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=SESSION)
    run(ap.parse_args().session)


if __name__ == "__main__":
    main()
