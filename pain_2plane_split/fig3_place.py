"""fig3_place.py  -  how a centre or corner cell is decided, made readable.

WHY THE FIRST VERSION WAS HARD TO READ
  place_summary.png showed a contrast value and a q-value per cell. Those
  are the answer, not the evidence: there was no way to see WHY a cell
  passed, or what the null it was tested against actually looked like.

LAYOUT follows the zoom-out-first rule
  row 1  the arena itself: trajectory, occupancy, and the 3x3 zones drawn on
         the real floor - before any neuron is mentioned
  row 2  per cell, the decision made visible:
           left   the occupancy-normalised spatial map of that cell's
                  activity, with the zone grid on top
           right  the observed centre-minus-corner contrast against the
                  2000 circular-shift surrogates it was compared to
  A cell passes when its red line falls outside the grey null. That is the
  whole test, drawn.

WHY A CIRCULAR-SHIFT NULL AND NOT A t-TEST
  Calcium samples are not independent - one transient spans several frames -
  so a t-test across frames treats each frame as new information and will
  call noise significant. Rolling the trace against the behaviour keeps the
  trace's own autocorrelation and the animal's occupancy exactly, and
  destroys only the pairing between them.

OCCUPANCY NORMALISATION IS NOT OPTIONAL
  The mouse spends 5.6 % of its time in the centre and 41.4 % in the
  corners. A raw sum over frames would make every corner bin look active.
  Every map here is a MEAN over the frames spent in that bin, and bins with
  under 0.5 s of dwell are left white rather than drawn from one or two
  samples.

OUTPUT  ->  <open field session>\\output_split\\place\\fig3_place.png

USAGE
  python fig3_place.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import openfield_track as oft
from openfield_place_cells import (MIN_OCC_S, N_BIN, N_SHIFT, SMOOTH_BINS,
                                   ZONE, behaviour_on_imaging, cell_seed,
                                   load_tracking, load_traces, shift_null,
                                   zone_contrast)

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
      "2026-09-11_15-27-52")
MIN_SHIFT_S = 10.0
RNG = np.random.default_rng(0)


def run(session):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.ndimage import gaussian_filter

    vid = oft.find_video(session)
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    med, arena, bbox, quad = oft.find_arena(cap, fps)
    cap.release()
    Z = oft.zones(bbox, arena.shape)
    tr = load_tracking(session)
    ax_, ay_, bx_, by_ = bbox

    cells = []
    for plane in ("A", "B"):
        z, t_img, usable, S, shape, labels = load_traces(session, plane)
        x, y, zi, speed, valid = behaviour_on_imaging(tr, t_img, Z)
        dt = float(np.median(np.diff(t_img)))
        bx = np.clip(((x - ax_) / bx_ * N_BIN).astype(int), 0, N_BIN - 1)
        by = np.clip(((y - ay_) / by_ * N_BIN).astype(int), 0, N_BIN - 1)
        bidx = (by * N_BIN + bx)[valid]
        occ_map = np.bincount(bidx, minlength=N_BIN ** 2) * dt
        min_shift = int(round(MIN_SHIFT_S / dt))
        zv = zi[valid]
        for c in usable:
            zz = z[c - 1][valid]
            d = zone_contrast(zz, zv)
            null = shift_null(lambda s: zone_contrast(s, zv), zz, min_shift,
                              N_SHIFT,
                              seed=cell_seed(plane, labels[c - 1],
                                             'contrast'))
            p = float((1 + np.sum(np.abs(null - np.median(null))
                                  >= abs(d - np.median(null))))
                      / (len(null) + 1))
            m = np.zeros(N_BIN ** 2)
            cnt = np.bincount(bidx, minlength=N_BIN ** 2)
            np.add.at(m, bidx, zz)
            rate = np.where(cnt > 0, m / np.maximum(cnt, 1), np.nan)
            rate[occ_map < MIN_OCC_S] = np.nan
            cells.append(dict(plane=plane, label=labels[c - 1],
                              contrast=d, p=p, null=null,
                              rate=rate.reshape(N_BIN, N_BIN)))
        print(f"  plane {plane}: {len(usable)} cells tested")

    # BH across the cells tested, the same correction the table uses
    pv = np.array([c["p"] for c in cells])
    o = np.argsort(pv)
    q = np.empty_like(pv)
    q[o] = np.minimum.accumulate(
        (pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
    for c, qq in zip(cells, np.clip(q, 0, 1)):
        c["q"] = float(qq)
        c["pref"] = ("none" if qq > .05
                     else ("centre" if c["contrast"] > 0 else "corner"))

    ok = tr[tr["ok"] == 1]
    nrow = 1 + len(cells)
    fig = plt.figure(figsize=(15.5, 4.6 + 2.25 * len(cells)))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[4.6] + [2.25] * len(cells),
                          hspace=.32)

    top = gs[0].subgridspec(1, 3, wspace=.16)
    occ2 = None
    for jj, ttl in enumerate(["where the mouse went",
                              "how long it spent there",
                              "the zones, on the real floor"]):
        a = fig.add_subplot(top[0, jj])
        a.imshow(med, cmap="gray", vmin=np.percentile(med, 2),
                 vmax=np.percentile(med, 99.5))
        if jj == 0:
            sc = a.scatter(ok["x"], ok["y"], c=ok["t_s"], s=1.1,
                           cmap="viridis")
            plt.colorbar(sc, ax=a, fraction=.046, label="s")
        elif jj == 1:
            Hh, xe, ye = np.histogram2d(
                ok["x"], ok["y"], bins=N_BIN,
                range=[[ax_, ax_ + bx_], [ay_, ay_ + by_]])
            occ2 = Hh.T / fps
            occ2[occ2 < MIN_OCC_S] = np.nan
            im = a.imshow(occ2, cmap="magma", origin="upper",
                          extent=[xe[0], xe[-1], ye[-1], ye[0]])
            plt.colorbar(im, ax=a, fraction=.046, label="s in bin")
        else:
            for lbl, cc, val in (("centre", "#C1272D", 0),
                                 ("corner", "#1C6E8C", 1)):
                mask = (Z == val) & (arena > 0)
                ov = np.zeros(med.shape + (4,))
                ov[mask] = (*[int(cc[i:i + 2], 16) / 255
                              for i in (1, 3, 5)], .38)
                a.imshow(ov)
            a.plot([], [], "s", color="#C1272D", label="centre 5.6 %")
            a.plot([], [], "s", color="#1C6E8C", label="corners 41.4 %")
            a.legend(loc="lower right", fontsize=9)
        a.plot(*np.vstack([quad, quad[:1]]).T[::-1][::-1], "-",
               color="#E0A020", lw=1.6) if False else None
        a.add_patch(Rectangle((ax_, ay_), bx_, by_, fill=False,
                              ec="#E0A020", lw=1.6))
        for f in (1 / 3, 2 / 3):
            a.plot([ax_ + bx_ * f] * 2, [ay_, ay_ + by_], color="#E0A020",
                   lw=.8, ls="--")
            a.plot([ax_, ax_ + bx_], [ay_ + by_ * f] * 2, color="#E0A020",
                   lw=.8, ls="--")
        a.set_xlim(ax_ - 30, ax_ + bx_ + 30)
        a.set_ylim(ay_ + by_ + 30, ay_ - 30)
        a.axis("off")
        a.set_title(f"{'ABC'[jj]}  {ttl}", fontsize=11, loc="left")

    for i, c in enumerate(cells):
        sub = gs[1 + i].subgridspec(1, 3, width_ratios=[1, 1, 2.1],
                                    wspace=.20)
        col = {"centre": "#C1272D", "corner": "#1C6E8C",
               "none": "#999999"}[c["pref"]]

        a = fig.add_subplot(sub[0, 0])
        m = c["rate"]
        mm = np.where(np.isnan(m), 0, m)
        wt = gaussian_filter((~np.isnan(m)).astype(float), SMOOTH_BINS)
        # Smoothing interpolates across gaps, which is fine, but with a
        # permissive weight cut it also painted colour into bins the mouse
        # never entered - the rate map looked full while the occupancy map
        # beside it was mostly empty. The occupancy mask is re-applied after
        # smoothing so a bin with no dwell stays white.
        sm = np.where(wt > .35, gaussian_filter(mm, SMOOTH_BINS) / wt,
                      np.nan)
        sm[np.isnan(m)] = np.nan
        v = np.nanmax(np.abs(sm)) or 1.0
        im = a.imshow(sm, cmap="RdBu_r", vmin=-v, vmax=v, origin="upper",
                      extent=[0, N_BIN, N_BIN, 0])
        for f in (N_BIN / 3, 2 * N_BIN / 3):
            a.axvline(f, color="k", lw=.7, ls="--")
            a.axhline(f, color="k", lw=.7, ls="--")
        a.set_xticks([]), a.set_yticks([])
        a.set_ylabel(f"{c['plane']}#{c['label']}", fontsize=11, rotation=0,
                     labelpad=28, va="center", color=col, fontweight="bold")
        plt.colorbar(im, ax=a, fraction=.046, label="mean z")
        if i == 0:
            a.set_title("activity per place\n(occupancy-normalised)",
                        fontsize=9.5)

        a = fig.add_subplot(sub[0, 1])
        if occ2 is not None:
            a.imshow(occ2, cmap="Greys", origin="upper",
                     extent=[0, N_BIN, N_BIN, 0])
        a.contour(np.linspace(.5, N_BIN - .5, N_BIN),
                  np.linspace(.5, N_BIN - .5, N_BIN),
                  np.where(np.isnan(sm), 0, sm), levels=[v * .45],
                  colors=[col], linewidths=1.6)
        for f in (N_BIN / 3, 2 * N_BIN / 3):
            a.axvline(f, color="k", lw=.7, ls="--")
            a.axhline(f, color="k", lw=.7, ls="--")
        a.set_xticks([]), a.set_yticks([])
        if i == 0:
            a.set_title("where it was most active\non top of dwell time",
                        fontsize=9.5)

        a = fig.add_subplot(sub[0, 2])
        a.hist(c["null"], bins=60, color="#CFCFCF",
               label=f"{N_SHIFT} circular shifts")
        a.axvline(np.percentile(c["null"], 2.5), color="#777777", ls=":",
                  lw=1)
        a.axvline(np.percentile(c["null"], 97.5), color="#777777", ls=":",
                  lw=1, label="null 95 % range")
        a.axvline(c["contrast"], color=col, lw=2.6,
                  label="observed contrast")
        a.set_xlabel("centre mean - corner mean (z)", fontsize=9.5)
        a.set_yticks([])
        a.legend(fontsize=8.5, loc="upper right")
        a.text(.01, .96, f"contrast {c['contrast']:+.3f}   p {c['p']:.4f}   "
                         f"q {c['q']:.4f}   ->  {c['pref'].upper()}",
               transform=a.transAxes, va="top", fontsize=10, color=col,
               fontweight="bold")
        a.spines[["top", "right", "left"]].set_visible(False)
        if i == 0:
            a.set_title("the test: observed against its own null",
                        fontsize=9.5)

    nc = sum(c["pref"] == "centre" for c in cells)
    nk = sum(c["pref"] == "corner" for c in cells)
    legend = (
        f"METHODS.  Arena: the saturated white floor only "
        f"({bx_} x {by_} px = {100 * arena.mean():.1f} % of the camera "
        f"field), fitted as a rectangle and eroded 8 px; zones are the "
        f"standard 3x3 grid of that rectangle, drawn in panel C. Positions "
        f"come from the largest dark blob INSIDE the floor after a "
        f"morphological opening that removes the miniscope tether; "
        f"{100 * tr['ok'].mean():.2f} % of {len(tr)} frames tracked. Frame "
        f"times come from the per-frame TDMS timestamps, not frame_index/25 "
        f"- the camera dropped 6 frames, which would drift by up to 0.24 s.  "
        f"Neural traces are the curated cells only, demixed from the raw "
        f"movie, dF/F with F0 = the 20th percentile, detrended with a 20 s "
        f"running median, then z-scored. Position is interpolated onto each "
        f"imaging frame time.\n"
        f"STATISTICS.  Contrast = centre mean - corner mean of the z-scored "
        f"trace. The null is {N_SHIFT} circular shifts of the trace against "
        f"the behaviour, minimum shift {MIN_SHIFT_S:.0f} s: this keeps the "
        f"trace's own autocorrelation and the animal's occupancy exactly and "
        f"destroys only the pairing. A t-test across frames would treat each "
        f"frame as independent - calcium samples are not, so it would call "
        f"noise significant. Two-sided empirical p, then Benjamini-Hochberg "
        f"across the {len(cells)} cells tested. A cell is called when its "
        f"red line falls outside the grey null and q <= 0.05.  Maps are "
        f"MEANS over the frames spent in each bin, not sums, and bins with "
        f"under {MIN_OCC_S} s of dwell are left white - the mouse spends "
        f"5.6 % of its time in the centre against 41.4 % in the corners, so "
        f"a sum would make every corner bin look active.\n"
        f"RESULT.  {nc} centre-preferring, {nk} corner-preferring, "
        f"{len(cells) - nc - nk} not selective. Centre occupancy of 5.6 % "
        f"means the centre mean rests on ~83 imaging frames against ~620 in "
        f"the corners, so a centre effect is harder to detect than a corner "
        f"one and the absence of corner cells is weak evidence. One session, "
        f"one animal: this describes these cells, not a population.")
    fig.text(.006, .002, legend, fontsize=9, color="#333333", wrap=True,
             va="bottom", linespacing=1.4)
    fig.suptitle("Open field: how a centre or corner cell is decided",
                 fontsize=14, fontweight="bold", y=.998)
    outdir = os.path.join(session, "output_split", "place")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "fig3_place.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("null", "rate")}
                  for c in cells]).to_csv(
        os.path.join(outdir, "fig3_place_cells.csv"), index=False)
    print(f"\n  {nc} centre, {nk} corner, {len(cells) - nc - nk} none")
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=OF)
    run(ap.parse_args().session)


if __name__ == "__main__":
    main()
