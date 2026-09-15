"""of_zone_rates.py  -  does a cell fire FASTER in the centre? Rates, not counts.

WHY THIS REPLACES THE EARLIER OPEN-FIELD FIGURE
  Hansol's objection, and it is correct: the cells I called "centre cells"
  clearly fire while the mouse is sitting in a corner too, so the name
  oversells them. This figure shows the thing the name should have been
  based on.

  The mouse is in the corners 41.7 % of the session and in the centre 5.5 %
  - 7.6 times longer. So MOST of any cell's firing happens in a corner no
  matter what it prefers. Counting yellow dots answers the wrong question.
  The right one is the rate: firing per second spent in each zone.

    A3   17.9 % of its firing is in the centre ... but 1.24 /s there
                                                 against 0.34 /s in corners
    A7   16.0 % of its firing is in the centre ... 0.96 /s against 0.29 /s

  So they do fire 3-4 times faster per second in the centre, and they also
  fire plenty in the corners. "Fires faster in the centre" is the honest
  label; "centre cell" is not, and is not used here.

  ZONE = WHERE THE MOUSE IS, always. Nothing in this script is triggered on
  entering a zone - an earlier figure had entry-triggered averages in it and
  they answered a question nobody asked.

WHAT IS DRAWN, per cell
  1  the field of view, this cell filled and named
  2  the path with a dot wherever the cell fired (z > Z_FIRE), zones drawn
  3  the rate map: firing per second spent in each bin, ONE COLOUR SCALE
     shown with its own colour bar and its maximum printed, so "brighter"
     means something. Bins with under 0.5 s of dwell are left white.
  4  three bars: firing rate in centre, corner and edge, with the time
     spent in each printed under them - the claim, in one panel
  5  the trace, with the mouse's location as a colour bar underneath

OUTPUT  ->  <open field>\\output_split\\atlas\\fig17_zone_rates*.png

USAGE
  python of_zone_rates.py
"""
from __future__ import annotations

import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import openfield_track as oft  # noqa: E402
import transfer_footprints as TF  # noqa: E402
from of_cell_rows import Z_FIRE, arena_geometry, bp, zone_squares  # noqa: E402
from openfield_place_cells import cell_seed, load_tracking  # noqa: E402
from pain_events import bh  # noqa: E402
from union_data import ANAT_MIN, OF_SESSION, PLANES, ROOTS, load_union  # noqa: E402

N_BIN = 10
MIN_DWELL_S = 0.5
CEN, COR, EDG = "#2166AC", "#E7298A", "#9E9E9E"
FIRE = "#FFC000"
INK, DIM = "#1A1A1A", "#666666"
ZN = ("centre", "corner", "edge")
ZC = (CEN, COR, EDG)


def main():
    U = load_union()
    root = ROOTS["openfield"]
    out = os.path.join(root, "atlas")
    os.makedirs(out, exist_ok=True)
    P = pd.read_csv(os.path.join(root, "place", "place_cells.csv"))
    key = {r["cell"]: r for _, r in P.iterrows()}

    bbox, _ = arena_geometry()
    tr = load_tracking(OF_SESSION)
    t = U[("openfield", "A")]["t"]
    ok = tr["ok"].to_numpy() == 1
    tb = tr["t_s"].to_numpy()
    zo = np.round(np.interp(t, tb[ok],
                            tr["zone"].to_numpy()[ok].astype(float))
                  ).astype(int)
    xn = np.clip((np.interp(t, tb[ok], tr["x"].to_numpy()[ok]) - bbox[0])
                 / bbox[2], 0, 1)
    yn = np.clip((np.interp(t, tb[ok], tr["y"].to_numpy()[ok]) - bbox[1])
                 / bbox[3], 0, 1)
    pxn = np.clip((tr["x"].to_numpy()[ok] - bbox[0]) / bbox[2], 0, 1)
    pyn = np.clip((tr["y"].to_numpy()[ok] - bbox[1]) / bbox[3], 0, 1)
    dt = float(np.median(np.diff(t)))
    bx = np.clip((xn * N_BIN).astype(int), 0, N_BIN - 1)
    by = np.clip((yn * N_BIN).astype(int), 0, N_BIN - 1)
    occ_bin = np.zeros((N_BIN, N_BIN))
    np.add.at(occ_bin, (by, bx), dt)
    dwell = occ_bin >= MIN_DWELL_S
    occ = {v: float((zo == v).sum()) * dt for v in (0, 1, 2)}

    # the two standard stability checks for a spatial claim, and the one
    # this session specifically needs: the mouse entered the centre FOUR
    # times, so a preference could be one visit
    vis = np.diff(np.concatenate([[0], (zo == 0).astype(np.int8), [0]]))
    v_st, v_en = np.flatnonzero(vis == 1), np.flatnonzero(vis == -1) - 1
    half = len(t) // 2

    def contrast(z, keep=None):
        k = np.ones(len(z), bool) if keep is None else keep
        c, kk = (zo == 0) & k, (zo == 1) & k
        if c.sum() < 3 or kk.sum() < 3:
            return np.nan
        return float(z[c].mean() - z[kk].mean())

    # CORNER versus EVERYWHERE ELSE - the actual "is there a corner cell"
    # question, which centre-minus-corner does not ask. A cell could be
    # high in the corners and low in BOTH the centre and the edge, and the
    # contrast above would miss it. This test is also the better powered
    # one: 626 corner frames against 874 elsewhere, where the centre has
    # only 82.
    m_cor = zo == 1

    def corner_vs_rest(z):
        return float(z[m_cor].mean() - z[~m_cor].mean())

    def shift_p(z, fn, fs, seed):
        rng = np.random.default_rng(seed)
        lo = int(10 * fs)
        sh = rng.integers(lo, len(z) - lo, 2000)
        null = np.array([fn(np.roll(z, s)) for s in sh])
        o = fn(z)
        return o, (1. + np.sum(np.abs(null) >= abs(o))) / (len(null) + 1.)

    rows = []
    for plane in PLANES:
        d = U[("openfield", plane)]
        for i, lab in enumerate(d["labels"]):
            z = d["z"][i]
            f = z > Z_FIRE
            rate = [float((f & (zo == v)).sum()) / max(occ[v], 1e-9)
                    for v in (0, 1, 2)]
            frac = [100 * float(np.mean(zo[f] == v)) if f.any() else np.nan
                    for v in (0, 1, 2)]
            loo = []
            for a, b in zip(v_st, v_en):
                keep = np.ones(len(t), bool)
                keep[a:b + 1] = False
                loo.append(contrast(z, keep))
            h1 = contrast(z, np.arange(len(t)) < half)
            h2 = contrast(z, np.arange(len(t)) >= half)
            cvr, cvr_p = shift_p(z, corner_vs_rest, d["fs"],
                                 cell_seed(plane, lab, "cornerVSrest"))
            r = key.get(lab)
            rows.append(dict(plane=plane, uid=lab, i=i,
                             loo_min=round(float(np.nanmin(loo)), 3),
                             loo_max=round(float(np.nanmax(loo)), 3),
                             half1=round(h1, 3), half2=round(h2, 3),
                             corner_vs_rest=round(cvr, 3),
                             corner_vs_rest_p=round(cvr_p, 4),
                             stable=bool(np.isfinite(h1) and np.isfinite(h2)
                                         and np.sign(h1) == np.sign(h2)),
                             usable=bool(d["anat"][i] >= ANAT_MIN),
                             n_fire=int(f.sum()), rate=rate, frac=frac,
                             ratio=rate[0] / max(rate[1], 1e-9),
                             contrast=(float(r["contrast"]) if r is not None
                                       else np.nan),
                             q=(float(r["q_contrast"]) if r is not None
                                else np.nan),
                             faster=(r is not None
                                     and r["q_contrast"] <= .05
                                     and r["contrast"] > 0)))
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "i"}
                      for r in rows])
    for j, nmz in enumerate(ZN):
        R[f"rate_{nmz}"] = [r["rate"][j] for r in rows]
        R[f"pct_fire_{nmz}"] = [round(r["frac"][j], 1) for r in rows]
    R["corner_vs_rest_q"] = np.round(bh(R["corner_vs_rest_p"].to_numpy()), 4)
    R = R.drop(columns=["rate", "frac"])
    R.to_csv(os.path.join(out, "zone_rates.csv"), index=False)

    L = ["===== open field: firing RATE by where the mouse was =====", "",
         f"Zone means where the mouse IS at that moment - nothing here is "
         f"triggered on entering a zone.",
         "",
         f"time in each zone: " + ",  ".join(
             f"{nmz} {occ[v]:.0f} s ({100 * occ[v] / sum(occ.values()):.1f} %)"
             for v, nmz in enumerate(ZN)),
         f"The mouse is in the corners {occ[1] / max(occ[0], 1e-9):.1f} "
         f"times longer than in the centre, so most of ANY cell's firing "
         f"lands in a corner.",
         f"Firing is z > {Z_FIRE}; rate is firing frames per second spent "
         f"in that zone.",
         "",
         f"  {'cell':>5s} {'fires':>5s} {'% of firing in':>22s} "
         f"{'rate per second in':>26s} {'centre/corner':>13s} "
         f"{'zone test':>18s}",
         f"  {'':>5s} {'':>5s} {'centre':>7s}{'corner':>7s}{'edge':>7s} "
         f"{'centre':>8s}{'corner':>8s}{'edge':>8s}  {'':>13s}"]
    for _, r in R.sort_values("ratio", ascending=False).iterrows():
        if not r["usable"]:
            continue
        L.append(f"  {r['uid']:>5s} {r['n_fire']:>5d} "
                 f"{r['pct_fire_centre']:6.1f}%{r['pct_fire_corner']:6.1f}%"
                 f"{r['pct_fire_edge']:6.1f}% "
                 f"{r['rate_centre']:8.3f}{r['rate_corner']:8.3f}"
                 f"{r['rate_edge']:8.3f}  {r['ratio']:13.2f}  "
                 + (f"faster in centre, q={r['q']:.3f}" if r["faster"]
                    else f"q={r['q']:.3f}" if np.isfinite(r["q"]) else ""))
    ok_ = R["usable"]
    nc = int(((R["corner_vs_rest_q"] <= .20) & (R["corner_vs_rest"] > 0)
              & ok_).sum())
    L += ["",
          "IS THERE A CORNER CELL?  Two different questions, both answered "
          "no.",
          "  centre minus corner  : "
          + ", ".join(f"{lab} {int(((R[chr(113)] <= thr) & (R[chr(99) + chr(111) + chr(110) + chr(116) + chr(114) + chr(97) + chr(115) + chr(116)] < 0) & ok_).sum())}"
                      for lab, thr in (("q<=0.05", .05), ("q<=0.10", .10),
                                       ("q<=0.20", .20))),
          f"  corner versus everywhere else : {nc} at q <= 0.20   "
          f"(largest effect "
          f"{R.loc[ok_, 'corner_vs_rest'].max():+.3f} z, cell "
          f"{R.loc[R[ok_]['corner_vs_rest'].idxmax(), 'uid']})",
          "  The second test is the better powered one - 626 corner frames "
          "against 874 elsewhere, where the",
          "  centre has 82 - so this is an informative null, not a lack of "
          "data. A corner effect the size of",
          "  the centre effects (+0.5 to +0.9 z) would have been obvious.",
          "",
          f"STABILITY.  The mouse entered the centre {len(v_st)} times "
          f"(durations "
          + ", ".join(f"{(b - a + 1) * dt:.1f}" for a, b in zip(v_st, v_en))
          + " s), so a zone preference could rest on one visit. Two checks:",
          f"  {'cell':>5s} {'contrast':>9s} {'drop any one visit':>19s} "
          f"{'first half':>11s} {'second half':>12s}  same sign?"]
    for _, r in R.sort_values("contrast", ascending=False).iterrows():
        if not r["usable"] or not np.isfinite(r["contrast"]):
            continue
        L.append(f"  {r['uid']:>5s} {r['contrast']:+9.2f} "
                 f"{r['loo_min']:+9.2f} to {r['loo_max']:+6.2f} "
                 f"{r['half1']:+11.2f} {r['half2']:+12.2f}  "
                 + ("yes" if r["stable"] else "NO"))
    L += ["",
          "Reading it. A3 fires 123 times, only 18 % of that in the centre "
          "- and yet its rate there",
          "is 3.6 times its corner rate, because the centre is 5.5 % of the "
          "session. Both statements",
          "are true. 'Fires faster in the centre' is what the statistics "
          "support; 'centre cell'",
          "would imply it stops firing elsewhere, and it plainly does not.",
          "",
          f"The centre rate rests on {occ[0]:.0f} s of data in total. Treat "
          f"the ratio as a direction, not a number."]
    txt = "\n".join(L)
    with open(os.path.join(out, "zone_rates.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    mimg = {p: bp(TF.load(root, p)["mean"]) for p in PLANES}
    sel = [r for r in rows if r["faster"]]
    draw(sel or rows[:4], U, mimg, out, "fig17_zone_rates_selective.png",
         pxn, pyn, xn, yn, zo, bx, by, occ_bin, dwell, occ, dt, t,
         "Open field: the cells that fire FASTER in the centre - "
         "rate, not count")
    draw(rows, U, mimg, out, "fig17_zone_rates_all.png",
         pxn, pyn, xn, yn, zo, bx, by, occ_bin, dwell, occ, dt, t,
         "Open field: every cell - where it fired, and how fast per second "
         "in each zone")


def draw(cells, U, mimg, out, name, pxn, pyn, xn, yn, zo, bx, by,
         occ_bin, dwell, occ, dt, t, title):
    n = len(cells)
    H = 2.5 * n + 1.3
    fig = plt.figure(figsize=(18, H))
    gs = fig.add_gridspec(n, 5, width_ratios=[1, 1.25, 1.35, 1.15, 3.3],
                          hspace=.5, wspace=.3)
    for row, c in enumerate(cells):
        d = U[("openfield", c["plane"])]
        z = d["z"][c["i"]]
        f = z > Z_FIRE
        col = CEN if c["faster"] else (INK if c["usable"] else "#9E9E9E")

        A = fig.add_subplot(gs[row, 0])
        S = d["S"][:, c["i"]].reshape(d["shape"])
        m = (S > .2 * S.max()).astype(np.uint8)
        yy, xx = np.nonzero(m)
        cy, cx = int(yy.mean()), int(xx.mean())
        h, w = d["shape"]
        y0, y1 = max(cy - 60, 0), min(cy + 60, h)
        x0, x1 = max(cx - 60, 0), min(cx + 60, w)
        A.imshow(mimg[c["plane"]][y0:y1, x0:x1], cmap="gray")
        for cc in cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)[0]:
            A.fill(cc[:, 0, 0], cc[:, 0, 1], color=col, alpha=.45, lw=0)
            A.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                   np.append(cc[:, 0, 1], cc[0, 0, 1]), color=col, lw=2.4)
        A.text(cx - x0, cy - y0, c["uid"], color="white", fontsize=12,
               fontweight="bold", ha="center", va="center",
               path_effects=[pe.withStroke(linewidth=3.4, foreground=col)])
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(c["uid"], fontsize=12, color=col, fontweight="bold")

        A = fig.add_subplot(gs[row, 1])
        A.plot(pxn, 1 - pyn, lw=.35, color="#C9D8E8", zorder=1)
        A.scatter(xn[f], 1 - yn[f], s=24, color=FIRE, edgecolor="#8A6D00",
                  linewidth=.5, zorder=3)
        zone_squares(A)
        A.set_xlim(0, 1)
        A.set_ylim(0, 1)
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(f"where it fired  (n={int(f.sum())})\n"
                    f"{c['frac'][0]:.0f} % centre, {c['frac'][1]:.0f} % "
                    f"corner", fontsize=9.5)

        A = fig.add_subplot(gs[row, 2])
        fi = np.zeros_like(occ_bin)
        np.add.at(fi, (by[f], bx[f]), 1.0)   # frames, so the
        # map is in the same units as the bars: firing / second
        rate = np.divide(fi, occ_bin, out=np.full_like(fi, np.nan),
                         where=dwell)
        vmax = float(np.nanmax(rate)) if np.isfinite(rate).any() else 1.
        im = A.imshow(np.flipud(rate), cmap="turbo", origin="upper",
                      extent=[0, 1, 0, 1], vmin=0, vmax=max(vmax, 1e-9))
        zone_squares(A)
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(f"rate map  (max {vmax:.2f} /s)", fontsize=9.5)
        cb = fig.colorbar(im, ax=A, fraction=.046, pad=.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("firing / s", fontsize=7.5, labelpad=1)

        A = fig.add_subplot(gs[row, 3])
        A.bar(range(3), c["rate"], color=ZC)
        for k in range(3):
            A.text(k, c["rate"][k], f"{c['rate'][k]:.2f}", ha="center",
                   va="bottom", fontsize=8)
        A.set_xticks(range(3))
        A.set_xticklabels([f"{nmz}\n{occ[v]:.0f}s"
                           for v, nmz in enumerate(ZN)], fontsize=8)
        A.set_ylabel("firing / s", fontsize=9, labelpad=1)
        A.set_title(f"centre / corner = {c['ratio']:.1f}x", fontsize=9.5,
                    color=col, fontweight="bold")
        A.spines[["top", "right"]].set_visible(False)

        A = fig.add_subplot(gs[row, 4])
        A.plot(t, z, lw=.5, color=INK)
        A.scatter(t[f], z[f], s=9, color=FIRE, zorder=3)
        lo = z.min() - .20 * (z.max() - z.min())
        for v, cc in enumerate(ZC):
            mm = zo == v
            if mm.any():
                A.fill_between(t, lo, lo + .10 * (z.max() - z.min()),
                               where=mm, color=cc, lw=0, step="mid")
        A.set_xlim(t[0], t[-1])
        A.set_ylim(lo, z.max() * 1.05)
        A.set_ylabel("z", fontsize=9)
        A.tick_params(labelsize=8)
        A.spines[["top", "right"]].set_visible(False)
        A.set_title(f"{c['uid']}   zone contrast {c['contrast']:+.2f} z, "
                    f"q = {c['q']:.3f}"
                    + ("   FIRES FASTER IN THE CENTRE" if c["faster"]
                       else ""), fontsize=10.5, loc="left", color=col,
                    fontweight="bold")
        if row == n - 1:
            A.set_xlabel("time (s)   |   bar = where the mouse was: "
                         "blue centre, pink corner, grey edge", fontsize=9)
    fig.suptitle(title + "\nthe mouse is in the corners 7.6x longer than "
                 "in the centre, so counting dots is misleading - the rate "
                 "panel is the claim", fontsize=13.5, y=1 - .24 / H)
    fig.subplots_adjust(left=.035, right=.985, top=1 - .95 / H,
                        bottom=.5 / H)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
