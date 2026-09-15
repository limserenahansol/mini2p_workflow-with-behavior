"""of_cell_rows.py  -  one row per neuron: where it is, where it fired, its trace.

The layout Hansol asked for, four columns per neuron:

  1  FOV          the field of view around that cell, its footprint filled
                  and numbered, so there is no doubt which cell this row is
  2  PATH         the mouse's path in grey, with a yellow dot at every
                  position where THIS cell was firing (z > 1.5). The blue
                  square is the centre zone, the four pink squares are the
                  corners - the same 3x3 grid the statistics use
  3  PLACE MAP    P(z > 1.5 | location): of the time spent in each bin, the
                  fraction with this cell firing. Occupancy-normalised, so
                  a bin is not bright merely for being visited often; bins
                  with under 0.5 s of dwell are left white
  4  TRACE        z over the session in black, the same firing frames in
                  yellow, and a colour bar underneath showing where the
                  mouse was at that moment (blue centre, pink corner, grey
                  elsewhere)

Row order: centre-preferring cells first, then corner-preferring, then the
rest, each labelled with its verdict so the figure answers "who is a centre
cell and how did it fire" by itself.

Z_FIRE = 1.5 is a display threshold for the dots and the map. It is NOT
what the statistics use - the zone test in openfield_place_cells.py works
on the continuous z with a circular-shift null and never thresholds.

OUTPUT  ->  <open field>\\output_split\\atlas\\
  fig15_rows_selective.png   the centre and corner cells
  fig15_rows_all.png         every cell, same layout

USAGE
  python of_cell_rows.py
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
from openfield_place_cells import load_tracking  # noqa: E402
from union_data import ANAT_MIN, OF_SESSION, PLANES, ROOTS, load_union  # noqa: E402

Z_FIRE = 1.5
N_BIN = 12
MIN_DWELL_S = 0.5
CEN, COR, OTH = "#2166AC", "#E7298A", "#BBBBBB"
FIRE = "#FFC000"
INK, DIM = "#1A1A1A", "#666666"


def bp(img):
    v = img.astype(np.float32)
    lo, hi = np.percentile(v, [2, 99.5])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def arena_geometry():
    cap = cv2.VideoCapture(oft.find_video(OF_SESSION))
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, arena, bbox, _ = oft.find_arena(cap, fps)
    cap.release()
    return bbox, arena.shape


def zone_squares(ax):
    """The 3x3 grid in normalised arena coordinates, as in the statistics."""
    c = oft.CENTER_FRAC
    k = oft.CORNER_FRAC
    ax.add_patch(plt.Rectangle((.5 - c / 2, .5 - c / 2), c, c, fill=False,
                               ec=CEN, lw=2.0))
    for xx in (0, 1 - k):
        for yy in (0, 1 - k):
            ax.add_patch(plt.Rectangle((xx, yy), k, k, fill=False, ec=COR,
                                       lw=2.0))


def rows_figure(cells, U, geom, tr, out, name, title):
    bbox, _ = geom
    t = U[("openfield", "A")]["t"]
    okf = tr["ok"].to_numpy() == 1
    tb = tr["t_s"].to_numpy()
    xs = np.interp(t, tb[okf], tr["x"].to_numpy()[okf])
    ys = np.interp(t, tb[okf], tr["y"].to_numpy()[okf])
    zo = np.round(np.interp(t, tb[okf],
                            tr["zone"].to_numpy()[okf].astype(float))
                  ).astype(int)
    xn = np.clip((xs - bbox[0]) / bbox[2], 0, 1)
    yn = np.clip((ys - bbox[1]) / bbox[3], 0, 1)
    pxn = np.clip((tr["x"].to_numpy()[okf] - bbox[0]) / bbox[2], 0, 1)
    pyn = np.clip((tr["y"].to_numpy()[okf] - bbox[1]) / bbox[3], 0, 1)
    dt = float(np.median(np.diff(t)))
    bx = np.clip((xn * N_BIN).astype(int), 0, N_BIN - 1)
    by = np.clip((yn * N_BIN).astype(int), 0, N_BIN - 1)
    occ = np.zeros((N_BIN, N_BIN))
    np.add.at(occ, (by, bx), dt)
    mask = occ >= MIN_DWELL_S

    n = len(cells)
    H = 2.45 * n + 1.2
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1.15, 1.15, 3.6],
                          hspace=.42, wspace=.22)
    mean_img = {p: bp(TF.load(ROOTS["openfield"], p)["mean"])
                for p in PLANES}

    for i, c in enumerate(cells):
        d = U[("openfield", c["plane"])]
        j = d["labels"].index(c["uid"])
        z = d["z"][j]
        fire = z > Z_FIRE
        col = {"centre": CEN, "corner": COR}.get(c["pref"], INK)

        # 1 FOV
        A = fig.add_subplot(gs[i, 0])
        S = d["S"][:, j].reshape(d["shape"])
        m = (S > .2 * S.max()).astype(np.uint8)
        yy, xx = np.nonzero(m)
        cy, cx = int(yy.mean()), int(xx.mean())
        h, w = d["shape"]
        r = 60
        y0, y1 = max(cy - r, 0), min(cy + r, h)
        x0, x1 = max(cx - r, 0), min(cx + r, w)
        A.imshow(mean_img[c["plane"]][y0:y1, x0:x1], cmap="gray")
        cs = cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)[0]
        for cc in cs:
            A.fill(cc[:, 0, 0], cc[:, 0, 1], color=col, alpha=.45, lw=0)
            A.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                   np.append(cc[:, 0, 1], cc[0, 0, 1]), color=col, lw=2.4)
        A.text(cx - x0, cy - y0, c["uid"], color="white", fontsize=12,
               fontweight="bold", ha="center", va="center",
               path_effects=[pe.withStroke(linewidth=3.4, foreground=col)])
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(f"{c['uid']}   {c['pref'].upper()}"
                    + ("" if c["usable"] else "   not on a soma"),
                    fontsize=11, color=col, fontweight="bold", loc="left")

        # 2 path + firing locations
        A = fig.add_subplot(gs[i, 1])
        A.plot(pxn, 1 - pyn, lw=.35, color="#AFC6DE", zorder=1)
        A.scatter(xn[fire], 1 - yn[fire], s=26, color=FIRE,
                  edgecolor="#8A6D00", linewidth=.5, zorder=3)
        zone_squares(A)
        A.set_xlim(0, 1)
        A.set_ylim(0, 1)
        A.set_xticks([0, .5, 1])
        A.set_yticks([0, .5, 1])
        A.tick_params(labelsize=7)
        A.set_title(f"yellow = z > {Z_FIRE} (n={int(fire.sum())})",
                    fontsize=9.5)

        # 3 P(fire | location)
        A = fig.add_subplot(gs[i, 2])
        fi = np.zeros((N_BIN, N_BIN))
        np.add.at(fi, (by[fire], bx[fire]), dt)
        P = np.divide(fi, occ, out=np.full_like(fi, np.nan), where=mask)
        A.imshow(np.flipud(P), cmap="inferno", origin="upper",
                 extent=[0, 1, 0, 1],
                 vmin=0, vmax=max(np.nanmax(P), 1e-9))
        zone_squares(A)
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(f"P(z>{Z_FIRE} | location)", fontsize=9.5)

        # 4 trace with the zone bar
        A = fig.add_subplot(gs[i, 3])
        A.plot(t, z, lw=.5, color=INK)
        A.scatter(t[fire], z[fire], s=9, color=FIRE, zorder=3)
        lo = z.min() - .18 * (z.max() - z.min())
        for v, cc in ((0, CEN), (1, COR), (2, OTH)):
            mm = zo == v
            if mm.any():
                A.fill_between(t, lo, lo + .10 * (z.max() - z.min()),
                               where=mm, color=cc, lw=0, step="mid")
        A.set_xlim(t[0], t[-1])
        A.set_ylim(lo, z.max() * 1.05)
        A.set_ylabel("z", fontsize=9)
        A.tick_params(labelsize=8)
        A.spines[["top", "right"]].set_visible(False)
        A.set_title(f"contrast {c['contrast']:+.2f} z, q = {c['q']:.3f}"
                    if np.isfinite(c["contrast"]) else "not tested",
                    fontsize=9.5, loc="left")
        if i == n - 1:
            A.set_xlabel("time (s)   |   bar = mouse location: "
                         "blue centre, pink corner, grey other", fontsize=9)

    fig.suptitle(title, fontsize=14, y=1 - .22 / H)
    # inches, not fractions: on a tall figure a fractional rect
    # leaves a band of white above the first row
    fig.subplots_adjust(left=.04, right=.985, top=1 - .85 / H,
                        bottom=.5 / H)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def main():
    U = load_union()
    root = ROOTS["openfield"]
    out = os.path.join(root, "atlas")
    os.makedirs(out, exist_ok=True)
    P = pd.read_csv(os.path.join(root, "place", "place_cells.csv"))
    key = {r["cell"]: r for _, r in P.iterrows()}
    print("open field: per-cell rows")
    geom = arena_geometry()
    print(f"  arena {geom[0][2]}x{geom[0][3]} px at "
          f"({geom[0][0]},{geom[0][1]})")
    tr = load_tracking(OF_SESSION)

    cells = []
    for plane in PLANES:
        d = U[("openfield", plane)]
        for i, lab in enumerate(d["labels"]):
            r = key.get(lab)
            cells.append(dict(
                plane=plane, uid=lab,
                usable=bool(d["anat"][i] >= ANAT_MIN),
                pref=(r["preference"] if r is not None else "not tested"),
                contrast=(float(r["contrast"]) if r is not None else np.nan),
                q=(float(r["q_contrast"]) if r is not None else np.nan)))
    order = {"centre": 0, "corner": 1}
    cells.sort(key=lambda c: (order.get(c["pref"], 2),
                             -abs(c["contrast"]) if np.isfinite(c["contrast"])
                             else 0))

    sel = [c for c in cells if c["pref"] in ("centre", "corner")]
    nc = sum(c["pref"] == "centre" for c in sel)
    nk = sum(c["pref"] == "corner" for c in sel)
    if sel:
        rows_figure(sel, U, geom, tr, out, "fig15_rows_selective.png",
                    f"Open field: the zone-selective neurons - "
                    f"{nc} centre-preferring, {nk} corner-preferring "
                    f"(q <= 0.05, circular-shift null)")
    rows_figure(cells, U, geom, tr, out, "fig15_rows_all.png",
                f"Open field: all {len(cells)} cells, ordered "
                f"centre-preferring first, then corner, then the rest")


if __name__ == "__main__":
    main()
