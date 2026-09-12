"""openfield_track.py  -  automatic mouse tracking in the open field.

No manual scoring: the floor is backlit white and the mouse is black, so the
animal is separable by brightness - but ONLY over the floor.

WHY THE ARENA MASK IS THE WHOLE PROBLEM
  The camera FOV (1600x1200) is far larger than the floor. In frame you also
  get the transparent box the floor sits in, the bench on the left, a DARK
  background filling the right third, white tape strips on the box walls and
  dark tape strips on the box lip. "Largest dark blob in the frame" locks onto
  the dark right-hand background: measured over the whole session it gave a
  body of 923,038-955,852 px (half the frame) sitting still at x~1060, i.e.
  1632/1632 frames tracking the wall. So the floor must be isolated first and
  everything outside it discarded, which is also what removes the tether's run
  out to the rig.

  The floor is the saturated white square and nothing else. Measured sweep on
  the session median image (largest bright component, holes filled):
    >180  640x680  fill 0.68   <- climbs into the white tape strips on the wall
    >200  623x669  fill 0.66   <- same, plus bleed to the right
    >220  571x625  fill 0.66   <- ragged right edge, tape spikes at the top
    >235  566x531  fill 0.75   <- close, right edge still ragged
    >245  469x527  fill 0.89   <- the floor, clean
    >250  453x527  fill 0.91
  245 it is. The residual 0.11 of unfilled bbox is a shadow bite on the left
  edge, so the mask is taken as the FITTED RECTANGLE (minAreaRect, measured
  525x463 at 89.4 deg = axis-aligned to within 0.6 deg) rather than the ragged
  threshold blob: the floor is a rectangle, and a rectangle makes the zone
  geometry well defined.

HOW THE MOUSE IS SEPARATED FROM ITS CABLE
  1  find the floor as above, fill it, then erode by ARENA_ERODE so the dark
     box wall just outside the floor edge cannot leak in as a thin ring.
  2  threshold dark pixels inside the arena -> mouse + the cable segment that
     is still over the floor.
  3  morphological OPENING with a disk wider than the cable but narrower than
     the body. The cable is ~10-15 px across, the body ~60-80 px, so a disk of
     radius 12 erases the tether and leaves the animal.
  4  largest remaining component = body, if its area is plausible. Centroid =
     position.

Every step is checked in the QC montage rather than assumed - run --check
first and look at it.

CENTRE VERSUS CORNER
  The floor rectangle is cut into the standard 3x3 grid of equal cells: the
  middle cell is "centre", the four corner cells are "corner", the four
  remaining edge cells are "edge". Fractions are taken of width and height
  separately, because the floor is 469x527 and not square. Reported per frame
  so the calcium analysis can use occupancy-normalised maps later.

OUTPUTS  ->  <session>\output_split\tracking\
  openfield_track.csv      per frame: t, x, y, speed, zone, area, ok
  openfield_track_qc.png   segmentation on sampled frames
  openfield_summary.txt    distance, speed, occupancy, time in each zone
  openfield_trajectory.png path, occupancy heatmap, speed histogram

USAGE
  python openfield_track.py --check          # look before trusting it
  python openfield_track.py
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

SESSION = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52")
ARENA_THR = 245        # only the saturated white floor; see the sweep above
ARENA_ERODE = 8        # keep the dark box wall outside the floor edge out
DARK_THR = 80          # a mouse pixel over the lit floor
OPEN_RADIUS = 12       # > cable half-width, < body half-width
MIN_BODY_PX = 800      # a real body at this magnification
MAX_BODY_PX = 60000    # measured bodies are 6k-12k px; 60k means it broke
CENTER_FRAC = 1 / 3    # 3x3 grid: middle cell
CORNER_FRAC = 1 / 3    # 3x3 grid: the four corner cells


def find_video(session):
    d = os.path.join(session, "MiceVideo2", "MiceVideo")
    v = sorted(f for f in os.listdir(d) if f.lower().endswith(".avi"))
    if not v:
        raise SystemExit(f"no AVI in {d}")
    return os.path.join(d, v[0])


def find_arena(cap, fps, n_sample=60):
    """The white floor, as a fitted rectangle, from a median over the session.

    A single frame would include the mouse as a dark hole; the median over
    frames spread across the recording removes it.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 2, 1), n_sample).astype(int)
    acc = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            acc.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    med = np.median(np.stack(acc), axis=0).astype(np.uint8)
    h, w = med.shape

    bright = (med > ARENA_THR).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (41, 41)))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    if nl < 2:
        raise SystemExit(f"no region brighter than {ARENA_THR} - is the "
                         "floor lit?")
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lab == i).astype(np.uint8)
    # fill interior holes so marks on the paper stay inside the arena
    ff = blob.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    blob = (blob | (1 - ff)).astype(np.uint8)

    # the floor is a rectangle: fit one rather than keep the ragged threshold
    cnt = max(cv2.findContours(blob, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[0],
              key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    quad = cv2.boxPoints(rect).astype(np.int32)
    mask = np.zeros_like(blob)
    cv2.fillConvexPoly(mask, quad, 1)
    if ARENA_ERODE:
        mask = cv2.erode(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ARENA_ERODE + 1,) * 2))
    x, y, ww, hh = cv2.boundingRect(mask)
    return med, mask, (int(x), int(y), int(ww), int(hh)), quad


def zones(bbox, shape):
    """centre / corner / edge label image: the standard 3x3 grid.

    Width and height fractions are taken separately - the floor is 469x527,
    so a single "side" would make the cells non-square in one axis and push
    the corner cells off the floor.
    """
    x, y, w, h = bbox
    Z = np.full(shape, 2, np.uint8)          # 2 = edge
    cw, ch = int(w * CENTER_FRAC), int(h * CENTER_FRAC)
    cx, cy = x + w // 2, y + h // 2
    Z[max(cy - ch // 2, 0):cy + ch // 2, max(cx - cw // 2, 0):cx + cw // 2] = 0
    kw, kh = int(w * CORNER_FRAC), int(h * CORNER_FRAC)
    for yy in (y, y + h - kh):
        for xx in (x, x + w - kw):
            Z[max(yy, 0):yy + kh, max(xx, 0):xx + kw] = 1
    return Z


def body(gray, arena, ring, se):
    """Mouse body: dark inside the arena, opened to drop the tether.

    Positions are floor-only by instruction, so when the animal presses
    against a wall the part of it hanging over the box lip is cut off and the
    centroid is pulled inward. Measured: a free-standing body is 6-9k px, one
    pressed against the bottom wall drops to ~2.5-3.3k. That is not hidden -
    `clipped` flags every frame where the body touches the arena boundary, so
    the wall-hugging frames can be excluded or down-weighted downstream.
    """
    dark = ((gray < DARK_THR) & (arena > 0)).astype(np.uint8)
    op = cv2.morphologyEx(dark, cv2.MORPH_OPEN, se)
    nl, lab, st, cen = cv2.connectedComponentsWithStats(op, 8)
    if nl < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    a = st[i, cv2.CC_STAT_AREA]
    if a < MIN_BODY_PX or a > MAX_BODY_PX:
        return None
    m = lab == i
    return dict(x=float(cen[i][0]), y=float(cen[i][1]),
                area=int(a), mask=m, dark=dark, opened=op,
                clipped=int(np.any(m & (ring > 0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=SESSION)
    ap.add_argument("--check", action="store_true",
                    help="write the QC montage on sampled frames and stop")
    ap.add_argument("--step", type=int, default=1,
                    help="analyse every Nth frame (1 = all)")
    ap.add_argument("--arena-cm", type=float, default=None,
                    help="arena side in cm, to report distance in cm")
    a = ap.parse_args()

    vid = find_video(a.session)
    outdir = os.path.join(a.session, "output_split", "tracking")
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(vid)}  {fps:.2f} fps  {total} frames  "
          f"{total / fps / 60:.2f} min")

    med, arena, bbox, quad = find_arena(cap, fps)
    print(f"white floor: bbox x={bbox[0]} y={bbox[1]} w={bbox[2]} h={bbox[3]}"
          f"  {arena.sum()} px = {100 * arena.mean():.1f} % of the FOV")
    # is DARK_THR actually in the gap between mouse and floor?
    inside = med[arena > 0]
    print(f"floor brightness (median image, inside the arena): "
          f"p1 {np.percentile(inside, 1):.0f}  "
          f"median {np.median(inside):.0f}  "
          f"p99 {np.percentile(inside, 99):.0f}  "
          f"-> DARK_THR {DARK_THR} is "
          f"{np.percentile(inside, 1) - DARK_THR:.0f} below the floor's 1st "
          f"percentile")
    Z = zones(bbox, arena.shape)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * OPEN_RADIUS + 1,) * 2)
    # boundary ring of the arena, to flag bodies cut off at the floor edge
    ring = arena - cv2.erode(arena, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)))

    if a.check:
        tiles = []
        for t in (10, 60, 120, 180, 240, 300):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            b = body(g, arena, ring, se)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, [quad], -1, (0, 200, 255), 2)
            vis[(Z == 0) & (arena > 0)] = (
                0.75 * vis[(Z == 0) & (arena > 0)]
                + 0.25 * np.array([0, 180, 0])).astype(np.uint8)
            vis[(Z == 1) & (arena > 0)] = (
                0.75 * vis[(Z == 1) & (arena > 0)]
                + 0.25 * np.array([180, 0, 180])).astype(np.uint8)
            if b is not None:
                vis[b["dark"] > 0] = (0.5 * vis[b["dark"] > 0]
                                      + 0.5 * np.array([0, 0, 255])).astype(
                    np.uint8)
                vis[b["mask"]] = (0.35 * vis[b["mask"]]
                                  + 0.65 * np.array([0, 255, 255])).astype(
                    np.uint8)
                cv2.circle(vis, (int(b["x"]), int(b["y"])), 12,
                           (255, 255, 255), 2)
                txt = f"{t}s  body {b['area']} px"
            else:
                txt = f"{t}s  NO BODY"
            cv2.putText(vis, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, .9,
                        (0, 255, 255), 2)
            tiles.append(cv2.resize(vis, (vis.shape[1] // 3,
                                          vis.shape[0] // 3)))
        cap.release()
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3),
                                       np.uint8)]) if r.shape[1] < wmax else r
                for r in rows]
        p = os.path.join(outdir, "openfield_track_qc.png")
        cv2.imwrite(p, np.vstack(rows))
        print(f"\nwrote {p}")
        print("red = all dark pixels inside the arena (mouse + tether)")
        print("yellow = what survived the opening and was taken as the body")
        print("green = centre zone, magenta = corners")
        print("If yellow follows the cable instead of the body, raise "
              "OPEN_RADIUS.")
        return

    # ---- full pass ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows = []
    k = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % a.step:
            k += 1
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        b = body(g, arena, ring, se)
        if b is None:
            rows.append(dict(frame=k, t=k / fps, x=np.nan, y=np.nan,
                             area=0, zone=-1, ok=0, clipped=0))
        else:
            zi = int(Z[int(round(b["y"])), int(round(b["x"]))])
            rows.append(dict(frame=k, t=k / fps, x=b["x"], y=b["y"],
                             area=b["area"], zone=zi, ok=1,
                             clipped=b["clipped"]))
        k += 1
    cap.release()

    D = pd.DataFrame(rows)
    # interpolate short dropouts so speed is not spiked by gaps
    D["x"] = D["x"].interpolate(limit=int(fps), limit_direction="both")
    D["y"] = D["y"].interpolate(limit=int(fps), limit_direction="both")
    dt = a.step / fps
    dx = D["x"].diff()
    dy = D["y"].diff()
    D["speed_px_s"] = np.hypot(dx, dy) / dt
    # a mouse cannot cross the arena in one frame; drop obvious jumps
    lim = np.nanpercentile(D["speed_px_s"], 99.5) * 3
    D.loc[D["speed_px_s"] > lim, "speed_px_s"] = np.nan

    px_per_cm = None
    if a.arena_cm:
        px_per_cm = min(bbox[2], bbox[3]) / a.arena_cm
        D["speed_cm_s"] = D["speed_px_s"] / px_per_cm

    csv = os.path.join(outdir, "openfield_track.csv")
    D.to_csv(csv, index=False)

    ZN = {0: "centre", 1: "corner", 2: "edge", -1: "lost"}
    det = 100 * D["ok"].mean()
    dist_px = np.nansum(np.hypot(dx, dy))
    lines = [
        f"video            {os.path.basename(vid)}",
        f"frames analysed  {len(D)} of {total} (step {a.step})",
        f"duration         {len(D) * dt:.1f} s",
        f"detection rate   {det:.2f} % of frames",
        f"white floor      {bbox[2]} x {bbox[3]} px "
        f"({100 * arena.mean():.1f} % of the FOV; everything outside it is "
        f"ignored)",
        f"body clipped     {100 * D['clipped'].mean():.1f} % of frames "
        f"(pressed against a wall, part of the body off the floor)",
        f"body area        median {np.median(D.loc[D['ok'] == 1, 'area']):.0f}"
        f" px, range {D.loc[D['ok'] == 1, 'area'].min():.0f}-"
        f"{D.loc[D['ok'] == 1, 'area'].max():.0f}",
        "",
        f"distance         {dist_px:.0f} px"
        + (f"  =  {dist_px / px_per_cm:.1f} cm" if px_per_cm else ""),
        f"median speed     {np.nanmedian(D['speed_px_s']):.1f} px/s"
        + (f"  =  {np.nanmedian(D['speed_px_s']) / px_per_cm:.2f} cm/s"
           if px_per_cm else ""),
        f"time moving      "
        f"{100 * np.nanmean(D['speed_px_s'] > 20):.1f} % (>20 px/s)",
        "",
        "zone occupancy (of detected frames)",
    ]
    ok = D[D["ok"] == 1]
    for z, nm in ZN.items():
        if z < 0:
            continue
        f = 100 * (ok["zone"] == z).mean() if len(ok) else np.nan
        lines.append(f"  {nm:7s} {f:5.1f} %   {(ok['zone'] == z).sum() * dt:6.1f} s")
    if not a.arena_cm:
        lines += ["", "pass --arena-cm <side in cm> to get distance in cm"]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(outdir, "openfield_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(med, cmap="gray")
    sc = ax[0].scatter(ok["x"], ok["y"], c=ok["t"], s=1.2, cmap="viridis")
    ax[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                  fill=False, color="orange", lw=1.5))
    ax[0].set_title("path, coloured by time")
    ax[0].axis("off")
    plt.colorbar(sc, ax=ax[0], label="s", fraction=.046)
    H, xe, ye = np.histogram2d(
        ok["x"], ok["y"], bins=40,
        range=[[bbox[0], bbox[0] + bbox[2]], [bbox[1], bbox[1] + bbox[3]]])
    ax[1].imshow(H.T, origin="upper", cmap="magma",
                 extent=[xe[0], xe[-1], ye[-1], ye[0]])
    ax[1].set_title("occupancy")
    ax[1].axis("off")
    s = D["speed_px_s"].dropna()
    ax[2].hist(s, bins=60, color="#1C6E8C")
    ax[2].set_xlabel("speed (px/s)")
    ax[2].set_ylabel("frames")
    ax[2].set_title(f"median {np.nanmedian(s):.0f} px/s")
    fig.suptitle(f"open field, {len(D) * dt / 60:.2f} min, "
                 f"detection {det:.1f} %", fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "openfield_trajectory.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {csv}\nwrote {p}")


if __name__ == "__main__":
    main()
