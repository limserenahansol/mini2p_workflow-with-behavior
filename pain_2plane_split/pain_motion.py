"""pain_motion.py  -  how much the mouse is moving, per frame, on the imaging clock.

WHY THIS EXISTS
  Two questions need a movement signal in the PAIN session, where there is no
  body tracking (the open field has tracking; the pain arena is a small box
  filmed from below and from the side):

  1  Is a neuron a movement or a freezing neuron rather than a pain neuron?
     Without a movement regressor, any cell that fires when the mouse moves
     looks stimulus-responsive, because the stimulus makes the mouse move.

  2  How late is the human's key press? The experimenter's hand entering the
     bottom view is a large, abrupt change in the image. Cross-correlating
     the scored stimulus taps against bottom-camera motion energy measures
     the tap delay from the video itself, instead of assuming it.

WHAT IS MEASURED
  Motion energy: mean absolute difference between consecutive frames, after
  converting to grey and downscaling to 160 x 120. The cameras are static, so
  everything this picks up is the mouse or the experimenter's hand. It is not
  a speed in cm/s - it is "how much of the image changed" - which is all that
  is needed for a regressor and for freezing bouts.

  bottom (MiceVideo1)  the stimulus side: hand, filament, laser, and the mouse
  side   (MiceVideo2)  the mouse's own movement, used for freezing

  Freezing is defined on the side camera: motion energy below a threshold for
  at least MIN_FREEZE_S continuously. The threshold is not a guess - it is the
  FREEZE_PCT-th percentile of this session's own smoothed energy, and the
  report shows how the bout count changes across nearby thresholds so the
  choice is visible rather than hidden.

TIME BASE
  Frames are mapped to the imaging clock through
  timestamps\\behav_frame_times.csv, never by dividing the frame index by 25:
  11 and 12 frames were dropped in these two cameras and the error is
  cumulative, reaching 0.44 s by the end of the session.

OUTPUT  ->  <pain session>\\output_split\\motion\\
  motion_energy.npz        per-camera per-frame energy and imaging-clock time
  motion_on_imaging.csv    energy resampled onto each plane's frame times
  freeze_bouts.csv         start, end, duration of every freezing bout
  motion_report.txt
  fig6_motion.png

USAGE
  python pain_motion.py                 # pain session, both cameras
  python pain_motion.py --session <dir>
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
CAMS = {"bottom": ("MiceVideo1", "MiceVideo"),
        "side": ("MiceVideo2", "MiceVideo")}
SMALL = (160, 120)
SMOOTH_S = 0.5          # boxcar on the energy before thresholding
MIN_FREEZE_S = 1.0      # a bout must last this long to count
FREEZE_PCT = 25.0       # threshold percentile of this session's own energy


def energy(path, small=SMALL):
    """Per-frame mean |difference| from the previous frame."""
    c = cv2.VideoCapture(path)
    if not c.isOpened():
        raise SystemExit(f"cannot open {path}")
    out, prev = [], None
    while True:
        ok, fr = c.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                       small).astype(np.float32)
        out.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    c.release()
    return np.asarray(out, np.float32)


def boxcar(x, n):
    if n <= 1:
        return x.copy()
    k = np.ones(int(n)) / float(int(n))
    return np.convolve(x, k, mode="same")


def bouts(mask, t, min_s):
    """Contiguous runs of mask lasting at least min_s, as (start, end) times."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    keep = [(t[a], t[b]) for a, b in zip(st, en) if t[b] - t[a] >= min_s]
    return keep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PAIN)
    a = ap.parse_args()
    root = os.path.join(a.session, "output_split")
    out = os.path.join(root, "motion")
    os.makedirs(out, exist_ok=True)

    B = pd.read_csv(os.path.join(root, "timestamps",
                                 "behav_frame_times.csv"))
    P = pd.read_csv(os.path.join(root, "timestamps",
                                 "plane_frame_times.csv"))

    E, T, L = {}, {}, []
    for cam, (d1, d2) in CAMS.items():
        folder = os.path.join(a.session, d1, d2)
        avi = [f for f in sorted(os.listdir(folder)) if f.endswith(".avi")]
        if len(avi) != 1:
            raise SystemExit(f"{folder}: expected one .avi, found {len(avi)}")
        print(f"  {cam}: {avi[0]} ...", flush=True)
        e = energy(os.path.join(folder, avi[0]))
        g = B[B["camera"] == d1].sort_values("frame")
        if len(g) != len(e):
            raise SystemExit(f"{cam}: {len(e)} decoded frames but "
                             f"{len(g)} frame times")
        E[cam], T[cam] = e, g["t_s"].to_numpy()
        fs = 1.0 / np.median(np.diff(T[cam]))
        L.append(f"  {cam:7s} {len(e)} frames, {fs:.2f} Hz, energy "
                 f"median {np.median(e):.3f}, p95 {np.percentile(e, 95):.3f}")
        print(L[-1], flush=True)

    np.savez(os.path.join(out, "motion_energy.npz"),
             **{f"e_{k}": v for k, v in E.items()},
             **{f"t_{k}": v for k, v in T.items()})

    # ---- freezing, on the side camera -------------------------------
    cam = "side"
    fs = 1.0 / np.median(np.diff(T[cam]))
    sm = boxcar(E[cam], max(1, round(SMOOTH_S * fs)))
    thr = np.percentile(sm, FREEZE_PCT)
    fz = bouts(sm < thr, T[cam], MIN_FREEZE_S)
    rows = [dict(start_s=round(s, 3), end_s=round(e_, 3),
                 duration_s=round(e_ - s, 3)) for s, e_ in fz]
    pd.DataFrame(rows).to_csv(os.path.join(out, "freeze_bouts.csv"),
                              index=False)
    tot = sum(r["duration_s"] for r in rows)
    L += ["",
          f"freezing on the {cam} camera: energy below the "
          f"{FREEZE_PCT:.0f}th percentile ({thr:.3f}) for >= "
          f"{MIN_FREEZE_S:.0f} s",
          f"  {len(rows)} bouts, {tot:.0f} s total "
          f"({100 * tot / (T[cam][-1] - T[cam][0]):.1f} % of the session), "
          f"median bout {np.median([r['duration_s'] for r in rows]):.1f} s"
          if rows else "  no bouts",
          "",
          "  threshold sensitivity (the choice is a threshold, so here is "
          "what it costs):"]
    for pct in (15, 20, 25, 30, 35):
        t2 = np.percentile(sm, pct)
        b2 = bouts(sm < t2, T[cam], MIN_FREEZE_S)
        s2 = sum(e_ - s for s, e_ in b2)
        L.append(f"    p{pct:<3d} thr {t2:.3f}  {len(b2):3d} bouts  "
                 f"{s2:5.0f} s  {100 * s2 / (T[cam][-1] - T[cam][0]):4.1f} %")

    # ---- resample onto the imaging frame times -----------------------
    cols = {}
    for plane in ("A", "B"):
        ts = P.loc[P["plane"] == plane, "t_s"].to_numpy()
        for cam in CAMS:
            f_ = boxcar(E[cam], max(1, round(SMOOTH_S /
                                             np.median(np.diff(T[cam])))))
            cols[(plane, cam)] = np.interp(ts, T[cam], f_)
        cols[(plane, "t_s")] = ts
    frames = []
    for plane in ("A", "B"):
        frames.append(pd.DataFrame({
            "plane": plane, "t_s": cols[(plane, "t_s")],
            "motion_bottom": cols[(plane, "bottom")],
            "motion_side": cols[(plane, "side")]}))
    M = pd.concat(frames, ignore_index=True)
    M.to_csv(os.path.join(out, "motion_on_imaging.csv"), index=False)
    L += ["",
          "resampled onto the imaging frame times (0.5 s boxcar first, then "
          "linear interpolation)",
          f"  {len(M)} rows in motion_on_imaging.csv, columns "
          f"motion_bottom and motion_side"]

    txt = "\n".join(["===== movement in the pain session =====", ""] + L)
    with open(os.path.join(out, "motion_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
