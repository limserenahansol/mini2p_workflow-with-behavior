"""split_timestamps.py  -  one time base for both depths and the behaviour cameras.

WHY THIS REPLACES pain_2plane_step3_parse_timestamps.m
  That step was written for the max-projected movie and keeps "the timestamp
  of each pair's first frame (odd-indexed raw frames)". For split planes
  that is right for plane A and gives plane B a fixed 109 ms bias, because
  plane B is the even raw frames. Its hard-coded frame counts (18000 raw /
  9000 paired) match neither session here either.

THE THREE THINGS THAT WOULD OTHERWISE GO WRONG

  1  Which depth is which. `CellVideo_CHA_Info.tdms` carries a `Slice`
     channel, so this is recorded, not inferred: Slice alternates 1,2,1,2...
     strictly in both sessions and odd raw frames are Slice 1. Parity would
     work, but reading the recorded value cannot drift.
       plane A = Slice 1 = ETL 65 um relative / 139.6 um absolute
       plane B = Slice 2 = ETL 15 um relative /  89.6 um absolute, 109 ms later

  2  Two clocks, ~37 s apart. The imaging side (`CHA/Time` and
     `SignalSync_N/Time`) runs about 37 s behind the behaviour side
     (`MiceVideoN/Ref Time`, which is the clock that agrees with the session
     folder name). The recordings really are simultaneous - the spans agree
     to within 0.05 s - so comparing ABSOLUTE timestamps across the two
     clocks would put the data 37 s out. The sync line is the bridge: it is
     the camera's frame clock recorded on the imaging clock, so anchoring the
     camera's first frame to the first sync pulse removes the offset without
     ever trusting either wall clock.

  3  Dropped behaviour frames. The cameras did not save every frame the sync
     line pulsed for: 6 missing in the open field, 11 and 12 in the pain
     session, clustered, and in the pain session BOTH cameras drop at the
     same indices, so it is a system stall rather than a camera fault.
     Therefore `frame_index / 25` is not time - the error is cumulative and
     reaches 0.24-0.48 s by the end. Per-frame `Ref Time` is used instead.

  Residual alignment uncertainty is one sync interval, because the gap
  arithmetic closes: observed intervals + missing frames predicts the camera
  span to within 2-4 ms, leaving the sync line with exactly 1 extra interval
  (2 for pain MiceVideo2). So the camera-frame-to-sync-pulse index offset is
  unknown by 1-2 pulses = 40-80 ms, not the ~0.3 s a pulse-count difference
  alone would suggest.

  The sync line cannot be pinned down further by matching jitter: its
  intervals are 40.000 ms with sd 0.022 ms and two distinct values, i.e. a
  generated clock with no jitter signature, and cross-correlating it against
  the camera intervals gives r ~ 0 at every lag.

TIME BASE
  Everything is returned as seconds since **plane A frame 1**, on the imaging
  clock. Plane B starts at +0.109 s. Behaviour frames can start slightly
  negative (the camera began before the first imaging frame: -0.58 s in the
  open field).

OUTPUTS  ->  <session>\\output_split\\timestamps\\
  timestamps.mat          all vectors, for MATLAB
  timestamps_report.txt   the measured numbers and the assumptions
  plane_frame_times.csv   plane, frame (1-based), t_s
  behav_frame_times.csv   camera, frame (1-based), t_s, gap_before

USAGE
  python split_timestamps.py --session "D:\\...openfield..."
  python split_timestamps.py            # both sessions
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from nptdms import TdmsFile

SESSIONS = [
    ("openfield", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
                  "2026-09-11_15-27-52"),
    ("pain", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
             "2026-09-11_15-56-48"),
]
CAM_NOMINAL_DT = 0.040      # s, 25 Hz behaviour camera
GAP_CLOSE_TOL = 0.050       # s; if the gap arithmetic misses by more, stop
PLANE_OF_SLICE = {1: "A", 2: "B"}


def ms(a):
    """datetime64[ms] array from a TDMS time channel."""
    return np.array(a[:], dtype="datetime64[ms]")


def sec(a, t0):
    return (a - t0).astype("float64") / 1000.0


# ------------------------------------------------------------------ imaging
def read_imaging(session):
    p = os.path.join(session, "CellVideo1", "CellVideo_CHA_Info.tdms")
    g = TdmsFile.read(p)["CHA"]
    t = ms(g["Time"])
    sl = np.array([int(x) for x in g["Slice"][:]])
    lost = np.array([int(x) for x in g["FrameLost"][:]])
    if set(np.unique(sl)) != {1, 2}:
        raise SystemExit(f"expected Slice values 1 and 2, got "
                         f"{np.unique(sl)}")
    # the interleave must be strict, or splitting by Slice is not a split
    if not (np.all(sl[::2] == sl[0]) and np.all(sl[1::2] == sl[1])):
        raise SystemExit("Slice does not alternate strictly - the depths are "
                         "not frame-interleaved in this session")
    if sl[0] != 1:
        raise SystemExit(f"first raw frame is Slice {sl[0]}, not 1; "
                         "pain_2plane_step1_split_planes assigns odd frames "
                         "to plane A and would mislabel the depths")
    return t, sl, lost


# ---------------------------------------------------------------- behaviour
def read_cameras(session):
    out = []
    for mv in sorted(glob.glob(os.path.join(session, "MiceVideo*",
                                            "*reference.tdms"))):
        folder = os.path.basename(os.path.dirname(mv))
        n = re.search(r"(\d+)$", folder)
        tf = TdmsFile.read(mv)
        grp = tf.groups()[0]
        ch = [c.name for c in grp.channels() if "Time" in c.name][0]
        t = ms(grp[ch])
        d = np.diff(t).astype("float64") / 1000.0
        gaps = np.maximum(np.round(d / CAM_NOMINAL_DT).astype(int) - 1, 0)

        # the gap count has to reproduce the span, otherwise the sync
        # alignment below is resting on a miscount
        n_nominal = len(d) + int(gaps.sum())
        span = float((t[-1] - t[0]).astype("float64") / 1000.0)
        err = abs(n_nominal * CAM_NOMINAL_DT - span)
        if err > GAP_CLOSE_TOL:
            raise SystemExit(
                f"{folder}: {len(d)} intervals + {gaps.sum()} missing predicts"
                f" {n_nominal * CAM_NOMINAL_DT:.3f} s but the span is "
                f"{span:.3f} s ({err * 1000:.0f} ms out) - the dropped-frame "
                f"count is wrong, do not trust the alignment")

        sync = None
        if n:
            sp = glob.glob(os.path.join(session, "SyncInformation",
                                        f"SignalSync_{n.group(1)}_*.tdms"))
            sp = [x for x in sp if not x.endswith("_index")]
            for cand in sp:
                s = TdmsFile.read(cand)["Sync"]["Time"][:]
                if len(s):
                    sync = (os.path.basename(cand), ms(s))
                    break
        out.append(dict(name=folder, t=t, gaps=gaps, n_nominal=n_nominal,
                        span=span, gap_err_ms=err * 1000, sync=sync))
    return out


# ---------------------------------------------------------------------- main
def run(tag, session):
    print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")
    raw_t, slice_id, lost = read_imaging(session)
    t0 = raw_t[slice_id == 1][0]          # plane A frame 1 = the time origin

    planes = {}
    for s, nm in PLANE_OF_SLICE.items():
        tt = sec(raw_t[slice_id == s], t0)
        planes[nm] = tt
        d = np.diff(tt)
        print(f"  plane {nm} (Slice {s}): {len(tt)} frames, "
              f"t {tt[0]:+.3f} .. {tt[-1]:.3f} s, "
              f"rate {(len(tt) - 1) / (tt[-1] - tt[0]):.4f} Hz, "
              f"dt {np.median(d) * 1000:.1f} ms")
    print(f"  frames flagged FrameLost: {int(np.sum(lost != 0))}")
    print(f"  plane B lags plane A by {planes['B'][0] * 1000:.0f} ms")

    cams, lines = [], []
    for c in read_cameras(session):
        if c["sync"] is None:
            print(f"  {c['name']}: NO sync channel with data - cannot place "
                  f"this camera on the imaging clock, skipped")
            continue
        sname, sy = c["sync"]
        # camera frame 1 is anchored to sync pulse 1 (imaging clock); the
        # index offset is unknown by the leftover sync intervals
        anchor = sy[0]
        t_cam = sec(c["t"], c["t"][0]) + sec(np.array([anchor]), t0)[0]
        amb = ((len(sy) - 1) - c["n_nominal"]) * CAM_NOMINAL_DT
        clock_off = (c["t"][0] - anchor).astype("float64") / 1000.0
        cams.append(dict(name=c["name"], t=t_cam, gaps=c["gaps"],
                         ambiguity_s=amb, clock_offset_s=clock_off,
                         sync_file=sname, n_sync=len(sy)))
        print(f"  {c['name']}: {len(c['t'])} frames saved, "
              f"{int(c['gaps'].sum())} missing in "
              f"{int(np.sum(c['gaps'] > 0))} gaps "
              f"(arithmetic closes to {c['gap_err_ms']:.0f} ms)")
        print(f"      t {t_cam[0]:+.3f} .. {t_cam[-1]:.3f} s   "
              f"behaviour clock is {clock_off:+.3f} s ahead of imaging   "
              f"index ambiguity {amb * 1000:+.0f} ms")

    # ---- overlap: how much of each stream has the other ----
    lines += [f"===== timestamps: {tag} =====",
              f"session: {session}",
              "",
              "time base: seconds since plane A frame 1, imaging clock",
              "",
              f"imaging, {len(raw_t)} raw frames, FrameLost "
              f"{int(np.sum(lost != 0))}"]
    for nm in ("A", "B"):
        tt = planes[nm]
        lines.append(f"  plane {nm}: {len(tt):5d} frames  "
                     f"{tt[0]:+8.3f} .. {tt[-1]:8.3f} s  "
                     f"{(len(tt) - 1) / (tt[-1] - tt[0]):.4f} Hz")
    lines += ["", "behaviour cameras (placed on the imaging clock via the "
                  "sync line)"]
    for c in cams:
        lines.append(f"  {c['name']}: {len(c['t']):6d} frames  "
                     f"{c['t'][0]:+8.3f} .. {c['t'][-1]:8.3f} s  "
                     f"{int(c['gaps'].sum())} dropped  "
                     f"clock offset {c['clock_offset_s']:+.3f} s  "
                     f"ambiguity {c['ambiguity_s'] * 1000:+.0f} ms  "
                     f"[{c['sync_file']}, {c['n_sync']} pulses]")
    if cams:
        lines += ["", "usable overlap (both imaging and behaviour present)"]
        for c in cams:
            for nm in ("A", "B"):
                lo = max(planes[nm][0], c["t"][0])
                hi = min(planes[nm][-1], c["t"][-1])
                ov = max(hi - lo, 0.0)
                lines.append(
                    f"  plane {nm} x {c['name']}: {lo:+.3f} .. {hi:.3f} s "
                    f"= {ov:.1f} s "
                    f"({100 * ov / (planes[nm][-1] - planes[nm][0]):.1f} % of "
                    f"the imaging, "
                    f"{100 * ov / (c['t'][-1] - c['t'][0]):.1f} % of the "
                    f"video)")
        lines += ["",
                  "Do NOT align by absolute timestamps across the two "
                  "clocks: the imaging",
                  "clock runs ~37 s behind the behaviour clock, while the "
                  "recordings are",
                  "simultaneous. Do NOT convert behaviour frame index to "
                  "time by dividing",
                  "by 25: frames were dropped and the error is cumulative."]

    outdir = os.path.join(session, "output_split", "timestamps")
    os.makedirs(outdir, exist_ok=True)
    txt = "\n".join(lines)
    print()
    print(txt)
    with open(os.path.join(outdir, "timestamps_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")

    pd.concat([pd.DataFrame(dict(plane=nm,
                                 frame=np.arange(1, len(planes[nm]) + 1),
                                 t_s=planes[nm]))
               for nm in ("A", "B")]).to_csv(
        os.path.join(outdir, "plane_frame_times.csv"), index=False)
    if cams:
        pd.concat([pd.DataFrame(dict(
            camera=c["name"], frame=np.arange(1, len(c["t"]) + 1),
            t_s=c["t"],
            gap_before=np.concatenate(([0], c["gaps"]))))
            for c in cams]).to_csv(
            os.path.join(outdir, "behav_frame_times.csv"), index=False)

    from scipy.io import savemat
    md = dict(time_base="seconds since plane A frame 1, imaging clock",
              plane_A_t=planes["A"], plane_B_t=planes["B"],
              plane_A_slice=1, plane_B_slice=2,
              plane_A_depth_um_rel=65.0, plane_B_depth_um_rel=15.0,
              plane_A_depth_um_abs=139.6, plane_B_depth_um_abs=89.6,
              n_frame_lost=int(np.sum(lost != 0)))
    for i, c in enumerate(cams, 1):
        md[f"behav{i}_name"] = c["name"]
        md[f"behav{i}_t"] = c["t"]
        md[f"behav{i}_gap_before"] = np.concatenate(([0], c["gaps"]))
        md[f"behav{i}_clock_offset_s"] = c["clock_offset_s"]
        md[f"behav{i}_align_ambiguity_s"] = c["ambiguity_s"]
    savemat(os.path.join(outdir, "timestamps.mat"), md)
    print(f"\nwrote {outdir}")
    return planes, cams


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append",
                    help="session folder (default: both)")
    a = ap.parse_args()
    if a.session:
        for s in a.session:
            run(os.path.basename(s.rstrip("/\\")), s)
    else:
        for tag, s in SESSIONS:
            run(tag, s)


if __name__ == "__main__":
    main()
