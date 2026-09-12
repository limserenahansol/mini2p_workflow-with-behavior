"""pain_cell_traces.py  -  raw Ca, dF/F and z-score per cell, ready for event locking.

Three trace forms per cell, as asked:
  raw F    the demixed fluorescence in movie units, no baseline removed.
           Shows the absolute brightness and any bleaching or drift, which
           dF/F hides.
  dF/F     (F - F0) / F0 with F0 the 20th percentile of that cell's own raw
           trace. NOT the mean: step 2's deltaF_over_F uses F0 = mean(trace),
           which inflates F0 for an active cell and shrinks its dF/F.
  z        the detrended dF/F divided by its robust sd (1.4826 x MAD), so
           amplitudes are comparable between cells with different baselines.
           Detrended with a 20 s running median first, because background
           ROIs in this session carry a raw lag-1 autocorrelation of 0.40 -
           without that step the z-score is dominated by drift.

Traces are solved jointly across all footprints on the raw movie
(least squares), so a footprint overlapping a neighbour does not carry the
neighbour's transients. Measured: this differs from a plain weighted average
only for the cells that actually overlap (r = 1.000 for all but two cells,
0.93 for the closest pair).

EVENT LOCKING IS SET UP BUT NOT RUN
  It needs the behaviour scoring, which Hansol will do. Everything else is
  in place: the output carries each cell's trace with a time vector on the
  same base as the behaviour cameras (seconds since plane A frame 1, imaging
  clock), so a stimulus time in that base indexes straight into these
  traces. lock_events() below is the function to call with the scored times.

OUTPUT  ->  <plane>\\curated\\
  pain_traces.mat / .csv    labels, t_s, raw F, dF/F, z
  pain_traces_raw.png       raw F per cell
  pain_traces_zscore.png    z per cell

USAGE
  python pain_cell_traces.py                 # pain session, both planes
  python pain_cell_traces.py --session <dir> # any session
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from extract_qc_cells import detrend, to_dff
from plot_cellmap_traces import demix_raw

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
VCOL = {"active": "#1C6E8C", "silent": "#B8860B", "SUSPECT": "#C1272D"}


def robust_z(x):
    med = np.median(x)
    sd = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / (sd if sd > 0 else 1e-12)


def lock_events(t_s, traces, event_times, pre=3.0, post=5.0, fs=4.6054):
    """Event-locked traces: (n_cells, n_events, n_samples) plus the lag axis.

    Not called yet - the stimulus times come from the manual scoring. Kept
    here so the conventions live next to the traces they apply to: event
    times must be in the same base as t_s (seconds since plane A frame 1,
    imaging clock), and a window is dropped rather than padded if it would
    run off either end of the recording.
    """
    n_pre, n_post = int(round(pre * fs)), int(round(post * fs))
    lag = np.arange(-n_pre, n_post + 1) / fs
    out, kept = [], []
    for et in np.atleast_1d(event_times):
        i = int(np.argmin(np.abs(t_s - et)))
        if i - n_pre < 0 or i + n_post >= traces.shape[1]:
            continue
        out.append(traces[:, i - n_pre:i + n_post + 1])
        kept.append(et)
    if not out:
        return lag, np.zeros((traces.shape[0], 0, len(lag))), []
    return lag, np.stack(out, 1), kept


def run(session, plane, outroot=None):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
        mn = np.array(h["output"]["info"]["summary_image"]).T
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    lb = pd.read_csv(os.path.join(cur, "curated_labels.csv"),
                     dtype={"label": str})
    lb = lb[lb["index"] > 0].sort_values("index")
    labels = lb["label"].tolist()
    qf = os.path.join(cur, "qc_cells", "qc_cells.csv")
    verd = ["?"] * k
    if os.path.exists(qf):
        q = pd.read_csv(qf)
        for _, r in q.iterrows():
            if 1 <= int(r["cell"]) <= k:
                verd[int(r["cell"]) - 1] = r["verdict"]

    F, _ = demix_raw(os.path.join(base, f"plane_{plane}_1.h5"), S,
                     (hh, w), mn)
    dff = to_dff(F)
    det, _ = detrend(dff)
    z = np.stack([robust_z(det[i]) for i in range(k)])

    tf = pd.read_csv(os.path.join(session, "output_split", "timestamps",
                                  "plane_frame_times.csv"))
    t_s = tf.loc[tf["plane"] == plane, "t_s"].to_numpy()
    if len(t_s) != F.shape[1]:
        raise SystemExit(f"plane {plane}: {F.shape[1]} samples but "
                         f"{len(t_s)} frame times")

    savemat(os.path.join(cur, "pain_traces.mat"),
            dict(labels=np.array(labels, dtype=object), verdict=
                 np.array(verd, dtype=object), t_s=t_s, F_raw=F, dff=dff,
                 z=z, fs_hz=1.0 / np.median(np.diff(t_s)),
                 time_base="seconds since plane A frame 1, imaging clock",
                 f0="20th percentile of each cell's raw trace",
                 z_def="detrended dF/F / (1.4826 * MAD)"))
    T = len(t_s)
    pd.DataFrame(dict(label=np.repeat(labels, T), verdict=np.repeat(verd, T),
                      t_s=np.tile(t_s, k), F_raw=F.ravel(),
                      dff=dff.ravel(), z=z.ravel())
                 ).to_csv(os.path.join(cur, "pain_traces.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for arr, name, ylab, fn in (
            (F, "raw Ca fluorescence (movie units, no baseline removed)",
             "F", "pain_traces_raw.png"),
            (z, "z-score of detrended dF/F  (robust sd = 1.4826 x MAD)",
             "z", "pain_traces_zscore.png")):
        step = np.percentile(arr, 99) - np.percentile(arr, 1)
        step = max(step * 0.8, 1e-9)
        fig, ax = plt.subplots(figsize=(15, max(5, 0.52 * k)))
        for i in range(k):
            ax.plot(t_s, arr[i] - np.median(arr[i]) + (k - 1 - i) * step,
                    lw=.45, color=VCOL.get(verd[i], "#555555"))
            ax.text(-0.006 * t_s[-1], (k - 1 - i) * step,
                    f"{labels[i]}", ha="right", va="center", fontsize=10,
                    fontweight="bold", color=VCOL.get(verd[i], "#555555"))
        ax.set_xlim(0, t_s[-1])
        ax.set_yticks([])
        ax.set_xlabel("time (s), imaging clock, 0 = plane A frame 1")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.plot([t_s[-1] * .99] * 2, [0, step], color="k", lw=2.5)
        ax.text(t_s[-1] * .985, step / 2, f"{step:.3g} {ylab}", rotation=90,
                ha="right", va="center", fontsize=9)
        ax.set_title(f"{os.path.basename(session)[:28]} plane {plane}: "
                     f"{name}\nblue = active, amber = silent, red = SUSPECT",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(cur, fn), dpi=140, bbox_inches="tight")
        plt.close(fig)

    nz = {v: verd.count(v) for v in set(verd)}
    print(f"  plane {plane}: {k} cells {nz}, {T} samples, "
          f"{t_s[-1]:.0f} s, F {F.min():.0f}-{F.max():.0f}, "
          f"|z| max {np.abs(z).max():.1f}")
    return labels, verd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PAIN)
    a = ap.parse_args()
    print(f"traces for {os.path.basename(a.session)}")
    for plane in ("A", "B"):
        run(a.session, plane)
    print("\nwrote pain_traces.mat / .csv and the two figures in each "
          "<plane>\\curated\\")
    print("event locking: call lock_events(t_s, z, stimulus_times) once the "
          "video is scored;\nstimulus times must be on the same imaging "
          "clock as t_s.")


if __name__ == "__main__":
    main()
