"""pain_cell_traces.py  -  raw Ca, dF/F and z per union cell, ready for event locking.

Reads the 39-cell union (`union_data.py`), not the per-session curated set.
The curated set gave the pain session 27 cells and the open field 28, with no
way to say which was which; the union gives both sessions the same 39 rows
under the same ids, so a row here is the same neuron as that row in the other
session, and the pain answer and the open-field answer can sit side by side.

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

All three come from `transfer_footprints.py`, which solves every union
footprint jointly against the raw movie by least squares, so a footprint
overlapping a neighbour does not carry the neighbour's transients. This
script does not recompute them - that is the point of one canonical source -
it re-checks them against the definitions above and then writes the
deliverable tables and figures.

READ THE anat COLUMN BEFORE USING A ROW
  A transferred footprint is a hypothesis. `anat` is the footprint's
  brightness over its surrounding ring in this session's mean image, in image
  SDs; below 1 it landed on nothing here and its trace is background whatever
  it looks like. Those rows are kept (dropping them silently would hide the
  transfer's failures) but drawn grey and flagged `on_soma = 0`.

EVENT LOCKING IS SET UP BUT NOT RUN
  It needs the behaviour scoring, which Hansol will do. Everything else is in
  place: every trace carries a time vector on the same base as the behaviour
  cameras (seconds since plane A frame 1, imaging clock), so a stimulus time
  in that base indexes straight into these traces. lock_events() below is the
  function to call with the scored times.

OUTPUT  ->  <session>\\output_split\\union\\plane_<X>\\
  union_cell_traces.mat / .csv    uid, verdict, anat, t_s, raw F, dF/F, z
  union_traces_raw.png            raw F per cell, A1..A18 / B1..B21
  union_traces_zscore.png         z per cell, same order

USAGE
  python pain_cell_traces.py                 # pain session, both planes
  python pain_cell_traces.py --session <dir> # or the open field
  python pain_cell_traces.py --both          # both sessions
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import savemat

from extract_qc_cells import detrend
from union_data import OF_SESSION, PA_SESSION, ANAT_MIN, load_union

VCOL = {"active": "#1C6E8C", "quiet": "#B8860B"}
OFF_SOMA = "#AAAAAA"


def robust_z(x):
    med = np.median(x)
    sd = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / (sd if sd > 0 else 1e-12)


def lock_events(t_s, traces, event_times, pre=3.0, post=5.0, fs=4.6083):
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


def qc_definitions(d):
    """Re-derive z from dff and report the agreement.

    Cheap, and it has caught a real problem before: if the stored z ever
    stops being 'robust z of the detrended dF/F', every amplitude statement
    downstream is wrong and nothing else would notice.
    """
    det, _ = detrend(d["dff"])
    mine = np.stack([robust_z(det[i]) for i in range(det.shape[0])])
    dev = np.abs(mine - d["z"]).max()
    r = min(float(np.corrcoef(mine[i], d["z"][i])[0, 1])
            for i in range(mine.shape[0]))
    return dev, r


def run(session, plane, U):
    ses = "openfield" if "openfield" in os.path.basename(session) else "pain"
    d = U[(ses, plane)]
    cells = U["cells"]
    k = len(d["labels"])
    uid, verd = d["labels"], d["verdict"]
    anat, t_s = d["anat"], d["t"]
    F, dff, z = d["F"], d["dff"], d["z"]
    on_soma = (anat >= ANAT_MIN).astype(int)
    src = d["source"]
    matched = d["matched"]
    if F.shape[1] != len(t_s):
        raise SystemExit(f"plane {plane}: {F.shape[1]} samples but "
                         f"{len(t_s)} frame times")
    dev, rmin = qc_definitions(d)
    if rmin < 0.999:
        raise SystemExit(f"plane {plane}: stored z does not match "
                         f"robust_z(detrend(dff)), worst r = {rmin:.4f}")

    out = os.path.join(session, "output_split", "union", f"plane_{plane}")
    savemat(os.path.join(out, "union_cell_traces.mat"),
            dict(uid=np.array(uid, dtype=object),
                 verdict=np.array(verd, dtype=object),
                 source=np.array(src, dtype=object), matched=matched,
                 anat_sd=anat, on_soma=on_soma, t_s=t_s,
                 F_raw=F, dff=dff, z=z, fs_hz=d["fs"],
                 session=ses,
                 time_base="seconds since plane A frame 1, imaging clock",
                 f0="20th percentile of each cell's raw trace",
                 z_def="detrended dF/F / (1.4826 * MAD)",
                 anat_def=f"footprint over ring in image SDs; "
                          f"< {ANAT_MIN} means the footprint landed on "
                          f"nothing in this session"))
    T = len(t_s)
    pd.DataFrame(dict(uid=np.repeat(uid, T),
                      verdict=np.repeat(verd, T),
                      anat_sd=np.repeat(np.round(anat, 3), T),
                      on_soma=np.repeat(on_soma, T),
                      t_s=np.tile(t_s, k), F_raw=F.ravel(),
                      dff=dff.ravel(), z=z.ravel())
                 ).to_csv(os.path.join(out, "union_cell_traces.csv"),
                          index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for arr, name, ylab, fn in (
            (F, "raw Ca fluorescence (movie units, no baseline removed)",
             "F", "union_traces_raw.png"),
            (z, "z-score of detrended dF/F  (robust sd = 1.4826 x MAD)",
             "z", "union_traces_zscore.png")):
        step = np.percentile(arr, 99) - np.percentile(arr, 1)
        step = max(step * 0.8, 1e-9)
        fig, ax = plt.subplots(figsize=(15, max(5, 0.52 * k)))
        for i in range(k):
            c = VCOL.get(verd[i], "#555555") if on_soma[i] else OFF_SOMA
            ax.plot(t_s, arr[i] - np.median(arr[i]) + (k - 1 - i) * step,
                    lw=.45, color=c)
            tag = uid[i] if on_soma[i] else f"{uid[i]}*"
            ax.text(-0.006 * t_s[-1], (k - 1 - i) * step, tag, ha="right",
                    va="center", fontsize=10, fontweight="bold", color=c)
        ax.set_xlim(0, t_s[-1])
        ax.set_yticks([])
        ax.set_xlabel("time (s), imaging clock, 0 = plane A frame 1")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.plot([t_s[-1] * .99] * 2, [0, step], color="k", lw=2.5)
        ax.text(t_s[-1] * .985, step / 2, f"{step:.3g} {ylab}", rotation=90,
                ha="right", va="center", fontsize=9)
        n_off = int((on_soma == 0).sum())
        ax.set_title(
            f"{os.path.basename(session)[:28]} plane {plane}: {name}\n"
            f"{k} union cells, ids shared with the other session  |  "
            f"blue = active here, amber = quiet here"
            + (f", grey* = footprint on nothing here "
               f"(anat < {ANAT_MIN} SD, n = {n_off}), trace is background"
               if n_off else ""), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fn), dpi=140, bbox_inches="tight")
        plt.close(fig)

    nz = {v: verd.count(v) for v in sorted(set(verd))}
    # "detected here" = EXTRACT found it in THIS session: either this is the
    # session whose footprint seeded the union row, or the row is a matched
    # pair and so was detected in both.
    seed = "pain" if ses == "pain" else "open field"
    here = int(np.sum([(s == seed) or bool(m) for s, m in zip(src, matched)]))
    print(f"  plane {plane}: {k} union cells {nz}, on a soma "
          f"{int(on_soma.sum())}/{k}, detected here {here}, "
          f"{T} samples, {t_s[-1]:.0f} s, F {F.min():.0f}-{F.max():.0f}, "
          f"|z| max {np.abs(z).max():.1f}  (z re-derived, max dev {dev:.2e})")
    _ = cells
    return uid, verd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PA_SESSION)
    ap.add_argument("--both", action="store_true",
                    help="both sessions, so the two sets of rows line up")
    a = ap.parse_args()
    U = load_union()
    sessions = [PA_SESSION, OF_SESSION] if a.both else [a.session]
    for s in sessions:
        print(f"union traces for {os.path.basename(s)}")
        for plane in ("A", "B"):
            run(s, plane, U)
    print("\nwrote union_cell_traces.mat / .csv and the two figures in each "
          "<session>\\output_split\\union\\plane_<X>\\")
    print("event locking: call lock_events(t_s, z, stimulus_times) once the "
          "video is scored;\nstimulus times must be on the same imaging "
          "clock as t_s. Cells with on_soma = 0 should be\ndropped first - "
          "their trace is background, not a neuron.")


if __name__ == "__main__":
    main()
