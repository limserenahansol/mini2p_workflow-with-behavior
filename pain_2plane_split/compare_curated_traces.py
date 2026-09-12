"""compare_curated_traces.py  -  does the trace method change the answer?

WHY THIS EXISTS
  The curated pass keeps two trace estimates per cell:
    EXTRACT T-step      robust, but it drops cells. One clause in
                        remove_redundant cannot be opened by any threshold
                        (is_S_tiny includes S_smooth_area_1 == 0), and it
                        removes exactly the footprints curation added -
                        3 of 16 in open-field plane B, 2 of 16 in pain B.
    joint least squares (S'S)\\(S'M), the T-step's core. Demixes overlapping
                        footprints jointly and by construction cannot drop a
                        cell, so the curated label mapping survives.

  Using EXTRACT where it works and least squares where it does not would put
  two different methods in one dataset. Least squares is therefore used
  everywhere - but that is only defensible if the two agree where both
  exist. This measures the agreement instead of asserting it.

  Agreement is reported on the DETRENDED traces, because a shared slow drift
  would inflate the correlation and hide a real difference in the fast
  component, which is the part every downstream analysis uses.

OUTPUT  ->  <this folder>\\curated_trace_comparison.txt and .png

USAGE
  python compare_curated_traces.py
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd

from extract_qc_cells import detrend

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = [
    ("OF", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
           "2026-09-11_15-27-52"),
    ("PA", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
           "2026-09-11_15-56-48"),
]


def load(folder):
    f = os.path.join(folder, "final_analysis_results.mat")
    if not os.path.exists(f):
        return None
    with h5py.File(f, "r") as h:
        o = h["output"]
        out = dict(method=None, ls=None, ex=None)
        if "trace_method" in o:
            v = np.array(o["trace_method"]).ravel()
            out["method"] = "".join(chr(int(c)) for c in v)
        if "temporal_weights_ls" in o:
            out["ls"] = np.array(o["temporal_weights_ls"])
        if "temporal_weights_extract" in o:
            out["ex"] = np.array(o["temporal_weights_extract"])
        out["tw"] = np.array(o["temporal_weights"])
    return out


def main():
    rows, panels = [], []
    for tag, sess in SESSIONS:
        for pl in ("A", "B"):
            folder = os.path.join(sess, "output_split", f"plane_{pl}",
                                  "curated")
            d = load(folder)
            if d is None:
                print(f"  {tag}-{pl}: no curated result yet")
                continue
            lab = pd.read_csv(os.path.join(folder, "curated_labels.csv"),
                              dtype={"label": str})
            lab = lab[lab["index"] > 0].sort_values("index")
            if d["ls"] is None or d["ex"] is None:
                rows.append(dict(plane=f"{tag}-{pl}", n=d["tw"].shape[0],
                                 method=d["method"] or "?",
                                 r_median=np.nan, r_min=np.nan,
                                 note="EXTRACT dropped cells, "
                                      "no comparison possible"))
                print(f"  {tag}-{pl}: {d['method']} only "
                      f"({d['tw'].shape[0]} cells) - EXTRACT unavailable")
                continue
            ls, ex = d["ls"], d["ex"]
            if ls.shape != ex.shape:
                print(f"  {tag}-{pl}: shape mismatch {ls.shape} vs "
                      f"{ex.shape}")
                continue
            lsd, _ = detrend(ls)
            exd, _ = detrend(ex)
            r = np.array([np.corrcoef(lsd[i], exd[i])[0, 1]
                          for i in range(ls.shape[0])])
            rows.append(dict(plane=f"{tag}-{pl}", n=len(r),
                             method=d["method"] or "?",
                             r_median=float(np.median(r)),
                             r_min=float(np.nanmin(r)),
                             note=""))
            print(f"  {tag}-{pl}: {len(r)} cells, detrended r "
                  f"median {np.median(r):.4f}, min {np.nanmin(r):.4f}")
            worst = int(np.nanargmin(r))
            panels.append((f"{tag}-{pl}", lab["label"].tolist(), r,
                           lsd[worst], exd[worst],
                           lab["label"].tolist()[worst]))

    D = pd.DataFrame(rows)
    L = ["===== curated traces: least squares versus EXTRACT T-step =====",
         "",
         "Correlation is computed on DETRENDED traces (20 s running median "
         "removed),",
         "so a shared slow drift cannot inflate the agreement.",
         "",
         D.to_string(index=False),
         ""]
    ok = D.dropna(subset=["r_median"])
    if len(ok):
        if ok["r_min"].min() >= 0.95:
            L.append(f"Every cell where both estimates exist agrees at "
                     f"r >= {ok['r_min'].min():.3f}, so using least squares")
            L.append("everywhere does not change the traces - it only keeps "
                     "the cells EXTRACT drops.")
        else:
            L.append(f"Worst agreement is r = {ok['r_min'].min():.3f}. The "
                     f"two estimates are NOT interchangeable;")
            L.append("report which method produced the traces in any result "
                     "that uses them.")
    else:
        L.append("No plane has both estimates, so no comparison was "
                 "possible.")
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(HERE, "curated_trace_comparison.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, len(panels),
                               figsize=(5.2 * len(panels), 6.4),
                               squeeze=False)
        for j, (name, labels, r, lsw, exw, wl) in enumerate(panels):
            ax[0, j].bar(range(len(r)), r, color="#1C6E8C")
            ax[0, j].set_xticks(range(len(r)))
            ax[0, j].set_xticklabels(labels, rotation=90, fontsize=8)
            ax[0, j].set_ylim(min(0.9, np.nanmin(r) - 0.02), 1.002)
            ax[0, j].set_ylabel("detrended r")
            ax[0, j].set_title(f"{name}: least squares vs EXTRACT")
            t = np.arange(len(lsw)) / 4.6054
            ax[1, j].plot(t, exw, lw=.4, color="#999999",
                          label="EXTRACT T-step")
            ax[1, j].plot(t, lsw, lw=.4, color="#C1272D",
                          label="least squares")
            ax[1, j].set_title(f"worst cell: {wl}  (r = {np.nanmin(r):.3f})")
            ax[1, j].set_xlabel("time (s)")
            ax[1, j].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(HERE, "curated_trace_comparison.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
