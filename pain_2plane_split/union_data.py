"""union_data.py  -  the canonical 31-cell set, for every downstream analysis.

WHY EVERYTHING SHOULD READ THIS
  Per-session curated sets cap any cross-session analysis at the neurons
  EXTRACT happened to detect twice. The audit showed that is a detection
  limit, not an anatomy one: 25 of 27 pain cells and 20 of 28 open-field
  cells sit on a soma in the OTHER session's mean image.

  transfer_footprints.py therefore registers the union of both sessions'
  footprints and solves all of them jointly on BOTH movies. 31 neurons
  (plane A 13, plane B 18) then have a trace in each session, with shared
  ids A1..A13 and B1..B18 - so "A1 in the pain session" and "A1 in the open
  field" are the same neuron by construction.

  The union is not simply both sets stacked. Eight open-field footprints sat
  on top of a cell already in it, closer than least squares can separate
  (footprint correlation 0.53-0.89, against 0.167 for the worst pair within
  a curated session), so they are recorded in union_cells.csv as merged_of
  and not added. Keeping them made the solve degenerate: one soma's signal
  came out as a positive trace on one copy and a negative trace on the
  other (trace r down to -0.85, one cell's raw F negative in every frame),
  which manufactured two "corner-preferring" cells that were nothing but
  the sign-flipped halves of centre cells.

  This module is the single place that loads it, so no analysis re-derives
  the transfer and they cannot drift apart.

WHAT A ROW CARRIES
  uid            shared id, e.g. A1
  plane          A or B
  source         which session detected it (pain, or open field)
  matched        was it detected independently in both (17 of 31)
  merged_of      an open-field footprint that overlapped this cell too
                 closely to be given its own trace, so it was folded in
                 here. Blank for most cells.
  <ses>_anat     footprint brightness over its ring in that session's mean
                 image, in image SDs. Below 1 means the transfer landed on
                 nothing there - drop those before interpreting.
  <ses>_active   is the trace separable from 200 shape-matched background
                 ROIs in that session (skew + lag-1 autocorrelation, Fisher,
                 p <= 0.05)

  A transferred footprint is a hypothesis, not a detection. Filter on anat
  before drawing conclusions; `usable()` below does it.

USAGE
  from union_data import load_union, usable
  U = load_union()                       # dict per (session, plane)
  cells = U["cells"]                     # the 39-row table
  z = U[("openfield", "A")]["z"]         # (18, 1500)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

OF_SESSION = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
              "2026-09-11_15-27-52")
PA_SESSION = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
              "2026-09-11_15-56-48")
ROOTS = {"openfield": os.path.join(OF_SESSION, "output_split"),
         "pain": os.path.join(PA_SESSION, "output_split")}
ANAT_MIN = 1.0      # image SDs; below this the transfer landed on nothing
PLANES = ("A", "B")


def _cellstr(a):
    return [str(np.asarray(x).ravel()[0]).strip()
            for x in np.asarray(a).ravel()]


def load_union():
    """Every union trace, keyed by (session, plane), plus the cell table."""
    out = {}
    tbl = os.path.join(ROOTS["pain"], "match", "union_cells.csv")
    if not os.path.exists(tbl):
        raise SystemExit("run transfer_footprints.py first - "
                         f"{tbl} is missing")
    out["cells"] = pd.read_csv(tbl, dtype={"pain_cell": str,
                                           "of_cell": str})
    for ses, root in ROOTS.items():
        for plane in PLANES:
            f = os.path.join(root, "union", f"plane_{plane}",
                             "union_traces.mat")
            if not os.path.exists(f):
                raise SystemExit(f"missing {f} - re-run "
                                 "transfer_footprints.py")
            m = loadmat(f)
            n = np.asarray(m["dff"]).shape[0]
            S3 = np.asarray(m["spatial_weights"])
            hh, w, _ = S3.shape
            out[(ses, plane)] = dict(
                labels=_cellstr(m["labels"]),
                verdict=_cellstr(m["verdict"]),
                source=_cellstr(m["source"]),
                matched=np.asarray(m["matched"]).ravel().astype(bool),
                t=np.asarray(m["t_s"]).ravel(),
                F=np.asarray(m["F_raw"]), dff=np.asarray(m["dff"]),
                z=np.asarray(m["z"]),
                anat=np.asarray(m["anat_sd"]).ravel(),
                p_active=np.asarray(m["p_active"]).ravel(),
                S=S3.reshape(hh * w, n), shape=(hh, w),
                fs=float(np.asarray(m["fs_hz"]).ravel()[0]))
    return out


def usable(U, session, plane, anat_min=ANAT_MIN):
    """Indices whose footprint actually lands on a soma in this session.

    Not filtered on activity: a neuron that was quiet here is still a valid
    row - that IS the measurement. Only transfers that landed on nothing are
    dropped, because their trace is background whatever it looks like.
    """
    a = U[(session, plane)]["anat"]
    return np.flatnonzero(np.isfinite(a) & (a >= anat_min))


def summary(U):
    rows = []
    for plane in PLANES:
        for ses in ("pain", "openfield"):
            d = U[(ses, plane)]
            rows.append(dict(plane=plane, session=ses, n=len(d["labels"]),
                             on_soma=int(np.nansum(d["anat"] >= ANAT_MIN)),
                             active=int(np.nansum(d["p_active"] <= .05)),
                             frames=d["dff"].shape[1],
                             seconds=round(float(d["t"][-1]), 1)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    U = load_union()
    print(f"{len(U['cells'])} union neurons\n")
    print(summary(U).to_string(index=False))
    print("\nper plane, usable in BOTH sessions "
          f"(footprint on a soma, anat >= {ANAT_MIN} SD):")
    for plane in PLANES:
        a = set(usable(U, "pain", plane).tolist())
        b = set(usable(U, "openfield", plane).tolist())
        print(f"  plane {plane}: {len(a & b)} of "
              f"{len(U[('pain', plane)]['labels'])}")
    C = U["cells"]
    print(f"\ndetected independently in both: {int(C['matched'].sum())}")
    print(f"active in both: "
          f"{int((C['pain_active'] & C['openfield_active']).sum())}")
