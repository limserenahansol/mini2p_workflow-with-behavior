"""linked_cells_report.py  -  the same neuron in both sessions, side by side.

This is the join that makes the question askable: for every cell that could
be followed from the open field into the pain session, show its open-field
place verdict next to its pain-session trace. Once the pain video is scored,
the pin-prick / heat columns drop straight into the same table and the
question becomes "was THIS neuron corner-preferring and pin-prick
responsive".

WHAT A ROW IS
  One of the 31 UNION neurons (transfer_footprints.py), which already has a
  trace in both sessions - so there is nothing left to pair here. This
  replaces an earlier version that joined matched detection pairs; that
  capped the table at the neurons EXTRACT found twice, and after both
  downstream analyses moved to union ids it silently matched nothing and
  reported that no zone-selective cell could be followed.

WHAT TO CHECK BEFORE BELIEVING A ROW
  anat, per session: the footprint's brightness over its surrounding ring in
  that session's mean image, in image SDs. Below 1 the transfer landed on
  nothing there and the trace is background whatever it looks like. 26 of 31
  clear 1 SD in both sessions.

OUTPUT  ->  <pain session>\\output_split\\match\\
  linked_cells.csv        one row per pair, with both sessions' verdicts
  linked_cells.png        open-field trace above pain trace, per pair
  linked_cells_report.txt

USAGE
  python linked_cells_report.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
SHAPE_MIN = 0.5
FS = 4.6054


def dff(root, plane):
    """Union dF/F for this session, keyed by the shared uid."""
    m = loadmat(os.path.join(root, "union", f"plane_{plane}",
                             "union_traces.mat"))
    labels = [str(np.asarray(x).ravel()[0]).strip()
              for x in np.asarray(m["labels"]).ravel()]
    return {lab: np.asarray(m["dff"], float)[i]
            for i, lab in enumerate(labels)}


def qc(root, plane):
    f = os.path.join(root, f"plane_{plane}", "curated", "qc_cells",
                     "qc_cells.csv")
    lb = pd.read_csv(os.path.join(root, f"plane_{plane}", "curated",
                                  "curated_labels.csv"), dtype={"label": str})
    lb = lb[lb["index"] > 0].sort_values("index")
    q = pd.read_csv(f)
    q = q.merge(lb[["index", "label"]], left_on="cell", right_on="index")
    return {r["label"]: (r["verdict"], int(r["usable"]))
            for _, r in q.iterrows()}


def main():
    """One row per UNION neuron, joined to the open-field zone result.

    This used to join match_cells.csv (per-session labels) to the place
    result. Both downstream analyses moved to the 39-cell union, whose ids
    are A1..A18 and B1..B21, so that join silently matched nothing and the
    report said no zone-selective cell could be followed. It now reads the
    union directly, where every row already has both sessions by
    construction and there is nothing to join on position at all.
    """
    up = os.path.join(PA, "match", "union_cells.csv")
    if not os.path.exists(up):
        raise SystemExit("run transfer_footprints.py first")
    U = pd.read_csv(up, dtype={"pain_cell": str, "of_cell": str})
    place_p = os.path.join(OF, "place", "place_cells.csv")
    P = pd.read_csv(place_p, dtype={"cell": str}) if os.path.exists(place_p) \
        else pd.DataFrame()

    rows = []
    for _, r in U.iterrows():
        pv, pc, pq = "not tested", np.nan, np.nan
        if len(P):
            hit = P[(P["plane"] == r["plane"]) & (P["cell"] == r["uid"])]
            if len(hit):
                pv = hit.iloc[0]["preference"]
                pc = float(hit.iloc[0]["contrast"])
                pq = float(hit.iloc[0]["q_contrast"])
        on_soma = (r["pain_anat"] >= 1) and (r["openfield_anat"] >= 1)
        rows.append(dict(
            plane=r["plane"], uid=r["uid"], source=r["source"],
            detected_twice="yes" if r["matched"] else "no",
            pain_anat=r["pain_anat"], of_anat=r["openfield_anat"],
            same_neuron="yes" if on_soma else "footprint on nothing",
            of_verdict="active" if r["openfield_active"] else "quiet",
            pa_verdict="active" if r["pain_active"] else "quiet",
            of_usable=int(r["openfield_anat"] >= 1),
            pa_usable=int(r["pain_anat"] >= 1),
            of_place=pv, of_zone_contrast=pc, of_zone_q=pq))
    D = pd.DataFrame(rows).sort_values(
        ["plane", "of_zone_q"], na_position="last")
    outdir = os.path.join(PA, "match")
    D.to_csv(os.path.join(outdir, "linked_cells.csv"), index=False)

    conf = D[D["same_neuron"] == "yes"]
    both = conf
    L = ["===== the same neuron in both sessions =====",
         "",
         f"{len(D)} union neurons, each measured in BOTH sessions; "
         f"{len(conf)} have a footprint that lands on a soma",
         f"in both (anat >= 1 image SD) and can be interpreted.",
         "",
         f"{int((D['detected_twice'] == 'yes').sum())} of them were "
         f"detected independently by EXTRACT in both sessions; the rest "
         f"got their",
         "trace by transferring the footprint, which is why every row has "
         "both columns.",
         "",
         f"  {'plane':5s} {'uid':>5s} {'found in':>11s} {'2x':>3s} "
         f"{'anat PA':>8s} {'anat OF':>8s} {'OF':>7s} {'PA':>7s} "
         f"{'OF place':>9s} {'contrast':>8s} {'q':>7s}"]
    for _, r in D.iterrows():
        cs = "" if not np.isfinite(r["of_zone_contrast"]) \
            else f"{r['of_zone_contrast']:+8.3f}"
        qs = "" if not np.isfinite(r["of_zone_q"]) \
            else f"{r['of_zone_q']:7.4f}"
        L.append(f"  {r['plane']:5s} {r['uid']:>5s} "
                 f"{r['source']:>11s} {r['detected_twice']:>3s} "
                 f"{r['pain_anat']:8.2f} {r['of_anat']:8.2f} "
                 f"{r['of_verdict']:>7s} {r['pa_verdict']:>7s} "
                 f"{r['of_place']:>9s} {cs:>8s} {qs:>7s}")
    place_hits = conf[conf["of_place"].isin(["centre", "corner"])]
    L += ["",
          (f"Zone-selective open-field cells that could be followed into the "
           f"pain session: "
           + (", ".join(f"{r['uid']} ({r['of_place']}, pain "
                        f"{r['pa_verdict']})"
                        for _, r in place_hits.iterrows())
              if len(place_hits) else "none")),
          "",
          "Next: once the pain video is scored, add pin-prick and heat "
          "response columns",
          "to linked_cells.csv. The pairing above is what lets a single "
          "neuron carry",
          "both answers.",
          "",
          "Caveat. 'Detected twice' rests on a footprint correlation of 0.5 "
          "in a dense",
          "plane, with residual positions of 2-7 px against an 8 px radius. "
          "Treat those",
          "as probable, not certain, and check the linked figure before "
          "using one.",
          "",
          "Eight further open-field footprints were dropped from the union "
          "because least",
          "squares cannot separate them from a cell already in it (footprint "
          "r 0.53-0.89,",
          "against 0.167 for the worst pair within a curated session). Their "
          "locations are",
          "in union_cells.csv as merged_of. Keeping them split one soma's "
          "signal into a",
          "positive and a negative copy: the two 'corner-preferring' cells "
          "an earlier run",
          "reported were the negative halves of centre cells (r = -0.82 and "
          "-0.84)."]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(outdir, "linked_cells_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")

    if not len(conf):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dO = {p: dff(OF, p) for p in ("A", "B")}
    dP = {p: dff(PA, p) for p in ("A", "B")}
    n = len(conf)
    fig, ax = plt.subplots(n, 2, figsize=(15, 1.5 * n),
                           gridspec_kw=dict(width_ratios=[1, 2]),
                           squeeze=False)
    for i, (_, r) in enumerate(conf.iterrows()):
        a = dO[r["plane"]][r["uid"]]
        b = dP[r["plane"]][r["uid"]]
        ax[i, 0].plot(np.arange(len(a)) / FS, a, lw=.45, color="#B23AA8")
        ax[i, 0].set_xlim(0, len(a) / FS)
        ax[i, 0].set_ylabel(f"{r['uid']}\nopen field", fontsize=9,
                            rotation=0, labelpad=30, va="center")
        ax[i, 1].plot(np.arange(len(b)) / FS, b, lw=.45, color="#1C6E8C")
        ax[i, 1].set_xlim(0, len(b) / FS)
        ax[i, 1].set_ylabel("pain", fontsize=9, rotation=0,
                            labelpad=22, va="center")
        ax[i, 1].text(.004, .93,
                      f"anat PA {r['pain_anat']:.1f} / OF {r['of_anat']:.1f} SD"
                      f"  |  OF {r['of_verdict']}"
                      f" / place {r['of_place']}  |  PA {r['pa_verdict']}",
                      transform=ax[i, 1].transAxes, va="top", fontsize=8,
                      color="#555555")
        for a_ in ax[i]:
            a_.spines[["top", "right"]].set_visible(False)
            if i < n - 1:
                a_.set_xticks([])
        if i == 0:
            ax[i, 0].set_title("open field, 326 s", fontsize=10)
            ax[i, 1].set_title("pain, 651 s", fontsize=10)
    ax[-1, 0].set_xlabel("time (s)")
    ax[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{n} union neurons measured in both sessions "
                 f"(dF/F; footprint on a soma in both)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .99])
    p = os.path.join(outdir, "linked_cells.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
