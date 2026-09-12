"""linked_cells_report.py  -  the same neuron in both sessions, side by side.

This is the join that makes the question askable: for every cell that could
be followed from the open field into the pain session, show its open-field
place verdict next to its pain-session trace. Once the pain video is scored,
the pin-prick / heat columns drop straight into the same table and the
question becomes "was THIS neuron corner-preferring and pin-prick
responsive".

WHAT COUNTS AS THE SAME NEURON
  Two gates, both measured (match_sessions.py):
    position   within one cell radius (8 px) after registering the two
               sessions' mean images. Registration is a direct correlation
               search, not phase correlation - phase correlation gave
               contradictory shifts between the two depths and found no
               pairs at all.
    shape      footprint correlation >= 0.5 after alignment. Position alone
               pairs cells that merely sit near each other, and these planes
               are dense, so this is the gate that decides.

  Measured registration: plane A r = +0.872 at (dy +9, dx -17),
  plane B r = +0.876 at (dy -11, dx -7); each depth matches itself across
  sessions at r ~ 0.87 against 0.33-0.44 for the wrong depth.

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
    m = loadmat(os.path.join(root, f"plane_{plane}", "curated",
                             "curated_dff.mat"))
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
    mp = os.path.join(PA, "match", "match_cells.csv")
    if not os.path.exists(mp):
        raise SystemExit("run match_sessions.py first")
    M = pd.read_csv(mp, dtype={"of_cell": str, "pa_cell": str})
    place_p = os.path.join(OF, "place", "place_cells.csv")
    P = pd.read_csv(place_p, dtype={"cell": str}) if os.path.exists(place_p) \
        else pd.DataFrame()

    rows = []
    for _, r in M.iterrows():
        pl = r["plane"]
        qo, qp = qc(OF, pl), qc(PA, pl)
        pv, pc, pq = "not tested", np.nan, np.nan
        if len(P):
            hit = P[(P["plane"] == pl) & (P["cell"] == r["of_cell"])]
            if len(hit):
                pv = hit.iloc[0]["preference"]
                pc = float(hit.iloc[0]["contrast"])
                pq = float(hit.iloc[0]["q_contrast"])
        rows.append(dict(
            plane=pl, of_cell=r["of_cell"], pa_cell=r["pa_cell"],
            residual_px=r["residual_px"], shape_r=r["shape_r"],
            same_neuron="yes" if r["shape_r"] >= SHAPE_MIN else "position "
                                                                "only",
            of_verdict=qo.get(r["of_cell"], ("?", 0))[0],
            pa_verdict=qp.get(r["pa_cell"], ("?", 0))[0],
            of_usable=qo.get(r["of_cell"], ("?", 0))[1],
            pa_usable=qp.get(r["pa_cell"], ("?", 0))[1],
            of_place=pv, of_zone_contrast=pc, of_zone_q=pq))
    D = pd.DataFrame(rows).sort_values(
        ["plane", "shape_r"], ascending=[True, False])
    outdir = os.path.join(PA, "match")
    D.to_csv(os.path.join(outdir, "linked_cells.csv"), index=False)

    conf = D[D["same_neuron"] == "yes"]
    both = conf[(conf["of_usable"] == 1) & (conf["pa_usable"] == 1)]
    L = ["===== the same neuron in both sessions =====",
         "",
         f"{len(D)} pairs within one cell radius; {len(conf)} also agree in "
         f"footprint shape (r >= {SHAPE_MIN})",
         f"and are treated as the same neuron.",
         "",
         f"Of those {len(conf)}, {len(both)} are usable in BOTH sessions "
         f"(active, single-blob footprint,",
         "not oversized) and so can carry a result from one session to the "
         "other.",
         "",
         f"  {'plane':5s} {'OF':>4s} {'PA':>4s} {'res':>5s} {'shape':>6s} "
         f"{'OF verdict':>11s} {'PA verdict':>11s} {'OF place':>9s} "
         f"{'contrast':>8s} {'q':>7s}"]
    for _, r in D.iterrows():
        cs = "" if not np.isfinite(r["of_zone_contrast"]) \
            else f"{r['of_zone_contrast']:+8.3f}"
        qs = "" if not np.isfinite(r["of_zone_q"]) \
            else f"{r['of_zone_q']:7.4f}"
        L.append(f"  {r['plane']:5s} {r['of_cell']:>4s} {r['pa_cell']:>4s} "
                 f"{r['residual_px']:5.2f} {r['shape_r']:6.3f} "
                 f"{r['of_verdict']:>11s} {r['pa_verdict']:>11s} "
                 f"{r['of_place']:>9s} {cs:>8s} {qs:>7s}")
    place_hits = conf[conf["of_place"].isin(["centre", "corner"])]
    L += ["",
          (f"Zone-selective open-field cells that could be followed into the "
           f"pain session: "
           + (", ".join(f"OF {r['plane']}#{r['of_cell']} ({r['of_place']}) "
                        f"= PA {r['plane']}#{r['pa_cell']}"
                        for _, r in place_hits.iterrows())
              if len(place_hits) else "none")),
          "",
          "Next: once the pain video is scored, add pin-prick and heat "
          "response columns",
          "to linked_cells.csv. The pairing above is what lets a single "
          "neuron carry",
          "both answers.",
          "",
          "Caveat. 'Same neuron' rests on a footprint correlation of 0.5 in "
          "a dense",
          "plane, and the residual positions are 2-7 px against an 8 px "
          "radius. Treat the",
          "pairs as probable, not certain, and check the linked figure before "
          "using one."]
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
        a = dO[r["plane"]][r["of_cell"]]
        b = dP[r["plane"]][r["pa_cell"]]
        ax[i, 0].plot(np.arange(len(a)) / FS, a, lw=.45, color="#B23AA8")
        ax[i, 0].set_xlim(0, len(a) / FS)
        ax[i, 0].set_ylabel(f"{r['plane']}\nOF#{r['of_cell']}", fontsize=9,
                            rotation=0, labelpad=26, va="center")
        ax[i, 1].plot(np.arange(len(b)) / FS, b, lw=.45, color="#1C6E8C")
        ax[i, 1].set_xlim(0, len(b) / FS)
        ax[i, 1].set_ylabel(f"PA#{r['pa_cell']}", fontsize=9, rotation=0,
                            labelpad=22, va="center")
        ax[i, 1].text(.004, .93,
                      f"shape r {r['shape_r']:.2f}, residual "
                      f"{r['residual_px']:.1f} px  |  OF {r['of_verdict']}"
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
    fig.suptitle(f"{n} neurons followed across both sessions "
                 f"(dF/F; footprint shape r >= {SHAPE_MIN})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .99])
    p = os.path.join(outdir, "linked_cells.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
