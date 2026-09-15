"""loosen.py  -  the same tests at every threshold, with the price of each.

WHY
  q <= 0.05 across 28 cells is the right bar for a population claim and
  the wrong one for a single animal whose purpose is to find candidates to
  follow up. Loosening is a legitimate choice - but it has a price, and
  the price is quotable, so it is quoted here rather than left implicit.

  At an UNCORRECTED p <= 0.05 with 28 cells, chance alone gives 28 x 0.05
  = 1.4 "responsive" cells per test. With 8 tests that is ~11 false
  positives across the table. So a loose hit means "worth a second
  animal", not "this neuron responds".

WHAT IS PRODUCED
  A ladder, per event: how many cells pass at
      q <= 0.05    strict, and what the results deck reports
      q <= 0.10    still FDR-controlled, just at 1 in 10
      q <= 0.20    exploratory FDR, common for candidate screens
      p <= 0.05    uncorrected, with the chance count next to it
      p <= 0.01    uncorrected
  and, for the grid, how many of the 15 window x statistic boxes each cell
  passes at each level - because a cell passing 10 boxes at p <= 0.05 is a
  different claim from one passing a single box.

  Nothing is recomputed. Every p-value comes from the tables the strict
  analysis already wrote, so the two cannot disagree.

OUTPUT  ->  <pain session>\\output_split\\events\\
  loosen_ladder.csv , loosen_report.txt , fig24_thresholds.png

USAGE
  python loosen.py                 # the ladder
  python loosen.py --level p05     # also rewrite the maps at that level
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from union_data import ROOTS

PA = ROOTS["pain"]
LEVELS = [("q<=0.05", "q", .05), ("q<=0.10", "q", .10),
          ("q<=0.20", "q", .20), ("p<=0.05", "p", .05),
          ("p<=0.01", "p", .01)]


def ladder(D, name, n_cells):
    """Counts and cell lists at each level, for one test column pair."""
    out = {}
    ok = D["usable"] == 1
    for lab, kind, thr in LEVELS:
        col = f"{name}_{kind}"
        if col not in D:
            continue
        s = D[ok & (D[col] <= thr)]
        up = s[s[f"{name}_resp"] > 0]["uid"].tolist() if \
            f"{name}_resp" in D else []
        dn = s[s[f"{name}_resp"] < 0]["uid"].tolist() if \
            f"{name}_resp" in D else []
        out[lab] = dict(n=len(s), up=up, down=dn,
                        expected=(n_cells * thr if kind == "p" else None))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="q05",
                    choices=["q05", "q10", "q20", "p05", "p01"])
    a = ap.parse_args()
    out = os.path.join(PA, "events")
    D = pd.read_csv(os.path.join(out, "responsive_cells.csv"))
    G = pd.read_csv(os.path.join(out, "search_grid.csv"))
    J = pd.read_csv(os.path.join(out, "search_joint.csv"))
    n_cells = int((D["usable"] == 1).sum())

    evs = [c[:-4] for c in D.columns if c.endswith("_dir")]
    tests = [e for e in evs]
    rows, L = [], []
    L += ["===== the same tests at every threshold =====", "",
          f"{n_cells} interpretable cells. Nothing is recomputed - these "
          f"are the p and q values the",
          "strict analysis already wrote.",
          "",
          "The price of loosening, stated once: at an UNCORRECTED "
          f"p <= 0.05, chance alone gives",
          f"{n_cells} x 0.05 = {n_cells * .05:.1f} cells per test. Across "
          f"{len(tests)} tests that is about {len(tests) * n_cells * .05:.0f} "
          f"false positives in the",
          "table below. A loose hit is a candidate for the next animal, "
          "not a result.",
          "",
          f"  {'test':<18s} " + " ".join(f"{lab:>9s}" for lab, _, _
                                         in LEVELS)]
    for e in tests:
        lad = ladder(D, e, n_cells)
        if not lad:
            continue
        L.append(f"  {e:<18s} "
                 + " ".join(f"{lad[lab]['n']:>9d}" if lab in lad
                            else f"{'-':>9s}" for lab, _, _ in LEVELS))
        for lab, _, _ in LEVELS:
            if lab in lad:
                rows.append(dict(test=e, level=lab, n=lad[lab]["n"],
                                 up=",".join(lad[lab]["up"]),
                                 down=",".join(lad[lab]["down"])))

    L += ["", "  who appears, per level (UP in the first list, DOWN in the "
               "second):"]
    for e in tests:
        lad = ladder(D, e, n_cells)
        if not lad:
            continue
        L.append(f"    {e}")
        for lab, _, _ in LEVELS:
            if lab not in lad:
                continue
            d = lad[lab]
            L.append(f"      {lab:<9s} n={d['n']:<3d} "
                     + (f"UP {', '.join(d['up'])}  " if d["up"] else "")
                     + (f"DOWN {', '.join(d['down'])}" if d["down"] else "")
                     + ("" if d["up"] or d["down"] else "-")
                     + (f"      (chance: {d['expected']:.1f})"
                        if d["expected"] else ""))

    # ---- the grid, which is the more informative place to loosen -----
    L += ["", "  THE GRID at each level: how many of the 15 "
               "window x statistic boxes each cell passes.",
          "  This is the number to read. One box at p <= 0.05 is noise; "
          "ten boxes is not, whatever",
          "  the correction says, because 15 correlated tests of the same "
          "cell do not all pass by luck."]
    for st in G["stimulus"].unique():
        L.append(f"    {st}")
        sub = G[G.stimulus == st]
        for lab, kind, thr in LEVELS:
            tal = {}
            for u in sub["uid"].unique():
                s = sub[sub.uid == u]
                up = int(((s[kind] <= thr) & (s["effect"] > 0)).sum())
                dn = int(((s[kind] <= thr) & (s["effect"] < 0)).sum())
                if up or dn:
                    tal[u] = (up, dn)
            srt = sorted(tal.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
            L.append(f"      {lab:<9s} "
                     + ("; ".join(f"{u} {u_}UP" + (f"/{d_}DN" if d_ else "")
                                  for u, (u_, d_) in srt) if srt
                        else "nothing"))
            for u, (u_, d_) in srt:
                rows.append(dict(test=f"grid:{st}", level=lab, n=u_ + d_,
                                 up=u if u_ else "", down=u if d_ else ""))

    L += ["", "  THE JOINT MODEL at each level (all predictors at once):"]
    for nm in J["predictor"].unique():
        s = J[J.predictor == nm]
        bits = []
        for lab, kind, thr in LEVELS:
            h = s[s[kind] <= thr]["uid"].tolist()
            bits.append(f"{lab} {len(h)}" + (f" ({', '.join(h)})"
                                             if h and len(h) <= 6 else ""))
        L.append(f"    {nm:<16s} " + "   ".join(bits))

    L += ["",
          "RECOMMENDATION.  For this data, read the grid consistency "
          "column, not a single threshold.",
          "A1, B12, B7 and A2 pass many boxes at every level and are the "
          "candidates worth a second",
          "animal. Cells that appear only when the threshold is loosened "
          "to an uncorrected p <= 0.05",
          "and only in one or two boxes are what chance produces at that "
          "level, and the count above",
          "says how many of them to expect."]

    pd.DataFrame(rows).to_csv(os.path.join(out, "loosen_ladder.csv"),
                              index=False)
    txt = "\n".join(L)
    with open(os.path.join(out, "loosen_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    figure(D, G, tests, n_cells, out)


def figure(D, G, tests, n_cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.4,
                         "axes.titlesize": 13})
    show = [e for e in tests if not e.endswith("_in")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.4))

    A = ax[0]
    w = .16
    for k, (lab, kind, thr) in enumerate(LEVELS):
        vals = []
        for e in show:
            col = f"{e}_{kind}"
            vals.append(int(((D["usable"] == 1) & (D[col] <= thr)).sum())
                        if col in D else 0)
        A.bar(np.arange(len(show)) + (k - 2) * w, vals, w,
              label=lab, edgecolor="#1A1A1A", lw=.9)
    A.axhline(n_cells * .05, color="#A63A3A", ls="--", lw=2)
    A.text(len(show) - .5, n_cells * .05 + .12,
           f"chance at p<=0.05  ({n_cells * .05:.1f} cells)", ha="right",
           fontsize=10.5, color="#A63A3A", fontweight="bold")
    A.set_xticks(range(len(show)))
    A.set_xticklabels(show, rotation=28, ha="right")
    A.set_ylabel(f"cells passing (of {n_cells})")
    A.set_title("the +-2 s / 10 s tests, per threshold")
    A.legend(fontsize=10, frameon=False, ncol=2)
    A.spines[["top", "right"]].set_visible(False)

    for j, st in enumerate(G["stimulus"].unique()[:2]):
        A = ax[j + 1]
        sub = G[G.stimulus == st]
        uids = list(sub["uid"].unique())
        for k, (lab, kind, thr) in enumerate(LEVELS):
            cnt = [int((sub[sub.uid == u][kind] <= thr).sum())
                   for u in uids]
            o = np.argsort(cnt)[::-1]
            A.plot(range(len(uids)), [cnt[i] for i in o], "-o", lw=2.2,
                   ms=5, label=lab)
        A.set_xticks(range(len(uids)))
        A.set_xticklabels([uids[i] for i in np.argsort(
            [int((sub[sub.uid == u]["q"] <= .05).sum()) for u in uids]
        )[::-1]], rotation=90, fontsize=8)
        A.set_ylabel("boxes passed (of 15)")
        A.set_xlabel("cells, sorted")
        A.set_title(f"grid consistency - {st}\n15 = every window and "
                    f"statistic")
        A.legend(fontsize=10, frameon=False)
        A.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Loosening the threshold: what it adds and what it costs\n"
                 "the dashed line is how many cells chance alone gives at "
                 "an uncorrected p <= 0.05, so anything near it is not "
                 "evidence", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .89])
    p = os.path.join(out, "fig24_thresholds.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
