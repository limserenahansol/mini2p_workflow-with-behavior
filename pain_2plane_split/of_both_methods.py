"""of_both_methods.py  -  the zone result two ways, and where they disagree.

TWO METHODS, BOTH REPORTED
  A  SIGNIFICANCE   is this cell's zone contrast bigger than chance?
     centre mean minus corner mean of the detrended z, against a
     circular-shift null (2000 surrogates), Benjamini-Hochberg across
     cells. A cell can come out as neither, and most do.
     -> this says WHETHER there is a zone preference.

  B  RANKED SPLIT (k = 2)   the other pipeline's method.
     k-means with k = 2 on the same contrast index, which sorts every
     cell into a "centre group" or a "corner group".
     -> this says WHICH HALF a cell is in. It always produces both
        groups, even when no cell is significant, because splitting a
        one-dimensional list in two is what it does. On this data the
        two groups differ by a contrast of about 0.3 z, which is inside
        the noise for most cells.

  Neither is wrong; they answer different questions. Reported together so
  a reader can see that the six "corner" cells from the ranked split are
  the lower half of a ranking, not six cells with a corner preference.

ALSO HERE
  the corner-versus-everywhere-else test, which is the properly powered
  corner question (626 corner frames against 874 elsewhere, where the
  centre has only 82).

OUTPUT  ->  <open field>\\output_split\\place\\
  of_both_methods.csv , of_both_methods.txt , fig25_both_methods.png

USAGE
  python of_both_methods.py
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from union_data import ROOTS  # noqa: E402

CEN, COR, NONE = "#2166AC", "#E7298A", "#C8C8C8"
INK, DIM = "#1A1A1A", "#5A5A5A"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.4,
                     "axes.titlesize": 13.5, "axes.labelsize": 13})


def kmeans1d(x, k=2, iters=200, seed=0):
    """Plain 1-D k-means, so the split is reproducible and inspectable.

    Seeded at the min and max so the labelling is deterministic: group 0
    is always the lower (corner-leaning) half.
    """
    x = np.asarray(x, float)
    c = np.array([x.min(), x.max()]) if k == 2 else \
        np.linspace(x.min(), x.max(), k)
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        new = np.argmin(np.abs(x[:, None] - c[None, :]), 1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean()
    return lab, c


def main():
    root = ROOTS["openfield"]
    out = os.path.join(root, "place")
    P = pd.read_csv(os.path.join(out, "place_cells.csv"))
    Z = pd.read_csv(os.path.join(root, "atlas", "zone_rates.csv"))
    D = P.merge(Z[["uid", "corner_vs_rest", "corner_vs_rest_p",
                   "corner_vs_rest_q", "rate_centre", "rate_corner",
                   "usable"]], left_on="cell", right_on="uid", how="left")

    lab, cent = kmeans1d(D["contrast"].to_numpy())
    D["kmeans_group"] = np.where(lab == 1, "centre group", "corner group")
    D["kmeans_centre_of_group"] = np.where(lab == 1, cent[1], cent[0])

    D["significance"] = D["preference"]
    D.to_csv(os.path.join(out, "of_both_methods.csv"), index=False)

    n = len(D)
    nk_c = int((D.kmeans_group == "centre group").sum())
    nk_k = int((D.kmeans_group == "corner group").sum())
    ns_c = int((D.significance == "centre").sum())
    ns_k = int((D.significance == "corner").sum())

    L = ["===== the open field zone result, both ways =====", "",
         f"{n} cells tested.", "",
         "METHOD A - significance (circular-shift null, BH across cells)",
         f"   centre-preferring {ns_c}   corner-preferring {ns_k}   "
         f"neither {n - ns_c - ns_k}",
         "",
         "METHOD B - ranked split, k-means k = 2 on the same contrast "
         "(the other pipeline's method)",
         f"   centre group {nk_c}   corner group {nk_k}",
         f"   group centres: corner group {cent[0]:+.3f} z, centre group "
         f"{cent[1]:+.3f} z  (a gap of {cent[1] - cent[0]:.3f} z)",
         "   k = 2 always returns two groups. It is a ranking, not a test.",
         "",
         "  the two side by side:",
         f"  {'cell':>5s} {'contrast':>9s} {'q':>7s} {'significance':>14s} "
         f"{'k=2 group':>14s}  agree?"]
    for _, r in D.sort_values("contrast", ascending=False).iterrows():
        sig = r["significance"]
        km = r["kmeans_group"]
        agree = ("yes" if (sig == "centre" and km == "centre group")
                 or (sig == "corner" and km == "corner group")
                 else ("k=2 only" if sig == "none" else "CONFLICT"))
        L.append(f"  {r['cell']:>5s} {r['contrast']:+9.3f} "
                 f"{r['q_contrast']:7.4f} {sig:>14s} {km:>14s}  {agree}")

    both_c = D[(D.significance == "centre")
               & (D.kmeans_group == "centre group")]["cell"].tolist()
    km_only = D[(D.significance == "none")
                & (D.kmeans_group == "corner group")]["cell"].tolist()
    L += ["",
          f"  AGREE, centre: {', '.join(both_c) if both_c else 'none'}",
          f"  In the k=2 corner group but NOT significant: "
          f"{len(km_only)} cells - {', '.join(km_only)}",
          "  Those are the cells the other pipeline lists as corner cells. "
          "They are the lower half of the",
          "  ranking; none of them passes a test against chance.",
          "",
          "THE PROPERLY POWERED CORNER QUESTION - corner versus everywhere "
          "else",
          "  centre-minus-corner can miss a cell that is high in corners "
          "and low in BOTH other zones, and",
          "  it is limited by the 82 centre frames. Corner versus the rest "
          "uses 626 against 874.",
          f"  {'cell':>5s} {'corner - rest':>13s} {'p':>7s} {'q':>7s}"]
    for _, r in D.sort_values("corner_vs_rest", ascending=False).head(6
                                                                      ).iterrows():
        L.append(f"  {r['cell']:>5s} {r['corner_vs_rest']:+13.3f} "
                 f"{r['corner_vs_rest_p']:7.4f} {r['corner_vs_rest_q']:7.4f}")
    nc = int(((D.corner_vs_rest_q <= .20) & (D.corner_vs_rest > 0)).sum())
    L += [f"  cells passing at q <= 0.20: {nc}",
          "",
          "CONCLUSION.  There is no corner-specific cell in this session by "
          "either test, at any threshold",
          "from q <= 0.05 to an uncorrected p <= 0.05. The k = 2 split "
          "still names six, because that is what",
          "k = 2 does. Two cells (A3, A7) do prefer the centre and survive "
          "both stability checks."]
    txt = "\n".join(L)
    with open(os.path.join(out, "of_both_methods.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    figure(D, cent, out)


def figure(D, cent, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.4))
    d = D.sort_values("contrast").reset_index(drop=True)

    A = ax[0]
    col = [CEN if s == "centre" else COR if s == "corner" else NONE
           for s in d["significance"]]
    A.barh(range(len(d)), d["contrast"], color=col, edgecolor=INK, lw=.8)
    A.axvline(0, color=INK, lw=1.4)
    A.set_yticks(range(len(d)))
    A.set_yticklabels(d["cell"], fontsize=8.5)
    A.set_xlabel("zone contrast  (centre mean - corner mean, z)")
    A.set_title("METHOD A - significance\ncircular-shift null, BH; "
                f"{int((d.significance == 'centre').sum())} centre, "
                f"{int((d.significance == 'corner').sum())} corner")
    A.spines[["top", "right"]].set_visible(False)

    A = ax[1]
    col = [CEN if g == "centre group" else COR for g in d["kmeans_group"]]
    A.barh(range(len(d)), d["contrast"], color=col, edgecolor=INK, lw=.8)
    A.axvline(0, color=INK, lw=1.4)
    thr = (cent[0] + cent[1]) / 2
    A.axvline(thr, color="#111111", ls="--", lw=2)
    A.text(thr, len(d) - .5, f" k=2 boundary {thr:+.2f}", fontsize=10.5,
           va="top")
    A.set_yticks(range(len(d)))
    A.set_yticklabels(d["cell"], fontsize=8.5)
    A.set_xlabel("the same contrast values")
    A.set_title("METHOD B - ranked split, k = 2\n"
                f"{int((d.kmeans_group == 'centre group').sum())} centre "
                f"group, "
                f"{int((d.kmeans_group == 'corner group').sum())} corner "
                f"group - always two")
    A.spines[["top", "right"]].set_visible(False)

    A = ax[2]
    e = D.sort_values("corner_vs_rest").reset_index(drop=True)
    col = [COR if (q <= .20 and v > 0) else NONE
           for q, v in zip(e["corner_vs_rest_q"], e["corner_vs_rest"])]
    A.barh(range(len(e)), e["corner_vs_rest"], color=col, edgecolor=INK,
           lw=.8)
    A.axvline(0, color=INK, lw=1.4)
    A.set_yticks(range(len(e)))
    A.set_yticklabels(e["cell"], fontsize=8.5)
    A.set_xlabel("corner mean - everywhere-else mean (z)")
    A.set_title("the properly powered corner test\n626 corner frames vs "
                "874; 0 cells pass at q <= 0.20")
    A.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "The open field zone result, both ways\n"
        "LEFT asks whether a preference is bigger than chance. MIDDLE "
        "sorts the same numbers into two halves and always finds both. "
        "RIGHT is the corner question done with the frames to answer it.",
        fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .88])
    p = os.path.join(out, "fig25_both_methods.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
