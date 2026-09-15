"""pain_final.py  -  THE final pain classification. One answer per cell.

This replaces every earlier pain classification. Those disagreed with each
other (v5 said one thing, v6 another) because they asked questions this
session cannot answer, and each window choice gave a different list. The
audit is recomputed by pain_audit.py -> pain_audit.txt; the
short version:

  - pin pricks come a median 3.2 s apart, so no pre-stimulus window is
    clean (another delivery falls inside [-2,0] s for 28 % of trials,
    [-4,-2] 47 %, [-6,-4] 45 %)
  - the population average PEAKS 1-2 s BEFORE the key press, so any
    pre-window starts the measurement at the top of the response. That
    single fact produced both "pin suppresses the population, p = 0.006"
    and "no pin cells": with a [-6,-4] baseline the same data gives
    excitation, p = 0.0006
  - an event-jitter null cannot be built: shifting events by 3-15 s puts
    31 % of them within 1 s of a real delivery
  - comparing stimulation time against quiet time DOES work, but it is
    not modality-specific - the UP lists for pin, withdrawal and flinch
    overlap at Jaccard 0.60 to 0.90, because those events all live inside
    the same epochs
  - only 12 of 53 pin deliveries have 6 s of clear time before them, and
    with those alone the smallest resolvable effect (0.43 z) is LARGER
    than the largest effect anything in the session shows (0.41 z)

SO THE CLASSIFICATION IS MADE AT THE LEVEL THE DATA SUPPORTS, IN THREE
TIERS, and each tier says what it does and does not mean.

  TIER 1  STIMULATION-MODULATED          robust
          Is the cell different while stimulation is going on than during
          the quiet time? Reference = free frames within +-60 s of each
          delivery, so a session-long drift cancels (the global free mask
          sits 97 s late and covers only 11 % of the dense pin block, and
          would not).
          Does NOT mean pin-specific: handling, arousal and movement are
          all inside "stimulation going on".

  TIER 2  MODALITY-SELECTIVE             WITHDRAWN - not determinable
          "Does the cell prefer pin or heat" looked like the one specific
          test available, because permuting the 53/34 labels over the same
          delivery times holds arousal, handling and movement fixed. It
          does not work on this session: pin and heat were given in
          SEPARATE BLOCKS - pin 13-204 s and 519-586 s, heat 269-518 s,
          with ZERO pin deliveries inside the heat block. So "pin versus
          heat" is also "early versus middle", and a label permutation
          mixes the blocks and cannot undo that. The full-session version
          flagged 14 of 28 cells at effect sizes of 0.05-0.13 z, which is
          what a slow drift looks like; restricted to the overlap window
          there is no overlap to restrict to. The columns are kept in the
          CSV as t2_* and marked invalid.

  TIER 2b BEHAVIOUR-LINKED               weak, n is tiny
          Escape episodes, inside versus outside. 5 episodes, 3.9 s total.
          Reported with its n so it is read as a hint.

  NOT CLAIMED  per-delivery phasic responses to pin or heat; any
          statement of the form "N cells respond to pin"; and pin-versus-
          heat selectivity. The first is set by the window rather than by
          the neurons (0, 6, 7, 8 cells across versions, sharing only A9);
          the last is confounded with time by the block design.

OUTPUT  ->  <pain session>\\output_split\\events\\
  pain_final.csv , pain_final.txt , fig29_pain_final.png

USAGE
  python pain_final.py
"""
from __future__ import annotations

import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import transfer_footprints as TF  # noqa: E402
from cell_atlas import bp  # noqa: E402
from pain_events import bh, cell_seed, shifts, surr_p  # noqa: E402
from pain_responsive import free_mask  # noqa: E402
from tap_delay import frame_to_imaging, load_scoring  # noqa: E402
from union_data import ANAT_MIN, PLANES, ROOTS, load_union  # noqa: E402

PA_SES = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
          "15-56-48")
WIN = (0.0, 10.0)      # stimulation is "going on" for this long
LOCAL = 60.0           # reference drawn from free frames within +-this
N_PERM = 2000
ALPHA = 0.20           # exploratory FDR, as agreed for a single animal
UP, DOWN, NONE, DEAD = "#C1272D", "#1C6E8C", "#C8C8C8", "#6E6E6E"
PIN, HEAT = "#C1272D", "#E08214"
INK, DIM = "#1A1A1A", "#5A5A5A"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.4,
                     "axes.titlesize": 13, "axes.labelsize": 12.5})


def load():
    S = load_scoring(PA_SES)
    tb = frame_to_imaging(PA_SES, "MiceVideo1")
    d_t = tb[np.clip(S["d_frame"], 1, len(tb)) - 1]
    r_t = tb[np.clip(S["reflex"][:, 0], 1, len(tb)) - 1]
    U = load_union()
    t = U[("pain", "A")]["t"]
    free = free_mask(t, np.sort(np.concatenate([d_t, r_t])))
    esc = np.zeros(len(S["score"]), bool)
    for k in range(1, len(S["aff"])):
        if S["aff"][k].lower().startswith("escape"):
            esc = S["score"] == k
    esc_img = np.interp(t, tb[:len(esc)], esc.astype(float)) > .5
    return S, U, t, d_t, free, esc_img


def tier1(x, t, ev, free):
    """Stimulation vs the local quiet time, averaged over deliveries."""
    v = []
    for e in ev:
        b = (t >= e + WIN[0]) & (t <= e + WIN[1])
        nr = free & (t >= e - LOCAL) & (t <= e + LOCAL)
        if b.sum() and nr.sum() >= 15:
            v.append(x[b].mean() - x[nr].mean())
    return float(np.mean(v)) if v else np.nan


def tier2(x, t, times, is_pin):
    """Pin minus heat, over the SAME delivery times."""
    def m(sel):
        w = np.zeros(len(t), bool)
        for e in times[sel]:
            w |= (t >= e + WIN[0]) & (t <= e + WIN[1])
        return x[w].mean() if w.sum() else np.nan
    return m(is_pin) - m(~is_pin)


def main():
    S, U, t, d_t, free, esc = load()
    out = os.path.join(ROOTS["pain"], "events")
    times = np.asarray(d_t, float)
    is_pin = S["d_type"] == 1
    pin, heat = times[is_pin], times[~is_pin]
    print(f"{len(pin)} pin + {len(heat)} heat deliveries; free reference "
          f"{100 * free.mean():.0f} % of the session")

    rows = []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            z, fs = d["z"][i], d["fs"]
            ok = d["anat"][i] >= ANAT_MIN
            r = dict(plane=plane, uid=lab,
                     anat=round(float(d["anat"][i]), 2), usable=int(ok))
            rng = np.random.default_rng(cell_seed(plane, lab, "final"))
            sh = shifts(len(z), fs, rng)

            # TIER 1, one value per stimulus and one pooled
            for nm, ev in (("pin", pin), ("heat", heat),
                           ("any", times)):
                o = tier1(z, t, ev, free)
                p, sd = surr_p(o, lambda y: tier1(y, t, ev, free), z, sh)
                r[f"t1_{nm}"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"t1_{nm}_p"] = round(p, 4) if np.isfinite(p) else np.nan

            # TIER 2 - kept for the record, marked invalid: pin and
            # heat are separate blocks, so this is confounded with time
            o = tier2(z, t, times, is_pin)
            rng2 = np.random.default_rng(cell_seed(plane, lab, "perm"))
            null = np.empty(N_PERM)
            for k in range(N_PERM):
                sel = rng2.permutation(is_pin)
                null[k] = tier2(z, t, times, sel)
            p = (1. + np.sum(np.abs(null - np.nanmean(null))
                             >= abs(o - np.nanmean(null)))) / (N_PERM + 1.)
            r["t2_pin_minus_heat"] = round(float(o), 4)
            r["t2_p"] = round(float(p), 4)

            # TIER 3
            o = (z[esc].mean() - z[~esc].mean()) if esc.sum() > 3 else np.nan
            p, _ = surr_p(o, lambda y: y[esc].mean() - y[~esc].mean(), z, sh)
            r["t3_escape"] = round(o, 4) if np.isfinite(o) else np.nan
            r["t3_p"] = round(p, 4) if np.isfinite(p) else np.nan
            rows.append(r)

    D = pd.DataFrame(rows)
    m = (D["usable"] == 1).to_numpy()
    for c in [c for c in D.columns if c.endswith("_p")]:
        q = np.full(len(D), np.nan)
        q[m] = bh(D.loc[m, c].to_numpy())
        D[c[:-2] + "_q"] = np.round(q, 4)

    def lab_of(r):
        if not r["usable"]:
            return "not interpretable"
        bits = []
        if np.isfinite(r["t1_any_q"]) and r["t1_any_q"] <= ALPHA:
            bits.append("stim-UP" if r["t1_any"] > 0 else "stim-DOWN")
        if np.isfinite(r["t3_q"]) and r["t3_q"] <= ALPHA:
            bits.append("escape-UP" if r["t3_escape"] > 0
                        else "escape-DOWN")
        return ", ".join(bits) if bits else "not modulated"

    D["final"] = [lab_of(r) for _, r in D.iterrows()]
    D.to_csv(os.path.join(out, "pain_final.csv"), index=False)
    report(D, S, pin, heat, esc, out)
    figure(D, U, out)
    return D


def report(D, S, pin, heat, esc, out):
    ok = D["usable"] == 1
    n = int(ok.sum())
    L = ["===== FINAL pain classification =====", "",
         f"{len(D)} cells, {n} with a footprint on a soma in this session.",
         f"Exploratory FDR, q <= {ALPHA:.2f}, across those {n} cells "
         f"within each test.",
         "",
         "This supersedes every earlier pain classification. Those "
         "disagreed with each other because",
         "they asked a question this session cannot answer - see "
         "pain_final.py for the audit. The three",
         "tiers below are what the data does support.",
         ""]
    for tier, col, qcol, desc in (
            ("TIER 1  STIMULATION-MODULATED", "t1_any", "t1_any_q",
             f"activity during the {WIN[1]:.0f} s after any of the "
             f"{len(pin) + len(heat)} deliveries, against the quiet time "
             f"within +-{LOCAL:.0f} s"),
            ("TIER 2b  BEHAVIOUR-LINKED", "t3_escape", "t3_q",
             f"escape episodes inside vs outside - only "
             f"{int(esc.sum())} imaging frames inside, so a hint at best")):
        up = D[ok & (D[qcol] <= ALPHA) & (D[col] > 0)]
        dn = D[ok & (D[qcol] <= ALPHA) & (D[col] < 0)]
        L += [tier, f"  {desc}",
              f"  UP   {len(up):>2d}  "
              + (", ".join(f"{r['uid']} ({r[col]:+.2f})"
                           for _, r in up.iterrows()) if len(up) else "-"),
              f"  DOWN {len(dn):>2d}  "
              + (", ".join(f"{r['uid']} ({r[col]:+.2f})"
                           for _, r in dn.iterrows()) if len(dn) else "-"),
              ""]
    L += ["WITHDRAWN  MODALITY-SELECTIVE (pin versus heat)",
          "  Not determinable on this session. Pin and heat were given in "
          "separate BLOCKS -",
          "  pin 13-204 s and 519-586 s, heat 269-518 s, with ZERO pin "
          "deliveries inside the heat",
          "  block - so 'pin versus heat' is also 'early versus middle'. "
          "The label-permutation",
          "  version flagged 14 of 28 cells at effect sizes of 0.05-0.13 z, "
          "which is what a slow",
          "  drift looks like, and there is no overlap window to restrict "
          "it to. Columns kept in",
          "  the CSV as t2_* and not used.",
          "",
          "  pin and heat separately, TIER 1 - shown to make the point "
          "that they are NOT separable:"]
    for nm in ("pin", "heat"):
        u = D[ok & (D[f"t1_{nm}_q"] <= ALPHA) & (D[f"t1_{nm}"] > 0)]["uid"]
        dd = D[ok & (D[f"t1_{nm}_q"] <= ALPHA) & (D[f"t1_{nm}"] < 0)]["uid"]
        L.append(f"    {nm:<6s} UP {len(u)} {', '.join(u) if len(u) else '-'}"
                 f"   DOWN {len(dd)} {', '.join(dd) if len(dd) else '-'}")
    L += ["    These two lists come from the same epochs and cannot be "
          "attributed to the modality.",
          "    TIER 2 is the test that can. Read TIER 2 for 'pin versus "
          "heat'.",
          "",
          "  every cell:",
          f"  {'cell':>5s} {'anat':>5s} {'T1 any':>7s} {'q':>6s} "
          f"{'T2 pin-heat':>12s} {'q':>6s} {'T3 esc':>7s} {'q':>6s}  final"]
    for _, r in D.iterrows():
        L.append(f"  {r['uid']:>5s} {r['anat']:5.1f} {r['t1_any']:+7.2f} "
                 f"{r['t1_any_q']:6.3f} {r['t2_pin_minus_heat']:+12.2f} "
                 f"{r['t2_q']:6.3f} {r['t3_escape']:+7.2f} "
                 f"{r['t3_q']:6.3f}  {r['final']}")
    L += ["",
          "NOT CLAIMED. The number of cells that 'respond to pin' or "
          "'respond to heat' as individual",
          "deliveries. Every version of that test gave a different list "
          "(0, 6, 7, 8 cells, sharing only",
          "A9 between the two most defensible versions), because with "
          "deliveries 3.2 s apart the answer is",
          "set by the window and not by the neurons. A session with pin "
          "pricks 15-20 s apart would settle",
          "it; this one cannot."]
    txt = "\n".join(L)
    with open(os.path.join(out, "pain_final.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def figure(D, U, out):
    ok = D["usable"] == 1
    sub = D[ok].reset_index(drop=True)
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=.4,
                          wspace=.26)

    # The middle panel used to be "TIER 1, pin deliveries only" with its
    # own significance colouring, and it marked A4 and A11 - while the
    # verdict list on the same figure called A11 not modulated. That is
    # the modality claim sneaking back in through a panel. It is now drawn
    # as the paired pin/heat diagnostic it is, with nothing marked
    # significant, because this test is not used for any verdict.
    for j, (col, qcol, ttl, xl) in enumerate((
            ("t1_any", "t1_any_q", "TIER 1  stimulation-modulated",
             "z during stimulation - z during quiet time"),
            (None, None, "NOT USED - the same test on pin only, and on "
                         "heat only",
             "z during those epochs - z during quiet"),
            ("t3_escape", "t3_q", "TIER 2b  escape-linked",
             "z inside escape - z outside"))):
        A = fig.add_subplot(gs[0, j])
        if col is None:
            o = sub.sort_values("t1_pin").reset_index(drop=True)
            y = np.arange(len(o))
            A.barh(y + .2, o["t1_pin"], height=.38, color=PIN, alpha=.75,
                   edgecolor=INK, lw=.7, label="pin epochs only")
            A.barh(y - .2, o["t1_heat"], height=.38, color=HEAT, alpha=.75,
                   edgecolor=INK, lw=.7, label="heat epochs only")
            A.set_yticks(y)
            A.set_yticklabels(o["uid"], fontsize=9)
            A.axvline(0, color=INK, lw=1.4)
            A.set_xlabel(xl, fontsize=11.5)
            A.set_title(f"{ttl}\nnothing is marked significant here - this "
                        f"test is not used", fontsize=12.5)
            A.legend(fontsize=10.5, frameon=False, loc="lower right")
            A.spines[["top", "right"]].set_visible(False)
            continue
        s = sub.sort_values(col).reset_index(drop=True)
        cols = [UP if (q <= ALPHA and v > 0) else
                DOWN if (q <= ALPHA and v < 0) else NONE
                for q, v in zip(s[qcol], s[col])]
        A.barh(range(len(s)), s[col], color=cols, edgecolor=INK, lw=.9)
        A.set_yticks(range(len(s)))
        A.set_yticklabels(s["uid"], fontsize=9)
        A.axvline(0, color=INK, lw=1.4)
        A.set_xlabel(xl, fontsize=11.5)
        nu = int(((s[qcol] <= ALPHA) & (s[col] > 0)).sum())
        nd = int(((s[qcol] <= ALPHA) & (s[col] < 0)).sum())
        A.set_title(f"{ttl}\n{nu} UP, {nd} DOWN of {len(s)}   "
                    f"(q <= {ALPHA:.2f})", fontsize=12.5)
        A.spines[["top", "right"]].set_visible(False)

    mimg = {p: bp(TF.load(ROOTS["pain"], p)["mean"]) for p in PLANES}
    for j, plane in enumerate(PLANES):
        A = fig.add_subplot(gs[1, j])
        d = U[("pain", plane)]
        A.imshow(mimg[plane], cmap="gray")
        for i, lab in enumerate(d["labels"]):
            row = D[D.uid == lab].iloc[0]
            dead = not row["usable"]
            hit = (np.isfinite(row["t1_any_q"])
                   and row["t1_any_q"] <= ALPHA)
            c = DEAD if dead else (
                (UP if row["t1_any"] > 0 else DOWN) if hit else NONE)
            m = (d["S"][:, i].reshape(d["shape"])
                 > .2 * d["S"][:, i].max()).astype(np.uint8)
            for cc in cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)[0]:
                A.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                       np.append(cc[:, 0, 1], cc[0, 0, 1]), color=c,
                       lw=3.0 if hit else 1.6)
            if hit:
                ys, xs = np.nonzero(m)
                A.text(xs.mean(), ys.min() - 9, lab, color="white",
                       fontsize=10.5, fontweight="bold", ha="center",
                       va="bottom",
                       path_effects=[pe.withStroke(linewidth=3.2,
                                                   foreground=c)])
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(f"plane {plane} - TIER 1 map", fontsize=12.5)

    A = fig.add_subplot(gs[1, 2])
    A.axis("off")
    lines = [f"{'cell':>5s}  final classification"]
    for _, r in sub.iterrows():
        if r["final"] != "not modulated":
            lines.append(f"{r['uid']:>5s}  {r['final']}")
    nn = int((sub["final"] == "not modulated").sum())
    lines += ["", f"not modulated: {nn} of {len(sub)}"]
    A.text(0, 1, "\n".join(lines), family="Consolas", fontsize=11,
           va="top", color=INK)

    fig.suptitle(
        "FINAL pain classification - each tier with what it does and does "
        "not mean.  TIER 1 is robust but NOT modality-specific (the pin, "
        "withdrawal and flinch lists overlap at Jaccard 0.6-0.9).\n"
        "MODALITY-SELECTIVITY IS WITHDRAWN: pin was given 13-204 s and "
        "519-586 s, heat 269-518 s, with zero pin inside the heat block, "
        "so 'pin versus heat' is also 'early versus middle' and no "
        "permutation undoes that.\n"
        "Per-delivery 'responds to pin' is NOT claimed either: every "
        "window choice gave a different list (0, 6, 7, 8 cells, sharing "
        "only A9 between the two best).", fontsize=13)
    fig.subplots_adjust(left=.05, right=.985, top=.865, bottom=.05)
    p = os.path.join(out, "fig29_pain_final.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


def redraw():
    """Report and figure from the saved CSV - no statistic is recomputed."""
    out = os.path.join(ROOTS["pain"], "events")
    D = pd.read_csv(os.path.join(out, "pain_final.csv"))
    S, U, t, d_t, free, esc = load()
    times = np.asarray(d_t, float)
    is_pin = S["d_type"] == 1
    report(D, S, times[is_pin], times[~is_pin], esc, out)
    figure(D, U, out)


if __name__ == "__main__":
    import sys
    redraw() if "--redraw" in sys.argv else main()
