"""summary_figures.py  -  the summary set, in the layout Hansol asked for.

fig26  AUC pre versus post, per event, in the same three-panel layout as
       the other pipeline's slides so the two can be put side by side:
         left    population AUC, pre against post, with the paired p
         right   population peri-event mean +- SEM
         bottom  per-cell AUC, pre grey and post red, * = per-cell p < 0.05
       Built twice, once on z and once on dF/F, because "did it go up" can
       depend on which signal is used and that should be visible rather
       than chosen.

fig27  the cross-session rows WITH the field of view: every cell that is
       either pain-responsive or fires faster in the centre, as
         field of view | open-field trace with the mouse's zone |
         pain trace with the scored events
       A3 and A7 are in here even though they are not pain-responsive -
       they are the centre cells, and the point of the figure is to show
       what those two do in the pain session.

fig28  the cells that answer to more than one thing, one panel per event
       they answer to, large enough to read. A1 is the case in point:
       heat, paw withdrawal, pin epoch and heat epoch.

OUTPUT  ->  <pain session>\\output_split\\results\\

USAGE
  python summary_figures.py               # q <= 0.20, the deck's criterion
  python summary_figures.py --level q05
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import transfer_footprints as TF  # noqa: E402
from cell_atlas import EVCOL, bp, pain_events  # noqa: E402
from results_figures import LEVELS, dirs_of  # noqa: E402
from union_data import ANAT_MIN, OF_SESSION, PLANES, ROOTS, load_union  # noqa: E402

SUF = ""     # filename suffix for a non-default verdict set
PA_SES = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
          "15-56-48")
UP, DOWN, NONE, DEAD = "#C1272D", "#1C6E8C", "#C8C8C8", "#6E6E6E"
CEN, COR, EDG = "#2166AC", "#E7298A", "#B0B0B0"
INK, DIM = "#1A1A1A", "#5A5A5A"
PRE, POST = (-2.0, 0.0), (0.0, 2.0)
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.4,
                     "axes.titlesize": 13, "axes.labelsize": 12.5,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4})


def auc_trials(x, t, ev, pre=PRE, post=POST):
    """Trial-baselined area under the curve, per trial, pre and post.

    trapz of (x - mean over the pre window) over each window, in z*s. No
    division, so it is an area and scales with the window length - the
    two windows here are the same length, which is what makes them
    comparable.
    """
    a_all, b_all = [], []
    for e in ev:
        ia = (t >= e + pre[0]) & (t < e + pre[1])
        ib = (t >= e + post[0]) & (t <= e + post[1])
        if ia.sum() < 2 or ib.sum() < 2:
            continue
        base = x[ia].mean()
        a_all.append(np.trapezoid(x[ia] - base, t[ia]))
        b_all.append(np.trapezoid(x[ib] - base, t[ib]))
    return np.array(a_all), np.array(b_all)


def fig_auc(U, EV, g, out, signal="z", level="q20"):
    c = []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            if d["anat"][i] >= ANAT_MIN:
                c.append((lab, d[signal][i], d["t"]))
    labs = [x[0] for x in c]
    evn = [e for e in EV]
    n = len(evn)
    fig = plt.figure(figsize=(5.4 * n, 11))
    gs = fig.add_gridspec(2, n, height_ratios=[1, .95], hspace=.42,
                          wspace=.3)
    lags = np.arange(-5, 5.01, .1)
    for j, ev in enumerate(evn):
        E = EV[ev]
        col = EVCOL.get(ev, INK)
        # per cell: mean AUC over trials, pre and post; and a per-cell p
        pre_c, post_c, pcell = [], [], []
        for lab, x, t in c:
            a, b = auc_trials(x, t, E)
            pre_c.append(a.mean())
            post_c.append(b.mean())
            try:
                pcell.append(wilcoxon(a, b)[1] if len(a) > 5 else np.nan)
            except ValueError:
                pcell.append(np.nan)
        pre_c, post_c = np.array(pre_c), np.array(post_c)
        try:
            _, p_pop = wilcoxon(pre_c, post_c)
        except ValueError:
            p_pop = np.nan

        A = fig.add_subplot(gs[0, j])
        A.bar([0], [pre_c.mean()], .55, color="#9E9E9E", edgecolor=INK,
              lw=1.4, label="pre")
        A.bar([1], [post_c.mean()], .55, color=col, edgecolor=INK, lw=1.4,
              label="post")
        for k, v in ((0, pre_c), (1, post_c)):
            A.errorbar([k], [v.mean()], yerr=[v.std() / np.sqrt(len(v))],
                       color=INK, lw=2.2, capsize=8, capthick=2.2)
        A.axhline(0, color=INK, lw=1.2)
        A.set_xticks([0, 1])
        A.set_xticklabels([f"pre [{PRE[0]:.0f},{PRE[1]:.0f}] s",
                           f"post [{POST[0]:.0f},{POST[1]:.0f}] s"])
        A.set_ylabel(f"AUC of {signal}  (z*s)" if signal == "z"
                     else "AUC of dF/F  (dF/F * s)")
        sig = "n.s." if not np.isfinite(p_pop) or p_pop > .05 else (
            "*" if p_pop > .01 else "**")
        A.set_title(f"{ev}  ({len(E)} events)\n"
                    f"signed-rank p = {p_pop:.3f}  {sig}", color=col,
                    fontsize=12.5)
        A.legend(fontsize=10, frameon=False)
        A.spines[["top", "right"]].set_visible(False)

        B = fig.add_subplot(gs[1, j])
        Y = []
        for lab, x, t in c:
            M = np.stack([np.interp(e + lags, t, x, left=np.nan,
                                    right=np.nan) for e in E])
            b0 = np.nanmean(M[:, (lags >= PRE[0]) & (lags < PRE[1])], 1,
                            keepdims=True)
            Y.append(np.nanmean(M - b0, 0))
        Y = np.stack(Y)
        m, se = np.nanmean(Y, 0), np.nanstd(Y, 0) / np.sqrt(len(Y))
        B.fill_between(lags, m - se, m + se, color=col, alpha=.25, lw=0)
        B.plot(lags, m, color=col, lw=2.6)
        B.axvline(0, color=INK, lw=1.6, ls="--")
        B.axhline(0, color=DIM, lw=1)
        B.axvspan(POST[0], POST[1], color="#999999", alpha=.16, lw=0)
        B.set_xlabel("time from event (s)")
        B.set_ylabel(f"{signal}, baselined")
        B.set_title(f"mean +- SEM, {len(labs)} cells",
                    fontsize=12)
        B.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"AUC pre versus post, per event - signal = "
        f"{'z (detrended dF/F / robust SD)' if signal == 'z' else 'dF/F'}"
        f"\nAUC = trapz of the trial-baselined signal over each window "
        f"(baseline = mean of [{PRE[0]:.0f},{PRE[1]:.0f}] s, no division). "
        f"Population p = Wilcoxon signed-rank across cells of the "
        f"trial-averaged AUC.", fontsize=14)
    fig.subplots_adjust(left=.05, right=.985, top=.855,
                        bottom=.06, hspace=.42, wspace=.34)
    p = os.path.join(out, f"fig26_auc_{signal}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p, {ev: None for ev in evn}


def fig_auc_percell(U, EV, out, signal="z"):
    """The per-cell AUC bars, the bottom panel of the attached layout."""
    c = []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            if d["anat"][i] >= ANAT_MIN:
                c.append((lab, d[signal][i], d["t"]))
    labs = [x[0] for x in c]
    evn = list(EV)
    fig, axes = plt.subplots(len(evn), 1, figsize=(16, 3.1 * len(evn)),
                             squeeze=False)
    for j, ev in enumerate(evn):
        A = axes[j, 0]
        col = EVCOL.get(ev, INK)
        pre_c, post_c, star = [], [], []
        for lab, x, t in c:
            a, b = auc_trials(x, t, EV[ev])
            pre_c.append(a.mean())
            post_c.append(b.mean())
            try:
                star.append(wilcoxon(a, b)[1] < .05 if len(a) > 5 else False)
            except ValueError:
                star.append(False)
        xi = np.arange(len(labs))
        A.bar(xi - .2, pre_c, .4, color="#9E9E9E", edgecolor=INK, lw=1,
              label="pre")
        A.bar(xi + .2, post_c, .4, color=col, edgecolor=INK, lw=1,
              label="post")
        top = max(np.max(np.abs(pre_c)), np.max(np.abs(post_c)))
        for k, s in enumerate(star):
            if s:
                A.text(k, top * 1.06, "*", ha="center", fontsize=17,
                       fontweight="bold")
        A.axhline(0, color=INK, lw=1.2)
        A.set_xticks(xi)
        A.set_xticklabels(labs, fontsize=9.5, rotation=90)
        A.set_ylabel(f"AUC of {signal}")
        A.set_title(f"{ev}   ({len(EV[ev])} events)   "
                    f"* = per-cell Wilcoxon across trials, p < 0.05 "
                    f"({sum(star)} cells)", color=col)
        A.legend(fontsize=10, frameon=False, ncol=2)
        A.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Per-cell AUC, pre and post, signal = {signal}\n"
                 f"no multiple-comparison correction on the stars - they "
                 f"mark a raw p < 0.05, and with {len(labs)} cells chance "
                 f"gives about {len(labs) * .05:.1f} per event",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .96])
    p = os.path.join(out, f"fig26b_auc_percell_{signal}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fov_panel(ax, U, ses, plane, uid, colour):
    d = U[(ses, plane)]
    j = d["labels"].index(uid)
    S = d["S"][:, j].reshape(d["shape"])
    m = (S > .2 * S.max()).astype(np.uint8)
    yy, xx = np.nonzero(m)
    cy, cx = int(yy.mean()), int(xx.mean())
    h, w = d["shape"]
    y0, y1 = max(cy - 62, 0), min(cy + 62, h)
    x0, x1 = max(cx - 62, 0), min(cx + 62, w)
    ax.imshow(bp(TF.load(ROOTS[ses], plane)["mean"])[y0:y1, x0:x1],
              cmap="gray")
    # outline only, no fill, and the name in the CORNER - a label sitting
    # on the soma hides the very shape the panel exists to show
    for cc in cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0]:
        ax.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                np.append(cc[:, 0, 1], cc[0, 0, 1]), color=colour, lw=2.8)
    ax.text(.04, .96, uid, transform=ax.transAxes, color="white",
            fontsize=13, fontweight="bold", ha="left", va="top",
            path_effects=[pe.withStroke(linewidth=3.6, foreground=colour)])
    ax.set_xticks([])
    ax.set_yticks([])


def fig_cross(U, EV, g, Z, out):
    """FOV + open-field trace + pain trace, for the cells that matter."""
    from openfield_place_cells import load_tracking
    tr = load_tracking(OF_SESSION)
    tof = U[("openfield", "A")]["t"]
    ok = tr["ok"].to_numpy() == 1
    zo = np.round(np.interp(tof, tr["t_s"].to_numpy()[ok],
                            tr["zone"].to_numpy()[ok].astype(float))
                  ).astype(int)
    tpa = U[("pain", "A")]["t"]
    faster = set(Z[Z["faster"]]["uid"])
    resp = [u for u, d in g.items() if d]
    order = ([u for u in faster] +
             [u for u in resp if u not in faster])
    n = len(order)
    H = 2.7 * n + 1.6
    fig = plt.figure(figsize=(19, H))
    gs = fig.add_gridspec(n, 6, width_ratios=[1, 1, 2.6, 2.6, 2.6, 2.6],
                          wspace=.24, hspace=.55)
    for i, uid in enumerate(order):
        plane = "A" if uid.startswith("A") else "B"
        dp, do = U[("pain", plane)], U[("openfield", plane)]
        k = dp["labels"].index(uid)
        is_cen = uid in faster
        col = CEN if is_cen else (UP if "UP" in g.get(uid, {}).values()
                                  else DOWN)
        fov_panel(fig.add_subplot(gs[i, 0]), U, "openfield", plane, uid,
                  col)
        fov_panel(fig.add_subplot(gs[i, 1]), U, "pain", plane, uid, col)

        A = fig.add_subplot(gs[i, 2:4])
        A.plot(tof, do["z"][k], lw=.8, color=INK)
        lo = do["z"][k].min() - .22 * np.ptp(do["z"][k])
        for v, cc in ((0, CEN), (1, COR), (2, EDG)):
            mm = zo == v
            if mm.any():
                A.fill_between(tof, lo, lo + .10 * np.ptp(do["z"][k]),
                               where=mm, color=cc, lw=0, step="mid")
        A.set_xlim(tof[0], tof[-1])
        A.set_ylim(lo, do["z"][k].max() * 1.05)
        A.set_ylabel("z", fontsize=11)
        zr = Z[Z.uid == uid]
        ztxt = (f"centre {zr.iloc[0]['rate_centre']:.2f}/s vs corner "
                f"{zr.iloc[0]['rate_corner']:.2f}/s, contrast "
                f"{zr.iloc[0]['contrast']:+.2f} z, q="
                f"{zr.iloc[0]['q']:.3f}" if len(zr) else "not tested")
        A.set_title(f"{uid}  OPEN FIELD - {ztxt}", fontsize=11.5,
                    fontweight="bold", loc="left", color=col)
        A.spines[["top", "right"]].set_visible(False)

        B = fig.add_subplot(gs[i, 4:])
        B.plot(tpa, dp["z"][k], lw=.8, color=INK)
        lo = dp["z"][k].min() - .30 * np.ptp(dp["z"][k])
        st = .075 * np.ptp(dp["z"][k])
        for j2, (nm, ev) in enumerate(EV.items()):
            B.plot(ev, np.full(len(ev), lo + j2 * st), "|", ms=7, mew=1.6,
                   color=EVCOL.get(nm, INK))
            if i == 0:
                B.text(tpa[-1] * 1.004, lo + j2 * st, nm, fontsize=8.5,
                       va="center", color=EVCOL.get(nm, INK),
                       fontweight="bold")
        B.set_xlim(tpa[0], tpa[-1])
        B.set_ylabel("z", fontsize=11)
        v = g.get(uid, {})
        B.set_title(f"{uid}  PAIN - "
                    + (", ".join(f"{e} {x}" for e, x in v.items())
                       if v else "no response"),
                    fontsize=11.5, fontweight="bold", loc="left", color=col)
        B.spines[["top", "right"]].set_visible(False)
        if i == n - 1:
            A.set_xlabel("time (s)   |   bar = zone: blue centre, pink "
                         "corner, grey edge")
            B.set_xlabel("time (s)   |   ticks = scored events")
    fig.suptitle("Field of view, open field and pain, for every cell that "
                 "matters\nthe two centre cells first (blue), then every "
                 "pain-responsive cell (red = up to something, blue = down)",
                 fontsize=15, y=1 - .3 / H)
    fig.subplots_adjust(left=.03, right=.955, top=1 - 1.1 / H,
                        bottom=.55 / H)
    p = os.path.join(out, f"fig27_cross_cells{SUF}.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_multi(U, EV, g, out):
    """Cells answering to more than one thing, one panel per answer."""
    multi = {u: d for u, d in g.items() if len(d) > 1}
    if not multi:
        return None
    lags = np.arange(-5, 8.01, .1)
    lag2 = np.arange(-6, 20.01, .2)
    order = sorted(multi, key=lambda u: -len(multi[u]))
    ncol = max(len(d) for d in multi.values())
    n = len(order)
    fig, axes = plt.subplots(n, ncol + 1, figsize=(3.6 * (ncol + 1),
                                                   3.2 * n), squeeze=False)
    free = {}
    for b_ in ("pin", "heat"):
        m = np.ones(len(U[("pain", "A")]["t"]), bool)
        for e in EV[b_]:
            m &= ~((U[("pain", "A")]["t"] >= e)
                   & (U[("pain", "A")]["t"] <= e + 10.0))
        free[b_] = m
    for i, uid in enumerate(order):
        plane = "A" if uid.startswith("A") else "B"
        d = U[("pain", plane)]
        k = d["labels"].index(uid)
        z, t = d["z"][k], d["t"]
        col = UP if "UP" in multi[uid].values() else DOWN
        fov_panel(axes[i, 0], U, "pain", plane, uid, col)
        axes[i, 0].set_title(f"{uid}\n{len(multi[uid])} responses",
                             fontsize=12, fontweight="bold", color=col)
        for j, (ev, dirv) in enumerate(multi[uid].items()):
            A = axes[i, j + 1]
            cj = EVCOL.get(ev.replace("_epoch", "").replace("_in", ""), INK)
            if ev.endswith("_epoch"):
                base = ev[:-6]
                M = np.stack([np.interp(e + lag2, t, z, left=np.nan,
                                        right=np.nan) for e in EV[base]])
                M = M - z[free[base]].mean()
                lg, win = lag2, (0, 10)
            else:
                base = ev.replace("_in", "")
                src = EV.get(base, EV.get(ev))
                M = np.stack([np.interp(e + lags, t, z, left=np.nan,
                                        right=np.nan) for e in src])
                b0 = np.nanmean(M[:, (lags >= PRE[0]) & (lags < PRE[1])],
                                1, keepdims=True)
                M = M - b0
                lg, win = lags, (0, 2)
            m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(len(M))
            A.fill_between(lg, m - se, m + se, color=cj, alpha=.25, lw=0)
            A.plot(lg, m, color=cj, lw=2.4)
            A.axvline(0, color=INK, lw=1.5)
            A.axhline(0, color=DIM, lw=1)
            A.axvspan(*win, color="#999999", alpha=.16, lw=0)
            A.set_title(f"{ev}  {dirv}  ({len(M)} events)",
                        fontsize=11.5, color=cj, fontweight="bold")
            A.set_xlabel("s from event", fontsize=10.5)
            if j == 0:
                A.set_ylabel("z, baselined", fontsize=11)
            A.spines[["top", "right"]].set_visible(False)
        for j in range(len(multi[uid]), ncol):
            axes[i, j + 1].axis("off")
    fig.suptitle("Cells that answer to more than one thing\n"
                 "one panel per response, mean +- SEM over events; grey "
                 "band = the window that test used", fontsize=14.5)
    fig.tight_layout(rect=[0, 0, 1, .95])
    p = os.path.join(out, "fig28_multi_response.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="q20", choices=list(LEVELS))
    a = ap.parse_args()
    U = load_union()
    root = ROOTS["pain"]
    out = os.path.join(root, "results")
    os.makedirs(out, exist_ok=True)
    D = pd.read_csv(os.path.join(root, "events", "responsive_cells.csv"))
    EV = pain_events(PA_SES)
    global SUF
    if a.level == "final":
        # fig27 on the FINAL verdicts. The AUC figures are the withdrawn
        # set and stay on the per-event test, which is the version they
        # were built to demonstrate.
        from results_figures import final_dirs
        _, g = final_dirs(root)
        SUF = "_final"
        Z = pd.read_csv(os.path.join(ROOTS["openfield"], "atlas",
                                     "zone_rates.csv"))
        print(f"summary figures, FINAL classification")
        print(f"  {fig_cross(U, EV, g, Z, out)}")
        return
    _, g = dirs_of(D, a.level)
    Z = pd.read_csv(os.path.join(ROOTS["openfield"], "atlas",
                                 "zone_rates.csv"))
    print(f"summary figures, {LEVELS[a.level][2]}")
    for sig in ("z", "dff"):
        p, _ = fig_auc(U, EV, g, out, sig, a.level)
        print(f"  {p}")
        print(f"  {fig_auc_percell(U, EV, out, sig)}")
    print(f"  {fig_cross(U, EV, g, Z, out)}")
    m = fig_multi(U, EV, g, out)
    if m:
        print(f"  {m}")


if __name__ == "__main__":
    main()
