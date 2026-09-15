"""results_figures.py  -  the result figures for the results deck.

Built to three rules, because the earlier figures broke all three:
  1  readable - thick lines, large labels, nothing thinner than 1.5 pt
  2  every panel states its statistic, its n and its p or q
  3  every axis has units, every colour map has a bar with its range

WHAT IS HERE
  fig19  before the first stimulus versus after the last one. 30 s of
         baseline against 30 s at the end, per cell, paired.
  fig20  where the responsive cells are - one field-of-view map per event
         type, up in red and down in blue
  fig21  group averages per event. EVERY panel is z minus the
         stimulus-free frames within +-60 s of that event, and the groups
         come from pain_final.csv - two different references in one figure
         is what made the earlier version contradict its own verdict.
  fig22  population summary - counts per event, and the overlap between
         events for cells that respond to more than one
  fig23  cross-session - the pain-responsive cells and what they did in the
         open field

Everything reads a table: pain_final.csv for the pain verdicts,
responsive_cells.csv and search_grid.csv for the withdrawn per-delivery
version, place_cells.csv for the open field. No statistic is recomputed
here, so a figure cannot disagree with its table.

  --level final   draw the pain figures from pain_final.csv: two tags,
                  stimulation and escape, and no per-delivery verdict.

OUTPUT  ->  <pain session>\\output_split\\results\\

USAGE
  python results_figures.py
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
from scipy.stats import wilcoxon  # noqa: E402

import transfer_footprints as TF  # noqa: E402
from cell_atlas import EVCOL, bp, pain_events  # noqa: E402
from pain_events import bh, cell_seed, shifts, surr_p  # noqa: E402
from union_data import ANAT_MIN, PLANES, ROOTS, load_union  # noqa: E402

PA_SES = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
          "15-56-48")
UP, DOWN, NONE, DEAD = "#C1272D", "#1C6E8C", "#C8C8C8", "#6E6E6E"
CEN, COR = "#2166AC", "#E7298A"
INK, DIM = "#1A1A1A", "#5A5A5A"
WIN_S = 30.0
ALPHA = 0.05
LEVELS = {"q05": ("q", .05, "q <= 0.05, FDR across cells"),
          "q10": ("q", .10, "q <= 0.10, FDR across cells"),
          "q20": ("q", .20, "q <= 0.20, exploratory FDR"),
          "p05": ("p", .05, "p <= 0.05, UNCORRECTED"),
          "p01": ("p", .01, "p <= 0.01, UNCORRECTED"),
          "final": ("q", .20, "FINAL classification, pain_final.csv")}
SUF = ""
LEVTXT = LEVELS["q05"][2]
plt.rcParams.update({"axes.linewidth": 1.4, "xtick.major.width": 1.4,
                     "ytick.major.width": 1.4, "font.size": 12,
                     "axes.labelsize": 13, "axes.titlesize": 13.5})


def load_all():
    U = load_union()
    root = ROOTS["pain"]
    D = pd.read_csv(os.path.join(root, "events", "responsive_cells.csv"))
    G = pd.read_csv(os.path.join(root, "events", "search_grid.csv"))
    EV = pain_events(PA_SES)
    P = pd.read_csv(os.path.join(ROOTS["openfield"], "place",
                                 "place_cells.csv"))
    out = os.path.join(root, "results")
    os.makedirs(out, exist_ok=True)
    return U, D, G, EV, P, out


def cell_list(U, ses="pain"):
    """(plane, uid, z, t, anat) for every cell, interpretable first."""
    c = []
    for plane in PLANES:
        d = U[(ses, plane)]
        for i, lab in enumerate(d["labels"]):
            c.append((plane, lab, d["z"][i], d["t"], float(d["anat"][i]), i))
    return c


def dirs_of(D, level="q05"):
    """uid -> {event: UP/DOWN}, at the chosen threshold.

    The _dir columns in the table are fixed at q <= 0.05. Here the
    direction is re-derived from the p or q column so the same figures can
    be drawn at a looser bar without recomputing any statistic - which
    also means the loose and strict versions cannot drift apart.
    """
    kind, thr, _ = LEVELS[level]
    evs = [c[:-4] for c in D.columns if c.endswith("_dir")]
    g = {}
    for _, r in D.iterrows():
        if r["usable"] != 1:
            g[r["uid"]] = {}
            continue
        d = {}
        for e in evs:
            col, val = f"{e}_{kind}", f"{e}_resp"
            if col not in D or val not in D:
                continue
            if np.isfinite(r[col]) and r[col] <= thr and np.isfinite(r[val]):
                d[e] = "UP" if r[val] > 0 else "DOWN"
        g[r["uid"]] = d
    return evs, g


def final_dirs(pa_root=None):
    """The FINAL classification, in the same shape as dirs_of().

    One tag per cell, not one per event: the final test asks "is the cell
    different while stimulation is going on than during the quiet time",
    and it deliberately makes NO per-delivery and NO pin-versus-heat
    claim, because pin pricks 3.2 s apart set the answer by the window
    (0, 6, 7, 8 cells across versions) and pin and heat were given in
    separate blocks. So the events collapse to two tags:

      stimulation   TIER 1, robust
      escape        TIER 2b, 18 imaging frames inside - a hint

    Any figure drawn from this dict therefore cannot print "responds to
    pin", which is the whole point.
    """
    F = pd.read_csv(os.path.join(pa_root or ROOTS["pain"], "events",
                                 "pain_final.csv"))
    g = {}
    for _, r in F.iterrows():
        s, d = str(r["final"]), {}
        for tag, key in (("stim-UP", ("stimulation", "UP")),
                         ("stim-DOWN", ("stimulation", "DOWN")),
                         ("escape-UP", ("escape", "UP")),
                         ("escape-DOWN", ("escape", "DOWN"))):
            if tag in s:
                d[key[0]] = key[1]
        g[r["uid"]] = d
    return ["stimulation", "escape"], g


# ------------------------------------------------------------------ fig 19
def fig_before_after(U, D, EV, out):
    """First 30 s before any stimulus against the last 30 s after them all."""
    c = [x for x in cell_list(U) if x[4] >= ANAT_MIN]
    t = U[("pain", "A")]["t"]
    first = min(v.min() for v in EV.values())
    last = max(v.max() for v in EV.values())
    m_pre = (t >= max(first - WIN_S, t[0])) & (t < first)
    m_post = (t > last) & (t <= min(last + WIN_S, t[-1]))
    if m_post.sum() < 10:
        m_post = t >= t[-1] - WIN_S
    # dF/F as well as z, and it matters which: z here is DETRENDED with a
    # 20 s running median, which by construction removes changes slower
    # than ~20 s. A real slow drift across ten minutes would be invisible
    # in z and visible in dF/F, so both are shown.
    dff = {}
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            dff[lab] = d["dff"][i]
    pre = np.array([x[2][m_pre].mean() for x in c])
    post = np.array([x[2][m_post].mean() for x in c])
    f_pre = np.array([dff[x[1]][m_pre].mean() for x in c])
    f_post = np.array([dff[x[1]][m_post].mean() for x in c])
    labs = [x[1] for x in c]

    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2))
    for A, a, b, ttl, yl in (
            (ax[0], f_pre, f_post, "mean dF/F  (NOT detrended)",
             "mean dF/F over the window"),
            (ax[1], pre, post, "mean z  (detrended, 20 s median)",
             "mean z over the window")):
        st, p = wilcoxon(a, b)
        for i in range(len(a)):
            A.plot([0, 1], [a[i], b[i]], color="#BBBBBB", lw=1.2, zorder=1)
        A.scatter(np.zeros(len(a)), a, s=55, color="#6BAED6",
                  edgecolor=INK, lw=.8, zorder=3, label="before")
        A.scatter(np.ones(len(b)), b, s=55, color="#FB6A4A",
                  edgecolor=INK, lw=.8, zorder=3, label="after")
        A.plot([0, 1], [a.mean(), b.mean()], color=INK, lw=3, zorder=4)
        A.errorbar([0, 1], [a.mean(), b.mean()],
                   yerr=[a.std() / np.sqrt(len(a)),
                         b.std() / np.sqrt(len(b))],
                   color=INK, lw=3, capsize=7, capthick=2.5, zorder=5)
        A.set_xticks([0, 1])
        A.set_xticklabels([f"before\nfirst stimulus\n({WIN_S:.0f} s)",
                           f"after\nlast stimulus\n({WIN_S:.0f} s)"])
        A.set_xlim(-.35, 1.35)
        A.set_ylabel(yl)
        sig = "n.s." if p > .05 else ("*" if p > .01 else "**")
        A.set_title(f"{ttl}\nWilcoxon signed-rank, n = {len(a)} cells, "
                    f"p = {p:.4f}  {sig}")
        A.legend(fontsize=11, frameon=False, loc="best")
        A.spines[["top", "right"]].set_visible(False)

    A = ax[2]
    d = f_post - f_pre
    o = np.argsort(d)
    A.barh(range(len(d)), d[o],
           color=[UP if v > 0 else DOWN for v in d[o]], height=.72)
    A.set_yticks(range(len(d)))
    A.set_yticklabels([labs[i] for i in o], fontsize=9)
    A.axvline(0, color=INK, lw=1.4)
    A.set_xlabel("change in mean dF/F  (after - before)")
    A.set_title(f"per cell\n{int((d > 0).sum())} up, "
                f"{int((d < 0).sum())} down of {len(d)}")
    A.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Did the whole population shift over the session?\n"
                 f"{WIN_S:.0f} s immediately before the first delivery "
                 f"({first:.0f} s) against {WIN_S:.0f} s after the last "
                 f"({last:.0f} s); paired, same cells", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .88])
    p_ = os.path.join(out, "fig19_before_after.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return p_, dict(pre=pre, post=post, labs=labs)


# ------------------------------------------------------------------ fig 20
def fig_event_maps(U, D, evs, g, out):
    """One field-of-view map per event type: who is up, who is down, where."""
    show = [e for e in evs if not e.endswith("_in")]
    n = len(show)
    fig, axes = plt.subplots(2, n, figsize=(max(3.6 * n, 8.5), 8.6),
                             squeeze=False)
    mimg = {p: bp(TF.load(ROOTS["pain"], p)["mean"]) for p in PLANES}
    for j, ev in enumerate(show):
        for r, plane in enumerate(PLANES):
            A = axes[r, j]
            d = U[("pain", plane)]
            A.imshow(mimg[plane], cmap="gray")
            nu = nd = 0
            for i, lab in enumerate(d["labels"]):
                dead = d["anat"][i] < ANAT_MIN
                dv = g.get(lab, {}).get(ev, "")
                col = DEAD if dead else (UP if dv == "UP" else
                                         DOWN if dv == "DOWN" else NONE)
                m = (d["S"][:, i].reshape(d["shape"])
                     > .2 * d["S"][:, i].max()).astype(np.uint8)
                for cc in cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)[0]:
                    A.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                           np.append(cc[:, 0, 1], cc[0, 0, 1]), color=col,
                           lw=3.0 if dv else 1.6)
                    if dv:
                        A.fill(cc[:, 0, 0], cc[:, 0, 1], color=col,
                               alpha=.45, lw=0)
                if dv:
                    # label ABOVE the footprint, never on it - a name
                    # printed inside the soma hides the cell shape
                    ys, xs = np.nonzero(m)
                    A.text(xs.mean(), ys.min() - 8, lab, color="white",
                           fontsize=11, fontweight="bold", ha="center",
                           va="bottom",
                           path_effects=[pe.withStroke(linewidth=3.2,
                                                       foreground=col)])
                    nu += dv == "UP"
                    nd += dv == "DOWN"
            A.set_xticks([])
            A.set_yticks([])
            if r == 0:
                A.set_title(f"{ev}", fontsize=13, fontweight="bold",
                            color=EVCOL.get(ev, INK))
            A.set_ylabel(f"plane {plane}", fontsize=12) if j == 0 else None
            A.text(.02, .02, f"{nu} up, {nd} down", transform=A.transAxes,
                   fontsize=11, color="white", fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=3,
                                               foreground="black")])
    fig.suptitle("Where the responsive cells are, one map per event\n"
                 "red = UP, blue = DOWN, light grey = not significant "
                 "(" + LEVTXT + "), dark grey = footprint not on a soma\n"
                 "each name sits ABOVE its cell so the soma stays visible",
                 fontsize=15)
    # tight_layout leaves a third of the canvas empty on image axes, and
    # the slide then shows a postage stamp
    fig.subplots_adjust(left=.03, right=.99, top=.865, bottom=.01,
                        wspace=.03, hspace=.03)
    p_ = os.path.join(out, f"fig20_event_maps{SUF}.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return p_


# ------------------------------------------------------------------ fig 21
def peri(z, t, ev, lo=-5, hi=8, step=.1, pre=(-2., 0.)):
    lags = np.arange(lo, hi + 1e-9, step)
    M = np.stack([np.interp(e + lags, t, z, left=np.nan, right=np.nan)
                  for e in ev])
    b = np.nanmean(M[:, (lags >= pre[0]) & (lags < pre[1])], 1, keepdims=True)
    return lags, M - b


def fig_group_means(U, D, EV, evs, g, out):
    """Group averages, EVERY panel against the same reference.

    The earlier version subtracted a [-2, 0] s pre-window on the five
    event panels and the stimulus-free time on the two epoch panels - two
    different references in one figure, and the wrong one on the panels
    that were read as the headline. The population average peaks 1-2 s
    BEFORE the key press (mean z: -0.05 at -6 s, +0.17 at -2 s, +0.17 at
    -1 s, +0.12 at 0, +0.08 at +2 s), so the pre-window put the zero at
    the top of the response and forced everything after 0 negative. That
    is why pin read "no response (n=28)".

    Now all panels use the reference the final classification uses: the
    stimulus-free frames within +-60 s of each event, so a session-long
    drift cancels and nothing is subtracted from inside the response.

    Groups are read from pain_final.csv's `final` column, not re-derived
    here, so the figure cannot disagree with the table. Note that means
    the SAME grouping on pin, heat and the reflex panels: TIER 1 asks
    "stimulation or quiet", and it is not modality-specific.
    """
    F = pd.read_csv(os.path.join(ROOTS["pain"], "events", "pain_final.csv"))
    fin = {r["uid"]: str(r["final"]) for _, r in F.iterrows()}
    from pain_responsive import free_mask
    from tap_delay import frame_to_imaging, load_scoring
    S = load_scoring(PA_SES)
    tb = frame_to_imaging(PA_SES, "MiceVideo1")
    d_t = tb[np.clip(S["d_frame"], 1, len(tb)) - 1]
    r_t = tb[np.clip(S["reflex"][:, 0], 1, len(tb)) - 1]
    t = U[("pain", "A")]["t"]
    free = free_mask(t, np.sort(np.concatenate([d_t, r_t])))

    NOTE = {"pin": "TIER 1 grouping - NOT pin-specific",
            "heat": "TIER 1 grouping - NOT heat-specific",
            "paw_withdrawal": "reflexes sit inside stimulation epochs",
            "flinch": "reflexes sit inside stimulation epochs",
            "escape": "TIER 2b, 18 frames inside - a hint"}
    show = [e for e in evs if e in EV]
    eps = [e for e in evs if e.endswith("_epoch") and e[:-6] in EV]
    c = [x for x in cell_list(U) if x[4] >= ANAT_MIN]
    n = len(show) + len(eps)
    # 4 across at most: seven panels in one row comes out 33 in wide and is
    # unreadable once it is on a slide
    nc = min(n, 4)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 5.2 * nr),
                             squeeze=False)

    def local_peri(z, ev, lags):
        """Peri-event mean, each event against its OWN local quiet time."""
        rows = []
        for e in ev:
            w = np.interp(e + lags, t, z, left=np.nan, right=np.nan)
            nr = free & (t >= e - 60.0) & (t <= e + 60.0)
            if nr.sum() >= 15:
                rows.append(w - z[nr].mean())
        return np.stack(rows) if rows else None

    for j, ev in enumerate(show + eps):
        base = ev[:-6] if ev.endswith("_epoch") else ev
        key = "escape" if base == "escape" else "stim"
        lags = (np.arange(-6, 20.01, .2) if ev.endswith("_epoch")
                else np.arange(-5, 8.01, .1))
        win = (0, 10) if ev.endswith("_epoch") else (0, 2)
        A = axes[j // nc, j % nc]
        for gname, col in ((f"{key}-UP", UP), (f"{key}-DOWN", DOWN),
                           ("not modulated", NONE)):
            idx = [i for i, x in enumerate(c)
                   if (gname in fin.get(x[1], "") if gname != "not modulated"
                       else f"{key}-" not in fin.get(x[1], ""))]
            if not idx:
                continue
            Y = [local_peri(c[i][2], EV[base], lags) for i in idx]
            Y = [np.nanmean(y, 0) for y in Y if y is not None]
            if not Y:
                continue
            Y = np.stack(Y)
            m = np.nanmean(Y, 0)
            se = np.nanstd(Y, 0) / max(np.sqrt(len(Y)), 1)
            A.fill_between(lags, m - se, m + se, color=col, alpha=.25, lw=0)
            A.plot(lags, m, color=col, lw=2.8,
                   label=f"{gname} (n={len(Y)})")
        A.axvline(0, color=INK, lw=1.8)
        A.axhline(0, color=DIM, lw=1)
        A.axvspan(*win, color="#999999", alpha=.16, lw=0)
        A.set_xlabel("time from event (s)")
        A.set_ylabel("z  -  local stimulus-free mean")
        A.set_title(f"{ev}   ({len(EV[base])} events)\n"
                    f"grey = the {win[1]:.0f} s window tested\n"
                    f"{NOTE.get(base, '')}",
                    color=EVCOL.get(base, INK))
        A.legend(fontsize=11, frameon=False)
        A.spines[["top", "right"]].set_visible(False)
    for j in range(n, nr * nc):
        axes[j // nc, j % nc].axis("off")

    fig.suptitle(
        "Group averages - every panel against the SAME reference:  z minus "
        "the stimulus-free frames within +-60 s of that event\n"
        "groups read from pain_final.csv:  stim-UP A1, A4     escape-UP "
        "A12, B7, B13, B14, B15, B16     shading = SEM across cells\n"
        "the time before 0 is ALREADY raised because pin pricks come 3.2 s "
        "apart - which is exactly why a pre-event window cannot be the "
        "baseline here", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .93])
    p_ = os.path.join(out, f"fig21_group_means{SUF}.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return p_


# ------------------------------------------------------------------ fig 22
def fig_population(U, D, G, evs, g, out):
    c = [x for x in cell_list(U) if x[4] >= ANAT_MIN]
    labs = [x[1] for x in c]
    show = [e for e in evs if not e.endswith("_in")]
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=.42,
                          wspace=.22)

    A = fig.add_subplot(gs[0, 0])
    up = [sum(1 for l in labs if g.get(l, {}).get(e) == "UP") for e in show]
    dn = [sum(1 for l in labs if g.get(l, {}).get(e) == "DOWN")
          for e in show]
    x = np.arange(len(show))
    A.bar(x - .2, up, .4, color=UP, label="UP", edgecolor=INK, lw=1.2)
    A.bar(x + .2, dn, .4, color=DOWN, label="DOWN", edgecolor=INK, lw=1.2)
    for xi, (u, v) in enumerate(zip(up, dn)):
        if u:
            A.text(xi - .2, u + .05, str(u), ha="center", fontsize=12,
                   fontweight="bold")
        if v:
            A.text(xi + .2, v + .05, str(v), ha="center", fontsize=12,
                   fontweight="bold")
    A.set_xticks(x)
    A.set_xticklabels(show, rotation=25, ha="right")
    A.set_ylabel(f"cells (of {len(labs)} interpretable)")
    A.set_title(f"how many respond, per event\nq <= 0.05, BH across the "
                f"{len(labs)} cells")
    A.legend(fontsize=11, frameon=False)
    A.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[0, 1])
    nresp = [len(g.get(l, {})) for l in labs]
    vals, cnts = np.unique(nresp, return_counts=True)
    A.bar(vals, cnts, color="#7B8794", edgecolor=INK, lw=1.2, width=.7)
    for v, ct in zip(vals, cnts):
        A.text(v, ct + .1, str(ct), ha="center", fontsize=12,
               fontweight="bold")
    A.set_xticks(vals)
    A.set_xlabel("number of events this cell responds to")
    A.set_ylabel("cells")
    A.set_title(f"most cells answer to one thing or nothing\n"
                f"{int(sum(1 for v in nresp if v == 0))} of {len(labs)} "
                f"respond to nothing")
    A.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[1, :])
    M = np.zeros((len(show), len(show)))
    for i, e1 in enumerate(show):
        for j, e2 in enumerate(show):
            M[i, j] = sum(1 for l in labs
                          if e1 in g.get(l, {}) and e2 in g.get(l, {}))
    im = A.imshow(M, cmap="YlOrRd", vmin=0, vmax=max(M.max(), 1))
    for i in range(len(show)):
        for j in range(len(show)):
            if M[i, j]:
                A.text(j, i, int(M[i, j]), ha="center", va="center",
                       fontsize=13, fontweight="bold",
                       color="white" if M[i, j] > M.max() * .6 else INK)
    A.set_xticks(range(len(show)))
    A.set_xticklabels(show, rotation=25, ha="right")
    A.set_yticks(range(len(show)))
    A.set_yticklabels(show)
    cb = fig.colorbar(im, ax=A, fraction=.025, pad=.01)
    cb.set_label("number of cells", fontsize=11)
    ov = [(l, list(g[l])) for l in labs if len(g.get(l, {})) > 1]
    A.set_title("overlap: cells that respond to more than one thing\n"
                + ("; ".join(f"{l}: " + ", ".join(f"{e} {g[l][e]}"
                                                  for e in es)
                             for l, es in ov) if ov else "none"),
                fontsize=12.5)
    fig.suptitle(f"Population summary, {len(labs)} interpretable cells of "
                 f"{len(cell_list(U))} total", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .93])
    p_ = os.path.join(out, f"fig22_population{SUF}.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return p_


# ------------------------------------------------------------------ fig 23
def fig_cross(U, D, P, EV, evs, g, out):
    """Pain-responsive cells: what did each do in the open field?"""
    pref = {r["cell"]: (r["preference"], r["contrast"], r["q_contrast"])
            for _, r in P.iterrows()}
    resp = [l for l in [x[1] for x in cell_list(U) if x[4] >= ANAT_MIN]
            if g.get(l)]
    if not resp:
        return None
    zo = None
    from openfield_place_cells import load_tracking
    from union_data import OF_SESSION
    tr = load_tracking(OF_SESSION)
    tof = U[("openfield", "A")]["t"]
    ok = tr["ok"].to_numpy() == 1
    zo = np.round(np.interp(tof, tr["t_s"].to_numpy()[ok],
                            tr["zone"].to_numpy()[ok].astype(float))
                  ).astype(int)
    tpa = U[("pain", "A")]["t"]

    n = len(resp)
    H = 2.6 * n + 1.6
    fig = plt.figure(figsize=(18, H))
    gs = fig.add_gridspec(n, 2, width_ratios=[1, 1], wspace=.16, hspace=.55)
    for i, lab in enumerate(resp):
        plane = "A" if lab.startswith("A") else "B"
        dp, do = U[("pain", plane)], U[("openfield", plane)]
        k = dp["labels"].index(lab)
        pr, con, q = pref.get(lab, ("not tested", np.nan, np.nan))

        A = fig.add_subplot(gs[i, 0])
        A.plot(tpa, dp["z"][k], lw=.8, color=INK)
        lo = dp["z"][k].min() - .28 * np.ptp(dp["z"][k])
        st = .075 * np.ptp(dp["z"][k])
        for j, (nm, ev) in enumerate(EV.items()):
            A.plot(ev, np.full(len(ev), lo + j * st), "|", ms=8, mew=1.8,
                   color=EVCOL.get(nm, INK))
            if i == 0:
                A.text(tpa[-1] * 1.004, lo + j * st, nm, fontsize=9,
                       va="center", color=EVCOL.get(nm, INK),
                       fontweight="bold")
        A.set_xlim(tpa[0], tpa[-1])
        A.set_ylabel("z", fontsize=12)
        A.set_title(f"{lab}  PAIN:  "
                    + ", ".join(f"{e} {v}" for e, v in g[lab].items()),
                    fontsize=12.5, fontweight="bold", loc="left",
                    color=UP if "UP" in g[lab].values() else DOWN)
        A.spines[["top", "right"]].set_visible(False)
        if i == n - 1:
            A.set_xlabel("time (s), pain session")

        A = fig.add_subplot(gs[i, 1])
        A.plot(tof, do["z"][k], lw=.8, color=INK)
        lo = do["z"][k].min() - .22 * np.ptp(do["z"][k])
        for v, cc in ((0, CEN), (1, COR)):
            mm = zo == v
            if mm.any():
                A.fill_between(tof, lo, lo + .10 * np.ptp(do["z"][k]),
                               where=mm, color=cc, lw=0, step="mid")
        A.set_xlim(tof[0], tof[-1])
        A.set_ylabel("z", fontsize=12)
        col = {"centre": CEN, "corner": COR}.get(pr, DIM)
        A.set_title(f"{lab}  OPEN FIELD:  {pr}"
                    + (f"  (contrast {con:+.2f} z, q = {q:.3f})"
                       if np.isfinite(con) else ""),
                    fontsize=12.5, fontweight="bold", loc="left", color=col)
        A.spines[["top", "right"]].set_visible(False)
        if i == n - 1:
            A.set_xlabel("time (s), open field   |   bar = blue centre, "
                         "pink corner")
    fig.suptitle("Every cell with a pain verdict, and what the SAME cell "
                 "did in the open field\nleft: pain session with the "
                 "scored events    right: open field with the mouse's "
                 "zone under the trace\nthe verdict on each row is "
                 f"whatever produced it - here, {LEVTXT}",
                 fontsize=15, y=1 - .3 / H)
    fig.subplots_adjust(left=.045, right=.955, top=1 - 1.35 / H,
                        bottom=.55 / H)
    p_ = os.path.join(out, f"fig23_cross_session{SUF}.png")
    fig.savefig(p_, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return p_


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="q05", choices=list(LEVELS))
    a = ap.parse_args()
    U, D, G, EV, P, out = load_all()
    global SUF, LEVTXT
    SUF = "" if a.level == "q05" else f"_{a.level}"
    LEVTXT = LEVELS[a.level][2]
    print(f"  threshold: {LEVTXT}")
    print(f"results figures -> {out}")
    if a.level == "final":
        # the final classification has two tags and no per-event columns,
        # so only the figures that take g as a label - the maps and the
        # cross-session rows - can be drawn from it. fig21 reads
        # pain_final.csv itself at any level, and fig22's counts are
        # per-event by construction and belong to the withdrawn set.
        evs, g = final_dirs()
        _, g20 = dirs_of(D, "q20")
        for f in (fig_event_maps(U, D, evs, g, out),
                  fig_group_means(U, D, EV,
                                  [e for e in dirs_of(D, "q20")[0]], g20,
                                  out),
                  fig_cross(U, D, P, EV, evs, g, out)):
            if f:
                print(f"  {f}")
        return
    evs, g = dirs_of(D, a.level)
    p, _ = fig_before_after(U, D, EV, out)
    print(f"  {p}")
    for f in (fig_event_maps(U, D, evs, g, out),
              fig_group_means(U, D, EV, evs, g, out),
              fig_population(U, D, G, evs, g, out),
              fig_cross(U, D, P, EV, evs, g, out)):
        if f:
            print(f"  {f}")


if __name__ == "__main__":
    main()
