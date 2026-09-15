"""cell_atlas.py  -  which neuron is which, what it did, and where it sits.

Four figures per session, built so a cell can be followed from the field of
view to its trace to its group average without guessing.

  1  WHERE IS IT        fig11  the field of view with every cell
                        outlined thick and labelled. Same ids in both
                        sessions, so A1 here is A1 there.
  2  WHAT IS IT         fig12  the same field, coloured by what the neuron
                        does - up / down / not responsive in the pain
                        session, centre / corner / none in the open field.
                        This is the "where are the up cells" map.
  3  WHAT IT DID        fig13  every neuron's z trace stacked, with the
                        behaviour underneath on the same time axis: zone
                        colour and speed for the open field, stimulus and
                        behaviour ticks for the pain session.
  4  THE AVERAGES       fig14  mean +- SEM across all neurons, then across
                        each group separately - responsive, up only, down
                        only, non-responsive - over the same behaviour
                        axis, plus the peri-event average per group.

GROUPS
  pain        from responsive_cells.csv: a neuron is UP or DOWN for a given
              event type if q <= 0.05 there. "responsive" = UP or DOWN to
              anything. Because a neuron can be UP to one event and DOWN to
              another, the group figures are drawn per event type.
  open field  from place_cells.csv: centre-preferring, corner-preferring or
              neither, by the sign of the zone contrast at q <= 0.05.

Neurons whose footprint did not land on a soma in that session (anat < 1
image SD) are drawn in grey and excluded from every average - their trace
is background.

OUTPUT  ->  <session>\\output_split\\atlas\\

USAGE
  python cell_atlas.py                 # both sessions
  python cell_atlas.py --session pain
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

import transfer_footprints as TF  # noqa: E402
from pain_responsive import free_mask  # noqa: E402
from union_data import ANAT_MIN, PLANES, ROOTS, load_union  # noqa: E402

UP, DOWN, NONE, DEAD = "#C1272D", "#1C6E8C", "#BDBDBD", "#5A5A5A"
SUF = ""     # filename suffix for a non-default threshold
CEN, COR, EDGE = "#2166AC", "#D6604D", "#CCCCCC"
INK, DIM = "#1A1A1A", "#666666"
EVCOL = {"pin": "#C1272D", "heat": "#E08214", "paw_withdrawal": "#5E3C99",
         "flinch": "#1B7837", "escape": "#00688B"}


def mean_image(session_key, plane):
    return TF.load(ROOTS[session_key], plane)["mean"]


def bp(img):
    v = img.astype(np.float32)
    lo, hi = np.percentile(v, [2, 99.5])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def outline(ax, col, shape, color, lw=2.6, fill=.18, label=None,
            fontsize=11):
    """A footprint drawn so it can actually be seen, and named."""
    m = (col.reshape(shape) > 0.2 * col.max()).astype(np.uint8)
    if not m.any():
        return
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    for c in cs:
        if len(c) > 4:
            ax.plot(np.append(c[:, 0, 0], c[0, 0, 0]),
                    np.append(c[:, 0, 1], c[0, 0, 1]), color=color, lw=lw,
                    solid_joinstyle="round")
            if fill:
                ax.fill(c[:, 0, 0], c[:, 0, 1], color=color, alpha=fill,
                        lw=0)
    if label:
        ys, xs = np.nonzero(m)
        ax.text(xs.mean(), ys.mean(), label, color="white",
                fontsize=fontsize, fontweight="bold", ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=3.4,
                                            foreground=color)])


# ------------------------------------------------------------------ groups
def pain_groups(pa_root, level="q05"):
    """uid -> {event: 'UP'|'DOWN'} at the chosen threshold.

    The _dir columns in the CSV are frozen at q <= 0.05. Reading them
    directly is what left ten slides of per-cell rows on the strict
    criterion after the deck had switched to q <= 0.20 - the figure and
    the slide titles disagreed. The direction is now re-derived from the
    p/q columns through the same helper the figures use.
    """
    f = os.path.join(pa_root, "events", "responsive_cells.csv")
    if not os.path.exists(f):
        raise SystemExit("run pain_responsive.py first")
    D = pd.read_csv(f)
    from results_figures import dirs_of
    evs, g = dirs_of(D, level)
    return D, evs, g


def of_groups(of_root):
    f = os.path.join(of_root, "place", "place_cells.csv")
    P = pd.read_csv(f)
    return P, {r["cell"]: r["preference"] for _, r in P.iterrows()}


def pain_events(session):
    from tap_delay import frame_to_imaging, load_scoring
    from pain_responsive import collect_events, MIN_EVENTS
    S = load_scoring(session)
    tb = frame_to_imaging(session, "MiceVideo1")
    EV, KIND = collect_events(S, tb)
    return {k: v for k, v in EV.items() if len(v) >= MIN_EVENTS}


def of_behaviour(session, t_img):
    """Zone and speed at each imaging frame."""
    import openfield_track as oft
    from openfield_place_cells import behaviour_on_imaging, load_tracking
    tr = load_tracking(session)
    arena = cv2.imread(os.path.join(session, "output_split", "tracking",
                                    "openfield_track_qc.png"))
    bbox = None
    # zones come from the tracking csv directly - it already stores the
    # zone per behaviour frame, so no arena re-fit is needed here
    tb = tr["t_s"].to_numpy()
    zo = tr["zone"].to_numpy().astype(float)
    ok = tr["ok"].to_numpy() == 1
    zi = np.round(np.interp(t_img, tb[ok], zo[ok])).astype(int)
    sp = np.interp(t_img, tb[ok], tr["speed_px_s"].to_numpy()[ok])
    x = np.interp(t_img, tb[ok], tr["x"].to_numpy()[ok])
    y = np.interp(t_img, tb[ok], tr["y"].to_numpy()[ok])
    _ = (arena, bbox, behaviour_on_imaging, oft)
    return zi, sp, x, y


# ------------------------------------------------------------------ fig 11
def fig_fov(U, ses, out, colour_by=None, tag="fig11_fov", title=""):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    for ax, plane in zip(axes, PLANES):
        d = U[(ses, plane)]
        ax.imshow(bp(mean_image(ses, plane)), cmap="gray")
        for i, lab in enumerate(d["labels"]):
            dead = d["anat"][i] < ANAT_MIN
            c = DEAD if dead else (colour_by(lab) if colour_by else "#2E7D5B")
            outline(ax, d["S"][:, i], d["shape"], c,
                    lw=1.8 if dead else 2.8, fill=.10 if dead else .20,
                    label=lab, fontsize=10.5 if dead else 12)
        ax.set_title(f"plane {plane} - {len(d['labels'])} cells",
                     fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    n_all = sum(len(U[(ses, q)]["labels"]) for q in PLANES)
    fig.suptitle(title or f"{ses}: all {n_all} cells, labelled", fontsize=15)
    fig.tight_layout(rect=[0, .02, 1, .96])
    p = os.path.join(out, f"{tag}_{ses}{SUF}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ fig 13
def rail_pain(ax, EV, t0, t1):
    names = list(EV)
    for j, nm in enumerate(names):
        y = -j
        ev = EV[nm]
        ax.plot(ev, np.full(len(ev), y), "|", ms=10, mew=1.8,
                color=EVCOL.get(nm, INK))
        ax.text(t0 - .012 * (t1 - t0), y, f"{nm} ({len(ev)})", ha="right",
                va="center", fontsize=9, color=EVCOL.get(nm, INK),
                fontweight="bold")
    ax.set_ylim(-len(names) + .5, .5)
    ax.set_xlim(t0, t1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("time (s), imaging clock")


def rail_of(ax, t, zi, sp, t0, t1):
    for v, c, nm in ((0, CEN, "centre"), (1, COR, "corner"),
                     (2, EDGE, "edge")):
        m = zi == v
        if m.any():
            ax.fill_between(t, 0, 1, where=m, color=c, lw=0, step="mid",
                            label=nm)
    ax.plot(t, .1 + .8 * sp / max(np.percentile(sp, 99), 1e-9), lw=.7,
            color=INK, label="speed")
    ax.set_ylim(0, 1)
    ax.set_xlim(t0, t1)
    ax.set_yticks([])
    ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("time (s), imaging clock")


def fig_traces(U, ses, out, colour_by, rail, n_rail_rows=1.0):
    rows, labs, cols = [], [], []
    for plane in PLANES:
        d = U[(ses, plane)]
        for i, lab in enumerate(d["labels"]):
            rows.append(d["z"][i])
            labs.append(lab)
            cols.append(DEAD if d["anat"][i] < ANAT_MIN else colour_by(lab))
    t = U[(ses, "A")]["t"]
    Z = np.stack(rows)
    n = len(Z)
    fig = plt.figure(figsize=(17, 1.6 + .42 * n + 2.4 * n_rail_rows))
    gs = fig.add_gridspec(2, 1, height_ratios=[.42 * n, 2.4 * n_rail_rows],
                          hspace=.06)
    ax = fig.add_subplot(gs[0])
    step = 6.0
    for i in range(n):
        ax.plot(t, Z[i] + (n - 1 - i) * step, lw=.5, color=cols[i])
        ax.text(t[0] - .008 * (t[-1] - t[0]), (n - 1 - i) * step, labs[i],
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=cols[i])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-step, n * step)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.plot([t[-1] * .995] * 2, [0, step], color=INK, lw=2.5)
    ax.text(t[-1] * .992, step / 2, f"{step:.0f} z", rotation=90,
            ha="right", va="center", fontsize=9)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    rail(ax2, t[0], t[-1])
    fig.suptitle(f"{ses}: all {n} cells (z) on one time axis, with "
                 f"the behaviour underneath", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .975])
    p = os.path.join(out, f"fig13_traces_{ses}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ fig 14
def peri(z, t, ev, lo=-4, hi=6, step=.1, pre=(-2., 0.)):
    lags = np.arange(lo, hi + 1e-9, step)
    M = np.stack([np.interp(e + lags, t, z, left=np.nan, right=np.nan)
                  for e in ev])
    b = np.nanmean(M[:, (lags >= pre[0]) & (lags < pre[1])], 1, keepdims=True)
    return lags, M - b


def band(ax, x, Y, color, label):
    if not len(Y):
        return
    m = np.nanmean(Y, 0)
    se = np.nanstd(Y, 0) / max(np.sqrt(len(Y)), 1)
    ax.fill_between(x, m - se, m + se, color=color, alpha=.22, lw=0)
    ax.plot(x, m, color=color, lw=1.8, label=f"{label} (n={len(Y)})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="both",
                    choices=["both", "pain", "openfield"])
    ap.add_argument("--level", default="q05",
                    help="threshold for UP/DOWN: q05 q10 q20 "
                         "p05 p01")
    a = ap.parse_args()
    global SUF
    SUF = "" if a.level == "q05" else f"_{a.level}"
    U = load_union()
    todo = ["pain", "openfield"] if a.session == "both" else [a.session]

    for ses in todo:
        root = ROOTS[ses]
        out = os.path.join(root, "atlas")
        os.makedirs(out, exist_ok=True)
        sess_dir = os.path.dirname(root)
        print(f"\n===== {ses} =====")

        if ses == "pain":
            D, evs_all, G = pain_groups(
                root, "q20" if a.level == "final" else a.level)
            if a.level == "final":
                # the FINAL classification: one tag per cell, and NO
                # per-event verdict - so the event column titles come out
                # bare, which is correct. pain_final does not claim that
                # any cell responds to an individual pin prick.
                from results_figures import final_dirs
                _, G = final_dirs(root)
                _, evs_all, _ = pain_groups(root, "q20")
            EV = pain_events(sess_dir)
            # evs_all keeps pin_epoch / heat_epoch; evs is the subset that
            # has event times. Filtering before fig_rows_pain is what
            # silently dropped the epoch columns and the epoch verdicts
            # from the per-cell figure.
            evs = [e for e in evs_all if e in EV]

            def colour(lab):
                g = G.get(lab, {})
                if not g:
                    return NONE
                return UP if "UP" in g.values() else DOWN

            p1 = fig_fov(U, ses, out,
                         title="pain session: all 31 cells, labelled - the same name means the same cell in both sessions")
            p2 = fig_fov(U, ses, out, colour_by=colour, tag="fig12_class",
                         title="pain: where the responsive neurons are  "
                               "(red = UP to something, blue = DOWN only, "
                               "grey = nothing, dark grey = not on a soma)")
            p3 = fig_traces(U, ses, out, colour,
                            lambda ax, t0, t1: rail_pain(ax, EV, t0, t1),
                            n_rail_rows=1.0)
            p4 = fig_groups_pain(U, D, EV, evs, out)
            p5 = fig_rows_pain(U, D, EV, evs_all, out, G)
            print(f"  wrote {p5}")
        else:
            P, pref = of_groups(root)
            t = U[(ses, "A")]["t"]
            zi, sp, x, y = of_behaviour(sess_dir, t)

            def colour(lab):
                return {"centre": CEN, "corner": COR}.get(
                    pref.get(lab, "none"), NONE)

            p1 = fig_fov(U, ses, out,
                         title="open field: all 31 cells, labelled - the same name means the same cell in both sessions")
            p2 = fig_fov(U, ses, out, colour_by=colour, tag="fig12_class",
                         title="open field: where the zone-selective "
                               "neurons are  (blue = centre-preferring, "
                               "red = corner-preferring, grey = neither)")
            p3 = fig_traces(U, ses, out, colour,
                            lambda ax, t0, t1: rail_of(ax, t, zi, sp,
                                                       t0, t1),
                            n_rail_rows=.8)
            p4 = fig_groups_of(U, P, pref, zi, out)
        for p in (p1, p2, p3, p4):
            print(f"  wrote {p}")


def fig_groups_pain(U, D, EV, evs, out):
    """Averages: all, responsive, up only, down only, non-responsive."""
    rows, labs, anat = [], [], []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            rows.append(d["z"][i])
            labs.append(lab)
            anat.append(d["anat"][i])
    Z = np.stack(rows)
    t = U[("pain", "A")]["t"]
    ok = np.array(anat) >= ANAT_MIN
    key = {r["uid"]: r for _, r in D.iterrows()}

    n = len(evs)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 9), squeeze=False)
    for j, nm in enumerate(evs):
        ev = EV[nm]
        up = np.array([ok[i] and key[labs[i]][f"{nm}_dir"] == "UP"
                       for i in range(len(labs))])
        dn = np.array([ok[i] and key[labs[i]][f"{nm}_dir"] == "DOWN"
                       for i in range(len(labs))])
        ns = ok & ~up & ~dn
        A = axes[0, j]
        lags, _ = peri(Z[0], t, ev)
        for mask, col, lb in ((ok, INK, "all on a soma"),
                              (up, UP, "UP"), (dn, DOWN, "DOWN"),
                              (ns, NONE, "not responsive")):
            if not mask.any():
                continue
            Y = np.stack([peri(Z[i], t, ev)[1].mean(0)
                          for i in np.flatnonzero(mask)])
            band(A, lags, Y, col, lb)
        A.axvline(0, color=INK, lw=1)
        A.axhline(0, color=DIM, lw=.6)
        A.axvspan(0, 2, color="#888888", alpha=.14, lw=0)
        A.set_title(f"{nm}  ({len(ev)} events)", fontsize=12)
        A.set_xlabel("time from event (s)")
        A.set_ylabel("z, baseline [-2,0] s subtracted")
        A.legend(fontsize=8.5, frameon=False)
        A.spines[["top", "right"]].set_visible(False)

        B = axes[1, j]
        r = np.array([key[l][f"{nm}_resp"] for l in labs])
        q = np.array([key[l][f"{nm}_q"] for l in labs])
        order = np.argsort(r)
        cols = [UP if (ok[i] and q[i] <= .05 and r[i] > 0)
                else DOWN if (ok[i] and q[i] <= .05 and r[i] < 0)
                else (NONE if ok[i] else DEAD) for i in order]
        B.barh(range(len(order)), r[order], color=cols)
        B.set_yticks(range(len(order)))
        B.set_yticklabels([labs[i] for i in order], fontsize=7)
        B.axvline(0, color=INK, lw=.8)
        B.set_xlabel("response (z)")
        B.set_title(f"every neuron, sorted - {nm}", fontsize=11)
        B.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pain session: group averages per event type, and every "
                 "neuron's response\nred = UP, blue = DOWN, light grey = "
                 "not significant, dark grey = not on a soma", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = os.path.join(out, "fig14_groups_pain.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_rows_pain(U, D, EV, evs, out, G=None):
    """One row per neuron: where it is, its peri-event averages, its trace.

    The pain counterpart of of_cell_rows.py, so a neuron can be read across
    the two sessions in the same layout. Responsive neurons come first.
    """
    key = {r["uid"]: r for _, r in D.iterrows()}
    fp = os.path.join(ROOTS["pain"], "events", "pain_final.csv")
    FIN = ({r["uid"]: r for _, r in pd.read_csv(fp).iterrows()}
           if os.path.exists(fp) else {})
    cells = []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            gd = (G or {}).get(lab, {})
            # carry the statistic into the tag. A1 is "stimulation UP" on
            # the POOLED test (+0.41 z, q = 0.007) while its pin-only
            # column reads q = 0.30, and a row title without its number
            # invites exactly that confusion.
            STAT = {"stimulation": ("t1_any", "t1_any_q"),
                    "escape": ("t3_escape", "t3_q")}
            tags = []
            for e, v in gd.items():
                r_ = FIN.get(lab)
                if r_ is not None and e in STAT:
                    ec, qc = STAT[e]
                    tags.append(f"{e} {v}  ({r_[ec]:+.2f} z, q={r_[qc]:.3f})")
                else:
                    tags.append(f"{e} {v}")
            cells.append(dict(plane=plane, uid=lab, i=i,
                              usable=bool(d["anat"][i] >= ANAT_MIN),
                              tags=tags))
    cells.sort(key=lambda c: (not c["tags"], not c["usable"], c["plane"],
                              int(c["uid"][1:])))

    # The verdict list includes pin_epoch and heat_epoch, so those need a
    # panel too - a label with no evidence behind it is exactly what this
    # figure exists to avoid. They are drawn from the same delivery times
    # over a longer window, with the 10 s the test actually used shaded.
    EPOCHS = [e for e in evs if e.endswith("_epoch")
              and e[:-6] in EV]
    ev_cols = [e for e in evs if e in EV]

    n = len(cells)
    ncol = 1 + len(ev_cols) + len(EPOCHS)
    H = 2.3 * n + 1.2
    fig = plt.figure(figsize=(19, H))
    gs = fig.add_gridspec(n, ncol + 3,
                          width_ratios=[1] + [.95] * len(ev_cols)
                          + [1.15] * len(EPOCHS) + [3.4] * 3,
                          hspace=.45, wspace=.28)
    mimg = {p: bp(mean_image("pain", p)) for p in PLANES}
    lags = np.arange(-4, 6.01, .1)

    # The five event columns used to be baselined to [-2, 0] s, which is
    # the error that made this figure disagree with its own verdict: the
    # population peaks 1-2 s BEFORE the key press, so subtracting that
    # window puts the zero at the top of the response. They now use the
    # same reference as the epoch columns and as pain_final - the
    # stimulus-free frames near each event - so every panel in the row is
    # on one scale.
    t0 = U[("pain", "A")]["t"]
    free_all = free_mask(t0, np.sort(np.concatenate(
        [np.asarray(v, float) for v in EV.values()])))

    for row, c in enumerate(cells):
        d = U[("pain", c["plane"])]
        z = d["z"][c["i"]]
        t = d["t"]
        col = UP if any("UP" in x for x in c["tags"]) else (
            DOWN if c["tags"] else NONE)
        if not c["usable"]:
            col = DEAD

        A = fig.add_subplot(gs[row, 0])
        S = d["S"][:, c["i"]].reshape(d["shape"])
        m = (S > .2 * S.max()).astype(np.uint8)
        yy, xx = np.nonzero(m)
        cy, cx = int(yy.mean()), int(xx.mean())
        h, w = d["shape"]
        y0, y1 = max(cy - 60, 0), min(cy + 60, h)
        x0, x1 = max(cx - 60, 0), min(cx + 60, w)
        A.imshow(mimg[c["plane"]][y0:y1, x0:x1], cmap="gray")
        for cc in cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)[0]:
            A.fill(cc[:, 0, 0], cc[:, 0, 1], color=col, alpha=.45, lw=0)
            A.plot(np.append(cc[:, 0, 0], cc[0, 0, 0]),
                   np.append(cc[:, 0, 1], cc[0, 0, 1]), color=col, lw=2.4)
        A.text(cx - x0, cy - y0, c["uid"], color="white", fontsize=12,
               fontweight="bold", ha="center", va="center",
               path_effects=[pe.withStroke(linewidth=3.4, foreground=col)])
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(c["uid"], fontsize=12, color=col, fontweight="bold")

        for j, nm in enumerate(ev_cols):
            B = fig.add_subplot(gs[row, 1 + j])
            M = np.stack([np.interp(e + lags, t, z, left=np.nan,
                                    right=np.nan)
                          - z[free_all & (t >= e - 60) & (t <= e + 60)].mean()
                          for e in EV[nm]
                          if (free_all & (t >= e - 60)
                              & (t <= e + 60)).sum() >= 15])
            mn = np.nanmean(M, 0)
            se = np.nanstd(M, 0) / max(np.sqrt(len(M)), 1)
            cj = EVCOL.get(nm, INK)
            B.fill_between(lags, mn - se, mn + se, color=cj, alpha=.22, lw=0)
            B.plot(lags, mn, color=cj, lw=1.4)
            B.axvline(0, color=INK, lw=.8)
            B.axhline(0, color=DIM, lw=.5)
            B.axvspan(0, 2, color="#888888", alpha=.14, lw=0)
            dirv = (G or {}).get(c["uid"], {}).get(nm, "")
            B.set_title(f"{nm} {dirv}",
                        fontsize=9, color=cj if (isinstance(dirv, str)
                                                 and dirv) else DIM)
            B.tick_params(labelsize=7)
            B.spines[["top", "right"]].set_visible(False)
            if j == 0:
                B.set_ylabel("z", fontsize=8)

        # the 10 s epoch tests, same delivery times, longer window. This
        # IS the TIER 1 test of pain_final, so it uses pain_final's own
        # reference: the stimulus-free frames within +-60 s of each
        # delivery. A global free mask sits 66 s late in this session
        # (only 4 % of the dense pin block is free) and a [-2,0] s
        # baseline put cells below zero while the table called them UP.
        lag2 = np.arange(-6, 20.01, .2)
        for j, nm in enumerate(EPOCHS):
            base = nm[:-6]
            B = fig.add_subplot(gs[row, 1 + len(ev_cols) + j])
            M = np.stack([np.interp(e + lag2, t, z, left=np.nan,
                                    right=np.nan)
                          - z[free_all & (t >= e - 60) & (t <= e + 60)].mean()
                          for e in EV[base]
                          if (free_all & (t >= e - 60)
                              & (t <= e + 60)).sum() >= 15])
            mn = np.nanmean(M, 0)
            se = np.nanstd(M, 0) / max(np.sqrt(len(M)), 1)
            cj = EVCOL.get(base, INK)
            B.fill_between(lag2, mn - se, mn + se, color=cj, alpha=.22,
                           lw=0)
            B.plot(lag2, mn, color=cj, lw=1.4)
            B.axvline(0, color=INK, lw=.8)
            B.axhline(0, color=DIM, lw=.5)
            B.axvspan(0, 10, color="#888888", alpha=.16, lw=0)
            B.set_xlabel("s from delivery", fontsize=7)
            r_ = FIN.get(c["uid"], key[c["uid"]])
            dirv = (G or {}).get(c["uid"], {}).get(nm, "")
            # pain_final's own TIER 1 numbers when they are there, so the
            # panel prints the statistic that produced the verdict
            eff = r_.get(f"t1_{base}", r_.get(f"{nm}_resp", np.nan))
            q_ = r_.get(f"t1_{base}_q", r_.get(f"{nm}_q", np.nan))
            B.set_title(f"{nm} {dirv if isinstance(dirv, str) else ''}\n"
                        f"{eff:+.2f} z, q={q_:.2f}"
                        if np.isfinite(eff) else nm,
                        fontsize=8.5,
                        color=cj if dirv else DIM)
            B.tick_params(labelsize=7)
            B.spines[["top", "right"]].set_visible(False)

        C = fig.add_subplot(gs[row, 1 + len(ev_cols) + len(EPOCHS):])
        C.plot(t, z, lw=.5, color=INK)
        lo = z.min() - .30 * (z.max() - z.min())
        st = .07 * (z.max() - z.min())
        for j, nm in enumerate(EV):
            C.plot(EV[nm], np.full(len(EV[nm]), lo + j * st), "|",
                   ms=6, mew=1.2, color=EVCOL.get(nm, INK))
            if row == 0:
                C.text(t[-1] * 1.006, lo + j * st, nm, fontsize=8,
                       va="center", color=EVCOL.get(nm, INK),
                       fontweight="bold")
        C.set_xlim(t[0], t[-1])
        C.set_ylim(lo - st, z.max() * 1.05)
        C.set_ylabel("z", fontsize=9)
        C.tick_params(labelsize=8)
        C.spines[["top", "right"]].set_visible(False)
        C.set_title(f"{c['uid']}   "
                    + (" | ".join(c["tags"]) if c["tags"]
                       else ("not on a soma - trace is background"
                             if not c["usable"] else "no response")),
                    fontsize=11, color=col, fontweight="bold", loc="left")
        if row == n - 1:
            C.set_xlabel("time (s)   |   ticks under the trace = scored "
                         "events", fontsize=9)

    fig.suptitle("Pain session: one row per neuron - where it is, its "
                 "peri-event average for every event type, and its whole "
                 "trace with the scored events\nresponsive neurons first; "
                 "red = UP to something, blue = DOWN, grey = neither, dark "
                 "grey = not on a soma\nEVERY peri-event panel is z minus "
                 "the stimulus-free frames within +-60 s of that event - "
                 "the same reference pain_final uses, and no pre-event "
                 "window\nthe pin_epoch / heat_epoch numbers are the "
                 "SINGLE-MODALITY version of the verdict test and are NOT "
                 "used for the verdict: pin was given 13-204 s and "
                 "519-586 s, heat 269-518 s, so a modality difference is "
                 "also an early-versus-middle difference", fontsize=13,
                 y=1 - .18 / H)
    # Fixed margins in inches, not tight_layout. On a 72-inch figure a
    # fractional rect leaves eight inches of white above the first row.
    # 1.45 in of header, not 0.85: the title is three lines now and at
    # 0.85 it sat on top of the first row's panels.
    fig.subplots_adjust(left=.035, right=.95, top=1 - 1.75 / H,
                        bottom=.5 / H)
    p = os.path.join(out, f"fig16_rows_pain{SUF}.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_groups_of(U, P, pref, zi, out):
    """Open field: zone-triggered averages and the per-cell contrast."""
    rows, labs, anat = [], [], []
    for plane in PLANES:
        d = U[("openfield", plane)]
        for i, lab in enumerate(d["labels"]):
            rows.append(d["z"][i])
            labs.append(lab)
            anat.append(d["anat"][i])
    Z = np.stack(rows)
    t = U[("openfield", "A")]["t"]
    ok = np.array(anat) >= ANAT_MIN
    cen = np.array([ok[i] and pref.get(labs[i]) == "centre"
                    for i in range(len(labs))])
    cor = np.array([ok[i] and pref.get(labs[i]) == "corner"
                    for i in range(len(labs))])
    ns = ok & ~cen & ~cor

    # entries into the centre and into a corner, as events
    ent = {}
    for v, nm in ((0, "centre entry"), (1, "corner entry")):
        m = (zi == v).astype(np.int8)
        d_ = np.diff(np.concatenate([[0], m]))
        ent[nm] = t[np.flatnonzero(d_ == 1)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for j, (nm, ev) in enumerate(ent.items()):
        A = axes[0, j]
        # Drawn even when there are only a handful of entries: the mouse
        # crossed into the centre 4 times in the whole session, and that
        # fact matters more than the curve does. The n is in the title.
        if len(ev) >= 2:
            lags, _ = peri(Z[0], t, ev)
            for mask, col, lb in ((ok, INK, "all on a soma"),
                                  (cen, CEN, "centre-preferring"),
                                  (cor, COR, "corner-preferring"),
                                  (ns, NONE, "neither")):
                if not mask.any():
                    continue
                Y = np.stack([peri(Z[i], t, ev)[1].mean(0)
                              for i in np.flatnonzero(mask)])
                band(A, lags, Y, col, lb)
            A.axvline(0, color=INK, lw=1)
            A.axhline(0, color=DIM, lw=.6)
        A.set_title(f"{nm}  ({len(ev)} entries)"
                    + ("   TOO FEW TO MEAN MUCH" if len(ev) < 8 else ""),
                    fontsize=12,
                    color="#A63A3A" if len(ev) < 8 else INK)
        A.set_xlabel("time from entry (s)")
        A.set_ylabel("z, baseline [-2,0] s subtracted")
        A.legend(fontsize=8.5, frameon=False)
        A.spines[["top", "right"]].set_visible(False)

    A = axes[1, 0]
    key = {r["cell"]: r for _, r in P.iterrows()}
    r = np.array([key[l]["contrast"] if l in key else np.nan for l in labs])
    order = np.argsort(np.nan_to_num(r))
    cols = [CEN if pref.get(labs[i]) == "centre"
            else COR if pref.get(labs[i]) == "corner"
            else (NONE if ok[i] else DEAD) for i in order]
    A.barh(range(len(order)), np.nan_to_num(r[order]), color=cols)
    A.set_yticks(range(len(order)))
    A.set_yticklabels([labs[i] for i in order], fontsize=7)
    A.axvline(0, color=INK, lw=.8)
    A.set_xlabel("zone contrast (centre mean - corner mean, z)")
    A.set_title("every neuron, sorted", fontsize=11)
    A.spines[["top", "right"]].set_visible(False)

    A = axes[1, 1]
    for mask, col, lb in ((cen, CEN, "centre-preferring"),
                          (cor, COR, "corner-preferring"),
                          (ns, NONE, "neither")):
        if not mask.any():
            continue
        m = np.nanmean(Z[mask], 0)
        A.plot(t, m, lw=.8, color=col, label=f"{lb} (n={int(mask.sum())})")
    for v, c in ((0, CEN), (1, COR)):
        A.fill_between(t, A.get_ylim()[0], A.get_ylim()[0] + .25,
                       where=zi == v, color=c, lw=0, step="mid")
    A.set_xlim(t[0], t[-1])
    A.set_xlabel("time (s); the band at the bottom is the mouse's zone")
    A.set_ylabel("group mean z")
    A.legend(fontsize=8.5, frameon=False)
    A.spines[["top", "right"]].set_visible(False)
    A.set_title("group means over the whole session", fontsize=11)

    fig.suptitle("Open field: group averages by zone preference, and every "
                 "neuron's zone contrast", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95])
    p = os.path.join(out, "fig14_groups_openfield.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    main()
