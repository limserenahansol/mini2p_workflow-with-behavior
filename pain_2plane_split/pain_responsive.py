"""pain_responsive.py  -  up cells and down cells, per stimulus and per behaviour.

Hansol's design, with one part changed for a reason that is measured:
  - the scored tap times as they are, with no reaction-delay correction
  - every single delivery is its own event; no grouping into bouts
  - response  [0, +2] s after the event
  - REFERENCE: the stimulus-free time, NOT a per-trial pre-window
  - every stimulus AND every behaviour tested the same way
  - a cell is UP or DOWN, not just "responsive"

WHY THE BASELINE HAD TO CHANGE
  The design as first specified used [-2, 0] s before each event. Two
  measurements on these event times say that cannot work here:

  1  No pre-window is clean. Pin pricks come a median 3.2 s apart, so
     another delivery falls inside the baseline for 28 % of trials at
     [-2,0] s, 47 % at [-4,-2], 45 % at [-6,-4] and 30 % at [-8,-6].
  2  The response starts BEFORE the tap. The population peri-event
     average runs -0.05 z at -6 s, +0.17 at -1 s, +0.12 at 0, +0.08 at
     +2 - it peaks one to two seconds before the key press. A [-2,0] s
     baseline therefore begins the measurement at the top of the
     response.

  The consequence was not subtle. With [-2,0] the population looked
  SUPPRESSED by pin (p = 0.006) and no cell passed; with [-6,-4] the same
  data gave EXCITATION (+0.26, p = 0.0006); restricted to the 12
  deliveries with 6 s of clear time before them it was +0.46 (p < 1e-4).
  The sign was a property of the window. The earlier rounds of this
  analysis, and the other pipeline's pin result, both came from a
  pre-window inside the rise.

  The fix uses no pre-window: every response is measured against the
  stimulus-free time - frames more than 2 s before and 10 s after ANY
  delivery or tap, 38 % of the session. The pre-window version is kept in
  the table as <event>_prewin_* so the difference stays visible.

WHAT COUNTS AS AN EVENT
  pin, heat                  each scored delivery
  paw withdrawal, flinch     each scored tap
  attending, licking/biting  the ONSET of each held episode, so a
  guarding, escape/rearing   behaviour goes through the identical test.
                             The inside-versus-outside version is next to
                             it as <behaviour>_in_*.

AND A SLOWER ONE
  <stimulus>_epoch  mean z over the 10 s after each delivery against the
                    same stimulus-free reference, because the response is
                    seconds long rather than a transient locked to the
                    tap.

STATISTICS
  p from a circular-shift null of that cell's own trace (2000 surrogates,
  minimum shift 10 s), which keeps the trace's autocorrelation and the
  event times and destroys only the pairing. Two-sided. q =
  Benjamini-Hochberg across the cells within each event type.

  UP    passes the criterion and the response is positive
  DOWN  passes the criterion and the response is negative

  Cells whose footprint did not land on a soma in this session (anat < 1
  image SD) are listed but excluded from the FDR and from every count.

OUTPUT  ->  <pain session>\output_split\events\
  responsive_cells.csv , responsive_matrix.csv , responsive_report.txt ,
  fig10_responsive.png

USAGE
  python pain_responsive.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from pain_events import bh, cell_seed, shifts, surr_p
from tap_delay import frame_to_imaging, load_scoring
from union_data import ANAT_MIN, PLANES, ROOTS, load_union

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
PRE = (-2.0, 0.0)
POST = (0.0, 2.0)
ALPHA = 0.05
EPOCH_S = 10.0      # the slower window: a stimulus is 'in effect'
                    # for this long after each delivery
MIN_EVENTS = 5      # below this the peri-event average is one or two
                    # windows and the test is not meaningful; such event
                    # types are listed with their n and not tested


def resp_stat(x, t, ev):
    """Mean z in [0,+2] s minus mean z in [-2,0] s, averaged over events."""
    if not len(ev):
        return np.nan
    v = []
    for e in ev:
        a = (t >= e + PRE[0]) & (t < e + PRE[1])
        b = (t >= e + POST[0]) & (t <= e + POST[1])
        if a.sum() and b.sum():
            v.append(x[b].mean() - x[a].mean())
    return float(np.mean(v)) if v else np.nan


def free_mask(t, all_ev, pre=2.0, post=10.0):
    """Frames with nothing happening: the reference that replaces the
    per-trial pre-window.

    WHY A PRE-WINDOW CANNOT BE USED IN THIS SESSION. Pin pricks come a
    median 3.2 s apart, so ANY 2 s window before a delivery contains
    another delivery for a large fraction of trials - measured on these
    times: 28 % for [-2,0] s, 47 % for [-4,-2], 45 % for [-6,-4], 30 %
    for [-8,-6]. There is no clean choice.

    Worse, the population peri-event average PEAKS 1-2 s before the tap
    (mean z: -0.05 at -6 s, +0.17 at -1 s, +0.12 at 0, +0.08 at +2), so a
    [-2,0] baseline starts the measurement at the top of the response and
    makes every cell look suppressed. That is what produced the earlier
    "pin inhibits the population, p = 0.006" result, and it reverses to
    +0.26 (p = 0.0006) with a [-6,-4] baseline. The sign was a property of
    the window, not of the neurons.

    This mask is the stimulus-free time: more than `pre` before and `post`
    after ANY delivery, and away from any reflex tap. Nothing is
    subtracted per trial, so no window choice can flip the answer.
    """
    m = np.ones(len(t), bool)
    for e in np.asarray(all_ev, float):
        m &= ~((t >= e - pre) & (t <= e + post))
    return m


def free_stat(x, t, ev, free, win=POST):
    """Mean in [0,+2] s after each event, minus the stimulus-free mean."""
    if not len(ev) or free.sum() < 20:
        return np.nan
    base = x[free].mean()
    v = []
    for e in ev:
        b = (t >= e + win[0]) & (t <= e + win[1])
        if b.sum():
            v.append(x[b].mean() - base)
    return float(np.mean(v)) if v else np.nan


def inside_stat(x, m):
    if m.sum() < 3 or (~m).sum() < 3:
        return np.nan
    return float(x[m].mean() - x[~m].mean())


def onsets(mask, t_cam):
    """Start time of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8)]))
    return t_cam[np.flatnonzero(d == 1)]


def collect_events(S, tb):
    """Every event type, as times on the imaging clock. No delay applied."""
    ev, meta = {}, {}
    d_t = tb[np.clip(S["d_frame"], 1, len(tb)) - 1]
    for k in sorted(set(S["d_type"])):
        nm = S["stim"][k - 1]
        ev[nm] = np.sort(d_t[S["d_type"] == k])
        meta[nm] = "stimulus"
    r_t = tb[np.clip(S["reflex"][:, 0], 1, len(tb)) - 1]
    for k in sorted(set(S["reflex"][:, 1])):
        nm = S["ref"][k - 1].split("/")[0].strip().lower().replace(" ", "_")
        ev[nm] = np.sort(r_t[S["reflex"][:, 1] == k])
        meta[nm] = "reflex"
    for k in range(1, len(S["aff"])):
        m = S["score"] == k
        if not m.any():
            continue
        nm = S["aff"][k].split("/")[0].strip().lower().replace(" ", "_")
        ev[nm] = onsets(m, tb[:len(m)])
        meta[nm] = "behaviour"
    return ev, meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PAIN)
    a = ap.parse_args()
    root = os.path.join(a.session, "output_split")
    out = os.path.join(root, "events")
    os.makedirs(out, exist_ok=True)

    S = load_scoring(a.session)
    tb = frame_to_imaging(a.session, "MiceVideo1")
    EV, KIND = collect_events(S, tb)
    U = load_union()

    # What the scoring actually contains, before anything is tested. Two of
    # the four held behaviours were never pressed in this session and one
    # was pressed once, so most of the affective side of the design has no
    # data behind it - that has to be visible, not discovered later.
    COVER = []
    for k in range(1, len(S["aff"])):
        m = S["score"] == k
        nm = S["aff"][k].split("/")[0].strip().lower().replace(" ", "_")
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        n_ep = int((d == 1).sum())
        COVER.append((S["aff"][k], nm, n_ep, float(m.sum()) / S["fps"]))
    EVALL = dict(EV)   # before the MIN_EVENTS filter, for the epochs
    skipped = {nm for nm, ev in EV.items() if len(ev) < MIN_EVENTS}
    EV = {nm: ev for nm, ev in EV.items() if len(ev) >= MIN_EVENTS}

    # episode masks, for the inside-versus-outside column
    aff_mask = {}
    for k in range(1, len(S["aff"])):
        m = S["score"] == k
        nm = S["aff"][k].split("/")[0].strip().lower().replace(" ", "_")
        if m.any() and nm not in skipped:
            aff_mask[nm] = m

    stim_codes = sorted(set(S["d_type"]))
    t0 = U[("pain", PLANES[0])]["t"]
    ALL_EV = np.concatenate([v for v in EVALL.values()]) if EVALL         else np.array([])
    FREE = free_mask(t0, ALL_EV)
    print(f"  stimulus-free reference: {FREE.sum()} of {len(t0)} frames "
          f"({100 * FREE.mean():.0f} % of the session)")
    rows = []
    for plane in PLANES:
        d = U[("pain", plane)]
        t, Z, fs = d["t"], d["z"], d["fs"]
        for i, lab in enumerate(d["labels"]):
            z = Z[i]
            rng = np.random.default_rng(cell_seed(plane, lab, "resp2s"))
            sh = shifts(len(z), fs, rng)
            r = dict(plane=plane, uid=lab,
                     anat=round(float(d["anat"][i]), 2),
                     usable=int(d["anat"][i] >= ANAT_MIN),
                     source=d["source"][i], matched=bool(d["matched"][i]))
            for nm, ev in EV.items():
                # PRIMARY: against the stimulus-free time. See free_mask
                # for why a per-trial pre-window is not usable here.
                o = free_stat(z, t, ev, FREE)
                p, sd = surr_p(o, lambda y: free_stat(y, t, ev, FREE),
                               z, sh)
                r[f"{nm}_resp"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"{nm}_p"] = round(p, 4) if np.isfinite(p) else np.nan
                r[f"{nm}_nullsd"] = round(sd, 4) if np.isfinite(sd) \
                    else np.nan
                r[f"{nm}_n"] = len(ev)
                # kept for comparison: the pre-window version that the
                # earlier rounds used, and that the sign depends on
                o2 = resp_stat(z, t, ev)
                p2, _ = surr_p(o2, lambda y: resp_stat(y, t, ev), z, sh)
                r[f"{nm}_prewin_resp"] = round(o2, 4) if np.isfinite(o2) \
                    else np.nan
                r[f"{nm}_prewin_p"] = round(p2, 4) if np.isfinite(p2) \
                    else np.nan
            # SLOWER TIMESCALE. The +-2 s test asks for a transient locked
            # to the tap. Diagnostics say that is not what is here: every
            # cell has 7-11 transients per minute, the design could detect
            # 0.17 z per cell, and yet 4 cells clearly differ between the
            # pin period, the heat period and the quiet period. So the
            # same question is asked again over 10 s: mean z while a
            # stimulus of this type is in effect, minus mean z when
            # NEITHER stimulus is. Same circular-shift null.
            for k in stim_codes:
                nm = S["stim"][k - 1]
                inside = np.zeros(len(t), bool)
                other = np.zeros(len(t), bool)
                for kk in stim_codes:
                    m_ = np.zeros(len(t), bool)
                    for e in EVALL[S["stim"][kk - 1]]:
                        m_ |= (t >= e) & (t <= e + EPOCH_S)
                    if kk == k:
                        inside = m_
                    else:
                        other |= m_
                keep = inside | ~(inside | other)

                def ep_stat(x, inside=inside, keep=keep):
                    a, b = inside & keep, (~inside) & keep
                    if a.sum() < 3 or b.sum() < 3:
                        return np.nan
                    return float(x[a].mean() - x[b].mean())

                o = ep_stat(z)
                p, _ = surr_p(o, ep_stat, z, sh)
                r[f"{nm}_epoch_resp"] = round(o, 4) if np.isfinite(o) \
                    else np.nan
                r[f"{nm}_epoch_p"] = round(p, 4) if np.isfinite(p) \
                    else np.nan
                r[f"{nm}_epoch_frac"] = round(float(inside.mean()), 4)

            for nm, m in aff_mask.items():
                mm = np.interp(t, tb[:len(m)], m.astype(float)) > 0.5
                o = inside_stat(z, mm)
                p, _ = surr_p(o, lambda y: inside_stat(y, mm), z, sh)
                r[f"{nm}_in_resp"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"{nm}_in_p"] = round(p, 4) if np.isfinite(p) else np.nan
                r[f"{nm}_in_frac"] = round(float(mm.mean()), 4)
            rows.append(r)

    D = pd.DataFrame(rows)
    ok = (D["usable"] == 1).to_numpy()
    names = (list(EV)
             + [f"{S['stim'][k - 1]}_epoch" for k in stim_codes]
             + [f"{n}_in" for n in aff_mask])
    for nm in names:
        q = np.full(len(D), np.nan)
        q[ok] = bh(D.loc[ok, f"{nm}_p"].to_numpy())
        D[f"{nm}_q"] = np.round(q, 4)
        lab = []
        for j, (_, r) in enumerate(D.iterrows()):
            if not ok[j] or not np.isfinite(q[j]) or q[j] > ALPHA:
                lab.append("")
            else:
                lab.append("UP" if r[f"{nm}_resp"] > 0 else "DOWN")
        D[f"{nm}_dir"] = lab

    M = D[["plane", "uid", "anat", "usable"]].copy()
    for nm in names:
        M[nm] = [x if x else "." for x in D[f"{nm}_dir"]]
    D.to_csv(os.path.join(out, "responsive_cells.csv"), index=False)
    M.to_csv(os.path.join(out, "responsive_matrix.csv"), index=False)
    report(D, M, EV, KIND, aff_mask, names, out, S, COVER,
           skipped)
    figure(D, M, EV, KIND, names, U, out)
    return D


def baseline_clean(ev):
    """Fraction of events whose [-2,0] s baseline has no other same event."""
    if len(ev) < 2:
        return 1.0
    g = np.diff(np.concatenate([[np.inf], ev]))
    return float(np.mean(g >= -PRE[0]))


def report(D, M, EV, KIND, aff_mask, names, out, S, COVER,
           skipped):
    ok = D["usable"] == 1
    L = ["===== pain session: up cells and down cells =====", "",
         f"{len(D)} cells; {int(ok.sum())} land on a soma in this "
         f"session and are counted.",
         "",
         f"DESIGN.  Scored tap times used as they are, no "
         f"reaction-delay correction.",
         f"Every single delivery is its own event, no bouts. Response "
         f"[{POST[0]:.0f}, +{POST[1]:.0f}] s after the event.",
         "Frame numbers are still mapped to the imaging clock through the "
         "sync line - without that the",
         "events drift up to 0.44 s from the traces by the end of the "
         "session.",
         "",
         "REFERENCE: the stimulus-free time, not a per-trial pre-window. "
         "A pre-window cannot be used here -",
         f"another delivery falls inside [{PRE[0]:.0f},{PRE[1]:.0f}] s for "
         f"28 % of trials (47 % at [-4,-2], 45 % at [-6,-4]), and the",
         "population average PEAKS 1-2 s BEFORE the tap, so a pre-window "
         "starts the measurement at the top",
         "of the response. With [-2,0] the population looked suppressed by "
         "pin (p = 0.006) and no cell passed;",
         "with [-6,-4] the same data gave excitation (p = 0.0006). The "
         "sign was a property of the window.",
         "The pre-window version is kept as <event>_prewin_* so the "
         "difference stays visible.",
         "",
         "Two-sided circular-shift null, 2000 surrogates, BH across cells "
         "within each event type.",
         "UP = q <= 0.05 and the response is positive; DOWN = q <= 0.05 and "
         "it is negative.",
         "",
         "WHAT THE SCORING CONTAINS.  The four held behaviours have almost "
         "no data behind them in this",
         "session - the keys were tapped rather than held, and two were "
         "never pressed at all:"]
    for full, nm, n_ep, secs in COVER:
        L.append(f"    {full:<30s} {n_ep:>2d} episodes, {secs:6.1f} s total"
                 + ("   NEVER SCORED" if n_ep == 0 else
                    f"   -> {'not tested, fewer than ' + str(MIN_EVENTS) + ' events' if nm in skipped else 'tested'}"))
    L += ["  The reflex taps are the substantive behaviour data here: "
          "76 paw withdrawal and 50 flinch.",
          "",
         f"  {'event':<18s} {'kind':<10s} {'n':>4s} "
         f"{'UP':>3s} {'DOWN':>4s}  cells"]
    for nm in names:
        if nm.endswith("_in"):
            base, kind = nm[:-3], "episode"
            n = int(round(float(D[f"{nm}_frac"].iloc[0]) * 100))
        elif nm.endswith("_epoch"):
            base, kind = nm[:-6], f"{EPOCH_S:.0f} s window"
            n = int(round(float(D[f"{nm}_frac"].iloc[0]) * 100))
        else:
            base, kind = nm, KIND.get(nm, "")
            n = int(D[f"{nm}_n"].iloc[0])
        cl = "-"   # the pre-window column is gone; see the header
        up = D[ok & (D[f"{nm}_dir"] == "UP")]["uid"].tolist()
        dn = D[ok & (D[f"{nm}_dir"] == "DOWN")]["uid"].tolist()
        L.append(f"  {nm:<18s} {kind:<10s} {n:>4d} "
                 f"{len(up):>3d} {len(dn):>4d}  "
                 + ("UP " + ", ".join(up) + "  " if up else "")
                 + ("DOWN " + ", ".join(dn) if dn else ""))

    L += ["", "  the matrix, one row per neuron "
               "(. = not significant, blank row = responds to nothing):",
          "  " + f"{'uid':>5s} {'anat':>5s} "
          + " ".join(f"{n[:9]:>9s}" for n in names)]
    for _, r in M.iterrows():
        if not r["usable"]:
            continue
        L.append(f"  {r['uid']:>5s} {r['anat']:5.1f} "
                 + " ".join(f"{r[n]:>9s}" for n in names))

    nores = [r["uid"] for _, r in M.iterrows()
             if r["usable"] and all(r[n] == "." for n in names)]
    L += ["",
          f"  responds to nothing: {len(nores)} of {int(ok.sum())}  "
          + (", ".join(nores) if nores else "-")]

    pl = os.path.join(ROOTS["openfield"], "place", "place_cells.csv")
    if os.path.exists(pl):
        P = pd.read_csv(pl)
        key = dict(zip(zip(P["plane"], P["cell"]), P["preference"]))
        L += ["", "  the zone-selective cells from the open field, and what "
                  "they do in the pain session:"]
        for _, r in M.iterrows():
            pref = key.get((r["plane"], r["uid"]), "not tested")
            if pref not in ("centre", "corner"):
                continue
            hits = [f"{n} {r[n]}" for n in names if r[n] != "."]
            L.append(f"    {r['uid']} ({pref}): "
                     + (", ".join(hits) if hits else "nothing"))

    L += ["",
          "Caveats.",
          f"  Baseline overlap. Pin pricks come a median 3.2 s apart, so a "
          f"2 s baseline sits inside the",
          f"  previous delivery's response for some of them - the "
          f"'baseline clean' column is the fraction",
          f"  with no same-type event in their own baseline. Where that "
          f"number is low, a real response is",
          f"  partly subtracted from itself and the test is conservative "
          f"(it loses cells, it does not invent",
          f"  them).",
          "  No delay correction. The key press lags the event it "
          "describes, so the +-2 s windows sit",
          "  slightly late relative to the true stimulus by whatever that "
          "lag is.",
          "  Episode onsets for the four held behaviours: the onset is when "
          "the scorer pressed the key",
          "  down, and the inside-versus-outside column next to it does not "
          "depend on onset timing at all.",
          "  One animal, one session."]
    txt = "\n".join(L)
    with open(os.path.join(out, "responsive_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def figure(D, M, EV, KIND, names, U, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ok = D["usable"] == 1
    sub = D[ok].reset_index(drop=True)
    ev_names = [n for n in names if n in EV]
    nE = len(ev_names)

    fig = plt.figure(figsize=(16, 9 + 1.5 * len(sub) / 2))
    gs = fig.add_gridspec(2 + int(np.ceil(len(sub) / 4)), max(nE, 4),
                          height_ratios=[2.6, 1.5]
                          + [1.0] * int(np.ceil(len(sub) / 4)),
                          hspace=.55, wspace=.3)

    # 1 the matrix
    ax = fig.add_subplot(gs[0, :])
    V = np.zeros((len(sub), len(names)))
    for j, nm in enumerate(names):
        for i in range(len(sub)):
            V[i, j] = {"UP": 1, "DOWN": -1}.get(sub[f"{nm}_dir"][i], 0)
    cm = ListedColormap(["#1C6E8C", "#F2F2F2", "#C1272D"])
    ax.imshow(V, cmap=cm, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["uid"], fontsize=8)
    for i in range(len(sub)):
        for j, nm in enumerate(names):
            if V[i, j]:
                ax.text(j, i, sub[f"{nm}_dir"][i], ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sub), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.8)
    ax.tick_params(which="minor", length=0)
    ax.set_title("red = UP, blue = DOWN, grey = not significant "
                 "(q <= 0.05, BH within each column)", fontsize=12)

    # 2 counts per event
    ax = fig.add_subplot(gs[1, :])
    up = [int((sub[f"{n}_dir"] == "UP").sum()) for n in names]
    dn = [int((sub[f"{n}_dir"] == "DOWN").sum()) for n in names]
    x = np.arange(len(names))
    ax.bar(x, up, .38, color="#C1272D", label="UP")
    ax.bar(x + .38, dn, .38, color="#1C6E8C", label="DOWN")
    for xi, (u, v) in enumerate(zip(up, dn)):
        if u:
            ax.text(xi, u, str(u), ha="center", va="bottom", fontsize=8)
        if v:
            ax.text(xi + .38, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + .19)
    ax.set_xticklabels([f"{n}\n(n={int(D[f'{n[:-3] if n.endswith(chr(95)+chr(105)+chr(110)) else n}_n'].iloc[0]) if (n in EV) else ''})"
                        if n in EV else n
                        for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("neurons")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"how many of the {len(sub)} interpretable neurons respond "
                 f"to each event", fontsize=12)

    # 3 peri-event averages, one panel per cell, all event types overlaid
    lags = np.arange(-4, 6.01, .1)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (_, r) in enumerate(sub.iterrows()):
        A = fig.add_subplot(gs[2 + i // 4, i % 4])
        d = U[("pain", r["plane"])]
        zi = d["labels"].index(r["uid"])
        z = d["z"][zi]
        for j, nm in enumerate(ev_names):
            Mx = np.stack([np.interp(e + lags, d["t"], z, left=np.nan,
                                     right=np.nan) for e in EV[nm]])
            b = np.nanmean(Mx[:, (lags >= PRE[0]) & (lags < PRE[1])], 1,
                           keepdims=True)
            m = np.nanmean(Mx - b, 0)
            A.plot(lags, m, lw=1.1, color=cols[j % 10],
                   label=nm if i == 0 else None)
        A.axvline(0, color="#1A1A1A", lw=.9)
        A.axhline(0, color="#888888", lw=.6)
        A.axvspan(POST[0], POST[1], color="#888888", alpha=.14, lw=0)
        tags = [f"{n[:4]}{r[f'{n}_dir'][0]}" for n in names
                if r[f"{n}_dir"]]
        A.set_title(f"{r['uid']}  " + (" ".join(tags) if tags else "-"),
                    fontsize=8.5)
        A.tick_params(labelsize=7)
        A.spines[["top", "right"]].set_visible(False)
        if i == 0:
            A.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(
        f"Up and down cells, {len(sub)} neurons x {len(names)} event types"
        f"\nbaseline [{PRE[0]:.0f}, {PRE[1]:.0f}] s, response "
        f"[{POST[0]:.0f}, +{POST[1]:.0f}] s, scored times as given, every "
        f"delivery its own event", fontsize=13, y=.997)
    fig.subplots_adjust(left=.06, right=.99, top=.95, bottom=.02)
    p = os.path.join(out, "fig10_responsive.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
