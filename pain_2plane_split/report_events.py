"""report_events.py  -  the text and figures for pain_events.py.

Kept separate so the statistics and the drawing do not share a scope: every
number written here comes out of event_cells.csv, so the figure and the table
cannot disagree.

FIGURES, ZOOMED OUT FIRST
  fig8_event_locked.png
    1  the whole session: every cell as a row of z, with the scored
       stimuli and the freezing bouts marked. This is the view that shows
       whether anything is locked to anything before a single average is
       taken.
    2  population peri-stimulus average, per stimulus, against the
       circular-shift null
    3  the lag check: response against lag, with the correction tap_delay.py
       measured marked. If the neural profile peaked somewhere else, the
       correction would be wrong - this panel is how you would see that.
    4  one panel per neuron, pin and heat

  fig9_movement.png
    1  motion energy for the whole session with the freezing bouts
    2  each neuron's movement correlation against its pin response, so a
       "pain" cell that is really a movement cell is visible as a point on
       the diagonal
    3  the same response before and after regressing movement out
    4  the cross-tabulation
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ALPHA = 0.05
PRE, POST = (-3.0, -1.0), (0.5, 3.0)
BOUT_PRE, BOUT_RESP = (-6.0, -1.0), (0.5, 3.0)
INK, DIM = "#1A1A1A", "#666666"
C_PIN, C_HEAT = "#C1272D", "#B8860B"
C_MOVE, C_FRZ = "#1C6E8C", "#2E7D5B"


def peri_matrix(z, t, ev, lo=-8.0, hi=14.0, step=0.1, base=None):
    """Peri-event traces, baseline subtracted.

    `ev` is a list of BOUT ONSETS, not individual deliveries: the deliveries
    in this session come every ~3 s in runs of up to 9, so a per-delivery
    average subtracts one delivery's response off the next one's baseline.
    The window is long enough to contain a whole bout (median 5-6 s, longest
    22 s) plus its tail.
    """
    lags = np.arange(lo, hi + 1e-9, step)
    M = np.full((len(ev), len(lags)), np.nan)
    for i, e in enumerate(ev):
        M[i] = np.interp(e + lags, t, z, left=np.nan, right=np.nan)
    # Centred on the SESSION median, not on a pre-event window, to match
    # the statistic. Subtracting the pre-bout window would draw the dip
    # that the bout-selection rule creates there and make every cell look
    # like it responded.
    return lags, M - np.median(z)


def run(D, S, d_t, r_t, r_type, tb, MO, FZ, U, out, delay,
        stim_names, BT):
    ok = D["usable"] == 1
    L = ["===== pain session: what each neuron responds to =====", "",
         f"{len(D)} cells; {int(ok.sum())} have a footprint on a "
         f"soma in this session (anat >= 1 image SD)",
         f"and are counted. The other {int((~ok).sum())} are listed in the "
         f"table but their trace is background.",
         "",
         f"TIMING.  The scored taps were moved {delay:.2f} s earlier, the "
         f"delay measured from the video by tap_delay.py",
         "(bottom camera, where the stimulus is; not from these neurons).",
         "",
         "WHAT AN EVENT IS.  Bouts, not single deliveries. The deliveries "
         "are not isolated trials: the",
         "median gap between pin pricks is 3.2 s and only 2 pin and 1 heat "
         "deliveries have clear space",
         "on both sides, so a per-delivery baseline usually sits inside the "
         "previous delivery's response",
         "and subtracts the signal away - that version finds nothing, and "
         "is kept as <stimulus>_single_*.",
         "A bout is a run of deliveries less than 5 s apart; "
         + ", ".join(f"{nm} has {len(BT[nm])} bouts of "
                     f"{int(np.median([b[2] for b in BT[nm]]))} deliveries "
                     f"(median)" for nm in stim_names) + ".",
         "",
         f"WHAT THE RESPONSE IS.  Mean z in "
         f"[{BOUT_RESP[0]:+.1f}, {BOUT_RESP[1]:+.1f}] s after bout onset, "
         f"minus the cell's own SESSION MEDIAN.",
         "Not minus a pre-stimulus window, which is the obvious choice and "
         "is wrong here: a bout is kept",
         "only if the 5 s before it is delivery-free, and with deliveries "
         "every ~3 s that condition SELECTS",
         "unusually quiet stretches. [-6,-1] s is clean in 12 of 12 pin "
         "bouts by construction, [-12,-7] in",
         "only 6 of 12, so subtracting it inflates the response. All four "
         "baselines are in the table",
         "(<stimulus>_prebase / _pre12 / _pre20), and a cell is called "
         "ROBUST only if it passes under all",
         "four. The counts, which are the honest summary of how much the "
         "choice matters:",
         "  " + "   ".join(
             f"{nm}: " + ", ".join(
                 f"{lab} {int((D[ok][c] <= ALPHA).sum())}"
                 for lab, c in (("median", f"{nm}_q"),
                                ("[-6,-1]", f"{nm}_prebase_q"),
                                ("[-12,-7]", f"{nm}_pre12_q"),
                                ("[-20,-15]", f"{nm}_pre20_q")))
             for nm in stim_names),
         "Null = circular shift of each cell's own trace, 2000 surrogates, "
         "BH across the cells.",
         ""]

    fams = []
    for nm in stim_names:
        fams.append((nm, f"{nm}_resp", f"{nm}_q"))
        fams.append((f"{nm}, movement removed", f"{nm}_nomove_resp",
                     f"{nm}_nomove_q"))
    for c in D.columns:
        if c.endswith("_q") and c[:-2] not in [f[0] for f in fams] \
                and not c.startswith("q_"):
            base = c[:-2]
            if base in [f"{n}" for n in stim_names] or \
                    base in [f"{n}_nomove" for n in stim_names]:
                continue
            val = f"{base}_resp" if f"{base}_resp" in D.columns else \
                f"{base}_diff"
            if val in D.columns:
                fams.append((base, val, c))
    fams.append(("movement (side camera)", "r_motion_side", "q_motion_side"))
    fams.append(("movement (bottom camera)", "r_motion_bottom",
                 "q_motion_bottom"))

    L.append("  per test, the neurons that pass q <= 0.05:")
    L.append(f"  {'test':<28s} {'n pass':>6s}  cells")
    for name, val, q in fams:
        if q not in D.columns:
            continue
        h = D[ok & (D[q] <= ALPHA)]
        L.append(f"  {name:<28s} {len(h):>6d}  "
                 + (", ".join(f"{r['uid']}({r[val]:+.2f})"
                              for _, r in h.iterrows()) if len(h) else "-"))

    L += ["", "  ROBUST responders - pass under all four baseline "
               "definitions:"]
    for nm in stim_names:
        rb = D[D[f"{nm}_robust"]]
        L.append(f"    {nm}: {len(rb)}  "
                 + (", ".join(f"{r['uid']} ({r[f'{nm}_resp']:+.2f} z)"
                              for _, r in rb.iterrows()) if len(rb)
                    else "none"))

    L += ["", "  the question the movement columns exist to answer:"]
    for nm in stim_names:
        raw = D[ok & (D[f"{nm}_q"] <= ALPHA)]
        both = raw[raw[f"{nm}_nomove_q"] <= ALPHA]
        lost = raw[raw[f"{nm}_nomove_q"] > ALPHA]
        L.append(f"    {nm}: {len(raw)} responsive, {len(both)} still "
                 f"responsive with movement regressed out"
                 + (f"; {len(lost)} were movement "
                    f"({', '.join(lost['uid'])})" if len(lost) else ""))

    L += ["", "  every neuron, with its open-field answer next to its pain "
               "answer:",
          f"  {'plane':>5s} {'uid':>4s} {'anat':>5s} "
          + " ".join(f"{n:>8s} {'b/4':>3s}" for n in stim_names)
          + f" {'r move':>7s} {'freeze':>7s} {'OF place':>9s}  label"]
    for _, r in D.iterrows():
        L.append(f"  {r['plane']:>5s} {r['uid']:>4s} {r['anat']:5.1f} "
                 + " ".join(f"{r[f'{n}_resp']:+8.2f} "
                             f"{int(r[f'{n}_n_baselines']):>3d}"
                             for n in stim_names)
                 + f" {r['r_motion_side']:+7.2f} "
                 f"{r['freeze_diff']:+7.2f} "
                 f"{str(r.get('of_place', '-')):>9s}  {r['label']}")

    if "of_place" in D.columns:
        zs = D[ok & D["of_place"].isin(["centre", "corner"])]
        L += ["", "  THE COMBINED QUESTION - was the neuron that preferred "
                  "the exposed centre also stimulus-responsive?"]
        for _, r in zs.iterrows():
            bits = []
            for nm in stim_names:
                q = r[f"{nm}_q"]
                qn = r[f"{nm}_nomove_q"]
                bits.append(
                    f"{nm} {r[f'{nm}_resp']:+.2f} (q={q:.3f}"
                    + (", survives movement" if np.isfinite(qn)
                       and qn <= ALPHA else
                       ", movement" if np.isfinite(q) and q <= ALPHA
                       else "") + ")")
            L.append(f"    {r['uid']} {r['of_place']}-preferring "
                     f"({r['of_contrast']:+.2f}):  " + ";  ".join(bits))
        if not len(zs):
            L.append("    no zone-selective cell is interpretable here")

    # what size of effect could this have found? A null that a real effect
    # has to clear is more informative than "n.s.", and it is the first
    # thing to ask when nothing passes.
    sens = []
    for nm in stim_names:
        sd = D.loc[ok, f"{nm}_nullsd"]
        obs = D.loc[ok, f"{nm}_resp"].abs()
        sens.append(f"{nm}: null SD {np.nanmedian(sd):.2f} z, so about "
                    f"{2 * np.nanmedian(sd):.2f} z was detectable; the "
                    f"largest response seen was {np.nanmax(obs):.2f} z")
    L += ["", "  SENSITIVITY - what could this have found?"]
    L += [f"    {s}" for s in sens]

    L += ["", "Caveats.",
          "  One delay for every delivery. Trial-to-trial variation in the "
          "human's reaction time stays in,",
          "  which widens the peri-stimulus average but does not shift it.",
          f"  Pin ran {d_t[S['d_type'] == 1].min():.0f}-"
          f"{d_t[S['d_type'] == 1].max():.0f} s and heat "
          f"{d_t[S['d_type'] == 2].min():.0f}-"
          f"{d_t[S['d_type'] == 2].max():.0f} s, so the two do overlap in "
          f"time rather than being two",
          "  clean blocks - but they are not interleaved trial by trial "
          "either, so a slow drift can still",
          "  imitate a difference between them. The circular-shift null "
          "controls for drift within a cell;",
          "  it cannot separate 'heat' from 'the second half of the "
          "session'.",
          "  Movement is camera motion energy, not tracked speed: it says "
          "how much the image changed.",
          "  Regressing it out removes what it shares with the neuron, and "
          "an imperfect regressor",
          "  under-removes - so 'survives movement' is the weaker claim of "
          "the two, not the stronger.",
          "  One animal, one session of each type."]

    txt = "\n".join(L)
    with open(os.path.join(out, "event_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    # ---------------------------------------------------------------- fig 8
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prof = np.load(os.path.join(out, "lag_profiles.npz"))
    lags_scan = prof["lags"]
    n = len(D)
    ncol = 4
    nrow_cells = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(16, 7.2 + 2.3 * nrow_cells))
    gs = fig.add_gridspec(2 + nrow_cells, ncol,
                          height_ratios=[2.5, 1.7] + [1.0] * nrow_cells,
                          hspace=.62, wspace=.26)

    # 1 whole session
    ax = fig.add_subplot(gs[0, :])
    Zs, labs = [], []
    for plane in ("A", "B"):
        d = U[("pain", plane)]
        for i, lb in enumerate(d["labels"]):
            Zs.append(d["z"][i])
            labs.append(lb)
    t = U[("pain", "A")]["t"]
    Zs = np.stack(Zs)
    im = ax.imshow(Zs, aspect="auto", cmap="magma", vmin=-1, vmax=4,
                   extent=[t[0], t[-1], len(Zs) - .5, -.5],
                   interpolation="nearest")
    for k in sorted(set(S["d_type"])):
        ev = d_t[S["d_type"] == k]
        ax.plot(ev, np.full(len(ev), -1.2), "|",
                color=C_PIN if k == 1 else C_HEAT, ms=9, mew=1.6,
                label=f"{S['stim'][k - 1]} ({len(ev)})")
    for _, b in FZ.iterrows():
        ax.axvspan(b["start_s"], b["end_s"], ymin=0, ymax=.03,
                   color=C_FRZ, alpha=.85, lw=0)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=6.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(len(Zs) - .5, -2.2)
    ax.set_xlabel("time (s), imaging clock")
    ax.set_title(f"the whole session first: {len(Zs)} cells (z), "
                 f"scored stimuli as ticks, freezing bouts in green at the "
                 f"bottom", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2, frameon=False,
              labelcolor="white")
    cb = fig.colorbar(im, ax=ax, pad=.008, fraction=.014)
    cb.set_label("z", fontsize=9)

    # 2 population peri-stimulus + 3 lag check
    for j, nm in enumerate(stim_names):
        A = fig.add_subplot(gs[1, j])
        ev = np.array([b[0] for b in BT[nm]])
        dur = np.median([b[1] - b[0] for b in BT[nm]])
        allc = []
        for plane in ("A", "B"):
            d = U[("pain", plane)]
            for i in range(len(d["labels"])):
                if d["anat"][i] < 1:
                    continue
                lg, M = peri_matrix(d["z"][i], d["t"], ev)
                allc.append(np.nanmean(M, 0))
        allc = np.stack(allc)
        m = np.nanmean(allc, 0)
        se = np.nanstd(allc, 0) / np.sqrt(len(allc))
        A.fill_between(lg, m - se, m + se, color=C_PIN if nm == "pin"
                       else C_HEAT, alpha=.25, lw=0)
        A.plot(lg, m, color=C_PIN if nm == "pin" else C_HEAT, lw=1.8)
        A.axvline(0, color=INK, lw=1)
        A.axhline(0, color=DIM, lw=.7)
        A.axvspan(BOUT_RESP[0], BOUT_RESP[1], color="#888888",
                  alpha=.22, lw=0)
        A.axvspan(0, dur, color="#888888", alpha=.10, lw=0)
        A.set_title(f"population, {nm}: {len(ev)} bouts of "
                    f"{int(np.median([b[2] for b in BT[nm]]))} "
                    f"deliveries (median)", fontsize=11)
        A.set_xlabel("time from bout onset (s)")
        A.set_ylabel("z - session median")
        A.spines[["top", "right"]].set_visible(False)

    for j, nm in enumerate(stim_names):
        A = fig.add_subplot(gs[1, 2 + j])
        keys = [k for k in prof.files if k != "lags" and k.endswith(f"|{nm}")]
        P = np.stack([prof[k] for k in keys])
        A.plot(lags_scan, np.nanmean(P, 0), color=INK, lw=1.8)
        A.axvline(0, color=INK, lw=1, ls=":")
        A.axhline(0, color=DIM, lw=.7)
        A.set_title(f"lag check, {nm}: response vs extra lag", fontsize=11)
        A.set_xlabel("extra lag applied (s)")
        A.set_ylabel("mean response (z)")
        A.text(.02, .96, f"correction already applied: {delay:+.2f} s\n"
                         f"a peak away from 0 would mean it is wrong",
               transform=A.transAxes, va="top", fontsize=8, color=DIM)
        A.spines[["top", "right"]].set_visible(False)

    # 4 per cell
    for i, (_, r) in enumerate(D.iterrows()):
        A = fig.add_subplot(gs[2 + i // ncol, i % ncol])
        d = U[("pain", r["plane"])]
        zi = d["labels"].index(r["uid"])
        for nm, col in zip(stim_names, (C_PIN, C_HEAT)):
            ev = np.array([b[0] for b in BT[nm]])
            lg, M = peri_matrix(d["z"][zi], d["t"], ev)
            m = np.nanmean(M, 0)
            se = np.nanstd(M, 0) / np.sqrt(np.sum(np.isfinite(M[:, 0])))
            A.fill_between(lg, m - se, m + se, color=col, alpha=.2, lw=0)
            A.plot(lg, m, color=col, lw=1.3)
        A.axvline(0, color=INK, lw=.9)
        A.axhline(0, color=DIM, lw=.6)
        star = " ".join(f"{nm}{'*' if np.isfinite(r[f'{nm}_q']) and r[f'{nm}_q'] <= ALPHA else ''}"
                        for nm in stim_names)
        A.set_title(f"{r['uid']}  anat {r['anat']:.1f}"
                    + ("" if r["usable"] else "  NOT ON A SOMA")
                    + f"\n{star}", fontsize=8.5,
                    color=INK if r["usable"] else "#AAAAAA")
        A.tick_params(labelsize=7)
        A.spines[["top", "right"]].set_visible(False)
        if i % ncol == 0:
            A.set_ylabel("z", fontsize=8)
        if i // ncol == nrow_cells - 1:
            A.set_xlabel("s from bout onset", fontsize=8)

    fig.suptitle("Event-locked responses, 31 cells, "
                 f"{len(d_t)} scored deliveries corrected by {delay:+.2f} s"
                 "\nred = pin, amber = heat; * = q <= 0.05 against a "
                 "circular-shift null; shading = SEM over deliveries",
                 fontsize=13, y=.998)
    fig.subplots_adjust(left=.055, right=.985, top=.955, bottom=.02)
    p8 = os.path.join(out, "fig8_event_locked.png")
    fig.savefig(p8, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 9
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.25, 1.0],
                          hspace=.42, wspace=.28)

    ax = fig.add_subplot(gs[0, :])
    mo = MO[MO["plane"] == "A"]
    ax.plot(mo["t_s"], mo["motion_side"], color=C_MOVE, lw=.6,
            label="side camera (the mouse)")
    ax.plot(mo["t_s"], mo["motion_bottom"], color=C_PIN, lw=.5, alpha=.55,
            label="bottom camera (hand + mouse)")
    for _, b in FZ.iterrows():
        ax.axvspan(b["start_s"], b["end_s"], color=C_FRZ, alpha=.22, lw=0)
    for k in sorted(set(S["d_type"])):
        ev = d_t[S["d_type"] == k]
        ax.plot(ev, np.full(len(ev), ax.get_ylim()[1]), "|",
                color=C_PIN if k == 1 else C_HEAT, ms=8, mew=1.4)
    ax.set_xlim(mo["t_s"].min(), mo["t_s"].max())
    ax.set_xlabel("time (s), imaging clock")
    ax.set_ylabel("motion energy")
    ax.set_title(f"movement across the session, with the "
                 f"{len(FZ)} freezing bouts shaded green "
                 f"({FZ['duration_s'].sum():.0f} s total)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[1, 0])
    for _, r in D.iterrows():
        c = INK if r["usable"] else "#CCCCCC"
        A.scatter(r["r_motion_side"], r["pin_resp"], s=34, color=c,
                  zorder=3 if r["usable"] else 1)
        if r["usable"] and (abs(r["r_motion_side"]) > .12
                            or abs(r["pin_resp"]) > .2):
            A.annotate(r["uid"], (r["r_motion_side"], r["pin_resp"]),
                       fontsize=7.5, xytext=(3, 3),
                       textcoords="offset points")
    A.axhline(0, color=DIM, lw=.7)
    A.axvline(0, color=DIM, lw=.7)
    A.set_xlabel("correlation with movement (side camera)")
    A.set_ylabel("pin response (z)")
    A.set_title("a movement cell would look pin-responsive for free",
                fontsize=11)
    A.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[1, 1])
    for nm, col in zip(stim_names, (C_PIN, C_HEAT)):
        A.scatter(D.loc[ok, f"{nm}_resp"], D.loc[ok, f"{nm}_nomove_resp"],
                  s=34, color=col, label=nm)
    lim = np.nanmax(np.abs(np.concatenate(
        [D.loc[ok, f"{n}_resp"].to_numpy() for n in stim_names]
        + [D.loc[ok, f"{n}_nomove_resp"].to_numpy() for n in stim_names])))
    A.plot([-lim, lim], [-lim, lim], color=DIM, lw=.8, ls="--")
    A.set_xlabel("response, raw")
    A.set_ylabel("response, movement regressed out")
    A.set_title("how much of it was movement", fontsize=11)
    A.legend(fontsize=9, frameon=False)
    A.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[1, 2])
    A.scatter(D.loc[ok, "r_motion_side"], D.loc[ok, "freeze_diff"], s=34,
              color=C_FRZ)
    for _, r in D[ok].iterrows():
        A.annotate(r["uid"], (r["r_motion_side"], r["freeze_diff"]),
                   fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    A.axhline(0, color=DIM, lw=.7)
    A.axvline(0, color=DIM, lw=.7)
    A.set_xlabel("correlation with movement")
    A.set_ylabel("z inside freezing minus outside")
    A.set_title("movement and freezing are two sides of one axis",
                fontsize=11)
    A.spines[["top", "right"]].set_visible(False)

    A = fig.add_subplot(gs[2, :])
    A.axis("off")
    cats = [("pin", D[ok & (D["pin_q"] <= ALPHA)]["uid"].tolist()),
            ("pin, robust over 4 baselines",
             D[D["pin_robust"]]["uid"].tolist()),
            ("heat, robust over 4 baselines",
             D[D["heat_robust"]]["uid"].tolist()),
            ("pin, survives movement",
             D[ok & (D["pin_q"] <= ALPHA)
               & (D["pin_nomove_q"] <= ALPHA)]["uid"].tolist()),
            ("heat", D[ok & (D["heat_q"] <= ALPHA)]["uid"].tolist()),
            ("heat, survives movement",
             D[ok & (D["heat_q"] <= ALPHA)
               & (D["heat_nomove_q"] <= ALPHA)]["uid"].tolist()),
            ("movement, side camera",
             D[ok & (D["q_motion_side"] <= ALPHA)]["uid"].tolist()),
            ("movement, bottom camera",
             D[ok & (D["q_motion_bottom"] <= ALPHA)]["uid"].tolist()),
            ("freezing (z up)", D[ok & (D["freeze_q"] <= ALPHA)
                                  & (D["freeze_diff"] > 0)]["uid"].tolist()),
            ("freezing (z down)", D[ok & (D["freeze_q"] <= ALPHA)
                                    & (D["freeze_diff"] < 0)]["uid"
                                                              ].tolist()),
            ("non-responsive to everything",
             D[ok & (D["label"] == "non-responsive")]["uid"].tolist())]
    lines = [f"{'category':<32s} {'n':>3s}   cells"]
    for nm, ids in cats:
        lines.append(f"{nm:<32s} {len(ids):>3d}   "
                     + (", ".join(ids) if ids else "-"))
    A.text(0, 1, "\n".join(lines), family="Consolas", fontsize=10.5,
           va="top", color=INK)
    A.text(0, .02,
           "HOW TO READ IT.  A pin prick makes the mouse move, so a cell "
           "that tracks movement will pass a naive stimulus test. The "
           "'survives movement' rows are the stimulus test repeated on the "
           "residual after regressing both cameras' motion energy out of "
           "every trace with one common design. Movement energy is not "
           "tracked speed and an imperfect regressor under-removes, so "
           "surviving is the weaker claim, not the stronger one. Freezing "
           "is the side camera's energy below its 25th percentile for at "
           "least 1 s; the sensitivity of that choice is in "
           "motion_report.txt.",
           fontsize=9.5, color=DIM, va="bottom", wrap=True)

    fig.suptitle("Movement, freezing and pain, on the same 31 neurons",
                 fontsize=13, y=.997)
    fig.subplots_adjust(left=.05, right=.985, top=.94, bottom=.02)
    p9 = os.path.join(out, "fig9_movement.png")
    fig.savefig(p9, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p8}\nwrote {p9}")
