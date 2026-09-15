"""pain_audit.py  -  every number the pain slides cite, recomputed on disk.

The final pain classification rests on nine measurements about the SESSION,
not about the neurons: how close the deliveries are, where the population
average peaks, whether a pre-window can be clean, whether a jitter null can
be built, and whether pin and heat can be told apart at all. Those numbers
decided which questions were dropped, so they are the numbers a reader will
want to check first.

They were computed once during the audit and quoted in the deck from
memory. That is exactly the failure mode this pipeline keeps hitting - a
figure or a slide carrying a number no file can confirm - so they are
recomputed here and written to pain_audit.txt, and the slides cite that
file.

WHAT IS MEASURED
  1  inter-delivery gaps, and the fraction of trials whose pre-window
     already contains another delivery, for four candidate pre-windows
  2  the population peri-event average, lag by lag, around the key press
  3  the same test with a [-2,0] and a [-6,-4] baseline - the sign flip
  4  the jitter null: how often a jittered event lands on a real one
  5  the pin and heat block boundaries, and the overlap between them
  6  Jaccard overlap between the pin, withdrawal and flinch UP lists
  7  the global stimulus-free mask: its mean time against the deliveries',
     and how much of the dense pin block it covers
  8  how many deliveries have a genuinely clear pre-window, and what
     effect size is detectable with only those
  9  scoring coverage per behaviour

OUTPUT  ->  <pain session>\\output_split\\events\\pain_audit.txt , .csv

USAGE
  python pain_audit.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pain_events import cell_seed, shifts, surr_p
from pain_responsive import collect_events, free_mask
from tap_delay import frame_to_imaging, load_scoring
from union_data import ANAT_MIN, PLANES, ROOTS, load_union

PA_SES = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
          "15-56-48")
PRE_WINDOWS = [(-2., 0.), (-4., -2.), (-6., -4.), (-8., -6.)]
POST = (0., 2.)
CLEAR = 6.0        # "a clear pre-window" means this many seconds of it
N_PERM = 2000
L = []


def say(s=""):
    print(s)
    L.append(s)


def head(n, t):
    say()
    say(f"----- {n}  {t}")


def traces(U):
    """z per interpretable cell, and the shared time base."""
    c = []
    for plane in PLANES:
        d = U[("pain", plane)]
        for i, lab in enumerate(d["labels"]):
            if d["anat"][i] >= ANAT_MIN:
                c.append((plane, lab, d["z"][i], d["fs"]))
    return c, U[("pain", "A")]["t"]


def main():
    out = os.path.join(ROOTS["pain"], "events")
    os.makedirs(out, exist_ok=True)
    S = load_scoring(PA_SES)
    tb = frame_to_imaging(PA_SES, "MiceVideo1")
    EV, KIND = collect_events(S, tb)
    U = load_union()
    cells, t = traces(U)
    d_t = np.sort(tb[np.clip(S["d_frame"], 1, len(tb)) - 1])
    is_pin = S["d_type"] == 1
    pin = np.sort(tb[np.clip(S["d_frame"][is_pin], 1, len(tb)) - 1])
    heat = np.sort(tb[np.clip(S["d_frame"][~is_pin], 1, len(tb)) - 1])
    rows = []

    say("===== pain audit - the session numbers the slides cite =====")
    say()
    say(f"{len(cells)} interpretable cells, {len(d_t)} deliveries "
        f"({len(pin)} pin, {len(heat)} heat), session "
        f"{t[0]:.1f}-{t[-1]:.1f} s")

    # ---------------------------------------------------------------- 1
    head(1, "how close the deliveries are, and whether a pre-window "
            "can be clean")
    for nm, ev in (("pin", pin), ("heat", heat), ("all", d_t)):
        g = np.diff(ev)
        say(f"  {nm:<5s} n={len(ev):>3d}  gap median {np.median(g):6.2f} s"
            f"   min {g.min():5.2f}   10th pct {np.percentile(g, 10):5.2f}"
            f"   90th pct {np.percentile(g, 90):6.2f}")
        rows.append(dict(item=f"gap_median_{nm}", value=round(
            float(np.median(g)), 3), unit="s"))
    say()
    say("  fraction of PIN trials whose pre-window already contains "
        "another delivery")
    say("  (any delivery, pin or heat, other than the trial itself)")
    for w in PRE_WINDOWS:
        bad = 0
        for e in pin:
            o = d_t[np.abs(d_t - e) > 1e-9]
            if ((o >= e + w[0]) & (o <= e + w[1])).any():
                bad += 1
        say(f"    [{w[0]:>5.1f}, {w[1]:>5.1f}] s   {bad:>3d} of {len(pin)}"
            f"   {100 * bad / len(pin):5.1f} %")
        rows.append(dict(item=f"pin_prewin_contaminated_{w[0]:g}_{w[1]:g}",
                         value=round(100 * bad / len(pin), 1), unit="%"))
    say()
    say("  NOTE the earlier version of this count used '.sum() > 1', which "
        "requires TWO other")
    say("  deliveries inside the window, and reported 4-6 %. The figures "
        "above are the right ones.")

    # ---------------------------------------------------------------- 2
    head(2, "where the population average peaks, relative to the key press")
    lags = np.arange(-8, 8.001, 1.0)
    M = []
    for _, _, z, _ in cells:
        w = np.stack([np.interp(e + lags, t, z, left=np.nan, right=np.nan)
                      for e in d_t])
        M.append(np.nanmean(w, 0))
    M = np.stack(M)
    pop = np.nanmean(M, 0)
    say("  mean z over all cells and all deliveries, no baseline "
        "subtracted anywhere:")
    for lg, v in zip(lags, pop):
        star = "  <-- peak" if abs(v - pop.max()) < 1e-12 else ""
        say(f"    {lg:>5.1f} s   {v:+6.3f}{star}")
    pk = float(lags[int(np.nanargmax(pop))])
    say(f"  PEAK AT {pk:+.0f} s. A pre-window baseline therefore starts the "
        f"measurement at the top")
    say(f"  of the response, and forces the post-window negative by "
        f"construction.")
    rows.append(dict(item="population_peak_lag", value=pk, unit="s"))

    # ---------------------------------------------------------------- 3
    head(3, "the same test with two baselines - the sign flip")
    from scipy.stats import wilcoxon
    for w in [(-2., 0.), (-6., -4.)]:
        v = []
        for _, _, z, _ in cells:
            d = []
            for e in pin:
                a = (t >= e + w[0]) & (t <= e + w[1])
                b = (t >= e + POST[0]) & (t <= e + POST[1])
                if a.any() and b.any():
                    d.append(z[b].mean() - z[a].mean())
            v.append(np.mean(d))
        v = np.asarray(v)
        _, p = wilcoxon(v)
        say(f"  baseline [{w[0]:>5.1f},{w[1]:>5.1f}] s  ->  mean over cells "
            f"{v.mean():+.4f} z,  Wilcoxon p = {p:.4g}   "
            f"({'SUPPRESSION' if v.mean() < 0 else 'EXCITATION'})")
        rows.append(dict(item=f"pin_effect_baseline_{w[0]:g}_{w[1]:g}",
                         value=round(float(v.mean()), 4), unit="z"))
        rows.append(dict(item=f"pin_p_baseline_{w[0]:g}_{w[1]:g}",
                         value=float(f"{p:.4g}"), unit="p"))
    say("  Same traces, same events, same post-window. Only the baseline "
        "moved, and the sign")
    say("  of the population result moved with it. That is why no "
        "pre-window version is reported.")

    # ---------------------------------------------------------------- 4
    head(4, "why an event-jitter null cannot be built here")
    rng = np.random.default_rng(20260914)
    hit = tot = 0
    for _ in range(200):
        j = d_t + rng.uniform(3, 15, len(d_t)) * rng.choice([-1, 1],
                                                            len(d_t))
        for e in j:
            tot += 1
            if np.abs(d_t - e).min() <= 1.0:
                hit += 1
    say(f"  jitter of 3-15 s in either direction, 200 draws: "
        f"{100 * hit / tot:.1f} % of jittered events land within 1 s of a "
        f"REAL delivery")
    say("  so the null contains the signal, is too narrow, and everything "
        "passes. The circular")
    say("  shift of the cell's own trace is used instead - it keeps the "
        "event times fixed.")
    rows.append(dict(item="jitter_lands_on_real_event",
                     value=round(100 * hit / tot, 1), unit="%"))

    # ---------------------------------------------------------------- 5
    head(5, "pin and heat were given in separate blocks")
    say(f"  pin   {pin.min():6.1f} - {pin.max():6.1f} s   "
        f"({len(pin)} deliveries)")
    say(f"  heat  {heat.min():6.1f} - {heat.max():6.1f} s   "
        f"({len(heat)} deliveries)")
    inside = int(((pin >= heat.min()) & (pin <= heat.max())).sum())
    say(f"  pin deliveries inside the heat block "
        f"[{heat.min():.0f}, {heat.max():.0f}] s:  {inside}")
    # the pin block, described as the two runs it actually is
    g = np.diff(pin)
    brk = np.nonzero(g > 60)[0]
    seg = np.split(pin, brk + 1)
    say("  pin, as the runs it actually is: "
        + ";  ".join(f"{s.min():.0f}-{s.max():.0f} s (n={len(s)})"
                     for s in seg))
    say(f"  So 'pin versus heat' is also 'early versus middle'. Permuting "
        f"the {len(pin)}/{len(heat)} labels")
    say("  over the same delivery times mixes the blocks and cannot undo a "
        "confound with time.")
    rows.append(dict(item="pin_inside_heat_block", value=inside,
                     unit="deliveries"))

    # ---------------------------------------------------------------- 6
    head(6, "the pin, withdrawal and flinch UP lists are the same list")
    R = pd.read_csv(os.path.join(out, "responsive_cells.csv"))
    from results_figures import dirs_of
    _, g20 = dirs_of(R, "q20")
    sets = {}
    for e in ("pin", "heat", "paw_withdrawal", "flinch", "pin_epoch",
              "heat_epoch"):
        sets[e] = {u for u, d in g20.items() if d.get(e) == "UP"}
    ks = [k for k in sets if sets[k]]
    say("  UP lists at the exploratory q <= 0.20 (the per-delivery test, "
        "now withdrawn):")
    for k in ks:
        say(f"    {k:<16s} {len(sets[k]):>2d}  "
            f"{', '.join(sorted(sets[k])) or '-'}")
    say()
    say("  Jaccard overlap between them  (|A n B| / |A u B|):")
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            u = sets[a] | sets[b]
            j = len(sets[a] & sets[b]) / len(u) if u else np.nan
            if j >= .5:
                say(f"    {a:<16s} vs {b:<16s} {j:.2f}")
                rows.append(dict(item=f"jaccard_{a}_{b}", value=round(j, 2),
                                 unit=""))
    say("  These events sit inside the same epochs, so one finding was "
        "being counted several times.")

    # ---------------------------------------------------------------- 7
    head(7, "the global stimulus-free mask is biased late")
    free = free_mask(t, d_t)
    say(f"  free frames: {100 * free.mean():.1f} % of the session")
    say(f"  mean time of the free frames   {t[free].mean():7.1f} s")
    say(f"  mean time of the deliveries    {d_t.mean():7.1f} s")
    say(f"  the reference therefore sits {t[free].mean() - d_t.mean():+.0f} "
        f"s later than what it is a reference for")
    blk = (t >= pin.min()) & (t <= 204.0)
    say(f"  inside the dense pin block [{pin.min():.0f}, 204] s, only "
        f"{100 * free[blk].mean():.1f} % of frames are free")
    say("  A cell that drifts up over ten minutes is therefore scored "
        "against mostly-late quiet time,")
    say("  which looks like suppression during stimulation. TIER 1 uses "
        "the free frames within")
    say("  +-60 s of each delivery instead, so the drift cancels.")
    rows.append(dict(item="free_mask_fraction",
                     value=round(100 * float(free.mean()), 1), unit="%"))
    rows.append(dict(item="free_mask_time_offset",
                     value=round(float(t[free].mean() - d_t.mean()), 1),
                     unit="s"))
    rows.append(dict(item="free_in_pin_block",
                     value=round(100 * float(free[blk].mean()), 1),
                     unit="%"))

    # ---------------------------------------------------------------- 8
    head(8, "how many deliveries have a genuinely clear pre-window")
    for nm, ev in (("pin", pin), ("heat", heat)):
        ok = [e for e in ev
              if not ((d_t >= e - CLEAR) & (d_t < e - 1e-9)).any()]
        say(f"  {nm:<5s} {len(ok):>3d} of {len(ev):>3d} have {CLEAR:.0f} s "
            f"with no other delivery before them")
        rows.append(dict(item=f"{nm}_with_clear_prewindow", value=len(ok),
                         unit=f"of {len(ev)}"))
    ok = [e for e in pin
          if not ((d_t >= e - CLEAR) & (d_t < e - 1e-9)).any()]
    # what effect size is detectable with only those trials: the width of
    # the circular-shift null for the mean over them, x1.96
    det = []
    for plane, lab, z, fs in cells:
        rg = np.random.default_rng(cell_seed(plane, lab, "audit"))
        sh = shifts(len(z), fs, rg)

        def stat(y):
            v = [y[(t >= e + POST[0]) & (t <= e + POST[1])].mean()
                 - y[(t >= e - CLEAR) & (t <= e - CLEAR + 2)].mean()
                 for e in ok]
            return float(np.mean(v))
        o = stat(z)
        _, sd = surr_p(o, stat, z, sh)
        if np.isfinite(sd):
            det.append(1.96 * sd)
    say(f"  with only those {len(ok)} pin trials, the smallest effect a "
        f"cell could show and still")
    say(f"  clear its own shift null is {np.median(det):.2f} z (median over "
        f"cells; 1.96 x null SD).")
    obs = pd.read_csv(os.path.join(out, "pain_final.csv"))
    say(f"  the largest effect anything actually shows is "
        f"{obs['t1_any'].abs().max():.2f} z, so this subset cannot resolve "
        f"real responses.")
    rows.append(dict(item="detectable_effect_clear_trials",
                     value=round(float(np.median(det)), 2), unit="z"))

    # ---------------------------------------------------------------- 9
    head(9, "scoring coverage - what was actually scored, and for how long")
    fr_s = 1.0 / 25.0
    say(f"  {'behaviour':<22s} {'episodes':>8s} {'frames':>8s} "
        f"{'seconds':>8s}")
    for k in range(1, len(S["aff"])):
        m = S["score"] == k
        if not m.any():
            say(f"  {S['aff'][k]:<22s} {0:>8d} {0:>8d} {0.0:>8.1f}")
            continue
        d = np.diff(m.astype(int))
        n_ep = int((d == 1).sum()) + int(m[0])
        say(f"  {S['aff'][k]:<22s} {n_ep:>8d} {int(m.sum()):>8d} "
            f"{m.sum() * fr_s:>8.1f}")
        rows.append(dict(item=f"scored_{S['aff'][k]}", value=n_ep,
                         unit="episodes"))
    say(f"  reflex taps: {len(EV.get('paw_withdrawal', [])):>3d} paw "
        f"withdrawal, {len(EV.get('flinch', [])):>3d} flinch - these are "
        f"instants, not episodes")
    say("  The held behaviours were tapped rather than held: licking and "
        "guarding were never")
    say("  scored at all, attending once. The reflex taps are the only "
        "substantial behaviour data.")

    say()
    say("===== what each number decided =====")
    say("  1, 8  ->  no pre-stimulus window is clean; per-delivery pin and "
        "heat responses WITHDRAWN")
    say("  2, 3  ->  a pre-window baseline inverts the population result; "
        "the AUC suppression WITHDRAWN")
    say("  4     ->  the jitter null is invalid; the circular shift is used "
        "instead")
    say("  5     ->  pin versus heat is confounded with time; "
        "modality-selectivity WITHDRAWN")
    say("  6     ->  pin, withdrawal and flinch are one finding, not four")
    say("  7     ->  the reference must be local; TIER 1 uses +-60 s "
        "around each delivery")
    say("  9     ->  only escape is testable among the held behaviours, "
        "and only as a hint")

    txt = "\n".join(L)
    with open(os.path.join(out, "pain_audit.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(out, "pain_audit.csv"),
                              index=False)
    print(f"\nwrote {os.path.join(out, 'pain_audit.txt')}")
    return rows


if __name__ == "__main__":
    main()
