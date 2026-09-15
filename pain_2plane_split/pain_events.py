"""pain_events.py  -  which of the 31 union neurons respond to what.

Four questions, one table. Every row is one of the union neurons, so its
pain answer sits next to its open-field answer.

  1  STIMULUS      pin prick, heat: mean z after the stimulus minus before
  2  BEHAVIOUR     the 2 reflex taps as events, the 4 affective states as
                   episodes (inside versus outside)
  3  MOVEMENT      correlation with camera motion energy, and mean z inside
                   freezing bouts versus outside
  4  IS IT REALLY THE STIMULUS?  the stimulus test again on the residual
                   after regressing movement out. A pin prick makes the
                   mouse move, so a movement cell looks stimulus-responsive
                   for free; only the cells that survive this are being
                   called pain-responsive.

TIMING
  Scored taps are late by a human reaction time, multiplied by the playback
  speed. tap_delay.py measures that from the video (not from these neurons)
  and this script subtracts it. The lag profile printed here - response
  against lag from -2 to +3 s - is a CHECK on that number. It is not used to
  choose the lag: maximising the neural response over lag and then testing
  significance at the winner would manufacture a result.

STATISTICS
  One null for everything: circular shift of the cell's own trace, at least
  MIN_SHIFT_S, 2000 surrogates. It keeps the trace's autocorrelation and the
  event times exactly as they are and destroys only the pairing between
  them. A t-test over trials would treat neighbouring imaging frames as
  independent, which at 4.6 Hz with a ~1 s calcium decay they are not.
  q-values are Benjamini-Hochberg within each family of tests.

  Cells whose footprint did not land on a soma in the pain session
  (anat < 1 image SD) are reported but excluded from the FDR correction and
  from every count, because their trace is background.

OUTPUT  ->  <pain session>\\output_split\\events\\
  event_cells.csv     one row per union neuron, every column below
  event_report.txt
  fig8_event_locked.png    peri-stimulus averages, per cell
  fig9_movement.png        movement, freezing and pain, side by side

USAGE
  python pain_motion.py && python tap_delay.py && python pain_events.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from tap_delay import frame_to_imaging, load_scoring
from union_data import ANAT_MIN, PLANES, ROOTS, load_union

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
PRE = (-3.0, -1.0)          # baseline window, s from a single delivery
POST = (0.5, 3.0)           # response window for a single delivery
BOUT_GAP = 5.0              # deliveries closer than this are one bout
BOUT_PRE = (-6.0, -1.0)     # pre-bout window; used for bout SELECTION, and
                            # as a sensitivity check only - see bout_stat
BOUT_RESP = (0.5, 3.0)      # primary response window, from bout onset
# (a second window tuple used to live here; see bout_stat for why not)
LAG_SCAN = np.arange(-2.0, 3.001, 0.25)
N_SURR = 2000
MIN_SHIFT_S = 10.0
ALPHA = 0.05


def bh(p):
    """Benjamini-Hochberg q-values; NaN in, NaN out."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    v = p[ok]
    if not len(v):
        return q
    o = np.argsort(v)
    n = len(v)
    r = np.empty(n)
    r[o] = np.minimum.accumulate((v[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q[ok] = np.minimum(r, 1.0)
    return q


def cell_seed(plane, label, what):
    """Reproducible per-cell seed. hash() is salted per process."""
    import hashlib
    h = hashlib.md5(f"{plane}|{label}|{what}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def shifts(n, fs, rng):
    lo = int(round(MIN_SHIFT_S * fs))
    return rng.integers(lo, n - lo, N_SURR)


def event_stat(x, t, ev, lag=0.0):
    """Mean of x in the post window minus the pre window, over events."""
    if not len(ev):
        return np.nan
    v = []
    for e in ev:
        a = (t >= e + lag + PRE[0]) & (t <= e + lag + PRE[1])
        b = (t >= e + lag + POST[0]) & (t <= e + lag + POST[1])
        if a.sum() and b.sum():
            v.append(x[b].mean() - x[a].mean())
    return float(np.mean(v)) if v else np.nan


def find_bouts(ev, others, gap=BOUT_GAP, pre=BOUT_PRE):
    """Group deliveries into bouts and keep those with a clean baseline.

    THIS IS WHY THE PER-DELIVERY TEST DOES NOT WORK HERE. The deliveries are
    not isolated trials: the median gap between pin pricks is 3.2 s and 40 of
    52 gaps are under 6 s, so a [-3, -1] s baseline usually sits inside the
    previous delivery's response and subtracts the signal away. Only 2 pin
    and 1 heat deliveries in this session are isolated on both sides.

    A bout is a run of deliveries separated by less than `gap`. Its baseline
    is the window before its onset, and a bout is dropped unless that window
    contains no delivery of EITHER type - which is what `others` is for.
    """
    ev = np.sort(np.asarray(ev, float))
    if not len(ev):
        return []
    st = np.concatenate([[0], np.flatnonzero(np.diff(ev) >= gap) + 1])
    en = np.concatenate([st[1:] - 1, [len(ev) - 1]])
    out = []
    for a, b in zip(st, en):
        on, off = ev[a], ev[b]
        if ((others >= on + pre[0]) & (others <= on + pre[1])).any():
            continue
        out.append((on, off, int(b - a + 1)))
    return out


def bout_stat(x, t, bts, base="median", to_end=False, lag=0.0):
    """Mean z in a window after bout onset, minus a baseline.

    base="median"  the cell's own session median. THIS IS THE PRIMARY.
    base=(lo, hi)  a window before bout onset. Reported as a sensitivity
                   check only, because it is not neutral here - see below.

    WHY NOT A PRE-STIMULUS BASELINE, WHICH IS THE OBVIOUS CHOICE.
    A bout is kept only if the 5 s before it contains no delivery. With
    deliveries every ~3 s, that condition SELECTS unusually quiet stretches:
    the [-6, -1] s window is delivery-free in 12 of 12 pin bouts by
    construction, while [-12, -7] is clean in only 6 of 12. Measuring the
    response against a selected quiet period inflates it, and the inflation
    is what the answer turns on - with that baseline 6 pin and 5 heat cells
    pass, with [-12, -7] it is 0 and 1, with [-20, -15] it is 2 and 0. A
    real stimulus response does not appear and disappear according to which
    5 s of pre-stimulus you subtract.

    The session median is immune: it uses no pre-window, so no selection
    enters. The cost is that slow drift is no longer removed per trial,
    which the circular-shift null handles because it preserves that drift.

    WHY A SHORT WINDOW. to_end=True runs to the end of the bout, up to 22 s,
    which dilutes a 2-3 s calcium transient: B2's heat response reads 1.23
    that way and 2.99 measured over 0.5-3 s after onset, which is where a
    GCaMP response to a stimulus is.

    to_end is a flag and not a second window tuple on purpose. It was a
    tuple, tested with `win is BOUT_LONG`, and BOUT_RESP and BOUT_LONG were
    both (0.5, 3.0) - which Python interns to ONE object, so the test was
    always true and every "short window" result was silently the long one.
    """
    if not bts:
        return np.nan
    med = np.median(x)
    v = []
    for on, off, _ in bts:
        hi = (off if to_end else on) + BOUT_RESP[1] + lag
        b = (t >= on + BOUT_RESP[0] + lag) & (t <= hi)
        if not b.sum():
            continue
        if base == "median":
            v.append(x[b].mean() - med)
        else:
            a = (t >= on + base[0] + lag) & (t <= on + base[1] + lag)
            if not a.sum():
                continue
            v.append(x[b].mean() - x[a].mean())
    return float(np.mean(v)) if v else np.nan


def mask_stat(x, m):
    """Mean inside minus mean outside a boolean mask."""
    if m.sum() < 3 or (~m).sum() < 3:
        return np.nan
    return float(x[m].mean() - x[~m].mean())


def surr_p(obs, fn, x, n_shift, two_sided=True):
    """p from the circular-shift null of the same statistic."""
    if not np.isfinite(obs):
        return np.nan, np.nan
    null = np.array([fn(np.roll(x, s)) for s in n_shift])
    null = null[np.isfinite(null)]
    if not len(null):
        return np.nan, np.nan
    if two_sided:
        p = (1. + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1.)
    else:
        p = (1. + np.sum(null >= obs)) / (len(null) + 1.)
    return float(p), float(np.std(null))


def episodes(score, t_cam, code, t_img):
    """Boolean mask on the imaging frames for one affective state."""
    m = (score == code)
    if not m.any():
        return np.zeros(len(t_img), bool)
    return np.interp(t_img, t_cam[:len(m)], m.astype(float)) > 0.5


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
    dd = pd.read_csv(os.path.join(out, "tap_delay.csv"))
    if "used" not in dd.columns:
        raise SystemExit("tap_delay.csv predates the camera fix - re-run "
                         "tap_delay.py")
    delay = float(dd.loc[dd["used"].astype(bool),
                         "delay_estimate_s"].mean())
    d_t = tb[S["d_frame"] - 1] - delay
    r_t = tb[np.clip(S["reflex"][:, 0], 1, len(tb)) - 1] - delay
    r_type = S["reflex"][:, 1]

    MO = pd.read_csv(os.path.join(root, "motion", "motion_on_imaging.csv"))
    FZ = pd.read_csv(os.path.join(root, "motion", "freeze_bouts.csv"))
    U = load_union()

    stim_codes = sorted(set(S["d_type"]))
    ref_codes = sorted(set(r_type))
    aff_codes = [c for c in range(1, len(S["aff"]))
                 if (S["score"] == c).any()]
    BT = {S["stim"][k - 1]: find_bouts(d_t[S["d_type"] == k], d_t)
          for k in stim_codes}

    rows, lagprof, peri_store = [], {}, {}
    for plane in PLANES:
        d = U[("pain", plane)]
        t = d["t"]
        Z = d["z"]
        fs = d["fs"]
        mo = MO[MO["plane"] == plane]
        mside = np.interp(t, mo["t_s"], mo["motion_side"])
        mbot = np.interp(t, mo["t_s"], mo["motion_bottom"])
        frz = np.zeros(len(t), bool)
        for _, b in FZ.iterrows():
            frz |= (t >= b["start_s"]) & (t <= b["end_s"])

        # movement regressed out, once per plane: the same design for
        # every cell, so a cell cannot be helped or hurt by its own fit
        Xm = np.column_stack([np.ones(len(t)), mside, mbot])
        Pm = Xm @ np.linalg.pinv(Xm)

        for i, lab in enumerate(d["labels"]):
            z = Z[i]
            zr = z - Pm @ z
            rng = np.random.default_rng(cell_seed(plane, lab, "events"))
            sh = shifts(len(z), fs, rng)
            r = dict(plane=plane, uid=lab, anat=round(float(d["anat"][i]), 2),
                     usable=int(d["anat"][i] >= ANAT_MIN),
                     verdict=d["verdict"][i], source=d["source"][i],
                     matched=bool(d["matched"][i]))

            for k in stim_codes:
                nm = S["stim"][k - 1]
                ev = d_t[S["d_type"] == k]
                bts = BT[nm]
                # PRIMARY: bout onset, short window, session-median
                # baseline. The three alternatives below are recorded so
                # the sensitivity of the answer is in the table, not just
                # in my head - they disagree, and bout_stat says why.
                o = bout_stat(z, t, bts)
                p, sd = surr_p(o, lambda y: bout_stat(y, t, bts), z, sh)
                r[f"{nm}_resp"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"{nm}_p"] = round(p, 4) if np.isfinite(p) else np.nan
                r[f"{nm}_nullsd"] = round(sd, 4) if np.isfinite(sd) else np.nan
                r[f"{nm}_n"] = len(bts)
                r[f"{nm}_n_deliveries"] = len(ev)
                for tag, kw in (("prebase", dict(base=BOUT_PRE)),
                                ("pre12", dict(base=(-12.0, -7.0))),
                                ("pre20", dict(base=(-20.0, -15.0))),
                                ("long", dict(to_end=True))):
                    ov = bout_stat(z, t, bts, **kw)
                    pv, _ = surr_p(ov, lambda y: bout_stat(y, t, bts, **kw),
                                   z, sh)
                    r[f"{nm}_{tag}_resp"] = round(ov, 4) if np.isfinite(ov) \
                        else np.nan
                    r[f"{nm}_{tag}_p"] = round(pv, 4) if np.isfinite(pv) \
                        else np.nan
                os_ = event_stat(z, t, ev)
                ps_, _ = surr_p(os_, lambda y: event_stat(y, t, ev), z, sh)
                r[f"{nm}_single_resp"] = round(os_, 4) if np.isfinite(os_) \
                    else np.nan
                r[f"{nm}_single_p"] = round(ps_, 4) if np.isfinite(ps_) \
                    else np.nan
                # again on the movement residual
                o2 = bout_stat(zr, t, bts)
                p2, _ = surr_p(o2, lambda y: bout_stat(y, t, bts), zr, sh)
                # named <stim>_nomove_p, not <stim>_p_nomove: the FDR loop
                # below finds families by the "_p" suffix, and the other
                # spelling silently skipped this one
                r[f"{nm}_nomove_resp"] = round(o2, 4) if np.isfinite(o2) \
                    else np.nan
                r[f"{nm}_nomove_p"] = round(p2, 4) if np.isfinite(p2) \
                    else np.nan
                lagprof[(plane, lab, nm)] = np.array(
                    [bout_stat(z, t, bts, lag=L) for L in LAG_SCAN])
                peri_store[(plane, lab, nm)] = ev

            for k in ref_codes:
                nm = S["ref"][k - 1].split("/")[0].strip().lower(
                ).replace(" ", "_")
                ev = r_t[r_type == k]
                o = event_stat(z, t, ev)
                p, _ = surr_p(o, lambda y: event_stat(y, t, ev), z, sh)
                r[f"{nm}_resp"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"{nm}_p"] = round(p, 4) if np.isfinite(p) else np.nan
                r[f"{nm}_n"] = len(ev)

            for k in aff_codes:
                nm = S["aff"][k].split("/")[0].strip().lower().replace(
                    " ", "_")
                m = episodes(S["score"], tb, k, t)
                o = mask_stat(z, m)
                p, _ = surr_p(o, lambda y: mask_stat(y, m), z, sh)
                r[f"{nm}_diff"] = round(o, 4) if np.isfinite(o) else np.nan
                r[f"{nm}_p"] = round(p, 4) if np.isfinite(p) else np.nan
                r[f"{nm}_frac"] = round(float(m.mean()), 4)

            for nm, mv in (("motion_side", mside), ("motion_bottom", mbot)):
                o = float(np.corrcoef(z, mv)[0, 1])
                p, _ = surr_p(o, lambda y: float(np.corrcoef(y, mv)[0, 1]),
                              z, sh)
                r[f"r_{nm}"] = round(o, 4)
                r[f"p_{nm}"] = round(p, 4) if np.isfinite(p) else np.nan
            o = mask_stat(z, frz)
            p, _ = surr_p(o, lambda y: mask_stat(y, frz), z, sh)
            r["freeze_diff"] = round(o, 4) if np.isfinite(o) else np.nan
            r["freeze_p"] = round(p, 4) if np.isfinite(p) else np.nan
            r["freeze_frac"] = round(float(frz.mean()), 4)
            rows.append(r)

    D = pd.DataFrame(rows)
    ok = D["usable"] == 1
    fams = [c[:-2] for c in D.columns if c.endswith("_p")
            and not c.startswith("p_")]
    for f in fams:
        q = np.full(len(D), np.nan)
        q[ok.to_numpy()] = bh(D.loc[ok, f"{f}_p"].to_numpy())
        D[f"{f}_q"] = np.round(q, 4)
    for nm in ("motion_side", "motion_bottom"):
        q = np.full(len(D), np.nan)
        q[ok.to_numpy()] = bh(D.loc[ok, f"p_{nm}"].to_numpy())
        D[f"q_{nm}"] = np.round(q, 4)

    # ---- how many baseline definitions does each cell survive? ------
    # The four differ only in what the response is measured against. A
    # stimulus response should not care; anything that passes under one
    # and not the others is a property of that baseline, not of the cell.
    stim_names = [S["stim"][k - 1] for k in stim_codes]
    BASELINES = ["", "prebase", "pre12", "pre20"]
    for nm in stim_names:
        cols = [f"{nm}_q" if not b else f"{nm}_{b}_q" for b in BASELINES]
        D[f"{nm}_n_baselines"] = sum(
            (D[c] <= ALPHA).fillna(False).astype(int) for c in cols)
        D[f"{nm}_robust"] = (D[f"{nm}_n_baselines"] == len(BASELINES)) \
            & (D["usable"] == 1)

    def hit(r, f, qcol=None):
        q = r.get(qcol or f"{f}_q", np.nan)
        return bool(np.isfinite(q) and q <= ALPHA)

    lab = []
    for _, r in D.iterrows():
        if not r["usable"]:
            lab.append("not interpretable")
            continue
        tags = []
        for nm in stim_names:
            if not hit(r, nm):
                continue
            nb = int(r[f"{nm}_n_baselines"])
            t_ = nm if r[f"{nm}_robust"] else f"{nm} ({nb}/4 baselines)"
            tags.append(t_ if hit(r, f"{nm}_nomove")
                        else f"{nm} (movement)")
        if hit(r, "", "q_motion_side"):
            tags.append("movement")
        if hit(r, "freeze"):
            tags.append("freezing" if r["freeze_diff"] > 0
                        else "freezing (down)")
        for nm in [S["ref"][k - 1].split("/")[0].strip().lower().replace(
                " ", "_") for k in ref_codes]:
            if hit(r, nm):
                tags.append(nm)
        for nm in [S["aff"][k].split("/")[0].strip().lower().replace(" ", "_")
                   for k in aff_codes]:
            if hit(r, nm):
                tags.append(nm)
        lab.append(", ".join(tags) if tags else "non-responsive")
    D["label"] = lab

    # ---- join the open-field answer ---------------------------------
    pl = os.path.join(ROOTS["openfield"], "place", "place_cells.csv")
    if os.path.exists(pl):
        P = pd.read_csv(pl)
        key = {(r["plane"], r["cell"]): (r["preference"], r["contrast"],
                                         r["q_contrast"])
               for _, r in P.iterrows()}
        D["of_place"] = [key.get((r["plane"], r["uid"]),
                                 ("not tested", np.nan, np.nan))[0]
                         for _, r in D.iterrows()]
        D["of_contrast"] = [key.get((r["plane"], r["uid"]),
                                    ("", np.nan, np.nan))[1]
                            for _, r in D.iterrows()]
    D.to_csv(os.path.join(out, "event_cells.csv"), index=False)
    np.savez(os.path.join(out, "lag_profiles.npz"),
             lags=LAG_SCAN,
             **{f"{p}|{u}|{n}": v for (p, u, n), v in lagprof.items()})
    return (D, S, d_t, r_t, r_type, tb, MO, FZ, U, out, delay,
            stim_names, BT)


if __name__ == "__main__":
    import report_events
    report_events.run(*main())
