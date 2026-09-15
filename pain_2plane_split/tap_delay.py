"""tap_delay.py  -  how late is the human's key press? Measured, not assumed.

THE PROBLEM
  The stimulus times come from a person watching the video and tapping 1 or
  2. That tap is late by their reaction time, and the scorer plays the video
  faster than real time, which multiplies the delay in video seconds: a
  300 ms reaction at 4.5x playback lands 1.35 s late on the video's clock.
  The scoring tool records a constant lagMs = 250 ms, but it applies that
  only to its training labels, not to the delivery times, and 250 ms is an
  assumption about a different setup.

  Event-locking on uncorrected taps smears every peri-stimulus average by
  whatever that delay is, and at 4.6 Hz imaging a 1 s error is 5 frames.

TWO MEASUREMENTS, AND ONLY ONE OF THEM ANSWERS THE QUESTION
  BOTTOM camera, peri-tap  <- this is the estimate used
     The stimulus itself is in this view: the experimenter's hand and the
     filament. Its motion onset is the closest thing in the recording to the
     moment of delivery. Measured: onset 0.32 s BEFORE the tap for pin and
     for heat alike, with a tight bootstrap interval, so the key press runs
     about a third of a second behind the stimulus.

  SIDE camera, peri-tap  <- NOT used as a delay, reported as a result
     The first version of this script used the side camera as the primary
     estimate, on the argument that a withdrawal follows a pin prick within
     ~0.1 s and so marks the stimulus. The data said otherwise: side-camera
     motion peaks 1.8 s AFTER the tap for pin, with a bootstrap interval
     spanning 3 s. That is not a broken measurement, it is the wrong
     quantity - whole-body motion energy is dominated by what the mouse does
     NEXT (licking, guarding, moving away), which builds over seconds, not by
     the reflex twitch. Using it would have shifted every stimulus 1.7 s in
     the wrong direction. It is kept in the figure because the behavioural
     response latency is worth knowing.

  A third check uses no video: the reflex taps against the stimulus taps.
  Both carry the same human delay, so their difference cannot measure it -
  but it shows whether the two tap streams are ordered sensibly, and it
  would catch a withdrawal scored before its own stimulus.

  The neurons are deliberately not used. Scanning the lag for the strongest
  neural response and then testing significance at that lag is circular;
  pain_events.py does report the neural lag profile, but it is checked
  against this figure rather than used to set the correction.

WHAT IS RETURNED
  delay_s, how late the tap is, to be SUBTRACTED from the scored times, with
  a bootstrap interval over deliveries and separately for pin and heat since
  a filament and a laser do not look the same from below. Onset is reported
  two ways - of the average curve, and the median of per-delivery onsets -
  because the onset of an average is biased towards the earliest delivery.

OUTPUT  ->  <pain session>\\output_split\\events\\
  tap_delay.csv , tap_delay.txt , fig7_tap_delay.png

USAGE
  python pain_motion.py      # first: it writes the motion energy
  python tap_delay.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
LAGS = np.arange(-3.0, 3.001, 0.04)     # s, relative to the tap
BASE = (-3.0, -2.0)                     # baseline window for the peri-average
N_BOOT = 2000
RNG = np.random.default_rng(0)


def load_scoring(session):
    """Deliveries and reflex taps, as BOTTOM-camera frame indices."""
    sc = os.path.join(session, "output", "scoring")
    mats = [f for f in sorted(os.listdir(sc)) if f.startswith("ScoringAB_")
            and f.endswith(".mat")]
    if not mats:
        raise SystemExit(f"no ScoringAB_*.mat in {sc} - score the video first")
    m = loadmat(os.path.join(sc, mats[0]), squeeze_me=True,
                struct_as_record=False)
    if int(np.asarray(m["offFrames"]).ravel()[0]) != 0:
        raise SystemExit("offFrames is not 0; the two cameras are offset and "
                         "the frame mapping below would be wrong")
    return dict(file=mats[0],
                d_frame=np.asarray(m["dFrames"], int).ravel(),
                d_type=np.asarray(m["dTypes"], int).ravel(),
                reflex=np.asarray(m["reflexEvents"], int).reshape(-1, 2),
                score=np.asarray(m["score"], int).ravel(),
                stim=[str(s) for s in np.asarray(m["stimNames"]).ravel()],
                ref=[str(s) for s in np.asarray(m["refNames"]).ravel()],
                aff=[str(s) for s in np.asarray(m["aff" "Names"]).ravel()],
                fps=float(np.asarray(m["frameRate"]).ravel()[0]),
                lag_ms=float(np.asarray(m["lagMs"]).ravel()[0]),
                n_used=int(np.asarray(m["nUsed"]).ravel()[0]))


def frame_to_imaging(session, camera="MiceVideo1"):
    """frame index (1-based) -> imaging-clock seconds, from the sync line."""
    B = pd.read_csv(os.path.join(session, "output_split", "timestamps",
                                 "behav_frame_times.csv"))
    g = B[B["camera"] == camera].sort_values("frame")
    fr = g["frame"].to_numpy()
    if not np.array_equal(fr, np.arange(1, len(fr) + 1)):
        raise SystemExit(f"{camera}: frame column is not 1..N contiguous")
    return g["t_s"].to_numpy()


def peri(e, t, events, lags, base=BASE):
    """Baseline-subtracted average of e(t) around each event time."""
    out = np.empty((len(events), len(lags)), float)
    for i, et in enumerate(events):
        out[i] = np.interp(et + lags, t, e, left=np.nan, right=np.nan)
    b = np.nanmean(out[:, (lags >= base[0]) & (lags <= base[1])], axis=1,
                   keepdims=True)
    return out - b


def peak_lag(curve, lags, frac=0.5):
    """Onset and peak of a peri-event curve.

    The maximum is the peak of the movement, which is later than its onset;
    the ONSET is what marks the event. Onset is the lag at which the curve
    last crosses `frac` of its own peak on the way up, found by walking back
    from the peak so a later bump cannot claim it.
    """
    k = int(np.nanargmax(curve))
    thr = frac * curve[k]
    j = k
    while j > 0 and curve[j] > thr:
        j -= 1
    return float(lags[j]), float(lags[k]), float(curve[k])


def per_trial_onset(M, lags, frac=0.5):
    """Median onset over individual deliveries.

    The onset of an averaged curve is biased early - one delivery with a
    long hand approach pulls the average up before any of the others. Doing
    it per delivery and taking the median is the less flattering number, so
    both are reported.
    """
    v = []
    for row in M:
        if not np.isfinite(row).any():
            continue
        v.append(peak_lag(np.nan_to_num(row, nan=0.0), lags, frac)[0])
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75))) if v else (np.nan,) * 3


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
    if S["d_frame"].max() > len(tb):
        raise SystemExit("a delivery frame is past the end of the camera")
    d_t = tb[S["d_frame"] - 1]
    r_t = tb[np.clip(S["reflex"][:, 0], 1, len(tb)) - 1]
    r_type = S["reflex"][:, 1]

    mz = np.load(os.path.join(root, "motion", "motion_energy.npz"))
    E = {c: mz[f"e_{c}"] for c in ("bottom", "side")}
    T = {c: mz[f"t_{c}"] for c in ("bottom", "side")}

    L = ["===== how late is the key press? =====", "",
         f"scoring file: {S['file']}",
         f"  {len(d_t)} deliveries: "
         + ", ".join(f"{S['stim'][k - 1]} {int((S['d_type'] == k).sum())}"
                     for k in sorted(set(S["d_type"]))),
         f"  {len(r_t)} reflex taps: "
         + ", ".join(f"{S['ref'][k - 1]} {int((r_type == k).sum())}"
                     for k in sorted(set(r_type))),
         f"  the tool's own assumed lag: {S['lag_ms']:.0f} ms "
         f"(applied to its training labels only, not to these times)",
         "",
         "frame -> imaging clock through the sync line, never frame/fps: the "
         "camera",
         f"dropped frames, and frame/{S['fps']:.0f} is off by up to "
         f"{np.max(np.abs((S['d_frame'] - 1) / S['fps'] - (d_t - tb[0]))):.3f}"
         f" s across these deliveries",
         f"(the scorer's own Time_s column carries exactly that error).",
         ""]

    rows, curves = [], {}
    for k in sorted(set(S["d_type"])):
        name = S["stim"][k - 1]
        ev = d_t[S["d_type"] == k]
        for cam in ("bottom", "side"):
            M = peri(E[cam], T[cam], ev, LAGS)
            c = np.nanmean(M, axis=0)
            curves[(name, cam)] = c
            onset, pk, amp = peak_lag(c, LAGS)
            med, q1, q3 = per_trial_onset(M, LAGS)
            bs = []
            for _ in range(N_BOOT):
                idx = RNG.integers(0, len(ev), len(ev))
                cb = np.nanmean(peri(E[cam], T[cam], ev[idx], LAGS), axis=0)
                bs.append(peak_lag(cb, LAGS)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(stimulus=name, camera=cam, n=len(ev),
                             onset_lag_s=round(onset, 3),
                             onset_ci_lo=round(float(lo), 3),
                             onset_ci_hi=round(float(hi), 3),
                             onset_per_trial_med=round(med, 3),
                             onset_per_trial_q1=round(q1, 3),
                             onset_per_trial_q3=round(q3, 3),
                             peak_lag_s=round(pk, 3),
                             peak_amplitude=round(amp, 4),
                             delay_estimate_s=round(-onset, 3),
                             used=(cam == "bottom")))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "tap_delay.csv"), index=False)

    L += ["  motion onset relative to the tap. Negative = the movement came "
          "FIRST, so the tap is late by that much.",
          f"  {'stimulus':>9s} {'camera':>7s} {'n':>4s} {'onset':>7s} "
          f"{'95 % CI':>16s} {'per-delivery med [IQR]':>24s} {'peak':>7s} "
          f"{'amp':>8s}  used"]
    for _, r in D.iterrows():
        L.append(f"  {r['stimulus']:>9s} {r['camera']:>7s} {r['n']:>4d} "
                 f"{r['onset_lag_s']:+7.2f} "
                 f"[{r['onset_ci_lo']:+6.2f},{r['onset_ci_hi']:+6.2f}] "
                 f"{r['onset_per_trial_med']:+10.2f} "
                 f"[{r['onset_per_trial_q1']:+5.2f},"
                 f"{r['onset_per_trial_q3']:+5.2f}]   "
                 f"{r['peak_lag_s']:+7.2f} {r['peak_amplitude']:8.4f}  "
                 + ("yes" if r["used"] else "no"))
    L += ["",
          "  Why the bottom camera and not the side camera. The bottom view "
          "contains the stimulus",
          "  itself - hand, filament - so its motion onset is the delivery. "
          "The side view contains the",
          "  mouse, and whole-body motion energy is dominated by what it "
          "does next (lick, guard, move",
          "  away), which builds over seconds: the side curve peaks "
          f"{D[(D['camera'] == 'side') & (D['stimulus'] == 'pin')].iloc[0]['peak_lag_s']:+.2f} s after the tap for pin",
          "  with an interval spanning 3 s. An earlier version of this "
          "script used the side camera as",
          "  the primary estimate and would have shifted every stimulus "
          "1.7 s the wrong way.",
          "  The side numbers are kept as the behavioural response latency, "
          "which is worth knowing."]

    # ---- the tap-order check, which uses no video at all -------------
    L += ["", "reflex taps against their own stimulus tap (this cancels the "
               "human delay, so it only checks ordering):"]
    for k in sorted(set(r_type)):
        rt = r_t[r_type == k]
        dt = np.array([rt_i - d_t[d_t <= rt_i].max() if (d_t <= rt_i).any()
                       else np.nan for rt_i in rt])
        fin = np.isfinite(dt)
        near = np.array([np.min(np.abs(d_t - rt_i)) for rt_i in rt])
        L.append(f"  {S['ref'][k - 1]:<16s} n={len(rt):3d}  "
                 f"after the previous stimulus by median "
                 f"{np.nanmedian(dt[fin]):5.2f} s  "
                 f"| nearest stimulus {np.median(near):.2f} s away  "
                 f"| {int(np.sum(near > 10))} taps more than 10 s from any "
                 f"stimulus")

    used = D[D["used"]]
    delay = float(np.mean(used["delay_estimate_s"]))
    L += ["",
          f"USING: delay = {delay:+.2f} s, the mean of the bottom-camera "
          f"estimates ({', '.join(f'{r.stimulus} {r.delay_estimate_s:+.2f}' for r in used.itertuples())}).",
          f"Corrected event time = scored tap time - {delay:.2f} s.",
          f"For comparison, the scoring tool's own built-in assumption is "
          f"{S['lag_ms'] / 1000:.2f} s, which it applies",
          "only to its training labels. The two agree to within "
          f"{abs(delay - S['lag_ms'] / 1000):.2f} s.",
          "",
          "Caveats, in order of how much they matter.",
          "  The bottom camera sees the hand APPROACH, which begins before "
          "contact, so this delay is an",
          "  upper bound: the true delay is somewhere between 0 and "
          f"{delay:.2f} s. The direction is certain",
          "  though - a person watching a recording cannot tap early.",
          "  The onset of an averaged curve is biased towards the earliest "
          "delivery, which is why the",
          "  per-delivery median is printed next to it. Where the two "
          "disagree, trust the median.",
          "  One delay is applied to every delivery. Human reaction time "
          "varies trial to trial, so this",
          "  removes the systematic part and leaves the jitter, which "
          "widens peri-stimulus averages but",
          "  does not shift them.",
          f"  At 4.6 Hz imaging, {abs(delay):.2f} s is "
          f"{abs(delay) * 4.6083:.1f} imaging frames, against a response "
          f"window 2.5 s wide. So this",
          "  correction matters for the shape of a peri-stimulus average "
          "and hardly at all for whether a",
          "  cell passes - pain_events.py prints the response against lag "
          "so that claim is checkable.",
          "  pain_events.py reports the neural response against lag as "
          "well. That profile is a CHECK on this",
          "  number, not its source: picking the lag that maximises the "
          "neural response and then testing",
          "  significance there would manufacture the result."]

    txt = "\n".join(L)
    with open(os.path.join(out, "tap_delay.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({n for n, _ in curves})
    fig, ax = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.6),
                           squeeze=False)
    for j, name in enumerate(names):
        A = ax[0, j]
        # bottom and side have very different amplitudes; each is scaled by
        # its own peak so the TIMING is comparable, which is the only thing
        # this figure is about. The amplitudes are in the table.
        for cam, col in (("bottom", "#B8860B"), ("side", "#1C6E8C")):
            c = curves[(name, cam)]
            r = D[(D["stimulus"] == name) & (D["camera"] == cam)].iloc[0]
            A.plot(LAGS, c / max(r["peak_amplitude"], 1e-9), color=col,
                   lw=1.7, label=f"{cam} camera (peak "
                                 f"{r['peak_amplitude']:.2f})")
            A.axvline(r["onset_lag_s"], color=col, ls="--", lw=1.1)
            A.plot([r["onset_per_trial_q1"], r["onset_per_trial_q3"]],
                   [-.12, -.12], color=col, lw=3, solid_capstyle="butt")
            A.plot([r["onset_per_trial_med"]], [-.12], "|", color=col,
                   ms=12, mew=2)
        A.axvline(0, color="#1A1A1A", lw=1.4)
        A.axvline(-delay, color="#2E7D5B", lw=1.4, ls=":")
        A.set_ylim(-.22, 1.15)
        A.text(0, 1.13, " tap", fontsize=9, rotation=90, va="top")
        A.text(-delay, 1.13, f" correction {-delay:+.2f} s ", fontsize=9,
               rotation=90, va="top", ha="right", color="#2E7D5B")
        A.set_xlabel("lag from the scored tap (s)")
        A.set_ylabel("motion energy / its own peak")
        n = int(D[(D['stimulus'] == name)
                  & (D['camera'] == 'side')].iloc[0]['n'])
        A.set_title(f"{name}, {n} deliveries", fontsize=12)
        A.legend(fontsize=9, frameon=False, loc="upper right")
        A.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "How late is the key press? Measured from the video, not assumed."
        f"\nBOTTOM camera contains the stimulus and is what the "
        f"{-delay:+.2f} s correction comes from. SIDE camera contains the "
        f"mouse's own response, which peaks seconds later - it is NOT a "
        f"delay estimate."
        "\nDashed = onset of the average curve; the bar below is the "
        "per-delivery median and IQR. Curves are scaled by their own peak "
        "so timing is comparable.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .88])
    p = os.path.join(out, "fig7_tap_delay.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
