"""fig1_processing.py  -  raw movie to footprints, with the quality measured at each stage.

LAYOUT follows the zoom-out-first rule
  row 1  the whole field at every stage, so the big picture comes first
  row 2  the same 140 px crop at every stage, so the detail is comparable
  row 3  three measurements, each matched to the stage it is meant to judge

EACH METRIC JUDGES ONE STAGE, NOT ALL OF THEM
  It would be easy and wrong to plot "frame-to-frame correlation" across all
  stages and call the rise an improvement: a spatial blur raises it for free.
  So each stage is judged by the thing it is actually supposed to fix.

  frame-to-mean       median correlation of each band-passed frame with
  correlation         the mean of all of them. If the field moves, a frame
                      stops matching the average and this falls. No shift
                      estimator is involved - the first version of this
                      panel reported a per-frame shift in px and was wrong
                      twice over; see frame_to_mean_corr().
  temporal SNR        per-pixel mean divided by the sd of the frame-to-frame
                      difference (which cancels any static structure), taken
                      inside the detected footprints. Denoising is supposed
                      to raise this.
  cell contrast       mean brightness inside a footprint over the ring
                      around it. The spatial band-pass is supposed to raise
                      this by removing the shared background; motion
                      correction should raise it too by stopping the soma
                      from smearing.

STAGES
  1 raw interleaved   CellVideo1\\CellVideo\\*.tif as acquired - the two
                      depths alternate frame by frame
  2 deinterleaved     output_split\\plane_X.tif, one depth
  3 motion corrected  analysis_results.mat / motion_corrected_data (NoRMCorre)
  4 denoised          plane_X_1.h5 / mov (gaussian, sigma 1.0)
  5 band-passed       what EXTRACT actually sees (spatial_bandpass,
                      radius 8, highpass 8)

OUTPUT  ->  <plane>\\curated\\fig1_processing.png

USAGE
  python fig1_processing.py --plane A
"""
from __future__ import annotations

import argparse
import os

import cv2
import h5py
import numpy as np
import tifffile

PAIN = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
        "15-56-48")
N_MEAN = 200        # frames averaged for each stage's mean image
N_MOTION = 200      # frames used for the frame-to-mean correlation
RADIUS = 8
HP = 8
CROP = 140


def bandpass(a, r=RADIUS, hp=HP):
    """Same form as EXTRACT's spatial_bandpass: keep the cell scale."""
    a = a.astype(np.float32)
    lo = cv2.GaussianBlur(a, (0, 0), r / 2.0)
    hi = cv2.GaussianBlur(a, (0, 0), r * hp / 2.0)
    return lo - hi


def read_tif_frames(path, idx):
    with tifffile.TiffFile(path) as t:
        n = len(t.pages)
        idx = [i for i in idx if 0 <= i < n]
        return np.stack([t.pages[i].asarray() for i in idx]).astype(
            np.float32), n


def read_mat_frames(path, var, idx):
    with h5py.File(path, "r") as f:
        d = f[var]
        n = d.shape[0]
        idx = [i for i in idx if 0 <= i < n]
        return np.stack([np.asarray(d[i]).T for i in idx]).astype(
            np.float32), n


def read_h5_frames(path, idx):
    return read_mat_frames(path, "mov", idx)


def frame_to_mean_corr(frames):
    """Median correlation of each band-passed frame with the stack's mean.

    This replaces a "residual motion in px" panel that was wrong twice over.
    That version estimated a per-frame shift by phase correlation and took
    the SD. Checked against a direct grid search on the same frames:

        plane A, deinterleaved   phase corr 0.29 px   grid search 2.60 px
        plane B, deinterleaved   phase corr 0.40 px   grid search 3.55 px
        the two disagreed on individual frames by a median of 1.9-2.6 px

    and with 200 frames instead of 60 the SD blew up to 16.6 px because
    phase correlation fails outright on a few frames and an SD is not robust
    to that. Two estimators that disagree by 10x, plus a non-robust
    summary, is not a number worth putting on a figure.

    Frame-to-mean correlation needs no shift estimator at all: if the field
    moves, each frame stops matching the average of all frames, and the
    correlation falls. It is bounded, robust when summarised by a median,
    and it is the same quantity for every stage.
    """
    ref = bandpass(frames.mean(0))
    ref = (ref - ref.mean()) / (ref.std() + 1e-9)
    out = []
    for i in range(len(frames)):
        b = bandpass(frames[i])
        b = (b - b.mean()) / (b.std() + 1e-9)
        out.append(float(np.mean(ref * b)))
    return float(np.median(out)), np.asarray(out)


def temporal_snr(frames, mask):
    """mean / sd(frame-to-frame difference), inside a mask.

    The difference cancels every static structure, so this is noise, not
    anatomy. Divided by sqrt(2) because differencing two independent frames
    doubles the variance.
    """
    m = frames[:, mask]
    noise = np.diff(m, axis=0).std(0) / np.sqrt(2)
    sig = m.mean(0)
    ok = noise > 0
    return float(np.median(sig[ok] / noise[ok]))


def cell_contrast(mean_img, S, shape, inner=3, outer=12):
    """(inside - ring) in units of the image's own robust SD.

    The obvious form, (inside - ring) / ring_mean, is invalid after the
    band-pass: that stage removes the mean, so the ring mean sits at ~0 and
    the ratio exploded to +15.5 - a broken metric, not a real gain. Dividing
    by the image's robust SD instead is well defined at every stage and
    stays comparable across them.
    """
    sd = 1.4826 * np.median(np.abs(mean_img - np.median(mean_img)))
    if sd <= 0:
        return np.nan
    out = []
    for i in range(S.shape[1]):
        m = (S[:, i].reshape(shape) > 0).astype(np.uint8)
        din = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inner + 1,) * 2))
        dout = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * outer + 1,) * 2))
        ring = (dout > 0) & (din == 0)
        if m.sum() == 0 or ring.sum() == 0:
            continue
        out.append((mean_img[m > 0].mean() - mean_img[ring].mean()) / sd)
    return float(np.median(out)) if out else np.nan


def run(session, plane):
    base = os.path.join(session, "output_split", f"plane_{plane}")
    cur = os.path.join(base, "curated")
    with h5py.File(os.path.join(cur, "final_analysis_results.mat"), "r") as h:
        S3 = np.array(h["output"]["spatial_weights"])
    k, w, hh = S3.shape
    S = S3.transpose(0, 2, 1).reshape(k, hh * w).T.astype(np.float32)
    shape = (hh, w)
    occupied = (S.sum(1).reshape(shape) > 0)

    raw_dir = os.path.join(session, "CellVideo1", "CellVideo")
    raw_tif = os.path.join(raw_dir, sorted(os.listdir(raw_dir))[0])
    plane_tif = os.path.join(session, "output_split", f"plane_{plane}.tif")
    mc_mat = os.path.join(base, "analysis_results.mat")
    dn_h5 = os.path.join(base, f"plane_{plane}_1.h5")

    # plane A is the odd raw frames, plane B the even ones
    off = 0 if plane == "A" else 1
    print(f"  reading {N_MEAN} frames per stage")
    with tifffile.TiffFile(plane_tif) as t:
        n_pl = len(t.pages)
    idx_pl = np.linspace(0, n_pl - 1, N_MEAN).astype(int)
    idx_raw = (idx_pl * 2 + off)

    stages = []
    fr, n_raw = read_tif_frames(raw_tif, idx_raw % 2000)
    stages.append(("1 raw, interleaved", fr, "two depths alternate"))
    fr, _ = read_tif_frames(plane_tif, idx_pl)
    stages.append((f"2 deinterleaved, plane {plane}", fr,
                   "one depth, from the Slice channel"))
    fr, _ = read_mat_frames(mc_mat, "motion_corrected_data", idx_pl)
    stages.append(("3 motion corrected", fr, "NoRMCorre, rigid"))
    fr, _ = read_h5_frames(dn_h5, idx_pl)
    stages.append(("4 denoised", fr, "gaussian, sigma 1.0 px"))
    bp = np.stack([bandpass(f) for f in stages[-1][1]])
    stages.append(("5 band-passed - what EXTRACT sees", bp,
                   f"radius {RADIUS}, highpass {HP}"))

    print("  measuring")
    metrics = []
    for si, (name, fr, _) in enumerate(stages, 1):
        mean_img = fr.mean(0)
        rm, _ = frame_to_mean_corr(fr[:N_MOTION])
        # mean / noise needs a mean to divide by. The band-pass removes it,
        # so the number there was 1.48 - an artefact of dividing by ~0, not
        # a real collapse in quality. Left out rather than plotted.
        snr = np.nan if si == len(stages) else temporal_snr(fr, occupied)
        ct = cell_contrast(mean_img, S, shape)
        metrics.append(dict(stage=name, mean=mean_img, motion=rm, snr=snr,
                            contrast=ct))
        ss = "  n/a " if not np.isfinite(snr) else f"{snr:6.2f}"
        print(f"    {name:38s} frame-to-mean r {rm:5.3f}   "
              f"temporal SNR {ss}   cell contrast {ct:+6.2f} sd")

    # crop centred on the brightest footprint, for the detail row
    i0 = int(np.argmax([(S[:, i] > 0).sum() * S[:, i].max()
                        for i in range(k)]))
    img = S[:, i0].reshape(shape)
    ys, xs = np.nonzero(img)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, hh - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = len(stages)
    # A 4th row is reserved for the legend so it cannot land on the x-axis
    # labels, and the per-stage notes go inside the image instead of under
    # it - the first version had all three of those overlapping.
    fig = plt.figure(figsize=(3.3 * ns, 12.2))
    gs = fig.add_gridspec(4, ns, height_ratios=[1.3, 1.0, 0.9, 0.42],
                          hspace=.34, wspace=.06)

    for j, (m, (name, _fr, note)) in enumerate(zip(metrics, stages)):
        im = m["mean"]
        lo, hi = np.percentile(im, [2, 99.6])
        a = fig.add_subplot(gs[0, j])
        a.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        for i in range(k):
            bw = (S[:, i].reshape(shape) > 0.3 * S[:, i].max()).astype(
                np.uint8)
            for c in cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]:
                a.plot(c[:, 0, 0], c[:, 0, 1], lw=.5, color="#E0A020",
                       alpha=.75)
        a.add_patch(plt.Rectangle((x0, y0), CROP, CROP, fill=False,
                                  ec="#1C8CA8", lw=1.4))
        a.set_xlim(0, w), a.set_ylim(hh, 0), a.axis("off")
        a.set_title(name, fontsize=11.5, fontweight="bold")
        a.text(.02, .02, note, transform=a.transAxes, ha="left",
               va="bottom", fontsize=8.5, color="#FFFFFF",
               bbox=dict(facecolor="#000000", alpha=.55, pad=2.2,
                         edgecolor="none"))

        b = fig.add_subplot(gs[1, j])
        sub = im[y0:y0 + CROP, x0:x0 + CROP]
        lo2, hi2 = np.percentile(sub, [2, 99.6])
        b.imshow(sub, cmap="gray", vmin=lo2, vmax=hi2)
        b.set_xticks([]), b.set_yticks([])
        for s in b.spines.values():
            s.set_color("#1C8CA8")
            s.set_linewidth(1.4)
        if j == 0:
            b.set_ylabel(f"{CROP} px crop", fontsize=10)

    keys = [("motion", "frame-to-mean correlation\nhigher is better",
             "#C1272D", 3),
            ("snr", "temporal SNR in cells\nhigher is better", "#1C6E8C", 4),
            ("contrast", "cell over surround\nhigher is better",
             "#2E7D5B", 5)]
    for jj, (key, lab, col, judges) in enumerate(keys):
        a = fig.add_subplot(gs[2, jj])
        v = np.array([m[key] for m in metrics], float)
        ok = np.isfinite(v)
        a.plot(np.arange(1, ns + 1)[ok], v[ok], "o-", color=col, lw=1.8,
               ms=8, zorder=3)
        lo_, hi_ = float(np.nanmin(v)), float(np.nanmax(v))
        pad = (hi_ - lo_) * .22 + 1e-9
        a.axvspan(judges - .35, judges + .35, color="#EDEDED", zorder=0)
        for x_ in np.arange(1, ns + 1)[~ok]:
            a.annotate("n/a", (x_, lo_), fontsize=9, ha="center",
                       va="bottom", color="#999999")
        a.set_ylim(lo_ - pad, hi_ + pad * 1.6)
        a.set_xticks(range(1, ns + 1))
        a.set_xlim(.6, ns + .4)
        a.set_xlabel("stage")
        a.set_title(f"{lab}\ngrey band = the stage it judges", fontsize=10,
                    pad=12)
        for x_, y_ in zip(np.arange(1, ns + 1)[ok], v[ok]):
            a.annotate(f"{y_:.2f}", (x_, y_), fontsize=8.5,
                       xytext=(0, 9), textcoords="offset points",
                       ha="center")
        a.spines[["top", "right"]].set_visible(False)
    for jj in range(3, ns):
        fig.add_subplot(gs[2, jj]).axis("off")
    fig.add_subplot(gs[3, :]).axis("off")

    legend = (
        "METHODS.  Means are over "
        f"{N_MEAN} frames spread across the session; footprint outlines "
        f"(orange) are the curated set, drawn at 30 % of each footprint's "
        f"peak.  Frame-to-mean correlation: median over {N_MOTION} frames of "
        f"the correlation between each band-passed frame and the mean of all "
        f"of them; if the field moves, a frame stops matching the average "
        f"and this falls. No shift estimator is used - it judges motion "
        f"correction (stage 3) and nothing else.  Temporal SNR: median over "
        f"footprint pixels of "
        f"mean / SD(frame-to-frame difference)/sqrt(2); differencing cancels "
        f"static structure, so this is shot noise, not anatomy - it judges "
        f"denoising (stage 4).  Cell contrast: median over cells of "
        f"(mean inside the footprint - mean in a 3-12 px ring) divided by "
        f"the image's robust SD - it judges the band-pass (stage 5), and "
        f"rises with motion correction too because a smeared soma has lower "
        f"contrast.\n"
        f"WHAT IS DELIBERATELY NOT SHOWN.  Frame-to-frame correlation: a "
        f"spatial blur raises it for free, which would make denoising look "
        f"better than it is.  Temporal SNR at stage 5: the band-pass removes "
        f"the mean, so mean/noise divides by ~0 - it read 1.48, an artefact, "
        f"and is marked n/a instead.  The same broken division inflated the "
        f"first version of cell contrast to +15.5 at stage 5, which is why "
        f"it is now normalised by the image SD.  Read the SNR gain at stage "
        f"4 with care as well: a gaussian does raise per-pixel SNR by "
        f"averaging neighbours, but it correlates the noise spatially, so "
        f"the gain per soma is smaller than the number suggests.  Residual "
        f"motion in px: an earlier version of the first panel reported it, "
        f"estimated per frame by phase correlation and summarised by an SD. "
        f"Checked against a direct grid search on the same frames the two "
        f"estimators disagreed by 10x (0.29 vs 2.60 px for plane A), and "
        f"with 200 frames instead of 60 the SD blew up to 16.6 px because "
        f"phase correlation fails outright on a few frames. Dropped.")
    fig.text(.01, .008, legend, fontsize=9.5, color="#333333", wrap=True,
             va="bottom", linespacing=1.45)
    fig.suptitle(f"Processing: raw movie to what EXTRACT sees "
                 f"- pain plane {plane}", fontsize=15, fontweight="bold",
                 y=.985)
    p = os.path.join(cur, "fig1_processing.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=PAIN)
    ap.add_argument("--plane", default="A")
    a = ap.parse_args()
    print(f"processing figure: plane {a.plane}")
    run(a.session, a.plane)


if __name__ == "__main__":
    main()
