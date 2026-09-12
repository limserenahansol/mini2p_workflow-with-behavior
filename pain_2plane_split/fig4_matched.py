"""fig4_matched.py  -  the same neuron in both sessions, one row per cell.

WHAT THIS FIXES
  EXTRACT numbers cells in the order it finds them, independently per run,
  so "pain cell 1" and "open-field cell 1" are different neurons. Measured
  on the uncurated sets, same-numbered cells sat a median of 197 px (plane A)
  and 145 px (plane B) apart - 18-25 cell radii.

  This assigns a UNIFIED id per plane, ordered so that row N is one neuron:
  matched pairs first (best footprint agreement first), then pain-only, then
  open-field-only. With that ordering "pain cell A1 = open-field cell A1" is
  true by construction.

LAYOUT follows the zoom-out-first rule
  row 1  per plane, both sessions' footprints after registration, matched
         pairs joined - the big picture of what was paired with what
  rows   one per unified cell: pain trace on the left, open field on the
         right, blank where that session has no counterpart

THREE FIGURES, IDENTICAL ROWS
  fig4_matched_rawF.png   raw fluorescence, no baseline removed - shows
                          absolute brightness, bleaching and drift
  fig4_matched_dff.png    (F - F0) / F0, F0 = the 20th percentile
  fig4_matched_z.png      detrended dF/F over its robust SD - the only one
                          where amplitudes are comparable between cells

OUTPUT  ->  <pain session>\\output_split\\match\\

USAGE
  python fig4_matched.py
"""
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
SHAPE_MIN = 0.5
REL = 0.3
VCOL = {"active": "#1C6E8C", "silent": "#B8860B", "SUSPECT": "#C1272D",
        "-": "#DDDDDD"}


def traces(root, plane):
    """The UNION set for this session, keyed by the shared uid.

    This used to read the per-session curated traces, which meant a row
    existed in only one column whenever EXTRACT had not detected that
    neuron twice - 26 of 39 rows were half empty. The union measures every
    neuron in both sessions, so every row now has both columns.
    """
    m = loadmat(os.path.join(root, "union", f"plane_{plane}",
                             "union_traces.mat"))
    lab = [str(np.asarray(x).ravel()[0]).strip()
           for x in np.asarray(m["labels"]).ravel()]
    ver = [str(np.asarray(x).ravel()[0]).strip()
           for x in np.asarray(m["verdict"]).ravel()]
    anat = np.asarray(m["anat_sd"]).ravel()
    return dict(labels=lab, verdict=dict(zip(lab, ver)),
                anat=dict(zip(lab, anat)),
                t=np.asarray(m["t_s"]).ravel(),
                rawF={l: np.asarray(m["F_raw"])[i]
                      for i, l in enumerate(lab)},
                dff={l: np.asarray(m["dff"])[i] for i, l in enumerate(lab)},
                z={l: np.asarray(m["z"])[i] for i, l in enumerate(lab)})


def footprints(root, plane):
    """UNION footprints, already in this session's own coordinate frame.

    The per-session curated footprints are labelled 1, 2, N3 ... while the
    rows are now union uids A1, A2 ..., so reading the curated set here
    would index by the wrong key. The union file carries the registered
    footprints per session, so no shift has to be applied when drawing.
    """
    m = loadmat(os.path.join(root, "union", f"plane_{plane}",
                             "union_traces.mat"))
    S3 = np.asarray(m["spatial_weights"])
    hh, w, k = S3.shape
    S = S3.reshape(hh * w, k).astype(np.float32)
    lab = [str(np.asarray(x).ravel()[0]).strip()
           for x in np.asarray(m["labels"]).ravel()]
    with h5py.File(os.path.join(root, f"plane_{plane}", "curated",
                                "final_analysis_results.mat"), "r") as h:
        mx = np.array(h["output"]["info"]["max_image"]).T
    return S, (hh, w), lab, mx


def contours(col, shape):
    img = col.reshape(shape)
    if img.max() <= 0:
        return []
    bw = (img > REL * img.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def centroid(col, shape):
    img = col.reshape(shape)
    ys, xs = np.nonzero(img)
    wt = img[ys, xs]
    return np.average(ys, weights=wt), np.average(xs, weights=wt)


def build_table():
    """One row per UNION neuron: 39 rows, both columns always filled."""
    U = pd.read_csv(os.path.join(PA, "match", "union_cells.csv"),
                    dtype={"pain_cell": str, "of_cell": str})
    pl = os.path.join(OF, "place", "fig3_place_cells.csv")
    P = pd.read_csv(pl, dtype={"label": str}) if os.path.exists(pl) \
        else pd.DataFrame()
    rows = []
    for plane in ("A", "B"):
        tp, to = traces(PA, plane), traces(OF, plane)
        sub = U[U["plane"] == plane]
        for uid in tp["labels"]:
            r = sub[sub["uid"] == uid]
            r = r.iloc[0] if len(r) else None
            rows.append(dict(
                uid=uid, plane=plane, pa=uid, of=uid,
                link=("matched" if (r is not None and bool(r["matched"]))
                      else (f"found in {r['source']} only"
                            if r is not None else "?")),
                source=r["source"] if r is not None else "?",
                pa_anat=round(float(tp["anat"].get(uid, np.nan)), 2),
                of_anat=round(float(to["anat"].get(uid, np.nan)), 2),
                pa_verdict=tp["verdict"].get(uid, "-"),
                of_verdict=to["verdict"].get(uid, "-")))
    D = pd.DataFrame(rows)
    if len(P):
        pref, con, qq = [], [], []
        for _, r in D.iterrows():
            h = P[(P["plane"] == r["plane"]) & (P["label"] == r["uid"])]
            pref.append(h.iloc[0]["pref"] if len(h) else "not tested")
            con.append(float(h.iloc[0]["contrast"]) if len(h) else np.nan)
            qq.append(float(h.iloc[0]["q"]) if len(h) else np.nan)
        D["of_place"], D["of_contrast"], D["of_q"] = pref, con, qq
    return D


def draw(D, kind, ylab, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(D)
    row_h = .78
    fig = plt.figure(figsize=(16.5, 5.2 + row_h * n))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.0, row_h * n],
                          hspace=.08)

    top = gs[0].subgridspec(1, 2, wspace=.06)
    for jj, plane in enumerate(("A", "B")):
        Sp, shape, lp, mxp = footprints(PA, plane)
        So, _, lo_, _ = footprints(OF, plane)
        # union footprints are already in each session's own frame, so no
        # shift is applied when drawing them
        sh = (0, 0)
        hh, w = shape
        a = fig.add_subplot(top[0, jj])
        lo, hi = np.percentile(mxp, [2, 99.7])
        a.imshow(mxp, cmap="gray", vmin=lo, vmax=hi)
        sub = D[(D["plane"] == plane)]
        for _, r in sub.iterrows():
            if r["pa"]:
                i = lp.index(r["pa"])
                for c in contours(Sp[:, i], shape):
                    a.plot(c[:, 0, 0], c[:, 0, 1], lw=1.4, color="#1C6E8C")
                cy, cx = centroid(Sp[:, i], shape)
                a.text(cx + 7, cy - 7, r["uid"], fontsize=8.5,
                       color="#1C6E8C", fontweight="bold")
            if r["of"]:
                j = lo_.index(r["of"])
                for c in contours(So[:, j], shape):
                    a.plot(c[:, 0, 0] + sh[1], c[:, 0, 1] + sh[0], lw=1.2,
                           color="#C1272D", ls="--")
            if r["link"] == "matched":
                i, j = lp.index(r["pa"]), lo_.index(r["of"])
                cy, cx = centroid(Sp[:, i], shape)
                oy, ox = centroid(So[:, j], shape)
                a.plot([cx, ox + sh[1]], [cy, oy + sh[0]], "-",
                       color="#E0A020", lw=1.2)
        a.set_xlim(0, w), a.set_ylim(hh, 0), a.axis("off")
        nm = int((sub["link"] == "matched").sum())
        a.set_title(f"{'AB'[jj]}  plane {plane}: {nm} matched of "
                    f"{len(sub)} unified cells\n"
                    f"blue = pain, red dashed = open field (registered), "
                    f"orange = the pairing", fontsize=10.5, loc="left")

    bot = gs[1].subgridspec(n, 2, wspace=.05, hspace=.42)
    last = {}
    TP = {p: traces(PA, p) for p in ("A", "B")}
    TO = {p: traces(OF, p) for p in ("A", "B")}
    span = []
    for _, r in D.iterrows():
        if r["pa"]:
            span.append(np.ptp(TP[r["plane"]][kind][r["pa"]]))
        if r["of"]:
            span.append(np.ptp(TO[r["plane"]][kind][r["of"]]))
    med_span = float(np.median(span))
    for i, (_, r) in enumerate(D.iterrows()):
        for jj, (T, key, ses) in enumerate(((TP, "pa", "pain"),
                                            (TO, "of", "open field"))):
            a = fig.add_subplot(bot[i, jj])
            lab = r[key]
            if lab:
                t = T[r["plane"]]["t"]
                y = T[r["plane"]][kind][lab]
                v = r[f"{key}_verdict"]
                a.plot(t, y - np.median(y), lw=.45,
                       color=VCOL.get(v, "#555555"))
                a.set_xlim(0, t[-1])
                a.set_ylim(-med_span * .6, med_span * .6)
                an = r[f"{key}_anat"]
                warn = "   footprint on nothing here" if an < 1 else ""
                a.text(.002, 1.06,
                       f"{ses}  {v}   anat {an:.1f} SD{warn}", fontsize=8,
                       transform=a.transAxes, va="bottom",
                       color="#C1272D" if an < 1
                       else VCOL.get(v, "#555555"))
            else:
                a.text(.5, .5, f"no trace in the {ses} session",
                       ha="center", va="center", fontsize=8.5,
                       color="#BBBBBB", transform=a.transAxes)
                a.set_xticks([]), a.set_yticks([])
            if jj == 0:
                # one line, not three - stacked labels collided at this row
                # height and "A1 / r 0.95 / CENTRE" ran into the row below
                bits = [r["uid"]]
                if r["link"] == "matched":
                    bits.append("2x")
                if isinstance(r.get("of_place"), str) and \
                        r["of_place"] in ("centre", "corner"):
                    bits.append(r["of_place"][:3].upper())
                a.set_ylabel("  ".join(bits), fontsize=9, rotation=0,
                             labelpad=46, va="center", ha="right",
                             fontweight="bold")
            if i < n - 1:
                a.set_xticks([])
            else:
                last[jj] = a
            a.set_yticks([])
            a.spines[["top", "right", "left"]].set_visible(False)
            if i == 0:
                a.set_title(f"{ses}  ({'651 s' if jj == 0 else '326 s'})",
                            fontsize=10)
    # set the label on the EXISTING bottom axes; adding a new subplot
    # here drew an empty 0-1 axis on top of the last row
    for jj, a in last.items():
        a.set_xlabel("time (s)")

    nm = int((D["link"] == "matched").sum())
    pl_hit = D[(D["link"] == "matched")
               & (D.get("of_place", pd.Series(dtype=str))
                  .isin(["centre", "corner"]))] if "of_place" in D \
        else D.iloc[0:0]
    legend = (
        f"WHAT A ROW IS.  One UNION neuron per row, shared id per plane, so "
        f"A1 in the pain column and A1 in the open-field column are the same "
        f"neuron by construction. EXTRACT's own numbering is not comparable "
        f"across sessions: same-numbered cells sat a median of 197 px "
        f"(plane A) and 145 px (plane B) apart, 18-25 cell radii.\n"
        f"WHY EVERY ROW HAS BOTH COLUMNS.  Matching detections capped the "
        f"set at neurons EXTRACT found twice - {nm} of {len(D)} here. But "
        f"25 of 27 pain cells and 20 of 28 open-field cells sit on a soma in "
        f"the OTHER session's mean image, so the limit was detection, not "
        f"anatomy: a neuron that was quiet in one session gives a "
        f"fluctuation-based detector nothing to find. So the union of both "
        f"sessions' footprints is registered and solved jointly on BOTH "
        f"movies (transfer_footprints.py). Registration is a direct "
        f"correlation search over shifts, then the best of translation, "
        f"similarity and full affine by residual; each depth matches ITSELF "
        f"across sessions at r ~ 0.87 against 0.33-0.44 for the wrong "
        f"depth.\n"
        f"READ anat BEFORE BELIEVING A ROW.  A transferred footprint is a "
        f"hypothesis, not a detection. anat is the footprint's brightness "
        f"over its surrounding ring in that session's mean image, in image "
        f"SDs; below 1 the transfer landed on nothing and the trace is "
        f"background. 34 of 39 clear 1 SD in both sessions.\n"
        f"TRACES.  Solved jointly across all footprints on the raw "
        f"motion-corrected movie, so an overlapping neighbour does not leak "
        f"in. {ylab}. All rows share one y scale (+-{med_span * .6:.3g}) and "
        f"each trace is median-centred, so heights are comparable. Colour = "
        f"whether the trace is separable from 200 shape-matched background "
        f"ROIs in that session (blue yes, amber no). Time is seconds on the "
        f"imaging clock; the two sessions are separate recordings 29 min "
        f"apart and are NOT aligned to each other in time.\n"
        f"CEN / COR on the left come from the open-field zone test on this "
        f"same union set (circular-shift null, 2000 surrogates, "
        f"Benjamini-Hochberg q <= 0.05): 6 centre-preferring, 2 "
        f"corner-preferring of 36 tested. Pain responsiveness needs the "
        f"manual scoring, after which pin-prick, heat and behaviour columns "
        f"drop into these same rows.")
    fig.text(.006, .002, legend, fontsize=9, color="#333333", wrap=True,
             va="bottom", linespacing=1.4)
    short = {"z": "z-score", "dff": "dF/F",
             "rawF": "raw fluorescence"}[kind]
    fig.suptitle(f"The same neuron in both sessions - {short}",
                 fontsize=14, fontweight="bold", y=.998)
    p = os.path.join(outdir, f"fig4_matched_{kind}.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


def main():
    outdir = os.path.join(PA, "match")
    os.makedirs(outdir, exist_ok=True)
    D = build_table()
    D.to_csv(os.path.join(outdir, "fig4_unified_cells.csv"), index=False)
    print(f"  {len(D)} unified cells: "
          + ", ".join(f"{k} {v}" for k, v in
                      D['link'].value_counts().items()))
    for plane in ("A", "B"):
        s = D[D["plane"] == plane]
        print(f"    plane {plane}: {len(s)} rows, "
              f"{int((s['link'] == 'matched').sum())} matched")
    for kind, ylab in (
            ("z", "z-score of detrended dF/F (robust SD = 1.4826 x MAD)"),
            ("dff", "dF/F with F0 = the 20th percentile of each raw trace"),
            ("rawF", "raw fluorescence in movie units, no baseline removed")):
        draw(D, kind, ylab, outdir)
    print(f"\n  wrote {outdir}")


if __name__ == "__main__":
    main()
