"""make_visual_ppt.py  -  the four visualisation sets, one deck, for confirmation.

Separate from make_pipeline_ppt.py on purpose. That deck is the narrative
report. This one exists so Hansol can check the FOUR figure sets that were
asked for:

  1 processing        raw movie to what EXTRACT sees, quality measured
  2 cell detection    every footprint against its own local control
  3 place coding      how a centre or corner cell is decided
  4 across sessions   the same neuron in the pain and open-field sessions

Every slide is one figure plus a one-line "what to look at" and the file
path. The figures carry their own methods legends, so the slides stay short.

Never uses PowerPoint COM: it attaches to the running instance and closes
whatever is open. Writes to a fresh version if the target is locked.

OUTPUT  ->  <this folder>\\CEANTSR1_visualisation_20260911.pptx

USAGE
  python make_visual_ppt.py
"""
from __future__ import annotations

import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
W, H = Inches(13.333), Inches(7.5)
INK, DIM = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x66, 0x66, 0x66)
ACC = RGBColor(0x1C, 0x6E, 0x8C)
GOOD = RGBColor(0x2E, 0x7D, 0x5B)


def facts():
    """Read the result numbers off the result files.

    Typed numbers in slide text went stale the moment an analysis was re-run
    - this deck claimed 13 followed neurons and 2 centre cells from two
    different vintages at the same time. Anything countable is counted here.
    """
    import pandas as pd
    f = {}
    u = os.path.join(PA, "match", "union_cells.csv")
    if os.path.exists(u):
        U = pd.read_csv(u)
        f["n_union"] = len(U)
        f["n_matched"] = int(U["matched"].sum())
        f["n_soma"] = int(((U["pain_anat"] >= 1)
                           & (U["openfield_anat"] >= 1)).sum())
        f["per_plane"] = ", ".join(
            f"plane {p} {int((U['plane'] == p).sum())}" for p in ("A", "B"))
        f["n_merged"] = int(U["merged_of"].notna().sum()) \
            if "merged_of" in U else 0
    p = os.path.join(OF, "place", "place_cells.csv")
    if os.path.exists(p):
        P = pd.read_csv(p)
        f["n_test"] = len(P)
        f["n_cen"] = int((P["preference"] == "centre").sum())
        f["n_cor"] = int((P["preference"] == "corner").sum())
        f["cen_ids"] = ", ".join(P.loc[P["preference"] == "centre", "cell"])
        f["cor_ids"] = ", ".join(P.loc[P["preference"] == "corner", "cell"]) \
            or "none"
    return f


F = facts()


def blank(p):
    return p.slides.add_slide(p.slide_layouts[6])


def text(s, t, x, y, w, h, size=14, bold=False, color=INK, mono=False,
         align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(t).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(2)
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold = Pt(size), bold
        r.font.color.rgb = color
        r.font.name = "Consolas" if mono else "Calibri"


SCRATCH = ("C:/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/"
           "05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad/ppt_bands")
OVERLAP = 0.06      # fraction of a band repeated in the next, for continuity


def fig_slide(prs, head, look, path, pathline, y=1.15, hbox=5.6,
              slice_tall=True):
    """One slide per figure, or several bands for a tall one.

    Fitting a tall figure into a 12.5 x 5.6 in box shrinks it by height, so
    the width collapses: fig4 came out 2.49 in wide on a 13.3 in slide - a
    sliver nobody can read. Anything taller than the slide box is cut into
    horizontal bands of the slide's own aspect ratio instead, with a small
    overlap so a row is never split without context.
    """
    if not os.path.exists(path):
        raise SystemExit(f"missing figure: {path}")
    iw, ih = Image.open(path).size
    target = 12.5 / hbox
    n_band = 1 if not slice_tall else max(1, int(round(ih / (iw / target))))
    os.makedirs(SCRATCH, exist_ok=True)
    for b in range(n_band):
        band = os.path.join(
            SCRATCH, f"{os.path.basename(path)[:-4]}_b{b + 1}of{n_band}.png")
        if n_band == 1:
            band = path
        else:
            step = ih / n_band
            top = int(max(0, b * step - (OVERLAP * step if b else 0)))
            bot = int(min(ih, (b + 1) * step
                          + (OVERLAP * step if b < n_band - 1 else 0)))
            Image.open(path).crop((0, top, iw, bot)).save(band)
        bw, bh = Image.open(band).size
        s = blank(prs)
        ttl = head if n_band == 1 else f"{head}   ({b + 1} of {n_band})"
        text(s, ttl, Inches(.4), Inches(.22), Inches(12.5), Inches(.45),
             size=21, bold=True)
        text(s, look if b == 0 else "continued", Inches(.4), Inches(.70),
             Inches(12.5), Inches(.38), size=13, color=ACC)
        mw, mh = Inches(12.5), Inches(hbox)
        sc = min(mw / bw, mh / bh)
        s.shapes.add_picture(band, Inches(.4) + Emu(int((mw - bw * sc) / 2)),
                             Inches(y), width=Emu(int(bw * sc)),
                             height=Emu(int(bh * sc)))
        text(s, pathline, Inches(.4), Inches(6.98), Inches(12.5),
             Inches(.4), size=9.5, color=DIM, mono=True)
    return n_band


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    s = blank(prs)
    text(s, "CEA-Ntsr1 two-plane imaging", Inches(.9), Inches(2.0),
         Inches(11.5), Inches(.8), size=32, bold=True)
    text(s, "Four visualisations, for confirmation", Inches(.9),
         Inches(2.9), Inches(11.5), Inches(.5), size=19, color=ACC)
    text(s,
         "1  processing        raw movie to what EXTRACT sees, with the "
         "quality measured at each stage\n"
         "2  cell detection    every footprint against its own local "
         "control, so the test can be seen\n"
         "3  place coding      how a centre or corner cell is decided, "
         "observed value against its own null\n"
         "4  across sessions   the same neuron in the pain and open-field "
         "sessions, one row per cell\n\n"
         "Each figure opens with the whole field of view, then goes to "
         "detail. Methods and statistics\n"
         "are written into each figure's own legend, so these slides stay "
         "short.\n\n"
         "Pain responsiveness is not here: it needs the manual scoring. "
         "Everything else is in place\n"
         "for it to drop straight into set 4.",
         Inches(.9), Inches(3.6), Inches(11.5), Inches(3.0), size=13,
         color=DIM)
    text(s, r"code C:\Users\hsollim\OneDrive\Desktop\cursor"
            r"\pain_2plane_split   |   figures <session>\output_split\ ",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    # ---------------- set 1: processing ----------------
    s = blank(prs)
    text(s, "Set 1 - processing", Inches(.4), Inches(.22), Inches(12.5),
         Inches(.5), size=24, bold=True)
    text(s, "Does each step actually improve the movie? Measured, not "
            "asserted.", Inches(.4), Inches(.78), Inches(12.5), Inches(.4),
         size=15, color=ACC)
    text(s,
         "Three rows: the whole field at every stage, then the same 140 px "
         "crop at every stage, then three measurements.\n\n"
         "Each measurement judges ONE stage - the one it is supposed to "
         "fix - and the grey band marks which:\n"
         "    frame-to-mean correlation  ->  motion correction\n"
         "    temporal SNR in cells      ->  denoising\n"
         "    cell over surround         ->  the spatial band-pass\n\n"
         "Plotting one metric across all stages and calling the rise an "
         "improvement is easy and wrong: a spatial\n"
         "blur raises frame-to-frame correlation for free. That is why it "
         "is deliberately not shown.",
         Inches(.4), Inches(1.45), Inches(12.5), Inches(3.2), size=13.5)
    text(s, "Headline: before motion correction the cell contrast is "
            "NEGATIVE (-0.30 SD) - motion smeared the somata until they "
            "were dimmer than their own surround. Correction flips it to "
            "+2.95 SD.",
         Inches(.4), Inches(4.9), Inches(12.5), Inches(.9), size=14,
         bold=True, color=GOOD)
    text(s, r"fig1_processing.py  ->  <plane>\curated\fig1_processing.png",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    fig_slide(prs, "Set 1 - pain plane A",
              "Follow one soma across the crops: smeared at stages 1-2, "
              "crisp from stage 3 on.",
              os.path.join(PA, "plane_A", "curated", "fig1_processing.png"),
              r"<pain>\output_split\plane_A\curated\fig1_processing.png")
    fig_slide(prs, "Set 1 - pain plane B",
              "Plane B was already steadier: frame-to-mean correlation "
              "0.74 before correction against 0.66 in plane A.",
              os.path.join(PA, "plane_B", "curated", "fig1_processing.png"),
              r"<pain>\output_split\plane_B\curated\fig1_processing.png")
    fig_slide(prs, "Set 1 - open field plane A",
              "Same pipeline, the cleaner recording: noise std 14.2 "
              "against 20.8 in the pain session.",
              os.path.join(OF, "plane_A", "curated", "fig1_processing.png"),
              r"<open field>\output_split\plane_A\curated\fig1_processing.png")

    # ---------------- set 2: detection ----------------
    s = blank(prs)
    text(s, "Set 2 - cell detection", Inches(.4), Inches(.22), Inches(12.5),
         Inches(.5), size=24, bold=True)
    text(s, "Show the comparison the statistics make, instead of asking "
            "anyone to trust a p-value.",
         Inches(.4), Inches(.78), Inches(12.5), Inches(.4), size=15,
         color=ACC)
    text(s,
         "Each cell is put next to ITS OWN control: the same footprint "
         "shape, moved to the nearest place in the\n"
         "same movie with no detected cell (28-52 px away). Same tissue, "
         "same depth, same shot noise, same\n"
         "extraction - only the soma is missing. If the cells look "
         "different from their controls, the detection is\n"
         "doing something real and you can see it.\n\n"
         "Nearest rather than random on purpose: a control in distant "
         "tissue has different brightness and noise,\n"
         "which would make the comparison easy in the wrong way.",
         Inches(.4), Inches(1.45), Inches(12.5), Inches(2.6), size=13.5)
    text(s,
         "Result, pain plane A:   lag-1 autocorrelation higher in the cell "
         "than in its own control for 11 of 11\n"
         "                        robust SNR higher for 8 of 11\n"
         "                        skew higher for only 6 of 11",
         Inches(.4), Inches(4.25), Inches(12.5), Inches(1.1), size=14,
         bold=True, color=GOOD)
    text(s, "So in this plane one of the two metrics is doing the work. "
            "That is stated rather than smoothed over - it is also why the "
            "verdict combines them against a measured null instead of "
            "relying on either alone.",
         Inches(.4), Inches(5.5), Inches(12.5), Inches(.9), size=12.5,
         color=DIM)
    text(s, r"fig2_detection.py  ->  <plane>\curated\fig2_detection.png",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    for tag, root, plane, look in (
            ("pain", PA, "A",
             "Panel C is the argument: 11 of 11 blue lines rise. Amber "
             "(silent) rise a little, the red SUSPECT line barely at all."),
            ("pain", PA, "B",
             "Cell 11a and 11b are the split pair - both compared against "
             "their own controls."),
            ("open field", OF, "B",
             "N3, N4 and N5 are the somata added by hand. All three came "
             "out silent: real footprints, no signal in 326 s.")):
        fig_slide(prs, f"Set 2 - {tag} plane {plane}", look,
                  os.path.join(root, f"plane_{plane}", "curated",
                               "fig2_detection.png"),
                  rf"<{tag}>\output_split\plane_{plane}\curated"
                  rf"\fig2_detection.png")

    # ---------------- set 3: place ----------------
    s = blank(prs)
    text(s, "Set 3 - centre versus corner", Inches(.4), Inches(.22),
         Inches(12.5), Inches(.5), size=24, bold=True)
    text(s, "The previous figure showed the answer. This one shows the "
            "evidence.",
         Inches(.4), Inches(.78), Inches(12.5), Inches(.4), size=15,
         color=ACC)
    text(s,
         "Top row, before any neuron is mentioned: where the mouse went, "
         "how long it spent there, and the\n"
         "3x3 zones drawn on the real floor.\n\n"
         "Then one row per cell. Left: that cell's activity per place, "
         "divided by how long the mouse spent in\n"
         "each bin. Right: the observed centre-minus-corner contrast (red) "
         "on top of the 2000 circular shifts\n"
         "it was compared against (grey). A cell is called when the red "
         "line falls outside the grey.\n\n"
         "The null is a circular shift of the trace against the behaviour, "
         "not a t-test across frames. One calcium\n"
         "transient spans several frames, so a t-test treats each frame as "
         "new information and calls noise\n"
         "significant. Rolling the trace keeps its autocorrelation and the "
         "animal's occupancy exactly, and\n"
         "destroys only the pairing.",
         Inches(.4), Inches(1.45), Inches(12.5), Inches(3.9), size=13.5)
    text(s, f"Result: {F['n_cen']} of {F['n_test']} union cells prefer the "
            f"centre ({F['cen_ids']}), {F['n_cor']} prefer the corners "
            f"({F['cor_ids']}). No cell is both zone- and speed-significant, "
            f"so the zone effects are not speed in disguise.",
         Inches(.4), Inches(5.5), Inches(12.5), Inches(1.0), size=14,
         bold=True, color=GOOD)
    text(s, r"fig3_place.py  ->  <open field>\output_split\place"
            r"\fig3_place.png",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    fig_slide(prs, "Set 3 - the decision, drawn",
              "A#3 and A#12 are the only red lines outside their grey "
              "nulls. Everyone else sits inside.",
              os.path.join(OF, "place", "fig3_place.png"),
              r"<open field>\output_split\place\fig3_place.png",
              hbox=5.7)

    # ---------------- set 4: across sessions ----------------
    s = blank(prs)
    text(s, "Set 4 - the same neuron in both sessions", Inches(.4),
         Inches(.22), Inches(12.5), Inches(.5), size=24, bold=True)
    text(s, "Row N is one neuron. EXTRACT's own numbering is not comparable "
            "across sessions.",
         Inches(.4), Inches(.78), Inches(12.5), Inches(.4), size=15,
         color=ACC)
    text(s,
         "Same-numbered cells across the two sessions sat a median of "
         "197 px (plane A) and 145 px (plane B)\n"
         "apart - 18 to 25 cell radii. So a unified id is assigned per "
         "plane: matched pairs first, best footprint\n"
         "agreement first, then cells seen only in the pain session, then "
         "only in the open field. Row A1 is then\n"
         "one neuron in both columns by construction.\n\n"
         "A pair must clear two gates. Position: within one cell radius "
         "(8 px) after registering the two sessions'\n"
         "mean images. Shape: footprint correlation at least 0.5 after "
         "alignment - position alone pairs\n"
         "neighbours in a dense plane.\n\n"
         "Registration is a direct correlation search over shifts. Each "
         "depth matches ITSELF across sessions at\n"
         "r about 0.87, against 0.33-0.44 for the wrong depth, so the same "
         "optical planes really were imaged\n"
         "29 minutes apart.",
         Inches(.4), Inches(1.45), Inches(12.5), Inches(4.1), size=13.5)
    text(s, f"Result: {F['n_union']} union neurons ({F['per_plane']}), each "
            f"with a trace in both sessions; {F['n_matched']} of them "
            f"detected independently twice, {F['n_soma']} on a soma in both. "
            f"All {F['n_cen']} centre-preferring cells could be followed "
            f"into the pain session.",
         Inches(.4), Inches(5.65), Inches(12.5), Inches(.9), size=14,
         bold=True, color=GOOD)
    text(s, r"fig4_matched.py  ->  <pain>\output_split\match"
            r"\fig4_matched_*.png , fig4_unified_cells.csv",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    # Only the z-score version goes in as bands. All three are 5 bands each,
    # and 15 slides of the same rows in three units is not a deck - the
    # other two are named here and shipped as files.
    fig_slide(prs, "Set 4 - z-score",
              "Every row on the same scale, so trace heights mean the same "
              "thing between cells. Plane A rows first, then plane B.",
              os.path.join(PA, "match", "fig4_matched_z.png"),
              r"<pain>\output_split\match\fig4_matched_z.png", hbox=5.7)
    s = blank(prs)
    text(s, "Set 4 - the same rows in dF/F and in raw fluorescence",
         Inches(.4), Inches(.22), Inches(12.5), Inches(.5), size=22,
         bold=True)
    text(s, "Identical row order; only the y quantity changes. Too tall to "
            "read on a slide, so they are files.",
         Inches(.4), Inches(.78), Inches(12.5), Inches(.4), size=14,
         color=ACC)
    text(s,
         "fig4_matched_z.png      the previous slides. Detrended dF/F over "
         "its robust SD, so amplitudes are\n"
         "                        comparable BETWEEN cells. Use this to "
         "compare one cell against another.\n\n"
         "fig4_matched_dff.png    (F - F0) / F0, F0 = the 20th percentile "
         "of that cell's own raw trace.\n"
         "                        Comparable within a cell over time, not "
         "between cells with different baselines.\n\n"
         "fig4_matched_rawF.png   raw movie units, no baseline removed. The "
         "only one that shows absolute\n"
         "                        brightness, bleaching and slow drift - "
         "dF/F hides all three by construction.\n\n"
         "fig4_unified_cells.csv  the row table: unified id, plane, pain "
         "cell, open-field cell, footprint\n"
         "                        agreement, both verdicts, and the "
         "open-field place preference. This is the\n"
         "                        file the pain-responsiveness columns get "
         "added to after the scoring.",
         Inches(.4), Inches(1.5), Inches(12.5), Inches(4.8), size=13)
    text(s, r"<pain>\output_split\match\ ", Inches(.4), Inches(6.98),
         Inches(12.5), Inches(.4), size=9.5, color=DIM, mono=True)

    # ---------------- set 4 QC: the separability bug ----------------
    s = blank(prs)
    text(s, "Set 4 QC - a mistake this deck had in it, and how it was found",
         Inches(.4), Inches(.22), Inches(12.5), Inches(.5), size=22,
         bold=True)
    text(s, "The union was built wrong the first time. Reported here because "
            "it changed the results.",
         Inches(.4), Inches(.78), Inches(12.5), Inches(.4), size=14,
         color=ACC)
    text(s,
         "What was wrong\n"
         "  The union took every pain footprint plus every open-field "
         "footprint that was not a shape-confirmed\n"
         "  match. An open-field footprint can sit ON TOP of a pain "
         "footprint and still fail the shape test, so the\n"
         "  same soma entered the design matrix twice.\n\n"
         "Why that is not a small problem\n"
         "  Traces come from one joint least-squares solve. Two "
         "near-identical footprints give it no way to choose,\n"
         "  so it pays for the soma's transient with a positive trace on one "
         "copy and a negative trace on the other.\n"
         "  Measured, by re-running the solve both ways: 7 of the 8 pairs "
         "came out anti-correlated at r = -0.25 to\n"
         "  -0.83, one cell's raw F was negative in every frame of both "
         "sessions, and cond(S'S) was 49 instead of 5.\n\n"
         "What it cost\n"
         "  The two 'corner-preferring' cells in the earlier version of this "
         "deck were the negative halves of\n"
         "  centre-preferring cells (r = -0.82 and -0.84). Centre minus "
         "corner is negative for a sign-flipped centre\n"
         "  cell by construction - so they were an artefact of the bug, not "
         "corner cells.\n\n"
         "How it is caught now\n"
         "  Overlap is measured between footprints AS REGRESSORS, which is "
         "what decides whether least squares can\n"
         "  separate them - not the distance between centres. Over 361 "
         "footprint pairs within a curated session\n"
         "  the worst pair of distinct neurons reaches 0.167, so the cut at "
         "0.30 is calibrated rather than chosen.\n"
         "  8 footprints failed it and are recorded as merged_of instead of "
         "added. transfer_footprints.py now\n"
         "  refuses to save a union that still contains an inseparable pair.",
         Inches(.4), Inches(1.32), Inches(12.5), Inches(5.4), size=12.5)
    text(s, r"transfer_footprints.py , fig5_separability.py", Inches(.4),
         Inches(6.98), Inches(12.5), Inches(.4), size=9.5, color=DIM,
         mono=True)

    fig_slide(prs, "Set 4 QC - shown",
              "Top: the whole field, kept footprints green, dropped red. "
              "Then one pair close up, the two traces the bad solve produces "
              "for it, and the calibration.",
              os.path.join(PA, "match", "fig5_separability.png"),
              r"<pain>\output_split\match\fig5_separability.png", hbox=5.7)

    # ---------------- what is next ----------------
    s = blank(prs)
    text(s, "What is ready, and what is waiting", Inches(.4), Inches(.22),
         Inches(12.5), Inches(.5), size=24, bold=True)
    text(s,
         "Ready\n"
         "  55 curated cells across four planes, verified against the "
         "curation decisions by verify_curation.py\n"
         "  every trace in three forms - raw F, dF/F, z - on one time base "
         "shared with the behaviour cameras\n"
         f"  {F['n_union']} union neurons measured in BOTH sessions "
         f"({F['n_matched']} of them detected twice by EXTRACT)\n"
         f"  {F['n_cen']} centre-preferring and {F['n_cor']} "
         f"corner-preferring cells of {F['n_test']} tested, all "
         f"followable into the pain session\n\n"
         "Waiting on the manual scoring\n"
         "  pin-prick / heat / behaviour-responsive classification, "
         "event-locked to the scored times\n"
         "  lock_events(t_s, z, stimulus_times) in pain_cell_traces.py "
         "takes the times and returns\n"
         "  (cells x events x lag) windows; stimulus times must be on the "
         "same imaging clock as t_s\n\n"
         "Then the question becomes one row of set 4: was the neuron that "
         "preferred the exposed centre\n"
         "also pin-prick responsive? For plane A that row is A1 - open "
         "field #12, pain #3, footprint\n"
         "agreement 0.95, and active in both sessions.",
         Inches(.4), Inches(1.1), Inches(12.5), Inches(5.4), size=13.5)
    text(s, r"nothing in <session>\output\ is touched - that is the other "
            r"agent's max-projection pipeline",
         Inches(.4), Inches(6.98), Inches(12.5), Inches(.4), size=9.5,
         color=DIM, mono=True)

    base = os.path.join(HERE, "CEANTSR1_visualisation_20260911")
    out, n = base + ".pptx", 1
    while True:
        try:
            with open(out, "ab"):
                pass
            break
        except OSError:
            n += 1
            out = f"{base}_v{n}.pptx"
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
    return out


if __name__ == "__main__":
    main()
