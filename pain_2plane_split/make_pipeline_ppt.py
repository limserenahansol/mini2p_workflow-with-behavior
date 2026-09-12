"""make_pipeline_ppt.py  -  step-by-step deck: pipeline, QC, then results.

One slide per step: a figure, the numbers it produced, and where the files
are. Deliberately not wordy - the figures carry the explanation.

Never uses PowerPoint COM: that attaches to the running instance and closes
whatever Hansol has open. python-pptx writes the file directly, every
embedded figure is checked to exist first, and check_deck.py verifies
afterwards that nothing runs off a slide or overlaps.

OUTPUT  ->  <this folder>\\CEANTSR1_pipeline_20260911.pptx

USAGE
  python make_pipeline_ppt.py
"""
from __future__ import annotations

import os

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = ("C:/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/"
           "05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad")
OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")

W, H = Inches(13.333), Inches(7.5)
INK, DIM = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x66, 0x66, 0x66)
ACC, BAD, GOOD = (RGBColor(0x1C, 0x6E, 0x8C), RGBColor(0xA6, 0x3A, 0x3A),
                  RGBColor(0x2E, 0x7D, 0x5B))


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(slide, s, x, y, w, h, size=14, bold=False, color=INK,
         align=PP_ALIGN.LEFT, mono=False, space=2):
    tf = slide.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(s).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(space)
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold = Pt(size), bold
        r.font.color.rgb = color
        r.font.name = "Consolas" if mono else "Calibri"


def title(slide, main, sub=None):
    text(slide, main, Inches(.45), Inches(.26), Inches(12.4), Inches(.5),
         size=22, bold=True)
    if sub:
        text(slide, sub, Inches(.45), Inches(.82), Inches(12.4), Inches(.36),
             size=12.5, color=ACC)


def path_line(slide, s):
    text(slide, s, Inches(.45), Inches(6.98), Inches(12.4), Inches(.4),
         size=9.5, color=DIM, mono=True)


def picture(slide, path, x, y, max_w, max_h, crop_frac=None):
    if not os.path.exists(path):
        raise SystemExit(f"figure missing: {path}")
    src = path
    if crop_frac:
        im = Image.open(path)
        src = os.path.join(SCRATCH, f"ppt_{crop_frac}_"
                                    f"{os.path.basename(path)}")
        im.crop((0, 0, im.size[0], int(im.size[1] * crop_frac))).save(src)
    iw, ih = Image.open(src).size
    sc = min(max_w / iw, max_h / ih)
    slide.shapes.add_picture(src, x + Emu(int((max_w - iw * sc) / 2)), y,
                             width=Emu(int(iw * sc)),
                             height=Emu(int(ih * sc)))


def table(slide, rows, x, y, w, h, size=12, head=True):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = bool(head and i == 0)
                    r.font.color.rgb = INK
                    r.font.name = "Calibri"
    return t


def fig_slide(prs, head, sub, path, note=None, pathline=None, crop=None,
              box=(.35, 1.3, 12.6, 5.2)):
    s = blank(prs)
    title(s, head, sub)
    picture(s, path, Inches(box[0]), Inches(box[1]), Inches(box[2]),
            Inches(box[3]), crop_frac=crop)
    if note:
        text(s, note, Inches(.45), Inches(box[1] + box[3] + .12),
             Inches(12.4), Inches(.8), size=11.5, color=DIM)
    if pathline:
        path_line(s, pathline)
    return s


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------- 1 title
    s = blank(prs)
    text(s, "CEA-Ntsr1 two-plane calcium imaging", Inches(.9), Inches(2.1),
         Inches(11.5), Inches(.8), size=33, bold=True)
    text(s, "Pipeline, QC and first results - sessions of 2026-09-11",
         Inches(.9), Inches(3.0), Inches(11.5), Inches(.5), size=18,
         color=ACC)
    text(s,
         "pain, 10 min pin prick + heat   |   open field, 5.4 min\n"
         "Everything below is measured on these two recordings.\n"
         "Still to do: manual scoring of the pain video, then event-locked "
         "responses.",
         Inches(.9), Inches(3.7), Inches(11.5), Inches(1.3), size=14,
         color=DIM)
    path_line(s, r"code C:\Users\hsollim\OneDrive\Desktop\cursor"
                 r"\pain_2plane_split   |   results <session>\output_split\ ")

    # ---------------------------------------------------- 2 the data
    s = blank(prs)
    title(s, "The data",
          "Both sessions image the SAME two depths, frame-interleaved - "
          "recorded in the TDMS, not assumed")
    table(s, [
        ["", "open field", "pain"],
        ["raw frames per channel", "3000", "6000"],
        ["frames per plane", "1500", "3000"],
        ["duration", "325.5 s", "651.2 s"],
        ["rate per plane", "4.6054 Hz", "4.6054 Hz"],
        ["FOV", "440 x 512 px at 0.898 um/px", "same"],
        ["plane A = Slice 1", "ETL 65 um (139.6 abs)", "same"],
        ["plane B = Slice 2", "ETL 15 um (89.6 abs), +109 ms", "same"],
        ["lost frames", "0", "0"],
        ["behaviour camera", "25 Hz, 8159 frames", "25 Hz, 16299 / 16296"],
    ], Inches(.6), Inches(1.45), Inches(7.2), Inches(4.3), size=13)
    text(s,
         "CellVideo_CHA_Info.tdms carries a Slice channel, so which depth\n"
         "each frame belongs to is recorded by the acquisition software.\n\n"
         "Measured: Slice alternates 1,2,1,2 strictly in both sessions and\n"
         "odd raw frames are Slice 1 - exactly what the split step assumes.\n\n"
         "etl1data.csv is byte-identical between the sessions, so the\n"
         "\"(100_50um)\" in the open-field folder name is not the depths.\n\n"
         "Same depths in both sessions is what makes cross-session cell\n"
         "matching possible at all (slides 19-20).",
         Inches(8.1), Inches(1.55), Inches(4.7), Inches(4.2), size=13)
    path_line(s, r"<session>\CellVideo1\Information-CHA.txt , "
                 r"CellVideo_CHA_Info.tdms , etl1data.csv")

    # ---------------------------------------------------- 3 split
    s = blank(prs)
    title(s, "Step 1 - split the two depths",
          "The previous pipeline max-projected them, which merges neurons")
    picture(s, os.path.join(PA, "plane_split_qc.png"), Inches(.5),
            Inches(1.4), Inches(7.5), Inches(3.1))
    text(s,
         "Why not max-project\n"
         "  5 of 17 cells at one depth sit within a cell radius of a cell\n"
         "  at the other, so ~29 % get merged, and max() is nonlinear - the\n"
         "  projected value flips between the two neurons in 50 % of\n"
         "  consecutive frames.\n\n"
         "Result\n"
         "  corr(mean A, mean B) = 0.592 pain, 0.564 open field, well below\n"
         "  1, so the depths hold different cells.",
         Inches(8.3), Inches(1.45), Inches(4.5), Inches(3.2), size=12.5)
    text(s, "pain 6000 -> 3000 frames per plane in 27 s   |   open field "
            "3000 -> 1500 in 17 s   |   0 frames lost",
         Inches(.5), Inches(4.75), Inches(12.3), Inches(.4), size=13,
         bold=True, color=GOOD)
    path_line(s, r"pain_2plane_step1_split_planes.m  ->  "
                 r"<session>\output_split\plane_A.tif , plane_B.tif")

    # ---------------------------------------------------- 4 ActSort
    s = blank(prs)
    title(s, "Blocker - ActSort was silently replacing EXTRACT's code",
          "EXTRACT died on every plane; the cause was the MATLAB path, not "
          "the data")
    text(s,
         "Not enough input arguments.\n"
         "    get_circularity_metrics line 12\n"
         "    remove_redundant line 22\n"
         "    run_extract line 471\n"
         "    extractor line 408",
         Inches(.55), Inches(1.45), Inches(5.4), Inches(1.3), size=12,
         mono=True, color=BAD)
    text(s,
         "ActSort ships its own fork of 20 EXTRACT helper files, 19 of which\n"
         "differ. addpath prepends, so ActSort won every name. Its\n"
         "get_circularity_metrics takes a 4th argument 'parallel' that\n"
         "EXTRACT never passes - and line 12 is \"if parallel\".\n\n"
         "Worse than a crash: the 73 and 60 candidate cells had already been\n"
         "found using ActSort's noise and SNR estimators. It corrupts the\n"
         "numbers before it stops. ActSort is in the SAVED MATLAB path\n"
         "(15 folders), so this affects every MATLAB session on this PC.",
         Inches(.55), Inches(2.95), Inches(6.6), Inches(2.6), size=12.5)
    table(s, [
        ["shadowed file", "EXTRACT", "ActSort"],
        ["get_circularity_metrics.m", "25 lines, 3 args", "46 lines, 4 args"],
        ["get_trace_noise.m", "49 lines", "6 lines"],
        ["kappa_of_epsilon.m", "22 lines", "3 lines"],
        ["estimate_noise_std, get_trace_snr,", "all differ", "all differ"],
        ["filter_images, smooth_images, ...", "(20 files)", ""],
    ], Inches(7.4), Inches(1.45), Inches(5.4), Inches(2.1), size=11.5)
    text(s,
         "Fix: rmpath every ActSort folder, then assert that all 141 EXTRACT\n"
         "functions resolve inside the EXTRACT tree. Load ActSort only after\n"
         "extraction, in a fresh session.\n\n"
         "After the fix all four planes completed: 234 s and 209 s (pain),\n"
         "123 s and 112 s (open field).",
         Inches(7.4), Inches(3.8), Inches(5.4), Inches(1.8), size=12.5,
         color=GOOD)
    path_line(s, r"mini2p_toolbox_paths.m (the guard)   |   "
                 r"EXTRACT_QC_20260911.md Finding 7")

    # ---------------------------------------------------- 5 EXTRACT
    s = blank(prs)
    title(s, "Step 2 - EXTRACT per plane",
          "Cells found on the 4x binned movie, traces re-solved at the full "
          "4.6 Hz")
    picture(s, os.path.join(PA, "plane_A", "cell_overlay_full_FOV.png"),
            Inches(.3), Inches(1.3), Inches(4.1), Inches(4.2))
    picture(s, os.path.join(OF, "plane_B", "cell_overlay_full_FOV.png"),
            Inches(4.5), Inches(1.3), Inches(4.1), Inches(4.2))
    text(s, "pain plane A, 12 cells", Inches(.3), Inches(5.55), Inches(4.1),
         Inches(.3), size=11.5, align=PP_ALIGN.CENTER, color=DIM)
    text(s, "open field plane B, 14 cells", Inches(4.5), Inches(5.55),
         Inches(4.1), Inches(.3), size=11.5, align=PP_ALIGN.CENTER,
         color=DIM)
    table(s, [
        ["", "OF A", "OF B", "PA A", "PA B"],
        ["frames", "1500", "1500", "3000", "3000"],
        ["noise std", "14.2", "14.3", "20.7", "21.3"],
        ["candidates", "38", "58", "73", "60"],
        ["after quality", "17", "14", "14", "18"],
        ["after morphology", "14", "14", "12", "15"],
    ], Inches(8.9), Inches(1.4), Inches(3.9), Inches(2.4), size=12)
    text(s,
         "Settings that matter\n"
         "  use_zscore_before_extract = false, so (F-F0)/F0 is real dF/F\n"
         "  find on binned / trace on unbinned, so a 2 s window holds\n"
         "  9 samples instead of 2\n"
         "  preset low_snr, avg_cell_radius 8 (measured median 7.9 px)\n\n"
         "The open-field movie is the cleaner one: noise std 14.2 against\n"
         "20.8, background lag-1 autocorrelation 0.07 against 0.40.",
         Inches(8.9), Inches(4.0), Inches(3.9), Inches(2.7), size=11)
    path_line(s, r"pain_2plane_step2_extract.m , split_master.m  ->  "
                 r"<session>\output_split\plane_A\ , plane_B\ ")

    # ---------------------------------------------------- 6 QC null
    s = blank(prs)
    title(s, "Step 3 - is it a cell, or background?",
          "At 4.6 Hz and SNR 0.75 a real cell's trace also looks like noise "
          "by eye, so the null is measured")
    picture(s, os.path.join(PA, "plane_A", "curated", "qc_cells",
                            "qc_cells_null.png"),
            Inches(.4), Inches(1.3), Inches(12.5), Inches(2.5))
    text(s,
         "Method   200 control ROIs cut from the SAME motion-corrected "
         "movie, with the SAME\n"
         "         footprint shapes moved to places >= 12 px from any "
         "detected cell, traces taken\n"
         "         by the identical weighted average. Metrics on detrended "
         "dF/F (20 s running\n"
         "         median): background raw autocorrelation is 0.40, so "
         "without detrending the\n"
         "         test measures drift, not calcium.\n"
         "Test     Fisher combination of the empirical p-values for skew and "
         "lag-1\n"
         "         autocorrelation, calibrated on the same statistic over "
         "the 200 controls.\n"
         "         Kept at p <= 0.05, so 5 % of background would pass by "
         "construction.",
         Inches(.45), Inches(3.95), Inches(8.0), Inches(2.8), size=11.5)
    table(s, [
        ["verdict", "OF A", "OF B", "PA A", "PA B", "all"],
        ["curated cells", "12", "16", "11", "16", "55"],
        ["active", "5", "5", "6", "6", "22"],
        ["silent", "7", "9", "4", "10", "30"],
        ["SUSPECT", "0", "2", "1", "0", "3"],
        ["usable", "4", "5", "6", "6", "21"],
    ], Inches(8.7), Inches(4.0), Inches(4.1), Inches(2.1), size=12)
    text(s, "silent = no signal but the footprint IS brighter than its "
            "surround: a real soma that did not fire",
         Inches(8.7), Inches(6.25), Inches(4.1), Inches(.6), size=10.5,
         color=GOOD)
    path_line(s, r"extract_qc_cells.py --curated  ->  "
                 r"<plane>\curated\qc_cells\ ")

    # ---------------------------------------------------- 7-9 curation
    s = blank(prs)
    title(s, "Step 4 - manual curation, decided on these sheets",
          "The automatic QC measures brightness against the surround and "
          "says nothing about SHAPE, so a bright fibre passes it")
    picture(s, os.path.join(PA, "curation", "curation_pain_A_tiles.png"),
            Inches(.35), Inches(1.35), Inches(7.2), Inches(5.3))
    text(s,
         "Left half of each tile = MEAN image, right half = MAX over time.\n"
         "Brightness stretched on the WHOLE plane, never per crop, so a dim\n"
         "candidate looks dim.\n\n"
         "Eccentricity was added after this view: pain A #6 and #12 are\n"
         "visibly fibres and are the two highest eccentricities in the plane\n"
         "(0.934, 0.941), just under the 0.95 the filter allowed.",
         Inches(7.8), Inches(1.45), Inches(5.1), Inches(1.9), size=11.5)
    table(s, [
        ["plane", "before", "decision", "after"],
        ["OF A", "14", "discard 7, 13", "12"],
        ["OF B", "14", "discard 9; trim 14 to its large blob;\n"
                       "add N3, N4, N5 (missed somata)", "16"],
        ["PA A", "12", "discard 9 (contour empty, anat 0.001)", "11"],
        ["PA B", "15", "split 11 into two cells 11a / 11b", "16"],
    ], Inches(7.8), Inches(3.6), Inches(5.1), Inches(2.2), size=11)
    text(s, "An interactive page was used to click these: "
            "curation_select.html",
         Inches(7.8), Inches(6.2), Inches(5.1), Inches(.4), size=11,
         color=ACC)
    path_line(s, r"cell_curation_sheet.py , make_curation_page.py  ->  "
                 r"<session>\output_split\curation\ ")

    fig_slide(
        prs, "Step 4 - the decisions on the field of view",
        "open field plane B: 1 discarded, 1 trimmed, 3 added",
        os.path.join(OF, "plane_B", "curated", "curation_on_fov.png"),
        note="Green = kept. Red dashed = discarded (#9 sits on fibrous "
             "texture, not a soma). Orange = trimmed, dotted outline is the "
             "original two blobs and solid is the large one that was kept. "
             "Cyan = the three somata EXTRACT missed.",
        pathline=r"plot_curation_on_fov.py  ->  "
                 r"<plane>\curated\curation_on_fov.png",
        box=(2.6, 1.25, 8.1, 5.2))

    fig_slide(
        prs, "Step 4 - zoom on what changed",
        "trim and the three added somata, open field plane B",
        os.path.join(OF, "plane_B", "curated", "curation_on_fov_zoom.png"),
        note="N3 and N4 are faint (band-pass peak 1.6 and 2.2 against 3.5 "
             "for N5 and 17.2 for #4), so whether they are dim cells or "
             "neuropil is left to the QC on their traces - both came out "
             "silent.",
        pathline=r"<plane>\curated\curation_on_fov_zoom.png",
        box=(.5, 1.25, 12.3, 5.2))

    # ---------------------------------------------------- 10 the bug
    s = blank(prs)
    title(s, "A bug I introduced, and how it was caught",
          "The curated export was scrambled by array order; detection itself "
          "was never touched")
    table(s, [
        ["one footprint, N3", "blobs", "largest area", "row span"],
        ["intended", "1", "212 px", "274-289"],
        ["what MATLAB built", "17", "16 px", "0-439 (whole FOV)"],
    ], Inches(.55), Inches(1.45), Inches(6.4), Inches(1.1), size=12.5)
    text(s,
         "numpy ravels row-major; MATLAB's reshape fills column-major. With\n"
         "440 != 512 it is not even a transpose - one soma scatters into\n"
         "stripes. Nothing errored, and every trace solved against it was\n"
         "wrong. Hansol spotted the stripes in the FOV figure.\n\n"
         "Two fixes\n"
         "  1  convert to Fortran order once, at the file boundary\n"
         "  2  MATLAB now asserts each footprint is one compact blob and\n"
         "     fails loudly, naming the likely cause, if it is not\n\n"
         "One earlier conclusion is RETRACTED. I had reported that EXTRACT's\n"
         "S-step relocates footprints by 127-143 px, making its T-step\n"
         "unusable. That was measured against the scrambled set. With the\n"
         "order fixed the movement is 0.8-8.1 px: EXTRACT was fine.",
         Inches(.55), Inches(2.75), Inches(6.4), Inches(3.9), size=12)
    text(s,
         "Verification now runs as a gate\n\n"
         "verify_curation.py checks the curated set against three\n"
         "independent sources - the original detection, the decisions typed\n"
         "in code, and the applied result - and exits non-zero on any\n"
         "mismatch. All checks pass:\n\n"
         "  original detection unchanged in all four planes\n"
         "  every curated footprint compact (row span 9-43 px)\n"
         "  discards exactly as decided\n"
         "  kept-unchanged footprints numerically identical to the original\n"
         "  trim 14: 2 blobs -> 1, 482 -> 371 px, subset of the original\n"
         "  N3, N4, N5 within 0.0 px of the requested positions\n"
         "  split 11 -> 11a 364 px + 11b 258 px = 622 px, a disjoint\n"
         "     partition of the original support, weights preserved exactly",
         Inches(7.3), Inches(1.45), Inches(5.5), Inches(5.2), size=12)
    path_line(s, r"verify_curation.py , apply_curation.py + .m")

    # ---------------------------------------------------- 11-13 per cell
    fig_slide(
        prs, "Per cell - where it is and what it does",
        "pain plane A, 11 curated cells",
        os.path.join(PA, "plane_A", "curated",
                     "cell_by_cell_location_trace.png"),
        note="dF/F is pulled from the RAW denoised movie with the curated "
             "footprints, F0 = 20th percentile. Traces are solved jointly "
             "across all footprints, so an overlapping neighbour does not "
             "leak in - measured to matter only for the one pair 25 px "
             "apart (r 0.93); every other cell is unchanged (r = 1.000).",
        pathline=r"plot_cell_by_cell.py  ->  "
                 r"<plane>\curated\cell_by_cell_location_trace.png",
        box=(.4, 1.25, 12.5, 5.1))

    fig_slide(
        prs, "Per cell - does the trace look like a calcium trace?",
        "pain plane A: 6 active, 4 silent, 1 SUSPECT",
        os.path.join(PA, "plane_A", "curated",
                     "cell_by_cell_evaluation.png"),
        note="Red marks = detected events. Shaded band on the "
             "autocorrelation = the plausible GCaMP decay range. Cells 1 and "
             "5 show repeated transients; cell 6 has a single clean one at "
             "620 s. Cells 7, 8, 10, 12 sit at tau 0.15-0.17 s, which is one "
             "frame - white noise. tau alone does not separate every cell "
             "(active runs 0.16-1.37 s), so the verdict uses skew and "
             "autocorrelation jointly against the background null.",
        pathline=r"<plane>\curated\cell_by_cell_evaluation.png",
        box=(.4, 1.25, 12.5, 5.0))

    fig_slide(
        prs, "Per cell - open field plane B",
        "16 curated cells including the three added somata (N3, N4, N5) and "
        "the trimmed #14",
        os.path.join(OF, "plane_B", "curated",
                     "cell_by_cell_evaluation.png"),
        note="All three added cells came out silent, and N4 is SUSPECT - the "
             "footprints are on real somata but they did not fire in 326 s. "
             "Adding them cost nothing and settled the question with data.",
        pathline=r"<plane>\curated\cell_by_cell_evaluation.png",
        crop=0.62,
        box=(.4, 1.25, 12.5, 5.0))

    # ---------------------------------------------------- 14 timestamps
    s = blank(prs)
    title(s, "Step 5 - one time base for both depths and the cameras",
          "Two clocks ~37 s apart, and the behaviour cameras dropped frames")
    table(s, [
        ["stream", "clock", "start - session start"],
        ["imaging CHA/Time", "imaging", "-36.64 s OF, -36.21 s pain"],
        ["SignalSync_N/Time", "imaging", "-37.41 s OF, -37.07 s pain"],
        ["MiceVideoN/Ref Time", "behaviour", "+0.21 s OF, +0.56 s pain"],
    ], Inches(.55), Inches(1.45), Inches(6.0), Inches(1.3), size=12)
    text(s,
         "This is NOT a 37 s misalignment of the data. The spans agree to\n"
         "within 0.05 s, so the recordings are simultaneous - only the\n"
         "timestamp clocks differ. Aligning by ABSOLUTE timestamps across the\n"
         "two clocks would put the data 37 s out; aligning by relative\n"
         "seconds from each stream's own start is off by only -0.58 s.",
         Inches(.55), Inches(3.0), Inches(6.0), Inches(1.8), size=12)
    table(s, [
        ["camera", "saved", "gaps", "missing", "sync pulses"],
        ["OF MiceVideo2", "8159", "5", "6", "8166"],
        ["PA MiceVideo1", "16299", "8", "11", "16311"],
        ["PA MiceVideo2", "16296", "10", "12", "16310"],
    ], Inches(6.9), Inches(1.45), Inches(5.9), Inches(1.3), size=12)
    text(s,
         "saved + missing reproduces the span to within 2-4 ms and lands 1-2\n"
         "pulses short of the sync count, so the residual alignment\n"
         "uncertainty is 40-80 ms, not the ~0.3 s a raw count difference\n"
         "would suggest.\n\n"
         "In the pain session BOTH cameras drop at the same frame indices\n"
         "(~2103-2107, 8799-8801, 13823-13830) - a system stall, not a camera\n"
         "fault. So behaviour frame index / 25 is NOT time; the error is\n"
         "cumulative and reaches 0.24-0.48 s by the end. My own open-field\n"
         "tracking output had that bug and was corrected by up to 775 ms.",
         Inches(6.9), Inches(3.0), Inches(5.9), Inches(2.7), size=12)
    text(s, "Result: 100 % of the imaging in both sessions has simultaneous "
            "behaviour video.",
         Inches(.55), Inches(5.9), Inches(12.3), Inches(.4), size=14,
         bold=True, color=GOOD)
    path_line(s, r"split_timestamps.py  ->  "
                 r"<session>\output_split\timestamps\ ")

    # ---------------------------------------------------- 15 tracking
    s = blank(prs)
    title(s, "Result 1 - open-field tracking",
          "The FOV is 8x the arena; only the white floor counts")
    picture(s, os.path.join(OF, "tracking", "openfield_track_qc.png"),
            Inches(.35), Inches(1.3), Inches(7.3), Inches(3.65))
    picture(s, os.path.join(OF, "tracking", "openfield_trajectory.png"),
            Inches(.35), Inches(5.05), Inches(7.3), Inches(1.8))
    text(s,
         "\"Largest dark blob in the frame\" tracked the dark right-hand\n"
         "background: a body of 923,038-955,852 px (half the frame) sitting\n"
         "still at x~1060, for 1632 of 1632 sampled frames.\n\n"
         "Threshold sweep on the session median image:\n"
         "  >180  640x680  climbs into the white wall tape\n"
         "  >220  571x625  ragged right edge\n"
         "  >245  469x527  the floor, clean\n"
         "Mask is the fitted rectangle, eroded 8 px; zones are the standard\n"
         "3x3 grid of the floor.",
         Inches(7.9), Inches(1.4), Inches(5.0), Inches(2.6), size=11.5)
    table(s, [
        ["metric", "value"],
        ["white floor", "454 x 516 px = 11.9 % of FOV"],
        ["frames analysed", "8159 of 8159 (5.44 min)"],
        ["detection rate", "99.96 %"],
        ["body area", "median 6411 px (827-10875)"],
        ["distance", "16476 px"],
        ["time moving", "46.0 % (>20 px/s)"],
        ["centre / corner / edge", "5.6 % / 41.4 % / 53.0 %"],
        ["body clipped at wall", "45.1 % of frames, flagged"],
    ], Inches(7.9), Inches(4.2), Inches(5.0), Inches(2.5), size=11.5)
    path_line(s, r"openfield_track.py  ->  "
                 r"<session>\output_split\tracking\ ")

    # ---------------------------------------------------- 16-17 place
    s = blank(prs)
    title(s, "Result 2 - centre versus corner neurons",
          "9 usable cells tested; 2 prefer the centre, 0 prefer the corners")
    picture(s, os.path.join(OF, "place", "place_summary.png"), Inches(.4),
            Inches(1.3), Inches(12.5), Inches(2.5))
    text(s,
         "Statistics   Zone contrast is the centre mean minus the corner "
         "mean of the detrended\n"
         "             z-scored dF/F. Its p-value is two-sided against a "
         "CIRCULAR-SHIFT null, which\n"
         "             keeps the trace's autocorrelation and the animal's "
         "occupancy and destroys\n"
         "             only the pairing. A t-test across frames would treat "
         "each frame as\n"
         "             independent and is not valid for calcium. 2000 "
         "surrogates, minimum shift\n"
         "             10 s. q-values are Benjamini-Hochberg across the 9 "
         "cells.\n"
         "Speed        reported per cell with its own shift test, because a "
         "\"centre cell\" could\n"
         "             just be a speed cell. 0 of 9 are speed-significant, "
         "so the zone effects\n"
         "             are not speed in disguise.",
         Inches(.45), Inches(3.95), Inches(8.1), Inches(2.9), size=11)
    table(s, [
        ["cell", "contrast", "q", "preference"],
        ["A#12", "+0.900", "0.0022", "centre"],
        ["A#3", "+0.520", "0.0022", "centre"],
        ["B#10", "+0.295", "0.1364", "none"],
        ["A#2", "+0.373", "0.3632", "none"],
        ["other 5", "-0.29 to +0.03", "> 0.5", "none"],
    ], Inches(8.8), Inches(4.0), Inches(4.0), Inches(2.0), size=12)
    text(s, "Centre occupancy is only 5.6 % (83 imaging frames against 623 "
            "in the corners), so a centre effect is much harder to detect "
            "than a corner one - absence of corner cells is weak evidence.",
         Inches(8.8), Inches(6.15), Inches(4.0), Inches(.8), size=10.5,
         color=DIM)
    path_line(s, r"openfield_place_cells.py  ->  "
                 r"<session>\output_split\place\ ")

    fig_slide(
        prs, "Result 2 - occupancy-normalised rate maps",
        "mean z-scored dF/F per spatial bin; white = under 0.5 s dwell",
        os.path.join(OF, "place", "place_ratemaps.png"),
        note="A#12 and A#3 are the two with a red centre. The maps are "
             "divided by occupancy, so the centre is not just brighter for "
             "being visited less.",
        pathline=r"<session>\output_split\place\place_ratemaps.png",
        box=(1.2, 1.25, 10.9, 5.2))

    # ---------------------------------------------------- 18-19 matching
    s = blank(prs)
    title(s, "Result 3 - following the same neuron across the two sessions",
          "By pixel position, after registering the fields - the cell IDs "
          "themselves do NOT correspond")
    picture(s, os.path.join(PA, "match", "match_A.png"), Inches(.4),
            Inches(1.25), Inches(12.5), Inches(2.6))
    table(s, [
        ["", "plane A", "plane B"],
        ["registration shift", "dy +9, dx -17 px", "dy -11, dx -7 px"],
        ["image correlation there", "+0.872", "+0.876"],
        ["same depth, wrong session", "-", "-"],
        ["against the OTHER depth", "+0.328", "+0.435"],
        ["pairs within 8 px", "8 of 11", "13 of 16"],
        ["median residual", "1.98 px", "2.00 px"],
        ["footprint shape r >= 0.5", "5 of 8", "8 of 13"],
    ], Inches(.5), Inches(4.05), Inches(5.9), Inches(2.6), size=11.5)
    text(s,
         "Each depth matches ITSELF across the two sessions at r ~ 0.87, "
         "against 0.33-0.44\n"
         "for the wrong depth - the same optical planes really were imaged "
         "29 minutes apart.\n\n"
         "Two gates, because position alone is not enough in a dense plane:\n"
         "  position   within one cell radius (8 px) after registration\n"
         "  shape      footprint correlation >= 0.5 after alignment\n"
         "13 of 21 position pairs pass both and are treated as the same "
         "neuron.\n\n"
         "Phase correlation was tried first and failed: it returned "
         "contradictory shifts\n"
         "between the depths (0 px for B, -21.7 px for A) and found no pairs "
         "at all. I had\n"
         "reported \"no cell can be followed across sessions\" on that basis; "
         "that conclusion\n"
         "is RETRACTED. A direct correlation search over shifts is what the "
         "numbers above\n"
         "come from, and it can be checked - which is why the wrong-depth "
         "controls are shown.",
         Inches(6.6), Inches(4.0), Inches(6.2), Inches(2.8), size=11)
    path_line(s, r"match_sessions.py  ->  "
                 r"<pain session>\output_split\match\ ")

    s = blank(prs)
    title(s, "Result 3 - the join that makes the question askable",
          "13 neurons followed across both sessions; 4 usable in both")
    picture(s, os.path.join(PA, "match", "linked_cells.png"), Inches(.35),
            Inches(1.25), Inches(7.6), Inches(5.4), crop_frac=0.55)
    table(s, [
        ["open field", "place", "contrast", "q", "-> pain", "shape r",
         "pain"],
        ["A#12", "centre", "+0.900", "0.0022", "A#3", "0.953", "active"],
        ["A#3", "centre", "+0.520", "0.0022", "A#7", "0.890", "silent"],
        ["A#2", "none", "+0.373", "0.3632", "A#5", "0.719", "active"],
        ["A#1", "none", "-0.293", "0.5325", "A#1", "0.696", "active"],
        ["B#13", "none", "+0.030", "0.9635", "B#7", "0.746", "active"],
    ], Inches(8.2), Inches(1.4), Inches(4.7), Inches(2.2), size=10.5)
    text(s,
         "Both centre-preferring cells could be followed into the pain\n"
         "session. A#12 - the strongly centre-preferring one - is A#3 there\n"
         "and is ACTIVE, so once the video is scored the question becomes\n"
         "directly answerable:\n\n"
         "   was the neuron that preferred the exposed centre also\n"
         "   pin-prick responsive?\n\n"
         "Caveat: \"same neuron\" rests on a footprint correlation of 0.5 in a\n"
         "dense plane, with residuals of 2-7 px against an 8 px radius.\n"
         "Treat the pairs as probable, not certain.",
         Inches(8.2), Inches(3.8), Inches(4.7), Inches(2.9), size=11.5)
    path_line(s, r"linked_cells_report.py  ->  "
                 r"<pain session>\output_split\match\linked_cells.csv , "
                 r"linked_cells.png")

    # ---------------------------------------------------- 20-21 pain traces
    fig_slide(
        prs, "Result 4 - pain session, raw Ca per cell",
        "pain plane B, 16 curated cells, fluorescence in movie units with no "
        "baseline removed",
        os.path.join(PA, "plane_B", "curated", "pain_traces_raw.png"),
        note="Raw F shows absolute brightness, bleaching and drift, which "
             "dF/F hides. Blue = active, amber = silent, red = SUSPECT.",
        pathline=r"pain_cell_traces.py  ->  "
                 r"<plane>\curated\pain_traces_raw.png , pain_traces.mat",
        box=(.5, 1.3, 12.3, 5.1))

    fig_slide(
        prs, "Result 4 - pain session, z-score per cell",
        "detrended dF/F divided by its robust sd (1.4826 x MAD)",
        os.path.join(PA, "plane_B", "curated", "pain_traces_zscore.png"),
        note="Detrended with a 20 s running median first, because background "
             "ROIs here carry a raw lag-1 autocorrelation of 0.40 - without "
             "that the z-score is dominated by drift. Amplitudes are now "
             "comparable between cells with different baselines; |z| reaches "
             "13.2.",
        pathline=r"<plane>\curated\pain_traces_zscore.png , "
                 r"pain_traces.csv",
        box=(.5, 1.3, 12.3, 5.1))

    # ---------------------------------------------------- 22 next
    s = blank(prs)
    title(s, "Ready for event locking, and where everything is")
    text(s,
         "What is waiting on the manual scoring\n"
         "  Every trace already carries a time vector on the same base as "
         "the behaviour cameras\n"
         "  (seconds since plane A frame 1, imaging clock), so a stimulus "
         "time in that base\n"
         "  indexes straight into it. lock_events() in pain_cell_traces.py "
         "takes the scored\n"
         "  times and returns (cells x events x lag) windows, dropping a "
         "window rather than\n"
         "  padding it if it would run off either end.\n"
         "  Then: pin-prick / heat / non-responsive / behaviour-responsive "
         "per cell, and the\n"
         "  linked table lets a single neuron carry both its open-field and "
         "its pain answer.",
         Inches(.55), Inches(1.3), Inches(12.3), Inches(2.2), size=12.5)
    table(s, [
        ["what", "where"],
        ["all code", r"C:\Users\hsollim\OneDrive\Desktop\cursor\pain_2plane_split\ "],
        ["measured QC report", r"...\pain_2plane_split\EXTRACT_QC_20260911.md"],
        ["EXTRACT per plane", r"<session>\output_split\plane_A\ , plane_B\ "],
        ["curated cells, traces, per-cell figures", r"<plane>\curated\ "],
        ["cell QC against background", r"<plane>\curated\qc_cells\ "],
        ["curation sheets and the click page", r"<session>\output_split\curation\ "],
        ["timestamps", r"<session>\output_split\timestamps\ "],
        ["open-field tracking / place cells", r"<session>\output_split\tracking\ , place\ "],
        ["cross-session matching", r"<pain session>\output_split\match\ "],
    ], Inches(.55), Inches(3.6), Inches(12.3), Inches(3.1), size=11.5)
    path_line(s, r"nothing in <session>\output\ is touched - that is the "
                 r"other agent's max-projection pipeline")

    # If the deck is open in PowerPoint the file is locked, and the previous
    # attempt destroyed it: python-pptx truncates the target before it finds
    # out it cannot write. So write to a fresh version instead of fighting
    # an open document.
    base = os.path.join(HERE, "CEANTSR1_pipeline_20260911")
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
    n = len(prs.slides._sldIdLst)
    print(f"wrote {out}  ({n} slides)")
    return out


if __name__ == "__main__":
    build()
