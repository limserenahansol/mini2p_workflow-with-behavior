"""make_processing_ppt.py  -  everything up to the final cell selection.

Split from the results deck on purpose: this one is the audit trail, so
anyone asking "how do you know these are cells" has it in one place, and
the results deck can start from a finished cell set.

  1  the recording, and why the two depths are kept apart
  2  motion correction and EXTRACT, measured stage by stage
  3  is each footprint a cell? the background-calibrated test
  4  manual curation, and proof that it was applied as decided
  5  one time base for imaging and the cameras
  6  the same cells in both sessions, and the separability trap
  7  the final cell set

Every mistake that was found and fixed is in here with its evidence, in
the order it was found. That is the point of the deck.

OUTPUT  ->  <this folder>\\CEANTSR1_processing_<date>.pptx

USAGE
  python make_processing_ppt.py
"""
from __future__ import annotations

import os

import pandas as pd
from pptx.util import Inches

from deck import (ACC, DIM, GOOD, WARN, blank, bullets, fig, new_deck,
                  path_line, save, section, table, text, title)

HERE = os.path.dirname(os.path.abspath(__file__))
OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
STAMP = "20260914"


def build():
    U = pd.read_csv(os.path.join(PA, "match", "union_cells.csv"))
    n_union = len(U)
    n_match = int(U["matched"].sum())
    p = new_deck()

    s = blank(p)
    text(s, "CEA-Ntsr1 two-plane calcium imaging", Inches(.9), Inches(1.9),
         Inches(11.5), Inches(.9), size=34, bold=True)
    text(s, "Processing - from the raw movie to the final cell set",
         Inches(.9), Inches(2.9), Inches(11.5), Inches(.5), size=19,
         color=ACC)
    text(s,
         "Sessions of 2026-09-11: pain 10 min (pin + heat), open field "
         "5.4 min.\n"
         "Every number here is measured on these two recordings, not "
         "quoted from a default.\n\n"
         "The results are in the RESULTS deck. This one is the audit "
         "trail: how the cells were found,\n"
         "what was rejected, and every mistake that had to be fixed along "
         "the way.",
         Inches(.9), Inches(3.6), Inches(11.5), Inches(2.0), size=14,
         color=DIM)
    path_line(s, r"nothing in <session>\output\ is touched - that is the "
                 r"other pipeline's folder")

    # ------------------------------------------------------- 1 the data
    section(p, 1, "The recording",
            "Two ETL depths, interleaved frame by frame.\n"
            "Why they are analysed apart instead of max-projected "
            "together.")
    s = blank(p)
    title(s, "Two depths, one movie, interleaved",
          "recorded in the Slice channel of the TDMS - read, not assumed")
    table(s, [
        ["", "open field", "pain"],
        ["raw frames per channel", "3000", "6000"],
        ["frames per plane", "1500", "3000"],
        ["duration", "325.5 s", "651.2 s"],
        ["rate per plane", "4.61 Hz", "4.61 Hz"],
        ["plane A = Slice 1", "ETL 65 um", "same"],
        ["plane B = Slice 2", "ETL 15 um, +109 ms", "same"],
    ], Inches(.6), Inches(1.6), Inches(7.4), Inches(3.0), size=13)
    text(s,
         "Why not max-project the two depths into one movie, which is the "
         "other pipeline's approach:\n\n"
         "  5 of 17 cells at one depth sit within a cell radius of a cell "
         "at the other, so ~29 % would merge\n"
         "  max() is nonlinear and neither depth dominates, so the "
         "projected value flips between the two\n"
         "  neurons in 50 % of consecutive frames\n\n"
         "The two sessions image the SAME depths: etl1data.csv is "
         "byte-identical between them, so the\n"
         "(100_50um) in the open-field folder name is not the depth.",
         Inches(8.3), Inches(1.6), Inches(4.5), Inches(4.4), size=12)
    path_line(s, r"pain_2plane_step1_split_planes.m  ->  "
                 r"<session>\output_split\plane_A.tif , plane_B.tif")

    bullets(p, "The bug that had to be fixed before anything would run",
            "ActSort silently replaces EXTRACT's own code",
            "ActSort ships its own fork of 20 EXTRACT helper files, 19 of "
            "which differ, and it was in the\n"
            "SAVED MATLAB path (15 folders). addpath prepends, so ActSort "
            "won every name. EXTRACT died with\n\n"
            "      Not enough input arguments.\n"
            "          get_circularity_metrics line 12\n\n"
            "because ActSort's version takes a 4th argument the caller "
            "never passes.\n\n"
            "Worse than the crash: cell finding had ALREADY run on "
            "ActSort's noise and SNR estimators, so the\n"
            "cell counts were wrong before anything errored - a silent "
            "failure that happened to become loud.\n\n"
            "mini2p_toolbox_paths.m now rmpaths every ActSort folder and "
            "asserts that all 141 EXTRACT functions\n"
            "resolve inside the EXTRACT tree before extraction starts.",
            size=13, color=WARN,
            pathline=r"mini2p_toolbox_paths.m")

    # ------------------------------------------------ 2 processing stages
    section(p, 2, "Motion correction and EXTRACT",
            "Each stage judged by the metric it is supposed to fix, not "
            "by one number that always rises.")
    for ses, root, nm in (("pain", PA, "pain"),
                          ("open field", OF, "open field")):
        for plane in ("A", "B"):
            f = os.path.join(root, f"plane_{plane}", "curated",
                             "fig1_processing.png")
            if os.path.exists(f) and plane == "A":
                fig(p, f"{nm} plane {plane} - raw to what EXTRACT sees",
                    "One metric per stage, each marked with the stage it "
                    "judges. Two metrics become invalid after the "
                    "band-pass, which removes the mean - they are marked "
                    "n/a rather than quoted.",
                    f, rf"<{nm}>\output_split\plane_{plane}\curated"
                       rf"\fig1_processing.png", hbox=5.3)

    # ---------------------------------------------- 3 is it a cell
    section(p, 3, "Is each footprint a cell?",
            "Two separate questions, because no signal and not a cell are "
            "different claims.")
    bullets(p, "The test, and why the null is measured rather than assumed",
            "at 4.6 Hz with a single-frame SNR of 0.75 a real cell also "
            "looks like noise by eye",
            "DOES IT HAVE CALCIUM SIGNAL?\n"
            "  200 control ROIs are cut from the SAME motion-corrected "
            "movie, using the same footprint shapes\n"
            "  moved to places containing no detected cell, and their "
            "traces taken by the identical method.\n"
            "  Skewness and lag-1 autocorrelation of the detrended dF/F "
            "are combined with Fisher's method and\n"
            "  calibrated against those 200 controls, so the cut sits at a "
            "5 % false-positive rate on background.\n\n"
            "  Detrending (20 s running median) is not optional: "
            "background ROIs carry a raw lag-1\n"
            "  autocorrelation of 0.40, so on raw traces the test measures "
            "drift, not calcium.\n\n"
            "IS THERE A SOMA THERE?\n"
            "  The footprint's brightness over the ring around it, in the "
            "session mean image. Background\n"
            "  median 0.000, p95 0.006-0.010.\n\n"
            "  active   signal yes                        fired during the "
            "session\n"
            "  silent   no signal, but brighter than its ring   a real "
            "soma that did not fire\n"
            "  SUSPECT  neither                           dropped\n\n"
            "The active/silent line is a threshold, not a state: pain "
            "plane B #7 passed at p = 0.0498 and #10\n"
            "failed at 0.0547 with practically identical metrics. Treat "
            "the labels as a ranking.",
            size=12.5,
            pathline=r"extract_qc_cells.py  ->  <plane>\curated\qc_cells\ ")
    for nm, root in (("pain", PA),):
        fig(p, f"{nm} plane A - each cell against its own local control",
            "Every cell next to its own footprint shape moved to the "
            "nearest clear spot in the same movie: same tissue, same "
            "noise, same extraction, no soma.",
            os.path.join(root, "plane_A", "curated", "fig2_detection.png"),
            rf"<{nm}>\output_split\plane_A\curated\fig2_detection.png",
            hbox=5.3)

    # ---------------------------------------------- 4 curation
    section(p, 4, "Manual curation",
            "Hansol's keep/discard decisions, and proof that what was "
            "applied is what was decided.")
    for nm, root, plane in (("pain", PA, "A"), ("open field", OF, "B")):
        f = os.path.join(root, "curation",
                         f"curation_{'pain' if nm == 'pain' else 'openfield'}"
                         f"_{plane}_overview.png")
        if os.path.exists(f):
            fig(p, f"{nm} plane {plane} - the selection page",
                "Every candidate numbered on the field of view, with its "
                "own tile and trace, so a decision can be made on the "
                "evidence rather than on a list of numbers.",
                f, rf"<{nm}>\output_split\curation\ ", hbox=5.3)
    bullets(p, "Two mistakes found here, and how they are prevented now",
            "both silent, both caught by checking rather than by an error",
            "1  numpy writes row-major, MATLAB reshapes column-major\n"
            "   Footprints handed to MATLAB as an (h*w, k) matrix with "
            "numpy's default C-order ravel were\n"
            "   scattered by MATLAB's reshape: one 212 px soma became 17 "
            "stripes of 16 px spanning all 440\n"
            "   rows. Nothing errored, and every trace solved against it "
            "was wrong. With 440 != 512 it is not\n"
            "   even a transpose.\n"
            "   -> footprints are re-raveled in Fortran order once, at the "
            "file boundary, and the MATLAB side\n"
            "      asserts each footprint is a single compact blob before "
            "using it.\n\n"
            "2  splitting a footprint leaked 22 px outside the original\n"
            "   Dilating each half by 9 px pulled in the parent's "
            "sub-threshold skirt.\n"
            "   -> the support is partitioned by nearest blob instead. "
            "11a 364 px + 11b 258 px = 622 px exactly,\n"
            "      zero overlap, weights preserved.\n\n"
            "verify_curation.py now gates the pipeline: it checks the "
            "curated set against the original detection,\n"
            "against the typed decisions, and against the applied result, "
            "and exits non-zero on any mismatch.",
            size=12.5, color=WARN,
            pathline=r"apply_curation.py , verify_curation.py")

    # ---------------------------------------------- 5 time base
    section(p, 5, "One time base",
            "Two clocks, two cameras, dropped frames. Everything later "
            "depends on this being right.")
    bullets(p, "Two things that look like time and are not",
            "both measured on these recordings",
            "THE TWO CLOCKS ARE 37 s APART\n"
            "  The imaging side (CHA/Time, SignalSync_N/Time) runs ~37 s "
            "behind the behaviour side\n"
            "  (MiceVideoN/Ref Time). The recordings are simultaneous - "
            "the spans agree to within 0.05 s - so\n"
            "  this is a clock offset, not a delay. Comparing absolute "
            "timestamps puts the data 37 s out.\n\n"
            "FRAME INDEX / 25 IS NOT TIME\n"
            "  The cameras dropped frames: 6 in the open field, 11 and 12 "
            "in the pain session, clustered, and\n"
            "  in the pain session BOTH cameras drop at the same indices "
            "(a system stall, not a camera fault).\n"
            "  The error is cumulative and reaches 0.44 s by the end - "
            "larger than the human's reaction time.\n\n"
            "Both are handled by anchoring each camera to its sync pulse "
            "and asserting the gap arithmetic\n"
            "closes to within 50 ms. Result: 100 % of the imaging in both "
            "sessions has simultaneous behaviour\n"
            "video, and every event time used later comes from the "
            "per-frame Ref Time, never from an index.",
            size=13,
            pathline=r"split_timestamps.py  ->  <session>\output_split"
                     r"\timestamps\ ")

    # ---------------------------------------------- 6 both sessions
    section(p, 6, "The same cells in both sessions",
            f"{n_union} neurons measured in both, {n_match} of them "
            f"detected independently twice.")
    bullets(p, "Why the cell set is not just 'what EXTRACT found twice'",
            "that is a detection limit, not an anatomical one",
            "Matching detections across sessions caps the set at the "
            "neurons EXTRACT happened to find twice.\n"
            "The audit shows that is not an anatomy limit: 25 of 27 pain "
            "cells have a soma at their location in\n"
            "the OPEN-FIELD mean image, and 20 of 28 open-field cells have "
            "one in the PAIN mean image. Nearly\n"
            "every cell is visible in both. EXTRACT only finds cells that "
            "fluctuate, so a neuron that was quiet\n"
            "in one session gives it nothing to find there.\n\n"
            "So the union of both sessions' footprints is registered and "
            "extracted from BOTH movies. Detection\n"
            f"then has to succeed once per neuron instead of twice: "
            f"{n_union} cells, each with a trace in each session,\n"
            "under one shared name.\n\n"
            "Registration is a direct correlation search over shifts. "
            "Phase correlation was tried first and\n"
            "failed - contradictory shifts between the depths (0 px for B, "
            "-21.7 px for A) and no pairs at all.\n"
            "I reported \"no cell can be followed across sessions\" on "
            "that basis; that conclusion is RETRACTED.\n"
            "Each depth matches ITSELF across sessions at r ~ 0.87 against "
            "0.33-0.44 for the wrong depth,\n"
            "which is the control that makes the registration checkable.",
            size=12.5,
            pathline=r"match_sessions.py , match_audit.py , "
                     r"transfer_footprints.py")
    fig(p, "The trap in building that set, and what it cost",
        "Two footprints on one soma cannot both have a trace. Found by "
        "checking, reported because it changed a result.",
        os.path.join(PA, "match", "fig5_separability.png"),
        r"transfer_footprints.py , fig5_separability.py  ->  "
        r"<pain>\output_split\match\fig5_separability.png", hbox=5.2,
        note="An open-field footprint can sit ON TOP of a pain footprint "
             "and still fail the shape test, so the same soma entered the "
             "design matrix twice - 8 times over the two planes. Least "
             "squares then splits one soma's signal into a positive and a "
             "negative copy: 7 of the 8 pairs came out anti-correlated at "
             "r = -0.25 to -0.83. Two of those negative copies had been "
             "reported as corner-preferring cells.")

    # ---------------------------------------------- 7 final
    section(p, 7, "The final cell set",
            "What the results deck starts from.")
    fig(p, "Pain session - the final cells",
        "This is the set every result is computed on. Grey = footprint "
        "not on a soma in this session.",
        os.path.join(PA, "atlas", "fig11_fov_pain.png"),
        r"<pain>\output_split\atlas\fig11_fov_pain.png", hbox=5.3)
    fig(p, "Open field - the same cells, same names",
        "Same numbering, so a neuron can be followed between the two "
        "decks and the two sessions.",
        os.path.join(OF, "atlas", "fig11_fov_openfield.png"),
        r"<open field>\output_split\atlas\fig11_fov_openfield.png",
        hbox=5.3)

    rows = [["", "OF A", "OF B", "PA A", "PA B"],
            ["EXTRACT detected", "14", "14", "12", "15"],
            ["after curation", "12", "16", "11", "16"],
            ["active / silent / SUSPECT", "5 / 7 / 0", "5 / 9 / 2",
             "6 / 4 / 1", "6 / 10 / 0"]]
    s = blank(p)
    title(s, "The cell set, in numbers",
          f"{n_union} cells measured in both sessions "
          f"({n_match} detected independently twice)")
    table(s, rows, Inches(.6), Inches(1.6), Inches(8.0), Inches(1.9),
          size=13)
    text(s,
         f"After registration and transfer:\n\n"
         f"  cells measured in both sessions      {n_union}\n"
         f"  detected independently in both       {n_match}\n"
         f"  footprint on a soma in BOTH          "
         f"{int(((U['pain_anat'] >= 1) & (U['openfield_anat'] >= 1)).sum())}\n"
         f"  dropped as inseparable duplicates    "
         f"{int(U['merged_of'].notna().sum()) if 'merged_of' in U else 0}\n\n"
         f"A cell whose footprint does not land on a soma in a given "
         f"session is kept in the table and\n"
         f"excluded from that session's statistics - its trace is "
         f"background whatever it looks like.",
         Inches(.6), Inches(3.8), Inches(12.2), Inches(2.8), size=13)
    path_line(s, r"<pain>\output_split\match\union_cells.csv")

    return save(p, f"CEANTSR1_processing_{STAMP}", HERE)


if __name__ == "__main__":
    build()
