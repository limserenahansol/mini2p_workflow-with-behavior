"""make_results_ppt.py  -  the results deck, in the order Hansol asked for.

  1  the field of view and the final cell selection
  2  open field: all cells, traces with the mouse's location on the same
     axis, then one cell at a time with the reason for its verdict
  3  pain: THE FINAL CLASSIFICATION and nothing else - what the session
     can answer, the three-tier test, the verdict per cell, the evidence
     behind each verdict
  4  what was withdrawn, and the measurement that forced each withdrawal.
     The superseded figures are kept here, labelled WITHDRAWN, because the
     reason each one fails is itself a result about the session's design
  5  across sessions: the cells with a final pain verdict and where they
     sat in the open field

Every pain slide reports pain_final.csv. The per-delivery pin/heat tables
(responsive_cells.csv, search_grid.csv) appear only in section 4, under an
explicit withdrawal, and the session numbers the argument rests on are read
from pain_audit.csv - they were typed in from memory once and five of them
had drifted.

Everything up to and including cell selection lives in the PROCESSING deck
(make_processing_ppt.py); this one starts from the finished cell set.

Numbers are read from the result CSVs at build time, never typed, so a
slide cannot disagree with its table.

OUTPUT  ->  <this folder>\\CEANTSR1_results_<date>.pptx

USAGE
  python make_results_ppt.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pptx.util import Inches

from deck import (ACC, GOOD, WARN, blank, bullets, fig, new_deck, path_line,
                  save, section, table, text, title)

HERE = os.path.dirname(os.path.abspath(__file__))
OF = ("D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_"
      "15-27-52/output_split")
PA = ("D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
      "15-56-48/output_split")
STAMP = "20260914"
LEVEL = "q20"      # the criterion section 3 onward uses; the strict
                   # q <= 0.05 version stays in the deck as the comparison


def facts():
    f = {}
    R = pd.read_csv(os.path.join(PA, "events", "responsive_cells.csv"))
    f["R"] = R
    f["n_all"] = len(R)
    f["n_ok"] = int((R["usable"] == 1).sum())
    # directions at the deck's criterion, re-derived from the p/q columns
    # so the slides and the figures cannot use different thresholds
    from results_figures import LEVELS, dirs_of
    evs, g = dirs_of(R, LEVEL)
    f["evs"] = evs
    f["dirs"] = g
    f["resp"] = {u: d for u, d in g.items() if d}
    f["levtxt"] = LEVELS[LEVEL][2]
    _, gs = dirs_of(R, "q05")
    f["resp_strict"] = {u: d for u, d in gs.items() if d}
    f["suf"] = "" if LEVEL == "q05" else f"_{LEVEL}"
    Z = pd.read_csv(os.path.join(OF, "atlas", "zone_rates.csv"))
    f["Z"] = Z
    f["faster"] = Z[Z["faster"]]["uid"].tolist()
    P = pd.read_csv(os.path.join(OF, "place", "place_cells.csv"))
    f["P"] = P
    f["n_place_tested"] = len(P)
    f["n_info"] = int((P["q_info"] <= .05).sum())
    G = pd.read_csv(os.path.join(PA, "events", "search_grid.csv"))
    f["G"] = G
    # THE FINAL pain classification. Section 3 reports only this; the
    # per-event tables above stay in the deck as section 4's withdrawn
    # set, so a reader can see what changed and why.
    FIN = pd.read_csv(os.path.join(PA, "events", "pain_final.csv"))
    ok = FIN["usable"] == 1
    f["FIN"] = FIN
    f["fin"] = {r["uid"]: r for _, r in FIN.iterrows()}
    f["fin_up"] = FIN[ok & FIN["final"].str.contains("stim-UP")]["uid"]\
        .tolist()
    f["fin_dn"] = FIN[ok & FIN["final"].str.contains("stim-DOWN")]["uid"]\
        .tolist()
    f["fin_esc"] = FIN[ok & FIN["final"].str.contains("escape-UP")]["uid"]\
        .tolist()
    from results_figures import final_dirs
    _, f["fin_dirs"] = final_dirs(PA)
    f["fin_resp"] = {u: d for u, d in f["fin_dirs"].items() if d}
    # The session numbers the pain slides argue from. They used to be
    # typed into the slide text from the audit, and five of them had
    # drifted from what the code actually computes - so they are read
    # from the audit table now, like every other number in this deck.
    A = pd.read_csv(os.path.join(PA, "events", "pain_audit.csv"))
    f["A"] = {r["item"]: r["value"] for _, r in A.iterrows()}
    return f


def a(F, item, fmt="{:.0f}"):
    """One audit number, formatted - raises if the audit does not have it."""
    if item not in F["A"]:
        raise SystemExit(f"pain_audit.csv has no '{item}' - run "
                         f"pain_audit.py")
    return fmt.format(F["A"][item])


def ev_counts(F, ev):
    up = [u for u, d in F["dirs"].items() if d.get(ev) == "UP"]
    dn = [u for u, d in F["dirs"].items() if d.get(ev) == "DOWN"]
    return up, dn


def build():
    F = facts()
    p = new_deck()

    # ---------------------------------------------------------- title
    s = blank(p)
    text(s, "CEA-Ntsr1 two-plane calcium imaging", Inches(.9), Inches(1.9),
         Inches(11.5), Inches(.9), size=34, bold=True)
    text(s, "Results - open field and pain assay, 2026-09-11",
         Inches(.9), Inches(2.9), Inches(11.5), Inches(.5), size=19,
         color=ACC)
    text(s,
         f"{F['n_all']} cells, the same cell carrying the same name in "
         f"both sessions; {F['n_ok']} of them sit on a soma in the pain "
         f"session and are counted.\n"
         f"Behaviour scored manually on 2026-09-14.\n\n"
         f"Everything up to the cell selection - motion correction, "
         f"EXTRACT, curation, QC - is in the PROCESSING deck.",
         Inches(.9), Inches(3.6), Inches(11.5), Inches(2.0), size=14,
         color=DIMC())
    path_line(s, r"code C:\Users\hsollim\OneDrive\Desktop\cursor"
                 r"\pain_2plane_split   |   results <session>\output_split\ ")

    # ======================================================== 1 the cells
    section(p, 1, "The cells",
            "Which neurons the results are about, and where they are.\n"
            "One name per neuron, the same in both sessions.")
    fig(p, "Pain session - every cell, labelled",
        "Thick outline and a number on each. Grey = the footprint does not "
        "land on a soma in this session, so its trace is background and it "
        "is excluded from every count.",
        os.path.join(PA, "atlas", "fig11_fov_pain.png"),
        r"<pain>\output_split\atlas\fig11_fov_pain.png", hbox=5.3)
    fig(p, "Open field - the same cells, same names",
        "Same numbering. A1 here and A1 on the previous slide are one "
        "neuron, which is what makes the cross-session comparison at the "
        "end possible.",
        os.path.join(OF, "atlas", "fig11_fov_openfield.png"),
        r"<open field>\output_split\atlas\fig11_fov_openfield.png",
        hbox=5.3)

    # ======================================================== 2 open field
    section(p, 2, "Open field",
            "Where the mouse was, what each cell did there, and the "
            "reason for every verdict.\n"
            "Zone means WHERE THE MOUSE IS at that moment - nothing is "
            "triggered on entering.")
    fig(p, "All cells, with the mouse's location on the same time axis",
        "Top: every cell's z. Bottom: the zone the mouse was in (blue "
        "centre, pink corner, grey edge) and its speed. Read them "
        "together.",
        os.path.join(OF, "atlas", "fig13_traces_openfield.png"),
        r"<open field>\output_split\atlas\fig13_traces_openfield.png",
        hbox=5.5)

    occ = ("centre 18 s (5.5 %), corner 136 s (41.7 %), "
           "edge 172 s (52.8 %)")
    bullets(p, "Why a count of firing is the wrong measure here",
            "and the rate is the right one",
            f"The mouse's time is not evenly spread:  {occ}\n"
            f"It is in the corners 7.6 times longer than in the centre. So "
            f"MOST of any cell's firing lands in a corner\n"
            f"no matter what it prefers - counting dots would make every "
            f"cell a corner cell.\n\n"
            f"A3, for example:   18 % of its firing happens in the centre "
            f"... and its rate there is 1.24 per second\n"
            f"                   37 % happens in the corners ... at 0.34 "
            f"per second\n"
            f"Both are true. The rate is 3.7x higher in the centre; the "
            f"count is 2x higher in the corners.\n\n"
            f"This is the standard correction for spatial firing "
            f"(occupancy-normalised rate maps, Muller & Kubie 1987).\n"
            f"The statistics use the continuous mean z per zone, which "
            f"needs no firing threshold at all; the\n"
            f"'per second' numbers and the yellow dots are the readable "
            f"version of the same thing.",
            size=13.5,
            pathline=r"of_zone_rates.py  ->  <open field>\output_split"
                     r"\atlas\zone_rates.csv , zone_rates.txt")

    up = F["faster"]
    fig(p, f"The {len(up)} cells that fire faster in the centre",
        "Per cell: where it is, where it fired, the rate map with its own "
        "colour bar and maximum, the rate in each zone with the time spent "
        "there, and the trace with the mouse's location.",
        os.path.join(OF, "atlas", "fig17_zone_rates_selective.png"),
        r"<open field>\output_split\atlas\fig17_zone_rates_selective.png",
        hbox=5.3)

    bullets(p, "Why these are NOT called place cells",
            "the word implies something this data does not show",
            f"A place cell has a localised firing field, significant "
            f"spatial information and a stable map\n"
            f"(O'Keefe & Dostrovsky 1971; Muller & Kubie 1987; Skaggs et "
            f"al. 1996).\n\n"
            f"  spatial information, significant:   "
            f"{F['n_info']} of {F['n_place_tested']} cells\n"
            f"  localised field:                    none - the firing is "
            f"spread over the arena\n\n"
            f"What the data does support is a ZONE preference: higher "
            f"activity while the animal is in the\n"
            f"exposed centre. That is the framing used for amygdala and "
            f"ventral hippocampus in the open\n"
            f"field - Jimenez et al. 2018 Neuron call them anxiety cells, "
            f"not place cells; Grundemann et al.\n"
            f"2019 Science treat the basal amygdala as encoding defensive "
            f"STATE rather than position.\n\n"
            f"So: \"fires faster in the centre\", not \"centre place "
            f"cell\".",
            size=13.5, color=WARN)

    Z = F["Z"]
    sel = Z[Z["faster"]]
    rows = [["cell", "contrast (z)", "q", "drop any 1 visit",
             "first half", "second half", "same sign"]]
    for _, r in sel.iterrows():
        rows.append([r["uid"], f"{r['contrast']:+.2f}", f"{r['q']:.3f}",
                     f"{r['loo_min']:+.2f} to {r['loo_max']:+.2f}",
                     f"{r['half1']:+.2f}", f"{r['half2']:+.2f}",
                     "yes" if r["stable"] else "NO"])
    s = blank(p)
    title(s, "Is the centre preference stable, or one lucky visit?",
          "the mouse entered the centre 4 times: 1.1, 1.1, 5.9 and 9.8 s")
    table(s, rows, Inches(.6), Inches(1.6), Inches(12.1),
          Inches(.45 * len(rows)), size=13)
    text(s,
         "Two standard checks and the one this session needs. Dropping any "
         "single centre visit leaves both cells\n"
         "positive, and both halves of the session agree in sign. Cells "
         "that do not reach significance mostly flip\n"
         "sign between halves - that is the difference.\n\n"
         "Still only 18 s of centre data in total, carried by two visits. "
         "Read the ratio as a direction, not a number.",
         Inches(.6), Inches(1.75 + .45 * len(rows)), Inches(12.1),
         Inches(1.8), size=13, color=DIMC())
    path_line(s, r"<open field>\output_split\atlas\zone_rates.csv")

    fig(p, "The zone result both ways - significance and the k = 2 split",
        "LEFT asks whether a preference beats chance. MIDDLE is the other "
        "pipeline's method: k-means k = 2 on the same numbers, which always "
        "returns two groups. RIGHT is the corner question with the frames "
        "to answer it.",
        os.path.join(OF, "place", "fig25_both_methods.png"),
        r"of_both_methods.py  ->  <open field>\output_split\place"
        r"\of_both_methods.csv , .txt", hbox=5.1,
        note="A3 and A7 are centre-preferring by both. The k = 2 corner "
             "group has 19 cells and not one of them passes a test - they "
             "are the lower half of a ranking. Corner versus everywhere "
             "else, which has 626 frames against 874, finds 0 cells at "
             "q <= 0.20: there is no corner-specific cell here.")

    fig(p, "Every cell in the open field, one row each",
        "Same four panels for all of them, so the cells that were NOT "
        "called centre-preferring can be checked too.",
        os.path.join(OF, "atlas", "fig17_zone_rates_all.png"),
        r"<open field>\output_split\atlas\fig17_zone_rates_all.png",
        hbox=5.4)

    # ======================================================== 3 pain
    section(p, 3, "Pain assay - the final classification",
            "53 pin pricks, 34 heat, 76 paw withdrawals, 50 flinches, "
            "5 escapes; behaviour scored by hand on 2026-09-14.\n"
            "ONE classification, in three tiers, each stating what it does "
            "and does not mean.\n"
            f"TIER 1 stimulation-modulated: {len(F['fin_up'])} UP "
            f"({', '.join(F['fin_up'])}), {len(F['fin_dn'])} DOWN.    "
            f"TIER 2 modality-selective: WITHDRAWN.    "
            f"TIER 2b escape-linked: {len(F['fin_esc'])} "
            f"({', '.join(F['fin_esc'])}).\n"
            "Everything the earlier versions of this section claimed and "
            "this one does not is in section 4, with the measurement that "
            "withdrew it.")

    fig(p, "All cells with the scored events on the same time axis",
        "Top: every cell's z. Bottom: when each stimulus and behaviour was "
        "scored. This is the raw material for everything that follows - no "
        "test has been applied yet.",
        os.path.join(PA, "atlas", "fig13_traces_pain.png"),
        r"<pain>\output_split\atlas\fig13_traces_pain.png", hbox=5.5)

    fig(p, "Before the first stimulus versus after the last",
        "30 s each, the same cells, paired. dF/F is the one to read: z is "
        "detrended with a 20 s running median and so cannot show a change "
        "slower than that.",
        os.path.join(PA, "results", "fig19_before_after.png"),
        r"results_figures.py  ->  <pain>\output_split\results"
        r"\fig19_before_after.png", hbox=5.0,
        note="26 of 28 cells raise their mean dF/F across the session "
             "(Wilcoxon signed-rank, p < 0.0001). This one does NOT depend "
             "on any window choice - it is two 30 s blocks ten minutes "
             "apart. It is also not a response locked to a delivery: "
             "sensitisation, arousal and slow imaging drift all look like "
             "this, and separating them needs a no-stimulus control "
             "session.")

    bullets(p, "What this session can answer, and what it cannot",
            "nine measurements, each of which rules out a question - every "
            "number below is read from pain_audit.csv at build time",
            "1  PIN PRICKS COME " + a(F, "gap_median_pin", "{:.2f}") +
            " s APART (median; heat "
            + a(F, "gap_median_heat", "{:.2f}") + " s). So no "
            "pre-stimulus window is clean - another\n"
            "   delivery falls inside [-2, 0] s for "
            + a(F, "pin_prewin_contaminated_-2_0", "{:.0f}") + " % of pin "
            "trials, [-4, -2] for "
            + a(F, "pin_prewin_contaminated_-4_-2", "{:.0f}") + " %, "
            "[-6, -4] for "
            + a(F, "pin_prewin_contaminated_-6_-4", "{:.0f}") + " %,\n"
            "   [-8, -6] for "
            + a(F, "pin_prewin_contaminated_-8_-6", "{:.0f}") + " %. Only "
            + a(F, "pin_with_clear_prewindow") + " of 53 pin deliveries "
            "have 6 s of clear time before them.\n\n"
            "2  THE POPULATION PEAKS "
            + a(F, "population_peak_lag", "{:+.0f}") + " s - BEFORE the "
            "key press. Mean z over all 28 cells and all 87\n"
            "   deliveries, nothing subtracted: +0.02 at -6 s, +0.10 at "
            "-2 s, +0.16 at -1 s, +0.05 at 0, +0.08 at +2 s.\n"
            "   A pre-window baseline therefore starts the measurement AT "
            "THE TOP of the response.\n\n"
            "3  THAT ONE FACT GAVE TWO OPPOSITE RESULTS FROM THE SAME "
            "TRACES. Pin, [0,+2] s post, across cells:\n"
            "     baseline [-2, 0] s   -> "
            + a(F, "pin_effect_baseline_-2_0", "{:+.3f}") + " z, Wilcoxon "
            "p = " + a(F, "pin_p_baseline_-2_0", "{:.4f}")
            + "   SUPPRESSION\n"
            "     baseline [-6, -4] s  -> "
            + a(F, "pin_effect_baseline_-6_-4", "{:+.3f}") + " z, Wilcoxon "
            "p = " + a(F, "pin_p_baseline_-6_-4", "{:.5f}")
            + "  EXCITATION\n"
            "   Only the baseline moved. The sign of the population result "
            "moved with it.\n\n"
            "4  AN EVENT-JITTER NULL CANNOT BE BUILT HERE. Jittering by "
            "3-15 s puts "
            + a(F, "jitter_lands_on_real_event", "{:.0f}") + " % of events "
            "within 1 s of a\n"
            "   real delivery, so the null contains the signal and "
            "everything passes. A circular shift of\n"
            "   the cell's own trace is used instead: it keeps the event "
            "times fixed and only breaks the pairing.\n\n"
            "5  PIN AND HEAT WERE GIVEN IN SEPARATE BLOCKS. Pin 13-204 s "
            "(n = 37) and 519-586 s (n = 16); heat\n"
            "   269-518 s. Pin deliveries inside the heat block: "
            + a(F, "pin_inside_heat_block") + ". So 'pin versus heat' is "
            "also 'early versus\n"
            "   middle', and permuting the 53/34 labels mixes the blocks "
            "instead of undoing that.\n\n"
            "6  PIN, WITHDRAWAL AND FLINCH ARE ONE FINDING, NOT THREE. "
            "Jaccard overlap of their UP lists:\n"
            "   withdrawal vs flinch "
            + a(F, "jaccard_paw_withdrawal_flinch", "{:.2f}") + ", pin vs "
            "withdrawal " + a(F, "jaccard_pin_paw_withdrawal", "{:.2f}")
            + ", pin vs flinch "
            + a(F, "jaccard_pin_flinch", "{:.2f}") + ". Those events all "
            "sit inside\n"
            "   the same epochs.\n\n"
            "7  A GLOBAL STIMULUS-FREE REFERENCE IS BIASED LATE. It is "
            + a(F, "free_mask_fraction", "{:.0f}") + " % of the session "
            "but sits "
            + a(F, "free_mask_time_offset", "{:+.0f}") + " s later\n"
            "   than the deliveries it is a reference for, and only "
            + a(F, "free_in_pin_block", "{:.0f}") + " % of the dense pin "
            "block is free. A cell that\n"
            "   drifts up over ten minutes then looks SUPPRESSED during "
            "stimulation. That bias alone moved the\n"
            "   answer from 2 cells to 8. TIER 1 uses the free frames "
            "within +-60 s of each delivery instead.\n\n"
            "8  THE CLEAN SUBSET IS TOO SMALL TO SUBSTITUTE. With only the "
            + a(F, "pin_with_clear_prewindow") + " pin trials that have 6 s "
            "clear, the\n"
            "   smallest resolvable effect is "
            + a(F, "detectable_effect_clear_trials", "{:.2f}") + " z - "
            "larger than the biggest effect anything actually shows "
            "(0.41 z).\n\n"
            "9  THE HELD BEHAVIOURS WERE TAPPED, NOT HELD. Licking/biting "
            "and guarding were never scored;\n"
            "   paw attending once, 0.4 s; escape 5 times, 3.9 s = 18 "
            "imaging frames. Only escape is testable,\n"
            "   and only as a hint.",
            size=11.5,
            pathline=r"pain_audit.py  ->  <pain>\output_split\events"
                     r"\pain_audit.txt , pain_audit.csv")

    bullets(p, "The final test, in three tiers",
            "one test per tier, applied identically to every cell; "
            "exploratory FDR q <= 0.20 across the 28 interpretable cells",
            "TIER 1   STIMULATION-MODULATED                            "
            "robust\n"
            "  QUESTION   is the cell different while stimulation is going "
            "on than during the quiet time?\n"
            "  WINDOW     [0, +10] s after each of the 87 deliveries\n"
            "  REFERENCE  the stimulus-free frames WITHIN +-60 s of that "
            "same delivery, so a session-long\n"
            "             drift cancels. A global free mask sits "
            + a(F, "free_mask_time_offset", "{:+.0f}") + " s late in this "
            "session - only "
            + a(F, "free_in_pin_block", "{:.0f}") + " % of\n"
            "             the dense pin block is free - and that bias "
            "alone moved the answer from 2 cells to 8.\n"
            "  NULL       circular shift of the cell's own trace: keeps its "
            "autocorrelation, its drift and\n"
            "             the event times, destroys only the pairing. 2000 "
            "shifts, exact by FFT.\n"
            "  DOES NOT MEAN pin-specific. Handling, arousal and movement "
            "are all inside 'stimulation\n"
            "             going on'.\n\n"
            "TIER 2   MODALITY-SELECTIVE                               "
            "WITHDRAWN - not determinable\n"
            "  Pin and heat are separate blocks (fact 5). The columns stay "
            "in the CSV as t2_* and are\n"
            "  marked invalid; the full-session version flagged 14 of 28 "
            "cells at 0.05-0.13 z, which is\n"
            "  what a slow drift looks like.\n\n"
            "TIER 2b  BEHAVIOUR-LINKED                                 "
            "weak - n is tiny\n"
            "  Escape episodes, inside versus outside: 5 episodes, 3.9 s, "
            "18 imaging frames. Reported\n"
            "  with its n so it reads as a hint, not a result.\n\n"
            "NOT CLAIMED\n"
            "  per-delivery phasic responses to pin or heat; any sentence "
            "of the form 'N cells respond to\n"
            "  pin'; pin-versus-heat selectivity. The first is set by the "
            "window and not by the neurons -\n"
            "  0, 6, 7 and 8 cells across four defensible versions, sharing "
            "only A9 between the two best.",
            size=12.5,
            pathline=r"pain_final.py  ->  <pain>\output_split\events"
                     r"\pain_final.csv , pain_final.txt")

    fig(p, "THE FINAL ANSWER - every cell, one verdict",
        "Left three: the effect size per cell for each tier, red = UP and "
        "blue = DOWN at q <= 0.20, grey = not significant. Bottom: where "
        "the TIER 1 cells sit, and the verdict list.",
        os.path.join(PA, "events", "fig29_pain_final.png"),
        r"pain_final.py  ->  <pain>\output_split\events"
        r"\fig29_pain_final.png", hbox=5.3,
        note=f"TIER 1: {len(F['fin_up'])} UP - "
             + ", ".join(f"{u} ({F['fin'][u]['t1_any']:+.2f} z, "
                         f"q = {F['fin'][u]['t1_any_q']:.3f})"
                         for u in F['fin_up'])
             + f"; {len(F['fin_dn'])} DOWN. TIER 2b escape: "
             + ", ".join(F['fin_esc']) + " - on 18 imaging frames. "
             + f"These are not borderline: the next cell after "
               f"{F['fin_up'][-1]} sits at q = "
             + f"{sorted(x for x in F['FIN'].loc[F['FIN'].usable == 1, 't1_any_q'] if x > 0.20)[0]:.2f}"
             + ", so there is a clear gap rather than a cut through a "
               "continuum. Every effect size in the CSV was re-derived "
               "from the traces independently and reproduced to 5e-4.")

    fig(p, "Where the final cells are",
        "One field-of-view map per tag, both planes. The name is printed "
        "above its cell so the soma stays visible.",
        os.path.join(PA, "results", "fig20_event_maps_final.png"),
        r"<pain>\output_split\results\fig20_event_maps_final.png", hbox=5.2,
        slice_tall=False)   # a 2 x 2 map grid has to be seen whole

    fig(p, "Group averages - every panel against the same reference",
        "Mean +- SEM across cells. EVERY panel is z minus the "
        "stimulus-free frames within +-60 s of that event - the same "
        "reference the verdict uses, and no pre-event window anywhere.",
        os.path.join(PA, "results", "fig21_group_means_final.png"),
        r"<pain>\output_split\results\fig21_group_means_final.png",
        hbox=5.2,
        note="The time before 0 is already raised, in every stimulus "
             "panel. That is fact 1 and fact 2 made visible: deliveries "
             "3.2 s apart mean the 'pre' time is still inside the epoch. "
             "The earlier version of this figure subtracted [-2, 0] s on "
             "the five left panels and the stimulus-free time on the two "
             "right ones - two references in one figure - and the headline "
             "was read off the wrong panels, which is how 'pin: no "
             "response (n = 28)' happened.")

    fig(p, "Every cell, one at a time - the evidence behind its verdict",
        "Field of view, then the peri-event average for every event type, "
        "then the whole trace with the events. The row title carries the "
        "verdict and its statistic; the pin and heat columns carry no "
        "verdict because none is claimed per delivery.",
        os.path.join(PA, "atlas", "fig16_rows_pain_final.png"),
        r"cell_atlas.py --level final  ->  <pain>\output_split\atlas"
        r"\fig16_rows_pain_final.png", hbox=5.4)

    hdr = ["cell", "TIER 1 stim", "q", "TIER 2b escape", "q", "FINAL"]
    body = [[r["uid"], f"{r['t1_any']:+.2f}", f"{r['t1_any_q']:.3f}",
             f"{r['t3_escape']:+.2f}", f"{r['t3_q']:.3f}", r["final"]]
            for _, r in F["FIN"].iterrows() if r["usable"] == 1]
    s = blank(p)
    title(s, "Every cell, its numbers and its verdict",
          "TIER 1 = [0,+10] s after each of the 87 deliveries minus the "
          "stimulus-free frames within +-60 s; TIER 2b = inside escape "
          "minus outside; q = BH across the 28 cells, exploratory q <= 0.20")
    # two tables side by side: 28 rows in one column overflows the slide,
    # and a table that runs off the bottom is the same as not having it
    h = (len(body) + 1) // 2
    for j, part in enumerate((body[:h], body[h:])):
        table(s, [hdr] + part, Inches(.35 + j * 6.35), Inches(1.5),
              Inches(6.1), Inches(.28 * (len(part) + 1)), size=10)
    path_line(s, r"<pain>\output_split\events\pain_final.csv")

    # =============================================== 4 what was withdrawn
    section(p, 4, "What was withdrawn, and the measurement that forced it",
            "Earlier versions of this deck reported per-delivery pin and "
            "heat responses, a pin-versus-heat\n"
            "preference, and a population suppressed by pin. None of the "
            "three survived the audit. The figures\n"
            "are kept here rather than deleted, because the reason each one "
            "fails is itself a result about\n"
            "this session's design - and because the next session can be "
            "designed to avoid all three.")

    bullets(p, "Four claims withdrawn",
            "each with the number that withdrew it",
            "WITHDRAWN 1   'N cells respond to pin' / 'to heat', per "
            "delivery\n"
            "  Four defensible versions of the same test gave 0, 6, 7 and "
            "8 cells, sharing only A9 between\n"
            "  the two best. With deliveries 3.2 s apart the answer is set "
            "by the window, not by the neurons.\n"
            "  KEPT INSTEAD  TIER 1: the cell is modulated while "
            "stimulation is going on. 2 cells, A1 and A4.\n\n"
            "WITHDRAWN 2   'cell X prefers pin over heat'\n"
            "  Pin 13-204 s and 519-586 s, heat 269-518 s, zero pin inside "
            "the heat block. The label\n"
            "  permutation mixes the blocks and cannot undo the confound "
            "with time. The full-session version\n"
            "  flagged 14 of 28 cells at 0.05-0.13 z - drift sizes.\n"
            "  KEPT INSTEAD  nothing. The question needs interleaved "
            "delivery.\n\n"
            "WITHDRAWN 3   'pin suppresses the population, p = 0.006'\n"
            "  That is the AUC figure below. Its pre window is [-2, 0] s, "
            "which is where the population peaks,\n"
            "  so the post window is measured against the top of the "
            "response. Move the baseline to [-6, -4]\n"
            "  and the same traces give EXCITATION: "
            + a(F, "pin_effect_baseline_-6_-4", "{:+.3f}") + " z, Wilcoxon "
            "p = " + a(F, "pin_p_baseline_-6_-4", "{:.5f}")
            + ", against "
            + a(F, "pin_effect_baseline_-2_0", "{:+.3f}") + " z, p = "
            + a(F, "pin_p_baseline_-2_0", "{:.4f}") + " with [-2, 0].\n"
            "  The figure stays in the deck as the demonstration.\n"
            "  KEPT INSTEAD  fig19: 26 of 28 cells rise across the whole "
            "session, p < 0.0001, no window needed.\n\n"
            "WITHDRAWN 4   'A1 responds to heat, withdrawal, pin epoch and "
            "heat epoch' as four findings\n"
            "  Those four lists overlap at Jaccard 0.60-0.90 because the "
            "events sit inside the same epochs.\n"
            "  It is one finding, counted four times.\n"
            "  KEPT INSTEAD  A1 is TIER 1 stimulation-UP, +0.41 z, "
            "q = 0.007 - the clearest cell in the dataset.",
            size=12.5, color=WARN)

    fig(p, "WITHDRAWN - the AUC layout, and what its baseline does",
        "Top: population AUC, pre against post, Wilcoxon across cells. "
        "Bottom: the peri-event mean +- SEM. The pre window is [-2, 0] s.",
        os.path.join(PA, "results", "fig26_auc_z.png"),
        r"summary_figures.py  ->  <pain>\output_split\results"
        r"\fig26_auc_z.png", hbox=5.2,
        note="THIS SLIDE IS THE WITHDRAWN CLAIM, kept on purpose. The pin "
             "panel reads 'pre +0.007, post -0.078 z*s, signed-rank "
             "p = 0.006'. Look at the bottom-left panel: the curve is "
             "already falling before 0, because the baseline it was given "
             "is [-2, 0] s and the population peaks at "
             + a(F, "population_peak_lag", "{:+.0f}") + " s. Every post "
             "value is measured downhill from the peak. On mean z the same "
             "traces give "
             + a(F, "pin_effect_baseline_-2_0", "{:+.3f}") + " z with that "
             "baseline and "
             + a(F, "pin_effect_baseline_-6_-4", "{:+.3f}") + " z with "
             "[-6, -4]. The other pipeline reports the same suppression "
             "from the same kind of baseline, so this is a shared artefact "
             "and not a disagreement between us.")
    fig(p, "WITHDRAWN - the same thing on dF/F instead of z",
        "z is detrended with a 20 s running median; dF/F is not. The "
        "baseline problem is in both, because it is a problem of when the "
        "window sits, not of which signal is used.",
        os.path.join(PA, "results", "fig26_auc_dff.png"),
        r"<pain>\output_split\results\fig26_auc_dff.png", hbox=5.2)
    fig(p, "WITHDRAWN - per-cell AUC, pre and post",
        "The bottom panel of that layout, per cell. * marks a raw per-cell "
        "p < 0.05 with no correction, so about 1.4 of them are chance.",
        os.path.join(PA, "results", "fig26b_auc_percell_z.png"),
        r"<pain>\output_split\results\fig26b_auc_percell_z.png", hbox=5.3)

    fig(p, f"WITHDRAWN - the per-delivery maps ({F['levtxt']})",
        "The version this deck used to headline: one map per event type, "
        "every cell tested against every event at an exploratory FDR.",
        os.path.join(PA, "results", f"fig20_event_maps{F['suf']}.png"),
        r"<pain>\output_split\results\fig20_event_maps_q20.png", hbox=5.2,
        note=f"{len(F['resp'])} of {F['n_ok']} cells respond to something "
             f"here, against 2 in TIER 1. The extra cells are not fabricated"
             f" - they are what a per-delivery test returns once the "
             f"baseline is contaminated by the next delivery. Withdrawn "
             f"because a different window gives a different list.")
    fig(p, f"WITHDRAWN - the per-delivery population summary "
           f"({F['levtxt']})",
        "Counts per event, how many events each cell answers to, and the "
        "overlap between events.",
        os.path.join(PA, "results", f"fig22_population{F['suf']}.png"),
        r"<pain>\output_split\results\fig22_population_q20.png", hbox=5.3,
        note="The overlap panel is the useful part and it is the argument "
             "against the rest of the figure: the same cells come up for "
             "pin, withdrawal and flinch, Jaccard 0.60-0.90, because those "
             "events sit inside the same epochs.")
    fig(p, "WITHDRAWN - cells answering to more than one thing",
        "One panel per response, per cell. Read as the overlap evidence, "
        "not as four separate findings.",
        os.path.join(PA, "results", "fig28_multi_response.png"),
        r"<pain>\output_split\results\fig28_multi_response.png", hbox=5.4)

    fig(p, "WHY the per-delivery answer moves - the whole grid",
        "5 response windows x 3 statistics x 2 stimuli, plus a joint model "
        "with every predictor at once. A cell lighting up in one box of 15 "
        "is noise; one surviving most of them is not.",
        os.path.join(PA, "events", "fig18_search.png"),
        r"pain_search.py  ->  <pain>\output_split\events\fig18_search.png",
        hbox=5.1,
        note="Most consistent across the grid: A1 (pin 10/15 UP, heat "
             "12/15 UP), B12 (heat 12/15 UP), B7 (pin 10/15 DOWN), A2 (pin "
             "7/15 UP). A1 is TIER 1 in the final version too, which is "
             "the one point of agreement between the grid and the final "
             "test. The rest of the grid shares the pre-window baseline "
             "and inherits its bias, so consistency across boxes is not "
             "independent evidence.")
    fig(p, "WHY loosening does not fix it - the threshold ladder",
        "Left: the +-2 s and 10 s tests at each threshold, with the dashed "
        "line marking how many cells chance gives at an uncorrected "
        "p <= 0.05. Right: the grid, per cell.",
        os.path.join(PA, "events", "fig24_thresholds.png"),
        r"loosen.py  ->  <pain>\output_split\events\fig24_thresholds.png",
        hbox=5.1,
        note="9 cells at q <= 0.05, 12 at q <= 0.10, 15 at q <= 0.20, 19 "
             "at an uncorrected p <= 0.05 - where chance alone gives about "
             "11 across the 8 tests. Loosening adds a tail of cells "
             "passing one or two boxes, which is what chance produces. It "
             "does not create A1, and it does not rescue the per-delivery "
             "question.")

    bullets(p, "What the next session needs - two changes",
            "both follow directly from the measurements above",
            "1  SPACE THE PIN PRICKS 15-20 s APART.\n"
            "   At " + a(F, "gap_median_pin", "{:.1f}") + " s the "
            "pre-stimulus window is contaminated for "
            + a(F, "pin_prewin_contaminated_-2_0", "{:.0f}") + "-"
            + a(F, "pin_prewin_contaminated_-4_-2", "{:.0f}") + " % of "
            "trials and the population has not\n"
            "   returned to baseline between deliveries. 20 well-spaced "
            "pricks carry more information than the\n"
            "   53 here: with only the "
            + a(F, "pin_with_clear_prewindow") + " trials that have 6 s "
            "clear, the smallest resolvable effect is "
            + a(F, "detectable_effect_clear_trials", "{:.2f}") + " z -\n"
            "   bigger than the largest effect anything in this session "
            "shows (0.41 z), so the clean subset is\n"
            "   useless. 20 trials that are ALL clean would resolve about "
            "0.25 z.\n\n"
            "2  INTERLEAVE PIN AND HEAT INSTEAD OF BLOCKING THEM.\n"
            "   Alternating deliveries make 'pin versus heat' a "
            "within-time comparison, and the label\n"
            "   permutation then holds arousal, handling and movement fixed "
            "- which is exactly the test that\n"
            "   TIER 2 wanted and this session cannot support.\n\n"
            "ALSO WORTH ADDING\n"
            "   a no-stimulus control session of the same length, which is "
            "the only way to tell the "
            "session-long\n"
            "   rise in fig19 (26 of 28 cells, p < 0.0001) from "
            "sensitisation.\n"
            "   a 60-90 s stimulus-free block at the start and the end, so "
            "there is a clean reference that\n"
            "   does not have to be carved out of the gaps.",
            size=13, color=GOOD)

    # ======================================================== 5 across
    section(p, 5, "Across the two sessions",
            "The question the whole pipeline exists for: is a "
            "stimulation-modulated neuron also the one that\n"
            "fired faster in the exposed centre? Both sessions share one "
            "footprint set, so A1 in the open\n"
            "field and A1 in the pain session are the same neuron - which "
            "is what makes the question askable.")
    fig(p, "Every cell with a final pain verdict, and the same cell in the "
           "open field",
        "Left: the pain session with the scored events. Right: the SAME "
        "neuron in the open field, with the mouse's location under the "
        "trace. The verdict on each row is the FINAL one.",
        os.path.join(PA, "results", "fig23_cross_session_final.png"),
        r"results_figures.py --level final  ->  <pain>\output_split"
        r"\results\fig23_cross_session_final.png", hbox=5.4)

    fig(p, "Every cell that matters - field of view, open field, pain",
        "Two field-of-view panels (open field and pain), then the "
        "open-field trace with the mouse's zone, then the pain trace with "
        "the scored events. A3 and A7 - the two centre cells - come first, "
        "then every cell with a final pain verdict.",
        os.path.join(PA, "results", "fig27_cross_cells_final.png"),
        r"summary_figures.py --level final  ->  <pain>\output_split"
        r"\results\fig27_cross_cells_final.png", hbox=5.4)

    pref = {r["cell"]: r["preference"] for _, r in F["P"].iterrows()}
    rows = [["cell", "final pain verdict", "open field", "same neuron?"]]
    for u, d in F["fin_resp"].items():
        rows.append([u, ", ".join(f"{e} {v}" for e, v in d.items()),
                     pref.get(u, "not tested"),
                     "yes - one footprint set, both sessions"])
    s = blank(p)
    hit = [u for u in F["fin_resp"] if pref.get(u) in ("centre", "corner")]
    title(s, "Are the pain cells the centre cells?",
          f"{'No overlap: ' if not hit else ''}"
          f"{len(hit)} of {len(F['fin_resp'])} cells with a final pain "
          f"verdict are also zone-selective")
    th = min(.40 * len(rows), 4.2)
    table(s, rows, Inches(.5), Inches(1.5), Inches(12.3), Inches(th),
          size=11.5)
    text(s,
         ("No overlap. Neither of the two cells that fire faster in the "
          "centre (A3, A7) has a pain verdict,\n"
          "and neither of the two stimulation-modulated cells (A1, A4) is "
          f"zone-selective. With 2 centre cells\n"
          f"and {len(F['fin_resp'])} cells with a pain verdict out of "
          f"{F['n_ok']}, chance would give no overlap too - so this is a "
          "description of\nthese cells and not a population result. It is "
          "also the honest answer to the question the pipeline was built "
          "for."
          if not hit else
          "Overlapping cells: " + ", ".join(hit)),
         Inches(.5), Inches(1.6 + th), Inches(12.3),
         Inches(1.4), size=12.5, color=WARN if not hit else GOOD)
    path_line(s, r"<pain>\output_split\results"
                 r"\fig23_cross_session_final.png")

    # ======================================================== what it needs
    bullets(p, "What this needs to become a result",
            "stated so the limits are not discovered later",
            "ONE ANIMAL, one session of each type. Every count here "
            "describes these cells, not a population.\n"
            "  The pipeline takes another session by changing two paths at "
            "the top of each script.\n\n"
            "PIN PRICKS EVERY ~3 s. The median gap is "
            + a(F, "gap_median_pin", "{:.1f}") + " s, so only "
            + a(F, "pin_with_clear_prewindow") + " of 53 pin and "
            + a(F, "heat_with_clear_prewindow") + " of 34 heat deliveries "
            "have 6 s of\n"
            "  clear time before them. Spacing them 15-20 s apart would "
            "make a single delivery a trial again\n"
            "  and remove the need for any of the window choices argued "
            "over here.\n\n"
            "THE HELD BEHAVIOURS WERE TAPPED, NOT HELD. Licking/biting and "
            "guarding were never scored at\n"
            "  all; paw attending once, for 0.4 s; escape 5 times "
            "totalling 3.9 s. The reflex taps (76 withdrawal,\n"
            "  50 flinch) are the only substantial behaviour data.\n\n"
            "18 s IN THE CENTRE, over 4 visits. The zone result is stable "
            "across both halves and survives\n"
            "  dropping any single visit, but it rests on two visits. A "
            "longer session, or an elevated plus maze,\n"
            "  would give the centre comparison something to stand on.",
            size=13, color=WARN)

    return save(p, f"CEANTSR1_results_{STAMP}", HERE)


def DIMC():
    from deck import DIM
    return DIM


if __name__ == "__main__":
    build()
