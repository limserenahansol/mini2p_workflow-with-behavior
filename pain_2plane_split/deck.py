"""deck.py  -  shared slide helpers for the processing and results decks.

Built to the three rules the figures follow, applied to slides:
  1  readable        nothing below 11 pt, no slide with more than ~10 lines
  2  statistics      every result slide names its test, n and p or q
  3  scales          figures carry their own axes and colour bars; the slide
                     says what to look at and where the file is

Never uses PowerPoint COM: that attaches to the running instance and closes
whatever Hansol has open. python-pptx writes the file directly, and a tall
figure is cut into slide-shaped bands rather than shrunk to a sliver.
"""
from __future__ import annotations

import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

W, H = Inches(13.333), Inches(7.5)
INK, DIM = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x5A, 0x5A, 0x5A)
ACC = RGBColor(0x1C, 0x6E, 0x8C)
GOOD = RGBColor(0x2E, 0x7D, 0x5B)
WARN = RGBColor(0xA6, 0x3A, 0x3A)
OVERLAP = 0.06
SCRATCH = ("C:/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/"
           "05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad/deckimg")


def new_deck():
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(p):
    return p.slides.add_slide(p.slide_layouts[6])


def text(s, t, x, y, w, h, size=14, bold=False, color=INK, mono=False,
         align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(t).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(3)
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold = Pt(size), bold
        r.font.color.rgb = color
        r.font.name = "Consolas" if mono else "Calibri"
    return tf


def title(s, main, sub=None, color=INK):
    text(s, main, Inches(.45), Inches(.22), Inches(12.4), Inches(.55),
         size=23, bold=True, color=color)
    if sub:
        text(s, sub, Inches(.45), Inches(.84), Inches(12.4), Inches(.42),
             size=13, color=ACC)


def path_line(s, t):
    text(s, t, Inches(.45), Inches(7.02), Inches(12.4), Inches(.36),
         size=9.5, color=DIM, mono=True)


def section(p, n, head, lines):
    """A divider so the deck's structure is visible while reading it."""
    s = blank(p)
    text(s, f"{n}", Inches(.7), Inches(2.0), Inches(2.0), Inches(1.4),
         size=72, bold=True, color=ACC)
    text(s, head, Inches(.7), Inches(3.3), Inches(11.9), Inches(.9),
         size=30, bold=True)
    text(s, lines, Inches(.75), Inches(4.35), Inches(11.9), Inches(2.2),
         size=15, color=DIM)
    return s


def bullets(p, head, sub, lines, size=14, pathline=None, color=INK):
    s = blank(p)
    title(s, head, sub, color=color)
    text(s, lines, Inches(.5), Inches(1.45), Inches(12.3), Inches(5.3),
         size=size)
    if pathline:
        path_line(s, pathline)
    return s


def table(s, rows, x, y, w, h, size=12):
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = (i == 0)
                    r.font.color.rgb = INK
                    r.font.name = "Calibri"
    return t


def fig(p, head, look, path, pathline=None, hbox=5.45, y=1.35,
        slice_tall=True, note=None):
    """One slide per figure, or bands for a figure taller than the slide.

    Shrinking a 70-inch figure to fit 5.5 inches of slide makes it 2 inches
    wide and unreadable, which is exactly the complaint this deck exists to
    answer. Anything taller than the box is cut into horizontal bands of the
    slide's own aspect ratio, with a 6 % overlap so a row is never split
    without context.
    """
    if not os.path.exists(path):
        raise SystemExit(f"missing figure: {path}")
    iw, ih = Image.open(path).size
    target = 12.4 / hbox
    n_band = 1 if not slice_tall else max(1, int(round(ih / (iw / target))))
    os.makedirs(SCRATCH, exist_ok=True)
    for b in range(n_band):
        if n_band == 1:
            band = path
        else:
            band = os.path.join(
                SCRATCH, f"{os.path.basename(path)[:-4]}_{b + 1}of"
                         f"{n_band}.png")
            step = ih / n_band
            top = int(max(0, b * step - (OVERLAP * step if b else 0)))
            bot = int(min(ih, (b + 1) * step
                          + (OVERLAP * step if b < n_band - 1 else 0)))
            Image.open(path).crop((0, top, iw, bot)).save(band)
        bw, bh = Image.open(band).size
        s = blank(p)
        title(s, head if n_band == 1 else f"{head}   ({b + 1} of {n_band})",
              look if b == 0 else "continued")
        mw, mh = Inches(12.4), Inches(hbox)
        sc = min(mw / bw, mh / bh)
        s.shapes.add_picture(band, Inches(.45)
                             + Emu(int((mw - bw * sc) / 2)), Inches(y),
                             width=Emu(int(bw * sc)),
                             height=Emu(int(bh * sc)))
        if note and b == n_band - 1:
            text(s, note, Inches(.45), Inches(y + hbox + .05),
                 Inches(12.4), Inches(.5), size=11, color=DIM)
        if pathline:
            path_line(s, pathline)
    return n_band


def save(p, stem, here):
    """Write to a fresh version rather than fight a file open in PowerPoint.

    python-pptx truncates the target before it discovers it cannot write,
    so overwriting an open deck destroys it.
    """
    out, n = os.path.join(here, stem + ".pptx"), 1
    while True:
        try:
            with open(out, "ab"):
                pass
            break
        except PermissionError:
            n += 1
            out = os.path.join(here, f"{stem}_v{n}.pptx")
    p.save(out)
    print(f"wrote {out}  ({len(p.slides.__iter__.__self__._sldIdLst)} "
          f"slides)")
    return out
