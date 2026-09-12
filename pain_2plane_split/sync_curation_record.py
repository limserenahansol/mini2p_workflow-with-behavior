"""sync_curation_record.py  -  make the curation folder state what was actually done.

WHY
  <session>\\output_split\\curation\\ holds the selection SHEETS - the input to
  the decision - so it still shows every original candidate, which is
  correct. But curation_template.csv in the same folder was left holding my
  pre-filled SUGGESTION, not Hansol's decisions, so opening that folder made
  it look as though the discards had never been applied. The applied result
  lives in <plane>\\curated\\curated_labels.csv.

  This writes curation_applied.csv next to the sheets, taken from the applied
  labels, and rewrites curation_template.csv's DECISION column to match, so
  nothing in that folder disagrees with what was actually run.

OUTPUT  ->  <session>\\output_split\\curation\\curation_applied.csv
            <session>\\output_split\\curation\\curation_template.csv (updated)
"""
from __future__ import annotations

import os

import pandas as pd

SESSIONS = [
    ("openfield", "D:/20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_"
                  "2026-09-11_15-27-52"),
    ("pain", "D:/20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_"
             "2026-09-11_15-56-48"),
]
SHORT = {"openfield": "OF", "pain": "PA"}


def main():
    rows = []
    for tag, sess in SESSIONS:
        for plane in ("A", "B"):
            f = os.path.join(sess, "output_split", f"plane_{plane}",
                             "curated", "curated_labels.csv")
            if not os.path.exists(f):
                print(f"  {tag} {plane}: no applied labels yet")
                continue
            d = pd.read_csv(f, dtype={"label": str})
            for _, r in d.iterrows():
                rows.append(dict(
                    session=tag, plane=plane,
                    id=f"{SHORT[tag]}-{plane}#{r['label']}",
                    label=r["label"],
                    original_cell=int(r["origin"]) if r["origin"] > 0 else "",
                    applied=("discard" if r["index"] == 0 else "kept"),
                    operation=r["op"],
                    curated_index=int(r["index"]),
                    area_px=int(r["area_px"])))
    if not rows:
        raise SystemExit("nothing applied yet")
    D = pd.DataFrame(rows)

    for tag, sess in SESSIONS:
        outdir = os.path.join(sess, "output_split", "curation")
        if not os.path.isdir(outdir):
            continue
        D.to_csv(os.path.join(outdir, "curation_applied.csv"), index=False)

        tpl = os.path.join(outdir, "curation_template.csv")
        if os.path.exists(tpl):
            t = pd.read_csv(tpl)
            key = D.set_index(["session", "plane", "label"])
            dec, note = [], []
            for _, r in t.iterrows():
                k = (r["session"], r["plane"], str(r["cell"]))
                if k in key.index:
                    v = key.loc[k]
                    dec.append(v["applied"])
                    note.append(v["operation"])
                else:
                    # a split parent no longer exists under its own number
                    kids = D[(D["session"] == r["session"])
                             & (D["plane"] == r["plane"])
                             & (D["original_cell"] == r["cell"])]
                    if len(kids):
                        dec.append("kept")
                        note.append("; ".join(
                            f"{x['label']}: {x['operation']}"
                            for _, x in kids.iterrows()))
                    else:
                        dec.append("?")
                        note.append("not found in the applied labels")
            t["DECISION"] = dec
            t["APPLIED_OP"] = note
            t.to_csv(tpl, index=False)

    print(D.groupby(["session", "plane", "applied"]).size().to_string())
    print("\nper plane: originals discarded, and what changed")
    for (tag, plane), g in D.groupby(["session", "plane"]):
        dropped = g[g["applied"] == "discard"]["label"].tolist()
        changed = g[(g["applied"] == "kept")
                    & (g["operation"] != "keep")]
        print(f"  {SHORT[tag]}-{plane}: {int((g['applied'] == 'kept').sum())} "
              f"cells kept"
              + (f", discarded {', '.join(dropped)}" if dropped else "")
              + ("".join(f", {r['label']} = {r['operation']}"
                         for _, r in changed.iterrows())))
    print("\nwrote curation_applied.csv and refreshed curation_template.csv "
          "in both curation folders")


if __name__ == "__main__":
    main()
