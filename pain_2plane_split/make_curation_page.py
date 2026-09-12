"""make_curation_page.py  -  a self-contained page for keeping or discarding cells.

WHY A FILE AND NOT AN INLINE WIDGET
  An inline chat widget would have to carry its images as base64 typed into
  the tool call, which forces the thumbnails down to ~56 px to stay
  affordable. Judging "is there a soma under this outline" wants full
  resolution. Writing a standalone page instead costs nothing per pixel, so
  the crops stay at their native 110 px and the whole-FOV context image is
  included too.

WHAT IS SHOWN PER CANDIDATE
  MEAN  output.info.summary_image - the session average. A real soma is a
        filled blob here whether or not it ever fired.
  MAX   output.info.max_image - the per-pixel maximum over time. An ACTIVE
        soma is brightest here; neuropil and vessels are not.
  Both are stretched on the WHOLE plane's percentiles, never per crop, so a
  dim candidate looks dim instead of being auto-brightened into looking
  convincing.

  Metrics come from step 2's QC_report.txt (area, eccentricity, solidity) and
  extract_qc_cells.py (anat contrast, events, p_joint, verdict, blob count).
  Eccentricity is included because the automatic QC measures brightness
  against the surround and says nothing about SHAPE - a bright elongated
  fibre passes it. Pain plane A #6 and #12 are visibly fibres and are the two
  highest eccentricities in that plane.

  The pre-selected suggestion answers "is this a cell", which is NOT "is this
  usable for the response analysis": a silent neuron is still a cell. So only
  no-signal-and-no-contrast, multi-blob, oversized and elongated footprints
  are pre-marked discard.

OUTPUT
  <session>\\output_split\\curation\\curation_select.html
  Open it, click cards to toggle, then use Copy result or Download CSV.

USAGE
  python make_curation_page.py
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd

from cell_curation_sheet import HI, LO, REL, SESSIONS, read_morph

CROP = 110
OVERVIEW_SCALE = 1.6


def stretch(img):
    lo, hi = np.percentile(img, [LO, HI])
    return (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(
        np.uint8)


def jpg(img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def outline(col, shape):
    bw = (col.reshape(shape) > REL * col.max()).astype(np.uint8)
    return cv2.findContours(bw, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def crop(img8, cy, cx, shape, cnts):
    h, w = shape
    y0 = int(np.clip(cy - CROP // 2, 0, max(h - CROP, 0)))
    x0 = int(np.clip(cx - CROP // 2, 0, max(w - CROP, 0)))
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, cnts, -1, (0, 240, 255), 1)
    sub = vis[y0:y0 + CROP, x0:x0 + CROP]
    if sub.shape[:2] != (CROP, CROP):
        pad = np.zeros((CROP, CROP, 3), np.uint8)
        pad[:sub.shape[0], :sub.shape[1]] = sub
        sub = pad
    return sub


def build():
    items, overviews = [], {}
    for tag, session in SESSIONS:
        short = {"openfield": "OF", "pain": "PA"}[tag]
        for plane in ("A", "B"):
            folder = os.path.join(session, "output_split", f"plane_{plane}")
            with h5py.File(os.path.join(folder,
                                        "final_analysis_results.mat"),
                           "r") as f:
                S3 = np.array(f["output"]["spatial_weights"])
                mean_img = np.array(f["output"]["info"]["summary_image"]).T
                max_img = np.array(f["output"]["info"]["max_image"]).T
            k, w, h = S3.shape
            S = S3.transpose(0, 2, 1).reshape(k, h * w).T
            qc = pd.read_csv(os.path.join(folder, "qc_cells",
                                          "qc_cells.csv"))
            morph = read_morph(folder)
            m8, x8 = stretch(mean_img), stretch(max_img)
            key = f"{short}-{plane}"

            ov = cv2.cvtColor(x8, cv2.COLOR_GRAY2BGR)
            for i in range(k):
                col = S[:, i]
                cn = outline(col, (h, w))
                cv2.drawContours(ov, cn, -1, (0, 240, 255), 1)
                img = col.reshape((h, w))
                ys, xs = np.nonzero(img)
                wt = img[ys, xs]
                cv2.putText(ov, str(i + 1),
                            (int(np.average(xs, weights=wt)) + 7,
                             int(np.average(ys, weights=wt)) - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 255, 255), 1)
            overviews[key] = jpg(cv2.resize(
                ov, (int(w * OVERVIEW_SCALE), int(h * OVERVIEW_SCALE)),
                interpolation=cv2.INTER_CUBIC), 88)

            for i in range(k):
                col = S[:, i]
                img = col.reshape((h, w))
                ys, xs = np.nonzero(img)
                wt = img[ys, xs]
                cy, cx = np.average(ys, weights=wt), np.average(xs, weights=wt)
                cn = outline(col, (h, w))
                r = qc[qc["cell"] == i + 1].iloc[0]
                ecc, sol = morph.get(i + 1, (float("nan"), float("nan")))
                elong = ecc == ecc and ecc >= 0.90
                flags = []
                if r["verdict"] == "SUSPECT":
                    flags.append("no signal + no contrast")
                if r["n_blobs"] > 1:
                    flags.append(f"{int(r['n_blobs'])} blobs")
                if r["big_footprint"] == 1:
                    flags.append("oversized")
                if elong:
                    flags.append(f"elongated ecc {ecc:.2f}")
                items.append(dict(
                    id=f"{key}#{i + 1}", group=key, cell=i + 1,
                    mean=jpg(crop(m8, cy, cx, (h, w), cn)),
                    max=jpg(crop(x8, cy, cx, (h, w), cn)),
                    row=int(round(cy)), col=int(round(cx)),
                    area=int(r["area_px"]),
                    ecc=None if ecc != ecc else round(float(ecc), 3),
                    sol=None if sol != sol else round(float(sol), 3),
                    blobs=int(r["n_blobs"]),
                    anat=round(float(r["anat"]), 4),
                    events=int(r["events"]),
                    p=round(float(r["p_joint"]), 4),
                    verdict=r["verdict"], flags=flags,
                    suggest="discard" if flags else "keep"))
            print(f"  {key}: {k} candidates")
    return items, overviews


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CEA-Ntsr1 cell curation</title>
<style>
:root{--bg:#101214;--fg:#e8eaed;--dim:#9aa0a6;--line:#2a2f36;--keep:#2e7d5b;
--drop:#a63a3a;--card:#171a1e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--bg);
border-bottom:1px solid var(--line);padding:10px 14px}
h1{font-size:15px;margin:0 0 8px}
button{background:#1e232a;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer}
button:hover{background:#262c34}
button.on{background:#2b3a4a;border-color:#3d5a7a}
.tabs{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.counts{margin-left:auto;font-size:12px;color:var(--dim)}
.ov{padding:10px 14px 0}
.ov img{max-width:100%;border:1px solid var(--line);border-radius:6px}
.ov p{color:var(--dim);font-size:12px;margin:6px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));
gap:10px;padding:12px 14px 90px}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;
padding:8px;cursor:pointer;user-select:none}
.card[data-d=keep]{border-color:var(--keep)}
.card[data-d=discard]{border-color:var(--drop);opacity:.62}
.card h3{margin:0 0 5px;font-size:13px;display:flex;gap:6px;
align-items:center}
.tick{font-size:11px;padding:1px 6px;border-radius:4px;font-weight:600}
.card[data-d=keep] .tick{background:var(--keep);color:#fff}
.card[data-d=discard] .tick{background:var(--drop);color:#fff}
.imgs{display:flex;gap:4px}
.imgs figure{margin:0;flex:1}
.imgs img{width:100%;display:block;border-radius:4px;
image-rendering:pixelated}
.imgs figcaption{font-size:11px;color:var(--dim);text-align:center;
padding-top:2px}
.m{font-size:11px;color:var(--dim);margin-top:5px;
font-family:ui-monospace,Consolas,monospace}
.f{font-size:11px;color:#e79a9a;margin-top:3px}
footer{position:fixed;bottom:0;left:0;right:0;background:#0c0e10;
border-top:1px solid var(--line);padding:9px 14px;display:flex;gap:8px;
align-items:center;flex-wrap:wrap}
#out{flex:1;min-width:240px;background:#171a1e;color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:6px;font-size:11px;
font-family:ui-monospace,Consolas,monospace;height:44px;resize:vertical}
</style></head><body>
<header>
<h1>Cell curation — click a card to keep or discard</h1>
<div class="tabs" id="tabs"></div>
</header>
<div class="ov" id="ov"></div>
<div class="grid" id="grid"></div>
<footer>
<button id="sug">Reset to suggestion</button>
<button id="allk">Keep all shown</button>
<button id="alld">Discard all shown</button>
<textarea id="out" readonly></textarea>
<button id="copy">Copy result</button>
<button id="csv">Download CSV</button>
</footer>
<script>
const ITEMS=__ITEMS__, OVERVIEW=__OV__;
const GROUPS=[...new Set(ITEMS.map(i=>i.group))];
let tab=GROUPS[0], flagged=false;
const dec={}; ITEMS.forEach(i=>dec[i.id]=i.suggest);
const tabs=document.getElementById('tabs');
GROUPS.concat(['ALL']).forEach(g=>{const b=document.createElement('button');
 b.textContent=g; b.onclick=()=>{tab=g;draw()}; b.dataset.g=g; tabs.append(b)});
const fb=document.createElement('button');
fb.textContent='Flagged only'; fb.onclick=()=>{flagged=!flagged;draw()};
tabs.append(fb);
const cnt=document.createElement('span'); cnt.className='counts';
tabs.append(cnt);
function shown(){return ITEMS.filter(i=>(tab==='ALL'||i.group===tab)
 &&(!flagged||i.flags.length))}
function result(){const d=ITEMS.filter(i=>dec[i.id]==='discard');
 const by={}; d.forEach(i=>{(by[i.group]=by[i.group]||[]).push(i.cell)});
 const parts=Object.keys(by).map(g=>g+' '+by[g].sort((a,b)=>a-b).join(','));
 return 'CURATION RESULT - discard '+d.length+' of '+ITEMS.length+': '+
  (parts.join(' | ')||'none')+' - keep all others';}
function draw(){
 [...tabs.querySelectorAll('button')].forEach(b=>
   b.classList.toggle('on', b.dataset.g===tab||(b===fb&&flagged)));
 const o=document.getElementById('ov');
 o.innerHTML = (tab!=='ALL'&&OVERVIEW[tab])
  ? '<img src="'+OVERVIEW[tab]+'" alt="whole field of view, '+tab+'">'+
    '<p>Whole field of view, MAX over time, numbered. Yellow = footprint '+
    'outline.</p>' : '';
 const g=document.getElementById('grid'); g.innerHTML='';
 shown().forEach(i=>{
  const c=document.createElement('div'); c.className='card';
  c.dataset.d=dec[i.id];
  c.onclick=()=>{dec[i.id]=dec[i.id]==='keep'?'discard':'keep';
   c.dataset.d=dec[i.id]; c.querySelector('.tick').textContent=dec[i.id];
   sync()};
  const ecc=i.ecc==null?'--':i.ecc.toFixed(2);
  const sol=i.sol==null?'--':i.sol.toFixed(2);
  c.innerHTML='<h3>'+i.id+' <span class="tick">'+dec[i.id]+'</span>'+
   '<span style="font-size:11px;color:var(--dim);font-weight:400">'+
   i.verdict+'</span></h3>'+
   '<div class="imgs"><figure><img src="'+i.mean+'" alt="mean image, '+
   i.id+'"><figcaption>mean</figcaption></figure>'+
   '<figure><img src="'+i.max+'" alt="max over time, '+i.id+
   '"><figcaption>max</figcaption></figure></div>'+
   '<div class="m">area '+i.area+'  ecc '+ecc+'  sol '+sol+
   '  blobs '+i.blobs+'<br>anat '+i.anat.toFixed(3)+'  ev '+i.events+
   '  p '+i.p.toFixed(3)+'  at row '+i.row+' col '+i.col+'</div>'+
   (i.flags.length?'<div class="f">'+i.flags.join(' | ')+'</div>':'');
  g.append(c)});
 sync();
}
function sync(){
 const k=ITEMS.filter(i=>dec[i.id]==='keep').length;
 cnt.textContent='keep '+k+'  /  discard '+(ITEMS.length-k)+
  '   (showing '+shown().length+')';
 document.getElementById('out').value=result();
}
document.getElementById('sug').onclick=()=>{
 ITEMS.forEach(i=>dec[i.id]=i.suggest); draw()};
document.getElementById('allk').onclick=()=>{
 shown().forEach(i=>dec[i.id]='keep'); draw()};
document.getElementById('alld').onclick=()=>{
 shown().forEach(i=>dec[i.id]='discard'); draw()};
document.getElementById('copy').onclick=()=>{
 const t=document.getElementById('out'); t.select();
 navigator.clipboard.writeText(t.value).catch(()=>document.execCommand('copy'));
 const b=document.getElementById('copy'); b.textContent='Copied';
 setTimeout(()=>b.textContent='Copy result',1200)};
document.getElementById('csv').onclick=()=>{
 const rows=[['id','group','cell','row','col','area','ecc','sol','blobs',
  'anat','events','p_joint','auto_verdict','flags','DECISION'].join(',')];
 ITEMS.forEach(i=>rows.push([i.id,i.group,i.cell,i.row,i.col,i.area,
  i.ecc==null?'':i.ecc,i.sol==null?'':i.sol,i.blobs,i.anat,i.events,i.p,
  i.verdict,'"'+i.flags.join('; ')+'"',dec[i.id]].join(',')));
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));
 a.download='curation_decisions.csv'; a.click()};
draw();
</script></body></html>"""


def main():
    items, overviews = build()
    html = (HTML.replace("__ITEMS__", json.dumps(items))
                .replace("__OV__", json.dumps(overviews)))
    paths = []
    for _tag, session in SESSIONS:
        outdir = os.path.join(session, "output_split", "curation")
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, "curation_select.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        paths.append(p)
    n_d = sum(1 for i in items if i["suggest"] == "discard")
    print(f"\n{len(items)} candidates, suggested discard {n_d}, "
          f"page {len(html) / 1e6:.2f} MB")
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
