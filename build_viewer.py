#!/usr/bin/env python3
"""
build_viewer.py
===============
Turns the narrative_XX.json files written by proposition_tracker.py into

  * viewer.html                  interactive, self-contained viewer (no server, no internet needed)
  * narrative_trajectories.csv   one row per narrative step, every similarity measured against DAY 1
  * narrative_summary.csv        one row per narrative: "1.00 > 0.71 > 0.35", total drop, biggest drop, ...

It runs automatically at the end of proposition_tracker.py. You can also run it on its own on an
existing output folder (older JSONs without day-1 fields get an approximation, marked as such):

    python build_viewer.py proposition_out/fff_blogs
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from datetime import date
from pathlib import Path

from emotions import EMOTION_DIRECTION

log = logging.getLogger("build_viewer")
ELEMENTS = ("actor", "action", "object")
# A narrative is flagged as "likely boilerplate" (unchanged text such as a newsletter footer) when
#   (a) its SVO never moves away from day 1 (similarity to day 1 >= BOILER_SIM at every step), or
#   (b) its claims are nearly the same sentence in every step (mean word overlap >= BOILER_CLAIM, >= 3 steps).
# Flagged narratives are hidden in the viewer by default (a checkbox shows them) and marked in the CSV.
BOILER_SIM = 0.95
BOILER_CLAIM = 0.90
VALENCE = EMOTION_DIRECTION  # decides what counts as an emotion "flip"; defined in emotions.py


def _orient(a: str, b: str) -> str:
    return "same" if a == b else ("flip" if {a, b} == {"supportive", "critical"} else "shift")


def _emo(a: str, b: str) -> str:
    return "same" if a == b else ("flip" if VALENCE.get(a, 0) * VALENCE.get(b, 0) < 0 else "shift")


def rnd(v, k=4):
    return None if v is None else round(float(v), k)


# --------------------------------------------------------------------------- #
# Loading / normalising
# --------------------------------------------------------------------------- #
def normalize(rec: dict) -> dict:
    """Make sure every step carries its similarity to day 1 (approximate for old JSONs)."""
    steps, trans = rec["steps"], rec["transitions"]
    d0 = date.fromisoformat(steps[0]["date"])
    approx = "sim_to_start" not in steps[0]
    if approx:  # old JSON: chain the step similarities (a rough lower-bound style proxy)
        sim, cos = 1.0, {e: 1.0 for e in ELEMENTS}
        for i, s in enumerate(steps):
            if i:
                t = trans[i - 1]
                sim *= t["sim"]
                for e in ELEMENTS:
                    c = t["cos"].get(e)
                    cos[e] = None if (c is None or cos[e] is None) else cos[e] * c
            s["sim_to_start"], s["cos_to_start"] = sim, dict(cos)
    rec["sim_approx"] = approx
    for s in steps:
        s["days_since_start"] = (date.fromisoformat(s["date"]) - d0).days
        s.setdefault("orientation_vs_start", _orient(steps[0]["orientation"], s["orientation"]))
        s.setdefault("emotion_vs_start", _emo(steps[0]["emotion"], s["emotion"]))
    return rec


def load_records(out_dir: Path) -> list:
    recs = []
    for p in sorted(out_dir.glob("narrative_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["id"] = p.stem
        recs.append(normalize(rec))
    return recs


# --------------------------------------------------------------------------- #
# CSV tables (always relative to day 1)
# --------------------------------------------------------------------------- #
def trajectory_rows(records: list) -> list:
    rows = []
    for r in records:
        for i, s in enumerate(r["steps"]):
            t = r["transitions"][i - 1] if i else None
            c0 = s.get("cos_to_start") or {}
            rows.append({
                "narrative": r["id"], "title": r["title"], "step": i + 1, "doc_id": s["doc_id"], "date": s["date"],
                "days_since_day1": s["days_since_start"],
                "sim_to_day1": rnd(s["sim_to_start"]), "sim_to_previous": rnd(t["sim"]) if t else None,
                "cos_actor_day1": rnd(c0.get("actor")), "cos_action_day1": rnd(c0.get("action")),
                "cos_object_day1": rnd(c0.get("object")),
                "orientation": s["orientation"], "orientation_vs_day1": s["orientation_vs_start"],
                "emotion": s["emotion"], "emotion_vs_day1": s["emotion_vs_start"],
                "actor": s["actor"], "action": s["action"], "object": s["object"], "claim": s["claim"],
                "change_vs_previous": t["label"] if t else "first step",
                "similarity_exact": not r["sim_approx"],
            })
    return rows


def _words(text) -> set:
    return set(re.findall(r"\w+", str(text).lower()))


def classify_boilerplate(rec: dict, sim_thr: float = BOILER_SIM, claim_thr: float = BOILER_CLAIM) -> dict:
    st = rec["steps"]
    min_sim = min(s["sim_to_start"] for s in st)
    ov = []
    for a, b in zip(st, st[1:]):
        wa, wb = _words(a["claim"]), _words(b["claim"])
        ov.append(len(wa & wb) / len(wa | wb) if wa | wb else 1.0)
    mean_ov = sum(ov) / len(ov) if ov else 0.0
    if min_sim >= sim_thr:
        flag, why = True, f"SVO unchanged (similarity to day 1 never below {min_sim:.2f})"
    elif mean_ov >= claim_thr and len(st) >= 3:
        flag, why = True, f"claims nearly identical (mean word overlap {mean_ov:.2f})"
    else:
        flag, why = False, ""
    return {"flag": flag, "reason": why, "claim_overlap": round(mean_ov, 3)}


def _trail(seq: list) -> str:
    return " > ".join(x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1])


def summary_rows(records: list) -> list:
    rows = []
    for r in records:
        st = r["steps"]
        sims = [s["sim_to_start"] for s in st]
        best = (0.0, "", "")
        for i in range(1, len(st)):
            d = sims[i - 1] - sims[i]
            if d > best[0]:
                best = (d, st[i - 1]["date"], st[i]["date"])
        last = st[-1]
        rows.append({
            "narrative": r["id"], "title": r["title"], "start_date": st[0]["date"], "end_date": last["date"],
            "n_documents": len(st), "days_span": last["days_since_start"],
            "sim_to_day1_trajectory": " > ".join(f"{x:.2f}" for x in sims),
            "sim_to_day1_last": rnd(sims[-1]), "sim_to_day1_min": rnd(min(sims)), "total_drop": rnd(max(0.0, 1 - sims[-1])),
            "largest_step_drop": rnd(best[0]), "largest_drop_from": best[1], "largest_drop_to": best[2],
            "actor_cos_day1_last": rnd((last.get("cos_to_start") or {}).get("actor")),
            "action_cos_day1_last": rnd((last.get("cos_to_start") or {}).get("action")),
            "object_cos_day1_last": rnd((last.get("cos_to_start") or {}).get("object")),
            "orientation_trajectory": _trail([s["orientation"] for s in st]),
            "emotion_trajectory": _trail([s["emotion"] for s in st]),
            "orientation_reversal": r["flags"].get("orientation_reversal"),
            "emotion_valence_flip": r["flags"].get("emotion_valence_flip"),
            "interrupted": r["flags"].get("interrupted"), "similarity_exact": not r["sim_approx"],
            "likely_boilerplate": r["boilerplate"]["flag"], "boilerplate_reason": r["boilerplate"]["reason"],
            "claim_overlap_mean": r["boilerplate"]["claim_overlap"],
        })
    return rows


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows umlauts correctly
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(out_dir, same_thr=None, drift_thr=None, boiler_sim=BOILER_SIM, boiler_claim=BOILER_CLAIM) -> None:
    out_dir = Path(out_dir)
    records = load_records(out_dir)
    if not records:
        log.warning("%s: no narrative_*.json found, nothing to build", out_dir)
        return
    for r in records:
        r["boilerplate"] = classify_boilerplate(r, boiler_sim, boiler_claim)
    n_b = sum(r["boilerplate"]["flag"] for r in records)
    log.info("%d of %d narratives look like unchanged boilerplate (hidden by default in viewer.html)", n_b, len(records))
    cfg = {"same_thr": 0.85, "drift_thr": 0.60}
    if (out_dir / "config.json").exists():
        cfg.update(json.loads((out_dir / "config.json").read_text()))
    if same_thr is not None:
        cfg["same_thr"] = same_thr
    if drift_thr is not None:
        cfg["drift_thr"] = drift_thr
    write_csv(out_dir / "narrative_trajectories.csv", trajectory_rows(records))
    write_csv(out_dir / "narrative_summary.csv", summary_rows(records))
    payload = json.dumps({"dataset": out_dir.name, "config": cfg, "narratives": records}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # keep the embedded JSON from closing the <script> tag
    (out_dir / "viewer.html").write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    log.info("Wrote viewer.html, narrative_trajectories.csv, narrative_summary.csv in %s", out_dir)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proposition narratives</title>
<style>
:root{--bg:#fff;--panel:#f6f6f3;--text:#1d1d1b;--muted:#6b6b66;--border:#dcdcd6;--accent:#2b6cb0}
@media (prefers-color-scheme: dark){:root{--bg:#171716;--panel:#222220;--text:#ecece8;--muted:#a3a39c;--border:#3a3a37;--accent:#6aa7e8}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
#app{display:grid;grid-template-columns:300px 1fr;height:100vh}
#side{border-right:1px solid var(--border);overflow:auto;padding:16px;background:var(--panel)}
#main{overflow:auto;padding:22px 28px 60px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 2px}h3{font-size:14px;margin:0 0 8px}
.muted{color:var(--muted)}.small{font-size:12px}
.btn{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:4px 10px;cursor:pointer}
.btn:hover{border-color:var(--accent)}.btn.on{background:var(--accent);border-color:var(--accent);color:#fff}.btn.wide{width:100%;margin:10px 0}
.sortl{display:block;margin:6px 0 10px}select{font:inherit;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px}
.item{padding:8px 10px;border:1px solid transparent;border-radius:8px;cursor:pointer;margin-bottom:4px}
.item:hover{background:var(--bg)}.item.on{background:var(--bg);border-color:var(--accent)}.it{font-weight:600}
.bar{height:5px;background:var(--border);border-radius:3px;margin-top:5px}.bar i{display:block;height:5px;border-radius:3px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:14px 0}
.summary{max-width:900px;color:var(--muted);margin:0 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.tag{border:1px solid var(--border);border-radius:99px;padding:1px 10px;font-size:12px;background:var(--panel)}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}.kpi{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:8px 14px;min-width:150px}
.kpi b{display:block;font-size:18px}
.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:10px 0}.seg{display:flex;gap:4px;align-items:center}.seg .muted{margin-right:4px}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--border);stroke-width:1}.tick{fill:var(--muted);font-size:11px}
.pt{cursor:pointer}.line{pointer-events:none}.hit{cursor:pointer}
.stepper{display:flex;gap:8px;align-items:center;margin:8px 0}.stepper input[type=range]{flex:1}
.claim{font-size:18px;margin:6px 0}blockquote{margin:8px 0;padding:6px 12px;border-left:3px solid var(--border);color:var(--muted)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px}th.sortable{cursor:pointer}
.chip{display:inline-block;border:2px solid var(--border);border-radius:8px;padding:2px 8px;background:var(--bg)}
.mwrap{overflow:auto}.mtab td,.mtab th{min-width:150px}.mtab td:first-child,.mtab th:first-child{min-width:110px;position:sticky;left:0;background:var(--panel);font-weight:600;color:var(--muted)}
.mtab th.sel,.mtab td.sel{background:rgba(43,108,176,.10)}.mtab th{cursor:pointer}
.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
tr.row{cursor:pointer}tr.row:hover{background:var(--panel)}
#tip{position:fixed;display:none;max-width:340px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 10px;box-shadow:0 4px 18px rgba(0,0,0,.18);pointer-events:none;z-index:9;font-size:12px}
.banner{background:#fff4e0;color:#6a4500;border:1px solid #e8c98a;border-radius:8px;padding:8px 12px;margin:0 0 12px;font-size:13px}
@media (max-width:800px){#app{grid-template-columns:1fr;height:auto}#side{max-height:260px}}
</style></head><body>
<div id="app"><aside id="side"></aside><main id="main"></main></div><div id="tip"></div>
<script>
(function(){
const DATA = __DATA__;
const CFG = DATA.config, NAR = DATA.narratives;
const C = {start:'#9a9a94', same:'#639922', drift:'#BA7517', shift:'#BA7517', changed:'#E24B4A', flip:'#E24B4A', na:'#bdbdb5'};
const ORI = {supportive:'#639922', critical:'#E24B4A', neutral:'#9a9a94', ambivalent:'#BA7517'};
const EMO = {hatred_hostility:'#7b1e1e', love:'#c2185b', solidarity_loyalty:'#2b6cb0', suspicion_paranoia:'#6a4c93', trust_respect:'#4d908e',
  anger:'#D85A30', grief_sorrow:'#4a78b5', outrage_indignation:'#b5361b', shame:'#8d6e63',
  compassion_sympathy:'#e07b9a', cynicism_depression:'#546e7a', defiance:'#1f7a3a', enthusiasm_pride:'#d4a017', envy_resentment:'#7a8b3a',
  fear_dread:'#8a5fb0', joy_hope:'#2a9d8f', resignation:'#78909c', neutral:'#9a9a94',
  sadness:'#4a78b5', disgust:'#7a8b3a', hope:'#2a9d8f', determination:'#1f7a3a', pride:'#d4a017', joy:'#e0b000', fear:'#8a5fb0'};
const PAL = ['#2b6cb0','#d9534f','#2a9d8f','#b8860b','#8a5fb0','#e07b39','#4d908e','#c2185b','#6d4c41','#546e7a'];
const state = {view:'overview', ni:0, step:0, ref:'prev', axis:'steps', oaxis:'days', extra:{actor:false, action:false, object:false}, listSort:'date', sort:{key:'sim', dir:1}, hideBoiler:true, timer:null};
let refs = null;

function mk(ns){ return function(tag, attrs, ...kids){
  const e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e; }; }
const h = mk(null), S = mk('http://www.w3.org/2000/svg');
const $ = id => document.getElementById(id);
const fmt = v => (v === null || v === undefined) ? 'n/a' : v.toFixed(2);
const simStatus = c => c == null ? 'na' : (c >= CFG.same_thr ? 'same' : (c >= CFG.drift_thr ? 'drift' : 'changed'));
const simColor = c => C[simStatus(c)];
const lastStep = r => r.steps[r.steps.length - 1];
const isBoiler = r => !!(r.boilerplate && r.boilerplate.flag);
const vis = r => !(state.hideBoiler && isBoiler(r));
const NB = NAR.filter(isBoiler).length;
const cap = s => { s = String(s).replace(/_/g, ' / '); return s.charAt(0).toUpperCase() + s.slice(1); };

function drops(r){
  let best = {d:0, a:null, b:null};
  for (let i = 1; i < r.steps.length; i++) {
    const d = r.steps[i-1].sim_to_start - r.steps[i].sim_to_start;
    if (d > best.d) best = {d, a:r.steps[i-1].date, b:r.steps[i].date};
  }
  return best;
}
function statusOf(r, i, key, ref){
  if (i === 0) return 'start';
  if (ref === 'prev') return (r.transitions[i-1].status || {})[key] || 'na';
  const s = r.steps[i];
  if (key === 'orientation') return s.orientation_vs_start;
  if (key === 'emotion') return s.emotion_vs_start;
  return simStatus((s.cos_to_start || {})[key]);
}
function flagList(r){
  return [['likely boilerplate: ' + ((r.boilerplate || {}).reason || ''), isBoiler(r)], ['orientation reversal', r.flags.orientation_reversal], ['emotion valence flip', r.flags.emotion_valence_flip],
          ['interrupted', r.flags.interrupted], ['approximate similarity', r.sim_approx]].filter(f => f[1]).map(f => f[0]);
}

const tip = $('tip');
function showTip(kids, e){
  tip.replaceChildren(...kids); tip.style.display = 'block';
  const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
}
const hideTip = () => { tip.style.display = 'none'; };
function setKids(el, ...kids){  // replaceChildren that flattens arrays and drops null/false
  el.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)
    .map(k => k.nodeType ? k : document.createTextNode(String(k))));
}

function drawChart(o){
  const W = o.width || 900, H = o.height || 320, m = {l:46, r:24, t:14, b:o.bottom || 50};
  const xs = o.lines.flatMap(l => l.pts.map(p => p.x));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x1 === x0) { x0 -= 0.5; x1 += 0.5; }
  const X = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), Y = y => m.t + (1 - y) * (H - m.t - m.b);
  const s = S('svg', {viewBox:`0 0 ${W} ${H}`, class:'chart', role:'img'});
  [0, .25, .5, .75, 1].forEach(v => s.append(S('line', {x1:m.l, x2:W-m.r, y1:Y(v), y2:Y(v), class:'grid'}),
                                              S('text', {x:m.l-6, y:Y(v)+4, 'text-anchor':'end', class:'tick'}, v.toFixed(2))));
  [[CFG.same_thr, 'same'], [CFG.drift_thr, 'drift']].forEach(([v, k]) =>
    s.append(S('line', {x1:m.l, x2:W-m.r, y1:Y(v), y2:Y(v), stroke:C[k], 'stroke-dasharray':'4 4', 'stroke-opacity':.7})));
  (o.ticks || []).forEach(t => { const tx = X(t.x), ty = H - m.b + 16;
    s.append(S('text', {x:tx, y:ty, 'text-anchor':o.rotate ? 'end' : 'middle', class:'tick', transform:o.rotate ? `rotate(-35 ${tx} ${ty})` : null}, t.label)); });
  if (o.selX !== undefined && o.selX !== null)
    s.append(S('line', {x1:X(o.selX), x2:X(o.selX), y1:m.t, y2:H-m.b, stroke:'var(--accent)', 'stroke-width':1.5, 'stroke-dasharray':'2 3'}));
  for (const l of o.lines) {
    const d = l.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.x)},${Y(p.y)}`).join(' ');
    s.append(S('path', {d, fill:'none', stroke:l.color, 'stroke-width':l.width || 2, 'stroke-dasharray':l.dashed ? '5 4' : null, class:'line', 'data-id':l.id}));
    if (o.onLine) {
      const hit = S('path', {d, fill:'none', stroke:'transparent', 'stroke-width':14, class:'hit'});
      hit.addEventListener('mouseenter', e => o.onLineHover && o.onLineHover(l, e));
      hit.addEventListener('mousemove', e => o.onLineHover && o.onLineHover(l, e));
      hit.addEventListener('mouseleave', () => o.onLineHover && o.onLineHover(null));
      hit.addEventListener('click', () => o.onLine(l));
      s.append(hit);
    }
    if (l.points) for (const p of l.pts) {
      const c = S('circle', {cx:X(p.x), cy:Y(p.y), r:p.r || 6, fill:p.color || l.color, stroke:'var(--bg)', 'stroke-width':2, class:'pt'});
      c.addEventListener('mouseenter', e => o.onHover && o.onHover(p, e));
      c.addEventListener('mousemove', e => o.onHover && o.onHover(p, e));
      c.addEventListener('mouseleave', () => o.onHover && o.onHover(null));
      c.addEventListener('click', () => o.onClick && o.onClick(p));
      s.append(c);
    }
    if (l.endMarker && l.pts.length) { const p = l.pts[l.pts.length - 1]; s.append(S('circle', {cx:X(p.x), cy:Y(p.y), r:4, fill:l.color, class:'line'})); }
  }
  return {svg:s, X, Y, H, m};
}

/* ---------- sidebar ---------- */
function order(){
  const f = {date:i => NAR[i].steps[0].date, docs:i => -NAR[i].steps.length,
             end:i => lastStep(NAR[i]).sim_to_start, drop:i => -drops(NAR[i]).d}[state.listSort];
  return NAR.map((_, i) => i).filter(i => vis(NAR[i])).sort((a, b) => (f(a) < f(b) ? -1 : (f(a) > f(b) ? 1 : 0)));
}
function renderSide(){
  const sel = h('select', {onchange:e => { state.listSort = e.target.value; renderSide(); }},
    [['date', 'start date'], ['docs', 'number of documents'], ['end', 'lowest end similarity'], ['drop', 'largest drop']]
      .map(([v, t]) => { const o = h('option', {value:v}, t); o.selected = state.listSort === v; return o; }));
  setKids($('side'), 
    h('h2', {}, DATA.dataset), h('div', {class:'muted'}, `${NAR.filter(vis).length} of ${NAR.length} narrative(s) shown`),
    NB ? h('label', {class:'muted sortl'}, h('input', {type:'checkbox', checked:state.hideBoiler, onchange:e => { state.hideBoiler = e.target.checked; renderAll(); }}), ` Hide likely boilerplate (${NB})`) : null,
    h('button', {class:'btn wide' + (state.view === 'overview' ? ' on' : ''), onclick:() => { stopPlay(); state.view = 'overview'; renderAll(); }}, 'Overview of all narratives'),
    h('label', {class:'muted sortl'}, 'Sort by ', sel),
    order().map(i => { const r = NAR[i], a = r.steps[0], b = lastStep(r);
      return h('div', {class:'item' + (state.view === 'narrative' && state.ni === i ? ' on' : ''), onclick:() => { stopPlay(); state.view = 'narrative'; state.ni = i; state.step = 0; renderAll(); }},
        h('div', {class:'it'}, `${i + 1}. ${r.title}`),
        h('div', {class:'muted small'}, `${a.date} → ${b.date} · ${r.steps.length} docs`),
        h('div', {class:'muted small'}, `similarity to day 1: ${fmt(a.sim_to_start)} → ${fmt(b.sim_to_start)}`),
        h('div', {class:'bar'}, h('i', {style:`width:${Math.max(2, Math.round(b.sim_to_start * 100))}%;background:${simColor(b.sim_to_start)}`}))); }));
}

/* ---------- narrative view ---------- */
const seg = (label, opts, cur, on) => h('div', {class:'seg'}, h('span', {class:'muted'}, label),
  opts.map(([v, t]) => h('button', {class:'btn' + (cur === v ? ' on' : ''), onclick:() => on(v)}, t)));

function renderNarrative(){
  const r = NAR[state.ni], n = r.steps.length;
  const holders = {controls:h('div', {class:'controls'}), chart:h('div', {class:'card'}), detail:h('div', {class:'card'}), matrix:h('div', {class:'card'})};
  const slider = h('input', {type:'range', min:0, max:n - 1, value:state.step, oninput:e => { stopPlay(); setStep(+e.target.value); }});
  const label = h('span', {class:'muted'});
  const play = h('button', {class:'btn', onclick:togglePlay}, '▶ Play');
  refs = {holders, slider, label, play, r};
  setKids($('main'), 
    r.sim_approx ? h('div', {class:'banner'}, 'Similarity to day 1 is approximated here (product of the step similarities) because this JSON was written by an older version. Re-run proposition_tracker.py for exact values.') : null,
    h('h1', {}, r.title), h('p', {class:'summary'}, r.summary),
    h('div', {class:'chips'}, flagList(r).map(f => h('span', {class:'tag'}, f))),
    kpis(r), holders.controls, holders.chart,
    h('div', {class:'stepper'}, h('button', {class:'btn', onclick:() => { stopPlay(); setStep(state.step - 1); }}, '◀'), slider,
      h('button', {class:'btn', onclick:() => { stopPlay(); setStep(state.step + 1); }}, '▶'), play, label),
    holders.detail, holders.matrix,
    h('div', {class:'card'}, h('h3', {}, 'Transformation summary'), h('ul', {}, r.notes.map(t => h('li', {}, t)))));
  refresh();
}
function kpis(r){
  const a = r.steps[0], b = lastStep(r), d = drops(r);
  const k = (t, v, col) => h('div', {class:'kpi'}, h('span', {class:'muted small'}, t), h('b', {style:col ? `color:${col}` : ''}, v));
  return h('div', {class:'kpis'}, k('Similarity to day 1', `${fmt(a.sim_to_start)} → ${fmt(b.sim_to_start)}`, simColor(b.sim_to_start)),
    k('Span', `${b.days_since_start} days`), k('Documents', r.steps.length),
    k('Largest step drop', d.a ? `${d.d.toFixed(2)}  (${d.a} → ${d.b})` : 'none'));
}
function setStep(i){ state.step = Math.max(0, Math.min(refs.r.steps.length - 1, i)); refs.slider.value = state.step; refresh(); }
function togglePlay(){
  if (state.timer) return stopPlay();
  if (state.step >= refs.r.steps.length - 1) setStep(0);
  refs.play.textContent = '⏸ Pause';
  state.timer = setInterval(() => { if (state.step >= refs.r.steps.length - 1) return stopPlay(); setStep(state.step + 1); }, 1400);
}
function stopPlay(){ if (state.timer) { clearInterval(state.timer); state.timer = null; } if (refs && refs.play) refs.play.textContent = '▶ Play'; }

function refresh(){
  const r = refs.r, hs = refs.holders, i = state.step, s = r.steps[i];
  refs.label.textContent = `step ${i + 1} of ${r.steps.length} · ${s.date}`;
  setKids(hs.controls, 
    seg('x-axis', [['steps', 'Documents'], ['time', 'True time']], state.axis, v => { state.axis = v; refresh(); }),
    seg('Colour vs', [['prev', 'previous step'], ['start', 'day 1']], state.ref, v => { state.ref = v; refresh(); }),
    h('div', {class:'seg'}, h('span', {class:'muted'}, 'Extra lines (cos to day 1)'),
      ['actor', 'action', 'object'].map(k => { const cb = h('input', {type:'checkbox', onchange:e => { state.extra[k] = e.target.checked; refresh(); }}); cb.checked = state.extra[k];
        return h('label', {}, cb, ' ', k, ' '); })));
  setKids(hs.chart, h('h3', {}, 'Similarity to day 1 over time'), narrativeChart(r));
  setKids(hs.detail, detail(r));
  setKids(hs.matrix, h('h3', {}, 'All elements over time'), matrixTable(r));
}

function narrativeChart(r){
  const steps = r.steps, n = steps.length, EXTRA = {actor:'#2b6cb0', action:'#d9534f', object:'#2a9d8f'};
  const xOf = (st, k) => state.axis === 'steps' ? k : st.days_since_start;
  const lines = [];
  for (const key of ['actor', 'action', 'object']) if (state.extra[key])
    lines.push({id:key, color:EXTRA[key], width:1.5, dashed:true, points:false,
      pts:steps.map((st, k) => ({x:xOf(st, k), y:(st.cos_to_start || {})[key]})).filter(p => p.y !== null && p.y !== undefined)});
  lines.push({id:'sim', color:'var(--text)', width:2.5, points:true,
    pts:steps.map((st, k) => ({x:xOf(st, k), y:st.sim_to_start, k, st, color:simColor(st.sim_to_start), r:7}))});
  let ticks;
  if (state.axis === 'steps') ticks = steps.map((st, k) => ({x:k, label:st.date}));
  else { const mx = Math.max(1, steps[n - 1].days_since_start), step = Math.max(1, Math.ceil(mx / 6));
    ticks = []; for (let d = 0; d <= mx; d += step) ticks.push({x:d, label:`+${d} d`}); }
  const c = drawChart({lines, ticks, rotate:state.axis === 'steps', bottom:118, height:370,
    selX:xOf(steps[state.step], state.step),
    onHover:(p, e) => p ? showTip([h('b', {}, p.st.date), h('div', {}, p.st.title), h('div', {}, `similarity to day 1: ${fmt(p.st.sim_to_start)}`), h('div', {class:'muted'}, p.st.claim)], e) : hideTip(),
    onClick:p => { stopPlay(); setStep(p.k); }});
  const yO = c.H - c.m.b + 70, yE = yO + 22;
  c.svg.append(S('text', {x:c.m.l - 6, y:yO + 10, 'text-anchor':'end', class:'tick'}, 'orientation'),
               S('text', {x:c.m.l - 6, y:yE + 10, 'text-anchor':'end', class:'tick'}, 'emotion'));
  steps.forEach((st, k) => { const x = c.X(xOf(st, k));
    [[yO, st.orientation, ORI], [yE, st.emotion, EMO]].forEach(([y, v, map]) => {
      const sq = S('rect', {x:x - 8, y, width:16, height:14, rx:3, fill:map[v] || '#9a9a94', stroke:k === state.step ? 'var(--text)' : 'none', 'stroke-width':2, class:'pt'});
      sq.addEventListener('mouseenter', e => showTip([h('b', {}, cap(v)), h('div', {class:'muted'}, st.date)], e));
      sq.addEventListener('mouseleave', hideTip);
      sq.addEventListener('click', () => { stopPlay(); setStep(k); });
      c.svg.append(sq); }); });
  return h('div', {}, c.svg, h('div', {class:'muted small'}, `Dashed lines: ${CFG.same_thr} (unchanged) and ${CFG.drift_thr} (drifted). Dot colour = similarity band. Squares: stance toward the object (top) and emotion (bottom). Click a point to inspect that step.`));
}

const chip = (text, status) => h('span', {class:'chip', style:`border-color:${C[status] || C.na}`}, text);
const elText = (s, key) => (key === 'orientation' || key === 'emotion') ? cap(s[key]) : s[key];

function detail(r){
  const i = state.step, s = r.steps[i], s0 = r.steps[0], prev = i ? r.steps[i - 1] : null, t = i ? r.transitions[i - 1] : null;
  const rows = ['actor', 'action', 'object', 'orientation', 'emotion'].map(key => {
    const cv = (key === 'orientation' || key === 'emotion') ? null : key;
    return h('tr', {}, h('td', {class:'muted'}, cap(key)), h('td', {}, chip(elText(s0, key), 'start')),
      h('td', {}, prev ? chip(elText(prev, key), 'start') : '–'), h('td', {}, chip(elText(s, key), statusOf(r, i, key, state.ref))),
      h('td', {class:'small'}, t ? (cv ? `cos ${fmt(t.cos[cv])}` : t.status[key]) : '–'),
      h('td', {class:'small'}, i ? (cv ? `cos ${fmt((s.cos_to_start || {})[cv])}` : s[key + '_vs_start']) : '–')); });
  const out = [
    h('h3', {}, `Step ${i + 1}: ${s.date}`),
    h('div', {class:'muted'}, `${s.title} · paragraph ${s.paragraphs} · ${s.n_mentions} mention(s)` + (t ? ` · ${t.label} (S = ${fmt(t.sim)}, +${t.days} d)` : '')),
    h('p', {class:'claim'}, `“${s.claim}”`),
    s.evidence ? h('blockquote', {}, s.evidence) : null,
    h('table', {}, h('tr', {}, ['Element', 'Day 1', 'Previous', 'Current', 'vs previous', 'vs day 1'].map(x => h('th', {}, x))), rows),
    h('div', {class:'muted small', style:'margin-top:6px'}, `Similarity to day 1: ${fmt(s.sim_to_start)}`)];
  if (s.mention_trajectory && s.mention_trajectory.length > 1)
    out.push(h('h3', {style:'margin-top:14px'}, 'Within this document (paragraph by paragraph)'),
      h('table', {}, h('tr', {}, ['¶', 'Actor', 'Action', 'Object', 'Orientation', 'Emotion'].map(x => h('th', {}, x))),
        s.mention_trajectory.map(m => h('tr', {}, h('td', {}, m.paragraph), h('td', {}, m.actor), h('td', {}, m.action), h('td', {}, m.object), h('td', {}, cap(m.orientation)), h('td', {}, cap(m.emotion))))));
  return out;
}

function matrixTable(r){
  const rows = [['claim', 'Claim'], ['actor', 'Actor'], ['action', 'Action'], ['object', 'Object'], ['orientation', 'Orientation'], ['emotion', 'Emotion'], ['sim', state.ref === 'prev' ? 'Similarity to previous' : 'Similarity to day 1']];
  const head = h('tr', {}, h('th', {}, ''), r.steps.map((s, k) => h('th', {class:k === state.step ? 'sel' : '', onclick:() => { stopPlay(); setStep(k); }}, `${k + 1}. ${s.date}`, h('div', {class:'small muted'}, s.title))));
  const body = rows.map(([key, lab]) => h('tr', {}, h('td', {}, lab), r.steps.map((s, k) => {
    const sel = k === state.step ? 'sel' : '';
    if (key === 'claim') return h('td', {class:sel}, h('i', {}, s.claim));
    if (key === 'sim') { const v = state.ref === 'prev' ? (k ? r.transitions[k - 1].sim : null) : s.sim_to_start;
      return h('td', {class:sel}, v === null ? '–' : chip(fmt(v), simStatus(v))); }
    return h('td', {class:sel}, chip(elText(s, key), statusOf(r, k, key, state.ref))); })));
  return h('div', {class:'mwrap'}, h('table', {class:'mtab'}, head, body));
}

/* ---------- overview ---------- */
function renderOverview(){
  const lines = NAR.map((r, i) => ({id:String(i), color:PAL[i % PAL.length], width:2, points:false, endMarker:true, i,
    pts:r.steps.map((s, k) => ({x:state.oaxis === 'days' ? s.days_since_start : k, y:s.sim_to_start}))})).filter(l => vis(NAR[l.i]));
  const xs = lines.flatMap(l => l.pts.map(p => p.x)), mx = Math.max(1, ...xs), tstep = Math.max(1, Math.ceil(mx / 8));
  const ticks = []; for (let d = 0; d <= mx; d += tstep) ticks.push({x:d, label:state.oaxis === 'days' ? `+${d} d` : `${d + 1}`});
  const c = drawChart({lines, ticks, bottom:40, height:380,
    onLine:l => { stopPlay(); state.view = 'narrative'; state.ni = l.i; state.step = 0; renderAll(); },
    onLineHover:(l, e) => { c.svg.querySelectorAll('.line').forEach(p => p.setAttribute('opacity', !l || p.getAttribute('data-id') === l.id ? 1 : 0.15));
      if (l) showTip([h('b', {}, `${l.i + 1}. ${NAR[l.i].title}`), h('div', {}, `similarity to day 1: ${fmt(1)} → ${fmt(lastStep(NAR[l.i]).sim_to_start)}`), h('div', {class:'muted'}, 'click to open')], e); else hideTip(); }});
  const cols = [['#', 'n', r => NAR.indexOf(r) + 1], ['Narrative', 'title', r => r.title.toLowerCase()], ['Start', 'start', r => r.steps[0].date],
    ['End', 'end', r => lastStep(r).date], ['Docs', 'docs', r => r.steps.length], ['Days', 'days', r => lastStep(r).days_since_start],
    ['Similarity to day 1 (first → last)', 'sim', r => lastStep(r).sim_to_start], ['Min', 'min', r => Math.min(...r.steps.map(s => s.sim_to_start))],
    ['Largest step drop', 'drop', r => drops(r).d], ['Flags', 'flags', r => flagList(r).length]];
  const sc = cols.find(x => x[1] === state.sort.key) || cols[6];
  const sorted = NAR.filter(vis).sort((a, b) => { const x = sc[2](a), y = sc[2](b); return (x < y ? -1 : (x > y ? 1 : 0)) * state.sort.dir; });
  const th = cols.map(([t, k]) => h('th', {class:'sortable', onclick:() => { state.sort = {key:k, dir:state.sort.key === k ? -state.sort.dir : 1}; renderOverview(); }}, t + (state.sort.key === k ? (state.sort.dir > 0 ? ' ▲' : ' ▼') : '')));
  const tr = sorted.map(r => { const i = NAR.indexOf(r), b = lastStep(r), d = drops(r);
    return h('tr', {class:'row', onclick:() => { state.view = 'narrative'; state.ni = i; state.step = 0; renderAll(); }},
      h('td', {}, i + 1), h('td', {}, h('span', {class:'swatch', style:`background:${PAL[i % PAL.length]}`}), r.title),
      h('td', {}, r.steps[0].date), h('td', {}, b.date), h('td', {}, r.steps.length), h('td', {}, b.days_since_start),
      h('td', {}, h('b', {style:`color:${simColor(b.sim_to_start)}`}, `${fmt(1)} → ${fmt(b.sim_to_start)}`)),
      h('td', {}, fmt(Math.min(...r.steps.map(s => s.sim_to_start)))), h('td', {}, d.a ? `${d.d.toFixed(2)} (${d.a} → ${d.b})` : '–'),
      h('td', {class:'small'}, flagList(r).join(', ') || '–')); });
  setKids($('main'), 
    (state.hideBoiler && NB) ? h('div', {class:'banner'}, `${NB} narrative(s) that look like unchanged boilerplate (e.g. newsletter footers) are hidden. Untick the box in the sidebar to show them.`) : null,
    NAR.some(r => r.sim_approx) ? h('div', {class:'banner'}, 'Some narratives come from an older JSON version: their similarity to day 1 is approximated. Re-run proposition_tracker.py for exact values.') : null,
    h('h1', {}, 'All narratives: similarity to day 1'),
    h('p', {class:'summary'}, 'Every line starts at 1.00 (the first document of that narrative) and shows how far the proposition has moved from its day-1 form. Hover a line to identify it, click to open it.'),
    h('div', {class:'controls'}, seg('x-axis', [['days', 'Days since day 1'], ['steps', 'Document index']], state.oaxis, v => { state.oaxis = v; renderOverview(); })),
    h('div', {class:'card'}, c.svg),
    h('div', {class:'card'}, h('div', {class:'mwrap'}, h('table', {}, h('tr', {}, th), tr))));
}

function renderAll(){ renderSide(); if (state.view === 'overview') renderOverview(); else renderNarrative(); }
document.addEventListener('keydown', e => {
  if (state.view !== 'narrative' || !refs || (e.target && e.target.tagName === 'SELECT')) return;
  if (e.key === 'ArrowRight') { stopPlay(); setStep(state.step + 1); }
  if (e.key === 'ArrowLeft') { stopPlay(); setStep(state.step - 1); }
});
if (NAR.length === 1) { state.view = 'narrative'; }
renderAll();
})();
</script></body></html>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="output folder of one dataset (contains narrative_XX.json)")
    p.add_argument("--same-thr", type=float, default=None)
    p.add_argument("--drift-thr", type=float, default=None)
    p.add_argument("--boiler-sim", type=float, default=BOILER_SIM,
                   help="flag a narrative as boilerplate if its similarity to day 1 never drops below this")
    p.add_argument("--boiler-claim", type=float, default=BOILER_CLAIM,
                   help="... or if the mean word overlap of consecutive claims is at least this")
    a = p.parse_args(argv)
    build(a.dir, a.same_thr, a.drift_thr, a.boiler_sim, a.boiler_claim)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
