#!/usr/bin/env python3
"""Build the human-readable report of a run (stdlib only; runs on the host or in the container).

    python3 multiframe_experiment/report.py <run-dir>            # rebuild results.json, report.html, report.md
    python3 multiframe_experiment/report.py --root <runs-root>    # rebuild every run and the runs index

Inputs: samples.json, config.json, progress.json, raw/answers-*.jsonl, raw/judge.jsonl, optional labels/<sample>.json.
"""
import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import infer  # noqa: E402
import prompts  # noqa: E402
import schema  # noqa: E402
import scoring  # noqa: E402


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write(path, text):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text)
    try:
        os.chmod(temp, 0o666)
    except OSError:
        pass
    os.replace(temp, path)


def load_answers(run_dir):
    records = {}
    for path in sorted((Path(run_dir) / 'raw').glob('answers-*.jsonl')):
        records.update(infer.load_records(run_dir, path.stem))
    return records


def load_labels(run_dir):
    labels = {}
    for path in (Path(run_dir) / 'labels').glob('*.json'):
        data = _read(path)
        if isinstance(data, dict) and any(any((p.get('flags') or {}).values()) for p in data.get('pairs', [])):
            labels[path.stem] = data
    return labels


def _slim_answer_record(record, keep_prompt):
    out = {k: record.get(k) for k in ('config', 'sample', 'window', 'ok', 'first_try_ok', 'seconds', 'answer', 'error')}
    out['attempts'] = [{k: a.get(k) for k in ('n', 'seconds', 'prompt_tokens', 'completion_tokens', 'predicted_per_second', 'finish', 'grammar', 'errors')}
                       | ({'raw': (a.get('raw') or '')[:4000]} if a.get('errors') else {}) for a in record.get('attempts', [])]
    if keep_prompt:
        out['prompt'] = record.get('prompt')
    return out


def build(run_dir):
    run_dir = Path(run_dir)
    samples = _read(run_dir / 'samples.json', []) or []
    config = _read(run_dir / 'config.json', {}) or {}
    progress = _read(run_dir / 'progress.json', {}) or {}
    answers = load_answers(run_dir)
    judge_records = infer.load_records(run_dir, 'judge')
    labels = load_labels(run_dir)
    rows = scoring.build_rows(samples, answers, judge_records, labels)
    complete = progress.get('status') == 'done'
    notes = scoring.findings(rows, complete)

    from judge import claim_text_key
    verdicts = {}
    for record in judge_records.values():
        if record.get('kind') == 'verify' and record.get('ok'):
            for text_key, verdict in record['verdicts'].items():
                verdicts[f"{record['sample']}|{record['window']}|{text_key}"] = verdict
    references = {f"{r['sample']}|{r['window']}": r['pairs'] for r in judge_records.values() if r.get('kind') == 'reference' and r.get('ok')}
    prompt_seen, calls = set(), []
    for record in sorted(answers.values(), key=lambda r: (r['sample'], r['config'], r['window'])):
        keep = record['config'] not in prompt_seen
        prompt_seen.add(record['config'])
        slim = _slim_answer_record(record, keep)
        if record.get('ok'):
            sample = next(s for s in samples if s['id'] == record['sample'])
            window = sample['windows'][record['window']]
            slim['claims'] = [{'t': f"{c['from']}→{c['to']} {c['text']}", 'v': verdicts.get(f"{record['sample']}|{record['window']}|{claim_text_key(c)}")}
                              for c in schema.answer_claims(record['answer'], window, record['config'], record['sample']) if c['judge']]
        calls.append(slim)
    baselines = []
    for sample in samples:
        view = (sample.get('baseline') or {}).get('view')
        if not view or len(sample['frames']) < 2:
            continue
        for window in sample['windows']:
            answer = schema.baseline_answer(view, window)
            baselines.append({'config': infer.BASELINE_KEY, 'sample': sample['id'], 'window': window['index'], 'ok': True, 'answer': answer,
                              'claims': [{'t': f"{c['from']}→{c['to']} {c['text']}", 'v': verdicts.get(f"{sample['id']}|{window['index']}|{claim_text_key(c)}")}
                                         for c in schema.answer_claims(answer, window, infer.BASELINE_KEY, sample['id']) if c['judge']]})
    results = {
        'run_id': run_dir.name, 'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'config': config, 'progress': progress,
        'weights': scoring.WEIGHTS, 'rows': rows, 'findings': notes, 'pareto': scoring.pareto(rows),
        'styles': {k: v['label'] for k, v in prompts.STYLES.items()}, 'contexts': {k: v['label'] for k, v in prompts.CONTEXTS.items()},
        'models': {k: v['label'] for k, v in infer.MODELS.items()}, 'categories': schema.CATEGORY_LABELS, 'importance': schema.IMPORTANCE,
        'samples': [{k: s.get(k) for k in ('id', 'cut_number', 'start', 'end', 'duration', 'roster', 'transcript', 'frames', 'windows', 'pairs', 'keyframes', 'no_characters')}
                    | {'baseline': {k: (s.get('baseline') or {}).get(k) for k in ('model', 'seconds', 'contract')}} for s in samples],
        'calls': calls + baselines, 'references': references, 'labels': sorted(labels),
    }
    _write(run_dir / 'results.json', json.dumps(results, ensure_ascii=False))
    _write(run_dir / 'report.html', render_html(results))
    _write(run_dir / 'report.md', render_markdown(results))
    return results


# -- markdown (for agents) ------------------------------------------------------------------------------------------------

def _pct(value):
    return '—' if value is None else f'{value * 100:.0f}%'


def _num(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def render_markdown(results):
    cfg, prog = results['config'], results['progress']
    lines = [f"# Multi-frame experiment — {results['run_id']}", '',
             f"Status: **{prog.get('status', '?')}** · updated {results['updated_at']} · episode {cfg.get('episode')} · cuts {cfg.get('cuts')} · "
             f"window {cfg.get('window_size')} · image width {cfg.get('image_width')} · judge {'on' if cfg.get('judge', True) else 'off'}", '',
             '## Findings', '', *[f'- {line}' for line in results['findings']], '',
             '## Leaderboard', '',
             '| Config | Score | Precision | Recall | Camera | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in sorted(results['rows'].values(), key=lambda r: -(r['score'] or -1)):
        c = row['components']
        lines.append(f"| {row['config']} | {_num(row['score'])} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['camera'])} | {_pct(c['speaking'])}"
                     f"{' (told)' if row['speaking']['told'] else ''} | {_pct(c['reliability'])} | {_pct(row['hallucination_rate'])} | "
                     f"{row['names']['invented']}/{row['names']['entries']} | {row['pixel']['claims_on_still']} in {row['pixel']['still_pairs']} | "
                     f"{_num(row['cost']['per_window'])} | {_num(row['cost']['per_footage_second'], 2)} | {row['cost']['prompt_tokens'] or '—'}/{row['cost']['completion_tokens'] or '—'} |")
    lines += ['', '## Precision / recall by category', '', '| Config | ' + ' | '.join(results['categories'][c] for c in schema.REFERENCE_FLAGS) + ' |',
              '|---|' + '---|' * len(schema.REFERENCE_FLAGS)]
    for row in sorted(results['rows'].values(), key=lambda r: r['config']):
        cells = [f"{_pct(row['categories'][c]['precision'])} / {_pct(row['categories'][c]['recall'])}" for c in schema.REFERENCE_FLAGS]
        lines.append(f"| {row['config']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## How to read', '',
              f"Score weights: {', '.join(f'{k} {v}' for k, v in results['weights'].items())} (missing components are left out and the rest renormalised).",
              'Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is checked against an ORB/RANSAC pixel estimate, '
              'speaking against the transcript (the `full` context is given the transcript, marked "told"). No hand labels unless `labels/*.json` exist.', '']
    return '\n'.join(lines)


# -- html -----------------------------------------------------------------------------------------------------------------

def render_html(results):
    data = json.dumps(results, ensure_ascii=False).replace('</', '<\\/')
    refresh = '<meta http-equiv="refresh" content="30">' if results['progress'].get('status') in ('running', 'starting') else ''
    return TEMPLATE.replace('/*REFRESH*/', refresh).replace('/*TITLE*/', html.escape(f"Multi-frame experiment · {results['run_id']}")).replace('/*DATA*/', data)


def build_index(root):
    root = Path(root)
    runs = []
    for path in root.glob('*/progress.json'):
        p = _read(path, {}) or {}
        runs.append({'id': path.parent.name, 'status': p.get('status'), 'started_at': p.get('started_at'), 'updated_at': p.get('updated_at'), 'dry': p.get('dry')})
    runs.sort(key=lambda r: r.get('started_at') or '', reverse=True)
    items = ''.join(f'<li><a href="{html.escape(r["id"])}/report.html">{html.escape(r["id"])}</a> — {html.escape(str(r["status"]))}'
                    f'{" (dry run)" if r["dry"] else ""} · started {html.escape(str(r["started_at"]))}</li>' for r in runs)
    _write(root / 'index.html', f'<!doctype html><meta charset="utf-8"><title>Multi-frame runs</title>'
                                f'<style>body{{font:15px system-ui;margin:24px;background:#fff;color:#1d2330}}@media(prefers-color-scheme:dark){{body{{background:#14171f;color:#e4e6ec}}a{{color:#9bb8ff}}}}</style>'
                                f'<h1>Multi-frame experiment runs</h1><ul>{items}</ul>')
    _write(root / 'index.json', json.dumps(runs))


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">/*REFRESH*/
<title>/*TITLE*/</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#5d6676;--line:#dde1e8;--accent:#4b5fd6;--good:#1f8a4c;--bad:#c23b3b;--warn:#b7791f;--soft:#eef0f6}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--panel:#1a1e28;--ink:#e4e6ec;--muted:#98a1b2;--line:#2c3242;--accent:#8ea0ff;--good:#4cc287;--bad:#ef6f6f;--warn:#e0b050;--soft:#232838}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1500px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}h3{font-size:14px;margin:14px 0 6px}
.muted{color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{background:var(--soft);border-radius:999px;padding:2px 10px;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{cursor:pointer;position:sticky;top:0;background:var(--panel);font-weight:600}
.scroll{overflow-x:auto}.best{font-weight:700;color:var(--good)}.bad{color:var(--bad)}.good{color:var(--good)}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--accent);vertical-align:middle;margin-left:6px}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}.strip figure{margin:0;flex:0 0 auto;width:220px}.strip img{width:220px;border-radius:6px;display:block;cursor:zoom-in}
.strip figcaption{font-size:12px;color:var(--muted)}.pair{font-size:12px;border-left:3px solid var(--line);padding-left:8px;margin:4px 0}
.seg{border:1px solid var(--line);border-radius:8px;padding:8px;margin:6px 0}.seg h4{margin:0 0 4px;font-size:13px}
.claim{display:block;padding:1px 0}.v-true{color:var(--good)}.v-false{color:var(--bad);text-decoration:line-through}.v-not_visible{color:var(--warn)}.v-none{color:var(--muted)}
select,button{font:inherit;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}@media (max-width:600px){.grid{grid-template-columns:1fr}.strip figure,.strip img{width:160px}}
pre{white-space:pre-wrap;font-size:12px;background:var(--soft);padding:8px;border-radius:6px;max-height:420px;overflow:auto}
details summary{cursor:pointer;color:var(--accent)}.heat{border-radius:4px;padding:2px 4px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}#zoom img{max-width:96vw;max-height:92vh}
</style></head><body><main>
<div id="app"></div><div id="zoom" onclick="this.style.display='none'"><img alt=""></div>
<script>
const R = /*DATA*/;
const $ = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = (v) => v == null ? '—' : Math.round(v * 100) + '%';
const num = (v, d = 1) => v == null ? '—' : Number(v).toFixed(d);
const label = (cfg) => { const [m, s, c] = cfg.split('|'); return m === 'baseline' ? 'Stored Qwen3-VL-8B video analysis (production)' : `${R.models[m] || m} · ${s} · ${c}`; };
const rows = Object.values(R.rows);
const app = document.getElementById('app');
function header() {
  const c = R.config, p = R.progress;
  const stages = Object.entries(p.stages || {}).map(([k, v]) => `<span class="chip">${esc(k)}: ${esc(v.status)} ${v.total ? v.done + '/' + v.total : ''}</span>`).join('');
  return `<h1>Multi-frame change experiment</h1><div class="muted">Run <b>${esc(R.run_id)}</b> · ${esc(p.status)}${p.dry ? ' · DRY RUN (fake models, numbers meaningless)' : ''} · started ${esc(p.started_at)} · updated ${esc(R.updated_at)}${p.elapsed_seconds ? ' · elapsed ' + Math.round(p.elapsed_seconds / 60) + ' min' : ''}</div>
  <div class="panel"><div class="chips">${stages}</div><p class="muted" style="margin:8px 0 0">Episode ${esc(c.episode)} · cuts ${esc(c.cuts)} (${(R.samples || []).length} cuts, ${c.windows ?? '?'} windows) · window ${esc(c.window_size)} frames · image width ${esc(c.image_width)} px · models ${esc((c.models || []).join(', '))} · styles ${esc((c.styles || []).join(', '))} · contexts ${esc((c.contexts || []).join(', '))} · judge ${c.judge === false ? 'off' : 'Qwen3.5-27B'} · grammar ${c.grammar === false ? 'off' : 'json_schema'}</p></div>`;
}
function findings() { return `<h2>Findings</h2><div class="panel"><ul>${R.findings.map(f => `<li>${esc(f)}</li>`).join('')}</ul><p class="muted">Deterministic summary of the numbers below; no model wrote it. Weights: ${Object.entries(R.weights).map(([k, v]) => k + ' ' + v).join(', ')}.</p></div>`; }
const COLS = [
  ['Score', r => r.score, v => num(v)], ['Precision', r => r.components.precision, pct], ['Recall', r => r.components.recall, pct],
  ['Camera', r => r.components.camera, pct], ['Speaking', r => r.components.speaking, pct], ['Reliability', r => r.components.reliability, pct],
  ['Efficiency', r => r.components.efficiency, pct], ['False claims', r => r.hallucination_rate, pct, true], ['Not visible', r => r.unverifiable_rate, pct, true],
  ['Invented names', r => r.names.entries ? r.names.invented / r.names.entries : null, pct, true], ['Claims on still pairs', r => r.pixel.claims_on_still, v => v ?? '—', true],
  ['Claims judged', r => r.claims.true + r.claims.false + r.claims.not_visible, v => v],
  ['Valid / windows', r => r.windows ? r.valid / r.windows : null, (v, r) => `${r.valid}/${r.windows}`],
  ['s / window', r => r.cost.per_window, v => num(v), true], ['s per footage s', r => r.cost.per_footage_second, v => num(v, 2), true],
  ['Tokens in/out', r => r.cost.prompt_tokens, (v, r) => `${r.cost.prompt_tokens ?? '—'}/${r.cost.completion_tokens ?? '—'}`], ['Decode tok/s', r => r.cost.decode_tps, v => num(v)],
];
let sortCol = 0, sortDir = -1;
function leaderboard() {
  const sorted = [...rows].sort((a, b) => { const x = COLS[sortCol][1](a), y = COLS[sortCol][1](b); return ((x ?? -1e9) - (y ?? -1e9)) * sortDir; });
  const best = COLS.map(([, get, , low]) => { const vals = rows.filter(r => !r.baseline).map(get).filter(v => v != null); return vals.length ? (low ? Math.min(...vals) : Math.max(...vals)) : null; });
  const body = sorted.map(r => `<tr><td>${esc(label(r.config))}${R.pareto.includes(r.config) ? ' <span class="chip" title="cost/quality frontier">frontier</span>' : ''}${r.speaking.told ? ' <span class="chip" title="given the transcript">told speech</span>' : ''}</td>${COLS.map(([, get, fmt], i) => { const v = get(r); return `<td class="${v != null && v === best[i] && !r.baseline ? 'best' : ''}">${fmt(v, r)}${i === 0 && v != null ? `<span class="bar" style="width:${Math.round(v * .6)}px"></span>` : ''}</td>`; }).join('')}</tr>`).join('');
  return `<h2>Leaderboard</h2><div class="panel scroll"><table id="lb"><thead><tr><th>Config</th>${COLS.map((c, i) => `<th data-i="${i}">${c[0]}${i === sortCol ? (sortDir < 0 ? ' ▾' : ' ▴') : ''}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>
  <p class="muted">Precision: judge-verified claims true ÷ (true + false), outfit/colour/lighting weighted 0.3. Recall: the judge's reference changes per pair and category that the answer also reports. Camera: agreement with an ORB/RANSAC pixel estimate. Speaking: speaking=yes against transcript speech in the pair's interval. Efficiency: cheapest compute per footage second ÷ this one. Claims on still pairs: claims made where the pixels barely changed (lower is better). Green = best among the candidates.</p></div>`;
}
function heat(v) { if (v == null) return ''; const h = Math.round(v * 120); return `background:hsla(${h},60%,45%,.25)`; }
function categories() {
  const cats = ['walking', 'speaking', 'blink', 'head', 'eyes_gaze', 'arms_hands', 'legs', 'body_pose', 'position', 'expression', 'outfit', 'color', 'lighting', 'object', 'background', 'effect'];
  const body = [...rows].sort((a, b) => a.config.localeCompare(b.config)).map(r => `<tr><td>${esc(label(r.config))}</td>${cats.map(c => { const v = r.categories[c]; return `<td><span class="heat" style="${heat(v.precision)}" title="precision: ${v.true} true, ${v.false} false, ${v.not_visible} not visible">${pct(v.precision)}</span> / <span class="heat" style="${heat(v.recall)}" title="recall: ${v.hit} of ${v.ref}">${pct(v.recall)}</span></td>`; }).join('')}</tr>`).join('');
  return `<h2>By category (precision / recall)</h2><div class="panel scroll"><table><thead><tr><th>Config</th>${cats.map(c => `<th>${esc(R.categories[c])}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table><p class="muted">Hover a cell for the counts. Recall exists only for the reference categories (the judge does not annotate outfit, colour, lighting, objects).</p></div>`;
}
const configs = [...new Set(R.calls.map(c => c.config))].sort();
let cutIndex = 0, shown = new Set(configs.filter(c => !c.startsWith('baseline')).slice(0, 2));
function claimLines(call, sample, w) {
  if (!call) return '<div class="muted">not run yet</div>';
  if (!call.ok) return `<div class="bad">invalid: ${esc(call.error)}</div>`;
  const states = (call.answer.frames || []).map(f => `<div class="pair"><b>${esc(f.frame)}</b> ${f.characters.map(c => `${esc(c.name)}: ${esc(c.state)}`).join(' · ') || '<span class="muted">nobody</span>'}</div>`).join('');
  return (states ? `<details><summary>Per-frame states</summary>${states}</details>` : '') + call.answer.segments.map((seg, si) => {
    const pair = (sample.pairs || []).find(p => p.from === seg.from && p.to === seg.to);
    const cv = pair ? pair.camera.move : '?';
    const camClass = seg.camera.move === cv ? 'good' : (cv === 'unclear' || seg.camera.move === 'unclear' ? 'muted' : 'bad');
    const people = seg.characters.map(p => {
      const flags = ['walking', 'speaking', 'blinking'].filter(f => p[f] === 'yes').map(f => `<span class="chip">${f}</span>`).join(' ');
      const ch = p.changes.map(c => `<span class="claim">• ${esc(c.part)}: ${esc(c.description)}</span>`).join('');
      return `<div><b>${esc(p.name)}</b> ${flags}${ch}</div>`;
    }).join('') || '<div class="muted">no characters</div>';
    const others = seg.other_changes.map(o => `<span class="claim">◦ ${esc(o.kind)}: ${esc(o.description)}</span>`).join('');
    return `<div class="seg"><h4>${seg.from}→${seg.to} <span class="muted">${num(seg.start, 2)}–${num(seg.end, 2)} s</span> · camera <span class="${camClass}">${esc(seg.camera.move)}</span> <span class="muted">(pixels: ${esc(cv)})</span></h4>${people}${others}</div>`;
  }).join('') + verdictList(call);
}
function verdictList(call) {
  const v = call.claims || []; if (!v.length) return '';
  const counts = { true: 0, false: 0, not_visible: 0, none: 0 }; v.forEach(x => counts[x.v || 'none']++);
  return `<div class="muted" style="font-size:12px">judge: <span class="good">${counts.true} true</span> · <span class="bad">${counts.false} false</span> · <span style="color:var(--warn)">${counts.not_visible} not visible</span> · ${counts.none} not judged yet</div>`;
}
function judgedText(call) {
  if (!call || !call.ok || !(call.claims || []).length) return '';
  return `<details><summary>Claims with verdicts (${call.claims.length})</summary>${call.claims.map(c => `<span class="claim v-${c.v || 'none'}">${esc(c.t)} <small>[${c.v || '—'}]</small></span>`).join('')}</details>`;
}
function drill() {
  const s = R.samples[cutIndex]; if (!s) return '<h2>Cuts</h2><div class="panel muted">No cuts yet.</div>';
  const opts = R.samples.map((x, i) => `<option value="${i}" ${i === cutIndex ? 'selected' : ''}>Cut ${x.cut_number} · ${num(x.duration, 2)} s · ${x.frames.length} frames</option>`).join('');
  const strip = s.frames.map(f => `<figure><img loading="lazy" src="${esc(f.path)}" alt="${esc(f.label)}"><figcaption><b>${esc(f.label)}</b> t=${num(f.t, 2)} s · ${esc(f.reason || '')} · change ${num(f.change_score)}</figcaption></figure>`).join('');
  const pairs = (s.pairs || []).map(p => `<div class="pair">${p.from}→${p.to}: camera <b>${esc(p.camera.move)}</b>${p.camera.dx != null ? ` (dx ${num(p.camera.dx * 100)}%, dy ${num(p.camera.dy * 100)}%, scale ${num(p.camera.scale, 3)}, ${p.camera.inliers} inliers)` : ''} · change global ${num(p.global, 2)} regional ${num(p.regional, 2)}${p.still ? ' · <i>still</i>' : ''}${Object.keys(p.boxes || {}).length ? ' · boxes ' + Object.entries(p.boxes).map(([n, v]) => `${esc(n)} ${num(v, 2)}`).join(', ') : ''}</div>`).join('');
  const speech = (s.transcript || []).map(l => `<div class="pair">${num(l.start, 2)}–${num(l.end, 2)} s · ${esc(l.speaker || '?')}: ${esc(l.text)}</div>`).join('') || '<div class="muted">no speech</div>';
  const toggles = configs.map(c => `<label class="chip"><input type="checkbox" data-c="${esc(c)}" ${shown.has(c) ? 'checked' : ''}> ${esc(label(c))}</label>`).join('');
  const windows = s.windows.map((win, w) => {
    const ref = R.references[`${s.id}|${w}`];
    const refHtml = ref ? `<details><summary>Judge reference for this window</summary>${ref.map(p => `<div class="pair">${p.from}→${p.to}: camera ${esc(p.camera)} · ${p.categories.length ? p.categories.map(esc).join(', ') : 'no change'}${p.characters.length ? ' — ' + p.characters.map(c => esc(c.name)).join(', ') : ''}</div>`).join('')}</details>` : '';
    const cols = [...shown].map(c => { const call = R.calls.find(x => x.config === c && x.sample === s.id && x.window === w); return `<div class="panel"><h3>${esc(label(c))}${call && call.seconds ? ` <span class="muted">${num(call.seconds)} s</span>` : ''}</h3>${claimLines(call, s, w)}${judgedText(call)}${call && call.answer && call.answer.summary ? `<p class="muted">${esc(call.answer.summary)}</p>` : ''}</div>`; }).join('');
    return `<h3>Window ${w + 1}: ${win.frames.map(f => f.label).join(' ')}</h3>${refHtml}<div class="grid">${cols}</div>`;
  }).join('');
  return `<h2>Cuts</h2><div class="panel"><select id="cut">${opts}</select> <span class="muted">roster: ${s.roster.length ? s.roster.map(p => esc(p.name) + ' (' + p.side + ')').join(', ') : (s.no_characters ? 'no characters (user)' : 'nobody identified')} · keyframes ${esc(s.keyframes.origin)} · change level ${esc(s.keyframes.change_level)} · baseline ${s.baseline.seconds ? num(s.baseline.seconds) + ' s' : '—'}</span>
  <div class="strip" style="margin-top:10px">${strip}</div><details><summary>Pixel evidence per pair</summary>${pairs}</details><details><summary>Transcript</summary>${speech}</details>
  <div class="chips" style="margin-top:10px">${toggles}</div></div>${windows}`;
}
function promptsSection() {
  const firsts = configs.map(c => R.calls.find(x => x.config === c && x.prompt)).filter(Boolean);
  return `<h2>Prompts</h2><div class="panel">${firsts.map(c => `<details><summary>${esc(label(c.config))}</summary><pre>${esc(c.prompt)}</pre></details>`).join('') || '<span class="muted">none yet</span>'}</div>
  <h2>Method and caveats</h2><div class="panel muted"><ul>
  <li>Frames: the keyframe selector's choice per cut (initial frame plus change peaks, at most 5 per started 5 s and 15 per cut). Windows of ${esc(R.config.window_size)} frames; each window starts on the previous one's last frame. One model call per window and config.</li>
  <li>Styles: ${Object.entries(R.styles).map(([k, v]) => `<b>${esc(k)}</b> = ${esc(v)}`).join('; ')}. Contexts: ${Object.entries(R.contexts).map(([k, v]) => `<b>${esc(k)}</b> = ${esc(v)}`).join('; ')}.</li>
  <li>The judge (Qwen3.5-27B, thinking off) is blind to which config wrote a claim, but it is the same model as one candidate: Qwen answers may read more naturally to it. Its reference table is a second opinion, not ground truth.</li>
  <li>The camera check uses the dominant image transform; a character filling the frame can mislead it, and animated background pans on a static camera count as pans.</li>
  <li>The stored video analysis (baseline) is mapped onto the pairs by its event times; its camera is one value for the whole cut, its cost is the production stage time.</li>
  <li>Ground truth: put <code>labels/&lt;cut&gt;.json</code> files (see README) into the run folder and rebuild the report to add true precision/recall.</li></ul></div>`;
}
function render() {
  app.replaceChildren($(header() + findings() + leaderboard() + categories() + drill() + promptsSection()));
  document.querySelectorAll('#lb th[data-i]').forEach(th => th.onclick = () => { const i = +th.dataset.i; sortDir = i === sortCol ? -sortDir : -1; sortCol = i; render(); });
  const cut = document.getElementById('cut'); if (cut) cut.onchange = () => { cutIndex = +cut.value; render(); };
  document.querySelectorAll('input[data-c]').forEach(box => box.onchange = () => { box.checked ? shown.add(box.dataset.c) : shown.delete(box.dataset.c); render(); });
  document.querySelectorAll('.strip img').forEach(img => img.onclick = () => { const z = document.getElementById('zoom'); z.querySelector('img').src = img.src; z.style.display = 'flex'; });
}
render();
</script></main></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', nargs='?')
    parser.add_argument('--root', help='rebuild every run under this folder and the index')
    args = parser.parse_args(argv)
    if args.root:
        for path in sorted(Path(args.root).glob('*/samples.json')):
            build(path.parent)
            print('built', path.parent / 'report.html')
        build_index(args.root)
        return 0
    if not args.run_dir:
        parser.error('give a run directory or --root')
    build(args.run_dir)
    build_index(Path(args.run_dir).parent)
    print('built', Path(args.run_dir) / 'report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
