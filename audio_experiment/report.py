"""report.html / report.md / spotcheck.html from results.json (stdlib only). `python3 report.py <run dir>`; also runs the analysis first."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
import common  # noqa: E402

CSS = """:root{--bg:#fff;--fg:#1d2330;--mut:#667;--line:#dde1ea;--acc:#2753d6;--ok:#17803d;--bad:#b3261e;--card:#f6f7fb}
@media(prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e4e6ec;--mut:#9aa1b2;--line:#2b303d;--acc:#9bb8ff;--ok:#5dd28a;--bad:#ff8a80;--card:#1b1f2a}}
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}main{max-width:1280px;margin:0 auto;padding:20px 16px 80px}
h1{font-size:22px}h2{font-size:18px;margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:15px;margin-top:22px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}th,td{border:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top}th{background:var(--card)}
td.n{text-align:right;font-variant-numeric:tabular-nums}.verdict{background:var(--card);border-left:4px solid var(--acc);padding:10px 14px;margin:12px 0}
.mut{color:var(--mut)}.sample{border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0;background:var(--card)}.sample b{color:var(--acc)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin:0 3px 0 0;font-size:12px}audio{height:30px;vertical-align:middle}
.wrap{overflow-x:auto}code{font-size:12px}"""


def pct(v, digits=0):
    return '–' if v is None else f'{v * 100:.{digits}f}%'


def num(v, digits=1):
    return '–' if v is None else f'{v:.{digits}f}'


def clock(t):
    return f'{int(t // 60)}:{t % 60:04.1f}'


class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, level, text):
        self.blocks.append(('h', level, text))

    def p(self, text, cls=None):
        self.blocks.append(('p', text, cls))

    def table(self, header, rows, numeric_from=1):
        self.blocks.append(('t', header, rows, numeric_from))

    def raw(self, html_text, md_text=''):
        self.blocks.append(('r', html_text, md_text))

    def html(self, title, refresh=False):
        out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>',
               '<meta http-equiv="refresh" content="60">' if refresh else '', f'<style>{CSS}</style></head><body><main>']
        for b in self.blocks:
            if b[0] == 'h':
                out.append(f'<h{b[1]}>{html.escape(b[2])}</h{b[1]}>')
            elif b[0] == 'p':
                out.append(f'<p class="{b[2] or ""}">{html.escape(b[1])}</p>' if b[2] != 'verdict' else f'<div class="verdict">{html.escape(b[1])}</div>')
            elif b[0] == 't':
                _, header, rows, start = b
                out.append('<div class="wrap"><table><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in header) + '</tr>')
                for row in rows:
                    out.append('<tr>' + ''.join(f'<td class="{"n" if i >= start and _is_num(c) else ""}">{html.escape(str(c))}</td>' for i, c in enumerate(row)) + '</tr>')
                out.append('</table></div>')
            else:
                out.append(b[1])
        out.append('</main></body></html>')
        return '\n'.join(out)

    def md(self):
        out = []
        for b in self.blocks:
            if b[0] == 'h':
                out.append('#' * b[1] + ' ' + b[2] + '\n')
            elif b[0] == 'p':
                out.append(('> ' if b[2] == 'verdict' else '') + b[1] + '\n')
            elif b[0] == 't':
                _, header, rows, _ = b
                out.append('| ' + ' | '.join(str(c) for c in header) + ' |\n|' + '---|' * len(header))
                out.extend('| ' + ' | '.join(str(c).replace('|', '/').replace('\n', ' ') for c in row) + ' |' for row in rows)
                out.append('')
            elif b[2]:
                out.append(b[2] + '\n')
        return '\n'.join(out)


def _is_num(value):
    s = str(value).replace('%', '').replace('–', '').replace('.', '', 1).replace('-', '', 1).replace(':', '').strip()
    return s.isdigit()


# ------------------------------------------------------------------ sections

def verdict(r):
    rows = [x for x in r['ranking'] if x['anchor_n'] and x['anchor_strict'] is not None]
    if not rows:
        return 'The run has no anchor lines scored yet; see the tables below for what finished.'
    loo = [x for x in rows if x.get('loo')]
    best = (loo or rows)[0]
    n = best['anchor_n']
    by = {x['method']: x for x in rows}
    parts = [f"Best speaker method on the anchor lines (voice-sample ranges and hand-corrected rows of Episodes 1 and 2, references left out where they overlap the line): {best['method']} with {pct(best['anchor_strict'])} of {n} lines right"
             f" ({pct(best['anchor_answered'])} of those it answered, coverage {pct(best['anchor_coverage'])})."]
    for key, label in (('V3', 'voice alone'), ('Q1', 'Qwen with everything'), ('R1', 'Qwen + episode-level voice resolution'), ('Q2', 'Qwen without the Portuguese subtitles')):
        if key in by and by[key]['anchor_strict'] is not None:
            parts.append(f"{label} ({key}): {pct(by[key]['anchor_strict'])}.")
    b0 = by.get('B0')
    if b0 and b0['anchor_strict'] is not None:
        parts.append(f"The stored transcript scores {pct(b0['anchor_strict'])}, but its links come from the very samples the anchors are cut from, so that number is optimistic.")
    if 'Q1' in by and 'Q2' in by and by['Q1']['anchor_strict'] is not None and by['Q2']['anchor_strict'] is not None:
        d = by['Q1']['anchor_strict'] - by['Q2']['anchor_strict']
        parts.append('The Portuguese subtitles ' + ('helped' if d > .01 else 'misled' if d < -.01 else 'made no measurable difference') + f' ({d * 100:+.1f} points Q1 vs Q2).')
    parts.append(f'Confidence: {"low" if n < 60 else "moderate" if n < 150 else "reasonable"}, because there are only {n} anchor lines; read the ranking as an ordering of approaches, not as final accuracy.')
    t = r.get('recommended_time') or {}
    if t.get('total_seconds'):
        parts.append(f"Estimated time for the recommended pipeline on a 25-minute episode: {t['total_seconds'] / 60:.0f} min" + (f" against {t['baseline_seconds'] / 60:.0f} min for the current pipeline re-run." if t.get('baseline_seconds') else '.'))
    return ' '.join(parts)


def ranking_table(r):
    rows = []
    for x in r['ranking']:
        rows.append([x['method'], 'yes' if x.get('loo') else 'no' if x.get('loo') is False else '–', x['anchor_n'], pct(x['anchor_strict']), pct(x['anchor_answered']), pct(x['anchor_coverage']),
                     pct(x['contradiction_rate']) + f" ({x['contradictions']})", pct(x['agreement']), pct(x['stability'])])
    return ['method', 'refs left out', 'anchor n', 'anchor right (strict)', 'right when answered', 'coverage', 'addressed-name contradictions', 'agreement (V3<->Q1)', 'stability Q1 vs shifted'], rows


def build(results):
    d = Doc()
    d.h(1, f"Audio experiment · {results['run_id']}")
    d.p(f"Generated {results['generated_at']}. " + ('Dry run with fake models: numbers are not measurements. ' if results.get('excerpt') else '') +
        f"Progress: {results['progress'].get('status')}, {round((results['progress'].get('elapsed_seconds') or 0) / 60)} min elapsed.", 'mut')
    d.h(2, 'Verdict')
    d.p(verdict(results), 'verdict')
    d.p('Chapter breakdown context item of Episode 1 ("Minute by minute breakdown"): ' + ('available but NOT used by any method, so Episode 1 is as hard as the others.' if results.get('breakdown_available', {}).get('1') else 'not available.'), 'mut')

    d.h(2, '1 · Speaker methods, ranked')
    d.p('Ranked by anchor accuracy first, then addressed-name contradictions, then agreement and stability. No weighted composite. "refs left out" means a line is scored with every voice-sample range that overlaps it removed.', 'mut')
    header, rows = ranking_table(results)
    d.table(header, rows)
    for n, ep in sorted(results['episodes'].items()):
        d.h(3, f"Episode {n}: methods (anchors {ep['anchors_n']})")
        rows = []
        for name, m in ep['methods'].items():
            a = m['anchors']
            dist = m['distribution']
            rows.append([name, f"{a['all']['correct']}/{a['all']['n']}", pct(a['all']['strict']), pct(a['sample']['strict']) + f" (n={a['sample']['n']})", pct(a['manual']['strict']) + f" (n={a['manual']['n']})",
                         f"{m['contradictions']['contradictions']}/{m['contradictions']['addressed_lines']}", num(m['pronouns']['weighted_entropy'], 2), dist['voices'], pct(dist['unknown_share']),
                         pct(dist['low_share']), pct(dist['on_screen_share'])])
        d.table(['method', 'anchors right', 'strict', 'voice-sample anchors', 'hand-corrected rows', 'contradictions', 'pronoun entropy (lower = steadier)', 'voices', 'unknown', 'low confidence', 'speaker on screen (off-screen is legitimate)'], rows)
        if ep.get('stability'):
            d.p(f"Stability Q1 vs half-window shift: {pct(ep['stability']['agreement'])} of {ep['stability']['both_answered']} lines; voice-only vs Qwen agreement: {pct((ep.get('voice_vs_qwen') or {}).get('agreement'))}.", 'mut')

    d.h(2, '2 · Diarization settings')
    for n, ep in sorted(results['episodes'].items()):
        rows = []
        for name, e in ep['diar'].items():
            rows.append([name, e['voices'], e['mapped_app'], e['mapped_range'], e['mixed_lines'], pct(e['app']['anchors']['all']['strict']) + ' (app rule, circular)',
                         pct(e['range']['anchors']['all']['strict']) + ' (per-range, refs left out)', f"{e['range']['contradictions']['contradictions']}/{e['range']['contradictions']['addressed_lines']}",
                         num(e['seconds'], 0)])
        d.h(3, f'Episode {n}')
        d.table(['setting', 'voices found', 'clusters linked (app rule)', 'clusters linked (per range)', 'lines with mixed speakers before the majority vote', 'anchor right, app rule', 'anchor right, per range', 'contradictions (per range)', 'diarization seconds'], rows)

    d.h(2, '3 · Segmentation and text')
    rows = []
    for n, ep in sorted(results['episodes'].items()):
        for name, t in ep['text'].items():
            if name == 'agreement':
                continue
            s, g = t['segmentation'], t['garbage']
            rows.append([f'Ep{n}', name, s['lines'], num(s['median_seconds'], 1), num(s['median_chars'], 0), s['short_lines'], s['unfinished'], s.get('window_cut'), s.get('speaker_switches_inside_sentence'),
                         pct((s.get('edge_start') or {}).get('precision')) + '/' + pct((s.get('edge_start') or {}).get('recall')), pct((s.get('edge_end') or {}).get('precision')) + '/' + pct((s.get('edge_end') or {}).get('recall')),
                         g['garbage_outside_mask'], pct(g['garbage_rate'], 1), g['glossary_hits'], num(t.get('asr_seconds'), 0) if name != 'B0' else (ep['methods']['B0'].get('job_seconds') or '–')])
    d.table(['ep', 'set', 'lines', 'median s', 'median chars', 'lines < 0.8 s', 'no sentence end', 'cut by window edge', 'speaker switches inside a sentence', 'start edge P/R (±0.35 s vs cues)', 'end edge P/R', 'garbage outside OP/ED', 'garbage rate', 'verified-name hits', 'ASR s (B0: whole job)'], rows, 2)
    rows = [[f'Ep{n}', k, v['n'], num(v['mean_cer'], 3), num(v['median_cer'], 3)] for n, ep in sorted(results['episodes'].items()) for k, v in ep['text']['agreement'].items()]
    d.p('Character error rate between independent readings of the same speech (not against a truth): a low value means the readings agree, not that they are right.', 'mut')
    d.table(['ep', 'readings', 'lines compared', 'mean CER', 'median CER'], rows, 2)
    for n, ep in sorted(results['episodes'].items()):
        if ep.get('second'):
            s = ep['second']
            d.p(f"Episode {n} second opinion: {s['considered']} weak lines considered, {s['reread']} re-read in {s['seconds']:.0f} s, median CER {num(s['median_cer'], 3)}, {s['disagree_over_20pct']} readings differ by more than 20 %, "
                f"{s['settled_by_voice']} settled (readings agree and the line's tight voice decides), tight voice agrees with Qwen on {s['voice_agrees_with_qwen']}.")
    d.h(3, 'Qwen text fixes (names and homophones only) for you to read')
    rows = [[f'Ep{n}', f['method'], clock(f['time']), f['before'], f['after'], f['reason'], f['cer']] for n, ep in sorted(results['episodes'].items()) for f in ep['fixes'][:40]]
    d.table(['ep', 'method', 'time', 'before', 'after', 'reason', 'CER'], rows, 9) if rows else d.p('No fixes proposed.', 'mut')

    d.h(2, '4 · OP / ED and music')
    o = results['oped']
    d.p('Method: ' + (o.get('method') or '') + '. Ranges below are checked with a click on the time in any player: they should begin and end on the song.', 'mut')
    rows = []
    for n, e in sorted(o['episodes'].items()):
        for rg in e['ranges']:
            rows.append([f'Ep{n}', rg['kind'], rg['from'], rg['to'], num(rg['end'] - rg['start'], 1), rg['support'], ','.join(str(p) for p in rg['partners'])])
    d.table(['ep', 'kind', 'from', 'to', 'seconds', 'episodes it matches', 'which'], rows, 4)
    d.table(['kind', 'episodes', 'median s', 'min s', 'max s'], [[k, ','.join(map(str, v['episodes'])), num(v['median_seconds']), num(v['min_seconds']), num(v['max_seconds'])] for k, v in o['consistency'].items()], 2)
    rows = []
    for n, e in sorted(o['episodes'].items()):
        rows.append([f'Ep{n}', e['baseline_lines'], e['inside'], e['garbage_removed'], e['probably_real_dialogue_lost'], e['garbage_left_outside'], e['subtitle_cues_inside'],
                     pct(e['lyric_vs_ranges']['recall']) + '/' + pct(e['lyric_vs_ranges']['precision']), pct(e['music_like_vs_ranges']['recall']) + '/' + pct(e['music_like_vs_ranges']['precision'])])
    d.table(['ep', 'baseline lines', 'inside ranges', 'of which garbage (removed)', 'probably real Japanese dialogue (lost)', 'garbage left outside', 'subtitle cues inside', '1.b lyric-looking ASR: recall/precision vs ranges', '1.b loud-without-speech: recall/precision'], rows)
    for n, e in sorted(o['episodes'].items()):
        if e['real_examples']:
            d.p(f"Ep{n} Japanese lines inside the ranges (check whether any is real dialogue): " + ' · '.join(f"{x['time']} {x['text'][:30]}" for x in e['real_examples'][:6]), 'mut')
    if results.get('lfm3'):
        l = results['lfm3']
        d.p(f"1.c LFM3-Omni audio probe: audio input {'works' if l.get('audio') else 'not available'}; {l.get('note') or l.get('error') or ''} probe answers {l.get('probe')}; {len(l.get('windows', []))} windows labelled.", 'mut')
        if l.get('windows'):
            from collections import Counter
            c = Counter((w['kind'], w['label']) for w in l['windows'])
            d.table(['kind', 'label', 'windows'], [[k, v, n] for (k, v), n in sorted(c.items())], 2)

    d.h(2, '5 · Time')
    t = results['timings']
    d.table(['stage', 'seconds', 'detail'], [[k, num(v['seconds'], 1), ', '.join(f'{a}={b}' for a, b in v.items() if a != 'seconds')] for k, v in sorted(t.items())], 1)
    rt = results.get('recommended_time') or {}
    d.p(f"Recommended pipeline per 25-minute episode (from Episode 3): {json.dumps(rt.get('parts'))}, total {num(rt.get('total_seconds'), 0)} s; the current pipeline re-run on Episode 3: {num(rt.get('baseline_seconds'), 0)} s.", 'mut')

    d.h(2, '6 · Changed, skipped and caveats')
    d.table(['when', 'kind', 'note'], [[x['at'], x['kind'], x['text']] for x in results['notes']], 3) if results['notes'] else d.p('Nothing skipped.', 'mut')
    g = results.get('glossary') or {}
    if g:
        d.p('ASR context (A2/A3): ' + g.get('asr_context', ''), 'mut')
        d.table(['character', 'spelling proposed by Qwen (unverified)', 'reading', 'sure'], [[k, v['japanese'], v['reading'], v['sure']] for k, v in g.get('proposed', {}).items()], 3)

    d.h(2, '7 · Where the methods disagree most (with audio)')
    d.p('Japanese per ASR variant (B0 stored, A1 pause windows, A2 with names, A3 names + forced Japanese), the Portuguese subtitle for context only, and each method\'s speaker (confidence, reason). Open the spot-check page to settle some of them: spotcheck.html', 'mut')
    for n, picks in sorted(results['samples'].items()):
        d.h(3, f'Episode {n}')
        for p in picks:
            texts = ''.join(f'<div><span class="tag">{k}</span>{html.escape(v or "–")}</div>' for k, v in p['text'].items())
            spk = ' '.join(f"<span class=\"tag\">{html.escape(k)}: {html.escape(str(v['speaker']))}" + (f" · {v['confidence']}" if v.get('confidence') else '') + (f" · {v['reason']}" if v.get('reason') else '') + '</span>' for k, v in p['speakers'].items())
            clip = f'<audio controls preload="none" src="clips/ep{n}/{p["id"]}.wav"></audio>'
            d.raw(f"<div class=\"sample\"><b>{clock(p['start'])}</b> {p['id']} <span class=\"mut\">diar {html.escape(str(p['diar']))}</span> {clip}{texts}"
                  f"<div class=\"mut\">pt: {html.escape(p['pt'] or '–')}</div><div>{spk}</div></div>",
                  f"- {clock(p['start'])} {p['id']}: {p['text']['A1']} — " + '; '.join(f"{k}={v['speaker']}" for k, v in p['speakers'].items()))
    return d


def spotcheck(results):
    data = {n: [{'id': p['id'], 'start': p['start'], 'text': p['text']['A1'], 'pt': p['pt'], 'candidates': p['candidates']} for p in picks] for n, picks in results['samples'].items()}
    characters = sorted({c['title'] for c in (results.get('characters') or [])})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spot check · {html.escape(results['run_id'])}</title><style>{CSS}
.row{{border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0;background:var(--card)}}label{{margin-right:12px;white-space:nowrap}}button{{padding:6px 12px}}</style></head><body><main>
<h1>Spot check · {html.escape(results['run_id'])}</h1><p class="mut">Listen, pick who speaks. Saved in this browser only. When you are done, Export JSON and run <code>experiment.py rescore {html.escape(results['run_id'])} --labels file.json</code>.</p>
<button id="export">Export JSON</button> <span id="count" class="mut"></span><div id="list"></div></main>
<script>
const DATA={json.dumps(data, ensure_ascii=False)};const ALL={json.dumps(characters, ensure_ascii=False)};const KEY='aex-spot-{results['run_id']}';
let saved={{}};try{{saved=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const list=document.getElementById('list');
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(saved))}}catch(e){{}};document.getElementById('count').textContent=Object.values(saved).reduce((a,b)=>a+Object.keys(b).length,0)+' labelled'}}
for(const [ep,items] of Object.entries(DATA)){{const h=document.createElement('h2');h.textContent='Episode '+ep;list.appendChild(h);
for(const it of items){{const div=document.createElement('div');div.className='row';
const opts=[...new Set([...(it.candidates||[]),...ALL])].slice(0,14).concat(['unknown','other']);
div.innerHTML='<b>'+Math.floor(it.start/60)+':'+(it.start%60).toFixed(1).padStart(4,'0')+'</b> '+it.id+' <audio controls preload="none" src="clips/ep'+ep+'/'+it.id+'.wav"></audio><div>'+(it.text||'')+'</div><div class="mut">'+(it.pt||'')+'</div>';
const box=document.createElement('div');for(const o of opts){{const l=document.createElement('label');const r=document.createElement('input');r.type='radio';r.name=ep+it.id;r.value=o;
if((saved[ep]||{{}})[it.id]===o)r.checked=true;r.onchange=()=>{{(saved[ep]=saved[ep]||{{}})[it.id]=o;save()}};l.append(r,' '+o);box.appendChild(l)}}div.appendChild(box);list.appendChild(div)}}}}
document.getElementById('export').onclick=()=>{{const out={{}};for(const [ep,m] of Object.entries(saved)){{const x={{}};for(const [k,v] of Object.entries(m))if(v!=='unknown'&&v!=='other')x[k]=v;if(Object.keys(x).length)out['ep'+ep]=x}}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));a.download='labels.json';a.click()}};save();
</script></body></html>"""


def main(argv):
    run_dir = Path(argv[1])
    labels = None
    if '--labels' in argv:
        labels = json.load(open(argv[argv.index('--labels') + 1]))
    run = common.Run(run_dir.parent, run_dir.name)
    results = analysis.compute(run, labels)
    if results is None:
        print('nothing to report yet (prep.json missing)')
        return 1
    results['characters'] = (run.load('prep.json') or {}).get('characters', [])
    run.save(results, 'results.json')
    doc = build(results)
    running = (results['progress'].get('status') == 'running')
    (run_dir / 'report.html').write_text(doc.html(f"Audio experiment {results['run_id']}", refresh=running))
    (run_dir / 'report.md').write_text(doc.md())
    (run_dir / 'spotcheck.html').write_text(spotcheck(results))
    for name in ('report.html', 'report.md', 'spotcheck.html', 'results.json'):
        try:
            (run_dir / name).chmod(0o666)
        except OSError:
            pass
    print('wrote', run_dir / 'report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
