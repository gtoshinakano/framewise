"""Render qwen_experiment/results.json into report.html: inference times and side-by-side answers.

Standalone (stdlib only), so it runs on the host after the GPU run. Reads the results next to it
(or --results) and writes report.html there. Frames live in frames/ beside the file.

    python3 report.py                 # report.html next to results.json in this folder
    python3 report.py --results X.json --out Y.html
"""
import argparse
import json
import html
import statistics
from pathlib import Path


def _pct95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return ordered[index]


def _stats(values):
    if not values:
        return {}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": _pct95(values), "min": min(values), "max": max(values), "total": sum(values)}


def _usage_total(frames, key):
    used = [a["usage"] for s in frames for a in [s.get("answers", {}).get(key)] if a and isinstance(a.get("usage"), dict)]
    prompts = [u.get("prompt_tokens", 0) for u in used if isinstance(u.get("prompt_tokens"), int)]
    completions = [u.get("completion_tokens", 0) for u in used if isinstance(u.get("completion_tokens"), int)]
    return {"prompt": sum(prompts), "completion": sum(completions)}


def _pretty(content):
    if content is None:
        return ""
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return content


def _fmt(value, digits=1, unit="s"):
    return f"{value:.{digits}f}{unit}" if isinstance(value, (int, float)) else "—"


def _badges(times):
    """Mark the faster answer of a pair green and the slower one amber."""
    usable = [t for t in times if isinstance(t, (int, float))]
    if len(usable) < 2:
        return ["", ""]
    fastest, slowest = min(usable), max(usable)
    return ["fast" if t == fastest and t != slowest else ("slow" if t == slowest and t != fastest else "") for t in times]


def build_html(meta, frames):
    keys = list(meta.get("models", {}).keys())
    labels = {key: meta["models"][key].get("label", key) for key in keys}
    per_model = {}
    for key in keys:
        times = [a["latency_seconds"] for s in frames for a in [s.get("answers", {}).get(key)]
                 if a and isinstance(a.get("latency_seconds"), (int, float))]
        usage = _usage_total(frames, key)
        stats = _stats(times)
        decode = (usage["completion"] / stats["total"]) if stats.get("total") else 0.0
        per_model[key] = {"stats": stats, "usage": usage, "decode": decode}

    existing = [s["existing_seconds"] for s in frames if isinstance(s.get("existing_seconds"), (int, float))]
    existing_models = {}
    for s in frames:
        if s.get("existing_model_id"):
            existing_models[s["existing_model_id"]] = existing_models.get(s["existing_model_id"], 0) + 1

    episode = meta.get("episode", {})
    canvas = meta.get("canvas", [])
    subtitle = meta.get("subtitle")

    # Summary table rows.
    summary_rows = ""
    for key in keys:
        st = per_model[key]["stats"]
        u = per_model[key]["usage"]
        summary_rows += f"""<tr>
          <td><b>{html.escape(labels[key])}</b></td>
          <td>{st.get('n', 0)}</td>
          <td>{_fmt(st.get('median'))}</td>
          <td>{_fmt(st.get('mean'))}</td>
          <td>{_fmt(st.get('p95'))}</td>
          <td>{_fmt(st.get('min'))} / {_fmt(st.get('max'))}</td>
          <td>{u['prompt']:,}</td>
          <td>{u['completion']:,}</td>
          <td>{per_model[key]['decode']:.1f}</td>
        </tr>"""
    if existing:
        ex = _stats(existing)
        summary_rows += f"""<tr class="historical">
          <td><b>Stored (historical)</b></td>
          <td>{ex['n']}</td>
          <td>{_fmt(ex['median'])}</td>
          <td>{_fmt(ex['mean'])}</td>
          <td>{_fmt(ex['p95'])}</td>
          <td>{_fmt(ex['min'])} / {_fmt(ex['max'])}</td>
          <td>—</td><td>—</td><td>—</td>
        </tr>"""

    # Bars for median inference time (pure CSS).
    bars = ""
    if keys:
        median_max = max((per_model[k]["stats"].get("median", 0) for k in keys), default=1) or 1
        for key in keys:
            median = per_model[key]["stats"].get("median", 0)
            width = (median / median_max * 100) if median_max else 0
            bars += f"""<div class="bar-row"><span class="bar-label">{html.escape(labels[key])}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <span class="bar-value">{_fmt(median)}</span></div>"""

    # Per-sample cards.
    samples = ""
    for s in sorted(frames, key=lambda f: f.get("cut_index", 0)):
        idx = s.get("cut_index", 0)
        image = s.get("image", "")
        people = ", ".join(p["name"] for p in s.get("people", [])) or "none"
        subtitle_line = s.get("subtitle_language") or "no subtitles"
        existing_line = (f"{s.get('existing_model_id', '?')} · {_fmt(s.get('existing_seconds'))}"
                         if s.get("existing_seconds") is not None else "no stored run")
        times = [s.get("answers", {}).get(key, {}).get("latency_seconds") for key in keys]
        badges = _badges(times)
        thumb = f"<img class='thumb' src='{html.escape(image)}' alt='cut {idx}' loading='lazy'>" if (image and Path(image).exists()) else ""
        cols = ""
        for position, key in enumerate(keys):
            answer = s.get("answers", {}).get(key, {})
            if not answer:
                continue
            latency = answer.get("latency_seconds")
            usage = answer.get("usage", {})
            tokens = f"prompt {usage.get('prompt_tokens', '?')}, completion {usage.get('completion_tokens', '?')}"
            status = answer.get("error") or answer.get("finish_reason") or ""
            badge = badges[position]
            cols += f"""<div class="answer">
              <div class="answer-head"><span class="model">{html.escape(labels[key])}</span>
              <span class="time {badge}">{_fmt(latency)}</span></div>
              <div class="answer-meta">{html.escape(tokens)} · {html.escape(status)}</div>
              <pre class="json">{html.escape(_pretty(answer.get('content')))}</pre>
            </div>"""
        samples += f"""<section class="sample">
          <div class="sample-head"><h3>Cut {idx}</h3><div class="sample-meta">
            marked: {html.escape(people)} · {html.escape(subtitle_line)} · stored: {html.escape(existing_line)}</div></div>
          <div class="sample-body">{thumb}<div class="answers">{cols}</div></div>
        </section>"""

    subtitle_text = (f"{subtitle.get('language')} at {subtitle.get('top'):.2f}–{subtitle.get('bottom'):.2f}"
                     if isinstance(subtitle, dict) and subtitle.get("language") else "none")
    existing_note = ", ".join(f"{count}× {html.escape(model)}" for model, count in existing_models.items()) or "none"
    models_note = "; ".join(f"{html.escape(labels[key])} ({meta['models'][key].get('model_id', '')})" for key in keys)

    css = """
:root { --fg:#1a1a1a; --muted:#666; --border:#d8d8d8; --surface:#fff; --accent:#2563eb; --fast:#1a7f37; --slow:#b45309; --bg:#f6f6f8; }
@media (prefers-color-scheme: dark) { :root { --fg:#e8e8e8; --muted:#aaa; --border:#3a3a3a; --surface:#1f1f24; --accent:#7aa4f0; --fast:#46c46d; --slow:#e09b4d; --bg:#131317; } }
body { font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 0 18px 60px; }
h1 { font-size: 22px; margin: 24px 0 4px; } .sub { color: var(--muted); margin: 0 0 24px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin: 16px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
td:first-child b { color: var(--accent); }
.historical td { color: var(--muted); background: rgba(0,0,0,0.03); }
.bar-row { display: grid; grid-template-columns: 200px 1fr 60px; align-items: center; gap: 12px; margin: 6px 0; }
.bar-track { background: var(--border); border-radius: 4px; height: 12px; }
.bar-fill { background: var(--accent); height: 100%; border-radius: 4px; }
.bar-label { color: var(--muted); } .bar-value { font-variant-numeric: tabular-nums; }
.method { color: var(--muted); font-size: 13px; } .method li { margin: 2px 0; }
.sample-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.sample-head h3 { margin: 0; font-size: 16px; } .sample-meta { color: var(--muted); font-size: 12px; }
.sample-body { display: grid; grid-template-columns: 200px 1fr; gap: 16px; margin-top: 8px; }
.thumb { width: 200px; border: 1px solid var(--border); border-radius: 6px; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.answer { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.answer-head { display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; background: rgba(0,0,0,0.03); border-bottom: 1px solid var(--border); }
.model { font-weight: 600; } .time { font-variant-numeric: tabular-nums; padding: 2px 8px; border-radius: 4px; background: var(--border); }
.time.fast { background: var(--fast); color: #fff; } .time.slow { background: var(--slow); color: #fff; }
.answer-meta { color: var(--muted); font-size: 12px; padding: 6px 10px 0; }
pre.json { margin: 6px 10px 10px; padding: 8px; background: rgba(0,0,0,0.04); border-radius: 6px; font: 12px/1.35 ui-monospace, SFMono, Menlo, Consolas, monospace; overflow-x: auto; white-space: pre-wrap; max-height: 560px; overflow-y: auto; }
@media (max-width: 900px) { .sample-body { grid-template-columns: 1fr; } .thumb { width: 100%; } }
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>First-frame model benchmark</title>
<style>{css}</style></head><body>
<h1>First-frame analysis: Qwen 3.6 vs Qwen 3.5</h1>
<p class="sub">{html.escape(episode.get('title', ''))} · {len(frames)} first frames · generated {html.escape(meta.get('generated_at', ''))}</p>

<section>
<h3 style="margin-top:0">Models</h3>
<p class="method">{models_note}</p>
<ul class="method">
  <li>Machine: {html.escape(meta.get('runtime', ''))} runtime; {canvas[0] if canvas else '?'}×{canvas[1] if canvas else '?'} first frames.</li>
  <li>Subtitle strip: {html.escape(subtitle_text)}.</li>
  <li>Both models ask the <b>same</b> message: one server at a time, <code>parallel=1</code>, <code>temperature=0</code>, JSON-schema grammar with the app's fallback.</li>
  <li>Time taken = one full request round-trip (encode + inference + response), so it is comparable between models.</li>
  <li>The app batches first frames at <code>FRAME_PARALLEL=4</code>; here each cut is one isolated call, so absolute times differ from a queued run.</li>
  <li>Historical stored runs: {existing_note}.</li>
</ul>
</section>

<section>
<h3 style="margin-top:0">Inference time per first frame</h3>
<table><thead><tr><th>Model</th><th>Calls</th><th>Median</th><th>Mean</th><th>p95</th><th>Min / Max</th><th>Prompt tok</th><th>Out tok</th><th>Out tok/s</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<div style="margin-top:14px">{bars}</div>
</section>

<section>
<h3 style="margin-top:0">Answers, side by side</h3>
<p class="method">Each card shows the extracted first frame and the two models' answers for the same image. The green badge is the faster call of the pair.</p>
</section>
{samples}
</body></html>"""

def main():
    parser = argparse.ArgumentParser(description="Render the benchmark results into report.html")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--out", default="report.html")
    args = parser.parse_args()
    results = Path(args.results)
    if not results.exists():
        print(f"no results at {results}; run benchmark.py first", file=sys.stderr)
        return 1
    data = json.loads(results.read_text(encoding="utf-8"))
    html_text = build_html(data["meta"], data["frames"])
    Path(args.out).write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out} ({len(data['frames'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
