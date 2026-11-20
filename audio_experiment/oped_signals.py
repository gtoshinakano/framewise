"""Pure (stdlib) twins of the 1.b signal helpers in oped.py, so the report does not need numpy."""
import lines as L


def lyric(lines, min_lines=3, pad=1.5, gap=12.):
    marked = [(l['start_time'], l['end_time']) for l in lines if L.garbage_reasons(l['text'])]
    ranges = []
    for start, end in L.merge_intervals(marked, gap=gap):
        count = sum(1 for s, e in marked if s >= start - 1e-6 and e <= end + 1e-6)
        if count >= min_lines:
            ranges.append((max(0., start - pad), end + pad))
    return {'ranges': [[round(a, 1), round(b, 1)] for a, b in ranges], 'garbage_lines': len(marked)}


def compare(candidate, reference):
    ref, cand = L.merge_intervals(reference), L.merge_intervals([tuple(c) for c in candidate])
    ref_total = sum(e - s for s, e in ref)
    cand_total = sum(e - s for s, e in cand)
    shared = sum(L.overlap(s, e, rs, re_) for s, e in cand for rs, re_ in ref)
    return {'reference_seconds': round(ref_total, 1), 'candidate_seconds': round(cand_total, 1), 'shared_seconds': round(shared, 1),
            'recall': shared / ref_total if ref_total else None, 'precision': shared / cand_total if cand_total else None}
