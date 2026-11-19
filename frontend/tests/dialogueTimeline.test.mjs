import { test } from 'node:test'
import assert from 'node:assert/strict'
import { boundaryMarkers, clamp, limits, lineColour, lineTracks, openingFrame, placeLabels, placePoint, pointWarnings, samePoints, snap, snapTargets, wordsAround } from '../src/dialogueTimeline.ts'

// Three cuts of a sequence that starts 100 s into the episode: 0-4, 4-10 and 10-12 s.
const cuts = [
  { id: 'a', number: 5, t0: 0, t1: 4, start_time: 100, end_time: 104, ready: true },
  { id: 'b', number: 6, t0: 4, t1: 10, start_time: 104, end_time: 110, ready: true },
  { id: 'c', number: 7, t0: 10, t1: 12, start_time: 110, end_time: 112, ready: true },
]
const boundary = (id, after, before, t, extra = {}) => ({ id, after_cut: 0, before_cut: 0, after_cut_id: after, before_cut_id: before, t, time: 100 + t, speech: null, options: {}, unavailable: {}, fixed: null, choice: null, reason: null, by: null, ...extra })
const boundaries = [boundary('K1', 'a', 'b', 4, { speech: { lines: ['C2'] } }), boundary('K2', 'b', 'c', 10)]
const line = (id, t0, t1, text, parts) => ({ id, t0, t1, text, parts: parts || [{ t0, t1, text }] })

test('every line has a colour and neighbours never share one', () => {
  const colours = Array.from({ length: 25 }, (_, index) => lineColour(index))
  colours.slice(1).forEach((colour, index) => assert.notEqual(colour, colours[index]))
  assert.equal(lineColour(-1), lineColour(9))
})

test('markers come from your points, else the plan, else the hard cut', () => {
  const shots = [{ index: 1, cut_id: 'a', part: null, t0: 0 }, { index: 2, cut_id: 'b', part: null, t0: 4.6 }, { index: 3, cut_id: 'c', part: null, t0: 10 }]
  const [first, second] = boundaryMarkers(boundaries, shots, [{ after_cut_id: 'b', time: 109.2 }], 100)
  assert.deepEqual([first.t, first.source, first.shot, first.speech], [4.6, 'plan', 2, true])
  assert.deepEqual([second.t, second.source, second.by, second.shot], [9.2, 'you', 'user', 3])
  // Saved as yours but handed back in the draft: the hard cut until saved, not the plan's old placement.
  const released = boundaryMarkers([boundaries[0], { ...boundaries[1], fixed: { t: 9.2, time: 109.2, shift: -0.8 } }], shots, [], 100)[1]
  assert.deepEqual([released.t, released.source], [10, 'cut'])
  assert.deepEqual(boundaryMarkers(boundaries, [], [], 100).map(item => [item.t, item.source, item.shot]), [[4, 'cut', 2], [10, 'cut', 3]])
})

test('compiled starts appear before Direction and user points override them', () => {
  const starts = [{ after_cut_id: 'a', time: 104.65, by: 'director', reason: 'Keep the phrase whole.', speech: true }]
  const [suggested] = boundaryMarkers(boundaries, [], [], 100, starts)
  assert.deepEqual([suggested.t, suggested.source, suggested.reason], [4.65, 'dialogue', 'Keep the phrase whole.'])
  const [placed] = boundaryMarkers(boundaries, [], [{ after_cut_id: 'a', time: 104 }], 100, starts)
  assert.deepEqual([placed.t, placed.source], [4, 'you'])
})

test('a boundary may move anywhere the hard rules allow', () => {
  const markers = boundaryMarkers(boundaries, [], [], 100)
  // K1: shot 1 keeps 0.2 s of cut 5; shot 2 keeps 0.2 s of cut 6, which ends at K2's hard cut.
  assert.deepEqual(limits(0, markers, cuts), { min: 0.2, max: 9.8 })
  // K2: shot 3 keeps 0.2 s of cut 7, and shot 2 starts at K1's hard cut.
  assert.deepEqual(limits(1, markers, cuts), { min: 4.2, max: 11.8 })
  // A placed neighbour counts where you put it; the director's do not.
  const placed = boundaryMarkers(boundaries, [], [{ after_cut_id: 'a', time: 107 }], 100)
  assert.deepEqual(limits(1, placed, cuts), { min: 7.2, max: 11.8 })
  // A shot of a cut that fits one take stays within it.
  assert.deepEqual(limits(0, markers, cuts, 5), { min: 0.2, max: 5 })
  assert.equal(clamp(20, { min: 1, max: 3 }), 3)
})

test('dragging snaps to hard cuts, the silence around lines and word edges', () => {
  const lines = [line('C1', 2, 3.5, 'もう一度', [{ t0: 2, t1: 2.8, text: 'もう' }, { t0: 2.8, t1: 3.5, text: '一度' }])]
  const targets = snapTargets(lines, cuts)
  assert.deepEqual(targets, [1.94, 2.8, 3.56, 4, 10])
  assert.equal(snap(3.98, targets, 0.1), 4)
  assert.equal(snap(3.7, targets, 0.1), 3.7)
  assert.equal(snap(2.85, targets, 0.1), 2.8)
})

test('the opening frame is the cut\'s first frame unless the shot starts inside its cut', () => {
  const early = openingFrame({ t: 3.5 }, cuts[1], 100)
  assert.equal(early.url, '/api/cuts/b/frame/first?v=104')
  assert.match(early.caption, /First frame of cut 6; it holds 0.50 s/)
  assert.equal(openingFrame({ t: 4 }, cuts[1], 100).derived, false)
  const late = openingFrame({ t: 4.75 }, cuts[1], 100)
  assert.deepEqual([late.url, late.derived], ['/api/cuts/b/boundary-preview-frame?time=104.750', true])
})

test('warnings name divided speech, long moves and hidden pictures', () => {
  const lines = [line('C2', 3, 5, 'ちょっと待って', [{ t0: 3, t1: 4, text: 'ちょっと' }, { t0: 4, t1: 5, text: '待って' }])]
  const [marker] = boundaryMarkers(boundaries, [], [{ after_cut_id: 'a', time: 103.6 }], 100)
  assert.deepEqual(pointWarnings(marker, cuts, lines), ['Divides C2: “ちょっと” ends the shot before, “待って” opens shot 2.'])
  const far = boundaryMarkers(boundaries, [], [{ after_cut_id: 'a', time: 100.5 }], 100)[0]
  assert.deepEqual(pointWarnings(far, cuts, []), ['3.50 s before the hard cut; the director never moves a boundary more than 3 s.', "Hides 3.50 s of cut 5's 4.00 s picture."])
  assert.deepEqual(wordsAround(line('L', 0, 2, 'good morning', [{ t0: 0, t1: 1, text: 'good' }, { t0: 1, t1: 2, text: 'morning' }]), 2.5), { before: 'good morning', after: '' })
})

test('overlapping lines get their own track and labels never overlap', () => {
  assert.deepEqual(lineTracks([line('A', 0, 2, ''), line('B', 1, 3, ''), line('C', 2.5, 4, ''), line('D', 1.5, 1.8, '')]), [0, 1, 0, 2])
  assert.deepEqual(placeLabels([0, 5, 40, 41], 13), [0, 13, 40, 53])
  assert.deepEqual(placeLabels([30, 0], 13), [30, 0])
})

test('placing and handing back points', () => {
  const points = placePoint([], 'a', 103.25)
  assert.deepEqual(points, [{ after_cut_id: 'a', time: 103.25 }])
  assert.deepEqual(placePoint(points, 'a', 103.5), [{ after_cut_id: 'a', time: 103.5 }])
  assert.deepEqual(placePoint(points, 'a', null), [])
  assert.equal(samePoints([{ after_cut_id: 'b', time: 1 }, { after_cut_id: 'a', time: 2 }], [{ after_cut_id: 'a', time: 2.0001 }, { after_cut_id: 'b', time: 1 }]), true)
  assert.equal(samePoints(points, []), false)
})
