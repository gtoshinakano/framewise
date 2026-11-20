#!/usr/bin/env node
// Drive one real Chromium session through the Framewise editor and do the work by hand:
// open Run settings -> Target frames, draw a box on a frame, and choose the character.
//
//   node target-frames.mjs inspect   --episode <id> --cut <n|a-b>
//   node target-frames.mjs draw      --episode <id> --cut <n> [--character NAME] [--frames 2,4-6|all]
//                          [--mode pick|carry] [--mark-reviewed] [--detect]
//   node target-frames.mjs name      --episode <id> --cut <n> --slot <n> --character NAME
//   … add --dry-run to print the plan, --headed to watch it, --cdp <url> to attach.
//
// The character is always the one the opening frame shows (the run's mirrored first frame):
// a colour that is not on that frame is never named, and a frame is never given a box when
// nothing supports where the person is. Every click and drag is real; every edit is checked
// against the API afterwards, because the API is what the screen reads.
//
// Exit codes: 0 done, 1 an error, 2 a person is needed.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'

const DEFAULT_URL = 'http://192.168.0.6:8080'
const WAIT_JOB = 600000       // a queued detection may wait for the GPU
const WAIT_BOX = 30000        // point detection loads its detector on the first call
const CHROMIUM = process.env.CHROMIUM_PATH || '/usr/bin/chromium'
// What the keyframe analysis writes when a person left the picture; carrying a box onto
// such a frame would invent someone the app already saw leaving.
const NOT_VISIBLE = /not visible|no longer visible|not (in|on|within) (the )?(frame|picture|shot|view|screen)|out of (the )?(frame|picture|shot|screen)|off( the)? (screen|frame|picture)|disappear|left the (frame|picture|shot|screen)|leaves? the (frame|picture|shot|screen)|no longer (shown|present)|gone from the (frame|picture|screen)/i

function usage() {
  console.log(`usage: node target-frames.mjs <inspect|draw|name> --episode <id> --cut <n|a-b> [options]

  --character NAME   the character to draw or name (default: the opening frame's only one)
  --frames LIST      2,4-6 | all (default: every frame after the first that lacks their box)
  --mode pick        double-click where they were: the app's detector finds the box (default)
  --mode carry       drag their last box; needs a crop or a keyframe state that puts them in the picture
  --allow-unmatched  also take a crop the tracker found for no one: the rule names it the first screen's character
  --force            name: allow a colour the opening frame does not show
  --slot N           name: colour to name (0 is colour 1)
  --mark-reviewed    mark each fixed frame reviewed afterwards
  --detect           press Detect characters first, and wait for the run
  --dry-run          print the plan, change nothing
  --headed           show the window (needs a display)
  --cdp URL          attach to a Chromium started with --remote-debugging-port
  --url URL          app address (default FRAMEWISE_URL, then .env APP_BIND_IP, then ${DEFAULT_URL})`)
}

function parseArgs(argv) {
  const out = { mode: 'pick', frames: 'all', framesSet: null, cut: null, url: '' }
  const flags = new Set(['dry-run', 'headed', 'mark-reviewed', 'detect', 'force', 'allow-unmatched'])
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (!arg.startsWith('--')) { out.extra?.push(arg) ?? (out.extra = [arg]); continue }
    const [key, inline] = arg.slice(2).split('=')
    if (flags.has(key)) { out[key] = true; continue }
    const value = inline !== undefined ? inline : argv[++i]
  if (value === undefined || value.startsWith('--')) fail(`--${key} needs a value`)
    out[key] = value
  }
  if (!out.url) out.url = process.env.FRAMEWISE_URL || fromEnvFile() || DEFAULT_URL
  if (!out.episode) fail('--episode is required')
  if (out.cut === null) fail('--cut is required')
  const [first, last] = String(out.cut).split('-')
  if (!Number.isInteger(Number(first)) || Number(first) < 1) fail('--cut must be a number or a-b')
  out.cuts = [Number(first), last === undefined ? Number(first) : Number(last)]
  if (out.cuts[1] < out.cuts[0]) fail('--cut range is backwards')
  if (out.frames !== 'all') out.framesSet = parseList(out.frames)
  if (out.mode !== 'pick' && out.mode !== 'carry') fail('--mode is pick or carry')
  return out
}

function parseList(text) {
  const out = new Set()
  for (const part of text.split(',')) {
    const [a, b] = part.trim().split('-')
    const from = Number(a), to = Number(b ?? a)
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 1 || to < from) fail(`--frames wants 2,4-6 or all: got ${part}`)
    for (let n = from; n <= to; n++) out.add(n)
  }
  return out
}

function fromEnvFile() {
  for (const dir of [process.cwd(), path.dirname(process.cwd())]) {
    const file = path.join(dir, '.env')
    if (!fs.existsSync(file)) continue
    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
      const [key, ...rest] = line.split('=')
      if (key?.trim() === 'APP_BIND_IP' && rest.join('=').trim()) return `http://${rest.join('=').trim()}:8080`
    }
  }
  return ''
}

function fail(message) {
  console.error(`target-frames: ${message}`)
  process.exit(1)
}

async function api(url, request) {
  const response = await fetch(`${url}/api${request?.path ?? ''}`, {
    method: request?.method ?? 'GET',
    headers: request?.body ? { 'content-type': 'application/json' } : undefined,
    body: request?.body ? JSON.stringify(request.body) : undefined,
  })
  const text = await response.text()
  if (!response.ok) {
    let detail = text.slice(0, 400)
    try { detail = JSON.parse(text).detail ?? detail } catch {}
    return { ok: false, status: response.status, detail }
  }
  return { ok: true, value: text ? JSON.parse(text) : null }
}
/** A short reason from a response body, whatever the app put in it. */
async function safeText(response) {
  try {
    const text = await response.text()
    return (JSON.parse(text).detail ?? text).slice(0, 200)
  } catch {
    return `status ${response.status()}`
  }
}

async function loadPlaywright() {
  const tries = [process.env.PLAYWRIGHT_MODULE]
  try { tries.push(createRequire(import.meta.url).resolve('playwright')) } catch {}
  const roots = [path.join(process.cwd(), 'node_modules'), path.join(import.meta.dirname, 'node_modules'),
    path.join(os.homedir(), '.local/share/mise/installs/npm-playwright'), '/usr/lib/node_modules', '/usr/local/lib/node_modules']
  for (const root of roots) {
    if (!fs.existsSync(root)) continue
    const candidates = root.includes('npm-playwright')
      ? fs.readdirSync(root).filter(name => name !== 'latest').map(name => path.join(root, name, 'node_modules', 'playwright'))
      : [path.join(root, 'playwright')]
    for (const dir of candidates) {
      for (const entry of ['index.mjs', 'index.js']) {
        const file = path.join(dir, entry)
        if (fs.existsSync(file)) tries.push(file)
      }
    }
  }
  for (const entry of tries) {
    if (!entry) continue
    try {
      const module = await import(entry.startsWith('node:') ? entry : pathToFileUrl(entry))
      const chromium = module.chromium ?? module.default?.chromium
      if (chromium) return { chromium, expect: module.expect ?? module.default?.expect }
    } catch {}
  }
  fail('Playwright is not installed here. Install it (npm install playwright) or set PLAYWRIGHT_MODULE to its index.mjs.')
}

function pathToFileUrl(file) {
  return `file://${file}`
}

/** The editor's cut numbers are 1-based over the active cuts, in cut_index order. */
async function cutIds(url, episode, wanted) {
  const response = await api(url, { path: `/v1/episodes/${episode}/cuts?limit=500` })
  if (!response.ok) fail(`Could not read the cuts: ${response.detail}`)
  const cuts = response.value.items.slice().sort((a, b) => a.cut_index - b.cut_index)
  return wanted.map(number => ({ number, cut: cuts[number - 1], total: cuts.length }))
}

async function loadState(url, episode, cutId) {
  const selection = await api(url, { path: `/cuts/${cutId}/keyframe-selection` })
  if (!selection.ok) fail(`Could not read the keyframes: ${selection.detail}`)
  const detections = await api(url, { path: `/cuts/${cutId}/target-detections` })
  if (!detections.ok) fail(`Could not read the Target frames run: ${detections.detail}`)
  const analysis = await api(url, { path: `/cuts/${cutId}/keyframe-analysis` })
  const state = { selection: selection.value, ...detections.value }
  const frames = state.selection?.result?.selected_frames ?? []
  const byOrdinal = new Map((state.run?.frames ?? []).map(frame => [frame.ordinal, frame]))
  state.onFrame = {}
  for (const frame of (analysis.value?.result?.frames ?? [])) {
    state.onFrame[frame.ordinal] = Object.fromEntries((frame.characters ?? []).map(person => [person.name, person.state ?? '']))
  }
  state.analysed = !!analysis.value?.result
  state.list = frames.map((frame, index) => ({
    index: index + 1, ordinal: frame.source_frame_index, size: byOrdinal.get(frame.source_frame_index)?.image_size ?? [1280, 720],
    mirror: !!byOrdinal.get(frame.source_frame_index)?.mirror,
    review: byOrdinal.get(frame.source_frame_index)?.review,
    boxes: (state.run?.detections ?? []).filter(box => box.frame_ordinal === frame.source_frame_index),
    candidates: (state.run?.candidates ?? []).filter(item => item.frame_ordinal === frame.source_frame_index),
  }))
  return state
}

/** The people the opening frame shows: the run's mirrored frame, in colour order. */
function firstScreen(state) {
  const first = state.list[0]
  if (!first || !state.run) return []
  const onFirst = new Set(first.boxes.map(box => box.slot))
  return state.run.slots.filter(slot => onFirst.has(slot.slot) && !slot.ignored)
}

function slotName(state, slot) {
  const found = state.run?.slots.find(item => item.slot === slot)
  return found?.character_name || `unnamed colour ${slot + 1}`
}

/**
 * What to do on this cut, with the rule applied: only the opening frame's character, and a
 * frame only when a detector or their own last frame says where they are.
 */
function plan(state, wanted, framesWanted, mode, allowUnmatched) {
  const actions = []
  if (!state.run) { actions.push({ kind: 'blocked', why: 'no run yet: press Detect characters (--detect) or wait for the queue' }); return actions }
  if (state.job?.status === 'running' || state.job?.status === 'queued') { actions.push({ kind: 'blocked', why: `detection is ${state.job.status}; wait or cancel it` }); return actions }
  if (state.run.stale) { actions.push({ kind: 'blocked', why: `the run is stale (${state.run.stale_reason || 'the frames changed'}); detect again` }); return actions }
  const cast = firstScreen(state)
  if (!cast.length) { actions.push({ kind: 'person', why: 'the opening frame has no boxes, so no one is on the first screen: box them in Characters in this cut first' }); return actions }
  let chosen = null
  if (wanted) {
    chosen = cast.find(slot => slot.character_name === wanted)
    if (!chosen) actions.push({ kind: 'person', why: `${wanted} is not on the first screen (${cast.map(slot => slotName(state, slot.slot)).join(', ')}); the rule is the first screen's character` })
  } else if (cast.length === 1) {
    chosen = cast[0]
  } else {
    const losing = cast.filter(slot => state.list.some(frame => frame.index > 1 && !frame.boxes.some(box => box.slot === slot.slot)))
    if (losing.length === 1) chosen = losing[0]
    else actions.push({ kind: 'person', why: `the first screen shows ${cast.map(slot => slotName(state, slot.slot)).join(', ')}: say --character which one` })
  }
  if (!chosen) return actions
  if (!chosen.character_name) { actions.push({ kind: 'person', why: `the first screen's person (colour ${chosen.slot + 1}) has no name: a person names them, then this runs with --character` }); return actions }
  actions.push({ kind: 'character', slot: chosen.slot, name: chosen.character_name })
  const limit = state.max_boxes ?? 3
  for (const frame of state.list) {
    if (frame.index === 1) continue            // the mirrored opening frame is edited in the other tab
    if (frame.boxes.some(box => box.slot === chosen.slot)) continue
    if (framesWanted && !framesWanted.has(frame.index)) continue
    if (frame.boxes.length >= limit) { actions.push({ kind: 'blocked', frame: frame.index, why: `colour ${chosen.slot + 1} cannot join: ${frame.boxes.length}/${limit} colours are used` }); continue }
    const seen = state.onFrame?.[frame.ordinal]?.[chosen.character_name]
    if (seen && NOT_VISIBLE.test(seen)) { actions.push({ kind: 'blocked', frame: frame.index, why: `their keyframe state says they are not in the picture (${seen.slice(0, 70)})` }); continue }
    const count = frame.review?.flags?.count_changed
    if (count?.here === 0) { actions.push({ kind: 'blocked', frame: frame.index, why: `the run itself sees nobody here (it saw ${count.first ?? 0} on the opening frame): a person decides` }); continue }
    const useful = frame.candidates.filter(item => !absurd(item.box, frame.size))
    const suggested = useful.find(item => item.suggested_slot === chosen.slot)
    if (suggested) { actions.push({ kind: 'promote', slot: chosen.slot, frame: frame.index, ordinal: frame.ordinal, candidate: suggested, from: 'the tracker suggests this colour' }); continue }
    const other = allowUnmatched ? useful.find(item => !item.suggested_slot) : null
    if (other) { actions.push({ kind: 'promote', slot: chosen.slot, frame: frame.index, ordinal: frame.ordinal, candidate: other, from: 'a crop no one has a colour for' }); continue }
    const carried = carriedBox(state, chosen.slot, frame.index)
    const support = useful.find(item => carried ? overlaps(item.box, carried) : true) ?? useful[0] ?? null
    const inPicture = !!seen && !NOT_VISIBLE.test(seen)
    if (mode === 'carry') {
      if (!carried || !(support || inPicture)) {
        actions.push({ kind: 'person', frame: frame.index, why: `${carried ? 'no crop there' : 'no box of theirs earlier'}${inPicture ? '' : ' and no keyframe state for them'}: a person decides whether they left the picture` })
        continue
      }
      actions.push({ kind: 'drag', slot: chosen.slot, frame: frame.index, ordinal: frame.ordinal, box: clamp(carried, frame.size), from: inPicture ? 'their last frame, the analysis puts them in F' + frame.index : 'their last frame' })
      continue
    }
    if (!support && !carried) { actions.push({ kind: 'person', frame: frame.index, why: 'no crop and no earlier box: nothing says where they are' }); continue }
    actions.push({ kind: 'pick', slot: chosen.slot, frame: frame.index, ordinal: frame.ordinal,
      point: centre(clamp(support?.box ?? carried, frame.size)), from: support ? 'a tracker crop' : 'their last frame' })
  }
  return actions
}

function carriedBox(state, slot, frameIndex) {
  for (let index = frameIndex - 1; index >= 1; index--) {
    const found = state.list[index - 1]?.boxes.find(box => box.slot === slot)
    if (found) return found.box
  }
  return null
}

function centre(box) { return [Math.round((box[0] + box[2]) / 2), Math.round((box[1] + box[3]) / 2)] }
/** A crop that is nearly the whole frame is a detector failure, not a person. */
function absurd(box, size) {
  return Math.abs(box[2] - box[0]) > size[0] * 0.9 && Math.abs(box[3] - box[1]) > size[1] * 0.9
}

function clamp(box, size) {
  const [x1, y1, x2, y2] = box.map(value => Math.max(0, Math.min(size[0], value)))
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]
}

function overlaps(a, b) {
  const wide = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const tall = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  return wide > 0 && tall > 0
}

async function openBrowser(opts) {
  const { chromium } = await loadPlaywright()
  const args = opts.headed ? ['--ozone-platform=wayland'] : []
  if (process.env.CHROMIUM_ARGS) args.push(...process.env.CHROMIUM_ARGS.split(' ').filter(Boolean))
  if (opts.cdp) {
    const browser = await chromium.connectOverCDP(opts.cdp)
    const context = browser.contexts()[0] ?? await browser.newContext({ viewport: { width: 1600, height: 1000 } })
    return { browser, context }
  }
  const browser = await chromium.launch({ executablePath: CHROMIUM, headless: !opts.headed, args })
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } })
  return { browser, context }
}

async function poll(give, ms, message) {
  const until = Date.now() + ms
  let last = null
  while (Date.now() < until) {
    last = await give()
    if (last) return last
    await new Promise(done => setTimeout(done, 400))
  }
  throw new Error(`timed out: ${message} (last: ${JSON.stringify(last)?.slice(0, 300) ?? 'nothing'})`)
}

/** Go to the episode, open Cut analysis, jump to the cut number, open Run settings -> Target frames. */
async function openScreen(page, url, episode, number) {
  await page.goto(`${url}/#/episode/${episode}`, { waitUntil: 'domcontentloaded' })
  const open = page.getByRole('button', { name: 'Analysis & results' })
  const cards = page.locator('[id^="cut-card-"]')
  // The app opens with no cut selected, so the inspector has no buttons until one is picked.
  await Promise.race([open.waitFor({ state: 'visible', timeout: 60000 }).catch(() => null), cards.first().waitFor({ state: 'visible', timeout: 60000 }).catch(() => null)])
  if (!(await open.count())) {
    await Promise.race([cards.first().waitFor({ state: 'visible', timeout: 30000 }).catch(() => null), page.getByTitle('Show the cut list').waitFor({ timeout: 30000 }).catch(() => null)])
    if (await page.getByTitle('Show the cut list').count()) await page.getByTitle('Show the cut list').click()
    await cards.first().waitFor({ state: 'visible', timeout: 30000 })
    await cards.first().click()
  }
  await open.waitFor({ state: 'visible', timeout: 30000 })
  await open.click()
  const dialog = page.getByRole('dialog', { name: /^Analysis for/ })
  await dialog.waitFor({ timeout: 15000 })
  const jump = dialog.getByRole('spinbutton', { name: 'Go to cut number' })
  await jump.waitFor({ timeout: 15000 })
  if (await jump.inputValue() !== String(number)) { await jump.fill(String(number)); await jump.press('Enter') }
  await poll(async () => (await jump.inputValue()) === String(number), 20000, `the dialog did not move to cut ${number}`)
  await dialog.getByRole('tab', { name: 'Run settings' }).click()
  await dialog.getByRole('tab', { name: 'Target frames' }).click()
  const screen = dialog.locator('section[aria-label="Target frames"]')
  await screen.waitFor({ timeout: 15000 })
  const filter = screen.getByRole('checkbox', { name: /^Needs review/ })
  if (await filter.count() && await filter.isChecked()) await filter.uncheck()
  return { dialog, screen }
}

/** The frame cards render from the run; wait until the ones we need are on the screen. */
async function ready(screen, list) {
  for (const frame of list) await screen.locator(`#target-frame-${frame.ordinal}`).waitFor({ state: 'visible', timeout: 20000 })
}

function frameScreen(screen, ordinal) { return screen.locator(`#target-frame-${ordinal}`) }

function layer(card) { return card.locator('svg[aria-label*="character boxes"]') }

/** The transparent layer a double-click reaches; it is absent while the app is saving. */
const PICK_LAYER = 'svg[aria-label*="character boxes"] rect[fill="transparent"]'

function toPage(rect, size, point) {
  return [rect.x + point[0] * rect.width / size[0], rect.y + point[1] * rect.height / size[1]]
}

/** Press Draw box, then drag with the mouse: the SVG reports the box when the pointer is released. */
async function dragBox(page, card, size, box) {
  await card.scrollIntoViewIfNeeded()
  const button = card.getByRole('button', { name: 'Draw box' })
  await button.waitFor({ state: 'visible', timeout: WAIT_BOX })
  await poll(() => button.isEnabled(), WAIT_BOX, 'the app is still saving, so this frame cannot be drawn on')
  await button.click()
  const rect = await layer(card).boundingBox()
  const [x1, y1] = toPage(rect, size, [Math.min(box[0] + 3, size[0]), Math.min(box[1] + 3, size[1])])
  const [x2, y2] = toPage(rect, size, [Math.max(box[2] - 3, 0), Math.max(box[3] - 3, 0)])
  const saved = page.waitForResponse(response => response.url().includes('/target-runs/') && response.url().includes('/boxes'), { timeout: 20000 })
  await page.mouse.move(x1, y1)
  await page.mouse.down()
  await page.mouse.move(x1, y2, { steps: 4 })
  await page.mouse.move(x2, y2, { steps: 4 })
  await page.mouse.up()
  const answer = await saved
  if (!answer.ok()) return { ok: false, why: await safeText(answer) }
  return { ok: true }
}
/**
 * Click Accept or Promote on one row of a frame card's Suggested changes list. `nth` is the
 * candidate's place in the run's own list for that frame.
 */
async function clickSuggestion(page, screen, nth, candidate) {
  const label = candidate.suggested_slot != null ? 'Accept' : 'Promote'
  const row = screen.locator('ul[aria-label="Suggested changes"] li').nth(nth)
  if (!(await row.locator(`button:has-text("${label}")`).count())) return { ok: false, why: `that crop offers no ${label} button (the colour may be full or the cut locked)` }
  const wait = page.waitForResponse(response => response.url().includes('/candidates/') && response.method() === 'POST', { timeout: WAIT_BOX })
  await row.locator(`button:has-text("${label}")`).click()
  const response = await wait
  return { ok: response.ok(), why: `the app refused the ${label}: ${await safeText(response)}` }
}

/** Double-click on a person: the app finds their box on the CPU, then adds it like a drawn one. */
async function pickBox(page, card, size, point) {
  await card.scrollIntoViewIfNeeded()
  await card.locator(PICK_LAYER).waitFor({ timeout: WAIT_BOX })
  const rect = await layer(card).boundingBox()
  const [x, y] = toPage(rect, size, point)
  const asked = page.waitForResponse(response => response.url().endsWith('/point-detections'), { timeout: WAIT_BOX }).catch(() => null)
  await page.mouse.dblclick(x, y)
  const answer = await asked
  if (!answer) return { found: false, why: 'the app did not ask for a box at that point' }
  if (answer.status() === 503) return { found: false, why: 'the point-detection worker is not running' }
  if (!answer.ok()) return { found: false, why: await answer.text() }
  const body = await answer.json()
  if (!body?.box) return { found: false, why: 'nothing is there' }
  return { found: true, box: body.box }
}

/** Move a box that landed in another colour, with the screen's own "Person for the box" select. */
async function reslot(page, card, frameIndex, fromSlot, slot) {
  const selects = card.locator('select[aria-label^="Person for the box in frame"]')
  let nth = -1
  for (let i = 0; i < await selects.count(); i++) if (await selects.nth(i).inputValue() === String(fromSlot)) { nth = i; break }
  if (nth < 0) throw new Error(`frame ${frameIndex} shows no box in colour ${fromSlot + 1}`)
  await selects.nth(nth).selectOption(String(slot))
  await page.waitForResponse(response => response.url().includes('/target-detections/') && response.url().endsWith('/slot') && response.ok(), { timeout: 15000 })
}

/** The character select for one colour on this screen. */
async function setName(page, screen, slot, name) {
  const select = screen.getByLabel(`Character for colour ${slot + 1}`)
  await select.waitFor({ state: 'visible', timeout: 10000 })
  await select.selectOption({ label: name })
  await page.waitForResponse(response => response.url().includes('/target-slots/') && response.ok(), { timeout: 15000 })
}

async function markReviewed(page, card) {
  const button = card.getByRole('button', { name: 'Mark reviewed' })
  if (!await button.count()) return false
  await button.click()
  await page.waitForResponse(response => response.url().includes('/reviewed') && response.ok(), { timeout: 15000 })
  return true
}

/** Read the run again and find a box you added: the API is the ground truth the screen shows. */
async function boxOn(url, cutId, ordinal, slot) {
  const response = await api(url, { path: `/cuts/${cutId}/target-detections` })
  if (!response.ok) return null
  const run = response.value.run
  if (!run) return null
  const boxes = run.detections.filter(item => (ordinal === undefined || item.frame_ordinal === ordinal) && (slot === undefined || item.slot === slot))
  return { boxes, run, slot: run.slots.find(item => item.slot === slot) }
}

function reportLine(row) {
  const mark = row.needs ? 'person' : row.skipped ? 'skip' : row.ok ? 'ok' : 'fail'
  const at = row.frame ? ` frame ${row.frame}` : ''
  console.log(`  ${mark.padEnd(6)}${at.padEnd(10)}${row.text}`)
}

async function inspect(url, episode, numbers) {
  for (const item of await cutIds(url, episode, numbers)) {
    if (!item.cut) { console.log(`cut ${item.number}: does not exist (${item.total} active cuts)`); continue }
    const state = await loadState(url, episode, item.cut.id)
    const cast = firstScreen(state)
    console.log(`\ncut ${item.number} (${item.cut.id.slice(0, 8)}) ${state.selection?.revision ? `${state.list.length} frames` : 'no keyframes'}` +
      `${state.run ? ` · run ${state.run.id.slice(0, 8)}${state.run.stale ? ' STALE' : ''}${state.job?.status ? ` · job ${state.job.status}` : ''}` : ' · no run'}`)
    console.log(`  first screen: ${cast.length ? cast.map(slot => `colour ${slot.slot + 1} ${slotName(state, slot.slot)}`).join(', ') : 'nobody (no boxes on the opening frame)'}`)
    for (const frame of state.list) {
      const people = frame.boxes.map(box => `${box.slot + 1}${box.source === 'manual' ? '*' : ''}`).join(' ') || '-'
      console.log(`  F${frame.index}${frame.mirror ? ' (opening)' : ''}  ord ${frame.ordinal}  colours ${people}${frame.candidates.length ? `  ${frame.candidates.length} candidate(s)` : ''}`)
    }
  }
}

async function draw(url, episode, numbers, opts) {
  const rows = []
  const work = []
  for (const item of await cutIds(url, episode, numbers)) {
    if (!item.cut) { rows.push({ ok: false, text: `cut ${item.number} does not exist (${item.total} active cuts)` }); continue }
    const state = await loadState(url, episode, item.cut.id)
    const actions = plan(state, opts.character, opts.framesSet, opts.mode, !!opts['allow-unmatched'])
    const character = actions.find(action => action.kind === 'character')
    console.log(`\ncut ${item.number} (${item.cut.id.slice(0, 8)}): ${character ? `the first screen shows ${character.name} (colour ${character.slot + 1})` : 'no character to work on'}`)
    for (const action of actions) {
      if (action.kind === 'character') { console.log(`  keep colour ${action.slot + 1} as ${action.name}${state.run.slots.find(s => s.slot === action.slot)?.character_name ? ' (already named)' : ''}`); continue }
      if (action.kind === 'drag') { console.log(`  drag ${JSON.stringify(action.box)} from ${action.from}`); continue }
      if (action.kind === 'pick') { console.log(`  double-click ${JSON.stringify(action.point)} (${action.from}) so the detector finds their box`); continue }
      if (action.kind === 'promote') { console.log(`  accept the crop ${JSON.stringify(action.candidate.box)} (${action.from}) for colour ${action.slot + 1}`); continue }
      const needs = action.kind === 'person' || (action.kind === 'blocked' && !action.frame)
      console.log(`  ${needs ? 'a person is needed' : 'skipped'}${action.frame ? ` on F${action.frame}` : ''}: ${action.why}`)
      rows.push({ ok: !needs, needs, skipped: action.kind === 'blocked' && !!action.frame, frame: action.frame, text: action.why })
    }
    const doable = actions.filter(action => action.kind === 'pick' || action.kind === 'drag' || action.kind === 'promote' || action.kind === 'character')
    if (doable.length) work.push({ item, state, actions: doable })
  }
  if (opts['dry-run']) {
    console.log(`\ndry run: ${work.length ? 'nothing was clicked' : 'nothing to click'}`)
    return rows.some(row => row.needs) ? 2 : 0
  }
  if (!work.length) return rows.some(row => row.needs) ? 2 : 0
  const { browser, context } = await openBrowser(opts)
  const page = await context.newPage()
  try {
    for (const job of work) {
      const cutId = job.item.cut.id
      if (opts.detect) await detect(page, url, episode, job.item, job.state)
      const { screen } = await openScreen(page, url, episode, job.item.number)
      await ready(screen, job.state.list)
      for (const action of job.actions) await runAction(page, screen, url, cutId, job.state, action, opts, rows)
    }
  } finally {
    await page.close()
    if (!opts.cdp) await browser.close()
  }
  console.log('\nresult:')
  for (const row of rows) reportLine(row)
  return rows.some(row => row.needs) ? 2 : rows.some(row => !row.ok && !row.needs) ? 1 : 0
}

async function detect(page, url, episode, item, state) {
  if (state.run && !state.run.stale && !state.job?.status) return
  await openScreen(page, url, episode, item.number)
  await page.locator('section[aria-label="Target frames"]')
    .getByRole('button', { name: /^(Detect characters|Detect again|Evaluate and detect)$/ }).click()
  const until = Date.now() + WAIT_JOB
  while (Date.now() < until) {
    const now = await loadState(url, episode, item.cut.id)
    if (now.run && !now.job?.status && now.selection?.revision === state.selection?.revision) return
    await new Promise(done => setTimeout(done, 5000))
  }
  throw new Error(`the detection job for cut ${item.number} did not finish`)
}

async function runAction(page, screen, url, cutId, state, action, opts, rows) {
  const frame = action.frame ? state.list.find(item => item.index === action.frame) : null
  if (action.kind === 'character') {
    const slot = state.run.slots.find(item => item.slot === action.slot)
    if (slot?.character_name === action.name) { rows.push({ ok: true, text: `cut: colour ${action.slot + 1} already ${action.name}` }); return }
    await setName(page, screen, action.slot, action.name)
    const named = await poll(() => boxOn(url, cutId, undefined, action.slot)?.slot?.character_name === action.name, 20000,
      `colour ${action.slot + 1} did not take ${action.name}`).catch(() => false)
    rows.push({ ok: !!named, text: `cut: colour ${action.slot + 1} is now ${action.name}` })
    return
  }
  const before = new Set((await boxOn(url, cutId, frame.ordinal))?.boxes?.map(box => box.id) ?? [])
  if (action.kind === 'promote') {
    const nth = state.list[action.frame - 1].candidates.findIndex(item => item.id === action.candidate.id)
    const answer = await clickSuggestion(page, frameScreen(screen, frame.ordinal), nth, action.candidate)
    if (!answer.ok) { rows.push({ ok: false, needs: true, frame: action.frame, text: `${slotName(state, action.slot)} on F${action.frame}: ${answer.why}` }); return }
  } else if (action.kind === 'drag') {
    const dragged = await dragBox(page, frameScreen(screen, frame.ordinal), frame.size, action.box)
    if (!dragged.ok) { rows.push({ ok: false, needs: true, frame: action.frame, text: `${slotName(state, action.slot)} on F${action.frame}: ${dragged.why}` }); return }
  } else {
    const found = await pickBox(page, frameScreen(screen, frame.ordinal), frame.size, action.point)
    if (!found.found) { rows.push({ ok: false, needs: true, frame: action.frame, text: `${slotName(state, action.slot)} on F${action.frame}: ${found.why}, so no box was drawn` }); return }
  }
  const landed = await poll(async () => {
    const read = await boxOn(url, cutId, frame.ordinal)
    if (!read) return null
    const fresh = read.boxes.filter(box => !before.has(box.id))
    return fresh.length ? { box: fresh[0], read } : null
  }, 20000, `no new box appeared on frame ${action.frame}`)
  let slotOfBox = landed.box.slot
  if (slotOfBox !== action.slot) {
    await reslot(page, frameScreen(screen, frame.ordinal), action.frame, slotOfBox, action.slot)
    slotOfBox = await poll(async () => (await boxOn(url, cutId, frame.ordinal, action.slot))?.boxes?.length ? action.slot : null, 15000, 'the box did not move to the colour')
  }
  const fixed = await boxOn(url, cutId, frame.ordinal, action.slot)
  if (opts['mark-reviewed'] && fixed?.boxes?.length) await markReviewed(page, frameScreen(screen, frame.ordinal))
  rows.push({ ok: !!fixed?.boxes?.length, frame: action.frame,
    text: `${slotName(state, action.slot)} on F${action.frame}: box ${JSON.stringify(fixed?.boxes?.[0]?.box ?? landed.box.box)}${slotOfBox === action.slot ? ` in colour ${action.slot + 1}` : ` but it stayed in colour ${slotOfBox + 1}`}${action.kind === 'drag' ? ' (carried: check it is really them)' : ''}` })
}

async function name(url, episode, numbers, opts) {
  if (!opts.character) fail('name needs --character')
  if (opts.slot === undefined) fail('name needs --slot (0 is colour 1)')
  const slot = Number(opts.slot)
  if (!Number.isInteger(slot) || slot < 0) fail('--slot must be a number, 0 is colour 1')
  const { browser, context } = await openBrowser(opts)
  const page = await context.newPage()
  const rows = []
  try {
    for (const item of await cutIds(url, episode, numbers)) {
      if (!item.cut) { rows.push({ ok: false, text: `cut ${item.number} does not exist` }); continue }
      const state = await loadState(url, episode, item.cut.id)
      const target = state.run?.slots.find(item => item.slot === slot)
      if (!target) { rows.push({ ok: false, text: `cut ${item.number}: colour ${slot + 1} does not exist yet` }); continue }
      const onFirst = firstScreen(state).some(item => item.slot === slot)
      if (!onFirst && !opts.force) {
        rows.push({ ok: false, needs: true, text: `cut ${item.number}: colour ${slot + 1} is not on the first screen (${firstScreen(state).map(s => slotName(state, s.slot)).join(', ')}); --force to override the rule` })
        continue
      }
      await openScreen(page, url, episode, item.number)
      const screen = page.locator('section[aria-label="Target frames"]')
      await screen.waitFor({ timeout: 10000 })
      const from = target.character_name || 'unnamed'
      let named = target.character_name === opts.character
      if (!named) {
        try {
          await setName(page, screen, slot, opts.character)
          named = await poll(() => boxOn(url, item.cut.id, undefined, slot)?.slot?.character_name === opts.character, 20000,
            `colour ${slot + 1} did not take ${opts.character}`).catch(() => false)
        } catch (error) {
          rows.push({ ok: false, text: `cut ${item.number}: ${String(error.message).slice(0, 200)}` }); continue
        }
      }
      rows.push({ ok: named, text: `cut ${item.number}: colour ${slot + 1} ${from} → ${opts.character}${onFirst ? '' : ' (forced)'}` })
    }
  } finally {
    await page.close()
    if (!opts.cdp) await browser.close()
  }
  console.log('\nresult:')
  for (const row of rows) reportLine(row)
  return rows.some(row => row.needs) ? 2 : rows.some(row => !row.ok && !row.needs) ? 1 : 0
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  const command = opts.extra?.shift()
  if (!['inspect', 'draw', 'name'].includes(command)) { usage(); process.exit(1) }
  const range = opts.cuts[1] > opts.cuts[0] ? Array.from({ length: opts.cuts[1] - opts.cuts[0] + 1 }, (unused, index) => opts.cuts[0] + index) : [opts.cuts[0]]
  let code = 1
  try {
    if (command === 'inspect') code = await inspect(opts.url, opts.episode, range)
    else if (command === 'draw') code = await draw(opts.url, opts.episode, range, opts)
    else code = await name(opts.url, opts.episode, range, opts)
  } catch (error) {
    console.error(`target-frames: ${error?.message ?? error}`)
    code = 1
  }
  process.exit(code)
}

main()
