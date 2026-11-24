export type ShortcutItem = { keys: string[]; does: string; when?: string }
export type ShortcutGroup = { title: string; items: ShortcutItem[] }
export type ShortcutSheet = { title: string; groups: ShortcutGroup[] }

const navigation: ShortcutGroup = { title: 'Cut analysis', items: [
  { keys: ['[', ']'], does: 'Previous / next cut', when: 'outside a text field' },
  { keys: ['Esc'], does: 'Close analysis' },
] }
const sheets: Record<string, ShortcutSheet> = {
  timeline: { title: 'Timeline', groups: [
    { title: 'Playback and editing', items: [
      { keys: ['Space'], does: 'Play / pause' }, { keys: ['S'], does: 'Split the cut at the playhead' },
      { keys: ['Ctrl+Z'], does: 'Undo cut edit' }, { keys: ['Ctrl+Shift+Z', 'Ctrl+Y'], does: 'Redo cut edit' },
      { keys: ['[', ']'], does: 'Previous / next cut' }, { keys: ['N'], does: 'Next cut needing attention or matching filters' },
      { keys: ['F'], does: 'Open filters' }, { keys: ['Esc'], does: 'Clear the cut range' },
      { keys: ['Ctrl+S'], does: 'Save cut notes', when: 'in the cut inspector' },
    ] },
    { title: 'Mouse', items: [
      { keys: ['Wheel'], does: 'Zoom timeline' }, { keys: ['Middle-drag'], does: 'Move timeline' },
      { keys: ['Drag'], does: 'Seek', when: 'on the waveform' }, { keys: ['Shift+Drag'], does: 'Select audio to transcribe again', when: 'on the waveform' },
      { keys: ['Double-click'], does: 'Open cut analysis' }, { keys: ['Shift+Click'], does: 'Select a range of cuts' },
      { keys: ['Drag'], does: 'Resize timeline', when: 'on its top edge' },
    ] },
  ] },
  'analysis:results': { title: 'Cut analysis · Results', groups: [navigation] },
  'analysis:assistant': { title: 'Cut analysis · Assistant', groups: [{ title: 'Assistant', items: [
    { keys: ['Enter'], does: 'Send message', when: 'in the message field' }, { keys: ['Shift+Enter'], does: 'New line', when: 'in the message field' },
  ] }, navigation] },
  'analysis:characters': { title: 'Cut analysis · Characters', groups: [{ title: 'First-frame boxes', items: [
    { keys: ['A'], does: 'Confirm automatic label', when: 'hovering a box or person card' },
    { keys: ['D'], does: 'Remove a box or start drawing', when: 'hovering a frame or person card' },
    { keys: ['@'], does: 'Start / finish a box', when: 'hovering a frame' },
    { keys: ['Esc'], does: 'Cancel drawing', when: 'drawing a box' },
    { keys: ['Double-click'], does: 'Find a missed person', when: 'on the frame' },
  ] }, navigation] },
  'analysis:targets': { title: 'Cut analysis · Target frames', groups: [{ title: 'Target frames', items: [
    { keys: ['A'], does: 'Mark frame reviewed', when: 'hovering a frame that needs review' },
    { keys: ['D'], does: 'Remove a box or start drawing', when: 'hovering a frame' },
    { keys: ['Double-click'], does: 'Find a person', when: 'on a frame' },
  ] }, navigation] },
  inbox: { title: 'Autopilot inbox', groups: [{ title: 'Questions', items: [
    { keys: ['J', '↓'], does: 'Next question' }, { keys: ['K', '↑'], does: 'Previous question' },
    { keys: ['Enter'], does: 'Accept proposed answer' }, { keys: ['1–9'], does: 'Choose an answer' },
    { keys: ['O'], does: 'Open cut' }, { keys: ['Esc'], does: 'Close inbox' },
  ] }] },
  'inbox:other': { title: 'Autopilot inbox', groups: [{ title: 'Inbox', items: [{ keys: ['Esc'], does: 'Close inbox' }] }] },
  'sequence:dialogue': { title: 'Sequence · Dialogue', groups: [{ title: 'Dialogue', items: [
    { keys: ['J', '↓'], does: 'Next line', when: 'focus in dialogue' }, { keys: ['K', '↑'], does: 'Previous line', when: 'focus in dialogue' },
    { keys: ['Space'], does: 'Play selected line', when: 'focus in dialogue' }, { keys: ['Esc'], does: 'Close sequence' },
  ] }] },
  sequence: { title: 'Sequence', groups: [{ title: 'Sequence', items: [{ keys: ['Esc'], does: 'Close sequence' }] }] },
  dialog: { title: 'Dialog', groups: [] },
}
export function sheetFor(view: string): ShortcutSheet {
  const sheet = sheets[view] ?? sheets.dialog
  return { ...sheet, groups: [...sheet.groups, { title: 'Everywhere', items: [{ keys: ['?'], does: 'Show / hide shortcuts', when: 'outside a text field' }] }] }
}
