/** One colour per person box. The first-frame editor colours a box by its position; the target frames colour a slot by its number, so the same colour is the same person. */
export const BOX_COLORS = ['#ff6b8b', '#4cc9f0', '#ffd166', '#7bdcb2', '#c3aeff', '#ff9f5a', '#8ecae6', '#f28482']

export const boxColor = (index: number) => BOX_COLORS[index % BOX_COLORS.length]
