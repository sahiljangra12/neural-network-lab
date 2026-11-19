// Diverging colour scale: orange (−1) → neutral (0) → blue (+1).
const NEG = [204, 54, 34], POS = [228, 232, 236];
let MID = [232, 234, 235];
let LUT = [];

export function rebuildColormap() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  MID = [12, 12, 12];
  LUT = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * 2 - 1, from = t < 0 ? NEG : POS, s = Math.abs(t);
    LUT.push([0, 1, 2].map(k => Math.round(MID[k] + (from[k] - MID[k]) * s)));
  }
}

export function colormap(v) {
  if (!LUT.length) rebuildColormap();
  const i = Math.max(0, Math.min(255, Math.round((v + 1) / 2 * 255)));
  return LUT[i];
}

export function rgb(v) { const c = colormap(v); return `rgb(${c[0]},${c[1]},${c[2]})`; }
