// Diverging colour scale, theme-aware: red (−1) → neutral (0) → ink (+1).
// Ink is near-black in the light theme and near-white in the dark theme.
const T = {
  light: { neg: [208, 23, 10], mid: [238, 241, 244], pos: [11, 11, 11], paper: [255, 255, 255] },
  dark: { neg: [255, 64, 46], mid: [28, 33, 40], pos: [240, 243, 246], paper: [14, 18, 22] },
};
let LUT = [];
export const isDark = () => document.documentElement.getAttribute('data-theme') !== 'light';
const cur = () => T[isDark() ? 'dark' : 'light'];

export function rebuildColormap() {
  const { neg, mid, pos } = cur();
  LUT = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * 2 - 1, from = t < 0 ? neg : pos, s = Math.abs(t);
    LUT.push([0, 1, 2].map(k => Math.round(mid[k] + (from[k] - mid[k]) * s)));
  }
}
export function colormap(v) {
  if (!LUT.length) rebuildColormap();
  return LUT[Math.max(0, Math.min(255, Math.round((v + 1) / 2 * 255)))];
}
export function rgb(v) { const c = colormap(v); return `rgb(${c[0]},${c[1]},${c[2]})`; }
// Ink colour at a given opacity, the panel ("paper") colour, and a paper→ink grey for v in [0, 1].
export const ink = (a = 1) => { const p = cur().pos; return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };
export const paper = () => { const p = cur().paper; return `rgb(${p[0]},${p[1]},${p[2]})`; };
export const paperRGB = () => cur().paper;
export const shadeRGB = v => { const { paper: a, pos: b } = cur(), t = Math.max(0, Math.min(1, v)); return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * (0.03 + 0.97 * t))); };
export const posName = () => (isDark() ? 'white' : 'black');
