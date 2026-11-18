// Dataset generators. Coordinates live in [-6, 6]; class labels are +1 / -1,
// regression targets are in [-1, 1].

export function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalFrom(r) {
  return () => {
    let u = 0;
    while (!u) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
}

const U = (r, a, b) => a + (b - a) * r();

function circle(n, noise, r) {
  const pts = [], R = 5;
  for (let i = 0; i < n; i++) {
    const inside = i < n / 2;
    const rad = inside ? U(r, 0, R * 0.5) : U(r, R * 0.7, R);
    const ang = U(r, 0, 2 * Math.PI);
    let x = rad * Math.sin(ang), y = rad * Math.cos(ang);
    x += U(r, -R, R) * noise; y += U(r, -R, R) * noise;
    pts.push({ x, y, label: inside ? 1 : -1 });
  }
  return pts;
}

function xor(n, noise, r) {
  const pts = [], pad = 0.3;
  for (let i = 0; i < n; i++) {
    let x = U(r, -5, 5), y = U(r, -5, 5);
    x += x > 0 ? pad : -pad; y += y > 0 ? pad : -pad;
    const label = x * y >= 0 ? 1 : -1;
    x += U(r, -5, 5) * noise; y += U(r, -5, 5) * noise;
    pts.push({ x, y, label });
  }
  return pts;
}

function gauss(n, noise, r) {
  const N = normalFrom(r), sd = Math.sqrt(0.5 + noise * 7);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const pos = i < n / 2, c = pos ? 2 : -2;
    pts.push({ x: c + N() * sd, y: c + N() * sd, label: pos ? 1 : -1 });
  }
  return pts;
}

function spiral(n, noise, r) {
  const pts = [], half = n / 2;
  for (const [delta, label] of [[0, 1], [Math.PI, -1]]) {
    for (let i = 0; i < half; i++) {
      const rad = (i / half) * 5;
      const t = (1.75 * i / half) * 2 * Math.PI + delta;
      pts.push({
        x: rad * Math.sin(t) + U(r, -1, 1) * noise * 5,
        y: rad * Math.cos(t) + U(r, -1, 1) * noise * 5,
        label,
      });
    }
  }
  return pts;
}

function moons(n, noise, r) {
  const N = normalFrom(r), pts = [];
  for (let i = 0; i < n; i++) {
    const top = i < n / 2, t = U(r, 0, Math.PI);
    let px = top ? Math.cos(t) : 1 - Math.cos(t);
    let py = top ? Math.sin(t) : 0.5 - Math.sin(t);
    pts.push({
      x: (px - 0.5) * 3.6 + N() * noise * 2.2,
      y: (py - 0.25) * 3.6 + N() * noise * 2.2,
      label: top ? 1 : -1,
    });
  }
  return pts;
}

function checker(n, noise, r) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = U(r, -6, 6), y = U(r, -6, 6);
    const label = (Math.floor((x + 6) / 4) + Math.floor((y + 6) / 4)) % 2 === 0 ? 1 : -1;
    pts.push({ x: x + U(r, -3, 3) * noise, y: y + U(r, -3, 3) * noise, label });
  }
  return pts;
}

function plane(n, noise, r) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = U(r, -6, 6), y = U(r, -6, 6);
    const label = (x + y) / 12;
    pts.push({ x: x + U(r, -6, 6) * noise, y: y + U(r, -6, 6) * noise, label });
  }
  return pts;
}

function multigauss(n, noise, r) {
  const bumps = [[-4, 2.5, 1], [0, 2.5, -1], [4, 2.5, 1], [-4, -2.5, -1], [0, -2.5, 1], [4, -2.5, -1]];
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = U(r, -6, 6), y = U(r, -6, 6);
    let v = 0;
    for (const [cx, cy, s] of bumps) v += s * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 3);
    v = Math.max(-1, Math.min(1, v));
    pts.push({ x: x + U(r, -6, 6) * noise, y: y + U(r, -6, 6) * noise, label: v });
  }
  return pts;
}

export const CLASS_DATASETS = {
  circle: { name: 'Circle', gen: circle },
  xor: { name: 'Exclusive or', gen: xor },
  gauss: { name: 'Gaussian', gen: gauss },
  spiral: { name: 'Spiral', gen: spiral },
  moons: { name: 'Two moons', gen: moons },
  checker: { name: 'Checkerboard', gen: checker },
};

export const REG_DATASETS = {
  plane: { name: 'Plane', gen: plane },
  multigauss: { name: 'Multi-Gaussian', gen: multigauss },
};

// Input features the network can use.
export const FEATURES = [
  { id: 'x1', label: 'X₁', f: (x, y) => x, help: 'Horizontal position' },
  { id: 'x2', label: 'X₂', f: (x, y) => y, help: 'Vertical position' },
  { id: 'x1sq', label: 'X₁²', f: (x, y) => x * x, help: 'Squared: lets a linear model bend' },
  { id: 'x2sq', label: 'X₂²', f: (x, y) => y * y, help: 'Squared: lets a linear model bend' },
  { id: 'x1x2', label: 'X₁X₂', f: (x, y) => x * y, help: 'Interaction term (perfect for XOR)' },
  { id: 'sinx1', label: 'sin(X₁)', f: (x, y) => Math.sin(x), help: 'Periodic feature' },
  { id: 'sinx2', label: 'sin(X₂)', f: (x, y) => Math.sin(y), help: 'Periodic feature' },
];

export function makeData(problem, name, n, noise, seed) {
  const r = rng(seed);
  const table = problem === 'class' ? CLASS_DATASETS : REG_DATASETS;
  const pts = (table[name] || Object.values(table)[0]).gen(n, noise, r);
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]];
  }
  return pts;
}
