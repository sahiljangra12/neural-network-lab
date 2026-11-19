import { Network, ACTIVATIONS } from './nn.js?v=202611182231';
import { makeData, normalFrom, rng } from './data.js?v=202611182231';
import { colormap, rebuildColormap } from './colors.js?v=202611182231';

const $ = id => document.getElementById(id);
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sig = z => 1 / (1 + Math.exp(-z));
rebuildColormap();

function setup(cv) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight || W;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  ctx.font = '11.5px Inter, sans-serif';
  return { ctx, W, H };
}
// Paint a function of (x, y) ∈ [-R, R]² onto a canvas as a heatmap.
const off = document.createElement('canvas');
function heat(ctx, W, H, fn, R = 6, res = 64) {
  off.width = off.height = res;
  const o = off.getContext('2d'), img = o.createImageData(res, res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const x = -R + (i + .5) * 2 * R / res, y = R - (j + .5) * 2 * R / res;
    const c = colormap(clamp(fn(x, y), -1, 1)), k = (j * res + i) * 4;
    img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
  }
  o.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
}
function points(ctx, W, H, pts, R = 6, r = 3) {
  for (const p of pts) {
    const c = colormap(p.label);
    ctx.beginPath(); ctx.arc((p.x + R) / (2 * R) * W, (R - p.y) / (2 * R) * H, r, 0, 7);
    ctx.fillStyle = `rgb(${c})`; ctx.fill(); ctx.strokeStyle = p.label > 0 ? 'rgba(220,220,220,.85)' : 'rgba(0,0,0,.85)'; ctx.lineWidth = 1; ctx.stroke();
  }
}
function pills(el, items, cur, onPick) {
  el.innerHTML = items.map(([k, n]) => `<button class="pill${k === cur ? ' on' : ''}" data-k="${k}">${n}</button>`).join('');
  el.querySelectorAll('.pill').forEach(b => b.onclick = () => { el.querySelectorAll('.pill').forEach(x => x.classList.toggle('on', x === b)); onPick(b.dataset.k); });
}
function trainEpoch(net, data, opts, batch = 10) {
  data.forEach((e, n) => { net.forward(e.input); net.backward(e.label); if ((n + 1) % batch === 0) net.update(opts); });
  net.update(opts);
}
const toEx = pts => pts.map(p => ({ ...p, input: [p.x, p.y] }));

/* =============== 1. NEURON =============== */
const NACT = {
  sigmoid: { n: 'Sigmoid', f: sig, show: a => 2 * a - 1, sym: 'σ' },
  tanh: { n: 'Tanh', f: Math.tanh, show: a => a, sym: 'tanh' },
  relu: { n: 'ReLU', f: z => Math.max(0, z), show: a => Math.min(1, a / 4), sym: 'ReLU' },
  step: { n: 'Step (perceptron)', f: z => (z >= 0 ? 1 : 0), show: a => 2 * a - 1, sym: 'step' },
};
let nAct = 'sigmoid';
pills($('nact'), Object.entries(NACT).map(([k, v]) => [k, v.n]), nAct, k => { nAct = k; drawNeuron(); });
['nw1', 'nw2', 'nb'].forEach(id => $(id).addEventListener('input', drawNeuron));
function drawNeuron() {
  const w1 = +$('nw1').value, w2 = +$('nw2').value, b = +$('nb').value, A = NACT[nAct];
  $('nw1v').textContent = w1.toFixed(2); $('nw2v').textContent = w2.toFixed(2); $('nbv').textContent = b.toFixed(1);
  const sgn = v => (v < 0 ? ' − ' : ' + ') + Math.abs(v).toFixed(2);
  $('nformula').innerHTML = `a = ${A.sym}( <span class="${w1 >= 0 ? 'p' : 'n'}">${w1.toFixed(2)}</span>·x₁${sgn(w2).replace(/([\d.]+)$/, `<span class="${w2 >= 0 ? 'p' : 'n'}">$1</span>`)}·x₂${sgn(b)} )`;
  const { ctx, W, H } = setup($('ncv'));
  heat(ctx, W, H, (x, y) => A.show(A.f(w1 * x + w2 * y + b)), 6, 90);
  // boundary line w1 x + w2 y + b = 0
  const X = x => (x + 6) / 12 * W, Y = y => (6 - y) / 12 * H;
  ctx.setLineDash([6, 5]); ctx.strokeStyle = cssv('--fg'); ctx.lineWidth = 1.6; ctx.beginPath();
  if (Math.abs(w2) > 1e-6) { ctx.moveTo(X(-6), Y((-b - w1 * -6) / w2)); ctx.lineTo(X(6), Y((-b - w1 * 6) / w2)); }
  else if (Math.abs(w1) > 1e-6) { ctx.moveTo(X(-b / w1), Y(-6)); ctx.lineTo(X(-b / w1), Y(6)); }
  ctx.stroke(); ctx.setLineDash([]);
  // weight vector arrow (points toward the positive side)
  const n = Math.hypot(w1, w2);
  if (n > 1e-6) {
    const px = -b * w1 / (n * n), py = -b * w2 / (n * n);
    if (Math.abs(px) < 5.5 && Math.abs(py) < 5.5) {
      const ex = px + w1 / n * 1.6, ey = py + w2 / n * 1.6;
      ctx.strokeStyle = cssv('--fg'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(px), Y(py)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
      const ang = Math.atan2(Y(ey) - Y(py), X(ex) - X(px));
      ctx.beginPath(); ctx.moveTo(X(ex), Y(ey)); ctx.lineTo(X(ex) - 9 * Math.cos(ang - .4), Y(ey) - 9 * Math.sin(ang - .4)); ctx.lineTo(X(ex) - 9 * Math.cos(ang + .4), Y(ey) - 9 * Math.sin(ang + .4)); ctx.closePath(); ctx.fillStyle = cssv('--fg'); ctx.fill();
      ctx.fillText('w', X(ex) + 6, Y(ey) - 4);
    }
  }
  ctx.fillStyle = cssv('--fg'); ctx.fillText('x₁ →', W - 34, H - 8); ctx.fillText('↑ x₂', 6, 14);
}
$('ncv').addEventListener('mousemove', e => {
  const r = e.currentTarget.getBoundingClientRect(), x = -6 + (e.clientX - r.left) / r.width * 12, y = 6 - (e.clientY - r.top) / r.height * 12;
  const z = +$('nw1').value * x + +$('nw2').value * y + +$('nb').value, a = NACT[nAct].f(z);
  $('nprobe').innerHTML = `at (${x.toFixed(1)}, ${y.toFixed(1)}): z = ${z.toFixed(2)} → a = <b>${a.toFixed(3)}</b>`;
});

/* =============== 2. ACTIVATIONS =============== */
const AF = {
  sigmoid: { n: 'Sigmoid', f: sig, d: z => sig(z) * (1 - sig(z)), formula: 'σ(z) = 1 / (1 + e⁻ᶻ)\nσ′(z) = σ(z)(1 − σ(z))',
    facts: [['Range', '(0, 1)'], ['Max slope', '0.25 at z = 0'], ['Zero-centred', 'No']],
    note: 'Turns any number into a probability, which is why it sits on the <b>output</b> of a binary classifier (giving logistic regression). In hidden layers its slope is at most 0.25 and almost 0 for large |z|, so in deep stacks the gradient <b>vanishes</b>.' },
  tanh: { n: 'Tanh', f: Math.tanh, d: z => 1 - Math.tanh(z) ** 2, formula: 'tanh(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)\ntanh′(z) = 1 − tanh²(z)',
    facts: [['Range', '(−1, 1)'], ['Max slope', '1 at z = 0'], ['Zero-centred', 'Yes']],
    note: 'A rescaled sigmoid: tanh(z) = 2σ(2z) − 1. Being zero-centred and having a slope of 1 at the origin usually makes it train better than sigmoid in hidden layers. It still <b>saturates</b> for large |z|.' },
  relu: { n: 'ReLU', f: z => Math.max(0, z), d: z => (z > 0 ? 1 : 0), formula: 'ReLU(z) = max(0, z)\nReLU′(z) = 1 if z > 0 else 0',
    facts: [['Range', '[0, ∞)'], ['Slope', '0 or 1'], ['Cost', 'Very cheap']],
    note: 'The default in modern deep learning. Its slope is exactly 1 for positive inputs, so gradients pass through many layers without shrinking. The risk is the <b>dead ReLU</b>: a neuron whose input is always negative has slope 0 and stops learning for good.' },
  leaky: { n: 'Leaky ReLU', f: z => (z > 0 ? z : 0.05 * z), d: z => (z > 0 ? 1 : 0.05), formula: 'f(z) = max(αz, z), α ≈ 0.01\nf′ = 1 or α',
    facts: [['Range', '(−∞, ∞)'], ['Slope', 'α or 1'], ['Dead neurons', 'Avoided']],
    note: 'A small slope α for negative inputs keeps a trickle of gradient flowing, so neurons cannot die. (Drawn here with α = 0.05 so the slope is visible.)' },
  linear: { n: 'Linear', f: z => z, d: () => 1, formula: 'f(z) = z\nf′(z) = 1',
    facts: [['Range', '(−∞, ∞)'], ['Slope', '1'], ['Bends?', 'Never']],
    note: 'Correct for the <b>output of a regression</b> network, where the answer can be any number. In hidden layers it adds nothing: a stack of linear layers collapses to a single linear map.' },
  step: { n: 'Step', f: z => (z >= 0 ? 1 : 0), d: () => 0, formula: 'step(z) = 1 if z ≥ 0 else 0\nslope = 0 (undefined at 0)',
    facts: [['Range', '{0, 1}'], ['Slope', '0 everywhere'], ['Used in', 'Perceptron (1958)']],
    note: 'The original perceptron. It gives a hard yes/no, but its slope is 0 everywhere, so <b>gradient descent cannot train it</b>: there is no signal saying which way to move. Smooth activations were the fix that made backpropagation possible.' },
};
let curAF = 'tanh';
pills($('actpills'), Object.entries(AF).map(([k, v]) => [k, v.n]), curAF, k => { curAF = k; drawAct(); });
function drawAct() {
  const A = AF[curAF], { ctx, W, H } = setup($('actcv'));
  const X = z => (z + 5) / 10 * W, Y = v => H - (v + 1.5) / 4 * H;
  ctx.strokeStyle = cssv('--line'); ctx.lineWidth = 1;
  for (let v = -1; v <= 2; v++) { ctx.beginPath(); ctx.moveTo(0, Y(v)); ctx.lineTo(W, Y(v)); ctx.stroke(); ctx.fillStyle = cssv('--faint'); ctx.fillText(v, 4, Y(v) - 3); }
  ctx.strokeStyle = cssv('--line2'); ctx.beginPath(); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();
  const curve = (fn, col, w) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); for (let i = 0; i <= 400; i++) { const z = -5 + 10 * i / 400; const v = clamp(fn(z), -1.6, 2.6); i ? ctx.lineTo(X(z), Y(v)) : ctx.moveTo(X(z), Y(v)); } ctx.stroke(); };
  curve(A.d, cssv('--neg'), 2); curve(A.f, cssv('--pos'), 2.6);
  ctx.fillStyle = cssv('--faint'); ctx.fillText('z →', W - 26, Y(0) - 5);
  $('actformula').textContent = A.formula;
  $('actfacts').innerHTML = A.facts.map(([a, b]) => `<div><span>${a}</span>${b}</div>`).join('');
  $('actnote').innerHTML = A.note;
}

/* =============== 3. DEPTH =============== */
let depthDS = 'circle', depthNets = [], depthData = [], depthRun = false, depthEp = 0;
const DEPTH_CFG = [['No hidden layer', [], 'tanh'], ['4 tanh neurons', [4], 'tanh'], ['4 linear neurons', [4], 'linear']];
$('depthgrid').innerHTML = DEPTH_CFG.map((c, i) => `<div><div class="sub" style="margin-bottom:4px"><b>${c[0]}</b></div><canvas class="plot" id="dcv${i}" style="aspect-ratio:1"></canvas><div class="probe" id="dst${i}"></div></div>`).join('');
pills($('depthds'), [['circle', 'Circle'], ['xor', 'XOR'], ['moons', 'Two moons'], ['gauss', 'Gaussian (easy)']], depthDS, k => { depthDS = k; resetDepth(); });
function resetDepth() {
  depthData = toEx(makeData('class', depthDS, 240, 0.02, 9));
  depthNets = DEPTH_CFG.map((c, i) => new Network([2, ...c[1], 1], c[2], 'class', 40 + i));
  depthEp = 0; depthRun = false; $('depthgo').textContent = '▶ Train all three'; drawDepth();
}
function drawDepth() {
  depthNets.forEach((net, i) => {
    const { ctx, W, H } = setup($('dcv' + i));
    heat(ctx, W, H, (x, y) => 2 * net.forward([x, y]) - 1, 6, 48);
    points(ctx, W, H, depthData, 6, 2.4);
    const acc = net.accuracy(depthData);
    $('dst' + i).innerHTML = `epoch ${depthEp} · accuracy <b>${(acc * 100).toFixed(0)}%</b>`;
  });
}
$('depthgo').onclick = () => { depthRun = !depthRun; $('depthgo').textContent = depthRun ? '❚❚ Pause' : '▶ Train all three'; };
$('depthreset').onclick = resetDepth;

/* =============== 4. LOSS =============== */
['lp', 'ly'].forEach(id => $(id).addEventListener('input', drawLoss));
function drawLoss() {
  const p = +$('lp').value, yh = +$('ly').value;
  $('lpv').textContent = p.toFixed(2); $('lyv').textContent = yh.toFixed(2);
  let { ctx, W, H } = setup($('lcv1'));
  const X1 = q => q * W, Y1 = v => H - 8 - v / 5 * (H - 16);
  ctx.strokeStyle = cssv('--line'); ctx.beginPath(); ctx.moveTo(0, Y1(0)); ctx.lineTo(W, Y1(0)); ctx.stroke();
  ctx.strokeStyle = cssv('--pos'); ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 1; i <= 200; i++) { const q = i / 200 * 0.995 + 0.005; const v = Math.min(5, -Math.log(q)); i > 1 ? ctx.lineTo(X1(q), Y1(v)) : ctx.moveTo(X1(q), Y1(v)); } ctx.stroke();
  ctx.fillStyle = cssv('--neg'); ctx.beginPath(); ctx.arc(X1(p), Y1(-Math.log(p)), 6, 0, 7); ctx.fill();
  ctx.fillStyle = cssv('--faint'); ctx.fillText('p → 1', W - 36, Y1(0) - 6); ctx.fillText('loss', 4, 12);
  $('lf1').textContent = `L = −log(p) = −log(${p.toFixed(2)}) = ${(-Math.log(p)).toFixed(3)}`;
  ({ ctx, W, H } = setup($('lcv2')));
  const X2 = v => (v + 2) / 4 * W, Y2 = v => H - 8 - v / 4 * (H - 16);
  ctx.strokeStyle = cssv('--line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X2(0), 0); ctx.lineTo(X2(0), H); ctx.stroke();
  ctx.strokeStyle = cssv('--pos'); ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i <= 200; i++) { const v = -2 + 4 * i / 200; i ? ctx.lineTo(X2(v), Y2(v * v)) : ctx.moveTo(X2(v), Y2(v * v)); } ctx.stroke();
  ctx.fillStyle = cssv('--neg'); ctx.beginPath(); ctx.arc(X2(yh), Y2(yh * yh), 6, 0, 7); ctx.fill();
  ctx.fillStyle = cssv('--faint'); ctx.fillText('ŷ', W - 14, H - 12); ctx.fillText('y = 0', X2(0) + 5, 12);
  $('lf2').textContent = `L = (ŷ − y)² = (${yh.toFixed(2)} − 0)² = ${(yh * yh).toFixed(3)}`;
}

/* =============== 5. GRADIENT DESCENT =============== */
const SURF = {
  bowl: { n: 'Round bowl', f: (x, y) => 0.5 * (x * x + y * y), start: [-2.5, 2.2] },
  valley: { n: 'Narrow valley', f: (x, y) => 0.5 * (x * x + 15 * y * y), start: [-2.7, 0.8] },
  two: { n: 'Two minima', f: (x, y) => (x * x - 4) ** 2 / 8 + y * y + 0.35 * x, start: [2.6, 1.8] },
  banana: { n: 'Curved valley', f: (x, y) => 0.15 * (1 - x) ** 2 + 2 * (y - 0.35 * x * x) ** 2, start: [-2.5, -1] },
};
let surf = 'valley', gdOpt = 'sgd', gdRuns = [], gdImg = null, gdStart = SURF.valley.start.slice();
const GDCOL = ['--neg', '--fg', '--mut', '--neg', '--fg'];
pills($('gdsurf'), Object.entries(SURF).map(([k, v]) => [k, v.n]), surf, k => { surf = k; gdStart = SURF[k].start.slice(); gdRuns = []; gdImg = null; launchGD(); });
pills($('gdopt'), [['sgd', 'Gradient descent'], ['momentum', 'Momentum'], ['adam', 'Adam']], gdOpt, k => { gdOpt = k; });
const lrv = () => 10 ** +$('gdlr').value;
$('gdlr').addEventListener('input', () => $('gdlrv').textContent = lrv().toFixed(3)); $('gdlrv').textContent = lrv().toFixed(3);
$('gdnoise').addEventListener('input', () => $('gdnoisev').textContent = (+$('gdnoise').value).toFixed(1)); $('gdnoisev').textContent = '0.0';
const gauss = normalFrom(rng(77));
function grad(f, x, y) { const h = 1e-4; return [(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)]; }
function launchGD() {
  if (gdRuns.length >= 5) gdRuns.shift();
  gdRuns.push({ pts: [gdStart.slice()], x: gdStart[0], y: gdStart[1], m: [0, 0], v: [0, 0], t: 0, opt: gdOpt, lr: lrv(), noise: +$('gdnoise').value, done: false, status: 'running', col: GDCOL[gdRuns.length % GDCOL.length] });
}
function stepGD() {
  const f = SURF[surf].f;
  for (const r of gdRuns) {
    if (r.done) continue;
    let [gx, gy] = grad(f, r.x, r.y);
    const gn = Math.hypot(gx, gy);
    if (r.noise) { gx += r.noise * gauss(); gy += r.noise * gauss(); }
    r.t++;
    if (r.opt === 'momentum') { r.m[0] = 0.9 * r.m[0] + gx; r.m[1] = 0.9 * r.m[1] + gy; r.x -= r.lr * r.m[0]; r.y -= r.lr * r.m[1]; }
    else if (r.opt === 'adam') {
      const b1 = .9, b2 = .999; r.m[0] = b1 * r.m[0] + (1 - b1) * gx; r.m[1] = b1 * r.m[1] + (1 - b1) * gy;
      r.v[0] = b2 * r.v[0] + (1 - b2) * gx * gx; r.v[1] = b2 * r.v[1] + (1 - b2) * gy * gy;
      const c1 = 1 - b1 ** r.t, c2 = 1 - b2 ** r.t;
      r.x -= r.lr * (r.m[0] / c1) / (Math.sqrt(r.v[0] / c2) + 1e-8); r.y -= r.lr * (r.m[1] / c1) / (Math.sqrt(r.v[1] / c2) + 1e-8);
    } else { r.x -= r.lr * gx; r.y -= r.lr * gy; }
    r.pts.push([r.x, r.y]);
    if (!isFinite(r.x) || Math.abs(r.x) > 1e3 || Math.abs(r.y) > 1e3) { r.done = true; r.status = 'diverged'; }
    else if (gn < 2e-3 && !r.noise && r.opt !== 'adam') { r.done = true; r.status = 'converged'; }
    else if (r.t >= 400) { r.done = true; r.status = 'stopped after 400 steps'; }
  }
}
function drawGD() {
  const { ctx, W, H } = setup($('gdcv')), f = SURF[surf].f, R = 3;
  if (!gdImg || gdImg.w !== W) {
    const dpr = window.devicePixelRatio || 1, res = 140, c = document.createElement('canvas');
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    const o = c.getContext('2d'); o.scale(dpr, dpr);
    const v = [], xs = i => -R + i * 2 * R / res, ys = j => R - j * 2 * R / res;
    let min = Infinity;
    for (let j = 0; j <= res; j++) { v.push([]); for (let i = 0; i <= res; i++) { const val = f(xs(i), ys(j)); v[j].push(val); min = Math.min(min, val); } }
    let lo = Infinity, hi = -Infinity;
    for (const row of v) for (let i = 0; i < row.length; i++) { row[i] = Math.log(row[i] - min + 0.03); lo = Math.min(lo, row[i]); hi = Math.max(hi, row[i]); }
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    const base = [6, 6, 6], tint = [150, 165, 185];
    const cw = W / res, ch = H / res;
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      const t = 1 - (v[j][i] - lo) / (hi - lo), m = 0.06 + 0.34 * t;
      o.fillStyle = `rgb(${base.map((b, k) => Math.round(b + (tint[k] - b) * m)).join(',')})`;
      o.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6);
    }
    // marching squares contour lines
    o.strokeStyle = 'rgba(255,255,255,.28)'; o.lineWidth = 1;
    const NL = 16;
    for (let k = 1; k < NL; k++) {
      const L = lo + (hi - lo) * k / NL; o.beginPath();
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
        const a = v[j][i], b = v[j][i + 1], cc = v[j + 1][i + 1], d = v[j + 1][i];
        const pts = [];
        const ed = (p, q, x1, y1, x2, y2) => { if ((p < L) !== (q < L)) { const t = (L - p) / (q - p); pts.push([(x1 + (x2 - x1) * t) * cw, (y1 + (y2 - y1) * t) * ch]); } };
        ed(a, b, i, j, i + 1, j); ed(b, cc, i + 1, j, i + 1, j + 1); ed(d, cc, i, j + 1, i + 1, j + 1); ed(a, d, i, j, i, j + 1);
        if (pts.length >= 2) { o.moveTo(pts[0][0], pts[0][1]); o.lineTo(pts[1][0], pts[1][1]); }
        if (pts.length === 4) { o.moveTo(pts[2][0], pts[2][1]); o.lineTo(pts[3][0], pts[3][1]); }
      }
      o.stroke();
    }
    gdImg = { c, w: W };
  }
  ctx.imageSmoothingEnabled = true; ctx.drawImage(gdImg.c, 0, 0, W, H);
  const X = x => (x + R) / (2 * R) * W, Y = y => (R - y) / (2 * R) * H;
  for (const r of gdRuns) {
    const col = cssv(r.col);
    ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath();
    r.pts.forEach(([x, y], k) => { const px = X(clamp(x, -9, 9)), py = Y(clamp(y, -9, 9)); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.stroke();
    ctx.fillStyle = col; r.pts.forEach(([x, y], k) => { if (k % 1 === 0 && Math.abs(x) < 9) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2, 0, 7); ctx.fill(); } });
    const [lx, ly] = r.pts[r.pts.length - 1]; ctx.beginPath(); ctx.arc(X(clamp(lx, -3, 3)), Y(clamp(ly, -3, 3)), 6, 0, 7); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  ctx.fillStyle = cssv('--fg'); ctx.beginPath(); ctx.arc(X(gdStart[0]), Y(gdStart[1]), 4, 0, 7); ctx.fill();
  const r = gdRuns[gdRuns.length - 1];
  if (r) {
    $('gdstat').textContent = `${r.opt.toUpperCase()}  η = ${r.lr.toFixed(3)}${r.noise ? '  noise ' + r.noise : ''}\nstep ${r.t}   w = (${r.x.toFixed(3)}, ${r.y.toFixed(3)})\nloss ${isFinite(f(r.x, r.y)) ? f(r.x, r.y).toFixed(5) : '∞'}   ${r.status}`;
    let note = '';
    if (r.status === 'diverged') note = 'Diverged: the step is so large it overshoots the steep direction and lands further out each time. On the narrow valley, plain GD needs η below 2/15 ≈ 0.13.';
    else if (surf === 'valley' && r.opt === 'sgd') note = 'See the zig-zag? The valley is 15× steeper across than along, so a step size safe for the steep direction is tiny for the flat one.';
    else if (surf === 'two') note = 'Non-convex: the minimum you reach depends on where you start. The left minimum is deeper (global); the right one is a local trap. Add noise to sometimes jump the ridge.';
    else if (r.opt === 'momentum') note = 'Momentum keeps a running velocity, so it speeds up along consistent directions and damps the zig-zag. It can also overshoot and swirl.';
    else if (r.opt === 'adam') note = 'Adam scales each coordinate\'s step by its recent gradient size, so steep and flat directions get similar progress.';
    $('gdnote').textContent = note;
  }
}
$('gdcv').addEventListener('click', e => {
  const rr = e.currentTarget.getBoundingClientRect();
  gdStart = [-3 + (e.clientX - rr.left) / rr.width * 6, 3 - (e.clientY - rr.top) / rr.height * 6]; launchGD();
});
$('gdgo').onclick = launchGD;
$('gdclear').onclick = () => { gdRuns = []; drawGD(); $('gdstat').textContent = 'Click the map or press Run.'; $('gdnote').textContent = ''; };

/* =============== 6. BACKPROP =============== */
const BP = { W1: [[0.4, -0.6], [0.7, 0.2]], b1: [0.1, -0.1], W2: [0.8, -0.5], b2: 0.05 };
let bpStep = 0;
const STEPS = ['Setup', 'Hidden sums', 'Hidden activations', 'Output & loss', 'Output error', 'Output gradients', 'Error flows back', 'Hidden gradients', 'Update'];
function bpCompute() {
  const x = [+$('bx1').value || 0, +$('bx2').value || 0], y = +$('by').value, lr = +$('blr').value || 0;
  const { W1, b1, W2, b2 } = BP;
  const z = W1.map((w, j) => w[0] * x[0] + w[1] * x[1] + b1[j]), h = z.map(Math.tanh);
  const zo = W2[0] * h[0] + W2[1] * h[1] + b2, p = sig(zo);
  const L = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  const dO = p - y, gW2 = h.map(v => dO * v), gb2 = dO;
  const dH = h.map((v, j) => dO * W2[j] * (1 - v * v));
  const gW1 = W1.map((w, j) => w.map((_, i) => dH[j] * x[i])), gb1 = dH.slice();
  const nW1 = W1.map((w, j) => w.map((v, i) => v - lr * gW1[j][i])), nb1 = b1.map((v, j) => v - lr * gb1[j]);
  const nW2 = W2.map((v, j) => v - lr * gW2[j]), nb2 = b2 - lr * gb2;
  const nz = nW1.map((w, j) => w[0] * x[0] + w[1] * x[1] + nb1[j]), nh = nz.map(Math.tanh), np = sig(nW2[0] * nh[0] + nW2[1] * nh[1] + nb2);
  const nL = -(y * Math.log(np) + (1 - y) * Math.log(1 - np));
  return { x, y, lr, z, h, zo, p, L, dO, gW2, gb2, dH, gW1, gb1, nW1, nb1, nW2, nb2, np, nL };
}
const f3 = v => (v >= 0 ? ' ' : '') + v.toFixed(3);
function bpRender() {
  const c = bpCompute(), s = bpStep;
  const P = { x: [[80, 85], [80, 215]], h: [[330, 85], [330, 215]], o: [[560, 150]] };
  const acc = cssv('--acc'), mut = cssv('--line2'), neg = cssv('--neg');
  const active = {
    1: ['e00', 'e01', 'e10', 'e11', 'h0', 'h1'], 2: ['h0', 'h1'], 3: ['f0', 'f1', 'o', 'L'], 4: ['o', 'L'], 5: ['f0', 'f1', 'o'],
    6: ['f0', 'f1', 'h0', 'h1'], 7: ['e00', 'e01', 'e10', 'e11', 'h0', 'h1'], 8: ['e00', 'e01', 'e10', 'e11', 'f0', 'f1'],
  }[s] || [];
  const on = id => active.includes(id);
  let g = '';
  const edge = (id, a, b, w, grad, newW, labelT) => {
    const col = on(id) ? (s >= 5 && s !== 6 ? neg : acc) : mut;
    g += `<line x1="${a[0] + 26}" y1="${a[1]}" x2="${b[0] - 26}" y2="${b[1]}" stroke="${col}" stroke-width="${on(id) ? 2.6 : 1.4}"/>`;
    const tx = a[0] + 26 + (b[0] - a[0] - 52) * labelT, ty = a[1] + (b[1] - a[1]) * labelT - 7;
    let lab = `w=${w.toFixed(2)}`;
    if (s === 8) lab = `w: ${w.toFixed(2)}→${newW.toFixed(3)}`;
    g += `<text x="${tx}" y="${ty}" text-anchor="middle">${lab}</text>`;
    if (grad !== null) g += `<text x="${tx}" y="${ty + 16}" text-anchor="middle" style="fill:${neg}">∂L/∂w=${grad.toFixed(3)}</text>`;
  };
  for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) edge(`e${j}${i}`, P.x[i], P.h[j], BP.W1[j][i], s >= 7 ? c.gW1[j][i] : null, c.nW1[j][i], i === j ? 0.5 : 0.2);
  for (let j = 0; j < 2; j++) edge(`f${j}`, P.h[j], P.o[0], BP.W2[j], s >= 5 ? c.gW2[j] : null, c.nW2[j], 0.45);
  const node = (id, [x, y], top, mid, bot, botCol) => {
    g += `<circle cx="${x}" cy="${y}" r="26" fill="${cssv('--card2')}" stroke="${on(id) ? acc : mut}" stroke-width="${on(id) ? 2.6 : 1.2}"/>`;
    g += `<text class="lab" x="${x}" y="${y - 34}" text-anchor="middle">${top}</text>`;
    if (mid) g += `<text x="${x}" y="${y + 4}" text-anchor="middle">${mid}</text>`;
    if (bot) g += `<text x="${x}" y="${y + 44}" text-anchor="middle" style="fill:${botCol || cssv('--fg')}">${bot}</text>`;
  };
  node('x0', P.x[0], 'input x₁', c.x[0].toFixed(2)); node('x1', P.x[1], 'input x₂', c.x[1].toFixed(2));
  for (let j = 0; j < 2; j++) node(`h${j}`, P.h[j], `hidden h${j + 1} (b=${BP.b1[j]})`, s >= 2 ? c.h[j].toFixed(3) : s >= 1 ? '' : '?', s === 1 ? `z=${c.z[j].toFixed(3)}` : s >= 6 ? `δ=${c.dH[j].toFixed(3)}` : (s >= 2 ? `z=${c.z[j].toFixed(3)}` : ''), s >= 6 ? neg : null);
  node('o', P.o[0], `output (b=${BP.b2})`, s >= 3 ? 'p=' + c.p.toFixed(3) : '?', s >= 4 ? `δ=p−y=${c.dO.toFixed(3)}` : (s >= 3 ? `z=${c.zo.toFixed(3)}` : ''), s >= 4 ? neg : null);
  if (s >= 3) g += `<text x="635" y="146" text-anchor="middle" style="fill:${on('L') ? acc : cssv('--fg')}">L=</text><text x="635" y="164" text-anchor="middle" style="fill:${on('L') ? acc : cssv('--fg')}">${c.L.toFixed(3)}</text>`;
  $('bpsvg').innerHTML = g;

  const E = {
    0: `<b>The setup.</b> Inputs x = (${c.x[0]}, ${c.x[1]}), target y = ${c.y}. The weights are fixed starting values. We will do <b>one</b> gradient-descent update with η = ${c.lr}: forward pass, then backward pass, then update.`,
    1: `<b>Forward, step 1: each hidden neuron takes a weighted sum.</b><div class="eq">z₁ = ${BP.W1[0][0]}·${c.x[0]} + (${BP.W1[0][1]})·${c.x[1]} + ${BP.b1[0]} = ${c.z[0].toFixed(4)}\nz₂ = ${BP.W1[1][0]}·${c.x[0]} + ${BP.W1[1][1]}·${c.x[1]} + (${BP.b1[1]}) = ${c.z[1].toFixed(4)}</div>`,
    2: `<b>Forward, step 2: squash with tanh.</b> These values are stored, because the backward pass needs them.<div class="eq">h₁ = tanh(${c.z[0].toFixed(4)}) = ${c.h[0].toFixed(4)}\nh₂ = tanh(${c.z[1].toFixed(4)}) = ${c.h[1].toFixed(4)}</div>`,
    3: `<b>Forward, step 3: output neuron (sigmoid), then the loss.</b><div class="eq">zₒ = ${BP.W2[0]}·${c.h[0].toFixed(4)} + (${BP.W2[1]})·${c.h[1].toFixed(4)} + ${BP.b2} = ${c.zo.toFixed(4)}\np  = σ(zₒ) = ${c.p.toFixed(4)}\nL  = −[y·log p + (1−y)·log(1−p)] = ${c.L.toFixed(4)}</div>`,
    4: `<b>Backward, step 1: the error at the output.</b> By the chain rule, ∂L/∂zₒ = ∂L/∂p · ∂p/∂zₒ. For sigmoid with log loss, the terms cancel to a very simple result:<div class="eq">δₒ = ∂L/∂zₒ = p − y = ${c.p.toFixed(4)} − ${c.y} = ${c.dO.toFixed(4)}</div>${c.dO < 0 ? 'Negative: the prediction is too low, so the update should push zₒ up.' : 'Positive: the prediction is too high, so the update should push zₒ down.'}`,
    5: `<b>Backward, step 2: gradients of the output weights.</b> zₒ = w·h + b, so ∂zₒ/∂wⱼ = hⱼ. Each gradient is the error times the input that weight carried.<div class="eq">∂L/∂w₁ = δₒ · h₁ = ${c.dO.toFixed(4)} · ${c.h[0].toFixed(4)} = ${c.gW2[0].toFixed(4)}\n∂L/∂w₂ = δₒ · h₂ = ${c.dO.toFixed(4)} · ${c.h[1].toFixed(4)} = ${c.gW2[1].toFixed(4)}\n∂L/∂bₒ = δₒ = ${c.gb2.toFixed(4)}</div>`,
    6: `<b>Backward, step 3: send the error back to the hidden neurons.</b> Chain rule again: through the weight wⱼ, then through tanh, whose slope is 1 − h².<div class="eq">δ₁ = δₒ · w₁ · (1 − h₁²) = ${c.dO.toFixed(4)} · ${BP.W2[0]} · ${(1 - c.h[0] ** 2).toFixed(4)} = ${c.dH[0].toFixed(4)}\nδ₂ = δₒ · w₂ · (1 − h₂²) = ${c.dO.toFixed(4)} · (${BP.W2[1]}) · ${(1 - c.h[1] ** 2).toFixed(4)} = ${c.dH[1].toFixed(4)}</div>This is the "back" in backpropagation. We reuse δₒ instead of redoing the whole chain for every weight.`,
    7: `<b>Backward, step 4: gradients of the first-layer weights.</b> Same pattern: (error at the neuron) × (input it saw).<div class="eq">∂L/∂w₁₁ = δ₁·x₁ = ${c.gW1[0][0].toFixed(4)}    ∂L/∂w₁₂ = δ₁·x₂ = ${c.gW1[0][1].toFixed(4)}\n∂L/∂w₂₁ = δ₂·x₁ = ${c.gW1[1][0].toFixed(4)}    ∂L/∂w₂₂ = δ₂·x₂ = ${c.gW1[1][1].toFixed(4)}</div>`,
    8: `<b>Update: w ← w − η · ∂L/∂w</b> for every weight and bias at once (η = ${c.lr}). Then check that it worked by running the forward pass again with the new weights:<div class="eq">before:  p = ${c.p.toFixed(4)}   L = ${c.L.toFixed(4)}\nafter:   p = ${c.np.toFixed(4)}   L = ${c.nL.toFixed(4)}   ${c.nL < c.L ? '✓ loss went down' : '✗ loss went up (η too large?)'}</div>Repeat this for every example, over many epochs, and you have trained a neural network.`,
  };
  $('bpexp').innerHTML = E[s];
  $('bpsteps').innerHTML = STEPS.map((t, i) => `<button class="pill${i === s ? ' on' : ''}" data-s="${i}">${i}. ${t}</button>`).join('');
  $('bpsteps').querySelectorAll('.pill').forEach(b => b.onclick = () => { bpStep = +b.dataset.s; bpRender(); });
  $('bpprev').disabled = s === 0; $('bpnext').textContent = s === 8 ? 'Start over ↺' : 'Next step →';
}
$('bpnext').onclick = () => { bpStep = bpStep === 8 ? 0 : bpStep + 1; bpRender(); };
$('bpprev').onclick = () => { bpStep = Math.max(0, bpStep - 1); bpRender(); };
['bx1', 'bx2', 'by', 'blr'].forEach(id => $(id).addEventListener('input', bpRender));

/* =============== 7. OVERFITTING =============== */
const LAMS = [0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
let of = {}, ofRun = false, ofSeed = 21;
$('ofl').addEventListener('input', () => { $('oflv').textContent = LAMS[+$('ofl').value]; resetOF(false); });
$('oflv').textContent = LAMS[+$('ofl').value];
function resetOF(newData) {
  if (newData || !of.train) {
    ofSeed++;
    const pts = toEx(makeData('class', 'moons', 450, 0.32, ofSeed));
    of.train = pts.slice(0, 50); of.test = pts.slice(50);
  }
  of.a = new Network([2, 8, 8, 1], 'tanh', 'class', 5); of.b = new Network([2, 8, 8, 1], 'tanh', 'class', 5);
  of.h = { at: [], aT: [], bt: [], bT: [] }; of.ep = 0;
  drawOF();
}
function ofStep() {
  const lam = LAMS[+$('ofl').value];
  for (let k = 0; k < 4; k++) {
    trainEpoch(of.a, of.train, { lr: 0.02, optimizer: 'adam' }, 10);
    trainEpoch(of.b, of.train, { lr: 0.02, optimizer: 'adam', reg: 'l2', regRate: lam }, 10);
    of.ep++;
  }
  of.h.at.push(of.a.loss(of.train)); of.h.aT.push(of.a.loss(of.test)); of.h.bt.push(of.b.loss(of.train)); of.h.bT.push(of.b.loss(of.test));
  if (of.ep >= 1600) { ofRun = false; $('ofgo').textContent = '▶ Train'; }
}
function drawOF() {
  [['ofa', of.a, 'ofas'], ['ofb', of.b, 'ofbs']].forEach(([id, net, st]) => {
    const { ctx, W, H } = setup($(id));
    heat(ctx, W, H, (x, y) => 2 * net.forward([x, y]) - 1, 6, 56);
    points(ctx, W, H, of.train, 6, 4);
    const wsum = net.W.flat(2).reduce((s, w) => s + w * w, 0);
    $(st).innerHTML = `epoch ${of.ep} · train ${net.loss(of.train).toFixed(3)} · <b>test ${net.loss(of.test).toFixed(3)}</b> · test acc ${(net.accuracy(of.test) * 100).toFixed(0)}% · ‖w‖² = ${wsum.toFixed(1)}`;
  });
  const { ctx, W, H } = setup($('ofloss'));
  const all = [...of.h.at, ...of.h.aT, ...of.h.bt, ...of.h.bT].filter(isFinite);
  const max = Math.max(0.3, ...all.slice(0)) * 1.05, n = Math.max(20, of.h.at.length);
  ctx.strokeStyle = cssv('--line'); ctx.beginPath(); ctx.moveTo(0, H - .5); ctx.lineTo(W, H - .5); ctx.stroke();
  const line = (arr, col, alpha, w) => { if (arr.length < 2) return; ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); arr.forEach((v, i) => { const x = i / (n - 1) * W, y = H - 4 - Math.min(v, max) / max * (H - 10); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.globalAlpha = 1; };
  line(of.h.at, cssv('--neg'), .4, 1.5); line(of.h.bt, cssv('--pos'), .4, 1.5); line(of.h.aT, cssv('--neg'), 1, 2.2); line(of.h.bT, cssv('--pos'), 1, 2.2);
  ctx.fillStyle = cssv('--faint'); ctx.fillText('loss over training →', 6, 12);
}
$('ofgo').onclick = () => { ofRun = !ofRun; $('ofgo').textContent = ofRun ? '❚❚ Pause' : '▶ Train'; };
$('ofreset').onclick = () => { ofRun = false; $('ofgo').textContent = '▶ Train'; resetOF(true); };

/* =============== GLOSSARY =============== */
const GLOSS = [
  ['Neuron', 'A unit computing a = f(w·x + b): a weighted sum of its inputs plus a bias, passed through an activation function.'],
  ['Weight', 'A number on a connection saying how strongly, and in which direction, one neuron\'s output affects the next.'],
  ['Bias', 'A constant added to a neuron\'s weighted sum. It shifts the decision boundary away from the origin, like an intercept.'],
  ['Activation function', 'The nonlinearity applied to each neuron\'s sum (ReLU, tanh, sigmoid…). Without it, the whole network is linear.'],
  ['Layer', 'A group of neurons that all receive the same inputs. Input → hidden layers → output.'],
  ['Hidden layer', 'Any layer between input and output. Its neurons learn intermediate features that are never directly observed.'],
  ['Forward pass', 'Computing the prediction by pushing inputs through the layers in order, storing every intermediate value.'],
  ['Loss function', 'A single number measuring how wrong the predictions are: log loss (cross-entropy) for classification, squared error for regression.'],
  ['Log loss / cross-entropy', '−[y log p + (1−y) log(1−p)]. Equal to the negative Bernoulli log-likelihood; minimising it is maximum likelihood.'],
  ['Gradient', 'The vector of partial derivatives ∂L/∂w for every parameter. It points in the direction of steepest increase of the loss.'],
  ['Gradient descent', 'Repeatedly updating w ← w − η∇L(w) to walk downhill on the loss.'],
  ['Learning rate (η)', 'The step size in gradient descent. Too small is slow; too large overshoots and can diverge.'],
  ['Backpropagation', 'Computing all gradients efficiently with the chain rule, from the output backwards, reusing stored forward values.'],
  ['Chain rule', 'd/dx f(g(x)) = f′(g(x))·g′(x). Multiply local derivatives along a path; add across paths.'],
  ['Epoch', 'One full pass through the training data.'],
  ['Batch / mini-batch', 'The number of examples whose gradients are averaged before one update. Batch size 1 is pure stochastic gradient descent.'],
  ['SGD', 'Stochastic gradient descent: gradient estimated from a random subset, so each step is cheap but noisy.'],
  ['Momentum', 'Keeps a running average of past gradients so steps build speed in consistent directions and zig-zags cancel.'],
  ['Adam', 'An optimizer that combines momentum with a per-parameter step size scaled by recent gradient magnitudes.'],
  ['Overfitting', 'Fitting noise in the training data: low training loss, higher test loss. A high-variance model.'],
  ['Regularization', 'Anything that limits model complexity to reduce variance: L2, L1, dropout, early stopping, smaller networks.'],
  ['L2 / weight decay', 'Adding λ‖w‖² to the loss, the same penalty as ridge regression. It shrinks all weights smoothly toward zero.'],
  ['L1', 'Adding λ‖w‖₁ (lasso). It pushes weak weights to exactly zero, giving sparse connections.'],
  ['Early stopping', 'Stopping training when validation loss stops improving; it acts as regularization.'],
  ['Vanishing gradient', 'Gradients shrink as they are multiplied by small activation slopes through many layers, so early layers barely learn.'],
  ['Dead ReLU', 'A ReLU neuron whose input is always negative: output 0 and gradient 0, so it never recovers.'],
  ['Universal approximation', 'One hidden layer with enough neurons and a nonlinear activation can approximate any continuous function on a bounded region.'],
  ['Feature engineering', 'Creating new inputs (x², x₁x₂, sin x) by hand. It is basis expansion: still a linear model, in richer features.'],
  ['Decision boundary', 'The set of inputs where the classifier switches class (p = 0.5). One neuron gives a straight line; hidden layers give curves.'],
  ['Validation / test set', 'Data held out from training to measure generalisation. Used to choose hyperparameters (validation) and to report final performance (test).'],
];
function drawGloss() {
  const q = $('gsearch').value.toLowerCase();
  $('gltable').innerHTML = GLOSS.filter(([a, b]) => (a + b).toLowerCase().includes(q)).map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('') || '<tr><td colspan="2">No match.</td></tr>';
}
$('gsearch').addEventListener('input', drawGloss);

/* =============== QUIZ =============== */
const QUIZ = [
  [false, 'A neural network with 3 hidden layers and linear activations can learn a circular decision boundary.', 'All-linear layers collapse: W₃W₂W₁x + c is one linear map, so the boundary is still a straight line. Nonlinearity, not depth, is what bends it.'],
  [true, 'A network with no hidden layer, a sigmoid output and log loss is equivalent to logistic regression.', 'P(y=1) = σ(wᵀx + b), with log-odds linear in x, fitted by minimising the negative Bernoulli log-likelihood, i.e. by maximum likelihood.'],
  [false, 'Neural networks must never use linear activation functions.', 'A linear output layer is standard for regression. Only a network that is linear everywhere reduces to a linear model.'],
  [false, 'Gradient descent always finds the global minimum of a neural network\'s loss.', 'The loss of a network with hidden layers is non-convex. GD can stop in a local minimum or slow down on a plateau, and the result depends on the initialisation.'],
  [true, 'For a sigmoid output with log loss, ∂L/∂z at the output equals p − y.', '∂L/∂p · ∂p/∂z = [−y/p + (1−y)/(1−p)] · p(1−p) = p − y.'],
  [false, 'A larger learning rate always makes training converge faster.', 'Past a threshold each step overshoots the minimum; the loss oscillates or diverges.'],
  [true, 'Backpropagation needs the values computed during the forward pass.', 'Each local derivative (e.g. tanh′ = 1 − h², ∂z/∂w = input) uses the stored activations.'],
  [false, 'Backpropagation is an optimization algorithm that replaces gradient descent.', 'Backprop only computes the gradients. Gradient descent (or Adam, etc.) uses them to update the weights.'],
  [true, 'L2 regularization on a neural network\'s weights is the same penalty as ridge regression.', 'Adding λ‖w‖² gives the gradient an extra 2λw term (weight decay), shrinking weights toward 0 and reducing variance.'],
  [false, 'If training loss keeps decreasing, the model is getting better on new data.', 'Training loss always falls with more capacity or more epochs. Test or validation loss can rise at the same time: overfitting.'],
  [true, 'Deep networks with sigmoid hidden activations can suffer vanishing gradients.', 'Backprop multiplies one σ′ ≤ 0.25 per layer, so the signal shrinks geometrically toward the input layers.'],
  [false, 'The perceptron\'s step activation can be trained by gradient descent.', 'Its derivative is 0 almost everywhere, so there is no gradient signal. Smooth activations made backprop possible.'],
  [true, 'Adding the feature X₁X₂ lets a model with no hidden layer solve XOR.', 'XOR\'s label is the sign of x₁x₂, so a linear boundary in that feature separates the classes perfectly (basis expansion).'],
  [false, 'Hyperparameters such as the number of layers and λ should be chosen to minimise training loss.', 'Training loss always prefers the biggest, least regularised model. Choose them by validation error or cross-validation.'],
  [true, 'With a fixed learning rate, gradient-descent steps get smaller as you approach a minimum.', 'Step = η × gradient, and the gradient shrinks toward 0 near a minimum.'],
  [false, 'SGD computes the exact gradient of the training loss at every step.', 'It uses a random mini-batch, so each step is a noisy (unbiased) estimate. The noise can even help escape poor minima.'],
];
let answers = {};
function drawQuiz() {
  const correct = Object.entries(answers).filter(([i, a]) => QUIZ[i][0] === a).length;
  $('score').textContent = `Score: ${correct} / ${Object.keys(answers).length} answered (${QUIZ.length} total)`;
  $('quizlist').innerHTML = QUIZ.map(([ans, q, why], i) => {
    const done = i in answers, right = done && answers[i] === ans;
    return `<div class="q ${done ? 'done ' + (right ? 'right' : 'wrong') : ''}"><p>${i + 1}. ${q}</p><div class="opts"><button class="btn sm" data-i="${i}" data-a="1" ${done ? 'disabled' : ''}>True</button><button class="btn sm" data-i="${i}" data-a="0" ${done ? 'disabled' : ''}>False</button></div><div class="why"><span class="verdict">${right ? '✓ Correct' : '✗ Not quite'}: ${ans ? 'True' : 'False'}.</span> ${why}</div></div>`;
  }).join('');
  $('quizlist').querySelectorAll('button[data-i]').forEach(b => b.onclick = () => { answers[+b.dataset.i] = b.dataset.a === '1'; drawQuiz(); });
}
$('quizreset').onclick = () => { answers = {}; drawQuiz(); };

/* =============== TOC + loop =============== */
const tocLinks = [...document.querySelectorAll('#toc a')];
const sections = tocLinks.map(a => document.querySelector(a.getAttribute('href')));
function onScroll() {
  let cur = 0; sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top < 140) cur = i; });
  tocLinks.forEach((a, i) => a.classList.toggle('on', i === cur));
  const h = document.documentElement; $('prog').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%';
}
window.addEventListener('scroll', onScroll, { passive: true });

function drawAll() { drawNeuron(); drawAct(); drawDepth(); drawLoss(); drawGD(); bpRender(); drawOF(); }
window.addEventListener('themechange', () => { rebuildColormap(); gdImg = null; drawAll(); });
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { gdImg = null; drawAll(); }, 150); });

resetDepth(); resetOF(true); launchGD(); drawGloss(); drawQuiz(); drawAll(); onScroll();
let fr = 0;
function loop() {
  fr++;
  if (depthRun && fr % 3 === 0) { depthNets.forEach(n => trainEpoch(n, depthData, { lr: 0.1, optimizer: 'sgd' }, 10)); depthEp++; if (depthEp >= 500) { depthRun = false; $('depthgo').textContent = '▶ Train all three'; } drawDepth(); }
  if (gdRuns.some(r => !r.done) && fr % 5 === 0) { stepGD(); drawGD(); }
  if (ofRun && fr % 2 === 0) { ofStep(); drawOF(); }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
