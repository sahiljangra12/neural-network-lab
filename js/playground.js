import { CLASS_DATASETS, REG_DATASETS, FEATURES, makeData } from './data.js?v=202611182231';
import { Network, ACTIVATIONS } from './nn.js?v=202611182231';
import { colormap, rebuildColormap } from './colors.js?v=202611182231';
import { CH } from './missions.js?v=202611182231';

const $ = id => document.getElementById(id);
const LRS = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
const REGRATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
const N_POINTS = 500, GRID = 50, MAX_LAYERS = 6, MAX_NEURONS = 8;
let NODE = 40, VGAP = 54; const HEAD = 52;

const DEFAULT = {
  problem: 'class', dataset: 'circle', regDataset: 'plane', lr: 0.03, act: 'tanh', opt: 'sgd',
  reg: 'none', regRate: 0, batch: 10, noise: 0, ratio: 50, layers: [4, 2], features: ['x1', 'x2'],
  showTest: false, discretize: false, seed: 1, speed: 10,
};
const clone = o => JSON.parse(JSON.stringify(o));

/* ---------------- state + URL hash ---------------- */
function parseHash() {
  if (location.hash.length < 3) return null;
  try {
    const p = new URLSearchParams(location.hash.slice(1)), s = clone(DEFAULT);
    if (!p.has('problem')) return null;
    for (const k of Object.keys(DEFAULT)) {
      if (!p.has(k)) continue;
      const v = p.get(k);
      if (k === 'layers') s.layers = v === '' ? [] : v.split(',').map(Number).filter(n => n >= 1 && n <= MAX_NEURONS).slice(0, MAX_LAYERS);
      else if (k === 'features') s.features = v.split(',').filter(f => FEATURES.some(F => F.id === f));
      else if (typeof DEFAULT[k] === 'number') s[k] = +v;
      else if (typeof DEFAULT[k] === 'boolean') s[k] = v === 'true';
      else s[k] = v;
    }
    if (!s.features.length) s.features = ['x1', 'x2'];
    return s;
  } catch { return null; }
}
function writeHash() {
  const p = new URLSearchParams();
  for (const k of Object.keys(DEFAULT)) p.set(k, Array.isArray(S[k]) ? S[k].join(',') : S[k]);
  history.replaceState(null, '', '#' + p.toString());
}

let S = parseHash() || clone(DEFAULT);
let net, train = [], test = [], epoch = 0, playing = false, netSeed = 7;
let hist = { train: [], test: [] }, bestTest = Infinity;
let hover = null;            // {l, j} node shown on output map
let gridInputs = [], nodeVals = [], featVals = [];
let lastTrainLoss = NaN, lastTestLoss = NaN, lossAtStart = NaN;

/* ---------------- data ---------------- */
function activeFeatures() { return FEATURES.filter(f => S.features.includes(f.id)); }
function toExample(p) { return { x: p.x, y: p.y, label: p.label, input: activeFeatures().map(f => f.f(p.x, p.y)) }; }

function genData() {
  const name = S.problem === 'class' ? S.dataset : S.regDataset;
  const pts = makeData(S.problem, name, N_POINTS, S.noise / 100, S.seed);
  const nTrain = Math.round(pts.length * S.ratio / 100);
  train = pts.slice(0, nTrain).map(toExample);
  test = pts.slice(nTrain).map(toExample);
}
function refreshInputs() { train = train.map(toExample); test = test.map(toExample); }

function buildGrid() {
  const feats = activeFeatures();
  gridInputs = [];
  featVals = FEATURES.map(() => new Float32Array(GRID * GRID));
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    const x = -6 + (i + 0.5) * 12 / GRID, y = 6 - (j + 0.5) * 12 / GRID, k = j * GRID + i;
    gridInputs.push(feats.map(f => f.f(x, y)));
    FEATURES.forEach((f, fi) => { featVals[fi][k] = f.f(x, y); });
  }
  featVals = featVals.map(arr => { let m = 0; for (const v of arr) m = Math.max(m, Math.abs(v)); return arr.map(v => v / (m || 1)); });
}

/* ---------------- network ---------------- */
function newNet() {
  net = new Network([activeFeatures().length, ...S.layers, 1], S.act, S.problem, netSeed++);
  epoch = 0; hist = { train: [], test: [] }; bestTest = Infinity;
  computeLosses(); lossAtStart = lastTrainLoss;
  hist.train.push(lastTrainLoss); hist.test.push(lastTestLoss);
}

function trainEpoch() {
  const order = train.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const opts = { lr: S.lr, optimizer: S.opt, reg: S.reg, regRate: S.regRate };
  order.forEach((idx, n) => {
    const e = train[idx];
    net.forward(e.input); net.backward(e.label);
    if ((n + 1) % S.batch === 0) net.update(opts);
  });
  net.update(opts);
  epoch++;
  computeLosses();
  hist.train.push(lastTrainLoss); hist.test.push(lastTestLoss);
  if (hist.train.length > 600) { hist.train = hist.train.filter((_, i) => i % 2 === 0); hist.test = hist.test.filter((_, i) => i % 2 === 0); }
  if (isFinite(lastTestLoss)) bestTest = Math.min(bestTest, lastTestLoss);
}

function computeLosses() { lastTrainLoss = net.loss(train); lastTestLoss = net.loss(test); }

function computeGrid() {
  const L = net.sizes.length - 1;
  nodeVals = net.sizes.map(n => Array.from({ length: n }, () => new Float32Array(GRID * GRID)));
  for (let k = 0; k < gridInputs.length; k++) {
    net.forward(gridInputs[k]);
    for (let l = 1; l <= L; l++) {
      const a = net.a[l], arr = nodeVals[l];
      for (let j = 0; j < a.length; j++) {
        let v = a[j];
        if (l === L && S.problem === 'class') v = 2 * v - 1;
        arr[j][k] = v;
      }
    }
  }
}

/* ---------------- drawing helpers ---------------- */
function paint(canvas, vals, discretize = false) {
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(GRID, GRID);
  for (let k = 0; k < vals.length; k++) {
    let v = vals[k]; if (!isFinite(v)) v = 0;
    if (discretize) v = v >= 0 ? 1 : -1;
    const c = colormap(v), o = k * 4;
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const offscreen = document.createElement('canvas'); offscreen.width = offscreen.height = GRID;
const out = $('out');
function drawOutput() {
  const dpr = window.devicePixelRatio || 1, size = out.clientWidth || 360;
  if (out.width !== Math.round(size * dpr)) { out.width = out.height = Math.round(size * dpr); }
  const ctx = out.getContext('2d'), W = out.width;
  const L = net.sizes.length - 1;
  let vals;
  if (hover && hover.l === 0) vals = featVals[hover.j];
  else if (hover) vals = nodeVals[hover.l][hover.j];
  else vals = nodeVals[L][0];
  paint(offscreen, vals, S.discretize && !hover);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(offscreen, 0, 0, W, W);
  const P = v => (v + 6) / 12 * W, Q = v => (6 - v) / 12 * W, r = 3.2 * dpr;
  const drawPts = (pts, isTest) => {
    for (const p of pts) {
      const c = colormap(S.problem === 'class' ? p.label : p.label);
      ctx.beginPath(); ctx.arc(P(p.x), Q(p.y), r, 0, 7);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fill();
      ctx.lineWidth = (isTest ? 1.8 : 1) * dpr; ctx.strokeStyle = isTest ? '#ffffff' : (p.label > 0 ? 'rgba(220,220,220,.85)' : 'rgba(0,0,0,.85)'); ctx.stroke();
    }
  };
  if (!hover) { drawPts(train, false); if (S.showTest) drawPts(test, true); }
  if (hover) {
    ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, 0, W, 26 * dpr);
    ctx.fillStyle = '#fff'; ctx.font = `${12 * dpr}px Inter, sans-serif`;
    const name = hover.l === 0 ? `Input feature ${FEATURES[hover.j].label}` : hover.l === L ? 'Output neuron' : `Layer ${hover.l}, neuron ${hover.j + 1}: what this neuron detects`;
    ctx.fillText(name, 9 * dpr, 17 * dpr);
  }
}

function drawLossChart() {
  const c = $('losschart'), dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight || 160;
  if (!w || !h) return;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement), cv = n => css.getPropertyValue(n).trim();
  const L = 44, B = 20, T = 6, Rm = 8, pw = w - L - Rm, ph = h - T - B;
  const all = hist.train.concat(hist.test).filter(isFinite);
  const max = niceMax(Math.max(0.05, ...all));
  ctx.font = '10.5px "Roboto Mono", monospace'; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = max * k / 4, y = Math.round(T + ph - v / max * ph) + .5;
    ctx.strokeStyle = k ? 'rgba(255,255,255,.07)' : cv('--line2'); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
    ctx.fillStyle = cv('--faint'); ctx.textAlign = 'right'; ctx.fillText(v.toFixed(max < 0.5 ? 2 : 1), L - 6, y + 3.5);
  }
  ctx.textAlign = 'left'; ctx.fillText('0', L, h - 5);
  ctx.textAlign = 'right'; ctx.fillText(`epoch ${epoch}`, L + pw, h - 5);
  ctx.textAlign = 'center'; if (epoch > 1) ctx.fillText(String(Math.round(epoch / 2)), L + pw / 2, h - 5);
  const line = (arr, color, width) => {
    if (arr.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.beginPath();
    arr.forEach((v, i) => { const x = L + i / (arr.length - 1) * pw, y = T + ph - (Math.min(isFinite(v) ? v : max, max) / max) * ph; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  line(hist.test, cv('--neg'), 1.6);
  line(hist.train, cv('--fg'), 1.8);
  const last = arr => arr.length ? arr[arr.length - 1] : NaN;
  ctx.textAlign = 'right'; ctx.font = '11px "Roboto Mono", monospace';
  if (isFinite(last(hist.train))) { ctx.fillStyle = cv('--fg'); ctx.fillText(`train ${last(hist.train).toFixed(3)}`, L + pw, T + 12); }
  if (test.length && isFinite(last(hist.test))) { ctx.fillStyle = cv('--neg'); ctx.fillText(`test ${last(hist.test).toFixed(3)}`, L + pw, T + 26); }
}
function niceMax(v) { const p = 10 ** Math.floor(Math.log10(v)); for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (v <= m * p) return m * p; return 10 * p; }

/* ---------------- network diagram ---------------- */
const netEl = $('net'), linksEl = $('links');
let nodeEls = [], linkEls = [], layout = null;

function buildNetwork() {
  [...netEl.querySelectorAll('.node,.featlabel,.colhead')].forEach(e => e.remove());
  linksEl.innerHTML = '';
  nodeEls = []; linkEls = [];
  const appMode = matchMedia('(min-width: 1100px) and (min-height: 600px)').matches;
  const narrow = netEl.parentElement.clientWidth < 520;
  const counts = [FEATURES.length, ...S.layers, 1];
  const maxN = Math.max(...counts);
  if (appMode) {
    const availH = netEl.parentElement.clientHeight;
    VGAP = availH > 150 ? Math.max(26, Math.min(60, Math.floor((availH - HEAD - 6) / maxN))) : 54;
  } else VGAP = narrow ? 42 : 54;
  NODE = VGAP < 32 ? 22 : VGAP < 38 ? 26 : VGAP < 46 ? 32 : VGAP < 52 ? 36 : 40;
  const nCols = S.layers.length + 2, left = narrow ? 64 : 74, colGap = narrow ? 58 : 90;
  const width = Math.max(netEl.parentElement.clientWidth, left + (nCols - 1) * colGap + NODE + 14);
  const right = width - NODE - 14;
  const xs = Array.from({ length: nCols }, (_, c) => left + (right - left) * c / (nCols - 1));
  const height = HEAD + maxN * VGAP;
  netEl.style.height = height + 'px'; netEl.style.width = width + 'px';
  linksEl.setAttribute('width', width); linksEl.setAttribute('height', height);
  const ys = counts.map((n, c) => Array.from({ length: n }, (_, j) => HEAD + (c === 0 ? 0 : (maxN - n) * VGAP / 2) + j * VGAP));
  layout = { xs, ys };

  // links (draw first so they sit beneath nodes)
  const active = activeFeatures().map(f => FEATURES.indexOf(f));
  for (let l = 1; l < net.sizes.length; l++) {
    for (let j = 0; j < net.sizes[l]; j++) {
      for (let i = 0; i < net.sizes[l - 1]; i++) {
        const srcCol = l - 1, srcRow = l === 1 ? active[i] : i;
        const x1 = xs[srcCol] + NODE, y1 = ys[srcCol][srcRow] + NODE / 2;
        const x2 = xs[l], y2 = ys[l][j] + NODE / 2, mx = (x1 + x2) / 2;
        const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        p.setAttribute('d', `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
        p.dataset.l = l - 1; p.dataset.j = j; p.dataset.i = i;
        linksEl.appendChild(p);
        linkEls.push({ el: p, l: l - 1, j, i });
      }
    }
  }

  // nodes
  const mk = (c, r, cls) => {
    const d = document.createElement('div'); d.className = 'node' + (cls ? ' ' + cls : '');
    d.style.left = xs[c] + 'px'; d.style.top = ys[c][r] + 'px'; d.style.width = d.style.height = NODE + 'px';
    const cv = document.createElement('canvas'); cv.width = cv.height = GRID; d.appendChild(cv);
    netEl.appendChild(d); return { d, cv };
  };
  nodeEls = [[]];
  FEATURES.forEach((f, r) => {
    const on = S.features.includes(f.id), n = mk(0, r, on ? '' : 'off');
    n.d.dataset.l = 0; n.d.dataset.j = r;
    nodeEls[0].push(n);
    const lab = document.createElement('div'); lab.className = 'featlabel';
    lab.style.left = (xs[0] - 62) + 'px'; lab.style.width = '56px'; lab.style.top = (ys[0][r] + NODE / 2 - 9) + 'px'; lab.textContent = f.label;
    netEl.appendChild(lab);
  });
  for (let l = 1; l < net.sizes.length; l++) {
    nodeEls[l] = [];
    for (let j = 0; j < net.sizes[l]; j++) {
      const n = mk(l, j); n.d.dataset.l = l; n.d.dataset.j = j;
      const b = document.createElement('div'); b.className = 'bias'; n.d.appendChild(b); n.bias = b;
      nodeEls[l].push(n);
    }
  }
  // column heads
  const hw = narrow ? colGap - 4 : NODE + 60;
  const head = (c, html) => { const h = document.createElement('div'); h.className = 'colhead'; h.style.left = (xs[c] + NODE / 2 - hw / 2) + 'px'; h.style.width = hw + 'px'; h.style.top = '0px'; h.innerHTML = html; netEl.appendChild(h); return h; };
  head(0, `<div style="height:25px"></div>${narrow ? 'Inputs' : 'Features'}`);
  S.layers.forEach((n, li) => {
    const h = head(li + 1, `<div class="pm"><button data-a="-" data-l="${li}" title="Remove a neuron">−</button><button data-a="+" data-l="${li}" title="Add a neuron">+</button></div>${narrow ? n : `${n} neuron${n > 1 ? 's' : ''}`}`);
    h.querySelectorAll('button').forEach(b => b.onclick = () => {
      const k = +b.dataset.l; S.layers[k] = Math.max(1, Math.min(MAX_NEURONS, S.layers[k] + (b.dataset.a === '+' ? 1 : -1)));
      structureChanged();
    });
  });
  head(nCols - 1, `<div style="height:25px"></div>${narrow ? 'Out' : 'Output'}`);
  $('nlayers').textContent = S.layers.length;
  $('nparams').textContent = `${net.weightCount()} parameters`;
}

function updateNetwork() {
  const css = getComputedStyle(document.documentElement);
  const pos = css.getPropertyValue('--pos').trim(), neg = css.getPropertyValue('--neg').trim();
  for (const k of linkEls) {
    const w = net.W[k.l][k.j][k.i];
    k.el.setAttribute('stroke', w >= 0 ? pos : neg);
    k.el.setAttribute('stroke-width', Math.max(0.4, Math.min(9, Math.abs(w) * 2.4)).toFixed(2));
    k.el.setAttribute('stroke-opacity', w === 0 ? '0.08' : Math.min(0.85, 0.25 + Math.abs(w) * 0.4).toFixed(2));
  }
  nodeEls[0].forEach((n, r) => paint(n.cv, featVals[r]));
  for (let l = 1; l < nodeEls.length; l++) nodeEls[l].forEach((n, j) => {
    paint(n.cv, nodeVals[l][j], l === nodeEls.length - 1 && S.discretize);
    const b = net.b[l - 1][j], c = colormap(Math.max(-1, Math.min(1, b)));
    n.bias.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
  });
}

/* ---------------- tooltips & weight editing ---------------- */
const tip = $('tip');
function showTip(e, html) { tip.innerHTML = html; tip.classList.add('show'); tip.style.left = Math.min(innerWidth - 270, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; }
function hideTip() { tip.classList.remove('show'); }
const nodeName = (l, j) => l === 0 ? activeFeatures()[j].label : l === net.sizes.length - 1 ? 'output' : `neuron ${j + 1} of layer ${l}`;

netEl.addEventListener('mousemove', e => {
  const n = e.target.closest('.node'), p = e.target.closest('path');
  if (n) {
    const l = +n.dataset.l, j = +n.dataset.j; hover = { l, j };
    if (l === 0) { const f = FEATURES[j]; showTip(e, `<b>${f.label}</b>: ${f.help}<br>${S.features.includes(f.id) ? 'On' : 'Off'}. Click to switch ${S.features.includes(f.id) ? 'off' : 'on'}.`); }
    else {
      const L = net.sizes.length - 1;
      showTip(e, `<b>${l === L ? 'Output neuron' : `Layer ${l}, neuron ${j + 1}`}</b><br><span class="mono">bias b = ${net.b[l - 1][j].toFixed(3)}</span><br>${l === L ? (S.problem === 'class' ? 'σ(w·h + b) = P(black). This is logistic regression on the last layer.' : 'Linear output w·h + b') : `${ACTIVATIONS[S.act].name}(w·inputs + b)`}`);
    }
  } else if (p) {
    hover = null;
    const l = +p.dataset.l, j = +p.dataset.j, i = +p.dataset.i;
    showTip(e, `<span class="mono">w = ${net.W[l][j][i].toFixed(4)}</span><br>from ${nodeName(l, i)} → ${nodeName(l + 1, j)}<br><i>Click to edit</i>`);
  } else { hover = null; hideTip(); }
});
netEl.addEventListener('mouseleave', () => { hover = null; hideTip(); });
netEl.addEventListener('click', e => {
  const n = e.target.closest('.node'), p = e.target.closest('path');
  if (n && +n.dataset.l === 0) {
    const id = FEATURES[+n.dataset.j].id;
    if (S.features.includes(id)) { if (S.features.length > 1) S.features = S.features.filter(f => f !== id); }
    else S.features = FEATURES.map(f => f.id).filter(f => f === id || S.features.includes(f));
    refreshInputs(); structureChanged();
  } else if (p) {
    const l = +p.dataset.l, j = +p.dataset.j, i = +p.dataset.i, box = $('wedit');
    box.style.display = 'block'; box.style.left = Math.min(innerWidth - 220, e.clientX) + 'px'; box.style.top = (e.clientY + 10) + 'px';
    $('weditlbl').textContent = `Weight ${nodeName(l, i)} → ${nodeName(l + 1, j)}`;
    const inp = $('weditin'); inp.value = net.W[l][j][i].toFixed(3); inp.focus(); inp.select();
    const apply = () => { const v = parseFloat(inp.value); if (isFinite(v)) { net.W[l][j][i] = v; computeLosses(); render(true); } box.style.display = 'none'; };
    $('weditok').onclick = apply; inp.onkeydown = ev => { if (ev.key === 'Enter') apply(); if (ev.key === 'Escape') box.style.display = 'none'; };
    e.stopPropagation();
  }
});
document.addEventListener('click', e => { if (!e.target.closest('#wedit') && !e.target.closest('path')) $('wedit').style.display = 'none'; });

function probe(e) {
  const r = out.getBoundingClientRect();
  const x = -6 + (e.clientX - r.left) / r.width * 12, y = 6 - (e.clientY - r.top) / r.height * 12;
  const p = net.forward(activeFeatures().map(f => f.f(x, y)));
  $('probe').innerHTML = S.problem === 'class'
    ? `X₁ = ${x.toFixed(2)}, X₂ = ${y.toFixed(2)} → P(black) = <b>${p.toFixed(3)}</b>, predicts <b style="color:var(${p >= .5 ? '--fg' : '--neg'})">${p >= .5 ? 'black' : 'red'}</b>`
    : `X₁ = ${x.toFixed(2)}, X₂ = ${y.toFixed(2)} → ŷ = <b>${p.toFixed(3)}</b>`;
}
out.addEventListener('pointermove', probe);
out.addEventListener('pointerdown', probe);

/* ---------------- insight engine ---------------- */
const NONLINEAR = ['circle', 'xor', 'spiral', 'checker', 'moons'];
function insight() {
  const el = $('insight'), set = (cls, title, body) => { el.className = 'insight ' + cls; el.innerHTML = `<b>${title}</b>${body}`; };
  const acc = net.accuracy(test.length ? test : train);
  if (!isFinite(lastTrainLoss) || lastTrainLoss > 50) return set('bad', 'Training blew up', `The loss is ${isFinite(lastTrainLoss) ? 'huge' : 'not a number'}. The learning rate (${S.lr}) is so large that each step overshoots the minimum and lands further away. Lower it and press reset.`);
  if (epoch === 0) return set('', 'Press play', 'The weights are random, so the output map is close to noise. Press play (or Space) and watch the neurons organise themselves.');
  if (S.act === 'linear' && S.layers.length > 0 && S.problem === 'class' && NONLINEAR.includes(S.dataset) && S.features.every(f => f === 'x1' || f === 'x2'))
    return set('warn', 'An all-linear network is still linear', `With a linear activation, W₂(W₁x + b₁) + b₂ = (W₂W₁)x + c. All ${S.layers.length} hidden layers collapse into a single straight line, however many neurons you add. Switch to tanh or ReLU.`);
  if (S.layers.length === 0 && S.problem === 'class' && NONLINEAR.includes(S.dataset) && S.features.every(f => f === 'x1' || f === 'x2') && epoch > 30)
    return set('warn', 'One straight line is not enough', 'With no hidden layers the network is plain logistic regression on X₁, X₂, so its boundary is one straight line. Either add a hidden layer or engineer a feature (try X₁² and X₂² for the circle, X₁X₂ for XOR).');
  if (S.act === 'relu' && S.layers.length) {
    let dead = 0, total = 0;
    for (let l = 1; l < net.sizes.length - 1; l++) for (const arr of nodeVals[l]) { total++; if (arr.every(v => v === 0)) dead++; }
    if (total && dead / total >= 0.3) return set('warn', `${dead} of ${total} neurons are dead`, 'A ReLU that outputs 0 everywhere has gradient 0, so it can never recover. This usually comes from a learning rate that is too high. Try Leaky ReLU, a lower rate, or reset.');
  }
  const overfit = epoch > 120 && test.length && lastTestLoss > bestTest * 1.15 && lastTrainLoss < hist.train[Math.max(0, hist.train.length - 40)];
  if (S.problem === 'class' && acc !== null && acc >= 0.97 && epoch > 5 && overfit)
    return set('ok', `Solved: ${(acc * 100).toFixed(1)}% test accuracy, but watch the test loss`, `Accuracy is excellent, yet test loss has crept up from ${bestTest.toFixed(3)} to ${lastTestLoss.toFixed(3)}: the network is becoming over-confident on points it gets wrong. Early stopping or a little L2 would keep it calibrated.`);
  if (overfit)
    return set('warn', 'Overfitting detected', `Train loss is still falling (${lastTrainLoss.toFixed(3)}), but test loss has risen to ${lastTestLoss.toFixed(3)} from its best of ${bestTest.toFixed(3)}. The network is memorising noise. Add L2 regularization, use fewer neurons, add more training data, or stop early.`);
  if (S.problem === 'class' && acc !== null && acc >= 0.97 && epoch > 5)
    return set('ok', `Solved: ${(acc * 100).toFixed(1)}% ${test.length ? 'test' : 'train'} accuracy`, `Done in ${epoch} epochs with ${net.weightCount()} parameters. Can you do it with fewer neurons? Hover each neuron to see the piece of the boundary it contributes.`);
  if (S.problem === 'reg' && lastTestLoss < 0.01 && epoch > 5)
    return set('ok', 'Good fit', `Test loss ${lastTestLoss.toFixed(4)}. The output surface matches the target. Hover the hidden neurons to see the building blocks it was made from.`);
  if (epoch > 200 && lastTrainLoss > lossAtStart * 0.85)
    return set('warn', 'Barely learning', `After ${epoch} epochs the loss has hardly moved. The learning rate may be too small (${S.lr}), the network too small, or the inputs not informative enough. Try a larger rate or Adam.`);
  set('', 'Training…', `Epoch ${epoch}. Loss has fallen from ${lossAtStart.toFixed(3)} to ${lastTrainLoss.toFixed(3)}. Each epoch the network sees all ${train.length} training points in batches of ${S.batch} and nudges every weight down its gradient.`);
}

/* ---------------- render ---------------- */
let frameN = 0;
function render(full) {
  computeGrid();
  updateNetwork();
  drawOutput();
  drawLossChart();
  if ($('fabep')) { $('fabep').textContent = 'Epoch ' + epoch; const a = S.problem === 'class' ? net.accuracy(test.length ? test : train) : null; $('fabacc').textContent = a === null ? `loss ${isFinite(lastTestLoss) ? lastTestLoss.toFixed(3) : '–'}` : `test acc ${(a * 100).toFixed(1)}%`; }
  $('epoch').textContent = String(epoch).padStart(6, '0').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const f = v => isFinite(v) ? v.toFixed(3) : '∞';
  $('trainloss').textContent = f(lastTrainLoss); $('testloss').textContent = test.length ? f(lastTestLoss) : '–';
  if (S.problem === 'class') {
    const a1 = net.accuracy(train), a2 = net.accuracy(test);
    $('trainacc').textContent = (a1 * 100).toFixed(1) + '%'; $('testacc').textContent = a2 === null ? '–' : (a2 * 100).toFixed(1) + '%';
    $('acclbl').textContent = 'Train acc.'; $('acclbl2').textContent = 'Test acc.';
  } else {
    $('trainacc').textContent = net.W.length ? net.W.reduce((s, L) => s + L.flat().filter(w => Math.abs(w) < 1e-3).length, 0) : '–';
    $('acclbl').textContent = 'Zero wts'; $('testacc').textContent = net.weightCount(); $('acclbl2').textContent = 'Params';
  }
  $('lossname').textContent = S.problem === 'class' ? 'log loss' : '½ squared error';
  $('sclo').textContent = S.problem === 'class' ? 'red' : '−1'; $('schi').textContent = S.problem === 'class' ? 'black' : '1';
  $('scmid').textContent = S.problem === 'class' ? 'P = 0.5' : '0';
  insight();
  frameN++;
}

let acc = 0, lastT = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  let trained = false;
  if (playing) {
    acc += dt * S.speed;
    let n = 0;
    while (acc >= 1 && n < 4) { trainEpoch(); acc -= 1; n++; trained = true; }
  }
  if (trained) render();
  else if (hover !== lastHover) render(true);
  lastHover = hover;
  requestAnimationFrame(loop);
}
let lastHover = null;

/* ---------------- controls ---------------- */
function setPlaying(p) {
  playing = p;
  const fp = $('fabplay'); if (fp) fp.innerHTML = p ? '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>' : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>';
  netEl.classList.toggle('training', p); $('play').classList.toggle('on', p);
  $('playicon').innerHTML = p ? '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>' : '<path d="M7 4.5v15l13-7.5z"/>';
  $('play').setAttribute('aria-label', p ? 'Pause' : 'Play');
}
function structureChanged() { buildGrid(); newNet(); buildNetwork(); render(true); writeHash(); }
function dataChanged() { genData(); newNet(); render(true); writeHash(); }

function fillSelect(id, values, fmt = v => v) { $(id).innerHTML = values.map(v => `<option value="${v}">${fmt(v)}</option>`).join(''); }
fillSelect('lr', LRS);
fillSelect('regRate', REGRATES);
$('act').innerHTML = Object.entries(ACTIVATIONS).map(([k, a]) => `<option value="${k}">${a.name}</option>`).join('');

function syncControls() {
  $('lr').value = S.lr; $('regRate').value = S.regRate; $('act').value = S.act; $('opt').value = S.opt;
  $('reg').value = S.reg; $('problem').value = S.problem; $('speed').value = S.speed;
  $('ratio').value = S.ratio; $('ratiov').textContent = S.ratio + '%';
  $('noise').value = S.noise; $('noisev').textContent = S.noise;
  $('batch').value = S.batch; $('batchv').textContent = S.batch;
  $('showTest').checked = S.showTest; $('discretize').checked = S.discretize;
  buildDatasetPicker();
}

$('lr').onchange = e => { S.lr = +e.target.value; writeHash(); };
$('regRate').onchange = e => { S.regRate = +e.target.value; writeHash(); };
$('reg').onchange = e => { S.reg = e.target.value; if (S.reg !== 'none' && !S.regRate) { S.regRate = 0.003; $('regRate').value = 0.003; } writeHash(); };
$('opt').onchange = e => { S.opt = e.target.value; if (S.opt === 'adam' && S.lr > 0.03) { S.lr = 0.01; $('lr').value = 0.01; } writeHash(); };
$('act').onchange = e => { S.act = e.target.value; structureChanged(); };
$('problem').onchange = e => { S.problem = e.target.value; buildDatasetPicker(); dataChanged(); };
$('ratio').oninput = e => { S.ratio = +e.target.value; $('ratiov').textContent = S.ratio + '%'; dataChanged(); };
$('noise').oninput = e => { S.noise = +e.target.value; $('noisev').textContent = S.noise; dataChanged(); };
$('speed').onchange = e => { S.speed = +e.target.value; acc = 0; writeHash(); };
$('batch').oninput = e => { S.batch = +e.target.value; $('batchv').textContent = S.batch; writeHash(); };
$('showTest').onchange = e => { S.showTest = e.target.checked; render(true); writeHash(); };
$('discretize').onchange = e => { S.discretize = e.target.checked; render(true); writeHash(); };
$('regen').onclick = () => { S.seed = Math.floor(Math.random() * 1e6); dataChanged(); };
$('play').onclick = () => setPlaying(!playing);
if ($('fabplay')) $('fabplay').onclick = () => setPlaying(!playing);
if (matchMedia('(hover: none)').matches) { const t = document.querySelector('.nettip'); if (t) t.textContent = 'Tap a neuron to inspect it · tap a line to edit its weight · tap an input to switch it on/off'; $('probe').textContent = 'Tap the map to read the network\'s prediction at any point.'; }
document.addEventListener('touchstart', e => { if (!e.target.closest('#net') && !e.target.closest('#wedit')) { hover = null; hideTip(); } }, { passive: true });
$('step').onclick = () => { setPlaying(false); trainEpoch(); render(true); };
$('reset').onclick = () => { newNet(); render(true); };
$('addlayer').onclick = () => { if (S.layers.length < MAX_LAYERS) { S.layers.push(2); structureChanged(); } };
$('remlayer').onclick = () => { if (S.layers.length) { S.layers.pop(); structureChanged(); } };
$('share').onclick = async () => {
  writeHash();
  try { await navigator.clipboard.writeText(location.href); $('share').textContent = '✓'; } catch { $('share').textContent = '!'; }
  setTimeout(() => $('share').textContent = '⧉', 1400);
};
document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === 's' || e.key === 'S') { setPlaying(false); trainEpoch(); render(true); }
  else if (e.key === 'r' || e.key === 'R') { newNet(); render(true); }
});

function buildDatasetPicker() {
  const table = S.problem === 'class' ? CLASS_DATASETS : REG_DATASETS;
  const cur = S.problem === 'class' ? S.dataset : S.regDataset;
  const g = $('dsgrid'); g.innerHTML = '';
  for (const [key, d] of Object.entries(table)) {
    const b = document.createElement('button'); b.className = 'ds' + (key === cur ? ' on' : ''); b.title = d.name;
    const cv = document.createElement('canvas'); cv.width = cv.height = 120; b.appendChild(cv);
    const t = document.createElement('div'); t.className = 'tip'; t.textContent = d.name; b.appendChild(t);
    const ctx = cv.getContext('2d'), mid = colormap(0);
    ctx.fillStyle = `rgb(${mid[0]},${mid[1]},${mid[2]})`; ctx.fillRect(0, 0, 120, 120);
    for (const p of makeData(S.problem, key, 220, 0, 3)) {
      const c = colormap(p.label); ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.beginPath(); ctx.arc((p.x + 6) * 10, (6 - p.y) * 10, 2.6, 0, 7); ctx.fill();
    }
    b.onclick = () => { if (S.problem === 'class') S.dataset = key; else S.regDataset = key; buildDatasetPicker(); dataChanged(); };
    g.appendChild(b);
  }
}

/* ---------------- challenges ---------------- */
const LVL = ['', 'Phase 1 · Fundamentals', 'Phase 2 · Core concept', 'Phase 3 · Anomaly', 'Phase 4 · Final'];
function buildCards() {
  $('cards').innerHTML = CH.map((c, i) => `<div class="card"><div class="lvl l${c.lvl}">${LVL[c.lvl]} · #${i + 1}</div><h4>${c.title}</h4><p>${c.desc}</p><button class="btn sm pri" data-ch="${i}">Launch</button></div>`).join('');
  $('cards').querySelectorAll('[data-ch]').forEach(b => b.onclick = () => loadChallenge(+b.dataset.ch));
}
function loadChallenge(i) {
  const c = CH[i];
  S = Object.assign(clone(DEFAULT), clone(c.cfg));
  syncControls(); genData(); structureChanged();
  $('missiontext').innerHTML = `<b>Mission ${String(i + 1).padStart(2, '0')}: ${c.title}</b><br>${c.mission}`;
  $('mission').classList.add('show'); document.body.classList.add('mission-on'); setTimeout(() => { buildNetwork(); render(true); }, 50);
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setPlaying(true);
}
$('missionclose').onclick = () => { $('mission').classList.remove('show'); document.body.classList.remove('mission-on'); setTimeout(() => { buildNetwork(); render(true); }, 50); };

/* ---------------- boot ---------------- */
window.addEventListener('themechange', () => { rebuildColormap(); buildDatasetPicker(); render(true); });
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { buildNetwork(); render(true); }, 150); });
rebuildColormap();
syncControls();
genData(); buildGrid(); newNet(); buildNetwork(); buildCards(); render(true); writeHash();
const chParam = new URLSearchParams(location.search).get('challenge');
if (chParam && CH[+chParam - 1]) loadChallenge(+chParam - 1);
requestAnimationFrame(t => { lastT = t; loop(t); });
