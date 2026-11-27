import { softmax, crossEntropy, softmaxGrad, deepProfile, DropNet, conv2d, relu, maxPool, rnnRun, attention } from './advmath.js?v=202611261914';
import { makeData } from './data.js?v=202611261914';
import { colormap, rebuildColormap, ink, paper, shadeRGB } from './colors.js?v=202611261914';

const $ = id => document.getElementById(id);
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const SANS = 'Space Grotesk, system-ui, sans-serif';
rebuildColormap();

function setup(cv) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight || W;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  ctx.font = `600 12.5px ${SANS}`; ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
  return { ctx, W, H };
}
function pills(el, items, cur, onPick) {
  el.innerHTML = items.map(([k, n]) => `<button class="pill${String(k) === String(cur) ? ' on' : ''}" data-k="${k}">${n}</button>`).join('');
  el.querySelectorAll('.pill').forEach(b => b.onclick = () => { el.querySelectorAll('.pill').forEach(x => x.classList.toggle('on', x === b)); onPick(b.dataset.k); });
}
const off = document.createElement('canvas');
function heat(ctx, W, H, fn, R = 6, res = 56) {
  off.width = off.height = res;
  const o = off.getContext('2d'), img = o.createImageData(res, res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const c = colormap(clamp(fn(-R + (i + .5) * 2 * R / res, R - (j + .5) * 2 * R / res), -1, 1)), k = (j * res + i) * 4;
    img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
  }
  o.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
}

/* ================= 1. SOFTMAX ================= */
let smTrue = 0;
const NAMES = ['A', 'B', 'C', 'D'];
pills($('smtrue'), NAMES.map((n, i) => [i, `True: ${n}`]), 0, k => { smTrue = +k; drawSoftmax(); });
['sm0', 'sm1', 'sm2', 'sm3', 'smT'].forEach(id => $(id).addEventListener('input', drawSoftmax));
function drawSoftmax() {
  const z = [0, 1, 2, 3].map(i => +$('sm' + i).value), T = 10 ** +$('smT').value;
  z.forEach((v, i) => $(`sm${i}v`).textContent = v.toFixed(1)); $('smTv').textContent = T.toFixed(2);
  const p = softmax(z, T), g = softmaxGrad(p, smTrue, 1), L = crossEntropy(p, smTrue);
  $('smout').textContent = `p = [${p.map(v => v.toFixed(3)).join(', ')}]   sum = ${p.reduce((a, b) => a + b, 0).toFixed(3)}\nloss = −log p(${NAMES[smTrue]}) = ${L.toFixed(3)}\np − y = [${g.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)).join(', ')}]`;
  const { ctx, W, H } = setup($('smcv')), padL = 40, padB = 26, pw = W - padL - 10, ph = H - padB - 14;
  const Y = v => 14 + (1 - v) / 2 * ph;
  ctx.lineWidth = 1;
  for (const v of [-1, -0.5, 0, 0.5, 1]) { ctx.strokeStyle = v === 0 ? cssv('--fg') : cssv('--line'); ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - 10, Y(v)); ctx.stroke(); ctx.fillStyle = cssv('--mut'); ctx.textAlign = 'right'; ctx.fillText(v.toFixed(1), padL - 6, Y(v) + 4); }
  const bw = pw / 4;
  z.forEach((_, i) => {
    const x = padL + i * bw + bw * 0.14, w = bw * 0.33;
    ctx.fillStyle = cssv('--fg'); ctx.fillRect(x, Y(p[i]), w, Y(0) - Y(p[i]));
    ctx.fillStyle = cssv('--neg'); const gy = Y(Math.max(g[i], 0)); ctx.fillRect(x + w + 3, gy, w, Math.abs(Y(g[i]) - Y(0)));
    ctx.fillStyle = cssv('--fg'); ctx.textAlign = 'center'; ctx.fillText((p[i] * 100).toFixed(0) + '%', x + w / 2, Y(p[i]) - 5);
    ctx.font = `700 13.5px ${SANS}`; ctx.fillText(NAMES[i] + (i === smTrue ? ' (true)' : ''), padL + i * bw + bw / 2, H - 7); ctx.font = `600 12.5px ${SANS}`;
  });
}

/* ================= 2. DEEP NETWORKS ================= */
const DP = { act: 'sigmoid', init: 'xavier', residual: false, seed: 1 };
pills($('dpact'), [['sigmoid', 'Sigmoid'], ['tanh', 'Tanh'], ['relu', 'ReLU']], DP.act, k => { DP.act = k; drawDeep(); });
pills($('dpinit'), [['tiny', 'Too small'], ['xavier', 'Xavier 1/n'], ['he', 'He 2/n'], ['large', 'Too large']], DP.init, k => { DP.init = k; drawDeep(); });
pills($('dpres'), [['0', 'Plain'], ['1', 'Skip connections']], '0', k => { DP.residual = k === '1'; drawDeep(); });
$('dpdepth').addEventListener('input', drawDeep);
$('dpseed').onclick = () => { DP.seed++; drawDeep(); };
function drawDeep() {
  const depth = +$('dpdepth').value; $('dpdepthv').textContent = depth;
  const pr = deepProfile({ depth, act: DP.act, init: DP.init, residual: DP.residual, seed: DP.seed });
  const lg = a => a.map(v => (isFinite(v) && v > 0 ? clamp(Math.log10(v), -14, 14) : (v === 0 ? -14 : 14)));
  const A = lg(pr.act), G = lg(pr.grad);
  const lo = Math.min(-2, Math.floor(Math.min(...A, ...G))), hi = Math.max(2, Math.ceil(Math.max(...A, ...G)));
  const { ctx, W, H } = setup($('dpcv')), padL = 52, padB = 30, pw = W - padL - 14, ph = H - padB - 12;
  const X = l => padL + l / depth * pw, Y = v => 12 + (hi - v) / (hi - lo) * ph, step = Math.max(1, Math.round((hi - lo) / 7));
  ctx.lineWidth = 1;
  for (let v = lo; v <= hi; v += step) { ctx.strokeStyle = v === 0 ? cssv('--line2') : cssv('--line'); ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - 14, Y(v)); ctx.stroke(); ctx.fillStyle = cssv('--mut'); ctx.textAlign = 'right'; ctx.fillText(v === 0 ? '1' : `10^${v}`, padL - 6, Y(v) + 4); }
  ctx.textAlign = 'left'; ctx.fillText('input', padL, H - 8); ctx.textAlign = 'right'; ctx.fillText(`layer ${depth} (output)`, W - 14, H - 8);
  const line = (arr, col) => { ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.6; ctx.beginPath(); arr.forEach((v, l) => (l ? ctx.lineTo(X(l), Y(v)) : ctx.moveTo(X(l), Y(v)))); ctx.stroke(); if (depth <= 24) arr.forEach((v, l) => { ctx.beginPath(); ctx.arc(X(l), Y(v), 3, 0, 7); ctx.fill(); }); };
  line(A, cssv('--fg')); line(G, cssv('--neg'));
  const ratio = pr.grad[0] / pr.grad[depth], rl = Math.log10(ratio);
  const verdict = !isFinite(rl) || rl > 3 ? 'The gradient EXPLODES on the way back: early layers would get enormous, unstable updates.' : rl < -3 ? 'The gradient VANISHES on the way back: the first layers receive almost no learning signal.' : 'Signal and gradient both survive: every layer can learn.';
  $('dpnote').innerHTML = `Gradient reaching the first layer is <b>${isFinite(ratio) ? ratio.toExponential(1) : '∞'}×</b> its size at the output. ${verdict}`;
}

/* ================= 3. DROPOUT ================= */
let dr = {}, drRun = false, drSeed = 30;
$('drp').addEventListener('input', () => { $('drpv').textContent = (+$('drp').value).toFixed(2); resetDrop(false); });
$('drpv').textContent = (+$('drp').value).toFixed(2);
function resetDrop(newData) {
  if (newData || !dr.train) {
    drSeed++;
    const pts = makeData('class', 'moons', 440, 0.34, drSeed).map(p => ({ ...p, input: [p.x, p.y] }));
    dr.train = pts.slice(0, 40); dr.test = pts.slice(40);
  }
  dr.a = new DropNet([2, 16, 16, 1], 'tanh', 'class', 5, 0); dr.b = new DropNet([2, 16, 16, 1], 'tanh', 'class', 5, +$('drp').value);
  dr.h = { at: [], aT: [], bt: [], bT: [] }; dr.ep = 0; drawDrop();
}
function stepDrop() {
  for (let k = 0; k < 3; k++) { dr.a.trainEpoch(dr.train, { lr: 0.02, optimizer: 'adam' }); dr.b.trainEpoch(dr.train, { lr: 0.02, optimizer: 'adam' }); dr.ep++; }
  dr.h.at.push(dr.a.loss(dr.train)); dr.h.aT.push(dr.a.loss(dr.test)); dr.h.bt.push(dr.b.loss(dr.train)); dr.h.bT.push(dr.b.loss(dr.test));
  if (dr.ep >= 1500) { drRun = false; $('drgo').textContent = '▶ Train'; }
}
function drawDrop() {
  [['dra', dr.a, 'dras'], ['drb', dr.b, 'drbs']].forEach(([id, net, st]) => {
    const { ctx, W, H } = setup($(id));
    heat(ctx, W, H, (x, y) => 2 * net.forward([x, y]) - 1);
    for (const p of dr.train) { const c = colormap(p.label); ctx.beginPath(); ctx.arc((p.x + 6) / 12 * W, (6 - p.y) / 12 * H, 4, 0, 7); ctx.fillStyle = `rgb(${c})`; ctx.fill(); ctx.strokeStyle = paper(); ctx.lineWidth = 1.2; ctx.stroke(); }
    $(st).innerHTML = `epoch ${dr.ep} · train ${net.loss(dr.train).toFixed(3)} · <b>test ${net.loss(dr.test).toFixed(3)}</b> · test acc ${(net.accuracy(dr.test) * 100).toFixed(0)}%`;
  });
  const { ctx, W, H } = setup($('drloss')), all = [...dr.h.at, ...dr.h.aT, ...dr.h.bt, ...dr.h.bT].filter(isFinite), max = Math.max(0.4, ...all) * 1.05, n = Math.max(20, dr.h.at.length);
  ctx.strokeStyle = cssv('--line2'); ctx.beginPath(); ctx.moveTo(0, H - .5); ctx.lineTo(W, H - .5); ctx.stroke();
  const line = (arr, col, alpha, w) => { if (arr.length < 2) return; ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); arr.forEach((v, i) => { const x = i / (n - 1) * W, y = H - 4 - Math.min(v, max) / max * (H - 12); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.globalAlpha = 1; };
  line(dr.h.at, cssv('--neg'), .4, 1.6); line(dr.h.bt, cssv('--fg'), .4, 1.6); line(dr.h.aT, cssv('--neg'), 1, 2.6); line(dr.h.bT, cssv('--fg'), 1, 2.6);
  ctx.fillStyle = cssv('--mut'); ctx.fillText('log loss over training →', 6, 14);
}
$('drgo').onclick = () => { drRun = !drRun; $('drgo').textContent = drRun ? '❚❚ Pause' : '▶ Train'; };
$('drreset').onclick = () => { drRun = false; $('drgo').textContent = '▶ Train'; resetDrop(true); };

/* ================= 4. CONVOLUTION ================= */
const N = 12;
const IMG = {
  cross: (r, c) => (c === 5 || c === 6 || r === 5 || r === 6) && r > 1 && r < 10 && c > 1 && c < 10,
  box: (r, c) => r >= 3 && r <= 8 && c >= 3 && c <= 8 && (r === 3 || r === 8 || c === 3 || c === 8),
  vbar: (r, c) => (c === 5 || c === 6) && r > 1 && r < 10,
  diag: (r, c) => Math.abs(r - c) <= 0 && r > 1 && r < 10 || (r - c === 1 && r > 1 && r < 10),
  seven: (r, c) => (r === 2 || r === 3) && c >= 3 && c <= 8 || (r > 3 && r < 10 && (c === 9 - Math.floor(r / 2) || c === 10 - Math.floor(r / 2))),
  blank: () => false,
};
const KER = {
  vedge: [-1, 0, 1, -1, 0, 1, -1, 0, 1], hedge: [-1, -1, -1, 0, 0, 0, 1, 1, 1], diag: [2, -1, -1, -1, 2, -1, -1, -1, 2],
  blur: Array(9).fill(0.11), sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0], ident: [0, 0, 0, 0, 1, 0, 0, 0, 0],
};
let img = new Float64Array(N * N), ker = KER.vedge.slice(), cvHover = null, painting = null;
const loadImg = k => { for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) img[r * N + c] = IMG[k](r, c) ? 1 : 0; drawConv(); };
pills($('cvimg'), [['cross', 'Cross'], ['box', 'Box'], ['vbar', 'Bar'], ['diag', 'Diagonal'], ['seven', 'Seven'], ['blank', 'Blank']], 'cross', loadImg);
pills($('cvker'), [['vedge', 'Vertical edge'], ['hedge', 'Horizontal edge'], ['diag', 'Diagonal'], ['blur', 'Blur'], ['sharpen', 'Sharpen'], ['ident', 'Identity']], 'vedge', k => { ker = KER[k].slice(); fillKer(); drawConv(); });
$('cvkgrid').innerHTML = Array.from({ length: 9 }, (_, i) => `<input type="number" step="0.5" data-i="${i}" aria-label="kernel weight ${i + 1}">`).join('');
const fillKer = () => $('cvkgrid').querySelectorAll('input').forEach((el, i) => { el.value = +ker[i].toFixed(2); });
$('cvkgrid').addEventListener('input', e => { const v = parseFloat(e.target.value); ker[+e.target.dataset.i] = isFinite(v) ? v : 0; drawConv(); });
function grid(ctx, W, n, val, shade, numbers) {
  const s = W / n;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const sv = clamp(shade(val[r * n + c]), 0, 1), g = sv > 0.5 ? 0 : 255;
    ctx.fillStyle = `rgb(${shadeRGB(sv)})`; ctx.fillRect(c * s, r * s, s, s);
    if (numbers) { ctx.fillStyle = g < 130 ? paper() : ink(); ctx.textAlign = 'center'; ctx.font = `600 ${Math.min(13, s * 0.36)}px ${SANS}`; ctx.fillText(val[r * n + c].toFixed(1), c * s + s / 2, r * s + s / 2 + 4); }
  }
  ctx.strokeStyle = ink(.18); ctx.lineWidth = 1; ctx.beginPath();
  for (let i = 1; i < n; i++) { ctx.moveTo(i * s, 0); ctx.lineTo(i * s, W); ctx.moveTo(0, i * s); ctx.lineTo(W, i * s); } ctx.stroke();
  return s;
}
function drawConv() {
  const { out, m } = conv2d(img, N, ker, 3), act = relu(out), pool = maxPool(act, m, 2), mx = Math.max(1e-9, ...act);
  let c = setup($('cvin')); const s1 = grid(c.ctx, c.W, N, img, v => v);
  let o = setup($('cvout')); const s2 = grid(o.ctx, o.W, m, act, v => v / mx);
  let p = setup($('cvpool')); grid(p.ctx, p.W, pool.m, pool.out, v => v / mx, true);
  if (cvHover) {
    const [r, cc] = cvHover; c.ctx.strokeStyle = cssv('--neg'); c.ctx.lineWidth = 3; c.ctx.strokeRect(cc * s1, r * s1, 3 * s1, 3 * s1);
    o.ctx.strokeStyle = cssv('--neg'); o.ctx.lineWidth = 3; o.ctx.strokeRect(cc * s2, r * s2, s2, s2);
    const terms = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (img[(r + i) * N + cc + j]) terms.push(ker[i * 3 + j]);
    $('cvcalc').innerHTML = `Patch at row ${r + 1}, column ${cc + 1}:<br>sum of kernel weights under ink = <b>${out[r * m + cc].toFixed(2)}</b><br>after ReLU = <b>${act[r * m + cc].toFixed(2)}</b>`;
  } else $('cvcalc').innerHTML = 'Hover or tap a feature-map cell to see the patch it was computed from.';
  $('cvnote').innerHTML = `Strongest response: <b>${mx.toFixed(2)}</b>. Parameters used: <b>9</b> shared weights, against 14,400 for a dense layer between the same sizes.`;
}
const cellOf = (e, el, n) => { const r = el.getBoundingClientRect(); return [clamp(Math.floor((e.clientY - r.top) / r.height * n), 0, n - 1), clamp(Math.floor((e.clientX - r.left) / r.width * n), 0, n - 1)]; };
$('cvin').addEventListener('pointerdown', e => { const [r, c] = cellOf(e, e.currentTarget, N); painting = img[r * N + c] ? 0 : 1; img[r * N + c] = painting; e.currentTarget.setPointerCapture(e.pointerId); drawConv(); });
$('cvin').addEventListener('pointermove', e => { if (painting === null) return; const [r, c] = cellOf(e, e.currentTarget, N); if (img[r * N + c] !== painting) { img[r * N + c] = painting; drawConv(); } });
['pointerup', 'pointercancel'].forEach(t => $('cvin').addEventListener(t, () => { painting = null; }));
const hov = e => { cvHover = cellOf(e, e.currentTarget, N - 2); drawConv(); };
$('cvout').addEventListener('pointermove', hov); $('cvout').addEventListener('pointerdown', hov);
$('cvout').addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') { cvHover = null; drawConv(); } });

/* ================= 5. RNN ================= */
['rnw', 'rnu', 'rng', 'rnT'].forEach(id => $(id).addEventListener('input', drawRnn));
function drawRnn() {
  const w = +$('rnw').value, u = +$('rnu').value, g = +$('rng').value, T = +$('rnT').value;
  $('rnwv').textContent = w.toFixed(2); $('rnuv').textContent = u.toFixed(2); $('rngv').textContent = g.toFixed(2); $('rnTv').textContent = T;
  const xs = Array.from({ length: T }, (_, t) => (t === 0 ? 1 : 0)), R = rnnRun(xs, w, u, g);
  const first = Math.abs(R.infl[1]);
  $('rnout').textContent = `h after the pulse      = ${R.h[1].toFixed(3)}\nh at the last step (T)  = ${R.h[T].toFixed(3)}\ninfluence ∂h_T/∂h_1     = ${first.toExponential(2)}\n${first < 1e-3 ? 'The start of the sequence is forgotten.' : first > 0.05 ? 'The start of the sequence is still remembered.' : 'The memory of the start is fading.'}`;
  const { ctx, W, H } = setup($('rncv')), padL = 36, padR = 50, padB = 24, pw = W - padL - padR, ph = H - padB - 12;
  const X = t => padL + t / T * pw, Y = v => 12 + (1 - v) * ph, LO = -12, Yl = v => 12 + (0 - clamp(v, LO, 0)) / (0 - LO) * ph;
  ctx.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) { ctx.strokeStyle = cssv('--line'); ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - padR, Y(v)); ctx.stroke(); ctx.fillStyle = cssv('--fg'); ctx.textAlign = 'right'; ctx.fillText(v.toFixed(2), padL - 5, Y(v) + 4); }
  ctx.fillStyle = cssv('--neg'); ctx.textAlign = 'left'; for (const v of [0, -4, -8, -12]) ctx.fillText(v === 0 ? '1' : `10^${v}`, W - padR + 6, Yl(v) + 4);
  ctx.fillStyle = cssv('--mut'); ctx.textAlign = 'left'; ctx.fillText('step 0', padL, H - 6); ctx.textAlign = 'right'; ctx.fillText(`step ${T}`, W - padR, H - 6);
  const line = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2.6; ctx.beginPath(); for (let t = 0; t <= T; t++) { const y = fn(t); t ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y); } ctx.stroke(); };
  line(t => Yl(Math.log10(Math.max(1e-300, Math.abs(R.infl[t])))), cssv('--neg'));
  line(t => Y(clamp(Math.abs(R.h[t]), 0, 1)), cssv('--fg'));
}

/* ================= 6. ATTENTION ================= */
const TOK = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired'];
const EMB = [[0, 0, 0, 1], [1, 0, .1, 0], [.2, .1, 1, 0], [0, .3, 0, .9], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, .1, 1], [.9, .25, 0, .15], [.1, 0, .7, .6], [.85, 0, .35, 0]];
let atQ = 7, atCausal = false;
pills($('atmode'), [['0', 'Full self-attention (BERT-style)'], ['1', 'Causal mask (GPT-style)']], '0', k => { atCausal = k === '1'; drawAtt(); });
$('atbeta').addEventListener('input', drawAtt);
function drawAtt() {
  const beta = +$('atbeta').value; $('atbetav').textContent = beta.toFixed(1);
  const { weights } = attention(EMB, beta, atCausal), w = weights[atQ], n = TOK.length;
  const { ctx, W, H } = setup($('atcv')), cw = W / n, fs = clamp(cw / 4.6, 9.5, 16), yT = 34, yB = H - 52;
  const box = (i, y, on, shade) => { const x = i * cw + 3, bw = cw - 6; ctx.fillStyle = on ? cssv('--neg') : ink(0.04 + 0.85 * shade); ctx.fillRect(x, y - 17, bw, 28); ctx.strokeStyle = cssv('--fg'); ctx.lineWidth = 1; ctx.strokeRect(x + .5, y - 16.5, bw - 1, 27); ctx.fillStyle = on || shade > 0.45 ? paper() : cssv('--fg'); ctx.textAlign = 'center'; ctx.font = `700 ${fs}px ${SANS}`; ctx.fillText(TOK[i], x + bw / 2, y + 2); };
  ctx.lineCap = 'round';
  w.forEach((v, j) => { if (v < 0.004) return; ctx.strokeStyle = ink(0.15 + 0.85 * v); ctx.lineWidth = 1 + 13 * v; ctx.beginPath(); ctx.moveTo(atQ * cw + cw / 2, yT + 13); ctx.lineTo(j * cw + cw / 2, yB - 19); ctx.stroke(); });
  TOK.forEach((_, i) => { box(i, yT, i === atQ, 0); box(i, yB, false, w[i]); ctx.fillStyle = cssv('--fg'); ctx.font = `600 ${clamp(fs - 1, 9.5, 13)}px ${SANS}`; ctx.fillText((w[i] * 100).toFixed(0) + '%', i * cw + cw / 2, yB + 30); });
  ctx.fillStyle = cssv('--mut'); ctx.textAlign = 'left'; ctx.font = `600 11.5px ${SANS}`; ctx.fillText('query (click a word)', 4, 11); ctx.fillText('attention paid to each word', 4, H - 4);
  const m = setup($('atmat')), s = m.W / n;
  weights.forEach((row, i) => row.forEach((v, j) => { m.ctx.fillStyle = `rgb(${shadeRGB(v)})`; m.ctx.fillRect(j * s, i * s, s, s); }));
  m.ctx.strokeStyle = ink(.15); m.ctx.lineWidth = 1; m.ctx.beginPath(); for (let i = 1; i < n; i++) { m.ctx.moveTo(i * s, 0); m.ctx.lineTo(i * s, m.W); m.ctx.moveTo(0, i * s); m.ctx.lineTo(m.W, i * s); } m.ctx.stroke();
  m.ctx.strokeStyle = cssv('--neg'); m.ctx.lineWidth = 3; m.ctx.strokeRect(1.5, atQ * s + 1.5, m.W - 3, s - 3);
  const top = w.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  $('atnote').innerHTML = `Query: <b>"${TOK[atQ]}"</b> (word ${atQ + 1})\nIt attends most to: ${top.map(([v, j]) => `<b>${TOK[j]}</b> ${(v * 100).toFixed(0)}%`).join(', ')}\n${atCausal ? 'Causal mask on: words to the right get exactly 0%.' : 'Every word can see the whole sentence.'}${beta === 0 ? '\nβ = 0: all words weighted equally (a plain average).' : ''}`;
}
$('atcv').addEventListener('click', e => { const r = e.currentTarget.getBoundingClientRect(); atQ = clamp(Math.floor((e.clientX - r.left) / r.width * TOK.length), 0, TOK.length - 1); drawAtt(); });
$('atmat').addEventListener('click', e => { const r = e.currentTarget.getBoundingClientRect(); atQ = clamp(Math.floor((e.clientY - r.top) / r.height * TOK.length), 0, TOK.length - 1); drawAtt(); });

/* ================= QUIZ ================= */
const QUIZ = [
  [true, 'With two classes, softmax gives the same probabilities as a sigmoid applied to the difference of the two logits.', 'p₁ = e^{z₁}/(e^{z₁}+e^{z₂}) = 1/(1+e^{−(z₁−z₂)}).'],
  [false, 'Raising the softmax temperature makes the predicted distribution more peaked.', 'Higher T flattens it toward uniform. Lower T sharpens it toward the largest logit.'],
  [false, 'Vanishing gradients are caused by the learning rate being too small.', 'They come from multiplying many factors below one (activation slopes times weights) during backpropagation, whatever the learning rate.'],
  [true, 'A skip connection gives the gradient a path back that is not multiplied by the layer\'s weights.', 'With a = x + f(Wx), ∂a/∂x = I + (…): the identity term carries the gradient through unchanged.'],
  [false, 'Dropout is applied both during training and when making predictions.', 'Only during training. At test time all neurons are used; the 1/(1−p) scaling in training keeps the expected activations equal.'],
  [true, 'A 3×3 convolution kernel uses the same nine weights at every position of the image.', 'This weight sharing is why a convolutional layer has far fewer parameters than a dense layer, and why it detects a pattern wherever it appears.'],
  [true, 'A plain recurrent network struggles with long sequences because backpropagation through time multiplies one factor per step.', 'If those factors are below one, the influence of early steps vanishes; gates (LSTM/GRU) or attention fix this.'],
  [true, 'In a causal (GPT-style) attention mask, a word gets zero attention weight on the words that come after it.', 'Their scores are set to −∞ before the softmax, so the model cannot look at the future while predicting the next word.'],
  [false, 'Attention weights for one query word can sum to more than one.', 'They come from a softmax, so they are positive and sum to exactly one: the output is a weighted average.'],
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

/* ================= page ================= */
const tocLinks = [...document.querySelectorAll('#toc a')], sections = tocLinks.map(a => document.querySelector(a.getAttribute('href')));
function onScroll() {
  let cur = 0; sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top < 140) cur = i; });
  tocLinks.forEach((a, i) => a.classList.toggle('on', i === cur));
  const h = document.documentElement; $('prog').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%';
}
window.addEventListener('scroll', onScroll, { passive: true });
function drawAll() { drawSoftmax(); drawDeep(); drawDrop(); drawConv(); drawRnn(); drawAtt(); }
window.addEventListener('themechange', () => { rebuildColormap(); drawAll(); });
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
fillKer(); loadImg('cross'); resetDrop(true); drawQuiz(); drawAll(); onScroll();
if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawAll);
(function loop() { if (drRun) { stepDrop(); drawDrop(); } requestAnimationFrame(loop); })();
