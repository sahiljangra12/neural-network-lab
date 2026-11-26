// Numerical checks of the Advanced section's maths (js/advmath.js).   node tools/check_advanced.mjs
import { softmax, crossEntropy, softmaxGrad, makeDeep, deepForward, deepBackward, DropNet, conv2d, maxPool, rnnRun, attention } from '../js/advmath.js';
import { Network } from '../js/nn.js';
import { rng } from '../js/data.js';

const r = rng(11), U = (a, b) => a + (b - a) * r();
let worst = {}, fail = 0;
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-9, Math.abs(a) + Math.abs(b));
const rec = (name, v, tol = 1e-5) => { worst[name] = Math.max(worst[name] || 0, v); if (!(v <= tol)) fail++; };

// 1. softmax: sums to one; analytic gradient of cross-entropy vs finite differences
for (let t = 0; t < 50; t++) {
  const z = Array.from({ length: 4 }, () => U(-3, 3)), T = U(0.2, 4), k = t % 4, p = softmax(z, T), g = softmaxGrad(p, k, T);
  rec('softmax sums to 1', Math.abs(p.reduce((a, b) => a + b, 0) - 1), 1e-12);
  z.forEach((_, j) => { const h = 1e-6, a = z.slice(), b = z.slice(); a[j] += h; b[j] -= h; rec('softmax gradient', rel(g[j], (crossEntropy(softmax(a, T), k) - crossEntropy(softmax(b, T), k)) / (2 * h))); });
}

// 2. deep network backward pass (plain and residual) vs finite differences of sum(g · a_L)
for (const act of ['sigmoid', 'tanh', 'relu']) for (const residual of [false, true]) {
  const Ws = makeDeep(6, 8, 'xavier', 5), x = Array.from({ length: 8 }, () => U(-1, 1)), g = Array.from({ length: 8 }, () => U(-1, 1));
  const obj = xx => deepForward(Ws, xx, act, residual).as[6].reduce((s, v, i) => s + v * g[i], 0);
  const ds = deepBackward(Ws, deepForward(Ws, x, act, residual), g, act, residual);
  x.forEach((_, i) => { const h = 1e-6, a = x.slice(), b = x.slice(); a[i] += h; b[i] -= h; rec('deep/residual backward', rel(ds[0][i], (obj(a) - obj(b)) / (2 * h)), 1e-4); });
}

// 3. dropout: p = 0 reproduces the site's engine exactly; with a fixed mask the gradient matches finite differences
{
  const a = new Network([2, 6, 5, 1], 'tanh', 'class', 3), b = new DropNet([2, 6, 5, 1], 'tanh', 'class', 3, 0);
  const data = Array.from({ length: 30 }, () => ({ input: [U(-3, 3), U(-3, 3)], label: r() < .5 ? 1 : -1 }));
  for (let e = 0; e < 20; e++) { data.forEach((d, n) => { a.forward(d.input); a.backward(d.label); if ((n + 1) % 10 === 0) a.update({ lr: 0.05 }); }); a.update({ lr: 0.05 }); b.trainEpoch(data, { lr: 0.05 }, 10); }
  rec('dropout p=0 equals engine', Math.max(...a.W.flat(2).map((w, i) => Math.abs(w - b.W.flat(2)[i]))), 1e-12);
  for (const act of ['tanh', 'relu', 'sigmoid']) {
    const n = new DropNet([3, 6, 5, 1], act, 'class', 9, 0.4); n.sampleMask();
    const x = [U(-2, 2), U(-2, 2), U(-2, 2)], y = 1, loss = () => -Math.log(n.forwardTrain(x));
    n.forwardTrain(x); n.backwardTrain(y);
    n.W.forEach((Lw, l) => Lw.forEach((row, j) => row.forEach((_, i) => { const h = 1e-6, o = row[i]; row[i] = o + h; const lp = loss(); row[i] = o - h; const lm = loss(); row[i] = o; const num = (lp - lm) / (2 * h); if (Math.abs(num) + Math.abs(n.gW[l][j][i]) > 1e-8) rec('dropout gradient', rel(n.gW[l][j][i], num), 1e-4); })));
  }
}

// 4. convolution vs an independent direct computation; max-pool
{
  const n = 12, img = Float64Array.from({ length: n * n }, () => U(0, 1)), ker = Array.from({ length: 9 }, () => U(-1, 1)), { out, m } = conv2d(img, n, ker, 3, 0.3);
  for (let t = 0; t < 40; t++) { const rr = Math.floor(U(0, m)), c = Math.floor(U(0, m)); let s = 0.3; for (let q = 0; q < 9; q++) s += img[(rr + Math.floor(q / 3)) * n + c + q % 3] * ker[q]; rec('convolution', Math.abs(s - out[rr * m + c]), 1e-12); }
  const pl = maxPool(out, m, 2); rec('max-pool', Math.abs(pl.out[0] - Math.max(out[0], out[1], out[m], out[m + 1])), 1e-12);
}

// 5. RNN influence d h_T / d h_0 vs finite differences
for (let t = 0; t < 30; t++) {
  const xs = Array.from({ length: 12 }, () => U(-1, 1)), w = U(0, 2.5), u = U(0.2, 2), g = U(0.05, 1), h = 1e-6;
  const num = (rnnRun(xs, w, u, g, h).h[12] - rnnRun(xs, w, u, g, -h).h[12]) / (2 * h);
  rec('RNN influence', rel(rnnRun(xs, w, u, g, 0).infl[0], num), 1e-4);
}

// 6. attention: rows sum to one, causal mask hides the future, outputs are convex combinations
{
  const E = Array.from({ length: 7 }, () => Array.from({ length: 4 }, () => U(0, 1)));
  for (const causal of [false, true]) {
    const { weights, out } = attention(E, 2.5, causal);
    weights.forEach((w, i) => { rec('attention rows sum to 1', Math.abs(w.reduce((a, b) => a + b, 0) - 1), 1e-12); if (causal) rec('causal mask', Math.max(0, ...w.slice(i + 1)), 0); });
    out.forEach((o, i) => o.forEach((v, t) => rec('attention output', Math.abs(v - weights[i].reduce((s, wj, j) => s + wj * E[j][t], 0)), 1e-12)));
  }
}

for (const [k, v] of Object.entries(worst)) console.log(k.padEnd(28), v.toExponential(1));
const sci = v => (v === 0 ? '0' : '$' + v.toExponential(1).replace(/e([-+]\d+)/, (_, e) => `\\times 10^{${+e}}`) + '$');
const LABEL = { 'softmax sums to 1': ['Softmax probabilities sum to one', 'exact identity'], 'softmax gradient': ['Softmax cross-entropy gradient $(p - y)/T$', 'finite differences'], 'deep/residual backward': ['Deep and residual backward pass', 'finite differences'], 'dropout p=0 equals engine': ['Dropout network with $p = 0$', "the site's engine, 20 epochs"], 'dropout gradient': ['Dropout gradient with a fixed mask', 'finite differences'], 'convolution': ['Convolution feature map', 'direct recomputation'], 'max-pool': ['Max-pooling', 'direct recomputation'], 'RNN influence': ['Recurrent influence $\\partial h_T/\\partial h_0$', 'finite differences'], 'attention rows sum to 1': ['Attention weights sum to one', 'exact identity'], 'attention output': ['Attention output is the weighted average', 'direct recomputation'], 'causal mask': ['Causal mask gives zero weight to later words', 'exact identity'] };
import('node:fs').then(fs => fs.writeFileSync(new URL('../paper/tab_advanced.tex', import.meta.url), Object.entries(worst).map(([k, v]) => `${LABEL[k][0]} & ${LABEL[k][1]} & ${sci(v)} \\\\\n`).join('')));
console.log(fail ? `FAILED: ${fail} checks outside tolerance` : 'ALL CHECKS PASSED');
process.exitCode = fail ? 1 : 0;
