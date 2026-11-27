// Pure maths for the Advanced section. No DOM, so tools/check_advanced.mjs can verify it.
import { Network, ACTIVATIONS } from './nn.js?v=202611261907';
import { rng, normalFrom } from './data.js?v=202611261907';

/* ---------- softmax ---------- */
export function softmax(logits, T = 1) {
  const m = Math.max(...logits), e = logits.map(z => Math.exp((z - m) / T)), s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
export const crossEntropy = (p, k) => -Math.log(Math.max(1e-300, p[k]));
// d(cross-entropy)/d(logit_j) at temperature T
export const softmaxGrad = (p, k, T = 1) => p.map((v, j) => (v - (j === k ? 1 : 0)) / T);

/* ---------- deep network signal / gradient propagation ---------- */
const F = {
  sigmoid: [z => 1 / (1 + Math.exp(-z)), (z, a) => a * (1 - a)],
  tanh: [Math.tanh, (z, a) => 1 - a * a],
  relu: [z => (z > 0 ? z : 0), z => (z > 0 ? 1 : 0)],
};
export const INIT = { tiny: n => 0.1 / n, xavier: n => 1 / n, he: n => 2 / n, large: n => 6 / n };
export function makeDeep(depth, width, init, seed) {
  const N = normalFrom(rng(seed)), sd = Math.sqrt(INIT[init](width));
  return Array.from({ length: depth }, () => Array.from({ length: width }, () => Float64Array.from({ length: width }, () => N() * sd)));
}
// Forward one input through `Ws`; returns activations per layer. residual: a_l = a_{l-1} + f(W a_{l-1}).
export function deepForward(Ws, x, act, residual) {
  const [f] = F[act], as = [Float64Array.from(x)], zs = [null];
  for (const W of Ws) {
    const prev = as[as.length - 1], n = W.length, z = new Float64Array(n), a = new Float64Array(n);
    for (let j = 0; j < n; j++) { let s = 0; const w = W[j]; for (let i = 0; i < n; i++) s += w[i] * prev[i]; z[j] = s; a[j] = f(s) + (residual ? prev[j] : 0); }
    zs.push(z); as.push(a);
  }
  return { as, zs };
}
// Backward: given dL/da at the last layer, returns dL/da at every layer (index 0 = input).
export function deepBackward(Ws, fw, dOut, act, residual) {
  const [f, d] = F[act], L = Ws.length, ds = new Array(L + 1); ds[L] = Float64Array.from(dOut);
  for (let l = L; l >= 1; l--) {
    const W = Ws[l - 1], n = W.length, up = ds[l], z = fw.zs[l], dz = new Float64Array(n), dp = new Float64Array(n);
    for (let j = 0; j < n; j++) dz[j] = up[j] * d(z[j], f(z[j]));
    for (let i = 0; i < n; i++) { let s = residual ? up[i] : 0; for (let j = 0; j < n; j++) s += W[j][i] * dz[j]; dp[i] = s; }
    ds[l - 1] = dp;
  }
  return ds;
}
export const rms = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
// Average RMS of activations and of gradients per layer over several random inputs.
export function deepProfile({ depth, width = 32, act, init, residual, seed = 1, samples = 12 }) {
  const Ws = makeDeep(depth, width, init, seed), N = normalFrom(rng(seed + 99));
  const actR = new Float64Array(depth + 1), grdR = new Float64Array(depth + 1);
  for (let s = 0; s < samples; s++) {
    const x = Array.from({ length: width }, () => N()), g = Array.from({ length: width }, () => N());
    const fw = deepForward(Ws, x, act, residual), ds = deepBackward(Ws, fw, g, act, residual);
    for (let l = 0; l <= depth; l++) { actR[l] += rms(fw.as[l]) / samples; grdR[l] += rms(ds[l]) / samples; }
  }
  return { act: Array.from(actR), grad: Array.from(grdR) };
}

/* ---------- dropout (inverted), as a subclass of the site's engine ---------- */
export class DropNet extends Network {
  constructor(sizes, act, problem, seed, p = 0) { super(sizes, act, problem, seed); this.p = p; this.r = rng(seed + 7); this.mask = sizes.map(n => new Float64Array(n).fill(1)); this.raw = sizes.map(n => new Float64Array(n)); }
  sampleMask() { const L = this.sizes.length - 1, k = 1 - this.p; for (let l = 1; l < L; l++) for (let j = 0; j < this.mask[l].length; j++) this.mask[l][j] = this.r() < k ? 1 / k : 0; }
  // Training-mode forward: hidden activations are multiplied by the current mask.
  forwardTrain(x) {
    const A = ACTIVATIONS[this.act], L = this.sizes.length - 1;
    for (let i = 0; i < x.length; i++) this.a[0][i] = x[i];
    for (let l = 1; l <= L; l++) {
      const W = this.W[l - 1], b = this.b[l - 1], prev = this.a[l - 1];
      for (let j = 0; j < W.length; j++) {
        let s = b[j]; const w = W[j]; for (let i = 0; i < w.length; i++) s += w[i] * prev[i];
        this.z[l][j] = s;
        if (l === L) this.a[l][j] = this.outAct(s);
        else { this.raw[l][j] = A.f(s); this.a[l][j] = this.raw[l][j] * this.mask[l][j]; }
      }
    }
    return this.a[L][0];
  }
  backwardTrain(y) {
    const A = ACTIVATIONS[this.act], L = this.sizes.length - 1;
    this.delta[L][0] = this.a[L][0] - (this.problem === 'class' ? (y > 0 ? 1 : 0) : y);
    for (let l = L; l >= 1; l--) {
      const W = this.W[l - 1], gW = this.gW[l - 1], gb = this.gb[l - 1], d = this.delta[l], prev = this.a[l - 1];
      for (let j = 0; j < W.length; j++) { gb[j] += d[j]; const g = gW[j]; for (let i = 0; i < g.length; i++) g[i] += d[j] * prev[i]; }
      if (l > 1) {
        const dp = this.delta[l - 1];
        for (let i = 0; i < dp.length; i++) { let s = 0; for (let j = 0; j < W.length; j++) s += W[j][i] * d[j]; dp[i] = s * this.mask[l - 1][i] * A.d(this.z[l - 1][i], this.raw[l - 1][i]); }
      }
    }
    this.nAcc++;
  }
  trainEpoch(data, opts, batch = 10) {
    data.forEach((e, n) => { this.sampleMask(); this.forwardTrain(e.input); this.backwardTrain(e.label); if ((n + 1) % batch === 0) this.update(opts); });
    this.update(opts);
  }
}

/* ---------- convolution ---------- */
// "Valid" cross-correlation of an n×n image (row-major array) with a k×k kernel, plus bias.
export function conv2d(img, n, ker, k, bias = 0) {
  const m = n - k + 1, out = new Float64Array(m * m);
  for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) { let s = bias; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) s += img[(r + i) * n + c + j] * ker[i * k + j]; out[r * m + c] = s; }
  return { out, m };
}
export const relu = a => a.map(v => (v > 0 ? v : 0));
export function maxPool(a, m, s = 2) {
  const q = Math.floor(m / s), out = new Float64Array(q * q);
  for (let r = 0; r < q; r++) for (let c = 0; c < q; c++) { let best = -Infinity; for (let i = 0; i < s; i++) for (let j = 0; j < s; j++) best = Math.max(best, a[(r * s + i) * m + c * s + j]); out[r * q + c] = best; }
  return { out, m: q };
}

/* ---------- recurrent memory ---------- */
// h_t = (1 - g) h_{t-1} + g tanh(w h_{t-1} + u x_t). g = 1 is a plain RNN; small g is a gated "carry".
export function rnnRun(xs, w, u, g = 1, h0 = 0) {
  const h = [h0], local = [];
  for (const x of xs) { const prev = h[h.length - 1], c = Math.tanh(w * prev + u * x); h.push((1 - g) * prev + g * c); local.push((1 - g) + g * w * (1 - c * c)); }
  // influence[t] = d h_T / d h_t  (product of the local derivatives after step t)
  const T = xs.length, infl = new Array(T + 1); infl[T] = 1;
  for (let t = T - 1; t >= 0; t--) infl[t] = infl[t + 1] * local[t];
  return { h, local, infl };
}

/* ---------- scaled dot-product self-attention ---------- */
export function attention(E, beta = 1, causal = false) {
  const n = E.length, d = E[0].length, Wt = [], out = [];
  for (let i = 0; i < n; i++) {
    const sc = E.map((k, j) => (causal && j > i ? -Infinity : beta * E[i].reduce((s, v, t) => s + v * k[t], 0) / Math.sqrt(d)));
    const m = Math.max(...sc), ex = sc.map(v => Math.exp(v - m)), s = ex.reduce((a, b) => a + b, 0), w = ex.map(v => v / s);
    Wt.push(w); out.push(Array.from({ length: d }, (_, t) => w.reduce((acc, wj, j) => acc + wj * E[j][t], 0)));
  }
  return { weights: Wt, out };
}
