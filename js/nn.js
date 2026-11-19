// A small, dependency-free multilayer perceptron with backpropagation.
// Classification: sigmoid output + log loss (the output neuron IS logistic regression).
// Regression: linear output + squared error.

import { rng, normalFrom } from './data.js?v=202611190245';

export const ACTIVATIONS = {
  relu: { name: 'ReLU', f: z => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0), formula: 'max(0, z)' },
  leaky: { name: 'Leaky ReLU', f: z => (z > 0 ? z : 0.01 * z), d: z => (z > 0 ? 1 : 0.01), formula: 'max(0.01z, z)' },
  tanh: { name: 'Tanh', f: Math.tanh, d: (z, a) => 1 - a * a, formula: '(eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)' },
  sigmoid: { name: 'Sigmoid', f: z => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a), formula: '1/(1 + e⁻ᶻ)' },
  linear: { name: 'Linear', f: z => z, d: () => 1, formula: 'z' },
};

const sigmoid = z => 1 / (1 + Math.exp(-z));

export class Network {
  /**
   * @param {number[]} sizes  e.g. [2, 4, 2, 1]
   * @param {string} act      hidden activation key
   * @param {'class'|'reg'} problem
   * @param {number} seed
   */
  constructor(sizes, act = 'tanh', problem = 'class', seed = 1) {
    this.sizes = sizes.slice();
    this.act = act;
    this.problem = problem;
    const r = rng(seed), N = normalFrom(r);
    this.W = []; this.b = [];
    for (let l = 1; l < sizes.length; l++) {
      const nIn = sizes[l - 1], nOut = sizes[l];
      const scale = Math.sqrt(1 / Math.max(1, nIn));
      this.W.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => N() * scale)));
      this.b.push(Array.from({ length: nOut }, () => 0.1));
    }
    this.resetGrads();
    this.mW = this.zeroLike(this.W); this.vW = this.zeroLike(this.W);
    this.mb = this.zeroLike(this.b); this.vb = this.zeroLike(this.b);
    this.t = 0;
    // forward caches
    this.z = sizes.map(n => new Float64Array(n));
    this.a = sizes.map(n => new Float64Array(n));
    this.delta = sizes.map(n => new Float64Array(n));
  }

  zeroLike(M) { return M.map(row => Array.isArray(row[0]) ? row.map(r => r.map(() => 0)) : row.map(() => 0)); }
  resetGrads() { this.gW = this.zeroLike(this.W); this.gb = this.zeroLike(this.b); this.nAcc = 0; }

  outAct(z) { return this.problem === 'class' ? sigmoid(z) : z; }

  forward(x) {
    const A = ACTIVATIONS[this.act], L = this.sizes.length - 1;
    const a0 = this.a[0];
    for (let i = 0; i < x.length; i++) a0[i] = x[i];
    for (let l = 1; l <= L; l++) {
      const W = this.W[l - 1], b = this.b[l - 1], prev = this.a[l - 1], z = this.z[l], a = this.a[l];
      for (let j = 0; j < W.length; j++) {
        let s = b[j]; const w = W[j];
        for (let i = 0; i < w.length; i++) s += w[i] * prev[i];
        z[j] = s;
        a[j] = l === L ? this.outAct(s) : A.f(s);
      }
    }
    return this.a[L][0];
  }

  // Accumulate gradient for one example (target y: class ±1, or regression value).
  backward(y) {
    const A = ACTIVATIONS[this.act], L = this.sizes.length - 1;
    const out = this.a[L][0];
    // For sigmoid + log loss and for linear + 0.5·squared error, dL/dz_out = prediction − target.
    const target = this.problem === 'class' ? (y > 0 ? 1 : 0) : y;
    this.delta[L][0] = out - target;
    for (let l = L; l >= 1; l--) {
      const W = this.W[l - 1], gW = this.gW[l - 1], gb = this.gb[l - 1];
      const d = this.delta[l], prev = this.a[l - 1];
      for (let j = 0; j < W.length; j++) {
        gb[j] += d[j];
        const g = gW[j];
        for (let i = 0; i < g.length; i++) g[i] += d[j] * prev[i];
      }
      if (l > 1) {
        const dp = this.delta[l - 1], zp = this.z[l - 1], ap = this.a[l - 1];
        for (let i = 0; i < dp.length; i++) {
          let s = 0;
          for (let j = 0; j < W.length; j++) s += W[j][i] * d[j];
          dp[i] = s * A.d(zp[i], ap[i]);
        }
      }
    }
    this.nAcc++;
  }

  // Apply accumulated gradients.
  update({ lr, optimizer = 'sgd', reg = 'none', regRate = 0 }) {
    if (!this.nAcc) return;
    this.t++;
    const n = this.nAcc, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const step = (val, grad, m, v, isWeight) => {
      let g = grad / n;
      if (isWeight && reg === 'l2') g += regRate * val;
      if (isWeight && reg === 'l1') g += regRate * Math.sign(val);
      if (optimizer === 'momentum') { m.val = 0.9 * m.val + g; return val - lr * m.val; }
      if (optimizer === 'adam') {
        m.val = b1 * m.val + (1 - b1) * g; v.val = b2 * v.val + (1 - b2) * g * g;
        const mh = m.val / (1 - b1 ** this.t), vh = v.val / (1 - b2 ** this.t);
        return val - lr * mh / (Math.sqrt(vh) + eps);
      }
      return val - lr * g;
    };
    const box = { val: 0 }, box2 = { val: 0 };
    for (let l = 0; l < this.W.length; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        for (let i = 0; i < this.W[l][j].length; i++) {
          box.val = this.mW[l][j][i]; box2.val = this.vW[l][j][i];
          let w = step(this.W[l][j][i], this.gW[l][j][i], box, box2, true);
          // L1 can push weights through zero; clip to exactly 0 when it crosses (sparsity).
          if (reg === 'l1' && Math.sign(w) !== Math.sign(this.W[l][j][i]) && this.W[l][j][i] !== 0) w = 0;
          this.W[l][j][i] = w; this.mW[l][j][i] = box.val; this.vW[l][j][i] = box2.val;
        }
        box.val = this.mb[l][j]; box2.val = this.vb[l][j];
        this.b[l][j] = step(this.b[l][j], this.gb[l][j], box, box2, false);
        this.mb[l][j] = box.val; this.vb[l][j] = box2.val;
      }
    }
    this.resetGrads();
  }

  // Mean loss over examples [{input, label}]
  loss(examples) {
    if (!examples.length) return 0;
    let s = 0;
    for (const e of examples) {
      const p = this.forward(e.input);
      if (this.problem === 'class') {
        const t = e.label > 0 ? 1 : 0, q = Math.min(1 - 1e-12, Math.max(1e-12, p));
        s += -(t * Math.log(q) + (1 - t) * Math.log(1 - q));
      } else s += 0.5 * (p - e.label) ** 2;
    }
    return s / examples.length;
  }

  accuracy(examples) {
    if (!examples.length || this.problem !== 'class') return null;
    let c = 0;
    for (const e of examples) if ((this.forward(e.input) >= 0.5 ? 1 : -1) === e.label) c++;
    return c / examples.length;
  }

  weightCount() { return this.W.reduce((s, L) => s + L.length * L[0].length, 0) + this.b.reduce((s, b) => s + b.length, 0); }
}
