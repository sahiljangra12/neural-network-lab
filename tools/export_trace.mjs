// Records complete training runs of the site's engine (js/nn.js) so that an independent
// implementation (tools/crosscheck.py, JAX + Optax) can replay them exactly:
// initial parameters, the training data, the shuffle order of every epoch, the loss after
// every epoch and the final parameters.   node tools/export_trace.mjs
import { writeFileSync } from 'node:fs';
import { Network } from '../js/nn.js';
import { makeData, FEATURES, rng } from '../js/data.js';
import { BP, bpStep } from '../js/backprop.js';

const ALL = FEATURES.map(f => f.id);
const C = (name, o) => Object.assign({ name, problem: 'class', dataset: 'circle', layers: [4, 3], act: 'tanh', opt: 'sgd', lr: 0.03, reg: 'none', regRate: 0, batch: 10, features: ['x1', 'x2'], noise: 0.1, epochs: 40 }, o);
const CONFIGS = [
  C('tanh · SGD', {}),
  C('ReLU · SGD', { act: 'relu' }),
  C('Leaky ReLU · SGD', { act: 'leaky' }),
  C('Sigmoid · SGD', { act: 'sigmoid', lr: 0.1 }),
  C('Linear · SGD', { act: 'linear', lr: 0.01 }),
  C('No hidden layer (logistic regression)', { layers: [], dataset: 'gauss', lr: 0.1 }),
  C('Momentum', { opt: 'momentum', lr: 0.01, dataset: 'xor' }),
  C('Adam', { opt: 'adam', lr: 0.01, act: 'relu', dataset: 'moons' }),
  C('L2 penalty · SGD', { reg: 'l2', regRate: 0.01 }),
  C('L2 penalty · Adam', { reg: 'l2', regRate: 0.003, opt: 'adam', lr: 0.01 }),
  C('L1 penalty · SGD', { reg: 'l1', regRate: 0.003, features: ALL }),
  C('L1 penalty · Momentum', { reg: 'l1', regRate: 0.001, opt: 'momentum', lr: 0.01, features: ALL }),
  C('Batch size 1', { batch: 1, lr: 0.01, epochs: 15 }),
  C('Batch size 7 (partial last batch)', { batch: 7 }),
  C('Deep network, all features, Adam', { dataset: 'spiral', layers: [8, 8, 8, 6], features: ALL, opt: 'adam', lr: 0.01, epochs: 30 }),
  C('Regression · tanh', { problem: 'reg', dataset: 'multigauss', layers: [8, 6] }),
  C('Regression · ReLU · Adam', { problem: 'reg', dataset: 'multigauss', layers: [8, 6], act: 'relu', opt: 'adam', lr: 0.01 }),
  C('Regression · sigmoid hidden', { problem: 'reg', dataset: 'plane', act: 'sigmoid', lr: 0.1 }),
];

const out = [];
CONFIGS.forEach((cfg, ci) => {
  const feats = FEATURES.filter(f => cfg.features.includes(f.id));
  const pts = makeData(cfg.problem, cfg.dataset, 200, cfg.noise, 7 + ci).slice(0, 100);
  const X = pts.map(p => feats.map(f => f.f(p.x, p.y))), Y = pts.map(p => p.label);
  const net = new Network([feats.length, ...cfg.layers, 1], cfg.act, cfg.problem, 300 + ci);
  const init = { W: structuredClone(net.W), b: structuredClone(net.b) };
  const r = rng(900 + ci), orders = [], losses = [];
  const opts = { lr: cfg.lr, optimizer: cfg.opt, reg: cfg.reg, regRate: cfg.regRate };
  const ex = X.map((x, i) => ({ input: x, label: Y[i] }));
  for (let e = 0; e < cfg.epochs; e++) {
    const order = X.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    orders.push(order);
    // identical to trainEpoch() in js/playground.js
    order.forEach((idx, n) => { net.forward(X[idx]); net.backward(Y[idx]); if ((n + 1) % cfg.batch === 0) net.update(opts); });
    net.update(opts);
    losses.push(net.loss(ex));
  }
  out.push({ ...cfg, X, Y, init, orders, losses, final: { W: net.W, b: net.b } });
});
// Long full-batch runs with no hidden layer: these must converge to the closed-form
// statistical estimates (logistic-regression MLE; ordinary least squares).
const closed = [];
for (const [problem, dataset, lr, noise] of [['class', 'gauss', 0.2, 0.3], ['reg', 'plane', 0.05, 0.15]]) {
  const pts = makeData(problem, dataset, 400, noise, 42).slice(0, 300);
  const X = pts.map(p => [p.x, p.y]), Y = pts.map(p => p.label);
  const net = new Network([2, 1], 'tanh', problem, 5);
  const opts = { lr, optimizer: 'sgd' };
  for (let e = 0; e < 20000; e++) { X.forEach((x, i) => { net.forward(x); net.backward(Y[i]); }); net.update(opts); }
  closed.push({ problem, dataset, X, Y, w: net.W[0][0], b: net.b[0][0] });
}
// The Academy's backpropagation lesson computes its own numbers: record them for several inputs.
const bp = [];
const rb = rng(4242);
for (let k = 0; k < 25; k++) {
  const x = [Math.round((rb() * 6 - 3) * 10) / 10, Math.round((rb() * 6 - 3) * 10) / 10], y = k % 2, lr = [0.1, 0.5, 1][k % 3];
  bp.push(bpStep(x, y, lr));
}
writeFileSync(new URL('../paper/crosscheck_trace.json', import.meta.url), JSON.stringify({ runs: out, closed, bp, bpInit: BP }));
console.log(`recorded ${out.length} runs`);
