// Reproducible experiments for the technical report.
// Runs the site's own engine (js/nn.js, js/data.js) in Node and writes
// paper/results.json and paper/stats.tex. Every number in the report comes from here.
//   node tools/benchmark.mjs
import { writeFileSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { Network, ACTIVATIONS } from '../js/nn.js';
import { makeData, FEATURES, rng } from '../js/data.js';
import { CH } from '../js/missions.js';

const SEEDS = [1, 2, 3, 4, 5];
const N_POINTS = 500;
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };

const DEFAULT = {
  problem: 'class', dataset: 'circle', regDataset: 'plane', lr: 0.03, act: 'tanh', opt: 'sgd',
  reg: 'none', regRate: 0, batch: 10, noise: 0, ratio: 50, layers: [4, 2], features: ['x1', 'x2'], seed: 1,
};

function setupData(S, nPoints = N_POINTS) {
  const feats = FEATURES.filter(f => S.features.includes(f.id));
  const name = S.problem === 'class' ? S.dataset : S.regDataset;
  const pts = makeData(S.problem, name, nPoints, S.noise / 100, S.seed);
  const nTrain = Math.round(pts.length * S.ratio / 100);
  const ex = p => ({ x: p.x, y: p.y, label: p.label, input: feats.map(f => f.f(p.x, p.y)) });
  return { feats, train: pts.slice(0, nTrain).map(ex), test: pts.slice(nTrain).map(ex) };
}

// Same procedure as the playground: shuffle, mini-batches, one update per batch.
function epoch(net, train, S, r) {
  const order = train.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const opts = { lr: S.lr, optimizer: S.opt, reg: S.reg, regRate: S.regRate };
  order.forEach((idx, n) => { const e = train[idx]; net.forward(e.input); net.backward(e.label); if ((n + 1) % S.batch === 0) net.update(opts); });
  net.update(opts);
}

function run(S, epochs, seed, { curve = false, data = null } = {}) {
  const { feats, train, test } = data || setupData(S);
  const net = new Network([feats.length, ...S.layers, 1], S.act, S.problem, 1000 + seed);
  const r = rng(77 + seed);
  const hist = { train: [], test: [] };
  let hit95 = null, bestTest = Infinity, bestEpoch = 0;
  for (let e = 1; e <= epochs; e++) {
    epoch(net, train, S, r);
    if (curve || e === epochs || (S.problem === 'class' && hit95 === null)) {
      const tl = net.loss(train), te = net.loss(test);
      if (curve) { hist.train.push(tl); hist.test.push(te); }
      if (isFinite(te) && te < bestTest) { bestTest = te; bestEpoch = e; }
      if (S.problem === 'class' && hit95 === null && net.accuracy(test) >= 0.95) hit95 = e;
    }
  }
  const out = { net, train, test, feats, hist, hit95, bestTest, bestEpoch,
    trainLoss: net.loss(train), testLoss: net.loss(test),
    trainAcc: net.accuracy(train), testAcc: net.accuracy(test), params: net.weightCount() };
  return out;
}

function grid(net, feats, problem, res = 60) {
  const g = [];
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const x = -6 + (i + .5) * 12 / res, y = 6 - (j + .5) * 12 / res;
    const p = net.forward(feats.map(f => f.f(x, y)));
    g.push(problem === 'class' ? 2 * p - 1 : p);
  }
  return g;
}

const R = {};
const T0 = Date.now();

/* ---------- E1: gradient check ---------- */
{
  const cases = [];
  for (const act of Object.keys(ACTIVATIONS)) for (const problem of ['class', 'reg']) {
    const r = rng(500 + cases.length);
    const net = new Network([3, 5, 4, 1], act, problem, 900 + cases.length);
    const x = [r() * 4 - 2, r() * 4 - 2, r() * 4 - 2], label = problem === 'class' ? (r() < .5 ? 1 : -1) : r() * 2 - 1;
    const ex = [{ input: x, label }];
    net.forward(x); net.backward(label);
    const analytic = [];
    net.W.forEach((L, l) => L.forEach((row, j) => row.forEach((_, i) => analytic.push([['W', l, j, i], net.gW[l][j][i]]))));
    net.b.forEach((L, l) => L.forEach((_, j) => analytic.push([['b', l, j], net.gb[l][j]])));
    let maxRel = 0, n = 0; const h = 1e-6;
    for (const [[k, l, j, i], a] of analytic) {
      const ref = k === 'W' ? net.W[l][j] : net.b[l];
      const idx = k === 'W' ? i : j, old = ref[idx];
      ref[idx] = old + h; const lp = net.loss(ex);
      ref[idx] = old - h; const lm = net.loss(ex);
      ref[idx] = old;
      const num = (lp - lm) / (2 * h);
      const rel = Math.abs(a - num) / Math.max(1e-7, Math.abs(a) + Math.abs(num));
      if (Math.abs(a) + Math.abs(num) > 1e-9) { maxRel = Math.max(maxRel, rel); n++; }
    }
    net.resetGrads();
    cases.push({ act, problem, params: analytic.length, checked: n, maxRel });
  }
  R.gradcheck = { cases, maxRel: Math.max(...cases.map(c => c.maxRel)), total: cases.reduce((s, c) => s + c.checked, 0) };
}

/* ---------- E2: missions ---------- */
{
  R.missions = CH.map((c, i) => {
    const S = Object.assign(structuredClone(DEFAULT), structuredClone(c.cfg));
    const epochs = c.cfg.dataset === 'spiral' ? 1000 : 400;
    const runs = SEEDS.map(s => run(S, epochs, s));
    const fin = runs.map(r => r.testLoss).filter(isFinite);
    return {
      n: i + 1, title: c.title, problem: S.problem, dataset: S.problem === 'class' ? S.dataset : S.regDataset,
      layers: S.layers, act: S.act, opt: S.opt, lr: S.lr, reg: S.reg, regRate: S.regRate, epochs,
      params: runs[0].params, features: S.features.length,
      testAccMean: S.problem === 'class' ? mean(runs.map(r => r.testAcc)) : null,
      testAccSd: S.problem === 'class' ? sd(runs.map(r => r.testAcc)) : null,
      testLossMean: fin.length ? mean(fin) : null, diverged: runs.length - fin.length,
      trainLossMean: mean(runs.map(r => r.trainLoss).filter(isFinite)),
      hit95: runs.map(r => r.hit95),
    };
  });
}

/* ---------- E3: linear vs engineered features vs hidden layer ---------- */
{
  const MODELS = [
    { key: 'lin', name: 'Logistic regression (X1, X2)', layers: [], features: ['x1', 'x2'], act: 'tanh' },
    { key: 'feat', name: 'Logistic regression + engineered features', layers: [], features: ['x1', 'x2', 'x1sq', 'x2sq', 'x1x2'], act: 'tanh' },
    { key: 'linhid', name: '4 linear hidden neurons', layers: [4], features: ['x1', 'x2'], act: 'linear' },
    { key: 'tanh', name: '4 tanh hidden neurons', layers: [4], features: ['x1', 'x2'], act: 'tanh' },
  ];
  const DS = ['gauss', 'circle', 'xor', 'moons'];
  R.capacity = { models: MODELS.map(m => ({ key: m.key, name: m.name })), datasets: DS, acc: {}, sd: {}, grids: {} };
  for (const ds of DS) for (const m of MODELS) {
    const S = Object.assign(structuredClone(DEFAULT), { dataset: ds, layers: m.layers, features: m.features, act: m.act, lr: 0.03 });
    const runs = SEEDS.map(s => run(S, 400, s));
    R.capacity.acc[`${ds}.${m.key}`] = mean(runs.map(r => r.testAcc));
    R.capacity.sd[`${ds}.${m.key}`] = sd(runs.map(r => r.testAcc));
    R.capacity.grids[`${ds}.${m.key}`] = { g: grid(runs[0].net, runs[0].feats, 'class'), pts: runs[0].train.slice(0, 160).map(p => [p.x, p.y, p.label]) };
  }
}

/* ---------- E4: overfitting and L2 ---------- */
{
  const LAMS = [0, 0.001, 0.003, 0.01, 0.03, 0.1];
  const base = { dataset: 'moons', noise: 32, layers: [8, 8], act: 'tanh', opt: 'adam', lr: 0.02, batch: 10, features: ['x1', 'x2'] };
  const out = { lams: LAMS, final: [], finalSd: [], best: [], trainFinal: [], curves: {} };
  // 50 training points, 400 test points, as in the Academy's overfitting module
  const mk = seed => {
    const pts = makeData('class', 'moons', 450, 0.32, 20 + seed);
    const ex = p => ({ x: p.x, y: p.y, label: p.label, input: [p.x, p.y] });
    return { feats: FEATURES.slice(0, 2), train: pts.slice(0, 50).map(ex), test: pts.slice(50).map(ex) };
  };
  for (const lam of LAMS) {
    const S = Object.assign(structuredClone(DEFAULT), base, { reg: lam ? 'l2' : 'none', regRate: lam });
    const runs = SEEDS.map(s => run(S, 1600, s, { curve: lam === 0 || lam === 0.01, data: mk(s) }));
    out.final.push(mean(runs.map(r => r.testLoss))); out.finalSd.push(sd(runs.map(r => r.testLoss)));
    out.best.push(mean(runs.map(r => r.bestTest))); out.trainFinal.push(mean(runs.map(r => r.trainLoss)));
    if (lam === 0 || lam === 0.01) out.curves[lam] = { train: runs[0].hist.train, test: runs[0].hist.test, bestEpoch: runs[0].bestEpoch };
    if (lam === 0) out.wsq0 = mean(runs.map(r => r.net.W.flat(2).reduce((s, w) => s + w * w, 0)));
    if (lam === 0.01) out.wsq01 = mean(runs.map(r => r.net.W.flat(2).reduce((s, w) => s + w * w, 0)));
  }
  R.overfit = out;
}

/* ---------- E5: learning-rate sweep ---------- */
{
  const LRS = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
  const out = { lrs: LRS, final: [], diverged: [], curves: {} };
  for (const lr of LRS) {
    const S = Object.assign(structuredClone(DEFAULT), { dataset: 'circle', layers: [4, 2], act: 'tanh', lr });
    const runs = SEEDS.map(s => run(S, 300, s, { curve: true }));
    const fin = runs.map(r => r.trainLoss);
    out.final.push(mean(fin.map(v => isFinite(v) ? v : NaN).filter(isFinite)) || null);
    out.diverged.push(fin.filter(v => !isFinite(v) || v > 5).length);
    out.curves[lr] = runs[0].hist.train;
  }
  R.lr = out;
}

/* ---------- E6: speed ---------- */
{
  const speed = (cfg, n = 200) => {
    const S = Object.assign(structuredClone(DEFAULT), cfg), { feats, train } = setupData(S);
    const net = new Network([feats.length, ...S.layers, 1], S.act, 'class', 3), r = rng(1);
    const t = process.hrtime.bigint(); for (let i = 0; i < n; i++) epoch(net, train, S, r);
    const ms = Number(process.hrtime.bigint() - t) / 1e6 / n;
    const g0 = process.hrtime.bigint(); grid(net, feats, 'class', 50); const gms = Number(process.hrtime.bigint() - g0) / 1e6;
    return { msPerEpoch: ms, params: net.weightCount(), gridMs: gms };
  };
  R.speed = { small: speed({}), large: speed({ dataset: 'spiral', layers: [8, 8, 8, 6], features: FEATURES.map(f => f.id), opt: 'adam', lr: 0.01 }) };
}

/* ---------- code metrics ---------- */
{
  const files = ['index.html', 'learn.html', 'advanced.html', 'css/style.css', 'css/journal.css', 'js/advanced.js', 'js/advmath.js', 'js/nn.js', 'js/data.js', 'js/playground.js', 'js/learn.js', 'js/backprop.js', 'js/missions.js', 'js/colors.js', 'js/fx.js'];
  R.code = files.map(f => ({ f, lines: readFileSync(new URL('../' + f, import.meta.url), 'utf8').split('\n').length, kb: statSync(new URL('../' + f, import.meta.url)).size / 1024 }));
  R.code.total = R.code.reduce((s, c) => s + c.lines, 0);
}
R.runtimeSec = (Date.now() - T0) / 1000;
R.date = new Date().toISOString().slice(0, 10);

writeFileSync(new URL('../paper/results.json', import.meta.url), JSON.stringify(R));

/* ---------- LaTeX macros ---------- */
const f = (v, d = 1) => v === null || v === undefined || !isFinite(v) ? '--' : Number(v).toFixed(d);
const pct = v => f(100 * v, 1);
const M = {};
M.buildDate = R.date;
M.nSeeds = SEEDS.length;
M.gcCases = R.gradcheck.cases.length;
M.gcParams = R.gradcheck.total;
M.gcMax = R.gradcheck.maxRel.toExponential(1).replace(/e([-+]\d+)/, (_, e) => `\\times 10^{${+e}}`);
M.nMissions = R.missions.length;
M.codeLines = R.code.total.toLocaleString('en-US').replace(',', '{,}');
M.nnLines = R.code.find(c => c.f === 'js/nn.js').lines;
M.msSmall = f(R.speed.small.msPerEpoch, 2); M.msLarge = f(R.speed.large.msPerEpoch, 2);
M.pSmall = R.speed.small.params; M.pLarge = R.speed.large.params;
M.gridMs = f(R.speed.small.gridMs, 1);
const cap = R.capacity.acc;
M.capCircleLin = pct(cap['circle.lin']); M.capCircleFeat = pct(cap['circle.feat']); M.capCircleTanh = pct(cap['circle.tanh']); M.capCircleLinHid = pct(cap['circle.linhid']);
M.capXorLin = pct(cap['xor.lin']); M.capXorFeat = pct(cap['xor.feat']); M.capXorTanh = pct(cap['xor.tanh']); M.capXorLinHid = pct(cap['xor.linhid']);
const o = R.overfit, i01 = o.lams.indexOf(0.01);
M.ofTestNoReg = f(o.final[0], 3); M.ofTestLtwo = f(o.final[i01], 3); M.ofBestNoReg = f(o.best[0], 3);
M.ofTrainNoReg = f(o.trainFinal[0], 3); M.ofTrainLtwo = f(o.trainFinal[i01], 3);
M.ofWsqNoReg = f(o.wsq0, 1); M.ofWsqLtwo = f(o.wsq01, 1);
M.ofBestEpoch = o.curves[0].bestEpoch;
const bestLamIdx = o.final.indexOf(Math.min(...o.final)); M.ofBestLam = o.lams[bestLamIdx]; M.ofBestLamLoss = f(o.final[bestLamIdx], 3);
const lr = R.lr;
M.lrDivFirst = lr.lrs[lr.diverged.findIndex(d => d > 0)] ?? '--';
const lrBestIdx = lr.final.reduce((b, v, i) => (v !== null && (b < 0 || v < lr.final[b]) ? i : b), -1); M.lrBest = lr.lrs[lrBestIdx]; M.lrBestLoss = f(lr.final[lrBestIdx], 3);
M.lrSlow = f(lr.final[0], 3);
const spiral = R.missions.find(m => m.dataset === 'spiral'); M.spiralAcc = pct(spiral.testAccMean); M.spiralSd = pct(spiral.testAccSd); M.spiralParams = spiral.params;
M.runtime = f(R.runtimeSec, 0);
let tex = '% Generated by tools/benchmark.mjs. Do not edit by hand.\n';
for (const [k, v] of Object.entries(M)) tex += `\\newcommand{\\${k}}{${v}}\n`;
writeFileSync(new URL('../paper/stats.tex', import.meta.url), tex);

/* ---------- tables ---------- */
const esc = s => s.replace(/&/g, '\\&').replace(/_/g, '\\_');
let tm = '';
for (const m of R.missions) {
  const arch = m.layers.length ? m.layers.join('--') : 'none';
  const res = m.problem === 'class'
    ? (m.diverged === SEEDS.length ? 'diverged' : `${pct(m.testAccMean)} $\\pm$ ${pct(m.testAccSd)}`)
    : `MSE ${f(2 * m.testLossMean, 4)}`;
  const h = m.hit95.filter(x => x !== null);
  tm += `${m.n} & ${esc(m.title)} & ${m.dataset} & ${arch} & ${m.act} & ${m.opt}, ${m.lr}${m.reg !== 'none' ? `, ${m.reg.toUpperCase()} ${m.regRate}` : ''} & ${m.params} & ${res} & ${m.problem === 'class' ? (h.length ? `${h.length}/${SEEDS.length}` : `0/${SEEDS.length}`) : '--'} \\\\\n`;
}
writeFileSync(new URL('../paper/tab_missions.tex', import.meta.url), tm);
let tc = '';
for (const mm of R.capacity.models) {
  tc += esc(mm.name) + R.capacity.datasets.map(d => ` & ${pct(cap[`${d}.${mm.key}`])} $\\pm$ ${pct(R.capacity.sd[`${d}.${mm.key}`])}`).join('') + ' \\\\\n';
}
writeFileSync(new URL('../paper/tab_capacity.tex', import.meta.url), tc);
let tg = '';
const actName = { relu: 'ReLU', leaky: 'Leaky ReLU', tanh: 'tanh', sigmoid: 'sigmoid', linear: 'linear' };
for (const c of R.gradcheck.cases) tg += `${actName[c.act]} & ${c.problem === 'class' ? 'classification' : 'regression'} & ${c.checked} & $${c.maxRel.toExponential(1).replace(/e([-+]\d+)/, (_, e) => `\\times 10^{${+e}}`)}$ \\\\\n`;
writeFileSync(new URL('../paper/tab_gradcheck.tex', import.meta.url), tg);
let tl = '';
lr.lrs.forEach((v, i) => { tl += `${v} & ${lr.final[i] === null ? '--' : f(lr.final[i], 3)} & ${lr.diverged[i]}/${SEEDS.length} \\\\\n`; });
writeFileSync(new URL('../paper/tab_lr.tex', import.meta.url), tl);
let tf = '';
for (const c of R.code) tf += `\\code{${c.f}} & ${c.lines} \\\\\n`;
writeFileSync(new URL('../paper/tab_code.tex', import.meta.url), tf);
console.log(JSON.stringify({ runtime: R.runtimeSec, grad: R.gradcheck.maxRel, missions: R.missions.map(m => [m.n, m.testAccMean, m.testLossMean, m.diverged]), cap: R.capacity.acc, of: [o.final, o.trainFinal], lr: [lr.final, lr.diverged], speed: R.speed }, null, 0));
