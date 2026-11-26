// The worked example of the Academy's backpropagation lesson (2-2-1 network, tanh hidden,
// sigmoid output, log loss). Pure function so it can be verified independently.
export const BP = { W1: [[0.4, -0.6], [0.7, 0.2]], b1: [0.1, -0.1], W2: [0.8, -0.5], b2: 0.05 };
const sig = z => 1 / (1 + Math.exp(-z));
export function bpStep(x, y, lr) {
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
