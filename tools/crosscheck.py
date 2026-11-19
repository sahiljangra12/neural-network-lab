"""Independent verification of the site's learning engine.

Replays every run recorded by tools/export_trace.mjs with a separate implementation built on
JAX (automatic differentiation) and Optax (reference optimisers), in 64-bit floating point,
using the same initial weights, data and per-epoch shuffle order. Then checks that a network
with no hidden layer converges to the closed-form statistical estimates computed by
scikit-learn (logistic-regression MLE) and NumPy (ordinary least squares).

    python3 tools/crosscheck.py     -> paper/crosscheck.json, paper/tab_crosscheck.tex
"""
import json, math, numpy as np
import jax, jax.numpy as jnp, optax
from sklearn.linear_model import LogisticRegression

jax.config.update("jax_enable_x64", True)
T = json.load(open("paper/crosscheck_trace.json"))

ACT = {"relu": jax.nn.relu, "leaky": lambda z: jax.nn.leaky_relu(z, 0.01), "tanh": jnp.tanh,
       "sigmoid": jax.nn.sigmoid, "linear": lambda z: z}

def forward(params, x, act):
    a = x
    for W, b in params[:-1]:
        a = ACT[act](W @ a + b)
    W, b = params[-1]
    return (W @ a + b)[0]                    # output pre-activation z

def objective(params, X, Y, cfg):
    z = jax.vmap(lambda x: forward(params, x, cfg["act"]))(X)
    if cfg["problem"] == "class":
        per = optax.sigmoid_binary_cross_entropy(z, (Y > 0).astype(jnp.float64))
    else:
        per = 0.5 * (z - Y) ** 2
    obj = jnp.mean(per)
    if cfg["reg"] == "l2":
        obj = obj + 0.5 * cfg["regRate"] * sum(jnp.sum(W ** 2) for W, _ in params)
    return obj

def report_loss(params, X, Y, cfg):           # same reporting formula as Network.loss()
    z = np.array(jax.vmap(lambda x: forward(params, x, cfg["act"]))(X))
    if cfg["problem"] == "class":
        p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12); t = (Y > 0)
        return float(np.mean(-(t * np.log(p) + (1 - t) * np.log(1 - p))))
    return float(np.mean(0.5 * (z - Y) ** 2))

def replay(run):
    X = jnp.array(run["X"], dtype=jnp.float64); Y = jnp.array(run["Y"], dtype=jnp.float64)
    params = [(jnp.array(W, dtype=jnp.float64), jnp.array(b, dtype=jnp.float64)) for W, b in zip(run["init"]["W"], run["init"]["b"])]
    opt = {"sgd": optax.sgd(run["lr"]),
           "momentum": optax.sgd(run["lr"], momentum=0.9),
           "adam": optax.adam(run["lr"], b1=0.9, b2=0.999, eps=1e-8)}[run["opt"]]
    state = opt.init(params)
    grad = jax.jit(jax.grad(lambda p, Xb, Yb: objective(p, Xb, Yb, run)))
    B, losses = run["batch"], []
    for order in run["orders"]:
        for s in range(0, len(order), B):
            idx = jnp.array(order[s:s + B])
            g = grad(params, X[idx], Y[idx])
            if run["reg"] == "l1":             # subgradient lambda * sign(w) on weights only
                g = [(gW + run["regRate"] * jnp.sign(W), gb) for (gW, gb), (W, _) in zip(g, params)]
            upd, state = opt.update(g, state, params)
            new = optax.apply_updates(params, upd)
            if run["reg"] == "l1":             # documented rule: a weight that changes sign is set to 0
                new = [(jnp.where((jnp.sign(Wn) != jnp.sign(Wo)) & (Wo != 0), 0.0, Wn), bn) for (Wn, bn), (Wo, _) in zip(new, params)]
            params = new
        losses.append(report_loss(params, X, np.array(Y), run))
    return params, losses

rows, worst_p, worst_l = [], 0.0, 0.0
for run in T["runs"]:
    params, losses = replay(run)
    ours = np.concatenate([np.ravel(np.array(a)) for W, b in zip(run["final"]["W"], run["final"]["b"]) for a in (W, b)])
    ref = np.concatenate([np.ravel(np.array(a)) for W, b in params for a in (W, b)])
    dp = float(np.max(np.abs(ours - ref)) / max(1.0, np.max(np.abs(ref))))
    dl = float(np.max(np.abs(np.array(run["losses"]) - np.array(losses)) / np.maximum(1e-12, np.abs(np.array(losses)))))
    worst_p, worst_l = max(worst_p, dp), max(worst_l, dl)
    updates = sum(math.ceil(len(o) / run["batch"]) for o in run["orders"])
    rows.append(dict(name=run["name"], params=int(ref.size), epochs=len(run["orders"]), updates=updates, dparam=dp, dloss=dl,
                     loss0=losses[0], lossN=losses[-1]))
    print(f"{run['name']:<42} params={ref.size:4d} updates={updates:5d}  max param diff={dp:.1e}  max loss rel diff={dl:.1e}  loss {losses[0]:.4f}->{losses[-1]:.4f}")

closed = []
for c in T["closed"]:
    X, Y = np.array(c["X"]), np.array(c["Y"])
    ours = np.array(c["w"] + [c["b"]])
    if c["problem"] == "class":
        m = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-12, max_iter=100000).fit(X, (Y > 0).astype(int))
        ref = np.concatenate([m.coef_[0], m.intercept_]); method = "Logistic-regression MLE (scikit-learn)"
    else:
        A = np.column_stack([X, np.ones(len(X))]); ref = np.linalg.lstsq(A, Y, rcond=None)[0]; method = "Ordinary least squares (NumPy)"
    d = float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))
    closed.append(dict(method=method, ours=ours.tolist(), ref=ref.tolist(), reldiff=d))
    print(f"{method:<42} site={np.round(ours, 6)}  reference={np.round(ref, 6)}  max rel diff={d:.1e}")

# Academy backpropagation lesson vs JAX autodiff
B0 = T["bpInit"]
P0 = [(jnp.array(B0["W1"]), jnp.array(B0["b1"])), (jnp.array([B0["W2"]]), jnp.array([B0["b2"]]))]
def bp_loss(P, x, y):
    z = forward(P, x, "tanh"); return optax.sigmoid_binary_cross_entropy(z, y)
bp_worst = 0.0
for c in T["bp"]:
    x, y, lr = jnp.array(c["x"], dtype=jnp.float64), float(c["y"]), c["lr"]
    g = jax.grad(bp_loss)(P0, x, y)
    ref = {"gW1": np.array(g[0][0]), "gb1": np.array(g[0][1]), "gW2": np.array(g[1][0][0]), "gb2": float(g[1][1][0]),
           "p": float(jax.nn.sigmoid(forward(P0, x, "tanh"))), "L": float(bp_loss(P0, x, y))}
    P1 = jax.tree_util.tree_map(lambda a, b: a - lr * b, P0, g)
    ref["nL"] = float(bp_loss(P1, x, y))
    site = {"gW1": np.array(c["gW1"]), "gb1": np.array(c["gb1"]), "gW2": np.array(c["gW2"]), "gb2": c["gb2"], "p": c["p"], "L": c["L"], "nL": c["nL"]}
    for k in ref:
        d = float(np.max(np.abs(np.array(site[k]) - np.array(ref[k]))) / max(1e-12, float(np.max(np.abs(np.array(ref[k]))))))
        bp_worst = max(bp_worst, d)
print(f"Academy backpropagation lesson, {len(T['bp'])} cases: worst relative difference {bp_worst:.1e}")

res = dict(bpCases=len(T["bp"]), bpWorst=bp_worst, runs=rows, worstParam=worst_p, worstLoss=worst_l, closed=closed, jax=jax.__version__, optax=optax.__version__)
json.dump(res, open("paper/crosscheck.json", "w"), indent=1)

def sci(v):
    if v == 0: return "0"
    m, e = f"{v:.1e}".split("e"); return f"${m}\\times 10^{{{int(e)}}}$"
with open("paper/tab_crosscheck.tex", "w") as f:
    for r in rows:
        f.write(f"{r['name']} & {r['params']} & {r['updates']} & {sci(r['dparam'])} & {sci(r['dloss'])} \\\\\n")
with open("paper/stats_crosscheck.tex", "w") as f:
    f.write("% Generated by tools/crosscheck.py. Do not edit by hand.\n")
    f.write(f"\\newcommand{{\\xcRuns}}{{{len(rows)}}}\n\\newcommand{{\\xcUpdates}}{{{sum(r['updates'] for r in rows):,}}}\n".replace(",", "{,}"))
    f.write(f"\\newcommand{{\\xcParam}}{{{sci(worst_p).strip('$')}}}\n\\newcommand{{\\xcLoss}}{{{sci(worst_l).strip('$')}}}\n")
    f.write(f"\\newcommand{{\\xcJax}}{{{jax.__version__}}}\n\\newcommand{{\\xcOptax}}{{{optax.__version__}}}\n")
    f.write(f"\\newcommand{{\\xcBpCases}}{{{len(T['bp'])}}}\n\\newcommand{{\\xcBp}}{{{sci(bp_worst).strip('$')}}}\n")
    f.write(f"\\newcommand{{\\xcLogit}}{{{sci(closed[0]['reldiff']).strip('$')}}}\n\\newcommand{{\\xcOls}}{{{sci(closed[1]['reldiff']).strip('$')}}}\n")
print(f"\nWORST over all runs: parameters {worst_p:.1e}, loss {worst_l:.1e}")
