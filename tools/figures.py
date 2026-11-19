"""Figures for the technical report, drawn from paper/results.json (made by tools/benchmark.mjs)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = json.load(open("paper/results.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})
CM = LinearSegmentedColormap.from_list("nnl", ["#cc3622", "#f4f1ee", "#2e3440"])
RED, INK, GRAY = "#cc3622", "#2e3440", "#9aa0a6"

# Figure: model capacity (decision maps)
cap = R["capacity"]; ds = cap["datasets"]; models = cap["models"]
names = {"gauss": "Gaussian", "circle": "Circle", "xor": "XOR", "moons": "Two moons"}
short = {"lin": "Logistic\n(X$_1$, X$_2$)", "feat": "Logistic +\nengineered features", "linhid": "4 linear\nhidden neurons", "tanh": "4 tanh\nhidden neurons"}
fig, ax = plt.subplots(len(ds), len(models), figsize=(6.3, 6.5))
for i, d in enumerate(ds):
    for j, m in enumerate(models):
        a = ax[i, j]; g = cap["grids"][f"{d}.{m['key']}"]
        n = int(len(g["g"]) ** .5)
        a.imshow(np.array(g["g"]).reshape(n, n), cmap=CM, vmin=-1, vmax=1, extent=(-6, 6, -6, 6), interpolation="bilinear")
        p = np.array(g["pts"])
        a.scatter(p[:, 0], p[:, 1], c=[RED if l < 0 else INK for l in p[:, 2]], s=3, linewidths=.25, edgecolors="white")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(True); s.set_linewidth(.4)
        acc = cap["acc"][f"{d}.{m['key']}"]
        a.set_xlabel(f"{100*acc:.1f}%", fontsize=8, labelpad=2)
        if i == 0: a.set_title(short[m["key"]], fontsize=8)
        if j == 0: a.set_ylabel(names[d], fontsize=9)
plt.tight_layout(pad=.4)
plt.savefig("paper/fig_capacity.pdf"); plt.close()

# Figure: overfitting
o = R["overfit"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5))
for lam, col, lab in [("0", RED, "no penalty"), ("0.01", INK, "L2, $\\lambda=0.01$")]:
    c = o["curves"][lam]; e = np.arange(1, len(c["train"]) + 1)
    ax[0].plot(e, c["train"], color=col, lw=.9, ls="--", alpha=.8)
    ax[0].plot(e, c["test"], color=col, lw=1.3, label=lab)
ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("log loss")
ax[0].set_title("(a) training (dashed) and test (solid) loss", fontsize=8.5)
ax[0].legend(frameon=False, fontsize=7.5)
lams = o["lams"]; xs = np.arange(len(lams))
ax[1].errorbar(xs, o["final"], yerr=o["finalSd"], color=INK, marker="o", ms=3.5, lw=1, capsize=2, label="test")
ax[1].plot(xs, o["trainFinal"], color=GRAY, marker="s", ms=3, lw=1, label="training")
ax[1].set_xticks(xs); ax[1].set_xticklabels([str(l) for l in lams], fontsize=7.5)
ax[1].set_xlabel("L2 strength $\\lambda$"); ax[1].set_ylabel("final log loss")
ax[1].set_title("(b) final loss after 1,600 epochs (mean $\\pm$ sd)", fontsize=8.5)
ax[1].legend(frameon=False, fontsize=7.5)
plt.tight_layout(pad=.4); plt.savefig("paper/fig_overfit.pdf"); plt.close()

# Figure: learning rate
lr = R["lr"]
fig, ax = plt.subplots(figsize=(6.3, 2.5))
cols = plt.cm.Greys(np.linspace(.35, 1, len(lr["lrs"])))
for k, v in enumerate(lr["lrs"]):
    c = np.array(lr["curves"][str(v)] if str(v) in lr["curves"] else lr["curves"][repr(v)], dtype=float)
    c = np.where(np.isfinite(c), c, np.nan)
    col = RED if v >= 3 else cols[k]
    ax.plot(np.arange(1, len(c) + 1), c, color=col, lw=1.1, label=f"$\\eta$ = {v}")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training log loss"); ax.set_ylim(top=20)
ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower left")
plt.tight_layout(pad=.4); plt.savefig("paper/fig_lr.pdf"); plt.close()
print("figures ok")
