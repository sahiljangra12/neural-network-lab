# Neural Network Lab

**Live site:** https://sahiljangra12.github.io/neural-network-lab/

A mission-control-style training console for neural networks. Design a network, launch training, and watch every neuron learn in real time. It includes guided missions and an academy of interactive lessons.

**Look and feel:** a light, futuristic console: pale grid background with a drifting node network, white panels with red corner brackets, condensed uppercase headings, black ink with red as the only accent. Animated (signal pulses, scanner sweep, scroll reveals) and switched off automatically for reduced-motion settings.

## Console (`index.html`)
- **8 datasets:** six classification (circle, XOR, Gaussian, spiral, two moons, checkerboard) and two regression (plane, multi-Gaussian). You can set the noise, the train/test split and the batch size.
- **7 input features:** X₁, X₂, X₁², X₂², X₁X₂, sin X₁, sin X₂. Click a feature to switch it on or off.
- **Network shape:** up to 6 hidden layers of up to 8 neurons each.
- **Training settings:** activation (ReLU, Leaky ReLU, tanh, sigmoid, linear), optimizer (SGD, Momentum, Adam), L1/L2 regularization, learning rate.
- **Live neuron maps:** every neuron shows its own output map. Hover a neuron to enlarge it on the output panel.
- **Weights:** line width and colour show each weight's size and sign. Click a line to edit the weight by hand.
- **Live insight engine:** spots overfitting, dead ReLUs, divergence, all-linear collapse and slow learning, and explains why.
- **11 training missions,** from "Hello, neuron" to "Beat the spiral".
- **Shareable setups:** the URL encodes the whole configuration. Keyboard shortcuts: Space, S, R.

## Academy (`learn.html`)
1. **The neuron:** drag the weights and bias and watch the decision boundary move.
2. **Activation functions:** each function and its derivative, with the trade-offs.
3. **Why hidden layers:** three networks train side by side (logistic regression, tanh, all-linear).
4. **Loss:** log loss and squared error, explored interactively.
5. **Gradient descent:** four loss surfaces, three optimizers, learning-rate and noise sliders, and click-to-start paths.
6. **Backpropagation:** every number in one full training update, step by step.
7. **Overfitting and regularization:** live train/test loss with and without L2 weight decay.
8. **Glossary** (searchable) and a **16-question true/false quiz** with justifications.

## Advanced (`advanced.html`)
Six interactive modules on the ideas behind modern deep networks, each computed live:
1. **Softmax and temperature:** logits to a probability distribution, cross-entropy and its gradient.
2. **Deep networks and skip connections:** watch gradients vanish or explode through up to 40 layers; fix them with Xavier/He initialization or residual paths.
3. **Dropout:** two networks train side by side on noisy data, one with dropout.
4. **Convolution:** draw an image, edit a 3 × 3 kernel, see the feature map and max-pooling.
5. **Recurrent memory:** how long a recurrent unit remembers, why it fades, and how a gate keeps it.
6. **Attention:** click a word to see what it attends to, with or without a causal (GPT-style) mask. The word vectors are a hand-made toy, not a trained model.

The maths is in `js/advmath.js` and is verified by `node tools/check_advanced.mjs` (finite-difference gradient checks and direct recomputation).

## Technical report
**[Read the technical report (PDF)](report/neural-network-lab-technical-report.pdf)**: 19 pages covering the engine's equations, the system design and a reproducible evaluation: an independent re-implementation in JAX/Optax that reproduces 18 complete training runs to machine precision, agreement with the closed-form logistic-regression MLE (scikit-learn) and least-squares solution, a gradient check, mission benchmarks, model capacity, overfitting with L2 and a learning-rate sweep. Every number, table and figure is generated from the site's own code:

```bash
node tools/benchmark.mjs      # runs the experiments, writes paper/results.json, stats.tex and tables
node tools/export_trace.mjs   # records complete training runs of the site's engine
python3 tools/crosscheck.py   # replays them in JAX/Optax; checks MLE/OLS and the backprop lesson (pip install jax optax scikit-learn)
python3 tools/figures.py      # draws the figures
cd paper && latexmk -pdf neural_network_lab.tex
```

## Inspiration
This project was inspired by **EN.553.413/613 Applied Statistics and Data Analysis (ASDA)**, taught by **Prof. Suryanarayana Maddu**, Department of Applied Mathematics and Statistics, Johns Hopkins University (Fall 2026). The course's treatment of likelihood, logistic regression, ridge and lasso regularisation, the bias–variance trade-off and model selection shaped the Academy's lessons and the missions.

## Tech
Plain HTML, CSS and ES modules with no build step and no dependencies. The network (`js/nn.js`) is a small, hand-written MLP with backpropagation, momentum, Adam and L1/L2. It is hosted on GitHub Pages.

Inspired by [TensorFlow Playground](https://playground.tensorflow.org). All code here is written from scratch.
