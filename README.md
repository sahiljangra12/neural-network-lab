# Neural Network Lab

**Live site:** https://sahiljangra12.github.io/neural-network-lab/

A sci-fi training console for neural networks. Design a network, launch training, and watch every neuron learn in real time, with signal pulses flowing along the synapses. It includes guided training missions and an academy of interactive lessons.

**Look and feel:** a deep-space HUD with a live neural-constellation background, glowing orb neurons, cyan (+) and magenta (−) signals, a boot sequence, and scanlines. Toggle the effects with **FX**. They are off automatically when your system asks for reduced motion.

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

## Tech
Plain HTML, CSS and ES modules with no build step and no dependencies. The network (`js/nn.js`) is a small, hand-written MLP with backpropagation, momentum, Adam and L1/L2. It is hosted on GitHub Pages.

Inspired by [TensorFlow Playground](https://playground.tensorflow.org). All code here is written from scratch.
