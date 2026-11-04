# Backpropagation and weight updates

The neural module trains networks with **mini-batch gradient descent**. This guide
describes the loss, the gradients, and the update implemented in `src/neural/`.
The command-line loop (epochs, shuffling, batches, validation, early stopping,
history, saving) belongs to the analyzer and is described in the
[training command guide](training-command.md).

The module stays generic: it works on numeric vectors of any compatible size and
has no knowledge of chess.

## API

| Method | Role |
| --- | --- |
| `double loss(samples) const` | Mean loss over the samples; never changes the network. |
| `double trainBatch(batch)` | One gradient descent step; returns the mean batch loss **before** the update. |
| `std::vector<Layer> gradients(samples) const` | Gradients of `loss(samples)` for every weight and bias, shaped like `layers()`. Used for verification. |
| `void validateSamples(samples) const` | Rejects samples that do not fit the network (see [Errors](#errors)). |

A `TrainingSample` holds `inputs` and the expected outputs `target`. Chess
training uses one-hot targets ordered `Nothing`, `Check`, `Checkmate`.

| File | Content |
| --- | --- |
| `Training.cpp` | `loss`, `gradients`, `trainBatch`, and the per-sample backpropagation |
| `Loss.cpp` | `sampleLoss`: stable per-sample loss from output logits |
| `Activation.cpp` | `activationDerivative`: derivatives used by backpropagation |
| `NeuralNetwork.cpp` | Forward pass (keeps every activation and the output logits) and sample validation |

## Notation

For a layer with weights `W` and biases `b`, receiving the previous activations
`a_prev`:

```text
z = W · a_prev + b      values before the activation (logits for the output layer)
a = f(z)                activations, passed to the next layer
p                       output probabilities (softmax or sigmoid of the output logits)
y                       target values
N                       number of samples in the batch
```

## Loss

The loss depends on the output activation; any other output activation is
rejected.

| Output | Loss for one sample |
| --- | --- |
| `softmax` | Categorical cross-entropy: `-Σ_k y_k · log(p_k)` |
| `sigmoid` | Binary cross-entropy summed over outputs: `-Σ_k [y_k · log(p_k) + (1 - y_k) · log(1 - p_k)]` |

The loss of a collection is the **mean** over its samples. Train and validation
losses therefore stay comparable whatever their sizes, and the learning rate does
not depend on the batch size.

### Numerical stability

A probability can round to exactly 0 or 1 (for example `sigmoid(40)` is `1.0` in
double precision), and `log(0)` is infinite. The forward pass therefore keeps the
output logits, and the loss is computed from them without forming probabilities:

```text
softmax:  -log(p_k) = (m - z_k) + log(Σ_j exp(z_j - m))     with m = max_j z_j
sigmoid:  per output  max(z, 0) - z · y + log1p(exp(-|z|))
```

No exponential can overflow, no logarithm of 0 is taken, and every term is
non-negative, so the loss is never negative. For logits `[800, 0, 0]` and the
third class expected, the loss is `800` instead of infinity.

For one-hot targets the minimum loss is 0. Soft targets are accepted; their
minimum loss is the entropy of the target, not 0.

## Gradients

For each sample, backpropagation computes the derivative of its loss with respect
to every parameter:

1. **Forward pass**, keeping the activations of every layer.
2. **Output delta** `δ = p - y`. This is the derivative of the loss with respect
   to the output logits, both for softmax with cross-entropy (targets summing to 1)
   and for sigmoid with binary cross-entropy.
3. **For each layer, from the last to the first:**
   - weight gradient `∂L/∂W[j][k] = δ[j] · a_prev[k]`;
   - bias gradient `∂L/∂b[j] = δ[j]`;
   - delta of the previous hidden layer `δ_prev = (Wᵀ · δ) ⊙ f'(a_prev)`, where `⊙`
     multiplies element by element.

Derivatives are computed from the activations already stored by the forward pass:

| Hidden activation | `f'` from its output `a` |
| --- | --- |
| `sigmoid` | `a · (1 - a)` |
| `tanh` | `1 - a²` |
| `relu` | `1` if `a > 0`, else `0` (0 at the kink) |
| `linear` | `1` |

Softmax has no element-wise derivative and is rejected as a hidden activation.

The gradients of a batch are the **mean** of the sample gradients, divided by the
actual batch size: a final partial batch uses its own size.

## Update

`trainBatch` applies one step of plain mini-batch stochastic gradient descent:

```text
θ ← θ - η · ∂L_batch/∂θ        for every weight and bias θ, with η = config().learning_rate
```

The loss and the gradients come from the same forward pass, so the returned
pre-update loss costs nothing extra. The analyzer reports epoch losses with
`loss()` after each epoch instead of averaging these batch values, which were
measured with different parameters.

The new parameters are computed aside and installed only when all of them are
finite. **Whenever `trainBatch` throws, the network is unchanged.** Training only
changes weights and biases, never the configuration (architecture, activations,
learning rate, seed). Networks are plain values, so a copy is an independent
checkpoint.

## Errors

| Situation | Exception |
| --- | --- |
| Empty collection, wrong number of inputs or targets | `std::invalid_argument` |
| Non-finite input or target, target outside `[0, 1]` | `std::invalid_argument` |
| Softmax targets not summing to 1 (tolerance `1e-6`) | `std::invalid_argument` |
| Output activation other than softmax or sigmoid | `std::invalid_argument` |
| Hidden activation without derivative (softmax) | `std::invalid_argument` |
| Learning rate not finite and positive | `std::invalid_argument` |
| Non-finite loss, gradient, or updated parameter | `std::runtime_error` (suggests a lower learning rate) |

Messages name the faulty sample, for example `training sample 1: expected 3
inputs, got 2`. Samples are validated before any computation.

## Design choices

| Choice | Reason | Trade-off |
| --- | --- | --- |
| Cross-entropy with softmax or sigmoid | Outputs read as probabilities; output delta is simply `p - y`; confident mistakes get large gradients | Only these two output activations can be trained |
| Loss computed from logits | Finite loss even when a probability rounds to 0 or 1 | The forward pass keeps one extra vector |
| Mean over samples | Comparable train/validation losses; learning rate independent of batch size | — |
| Mini-batch SGD (batch size chosen by the analyzer, 32 by default) | Several updates per epoch with less noise than single samples; one hyperparameter | No momentum or adaptive optimizer (Adam); learning rate must be tuned |
| Derivatives from activations | No need to store hidden pre-activations | ReLU derivative is taken as 0 at exactly 0 |
| Update applied only if finite | A diverging step cannot corrupt the network or a checkpoint | One temporary copy of the parameters per batch |
| Public `gradients()` | Lets tests compare backpropagation with measured slopes | Slightly larger public interface |

## Verification

| Test file | What it checks |
| --- | --- |
| `tests/Loss.cpp` | Hand-computed losses, invariance to shifting all logits, extreme logits that break the textbook formula, agreement with it on ordinary values |
| `tests/Activation.cpp` | Derivatives against hand-computed values and measured slopes |
| `tests/Backpropagation.cpp` | Sample validation; loss averaging and agreement with `predict`; hand-computed gradients with and without a hidden layer; batch averaging; a hand-computed update; failures leave the network unchanged; gradients of every parameter against finite differences for the 8 activation pairs; XOR learned with mini-batches, but not without a hidden layer |
| `tests/cli.py` | A real `--train` run saving the trained model and its history, in both save modes |

The finite-difference test nudges each parameter by `±1e-6` and compares the
measured loss slope with the gradient; observed differences stay below `1e-9`.

## Performance and limitations

A training step costs about three forward passes per sample (forward, backward,
and gradient accumulation). With the current build flags (no `-O2`), on an
Apple M5, an epoch of 2,400 positions on the 777-128-64-3 reference network,
including the train and validation loss evaluations, takes about 2.5 s. A train
set of about 570,000 positions would take roughly 10 minutes per epoch.

Possible improvements, not implemented: compiler optimization (`-O2`), skipping
zero inputs in the first layer (about 34 of the 777 chess inputs are non-zero),
parallelism, momentum or adaptive optimizers, and regularization such as dropout
or weight decay.
