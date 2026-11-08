# Architecture and configuration

MyTorch separates a generic dense neural network from its chess application.
**C++17 performs inference and learning; Python prepares data, runs protocols and
analyzes saved results.** Python never computes the shipped model's predictions
or weight updates.

## Components and data flow

| Component | Responsibility |
| --- | --- |
| `src/neural/` | Network configuration, initialization, forward pass, backpropagation, SGD/Adam and serialization; no chess dependencies |
| `src/chess/` | Six-field FEN parsing, 777-input encoding and three-class target mapping |
| `src/generator/` | `my_torch_generator`: create initialized networks from YAML |
| `src/analyzer/` | `my_torch_analyzer`: prediction, validation, mini-batches, checkpoint selection and saving |
| `src/logging/` | Per-run generation/training logs |
| `scripts/` | Dataset preparation/augmentation/freezing, experiment orchestration, metrics and charts |
| `notebooks/` | Dataset inspection and visualization of saved results |

```text
Raw FENs → clean and split → frozen datasets + manifests
Network YAML → C++ generator → initialized .nn model
Model + train + validation → C++ analyzer --train → selected checkpoint + losses
Saved checkpoint + FENs → C++ analyzer --predict → ordered class predictions
Run artifacts → Python evaluation/comparison → metrics, SVG charts and reports
```

The experiment runner invokes the C++ programs through subprocesses. It records
commands, configuration, seeds, data/build fingerprints and execution status.
Standalone C++ commands remain usable without Python or a chess-rules library.

## Network configuration

The [basic configuration](../configs/networks/basic-network.yml) is a starting
point, not a declaration of the final selected model:

```yaml
layers: [777, 128, 64, 3]
hidden_activation: sigmoid
output_activation: softmax
initialization: xavier
learning_rate: 0.01
seed: 42
optimizer: sgd
```

| Setting | Supported values and meaning |
| --- | --- |
| `layers` | At least two positive widths, including input and output; arbitrary dimensions in the generic module, 777 inputs/3 outputs in the chess analyzer |
| `hidden_activation` | `sigmoid`, `tanh`, `relu` |
| `output_activation` | `softmax` or `sigmoid`; chess uses mutually exclusive classes and softmax/categorical cross-entropy |
| `initialization` | `xavier`: uniform ±sqrt(6 / (fan-in + fan-out)); `he`: normal with standard deviation sqrt(2 / fan-in); biases start at zero |
| `learning_rate` | Finite positive value; default `0.01` |
| `seed` | Unsigned 32-bit integer; default `42`; successive generated models use successive seeds |
| `optimizer` | `sgd` (default) or `adam`; loaded with the model, not selected by a training CLI flag |
| `dropout` | Probability in [0, 1) of zeroing each hidden activation during training; default `0` (disabled); saved in the model only when enabled |

Unknown/duplicate keys, unsupported values and missing required fields are
rejected. Only learning rate, seed, optimizer and dropout have defaults. Protocols can
override network settings and specify a seed list; see [evaluation](evaluation.md).

Sigmoid is retained as a baseline. Tanh and ReLU test different gradient and
convergence behavior; He is a ReLU-oriented alternative to Xavier. These are
choices to compare, not automatic improvements. Architecture, learning rate and
optimizer interact, so changing several together measures a complete setting.
See [training calculations](training.md#5-neural-calculations) for the formulas
and [experiment reports](report/README.md) for observed outcomes.
The [hyperparameter guide](hyperparameters.md) connects each setting to its
tested alternatives and distinguishes the delivered model from later candidates.

## Saved models and continuation

`.nn` files are versioned JSON containing configuration, weights and biases.
Adam checkpoints also retain the two moment tensors and update counter. Loading
checks dimensions and finite values; legacy files without an optimizer field use
SGD. See [`ModelSerializer`](../src/neural/ModelSerializer.hpp).

Saving and loading preserves learned parameters. Continuing training also restores
Adam state, but starts a new shuffle sequence and early-stopping window; it does
not reproduce an uninterrupted run. Model files do not contain training datasets
or authenticate the provenance of a separate loss history.

`my_torch_network.nn` at the root is the delivered model. The explicitly tracked
candidate in [models/](../models/README.md) is a separate artifact; other generated
networks and raw results are ignored. Promotion must update the delivered file
and its provenance together.

## Performance, correctness and limits

Training stores binary chess inputs as active indices and expands them for
computation, preserving the 777-input contract. Sparse first-layer operations
avoid multiplication by zero; dense blocks optionally use CPU BLAS. Matrix
multiplication is delegated to BLAS, while activations, losses, gradients and
optimizer updates remain project code. There is no GPU backend.

Validation loss, best-checkpoint restoration and early stopping address
overfitting; they do not guarantee generalization. The encoder validates syntax,
not full chess legality. Prediction classifies a position; it does not play a
move or run a rules engine. See [input](datasets.md#1-fen-encoding-and-c-dataset-contract)
and [output](output-encoding.md) contracts.

C++ tests check calculations, finite-difference gradients, serialization and
failure guarantees. Python tests cover CLI behavior, data preparation and the
runner's real integration. Build/test commands and dependencies are in the
[README](../README.md); datasets stay outside result folders.
