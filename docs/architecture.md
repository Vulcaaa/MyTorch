# Project architecture

This is the proposed architecture and may evolve during development.

MyTorch uses **C++17 for network computation and executables**, and **Python for
dataset preparation, experiment orchestration, and result analysis**. The
Makefile handles builds and tests; the notebook launches dataset cleaning.

## Components

| Component | Language | Responsibility |
| --- | --- | --- |
| `src/neural/` | C++ | Generic network: configuration, initialization, serialization, then forward propagation and training. |
| `src/chess/` | C++ | FEN parsing, feature encoding, and dataset label handling. |
| `src/generator/` | C++ | `my_torch_generator`: create models from configuration files. |
| `src/analyzer/` | C++ | `my_torch_analyzer`: coordinate training and prediction. |
| `scripts/` | Python | Clean datasets; later run experiments, compute metrics, and plot results. |
| `tests/` | C++ and Python | Function tests and executable tests. |

The network module operates on numeric vectors without knowledge of chess or
FEN. The chess module implements the [input](data-encoding.md) and
[output](output-encoding.md) contracts. The analyzer connects both modules.

## Planned data flow

Training:

```text
Source datasets → Python preparation → train / validation / test files
Configuration → C++ generator → initial model
Initial model + data → C++ analyzer --train → trained model
Experiment results → Python analysis → metrics and plots
```

Prediction:

```text
FEN file → C++ parsing and encoding → C++ network → class predictions
```

Python scripts invoke the executables through `subprocess`, exchanging files
and command output. Language bindings are unnecessary. Computation and weight
updates stay in C++; Python organizes the experiments.

## Testing

- **C++:** test functions directly, including encoding, layer calculations,
  gradients, and serialization, using known expected results.
- **Python:** run the executables and check arguments, stdout, stderr, exit
  codes, and generated files. Cover generation, training, saving, and prediction
  together once that workflow is available.

These levels cover calculations and executable behavior respectively; tests
need not be duplicated across languages.

## Build and scripts

| Command | Purpose |
| --- | --- |
| `make` | Build both C++ executables |
| `make test` | Run C++ tests and Python executable tests |
| `make clean` | Remove build artifacts |
| `make fclean` | Also remove executables |
| `make re` | Clean and rebuild |

Dataset cleaning is launched through
[`notebooks/clean_dataset.ipynb`](../notebooks/clean_dataset.ipynb), which calls
`scripts/clean_dataset.py`. The notebook reads `dataset/raw/` and writes results
to `dataset/processed/`. Other Python commands may be added later.

Scripts will be versioned `.py` files. Experiments must retain configurations,
seeds, commands, and dataset references for reproducibility. Training datasets
must remain available for the project defense, without requiring Git tracking.

## Current status

Model generation, initialization, serialization, FEN encoding, dataset loading,
forward propagation, and analyzer prediction are implemented. C++ tests and
Python CLI tests run through `make test`.

Dataset cleaning is implemented. Training, dataset splitting, and evaluation
scripts remain to be implemented.
