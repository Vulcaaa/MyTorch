# Project architecture

This is the proposed architecture and may evolve during development.

MyTorch uses **C++17 for network computation and executables**, and **Python for
dataset preparation, experiment orchestration, and result analysis**. The
Makefile handles builds, tests, and experiments; notebooks launch dataset preparation.

## Components

| Component | Language | Responsibility |
| --- | --- | --- |
| `src/neural/` | C++ | Generic network: configuration, initialization, serialization, forward propagation, and [training](training.md). |
| `src/chess/` | C++ | FEN parsing, feature encoding, and dataset label handling. |
| `src/generator/` | C++ | `my_torch_generator`: create models from configuration files. |
| `src/analyzer/` | C++ | `my_torch_analyzer`: coordinate training and prediction. |
| `src/logging/` | C++ | Timestamped [run logs](logging.md) for generation and training. |
| `scripts/` | Python | Clean/split datasets, orchestrate experiments, evaluate models, and aggregate comparable runs. |
| `notebooks/` | Python | Launch dataset preparation, inspect evaluation reports, and compare validation runs. |
| `tests/` | C++ and Python | Function tests and executable tests. |

The network module operates on numeric vectors without knowledge of chess or
FEN. The chess module implements the [input](datasets.md) and
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
| `make experiment OUTPUT=PATH OPTIONS=--prepare` | Freeze an experiment and generate initial models; full execution awaits training integration |
| `make clean` | Remove build artifacts |
| `make fclean` | Also remove executables |
| `make re` | Clean and rebuild |

Dataset preparation runs in
[`notebooks/prepare_dataset.ipynb`](../notebooks/prepare_dataset.ipynb): first
`scripts/clean_dataset.py` reads `dataset/raw/` and writes `dataset/processed/`,
then `scripts/split_dataset.py` writes train/validation/test files to
`dataset/splits/`. Settings and reports are grouped in this single notebook.
These are disposable outputs.
`scripts/reference_dataset.py` reuses both preparation tools to freeze a named
reference under `dataset/references/`, with a tracked manifest in
`dataset/manifests/`. Experiments use the [frozen reference](datasets.md).
`make dataset` does not exist.

Experiment commands use `CONFIG` (defaults to `configs/experiment.json`),
`OUTPUT` (required new directory) and optional `OPTIONS`. Python commands use
the `PYTHON` variable, which defaults to `python3`. Third-party dependencies
will be declared in a dedicated file and installed in a virtual environment.

Scripts will be versioned `.py` files. Experiments must retain configurations,
seeds, commands, and dataset references for reproducibility. Training datasets
must remain available for the project defense, without requiring Git tracking.

## Current status

Model generation, initialization, serialization, FEN encoding, dataset loading,
forward propagation, backpropagation with mini-batch gradient descent, and
analyzer training and prediction are implemented. Generation and training runs
are [logged](logging.md) in `logs/`. C++ tests and Python CLI tests run through
`make test`.

The Python evaluation script computes classification metrics through the analyzer;
see the [evaluation guide](evaluation.md). `evaluation.ipynb` inspects one model's
report; `comparison.ipynb` compares individual validation runs and configuration
statistics independently. `experiment.py` prepares/runs the configuration-by-seed
plan, reusing `evaluate.py` and `comparison.py`. Dataset cleaning and splitting
are implemented. The backend training adapter remains to be implemented.
