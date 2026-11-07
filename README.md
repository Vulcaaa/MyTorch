# MyTorch

MyTorch is a neural network project written from scratch in C++17 to classify
chess positions as **check**, **checkmate**, or **neither**, using FEN notation.
It combines a reusable neural network module with two command-line programs:

- `my_torch_generator`: creates neural networks from configuration files.
- `my_torch_analyzer`: trains networks and classifies chess positions.

Network generation, saving, loading, training, and prediction are implemented.
No neural network framework is used.

See the [training guide](docs/training.md) for commands, the training flow,
backpropagation, checkpoints, and experiment integration.

## Build

Requires GNU Make and a C++17 compiler (GCC by default). Python tools and tests
require Python 3 and `python3 -m pip install -r requirements.txt`.
Training uses Accelerate on macOS or OpenBLAS when detected on Linux, with a
portable scalar fallback. See [CPU acceleration](docs/training.md#cpu-blas-backend)
for dependencies, thread settings and benchmarks.

- `make` — build both programs
- `make smoke` — quickly check generation, real training and evaluation
- `make test` — build and run all tests
- `make clean` — remove build files
- `make fclean` — remove build files and executables
- `make re` — rebuild everything

## Usage

Generate three networks from the example configuration:

```sh
./my_torch_generator configs/networks/basic-network.conf 3
```

Models are saved in `models/` under the current directory as `basic-network_1.nn`,
`basic-network_2.nn`, and `basic-network_3.nn`. The folder is created automatically
and ignored by Git. Existing names are skipped.
The configuration controls the network architecture and initialization seed.
Each run is recorded in `logs/generation_<UTC time>.log`; see the
[logging guide](docs/logging.md).

Classify positions from a FEN dataset using a saved model:

```sh
./my_torch_analyzer --predict models/basic-network_1.nn positions.txt
```

The analyzer prints one class (`Nothing`, `Check`, or `Checkmate`) per position,
in order. A generated model is untrained, so its predictions are only
illustrative. See the [chess data guide](docs/datasets.md#prediction-command)
for input and model requirements.

Train the reference network with the frozen train/validation datasets:

```sh
python3 scripts/experiment.py configs/experiments/train-reference.yml \
  --train-only --output results/training/YYYY-MM-DD-reference-01
```

The result directory groups the selected `model.nn`, loss history, execution log,
parameters and a readable `README.md`. See the [training guide](docs/training.md)
for the analyzer's direct command-line interface and checkpoint policy.

Run either executable with `--help` for its command-line options.

Use the [training and evaluation workflow](docs/evaluation.md) to run standalone
training, evaluate a saved model, or compare configurations. Results are grouped
under `results/training/`, `results/evaluation/`, `results/experiments/` only. Each result starts with a readable `README.md`; notebooks
show curves and confusion matrices.

## Development

See the [project roadmap](docs/project-roadmap.md) for the implementation,
training, experiment, and defense checklist.

- `src/neural/` — shared neural network code (headers beside sources)
- `src/generator/` — model generator
- `src/analyzer/` — chess analyzer
- `src/logging/` — run logs shared by both programs
- `configs/networks/` — network architectures and parameters
- `configs/experiments/` — training and comparison protocols
- `tests/` — tests

C++ handles network computation and the command-line programs. Python handles
CLI tests, dataset preparation, experiment orchestration, and result analysis.
Clean and split data with the
[preparation notebook](notebooks/prepare_dataset.ipynb), then inspect class balance
and position diversity with the [dataset analysis notebook](notebooks/dataset_analysis.ipynb). See the
[dataset guide](docs/datasets.md) for encoding, cleaning, and splitting, and the
[architecture guide](docs/architecture.md) for details.
Experiments use the [frozen reference dataset](docs/datasets.md); verify it with
`python3 -m scripts.reference_dataset verify` before a campaign.

CI builds and tests the project, then synchronizes successful pushes on `main`
to the Epitech repository.
