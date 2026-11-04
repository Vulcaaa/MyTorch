# MyTorch

MyTorch is a neural network project written from scratch in C++17 to classify
chess positions as **check**, **checkmate**, or **neither**, using FEN notation.
It combines a reusable neural network module with two command-line programs:

- `my_torch_generator`: creates neural networks from configuration files.
- `my_torch_analyzer`: classifies chess positions; training is planned.

Network generation, saving, loading, and prediction are implemented. Training
is still in development. No neural network framework is used.

See the [training guide](docs/training-command.md) for usage, options,
early stopping, history, and saving.

## Build

Requires GNU Make and a C++17 compiler (GCC by default); tests also require Python 3.

- `make` — build both programs
- `make test` — build and run all tests
- `make clean` — remove build files
- `make fclean` — remove build files and executables
- `make re` — rebuild everything

## Usage

Generate three networks from the example configuration:

```sh
./my_torch_generator configs/basic_network.conf 3
```

Models are saved in `models/` under the current directory as `basic_network_1.nn`,
`basic_network_2.nn`, and `basic_network_3.nn`. The folder is created automatically
and ignored by Git. Existing names are skipped.
The configuration controls the network architecture and initialization seed.

Classify positions from a FEN dataset using a saved model:

```sh
./my_torch_analyzer --predict models/basic_network_1.nn positions.txt
```

The analyzer prints one class (`Nothing`, `Check`, or `Checkmate`) per position,
in order. A generated model is untrained, so its predictions are only
illustrative. See the [chess data guide](docs/datasets.md#prediction-command)
for input and model requirements.

Run either executable with `--help` for its command-line options.

Evaluate a saved model on labeled positions with
`python3 scripts/evaluate.py MODEL DATASET --output results/RUN`.
Prepare a configuration/seed experiment with
`make experiment OUTPUT=results/experiments/preparation OPTIONS=--prepare`.
Full experiments await the training backend. See the
[evaluation guide](docs/evaluation.md) for the workflow and two analysis notebooks.

## Development

See the [project roadmap](docs/project-roadmap.md) for the implementation,
training, experiment, and defense checklist.

- `src/neural/` — shared neural network code (headers beside sources)
- `src/generator/` — model generator
- `src/analyzer/` — chess analyzer
- `configs/` — example configurations
- `tests/` — tests

C++ handles network computation and the command-line programs. Python handles
CLI tests, dataset preparation, experiment orchestration, and result analysis.
Prepare data with the [cleaning notebook](notebooks/clean_dataset.ipynb) and
[split notebook](notebooks/split_dataset.ipynb). See the
[dataset guide](docs/datasets.md) for encoding, cleaning, and splitting, and the
[architecture guide](docs/architecture.md) for details.
Experiments use the [frozen reference dataset](docs/datasets.md); verify it with
`python3 -m scripts.reference_dataset verify` before a campaign.

CI builds and tests the project, then synchronizes successful pushes on `main`
to the Epitech repository.
