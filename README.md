# MyTorch

MyTorch is a neural network project written from scratch in C++17 to classify
chess positions as **check**, **checkmate**, or **neither**, using FEN notation.
It combines a reusable neural network module with two command-line programs:

- `my_torch_generator`: creates neural networks from configuration files.
- `my_torch_analyzer`: classifies chess positions; training is planned.

Network generation, saving, loading, and prediction are implemented. Training
is still in development. No neural network framework is used.

## Build

Requires GNU Make and a C++17 compiler (GCC by default).

```sh
make
```

Use `make clean` to remove object files, `make fclean` to also remove the
executables, or `make re` to rebuild everything.

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

The analyzer prints one of `Nothing`, `Check`, or `Checkmate` per position, in
file order. A newly generated model has not been trained, so its predictions
are only useful for checking the command. See the [chess data guide](docs/data-encoding.md#prediction-command)
for the input format and model requirements.

Run either executable with `--help` for its command-line options.

## Development

Shared network code is in `src/neural/`, with headers beside their source files.
The two programs are in `src/generator/` and `src/analyzer/`; example configurations
and tests are in `configs/` and `tests/`.

C++ handles network computation and the command-line programs. Python handles
CLI tests and will provide dataset preparation, experiment orchestration, and
result analysis through Makefile targets. See the [architecture guide](docs/architecture.md)
for component responsibilities, data flow, and the testing strategy.

Run `make test` to execute the test suite (requires Python 3). CI builds and tests
the project, then synchronizes successful pushes on `main` to the Epitech repository.
