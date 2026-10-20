# MyTorch

MyTorch is a neural network project written from scratch in C++17 to classify
chess positions as **check**, **checkmate**, or **neither**, using FEN notation.
It combines a reusable neural network module with two command-line programs:

- `my_torch_generator`: creates neural networks from configuration files.
- `my_torch_analyzer`: will train models and classify chess positions.

Network generation, saving, and loading are implemented. Training and prediction
are still in development. No neural network framework is used.

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

Models are saved in the current directory as `basic_network_1.nn`,
`basic_network_2.nn`, and `basic_network_3.nn`. Existing names are skipped.
The configuration controls the network architecture and initialization seed.

Run either executable with `--help` for its command-line options.

## Development

Shared network code is in `src/neural/`, with headers beside their source files.
The two programs are in `src/generator/` and `src/analyzer/`; example configurations
and tests are in `configs/` and `tests/`.

Run `make test` to execute the test suite (requires Python 3). CI builds and tests
the project, then synchronizes successful pushes on `main` to the Epitech repository.
