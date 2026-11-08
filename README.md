# MyTorch

A supervised neural network, implemented from scratch in **C++17**, that classifies
chess positions in FEN notation as **Nothing**, **Check**, or **Checkmate**.
The generic neural module is independent of chess; Python prepares datasets and
organizes experiments. No neural network framework is used.

## Build

Requires GNU Make and a C++17 compiler (`g++` by default).
[yaml-cpp](vendor/yaml-cpp/README.mytorch.md) and the JSON headers are bundled.
Building and using the two executables does not require Python.

```sh
make
```

The build uses Accelerate on macOS, OpenBLAS when detected on Linux, or a portable
scalar backend. Use `make BLAS=none` when the target machine has no BLAS library.
See [CPU acceleration](docs/training.md#cpu-blas-backend) for installation and
thread controls. `make clean`, `make fclean` and `make re` are also available.

## Predict with the delivered model

Run from the repository root. Create a file containing one six-field FEN per line,
then load the pre-trained `my_torch_network.nn`:

```sh
printf '%s\n' 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' > /tmp/mytorch-positions.fen
./my_torch_analyzer --predict my_torch_network.nn /tmp/mytorch-positions.fen
```

Predictions appear on stdout in input order: `Nothing`, `Check White`,
`Check Black`, `Checkmate White`, or `Checkmate Black`. The color is the attacking
side, derived from the FEN side to move. Optional input labels do not influence
predictions. Errors go to stderr with exit code `84`; success returns `0`.

The [delivered-model provenance](docs/report/2026-11-08-mirror-view-ensemble.md)
describes the checkpoint currently at the repository root; its scores are in
[Results](#results).
Other [shared candidates](models/README.md) are separate from the delivered model.

## Generate and train

```sh
./my_torch_generator configs/networks/basic-network.yml 1
./my_torch_analyzer --train --validation validation.txt --save models/trained.nn \
  models/basic-network_1.nn train.txt
```

Replace `train.txt` and `validation.txt` with prepared labeled datasets. The
generator creates `models/basic-network_1.nn`, skipping existing names; use the
path it reports. Generated networks are **untrained**. Omitting `--validation`
uses a seeded 90/10 split; omitting `--save` replaces the input model after
successful training. Both executables provide `--help`.

The [training guide](docs/training.md) explains settings, early stopping and
SGD/Adam. For reproducible runs with recorded configurations, dataset hashes,
loss curves and metrics, use the [experiment runner](docs/evaluation.md).

## Python tools and tests

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
make test
```

`make test` runs C++ checks and Python integration/analysis tests. The optional
`scripts/requirements.txt` enables legal-game generation and its tests;
`notebooks/requirements.txt` adds notebook tools. `make smoke` runs a short
generation/training/evaluation integration check on temporary data.

CI builds and tests pull requests. Successful builds on `main` are mirrored to
the Epitech repository.

## Results

The delivered `my_torch_network.nn` reaches **93.66% accuracy and 93.15% macro F1**
on the reserved test split (71,555 positions), against 88.06% validation accuracy
for the previously delivered model. It is a 777 → 256 → 128 → 3 ReLU network trained with
Adam, dropout and left-right mirrored data; [`train-delivered-model.yml`](configs/protocols/train-delivered-model.yml)
retrains it. Benchmarks and figures: [benchmark summary](docs/benchmarks.md) and
[report index](docs/report/README.md).

## Documentation

- [Architecture and configuration](docs/architecture.md): module boundaries, model format and design choices.
- [Datasets](docs/datasets.md): 777-input encoding, frozen splits, color mirrors and generated positions.
- [Data rights and predictive limitations](docs/datasets.md#8-data-rights-and-predictive-limitations): Epitech-provided data, scope of use and model reliability.
- [Training](docs/training.md), [evaluation](docs/evaluation.md) and [logs](docs/logging.md): commands and reproducibility.
- [Hyperparameter choices](docs/hyperparameters.md): model versus training settings, measured alternatives and remaining decisions.
- [Trial and failure log](docs/experiment-decisions.md): negative results, corrections and what the evidence supports.
- [Training performance history](docs/report/training-history.md): experiments, measured gains, explanations and figures for the defense.
- [Detailed experiment archive](docs/report/README.md) and [CPU benchmarks](benchmarks/cpu_blas/README.md): protocols and supporting evidence.
- [Project requirements and defense checklist](docs/project-roadmap.md): competencies and where to find supporting evidence.

Large datasets and most raw run artifacts remain local. Tracked manifests,
protocols and portable report data identify them; retain the original files for
reproduction and the defense. See the [dataset guide](docs/datasets.md#4-frozen-reference-datasets)
before preparing or restoring training data.
