# MyTorch
A neural network built from scratch in C++ to classify chess positions as check, checkmate, or neither.

The project will provide a reusable neural network module for creating, saving,
loading and training models, and a supervised chess position classifier using FEN
input. Neural network frameworks are not used.

## Build

Requires GNU Make and a C++17 compiler (GCC by default).

```sh
make
./my_torch_generator --help
./my_torch_analyzer --help
```

`make` builds both executables. Use `make my_torch_generator` or
`make my_torch_analyzer` to build one executable.
`make clean` removes object files, `make fclean` also removes both executables, and
`make re` rebuilds from scratch. Build files are in `build/`.

- `include/mytorch/`: shared module headers.
- `src/neural/`: shared network, model I/O and configuration implementation.
- `src/generator/`: network generator entry point.
- `src/analyzer/`: training and prediction entry point.

The generator writes `<config-name>_<number>.nn` files in the current directory,
starting at 1 and skipping occupied names (including directories and symbolic
links). Repeated runs and configurations with the same base name use the next
available number.

## Model serialization

`ModelSerializer` saves and loads JSON models, regardless of the file extension:

```cpp
#include "mytorch/ModelSerializer.hpp"

ModelSerializer::save(network, "model.nn");
NeuralNetwork restored = ModelSerializer::load("model.nn");
```

The format contains `"format": "mytorch"`, `"version": 1`, the complete
`config`, and a `layers` array containing the weights and biases of each parameter
layer. Weight rows correspond to neurons, and columns correspond to their inputs.
Loading restores the saved parameters without random initialization. JSON number
output preserves the precision needed to recover the original `double` values.

Invalid configuration, incompatible dimensions, non-numeric or non-finite
parameters, unsupported versions, and file errors throw `std::runtime_error`
with the file path. Run `make test` to check serialization and invalid input handling.

The JSON parser is nlohmann/json v3.11.3, included in `include/nlohmann/`
under the MIT license. No download is required to build.

## Tests

Run `make test` (requires Python 3 for the executable tests). This runs the C++
tests for configuration parsing, Xavier/He initialization, network construction,
and model serialization, followed by end-to-end checks of the generator and the
analyzer's help command. Training and prediction are not yet
implemented and are not tested. All test files are created in temporary directories.
The same suite runs in CI.

## Implementation status

The shared module supports configuration parsing, weight initialization, and JSON
model serialization. The generator and analyzer entry points are still under
development; training and prediction are not implemented yet.

## CI and Epitech synchronization

Pull requests and pushes to `main` compile the project. After a successful build
on `Vulcaaa/MyTorch`'s `main`, the same commit is pushed to:

`EpitechPGE3-2026/G-CNA-501-MPL-5-1-mytorch-1`

Only `main` is synchronized; branches and tags are not mirrored. Work on the
development repository and avoid direct commits on Epitech's `main`. The workflow
does not force pushes: a divergent Epitech history causes synchronization to fail.

Configure the Actions secret `SSH_PRIVATE_KEY` on `Vulcaaa/MyTorch` with a private
SSH key authorized to push to the Epitech repository. When repository admin access
is unavailable, add the matching public key under the pushing GitHub account's
Settings > SSH and GPG keys. That account must already have write access to the
Epitech repository. A deploy key with write access is an alternative when an
Epitech repository administrator can configure it. Never commit the private key.
