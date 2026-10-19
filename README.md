# MyTorch
A neural network built from scratch in C++ to classify chess positions as check, checkmate, or neither.

The project will provide a reusable neural network module for creating, saving,
loading and training models, and a supervised chess position classifier using FEN
input. Neural network frameworks are not used.

## Build

Requires GNU Make and a C++17 compiler (GCC by default).

```sh
make
./my_torch_analyzer --help
```

`make clean` removes object files, `make fclean` also removes the executable, and
`make re` rebuilds from scratch. Sources are in `src/`; build files are in `build/`.

## Current status

The build system, help command and CI are ready. Training, prediction and the
neural network module are not implemented yet. Unsupported invocations report an
error on stderr and return 84.

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
