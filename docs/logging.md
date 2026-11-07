# Run logs

Both programs record each generation or training run in its own timestamped
log file. Logs make it possible to inspect what a run did and why it failed
after the fact. They complement the messages printed on stderr and the
training loss history; they replace neither.

## Where logs are written

| Command | Log file |
| --- | --- |
| `my_torch_generator ...` | `logs/generation_<YYYY-MM-DD_HHMMSS>.log` |
| `my_torch_analyzer --train ...` | `logs/train_<YYYY-MM-DD_HHMMSS>.log` |

- `logs/` is relative to the current directory, like `models/`. It is created
  automatically and ignored by Git.
- The time in file names and entries is **UTC**.
- **One file per run, never overwritten.** When the name is already taken (two
  runs in the same second, an older file, or a symbolic link), a suffix `_2`,
  `_3`, ... is added. The name is reserved atomically, so concurrent runs always
  get distinct files.
- **Not logged:** `--help`, prediction (`--predict`), and command-line argument
  errors, which are detected before a run starts and reported on stderr.

## Entry format

```text
2026-11-05T19:00:28.070Z [INFO] Generated models/basic-network_1.nn (seed 42)
```

Each entry is one line: a UTC timestamp with milliseconds, a level, and a
message. Line breaks inside a message are escaped as `\n` and `\r`. Every entry
is written to disk immediately, so the log of an interrupted run stays readable
up to its last entry. Durations are written in seconds with millisecond
precision, such as `0.081 s`.

| Level | Meaning |
| --- | --- |
| `INFO` | Normal progress of the run |
| `WARN` | Unusual but non-fatal situation (available to the code, not emitted by the current programs) |
| `ERROR` | Failure that stops the run; the same message is printed on stderr |

## Generation log

The log records each configuration with its settings, every saved model with
its seed, and the total time. Example for `./my_torch_generator basic-network.yml 2`:

```text
2026-11-05T19:00:28.028Z [INFO] Generation started
2026-11-05T19:00:28.028Z [INFO] Configuration basic-network.yml: layers=777,128,64,3 hidden_activation=sigmoid output_activation=softmax initialization=xavier learning_rate=0.01 seed=42
2026-11-05T19:00:28.070Z [INFO] Generated models/basic-network_1.nn (seed 42)
2026-11-05T19:00:28.109Z [INFO] Generated models/basic_network_2.nn (seed 43)
2026-11-05T19:00:28.109Z [INFO] Generation complete: 2 models in 0.081 s
```

A failed run ends with an `ERROR` entry, while stderr and the exit code (`84`)
are unchanged:

```text
2026-11-05T19:00:28.114Z [ERROR] Generation failed after 0.000 s: network count must be a positive integer
```

## Training log

The log records the loaded model and its configuration, the train and
validation sets, the [training settings](training.md), the initial validation
loss, the history path, every epoch with its losses and duration, early
stopping, and the saved model with the epoch it comes from. Example on 1,200
positions with an explicit validation file:

```text
2026-11-05T20:21:16.927Z [INFO] Training started
2026-11-05T20:21:16.945Z [INFO] Model models/basic-network_1.nn: layers=777,128,64,3 hidden_activation=sigmoid output_activation=softmax initialization=xavier learning_rate=0.01 seed=42
2026-11-05T20:21:16.949Z [INFO] Train set train.txt: 1200 positions
2026-11-05T20:21:16.949Z [INFO] Validation set val.txt: 300 positions
2026-11-05T20:21:16.949Z [INFO] Settings: epochs=3 batch_size=32 patience=10 min_delta=0.0001
2026-11-05T20:21:16.963Z [INFO] Initial validation_loss=1.37984
2026-11-05T20:21:16.963Z [INFO] History: logs/train_20261105T202116Z_92967_0/history.csv
2026-11-05T20:21:17.103Z [INFO] Epoch 1/3: train_loss=1.07059 validation_loss=1.0809 (0.140 s)
2026-11-05T20:21:17.243Z [INFO] Epoch 2/3: train_loss=1.07023 validation_loss=1.08041 (0.139 s)
2026-11-05T20:21:17.383Z [INFO] Epoch 3/3: train_loss=1.06995 validation_loss=1.08096 (0.140 s)
2026-11-05T20:21:17.394Z [INFO] Saved model from epoch 2 (validation_loss=1.08041) to trained.nn
2026-11-05T20:21:17.394Z [INFO] Training complete in 0.466 s: epochs_completed=3 best_epoch=2
```

Without `--validation`, the validation line describes the automatic split, and
the train set counts the remaining positions:

```text
[INFO] Validation set: 1 position held out from positions.txt (90/10 per class, seed 42)
```

When early stopping triggers, an entry such as
`Early stopping after epoch 4 (patience=2)` precedes the saved model.

The two training records serve different purposes:

| Record | Content | Use |
| --- | --- | --- |
| `history.csv` | Train and validation loss per epoch, full precision | Plots and automated comparisons |
| Training log | Timeline with context: model, data, settings, durations, saved model; losses with six significant digits | Reading a run, diagnosing failures, measuring training cost |

## When logging fails

**Logging never changes the result of a command.** If `logs/` or the log file
cannot be created, or an entry cannot be written (permissions, a file named
`logs`, a full disk), the program prints one warning on stderr and continues
without a log:

```text
warning: logging disabled: filesystem error: in create_directories: File exists ["logs"]
```

The reason comes from the system, so its wording varies. Exit codes and stdout
are unchanged. After a write failure, earlier entries
stay in the file and nothing more is written to it.

Automatic training histories are also stored under `logs/`. When `logs/` is
unusable, training therefore still needs `--history FILE` elsewhere; this is a
history error, not a logging one.

## For developers

The logger lives in `src/logging/` and depends on neither the neural nor the
chess module.

```cpp
#include "logging/Logger.hpp"

Logger log("logs", "train", std::cerr);
log.info("Training started");
log.error("Training failed after " + formatSeconds(elapsed) + ": " + reason);
```

| Function | Role |
| --- | --- |
| `Logger(directory, prefix, diagnostics, start = now)` | Creates the run's file; reports failures once on `diagnostics` |
| `info`, `warn`, `error`, `log(level, message)` | Append one entry; never throw |
| `enabled()`, `path()` | Whether entries are still written; the file, empty if it could not be created |
| `formatLogEntry(time, level, message)` | Formats one entry |
| `formatSeconds(duration)` | Formats a duration such as `0.081 s` |

An empty prefix or one containing `/` throws `std::invalid_argument`: it is a
programming error, not a runtime condition. The analyzer passes its logger to
`runTraining(network, arguments, progress, log)`; the configuration summary
comes from `describeNetworkConfig` in `src/neural/NetworkConfig.hpp`.

## Verification

| Test file | What it checks |
| --- | --- |
| `tests/Logger.cpp` | Entry format, durations, file names, collisions and symbolic links, immediate writes, creation and write failures |
| `tests/Training.cpp` | Training log content with the scripted backend, including early stopping and a failed run |
| `tests/cli.py` | Generation and training logs end to end, failure entries, unchanged stderr and exit codes, no log for help or prediction, runs that continue when `logs/` is unusable |

## Limitations

- A training log (`train_2026-11-05_202116.log`) and its automatic history
  directory (`train_20261105T202116Z_<pid>_<n>/`) use different names; the log
  records the history path to link them.
- Logs are never rotated or deleted; remove old ones manually.
- Command-line argument errors are not logged.
