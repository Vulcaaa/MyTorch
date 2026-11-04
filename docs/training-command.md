# Training command: implementation contract

Status: argument parsing, help, and orchestration are implemented for issue #7.
`runTraining` calls public `NeuralNetwork::trainBatch` and `NeuralNetwork::loss`
methods directly and implements epochs, batches, validation, checkpoints,
history, and saving. `runAnalyzer` in `main.cpp` calls `runTraining` in train mode
and `runPrediction` from `Prediction.cpp` otherwise. Issue #5 implements the
neural calculations behind the contract below; see
[backpropagation](backpropagation.md). CLI parameters and automatic history
export are approved.

## Responsibilities

- Issue #5 owns loss calculation, derivatives, gradient
  aggregation, and parameter updates in `src/neural/`.
- Issue #7 owns arguments, labeled dataset loading, chess compatibility checks,
  reproducible epoch ordering, batches, validation, early stopping, history,
  checkpoint selection, and saving in `src/analyzer/`.
- `main.cpp` dispatches to `Training.cpp` or `Prediction.cpp` and handles parsing,
  help, and the existing exit codes.
  Neither the neural module nor the dataset loader owns the training loop.

## Neural API for issue #5

`TrainingSample` is defined in `src/neural/NeuralNetwork.hpp`, and the read-only
`config()` accessor is implemented. The public `trainBatch` and `loss` methods
are declared in the same header and implemented in `src/neural/Training.cpp` as
specified below; the [backpropagation guide](backpropagation.md) details the
calculations:

```cpp
struct TrainingSample {
    std::vector<double> inputs;
    std::vector<double> target;
};

// Public NeuralNetwork methods:
double trainBatch(const std::vector<TrainingSample>& batch);
double loss(const std::vector<TrainingSample>& samples) const;
const NetworkConfig& config() const;
```

`TrainingSample` belongs to the neural module and contains only numeric vectors.
Issue #7 converts labeled `DatasetSample` values to this type. Chess inputs have
777 values, with one-hot targets ordered `Nothing`, `Check`, `Checkmate`; the
neural API must work with other compatible dimensions as well.

### Mini-batch update

`trainBatch` evaluates every sample against the same parameters, averages its
gradients over the actual batch size, and performs one SGD update of weights and
biases using the network configuration's `learning_rate`. The final partial batch
uses its own size. Return the mean sample loss **before** the update. This is
mini-batch SGD by default (32 samples per update), not one update per epoch.

Do not shuffle, seed a generator, load files, print metrics, or save a model in
this method. Issue #7 supplies the samples in their intended order.

### Loss and validation

`loss` returns the mean sample loss at the current parameters and never updates
parameters. Both methods use the same loss definition:

- Softmax output: categorical cross-entropy, summed over outputs per sample.
- Sigmoid output: binary cross-entropy, summed over outputs per sample.

Use numerically stable calculations from logits where needed. Hidden activations
remain `sigmoid`, `relu`, or `tanh`. Reject unsupported activation/loss
combinations explicitly. Softmax targets are finite values in [0, 1] summing to
one within a documented tolerance; sigmoid targets are finite values in [0, 1].

Reject empty collections, wrong input/target dimensions, nonfinite inputs or
targets, and invalid learning rates before any batch update. Use
`std::invalid_argument` for invalid arguments and `std::runtime_error` for
nonfinite calculated loss, gradients, or parameters. An execution error aborts
the run; issue #7 does not save a failed run's in-memory model.

`config()` exposes a read-only reference to the existing configuration, including
learning rate and seed. Training changes parameters, not architecture,
activations, initialization metadata, learning rate, or seed. Existing prediction,
serialization, and value-copy behavior must remain usable for checkpoints.

## Approved command-line parameters

```sh
./my_torch_analyzer --train \
  --validation ../dataset/splits/validation.txt \
  --epochs 100 --batch-size 32 --patience 10 --min-delta 0.0001 \
  --history results/history.csv --save models/trained.nn \
  models/initial.nn ../dataset/splits/train.txt
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `LOADFILE` | Required positional path | Existing model to train |
| `CHESSFILE` | Required positional path | Prepared, labeled training dataset |
| `--validation FILE` | Required | Independent labeled validation dataset; never used for updates |
| `--epochs N` | 100 | Maximum complete passes over the train set; positive integer |
| `--batch-size N` | 32 | Samples per SGD update; positive integer |
| `--patience N` | 10 | Consecutive epochs without sufficient validation-loss improvement; positive integer |
| `--min-delta X` | 0.0001 | Absolute validation-loss improvement needed to reset patience; finite and nonnegative |
| `--history FILE` | Automatic per-run path | Override the CSV destination; history is always enabled |
| `--save FILE` | `LOADFILE` | Model destination; a distinct destination preserves the source model |

Reuse `learning_rate` and `seed` from the loaded model; there are no CLI overrides
for this first version. A fixed SGD algorithm keeps the initial scope small.
Accept training options in any order before the two positional paths. Reject
unknown or repeated options, missing values, and training options in predict mode.

**Approved syntax extension:** mandatory `--validation` extends the minimal syntax in
issue #7. It avoids silently splitting an already prepared training set and makes
early stopping available for every successful training run. The old bare
`--train LOADFILE CHESSFILE` invocation returns 84 with a missing-validation
diagnostic. Parsing validates syntax and values without opening input files or
creating output paths. An empty `training.historyFile` represents the approved
automatic-history mode; the training loop selects its concrete path.

### Approved history behavior

Without `--history`, create a separate directory for each run under
`logs/train_<date-time-id>/` and write `history.csv` there. Reserve each directory
exclusively and retry with another identifier on collision, so repeated or
concurrent runs never overwrite a previous history. Create the directory
automatically and print the selected path on stderr.

Keep `--history FILE` as an optional explicit destination, especially for the
experiment runner. Create missing parent directories and reject an existing
history file rather than overwriting it. Reject destinations that would overwrite
the source model, datasets, or output model. Report history creation or write
failures on stderr and return 84; do not silently disable history or save a model
as if the run had completed successfully.

This CSV export belongs to issue #7. It does not implement the shared severity
logger or generation logs requested by issue #13. `/logs/` is ignored in Git.

## Implemented orchestration behavior

`runTraining(network, arguments, progress)` in `src/analyzer/Training.*` calls
`working.trainBatch(batch)` for updates and `working.loss(samples)` for
measurements. There is no operations struct, callback injection, or virtual
dispatch in production. Both public neural methods consume numeric
`TrainingSample` vectors and return a mean loss. The const qualifier on `loss`
prevents ordinary modification of the network during validation.

The orchestration test binary links test-only definitions of those same methods
in place of `src/neural/Training.cpp`; generated programs always link the real
implementation, never the test definitions.

Load and validate both labeled datasets before updates. Reject shared
model-visible inputs across train and validation, including FENs differing only
in unencoded fields. Repeated rows inside train are allowed for class balancing.
Never load a test set in the training command.

Seed one `std::mt19937` from the model's seed. Each epoch starts from canonical
dataset indices and shuffles them with that generator, then processes successive
batches. Reproducibility is promised for identical inputs, settings, build, and
runtime environment, not bit-identical results across standard libraries.

At the end of each epoch, evaluate train and validation losses using the same
current parameters and without updates. Do not use the average of batch-returned
losses as the reported epoch train loss: those batches used different parameters.
CSV epochs start at 1, as required by the existing experiment runner.

Evaluate initial validation loss before epoch 1 and keep the initial model as
checkpoint 0. Save an in-memory checkpoint whenever validation loss is strictly
lower than the lowest observed value. Maintain a separate patience reference,
initially the initial loss: reset patience and update this reference only when
`validation_loss < patience_reference - min_delta`. Otherwise increment the
counter. Stop at the epoch limit or when the counter reaches patience. Restore
the lowest-loss checkpoint, including checkpoint 0 if no epoch improves it.

Save only after successful training and history export. With an explicit
distinct `--save` destination, leave `LOADFILE` unchanged. Without `--save`, use
the source path. Write the selected model to a uniquely reserved temporary file
beside its destination, then rename it into place. Serialization or rename failure
leaves an existing destination intact and removes the temporary file. This uses
POSIX exclusive file creation, as does history reservation; it does not promise
durability across a power failure. Reject model destinations aliasing either
dataset, history destinations aliasing any model/dataset, and explicit alternate
save paths aliasing the source model.

The loop operates on a copy: failures leave the caller's network unchanged.
Completed history rows are retained for diagnosis even if a later batch or save
fails. A failed run's history may contain only its header. A `TrainingResult`
reports completed epochs, selected epoch (0 for the initial model), selected
validation loss, early stopping, and the history path. Progress is written to the
provided stream, which the CLI integration must connect to stderr. Keep
prediction output on stdout, success code 0, and error code 84.

## Verification

`make test` includes `tests/Training.cpp`. Scripted test neural methods check scheduling
and checkpoints with controlled parameters/losses; they are not evidence of
learning. Tests cover partial batches, sample coverage, identical-seed
reproducibility, changed-seed ordering, independent validation, cumulative patience improvements,
restoring an earlier or initial checkpoint, both save modes, model reload,
overlapping inputs, invalid data/backend losses, automatic histories, existing
history files, path aliases, and history/save failures. A POSIX file-size limit
forces a real history write failure. Learning itself is verified by the neural
tests in `tests/Backpropagation.cpp` and by a real training run in `tests/cli.py`.

## Experiment runner integration

The runner's `training.json` envelope already carries the seed, train and
validation paths, and a `settings` object. Once implemented, an adapter can map
`settings` fields `epochs`, `batch_size`, `patience`, and `min_delta` to these
options, and map `{history}` and `{trained_model}` to the corresponding paths.
The model seed must match the request seed. The adapter must reject unknown or
invalid settings instead of ignoring them.

Do not fill in `configs/experiment.json`'s training command or settings until the
real backend and adapter exist. See [training integration](training-integration.md)
for the existing runner contract; no training adapter is implemented here.
