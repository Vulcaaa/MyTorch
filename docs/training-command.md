# Training with `--train`

Train a saved chess model using separate labeled training and validation files.
The command handles mini-batches, early stopping, loss history, and model saving.
The neural calculations from #5 are still pending in this branch; training
currently fails with an explicit error until they are integrated.

## Usage

Run from the repository root after building and preparing the
[reference dataset](datasets.md#4-frozen-reference-datasets):

```sh
./my_torch_analyzer --train \
  --validation dataset/references/chess-v1/splits/validation.txt \
  --save models/trained.nn \
  models/basic_network_1.nn dataset/references/chess-v1/splits/train.txt
```

`LOADFILE` must be an existing model with 777 inputs and 3 outputs. Both datasets
must contain six-field FENs followed by a class label. Train and validation must
have no shared model-visible positions; the command rejects overlap. Keep the
test set for final evaluation.

## Options

Place options before the two positional paths, `LOADFILE CHESSFILE`.

| Option | Default | Purpose |
| --- | --- | --- |
| `--validation FILE` | Required | Separate validation set; never used for weight updates |
| `--epochs N` | `100` | Maximum epochs |
| `--batch-size N` | `32` | Training samples per update |
| `--patience N` | `10` | Epochs without sufficient improvement before stopping |
| `--min-delta X` | `0.0001` | Validation-loss improvement required to reset patience |
| `--history FILE` | Automatic path | CSV destination; an existing file is rejected |
| `--save FILE` | `LOADFILE` | Output model; use a distinct path to preserve the input model |

Epochs, batch size, and patience must be positive integers. Minimum improvement
must be finite and nonnegative. Learning rate and seed come from the loaded
model. The seed controls training order; reproducibility requires the same
inputs, settings, build, and runtime environment.

## Stopping and saved model

After each epoch, the command measures mean training and validation losses
without updating weights. It stops at the epoch limit or when patience runs out.
Patience resets when validation loss improves by more than `--min-delta`
relative to the last patience reset.

The saved model always has the lowest observed validation loss, including the
initial model if no epoch improves it. Checkpoint selection tracks every decrease,
including improvements smaller than `--min-delta`.

With `--save`, the source model remains unchanged. Without it, successful training
replaces `LOADFILE`. Saving uses a temporary file and rename; failed training
does not replace the destination model.

## History and diagnostics

Each run writes `epoch,train_loss,validation_loss` to a CSV file. Without
`--history`, it creates a unique `logs/train_<date-time-id>/history.csv` and prints
the path on stderr. With an explicit path, missing parent directories are
created and existing files are never overwritten.

Progress and errors go to stderr. Success returns `0`; invalid arguments,
invalid data, training failures, and history/save errors return `84`. A failed
run may retain a partial history for diagnosis.

For experiment comparison and charts, see the [evaluation guide](evaluation.md).
