# Training integration (pending backend)

The experiment runner is ready to call a training backend; it does not implement
backpropagation. The analyzer parses the approved training options and calls
`runTraining` from its dispatcher in `src/analyzer/main.cpp`. Prediction is
implemented separately in `src/analyzer/Prediction.cpp`. The training loop loads
both datasets and checks their independence, but `--train` returns 84 when it
reaches the pending neural loss calculation from issue #5.
Therefore `configs/experiment.json` leaves `training.command` and `training.settings`
null, and full experiments stop before creating an output directory. `--prepare`
can generate initial models independently.

## Runner contract

The analyzer CLI and checkpoint behavior are documented in the
[training guide](training-command.md). An adapter still needs to connect the
experiment request below to that CLI.

Set `training.command` to an argument array for the agreed backend/adapter. No
shell is used. It must consume `{initial_model}`, `{trained_model}`, and
`{training_config}`. The command runs from the MyTorch root. Available placeholders:

| Placeholder | Value |
| --- | --- |
| `{initial_model}` | Generated model for this configuration and seed; must remain unchanged |
| `{trained_model}` | New output file `trained.nn` for this run |
| `{training_config}` | Per-run `training.json`: seed, common settings, train/validation paths and hashes |
| `{history}` | Per-run `history.csv` destination; the runner permits backends without history |
| `{seed}`, `{train}`, `{validation}` | Seed and dataset paths, also present in the request file |
| `{analyzer}`, `{root}` | Analyzer executable and project directory |

`training.settings` must be a non-empty object with the backend's agreed optimizer,
training budget and checkpoint rule. Its exact fields are deliberately not invented
here. The backend must validate/apply them and seed its own random generators.
The request file records the common settings but does not implement them.

The backend must return nonzero on failure, or zero after writing a valid MyTorch
model to `{trained_model}`. Preserve its network configuration and seed. The runner
rejects missing output, a changed initial model/configuration, and unchanged
weights/biases; it then uses the real analyzer to validate and predict the output.
This detects a no-op adapter but does not prove that a backend actually learned.

Stdout/stderr, exact command, working directory and elapsed time are retained.
Errors/interruption stop the experiment and preserve completed results. No initial
model is silently substituted for a failed training result. No test path is supplied.

## Optional loss history

The analyzer always exports history: it uses an automatic path
without `--history`, or the supplied path when that option is present. The runner
still permits missing histories from other backends.

Emit this CSV at `{history}` after training, or adapt the backend logs to it:

```csv
epoch,train_loss,validation_loss
1,1.12,1.18
2,0.93,1.01
```

Epochs are positive increasing integers; losses are finite nonnegative values.
Columns and order are fixed. Existing malformed histories fail the run; missing
histories are allowed and reported in the notebook. Validation loss is measured
without weight updates. The backend must document loss definition/averaging and
whether the model is the last or a selected checkpoint. Curves are descriptive;
checkpoint selection follows the common frozen training settings. See the
[training guide](training-command.md#stopping-and-saved-model) for the analyzer's
loss measurement and checkpoint rule.

The adapter command, settings schema, and final configuration grid still need
to be configured. The supplied network variants are a starting comparison.
