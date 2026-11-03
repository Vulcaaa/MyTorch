# Training integration (pending backend)

The experiment runner is ready to call a training backend; it does not implement
backpropagation. The current analyzer's `--train` branch returns without training.
Therefore `configs/experiment.json` leaves `training.command` and `training.settings`
null, and full experiments stop before creating an output directory. `--prepare`
can generate initial models independently.

## Contract to connect with Gustave

Set `training.command` to an argument array for the agreed backend/adapter. No
shell is used. It must consume `{initial_model}`, `{trained_model}`, and
`{training_config}`. The command runs from the MyTorch root. Available placeholders:

| Placeholder | Value |
| --- | --- |
| `{initial_model}` | Generated model for this configuration and seed; must remain unchanged |
| `{trained_model}` | New output file `trained.nn` for this run |
| `{training_config}` | Per-run `training.json`: seed, common settings, train/validation paths and hashes |
| `{history}` | Optional per-run `history.csv` output |
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
checkpoint selection follows the common frozen training settings.

**Still to agree:** executable/arguments, settings schema, checkpoint policy, loss
logging and the small final configuration grid. The supplied network variants are
only a starting comparison. No adapter pretending to train is shipped.
