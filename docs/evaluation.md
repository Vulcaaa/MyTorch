# Evaluation workflow

## One experiment, one configuration file

Edit [configs/experiment.json](../configs/experiment.json): it defines the network
configurations, common training settings, dataset paths, and seeds **once**.
Paths in this file are relative to its directory. The supplied compact/reference
pair is an initial capacity comparison, not a tuned grid or a final choice.

The default paths use the frozen `chess-v1` reference. On a fresh checkout,
[recreate and verify it](datasets.md) before preparing an experiment. The
reference files are retained locally, separately from notebook outputs.

While training is unavailable, prepare all initial models:

```sh
make experiment OUTPUT=results/experiments/preparation OPTIONS=--prepare
```

This generates one model per configuration/seed, freezes the plan and file hashes,
and writes a manifest and an empty-results summary. It performs **no training or
evaluation**. `training.command` and `training.settings` intentionally remain null.

Once the neural calculations and the adapter connecting the runner to
[`--train`](training-command.md) are implemented and `training.command` /
`training.settings` are configured, full experiments can run:

```sh
make experiment OUTPUT=results/experiments/comparison-01
```

The same command now generates, trains, evaluates on validation, and summarizes.
Use `CONFIG=PATH` for another experiment file. Each output directory must be new;
there is no automatic resume. Start a new experiment after changing the plan.

## Read the results

| Goal | Notebook | Setting |
| --- | --- | --- |
| Compare configurations, variability across seeds, costs, and available training curves | [comparison.ipynb](../notebooks/comparison.ipynb) | `EXPERIMENT`: experiment output directory |
| Inspect one model's class scores and confusion matrix | [evaluation.ipynb](../notebooks/evaluation.ipynb) | `RUN`: its `runs/CONFIG/seed-SEED/evaluation` directory |

Each notebook runs independently; neither trains nor predicts. Use the project's
`.venv` kernel. Install dependencies there with
`python -m pip install -r notebooks/requirements.txt`.

The comparison notebook can still compare standalone validation reports with
`EXPERIMENT = None` and `RUNS`. Without an experiment plan, it cannot infer seeds
or group configurations. This mode is for individual-run inspection.

An experiment retains:

- `experiment.json`: frozen plan, datasets/class counts, source and binary hashes,
  machine information, run states, and errors.
- `summary.json`: configuration means/sample standard deviations and completed
  run details. The notebook recomputes it from the reports rather than trusting a stale cache.
- `runs/CONFIG/seed-SEED/`: network/training settings, initial/trained models,
  generation/training commands and logs, elapsed training time, model size,
  optional history, and evaluation reports.

Failed experiments retain completed runs and logs. Partial groups show missing
seeds; they cannot establish a leading configuration. A singleton has no estimated
standard deviation. Preparing models does not produce quality scores.

## Evaluate one model independently

```sh
python3 scripts/evaluate.py MODEL DATASET --output results/RUN
```

Use a saved `.nn` model, a labeled [FEN dataset](datasets.md), and a new output
directory. This command remains the entry point for the selected model's final
test evaluation, and for inspecting an existing model without retraining.

| Report | Contents |
| --- | --- |
| `metrics.json` | Accuracy, macro F1, precision/recall/F1 and support per class; artifact hashes, prediction command and elapsed time. |
| `predictions.csv` | Original dataset line number, actual class, predicted class. |
| `confusion_matrix.csv` | Actual classes in rows, predictions in columns: Nothing, Check, Checkmate. |

Undefined scores are zero. Macro F1 includes all three classes. Prediction time
includes startup, model loading, parsing, inference, and output capture; training
time includes the training command's startup/loading. These are single elapsed
measurements, not isolated speed benchmarks.

## Comparison protocol v1

1. Use prepared, disjoint train/validation/test splits. Train on train; compare on
   validation. Keep the test set out of the experiment file and reserve it for the
   final selected model. Retain datasets and split metadata for the defense.
2. Freeze the configuration grid, training budget, optimizer, and checkpoint rule
   before comparing. Common settings apply to every run; only declared network
   options and seeds vary. The training backend remains responsible for applying
   the requested seed to initialization and training randomness.
3. Use every seed listed in the experiment file for every configuration. Seeds
   override the base network file's default. Repeating prediction on one model
   does not measure training variability. Never discard a poor seed.
4. Rank configurations by mean validation macro F1; exact ties use mean
   worst-class recall. Report sample standard deviation and inspect per-class
   scores/confusions. Equal scores remain tied. These statistics are descriptive,
   not proof of significance. A single-seed exploratory experiment cannot establish
   a leading configuration.
5. Use the same validation SHA-256, analyzer SHA-256, class order and supports.
   Compare costs on the same machine without concurrent heavy work. Do not use
   single-run timing to break score ties. All planned configurations must finish
   all planned seeds before the summary identifies leaders.
6. Freeze the selected configuration **and the rule for choosing its delivered
   model** before test evaluation. Do not cherry-pick a seed or tune using test
   results. The runner reports leaders; it does not choose/export a final model.

The training adapter and experiment settings still need to be configured.
Final training comparisons, test results, interpretation, and model delivery
remain open parts of issue #8.
