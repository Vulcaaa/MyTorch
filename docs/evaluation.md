# Training and evaluation workflow

## Where files belong

| Directory | Purpose |
| --- | --- |
| `configs/networks/` | Network architecture and parameters |
| `configs/protocols/` | YAML training and comparison protocols: data, initial model, budget, hypothesis |
| `dataset/` | Shared datasets and frozen versions; new runs reference these files rather than copying them |
| `results/training/` | Standalone training or checkpoint continuation |
| `results/evaluation/` | Standalone validation or final test evaluation |
| `results/experiments/` | Controlled comparisons of configurations and seeds |
| `models/` | Reusable reference and selected models |
| `notebooks/` | Dataset exploration, curves and result inspection |
| `scripts/` | Reusable processing and the single experiment runner |
| `docs/` | Methods, conventions and conclusions |

Use English kebab-case and name result directories `YYYY-MM-DD-purpose-01`,
for example `2026-11-07-continue-reference-01`. Use a new directory for every
execution; existing results are never overwritten. Runs inside an experiment
stay there; do not duplicate them under `training/` or `evaluation/`.

Open **README.md first** in any result directory. No HTML reports, source-code
snapshots, exported PNGs, or dataset-analysis directory are generated. Dataset
analysis remains in its notebook and [dataset documentation](datasets.md).

Historical one-off campaigns may keep frozen input snapshots under ignored
`results/`. In particular, `train-large-relu-adam.yml` references retained local
Adam-campaign data. Preserve those bytes and hashes until dependencies are
verified; these are historical archives, not the directory pattern for new runs.
See the [preparation inventory](datasets.md#why-there-are-several-preparation-directories).

## Network configuration

`configs/networks/basic-network.yml` defines the architecture and defaults:

```yaml
layers: [777, 128, 64, 3]
hidden_activation: sigmoid
output_activation: softmax
optimizer: sgd
learning_rate: 0.01
initialization: xavier
seed: 42
```

Protocol `network` paths reference this file. Architecture overrides also use
YAML lists, for example `layers: [777, 64, 3]`. The protocol's seed list overrides
the network seed. Continuations read settings from the saved `.nn` model instead.

## Local model collection

`models/` holds locally generated networks and selected checkpoints. The tracked
[model inventory](../models/README.md) identifies the shared candidate
`models/large-relu-adam-seed46-lr0.001.nn` and its provenance. Historical local
models use the `legacy-` prefix; most generated models are ignored by Git.
The delivered checkpoint is `my_torch_network.nn` at the repository root.
Treat candidate validation results separately from delivered-model results.

## Train one model

Build with `make`. Protocol paths are relative to the configuration file.
For a fresh reference model:

```sh
python3 scripts/experiment.py configs/protocols/train-reference.yml \
  --train-only --output results/training/YYYY-MM-DD-reference-01
```

The supplied continuation starts from `models/legacy-sigmoid-seed43-epoch100.nn` (seed 43).
It allows **400 additional epochs**, batch size 32, patience 20 and min_delta
0.0001. To reuse it, retain that checkpoint and its referenced history, or edit
the protocol to select your own initial model.

```sh
python3 scripts/experiment.py configs/protocols/continue-reference.yml \
  --train-only --output results/training/2026-11-07-continue-reference-02
```

A single run is flat:

```text
README.md       Goal, status, checkpoint and conclusion
config.json     Resolved parameters, dataset/model hashes, commands and Git revision
model.nn        Selected checkpoint
history.csv     Per-epoch train and validation losses
history.svg     Loss curves with selected checkpoint and stopping epoch
training.log    Execution diagnostics and detailed analyzer log
```

The source model stays in place and is not modified or copied. Previous history
is referenced rather than copied. A fresh initialization is generated in a
temporary directory; its network settings, seed and fingerprint are recorded.
`--prepare` retains `initial-model.nn` instead and performs no training/evaluation.
Generation failures retain `generation.log`.

`--train-only` requires one configuration and one seed. Omit it to also evaluate
the initial and selected models on validation; this adds one `metrics.json`
containing `before` and `validation` scores and two corresponding
`confusion-matrix-*.svg` charts. Counts stay in `metrics.json`; no confusion-matrix
CSV is exported.

Continuation restores weights and biases, plus moments and update counter for
Adam checkpoints; shuffle randomness and early-stopping state restart.
`previous_history` and `previous_checkpoint_epoch` describe the earlier
trace, but the model format does not authenticate that relationship.

## Evaluate an existing model

Use validation while tuning. This command scores the delivered model without
training or modifying it:

```sh
python3 scripts/evaluate.py \
  my_torch_network.nn \
  dataset/references/chess-v1/splits/validation.txt \
  --output results/evaluation/YYYY-MM-DD-delivered-validation-01
```

The directory contains only:

```text
README.md               Model, dataset and readable scores
metrics.json            Detailed scores, fingerprints and inference timing
confusion-matrix.svg    Counts and row percentages: actual rows, predicted columns
```

Add `--predictions` only when individual predictions are needed for error
analysis. It exports `predictions.csv` with original line numbers and labels.
Undefined scores are zero; macro F1 includes Nothing, Check and Checkmate.
Color suffixes in analyzer output are removed before scoring and CSV export;
these metrics measure the three states, not color accuracy.

Only after final choices are frozen, use the test split for the final evaluation.
A historical reference was already evaluated on this test set; it is not a newly
unseen sample for later work. Keep its results out of tuning decisions.

## Compare configurations

Edit `configs/protocols/compare-network-sizes.yml`. It declares architectures,
seeds, prepared train/validation datasets and common training settings once.
The compact/reference pair is a starting comparison, not a tuned final choice.
Verify the [frozen dataset](datasets.md) before a campaign.

```sh
python3 scripts/experiment.py configs/protocols/compare-network-sizes.yml \
  --output results/experiments/2026-11-10-compare-network-sizes-01

# Quick technical check on tiny temporary datasets; automatically cleaned.
make smoke
```

Only campaigns with multiple runs add a level of folders:

```text
README.md
config.json
comparison.csv
runs/
  compact-seed-42/
  compact-seed-43/
  reference-seed-42/
  reference-seed-43/
```

Each run contains `model.nn`, `history.csv`, `history.svg`, `training.log` and
`metrics.json`, plus confusion-matrix charts when evaluated. When at least two
trials have completed with validation scores and training durations, the campaign
root also contains `comparison.svg`: per-trial macro F1 and training time. Failed
or missing trials are excluded; the campaign status stays visible on the chart.
The root README groups their results; `comparison.csv` gives before/after scores.
`config.json` records run states, costs and provenance. No separate summary,
command, training-config or decision files are needed. Write your conclusion in
the README's Conclusion section; refreshing the overview preserves that section.

Failures retain completed runs and diagnostics. Missing seeds remain visible;
a single seed is exploratory, with no estimate of variability. Timing includes
process startup/loading and is not an isolated benchmark.

## Charts and existing results

Chart exports are SVG files, viewable in a browser and embedded in result READMEs.
Python exports require the dependencies in `requirements.txt` (including matplotlib)
and run headlessly. Direct C++ `--train` exports an SVG beside its history CSV,
using the same basename, without requiring Python. An existing SVG is never
replaced by the C++ exporter; an export failure warns without losing the model.

The selected checkpoint and stopping epoch are different markers: training can
continue for the patience window after its best checkpoint. A checkpoint at epoch
zero means the initial weights were retained. Continuation charts offset epochs
by the declared previous checkpoint; they show the current training segment.
For historical runs without selection metadata, the marker explicitly says
“lowest recorded validation” rather than claiming it is the saved checkpoint.
Partial histories show the last recorded epoch, not a successful stopping point.

Regenerate missing charts from existing histories and metrics, without training,
prediction, or changes to model/configuration/numerical source files:

```sh
python3 scripts/charts.py results --recursive
```

This preserves written conclusions and refreshes the chart section of existing
READMEs. It replaces legacy confusion-matrix CSVs only after exporting their chart
from `metrics.json`. Campaign manifests still marked as active are skipped; check
their real state before updating a stale status. Missing histories or metrics
cannot be reconstructed from model weights. `--prepare` has no learning data to plot.

## Read results in notebooks

Use the project's `.venv` kernel with `notebooks/requirements.txt` installed.
Notebooks read saved results; they do not launch training or inference.

| Notebook | Select |
| --- | --- |
| `evaluation.ipynb` | `RUN`: a standalone evaluation or a training/experiment run containing metrics; `PHASE`: before or validation for grouped metrics |
| `comparison.ipynb` | `EXPERIMENT`: a directory containing `config.json`; or `RUNS`: standalone validation directories relative to the project |
| `dataset_analysis.ipynb` | Frozen dataset version; displays findings directly, without exporting files |

Training curves are also shown in the evaluation notebook when `history.csv`
is present. The comparison notebook recomputes statistics from the saved scores
and checks dataset/build compatibility and model integrity.

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

The built-in command applies every declared training setting and rejects unknown
keys. Checkpoint selection uses minimum validation loss, including the starting
model if no epoch improves it. Macro-F1 is measured afterwards, not used to
select an epoch. Completed comparisons are recorded in the [reports](report/README.md).
Those reports, the final selected model and its final evaluation must remain
distinct artifacts; a campaign summary does not promote a checkpoint.

## Test artifacts

`make smoke` selects the existing real generation/training/evaluation integration
test; `make test` includes it in the full suite. Both use temporary directories
that are removed automatically, including when a test fails. No keep-files option
or permanent `results/smoke-tests/` directory is used. Tests do not delete real
training results or models.

Install `requirements.txt` for the YAML protocol reader. Network files are YAML too; generated `config.json` and metrics remain JSON. YAML comments document
choices in editable protocols without adding extra metadata files.
