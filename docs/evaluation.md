# Model evaluation

## Run

Requires Python 3 and `make`; no Python packages are needed.

```sh
make
python3 scripts/evaluate.py MODEL DATASET --output results/RUN
```

Use a saved `.nn` model, a labeled [FEN dataset](data-encoding.md), and a new
output directory for each run. The script calls `my_torch_analyzer --predict`
and compares predictions with labels in input order. Invalid input, analyzer
failure, or mismatched prediction counts stop evaluation without a report.

## Read the results

| File | Contents |
| --- | --- |
| `metrics.json` | Accuracy, macro F1, precision/recall/F1 and sample count per class; command, file paths, SHA-256 hashes, and elapsed prediction time. |
| `predictions.csv` | Dataset line number, actual class, predicted class. |
| `confusion_matrix.csv` | Actual classes in rows, predictions in columns: `Nothing`, `Check`, `Checkmate`. |

Check per-class recall and F1 alongside accuracy to expose missed classes.
Off-diagonal matrix cells show which classes are confused. Undefined metrics
are zero; macro F1 averages all three classes, including absent ones.
Elapsed time includes analyzer startup, loading, parsing, and prediction.

For charts, open [the evaluation notebook](../notebooks/evaluation.ipynb), select
`RUN`, and run all cells. Install its dependencies with
`python3 -m pip install -r notebooks/requirements.txt` in your virtual environment.
It reads saved results without rerunning predictions.

## Comparison protocol v1

| Rule | Decision |
| --- | --- |
| Data | Train on `train.txt`; compare on the unchanged `validation.txt`. Use `test.txt` only after selection. Retain split metadata and sources. |
| Compatibility | Same validation SHA-256, analyzer SHA-256, class order, and class supports. All three classes must be represented. |
| Primary metric | Macro F1, giving each class equal weight. Inspect per-class F1/recall and confusion counts as well. |
| Run ordering | Highest macro F1 first; exact ties use highest worst-class recall. Equal scores remain tied regardless of display order. |
| Training seeds | 42, 43, 44 for every configuration; retain each run, including poor results. |
| Configuration choice | Compare mean macro F1 and sample standard deviation across all three seeds. Exact mean ties use mean worst-class recall. No choice from incomplete seed sets. |
| Timing | Same machine/build and no concurrent heavy work. Current elapsed time is a single measurement including loading; do not use it to break score ties. |
| Traceability | Retain training command, configuration, seed, train/validation hashes, model, and logs alongside each evaluation. |

In the notebook, fill `RUNS` with validation run directory names. The table and
per-class charts compare individual runs; compatibility checks reject mixed data
or builds. Seed aggregation is not implemented yet. `RUN1` is a test-set pipeline
check with an untrained model, so exclude it from tuning.

**Before training comparisons:** agree with Gustave on the small configuration
grid, optimizer, training budget, and checkpoint rule. Those depend on the training
interface and are not fixed here. Keep these settings common except for the
parameter being compared, and freeze them before inspecting validation results.

Generated models are untrained: evaluating them checks the pipeline only.
Training comparisons, seed aggregation, curves, and final model delivery remain;
the evaluation script does not orchestrate training or record its configuration.
