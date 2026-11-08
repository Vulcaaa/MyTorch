# Reference network: checkpoints at epochs 100 and 338

Date: 2026-11-07. First report, based on existing artifacts; no new training or
evaluation was launched for this report.

## Finding

Keep the epoch-338 checkpoint as the improved reference. On the same validation
split and analyzer build, continuing training raises accuracy from **72.73% to
82.57%** (+9.84 percentage points) and macro F1 from **72.28% to 81.96%** (+9.69
points). Recall improves for all three classes. These are two checkpoints of
the **same network**, not independent architectures or seed repetitions.

## Network, data and training

Both checkpoints use layers `777 → 128 → 64 → 3`, sigmoid hidden activations,
softmax output, Xavier initialization, learning rate 0.01 and network seed 43.

The continuation records the frozen `chess-v1` split: 572,445 training positions,
71,555 validation positions and 71,555 test positions. Split seed is 42, ratios
are 80/10/10, training is not balanced, and split metadata records zero overlapping
inputs. Validation supports are 29,756 Nothing, 24,813 Check and 16,986 Checkmate.
The original training protocol was not fully recorded; its missing settings and
original training-data provenance cannot be reconstructed from its metadata.

| Item | Initial checkpoint | Continued checkpoint |
| --- | --- | --- |
| Training date | 2026-11-05 | 2026-11-07 |
| Selected total epoch | 100 | 338 (100 + 238) |
| Epochs actually completed | 100 | 358 total (100 + 258) |
| Additional budget | — | 400 epochs; maximum 500 total |
| Batch size | Not recorded | 32 |
| Early stopping | Not fully recorded | Patience 20; min_delta 0.0001 |
| Selection rule | Historical checkpoint at epoch 100 | Minimum validation loss |
| Validation loss at selected epoch | 0.594914 | 0.432493 |
| Recorded training duration | Not recorded | 4,120.75 s (68.68 min), continuation only |

Continuation restores weights and restarts shuffle randomness and early-stopping
state. It is not guaranteed equivalent to an uninterrupted 358-epoch run. The
saved model is epoch 338, not the last epoch or the maximum budget of 500.

Sources: [initial training metadata](../../../results/training/2026-11-05-reference-training-01/config.json),
[continuation metadata](../../../results/training/2026-11-07-continue-reference-01/config.json),
[initial history](../../../results/training/2026-11-05-reference-training-01/history.csv),
[continuation history](../../../results/training/2026-11-07-continue-reference-01/history.csv).
Curves can be inspected in [the evaluation notebook](../../../notebooks/evaluation.ipynb).

## Paired validation comparison

The continuation's `before` and `validation` metrics use the same validation
dataset fingerprint and the same analyzer fingerprint. This paired evaluation
is the source for the comparison below.

| Metric | Epoch 100 | Epoch 338 | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Accuracy | 72.73% | 82.57% | +9.84 |
| Macro F1 | 72.28% | 81.96% | +9.69 |
| Nothing recall | 81.57% | 88.81% | +7.24 |
| Check recall | 61.10% | 76.18% | +15.08 |
| Checkmate recall | 74.26% | 80.98% | +6.72 |
| Nothing F1 | 79.79% | 88.73% | +8.95 |
| Check F1 | 62.78% | 76.41% | +13.63 |
| Checkmate F1 | 74.27% | 80.76% | +6.49 |

Changes are calculated before rounding. Check benefits most but remains the
weakest class. At epoch 338, 2,979 actual Check positions are predicted Nothing
and 2,931 are predicted Checkmate; 2,828 actual Checkmate positions are predicted
Check. These confusions provide concrete targets for error analysis.

Source: [paired validation metrics and confusion matrices](../../../results/training/2026-11-07-continue-reference-01/metrics.json).

## Separate test evaluation

Only the continued checkpoint has a recorded test evaluation in the inspected
results. Test scores must not be compared against epoch-100 validation scores
as an estimate of improvement.

| Metric | Epoch 338, test |
| --- | ---: |
| Accuracy | 82.13% |
| Macro F1 | 81.49% |
| Nothing recall | 88.56% |
| Check recall | 75.12% |
| Checkmate recall | 81.11% |

Test macro F1 is 0.47 percentage points below validation. This is consistent
with similar performance on these two splits; it does not establish robustness
across seeds, datasets or positions from a different source. The test set was
not used for checkpoint selection, according to the recorded workflow.

Source: [test evaluation](../../../results/evaluation/2026-11-07-reference-test-01/README.md)
and [test metrics](../../../results/evaluation/2026-11-07-reference-test-01/metrics.json).

## Limits and next experiment

This single-seed continuation supports keeping the later checkpoint, but does
not isolate training duration from historical code changes: the original
training protocol is incomplete and the continuation records a dirty working
tree. It cannot establish the best architecture or seed variability. Recorded
prediction times include loading/startup and are not a controlled speed benchmark.

The follow-up [three-model comparison](../2026-11-07-three-models/README.md) evaluates
compact, reference and large checkpoints with one frozen analyzer. It remains
a single-seed comparison with unequal training histories; it does not implement
the previously proposed multi-seed architecture campaign. That report contains
the updated decision and next experiment.

The current test score is already known, so it is a reference rather than a
fresh blind holdout for subsequent experimentation.

## Checkpoints

- [Epoch 100 model](../../../models/legacy-sigmoid-seed43-epoch100.nn)
- [Epoch 338 model](../../../results/training/2026-11-07-continue-reference-01/model.nn)
- [Result index](../../../results/README.md)
