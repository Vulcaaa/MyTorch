# Mirror-view ensemble: the delivered model

**The delivered `my_torch_network.nn` reaches 95.06% accuracy and 94.58% macro F1
on the reserved test split**, evaluated once. It replaces the augmented sigmoid
model (88.06% validation accuracy). It is an ordinary MyTorch network that the
current analyzer loads without code changes.

## Method

1. **Left-right mirrors.** Reflecting a position across the board's vertical
   axis (a-file ↔ h-file) keeps Check and Checkmate unchanged: movement rules are
   symmetric between files, and castling is never a legal reply to check.
   [`mirror_files.py`](../../scripts/mirror_files.py) adds these mirrors to the
   color-mirror train. The result has 2,275,774 positions: 1,134,631 mirrors
   added, 5,630 duplicates skipped, 882 skipped because their inputs occur in
   validation or test. python-chess 1.11.2 agrees with every added label.
2. **Training.** Two networks, 777 → 256 → 128 → 3, ReLU/softmax, Xavier, Adam
   learning rate 0.001, batch 128, seeds 46 and 48. After three epochs without
   validation improvement, training restarts from the best checkpoint with the
   learning rate multiplied by 0.3, at most three times. Both selected epoch 22
   of 28.
3. **Mirror views.** Each network also reads the left-right mirror, the color
   mirror and both mirrors of the position. A view only permutes first-layer input
   columns, so it is exact and costs no training.
4. **Merge.** [`merge_networks.py`](../../scripts/merge_networks.py) places the
   eight views side by side, with block-diagonal hidden weights, and averages their
   logits: 777 → 2048 → 1024 → 3, weights rounded to seven significant digits
   (27 MB).

## Results

| Step (validation accuracy) | Accuracy |
| --- | ---: |
| Previous shared candidate: 256/128 ReLU + Adam, color-mirror train | 89.25% |
| Same network on the ×4 train with learning-rate decay, seed 46 | 92.89% |
| Same network, four views | 94.91% |
| **Two networks, four views each (delivered)** | **95.20%** |

| Split | Accuracy | Macro F1 | Nothing F1 | Check F1 | Checkmate F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation (71,555) | 95.20% | 94.73% | 98.40% | 93.26% | 92.52% |
| **Test (71,555), once** | **95.06%** | **94.58%** | 98.32% | 93.08% | 92.33% |

Test confusion matrix (rows actual, columns predicted):

| | Nothing | Check | Checkmate |
| --- | ---: | ---: | ---: |
| Nothing | 29,007 | 659 | 90 |
| Check | 231 | 23,086 | 1,496 |
| Checkmate | 12 | 1,047 | 15,927 |

### Size against accuracy

Fewer members or views give lighter networks. Validation accuracy; operations
count every multiply-add of a prediction, including the zero blocks.

| Variant | Network | Operations per position | Validation accuracy |
| --- | --- | ---: | ---: |
| Previous delivered sigmoid | 777 → 128 → 64 → 3 | 108k | 88.06% |
| One member, one view | 777 → 256 → 128 → 3 | 232k | 92.89% |
| One member, two views (original + color) | 777 → 512 → 256 → 3 | 530k | 94.24% |
| One member, four views | 777 → 1024 → 512 → 3 | 1.32M | 94.91% |
| **Two members, four views (delivered)** | 777 → 2048 → 1024 → 3 | 3.69M | **95.20%** |

The delivered variant keeps the best accuracy; one member with four views is a
lighter alternative at −0.29 points.

## Interpretation

- **The left-right mirror teaches rules rather than positions.** A network trained
  on the color-mirror train alone scores 81% on left-right mirrored validation
  positions, against 89% on the originals. It had learned file-specific
  patterns.
- **Views disagree, so their average corrects errors.** Even after training on all
  four symmetries, each view of the same network makes partly different mistakes;
  averaging four views adds about two points.
- **Remaining errors are Check ↔ Checkmate** (2,543 of 3,535 test errors), as in
  the [label and error audit](2026-11-08-label-and-error-audit.md).

## Limitations

- Validation chose the learning rate, the checkpoints and the members, so its
  scores are optimistic; the test split, used once, confirms them (−0.14 points).
- The learning-rate decay was run from a local experimental build; it is not yet
  a training option on `main`.
- The merged network is about 20× larger than one member: on a loaded M5, 71,555
  predictions took about 90 seconds.

## Evidence

- [Portable metrics, confusion matrices, hashes and member histories](data/2026-11-08-mirror-view-ensemble.json)
- Rebuild: `python3 scripts/mirror_files.py TRAIN OUTPUT --heldout VALIDATION TEST`,
  then `python3 scripts/merge_networks.py my_torch_network.nn MEMBER_46 MEMBER_48 --mirror-views`.
