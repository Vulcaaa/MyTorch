# Left-right mirrors, dropout and mirror views: the delivered model

**The delivered `my_torch_network.nn` reaches 93.66% accuracy and 93.15% macro F1
on the reserved test split.** It replaces the augmented sigmoid model (88.06%
validation accuracy). It is one 777 → 256 → 128 → 3 network: 232,451 parameters,
2.4 MB, 1.2 seconds for 71,555 predictions. The current analyzer loads it
without code changes.

Combining mirror views and networks reaches up to 95.06% test accuracy, at 12 to
34 times the computation. These variants are documented below as alternatives.

## Method

1. **Left-right mirrors.** Reflecting a position across the board's vertical
   axis (a-file ↔ h-file) keeps Check and Checkmate unchanged: movement rules are
   symmetric between files, and castling is never a legal reply to check.
   [`mirror_files.py`](../../scripts/mirror_files.py) adds these mirrors to the
   color-mirror train: 2,275,774 positions
   ([dataset guide](../datasets.md#7-left-right-mirror-augmentation)).
2. **Training.** 777 → 256 → 128 → 3, ReLU/softmax, Xavier, Adam learning rate
   0.001, batch 128, seed 49. During training, dropout sets each hidden activation
   to zero with probability 0.1 (and scales the others by 1/0.9); prediction uses
   every neuron. After five epochs without validation improvement, training
   restarts from the best checkpoint with the learning rate multiplied by 0.3, at
   most three times. Epoch 47 of 57 was selected (validation loss 0.1620). The run
   took 86 minutes on one thread of a loaded M5. Weights are rounded to seven
   significant digits; validation accuracy is unchanged.
3. **Mirror views (alternatives).** A network can also read the left-right
   mirror, the color mirror and both mirrors of a position. A view only permutes
   first-layer input columns, so it costs no training.
   [`merge_networks.py`](../../scripts/merge_networks.py) places views and networks
   side by side, with block-diagonal hidden weights, and averages their logits.

## Results

| Variant (validation accuracy) | Network | Operations per position | Accuracy |
| --- | --- | ---: | ---: |
| Previous delivered sigmoid | 777 → 128 → 64 → 3 | 108k | 88.06% |
| Previous shared candidate: ReLU + Adam, color-mirror train | 777 → 256 → 128 → 3 | 232k | 89.25% |
| ×4 train, learning-rate decay, seed 46 | 777 → 256 → 128 → 3 | 232k | 92.89% |
| **×4 train, decay, dropout 0.1, seed 49 (delivered)** | 777 → 256 → 128 → 3 | **232k** | **93.78%** |
| Same network, two views (original + color) | 777 → 512 → 256 → 3 | 530k | 94.64% |
| Same network, four views | 777 → 1024 → 512 → 3 | 1.32M | 95.05% |
| Seeds 46 and 48 without dropout, four views each | 777 → 2048 → 1024 → 3 | 3.69M | 95.20% |

Operations count every multiply-add of a prediction, including zero blocks.

| Delivered model | Accuracy | Macro F1 | Nothing F1 | Check F1 | Checkmate F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation (71,555) | 93.78% | 93.27% | 97.47% | 91.34% | 91.00% |
| **Test (71,555)** | **93.66%** | **93.15%** | 97.37% | 91.17% | 90.90% |

Test confusion matrix (rows actual, columns predicted):

| | Nothing | Check | Checkmate |
| --- | ---: | ---: | ---: |
| Nothing | 28,710 | 898 | 148 |
| Check | 460 | 22,423 | 1,930 |
| Checkmate | 46 | 1,055 | 15,885 |

The two heavier alternatives were also evaluated on test: 94.95% (one network,
four views) and 95.06% (two networks, four views). The delivered variant was
chosen for its size, not for its test score.

## Interpretation

- **The left-right mirror teaches rules rather than positions.** A network trained
  on the color-mirror train alone scores 81% on left-right mirrored validation
  positions, against 89% on the originals. It had learned file-specific patterns.
  With the ×4 train, the same 256/128 network gains 3.6 points.
- **Dropout reduces overfitting.** At the same size (another seed), validation loss falls from
  0.197 to 0.162 and accuracy rises by 0.9 points.
- **Views disagree, so their average corrects errors.** Each view of the same
  network makes partly different mistakes; four views add 1.3 points at 5.7 times
  the computation.
- **Remaining errors are mostly Check ↔ Checkmate** (2,985 of 4,537 test errors),
  as in the [label and error audit](2026-11-08-label-and-error-audit.md).

## Limitations

- Validation chose the learning rate, the checkpoint and the variant, so its
  scores are optimistic; the single test evaluation of the delivered model
  confirms them (−0.12 points).
- Dropout and the learning-rate decay ran from a local experimental build; they
  are not yet training options on `main`.

## Evidence

- [Portable metrics, confusion matrices, hashes and training histories](data/2026-11-08-mirror-view-ensemble.json)
- Rebuild the ×4 train: `python3 scripts/mirror_files.py TRAIN OUTPUT --heldout VALIDATION TEST`.
- Rebuild an alternative: `python3 scripts/merge_networks.py OUTPUT MODEL... --mirror-views`
  (or `--views original,color`).
