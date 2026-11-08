# Label and error audit: what limits accuracy

## Purpose and method

Validation macro F1 has stayed below 89% across architectures, learning rates,
activations and augmentation. This audit asks whether the limit comes from the
data or from the model, and which positions the model gets wrong.

- **Labels:** every train and validation position is checked against the chess
  rules with python-chess 1.11.2, used as an oracle. A position is Checkmate if the
  side to move is checkmated, Check if it is in check, and Nothing otherwise.
- **Model:** the best checkpoint so far, ReLU + Xavier, LR 0.16, seed 46, trained on
  the color-mirror train ([ReLU report](2026-11-08-relu-color-mirror.md)). It is
  evaluated on its own training data and on validation. Validation errors are then
  grouped by chess features computed with the oracle. The test split is not used.

## Results

**Labels match the rules exactly.**

| Split | Positions | Labels matching the rules | Not valid per python-chess | Stalemates labeled Nothing |
| --- | ---: | ---: | ---: | ---: |
| Train | 572,445 | 572,445 (100%) | 9 | 0 |
| Validation | 71,555 | 71,555 (100%) | 2 | 0 |

The 777 inputs also contain everything that decides the label: pieces, side to move
and the en passant file. Castling rights are not encoded, but castling is never a
legal reply to check, so they cannot change Check or Checkmate. **A perfect
classifier is possible from these inputs.**

**The model does not fit its own training data.**

| Data | Accuracy | Macro F1 | Recall Nothing | Recall Check | Recall Checkmate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Color-mirror train (1,141,143 positions) | 92.66% | 92.11% | 96.34% | 88.33% | 92.54% |
| Validation | 89.11% | 88.47% | 93.92% | 83.69% | 88.62% |

The model misclassifies 7.3% of the positions it was trained on. The main limit is
what this model can represent, not overfitting: the train–validation gap is
3.5 points.

**Validation errors by confusion (7,791 errors)**

| Confusion | Errors | Share |
| --- | ---: | ---: |
| Check ↔ Checkmate | 4,367 (2,603 + 1,764) | 56% |
| Nothing ↔ Check | 2,984 (1,539 + 1,445) | 38% |
| Nothing ↔ Checkmate | 440 (271 + 169) | 6% |

**Error rate by position pattern (validation)**

| Pattern | Errors / positions | Error rate |
| --- | ---: | ---: |
| Nothing, enemy bishop/rook/queen on a line to the king with exactly one piece between | 992 / 3,756 | **26.4%** |
| Nothing, no such line | 818 / 26,000 | 3.1% |
| Check with 1–2 legal replies | 2,696 / 11,012 | **24.5%** |
| Check with 3–5 legal replies | 1,103 / 11,532 | 9.6% |
| Check given by a knight | 399 / 3,751 | 10.6% |
| Check given by a bishop / rook / queen / pawn | 493 / 2,397 · 1,288 / 7,094 · 1,608 / 10,167 · 246 / 1,275 | 20.6% · 18.2% · 15.8% · 19.3% |
| Checkmate given by a queen | 786 / 9,964 | 7.9% |
| Checkmate given by a bishop / knight / pawn | 247 / 868 · 200 / 799 · 131 / 493 | 28.5% · 25.0% · 26.6% |

## Interpretation

The two dominant errors match two operations that one-hot square inputs make hard
for a small dense network:

- **Line of sight.** Deciding Nothing or Check often needs "every square between
  this rook and the king is empty". The model fails eight times more often when
  one piece blocks an enemy line to the king, and less often on knight checks,
  which never depend on blocking.
- **Escape analysis.** Deciding Check or Checkmate needs to know whether any legal
  reply exists. Errors concentrate on checks with only one or two replies, which
  are closest to mate.
- **Rare patterns.** Mates by a bishop, knight or pawn have about 500–900
  validation examples each, against about 10,000 queen mates, and are missed
  three times more often.

## Leads to investigate

| Lead | Targets | Expected effect | Cost | Fit with the subject |
| --- | --- | --- | --- | --- |
| Larger or deeper ReLU network | Underfitting (92.7% on train) | Moderate; the earlier "large" test used sigmoid, which learns too slowly to show it | Protocol only | Pure learning |
| Left–right mirror augmentation | Data, rare patterns | Moderate; the color mirror gave +4.9 points | Small; verify labels with the oracle | Pure learning; mirroring files keeps Check and Checkmate unchanged |
| Learning-rate decay at the plateau | Final convergence | Small | Training option | Pure learning |
| Attacked-square maps added to the input: 64 squares attacked by White and 64 by Black | Line of sight, escape squares | Largest expected effect; to measure against the 777-input baseline | About a day: encoder, tests against the oracle, new input size, new model | **To confirm with the teaching team.** The network still learns Check and Checkmate, but the input then encodes piece-movement rules. Never input "in check" or legal-move counts |

## Evidence

- [Portable measurements: label/rule counts, train and validation fit, confusion counts and pattern error rates](data/2026-11-08-label-and-error-audit.json)
- The audit scripts and the oracle install stay local, as for earlier one-off
  reports. The oracle is installed as in the
  [color-mirror report](2026-11-08-color-mirror-augmentation.md#reproduction-and-evidence).
