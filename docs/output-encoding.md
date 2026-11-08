# Output encoding

Each position belongs to one of three classes. The output order must remain
identical in dataset loading, training, and prediction:

| Index | Class | Meaning | Training target |
| --- | --- | --- | --- |
| 0 | `Nothing` | Neither check nor checkmate | `[1, 0, 0]` |
| 1 | `Check` | Check without checkmate | `[0, 1, 0]` |
| 2 | `Checkmate` | Checkmate | `[0, 0, 1]` |

Classes are mutually exclusive: checkmate uses only `Checkmate`.
Stalemate has no separate class in this contract.

## Dataset labels

Source datasets remain unchanged. The reader maps labels as follows:

| Dataset label | Class |
| --- | --- |
| `Nothing` | `Nothing` |
| `Check`, `Check White`, `Check Black` | `Check` |
| `Checkmate`, `Checkmate White`, `Checkmate Black` | `Checkmate` |

Color suffixes are ignored. Unknown labels must raise an error rather than
default to `Nothing`.

Example from `dataset/raw/check/10_pieces.txt`:

```text
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White

FEN:    8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59
Target: [0, 1, 0]
```

The target is used to compare the prediction with the expected result during
training. It is never included in the network inputs.

## Prediction

The example configuration uses `softmax`, producing three scores in `[0, 1]`
that sum to one within numerical precision.

```text
Class order: [Nothing, Check, Checkmate]
Scores:      [0.05,    0.90,  0.05]  → Check
```

Select the largest score (`argmax`), without a `0.5` threshold. Break exact ties
by choosing the first index. A score of `0.90` is not a guarantee of 90% accuracy;
model performance must be measured separately. Prediction requires no target.

## Model contract

The final layer must have exactly three neurons. Training targets, saved models,
and predictions must use the same class order. The analyzer displays `Nothing`,
or appends `White` / `Black` to `Check` and `Checkmate` by default.
The suffix identifies the attacking side: `White` when the FEN says Black moves
next (`b`), and `Black` when White moves next (`w`). For legal positions in check,
the attacker is the opponent of the player who must move. The network still
predicts only the three states; color is derived from FEN input 768, independently
of any expected label. Existing three-output models need no retraining.

The evaluation tools accept both colored and legacy predictions and measure the
three states, removing the suffix before computing metrics and exporting CSV.

Class mapping belongs to the chess module; the neural network operates on
numeric vectors. See [data encoding](datasets.md#dataset-and-model-contract)
for input dimensions and model compatibility requirements.
