# Dataset splitting

Run all cells in [the split notebook](../notebooks/split_dataset.ipynb) after
[cleaning the data](dataset-cleaning.md). It reads
`dataset/processed/cleaned.txt` and writes:

```text
dataset/splits/
├── train.txt
├── validation.txt
├── test.txt
└── split_info.json
```

Each `.txt` file uses the same six-field FEN and label format as the cleaned
input, so it can be read by the C++ dataset loader.

## Settings

| Notebook setting | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | Makes the selection and order repeatable. |
| `ratios` | `(0.8, 0.1, 0.1)` | Train, validation, and test shares; must be positive and sum to 1. |
| `balance_training` | `False` | Optionally repeat minority-class rows in train only. |

Run all cells again after changing a setting. The same input, seed, ratios,
and balancing choice produce identical files. `split_info.json` records these
settings, the input file name and SHA-256 hash, and class counts before and
after balancing.

## How positions are assigned

The splitter groups rows by the **inputs visible to the model**: board, side
to move, and en passant target. Rows in one group stay in the same split.
Conflicting labels for one group cause an error.

Within each class, groups are shuffled with the recorded seed and distributed
according to the ratios. Thus each split keeps approximately the original
class proportions. This is **stratification**, not class balancing. Rounding
can cause small differences. Each class needs at least three distinct groups
so it can appear in all three files; otherwise the notebook reports why it
cannot split the input.

The splitter checks that every group was assigned and that no model input
appears in more than one split. The check runs before any output is written.

Optional balancing repeats training rows from smaller classes until all three
training classes have the same number of rows. Repetitions stay inside train.
Validation and test retain their original rows and order, even when the
balancing option changes. Leave balancing off unless training results show a
need for it.

Use **train** to fit weights, **validation** to choose settings and a model,
and **test** once for final evaluation.
