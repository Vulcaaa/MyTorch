# Dataset cleaning

Run all cells in [the cleaning notebook](../notebooks/clean_dataset.ipynb).
It calls [`scripts/clean_dataset.py`](../scripts/clean_dataset.py) and shows a
short summary by class and source file.

```text
dataset/raw/                  dataset/processed/
├── check/*.txt      ────►     ├── cleaned.txt
├── checkmate/*.txt             ├── provenance.jsonl
└── nothing/*.txt               └── report.json
```

The notebook reads all `.txt` files recursively from `dataset/raw/`. If that
folder is absent, the cleaner supports the older layout under `dataset/`.
It never reads `processed/` or `splits/` as sources. Source files are left
unchanged.

## Input and output

Each nonblank input line needs a **six-field FEN followed by a label**:

```text
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
```

The accepted labels are `Nothing`, `Check`, `Check White`, `Check Black`,
`Checkmate`, `Checkmate White`, and `Checkmate Black`. The color suffix is
removed in `cleaned.txt`, leaving one of `Nothing`, `Check`, or `Checkmate`.
The file is ready for the C++ dataset loader in training mode.

| Output | Contains |
| --- | --- |
| `cleaned.txt` | One retained position per line, with its normalized label. |
| `provenance.jsonl` | Source references for each output line, in the same order. |
| `report.json` | Counts and details of rejected rows and conflicts. |

Each provenance line is a JSON array with every source file and physical line
number for that position. Paths are relative to `dataset/`. Blank lines are
ignored; all other lines count toward `records`.

## Cleaning rules

1. **Validate.** The FEN and label must follow the same format as the
   [C++ dataset contract](data-encoding.md#dataset-and-model-contract).
   Invalid rows are excluded, with their file, line, and reason in `report.json`.
2. **Group by model input.** Two rows are the same position when their board,
   side to move, and en passant target match. The model does not receive
   castling rights or move counters, so differences in those fields do not
   make separate inputs. All six FEN fields are still validated.
3. **Remove duplicates.** If a group has one class, keep its first valid row.
   Files are read in sorted path order and then line order, so the choice is
   repeatable. `duplicates` counts the other rows in retained groups.
4. **Exclude conflicts.** If one group has different classes, exclude **all**
   its rows. `conflicts` lists each source file, line, and label; no label is
   chosen silently. `conflicting_records` counts those excluded rows.

For example, the same model input labeled `Check White` in one file and
`Checkmate Black` in another is a conflict. The cleaner does not decide which
label is correct.

`by_source.retained` counts distinct output positions that came from each
source. A position present in two files contributes to both source counts, but
appears only once in `cleaned.txt`. Class counts describe the final output.

The cleaner checks the FEN **format**, not whether a chess position or its
label is legally correct. It does not balance classes or create
train/validation/test splits.
