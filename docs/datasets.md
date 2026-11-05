# Chess datasets

This guide covers the input format, cleaning, and reproducible dataset splits.
Follow the preparation workflow: **raw data → cleaning → train/validation/test**.

## Contents

1. [FEN encoding and C++ dataset contract](#1-fen-encoding-and-c-dataset-contract)
2. [Dataset cleaning](#2-dataset-cleaning)
3. [Dataset splitting](#3-dataset-splitting)
4. [Frozen reference datasets](#4-frozen-reference-datasets)

---

## 1. FEN encoding and C++ dataset contract

`encodeFEN(fen)` converts a six-field FEN into a `std::vector<double>` of
**777 binary values** (`0.0` or `1.0`). The layout is fixed:

| Input | Count | Indices |
| --- | --- | --- |
| Piece placement | 64 × 12 = 768 | 0–767 |
| Side to move | 1 | 768 |
| En passant target file | 8 | 769–776 |

Castling rights and move counters are validated but not encoded. All indices
are zero-based.

### Piece placement

Squares follow FEN order: `a8 … h8`, `a7 … h7`, down to `a1 … h1`.
Within each rank, digits denote consecutive empty squares; `/` separates ranks.
Uppercase symbols represent white pieces, lowercase symbols black pieces.

Each square occupies 12 consecutive values. Set the slot for its piece to
`1.0`; leave all other slots at `0.0`. An empty square has 12 zeros.

| FEN symbol | P | N | B | R | Q | K | p | n | b | r | q | k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Piece index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| White rook example | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`P/N/B/R/Q/K` stand for pawn, knight, bishop, rook, queen, and king.

```text
input index = square index × 12 + piece index
```

For `R2k4` on rank 6, the white rook on `a6` activates index
`16 × 12 + 3 = 195`; the black king on `d6` activates `19 × 12 + 11 = 239`.

### Side to move and en passant

`inputs[768]` is `0.0` for `w` and `1.0` for `b`.

The en passant field specifies the capturing pawn’s destination square.
Encode its file as follows; `-` leaves all eight values at zero:

| File | a | b | c | d | e | f | g | h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Input index | 769 | 770 | 771 | 772 | 773 | 774 | 775 | 776 |
| Example: `d6` | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 |

The target rank must be **6 for white to move** or **3 for black to move**.
It is therefore recoverable from the side-to-move value. The encoder preserves
the declared target without checking whether a legal capture exists.

### Validation

The function requires exactly six fields and throws `std::invalid_argument`
with a diagnostic for malformed input.

| Field | Requirements |
| --- | --- |
| Board | Eight ranks of eight squares; recognized pieces and empty-square counts `1`–`8`; no consecutive counts |
| Side to move | `w` or `b` |
| Castling | `-` or distinct letters from `KQkq`: white kingside, white queenside, black kingside, black queenside |
| En passant | `-` or a file `a`–`h` with the rank required by the side to move |
| Halfmove clock | Decimal integer ≥ 0; plies since the last capture or pawn move |
| Fullmove number | Decimal integer ≥ 1; incremented after Black’s move |

Full position legality and check/checkmate detection are outside the encoder’s
scope.

### Dataset and model contract

A dataset reader must separate the six-field FEN from its training label
before calling `encodeFEN`. Labels never enter the input vector.
For example, `Check White` maps to `[0, 1, 0]`; see
[output encoding](output-encoding.md).

`loadDataset(path, DatasetMode::Prediction)` returns a `Dataset`, a vector of
`DatasetSample` objects in file order. Each sample holds `inputs` and an optional
`target`: a missing label is represented by `std::nullopt`, never by `Nothing`.
Prediction accepts both six-field FEN lines and labeled lines.
Use `DatasetMode::Training` to require a label for every position.

The loader skips blank lines and accepts whitespace-separated fields, including
Windows line endings. It rejects unknown labels, extra fields, and malformed
FENs with the file path and physical line number. Missing files, read errors,
and files containing no positions are also rejected.

Chess models require **777 inputs and 3 outputs**, ordered
`[Nothing, Check, Checkmate]`. Models with 768 inputs must be regenerated and
trained for this layout. Dimensions alone cannot verify feature or class order.
The `.nn` file stores model configuration and parameters, not input positions.

The generator remains generic. The analyzer checks that a loaded chess model has
777 inputs and 3 outputs before reading the dataset, in both prediction and
[training](training.md).

### Prediction command

Use a saved chess model and a text file containing one six-field FEN per line:

```text
8/8/8/8/8/8/8/8 w - - 0 1
8/8/8/8/8/8/8/8 b - - 0 1 Check
```

The optional label after a FEN is accepted but does not affect the prediction.
Blank lines are skipped. The encoder validates FEN fields, but does not check
whether a position is legal chess.

```sh
./my_torch_analyzer --predict models/basic_network_1.nn positions.txt
```

The command prints one class per nonblank line in the same order. It returns 0
on success, or 84 with a diagnostic on stderr for invalid arguments, models,
or datasets. A newly generated model is untrained; its class outputs have no
established predictive quality. See [output encoding](output-encoding.md) for
the class mapping.

---

## 2. Dataset cleaning

Run all cells in [the preparation notebook](../notebooks/prepare_dataset.ipynb).
Its cleaning stage calls [`scripts/clean_dataset.py`](../scripts/clean_dataset.py)
and shows a short summary by class and source file.

```text
dataset/raw/                  dataset/processed/
├── check/*.txt      ────►     ├── cleaned.txt
├── checkmate/*.txt             ├── provenance.jsonl
└── nothing/*.txt               └── report.json
```

The notebook reads all `.txt` files recursively from `dataset/raw/`. If that
folder is absent, the cleaner supports the older layout under `dataset/`.
It never reads `processed/`, `splits/`, or `references/` as sources. Source files
are left unchanged.

### Input and output

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

### Cleaning rules

1. **Validate.** The FEN and label must follow the same format as the
   [C++ dataset contract](#dataset-and-model-contract).
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

---

## 3. Dataset splitting

The splitting stage of [the preparation notebook](../notebooks/prepare_dataset.ipynb)
runs after [cleaning the data](#2-dataset-cleaning). It reads
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

### Settings

| Notebook setting | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | Makes the selection and order repeatable. |
| `ratios` | `(0.8, 0.1, 0.1)` | Train, validation, and test shares; must be positive and sum to 1. |
| `balance_training` | `False` | Optionally repeat minority-class rows in train only. |

Run all cells again after changing a setting. The same input, seed, ratios,
and balancing choice produce identical files. `split_info.json` records these
settings, the input file name and SHA-256 hash, and class counts before and
after balancing.

### How positions are assigned

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

---

## 4. Frozen reference datasets

`chess-v1` is the fixed dataset for MyTorch experiments. The existing cleaner and
splitter already produced reproducible data; this reference adds a retained
version, file inventory, integrity checks, and experiment paths that notebook
reruns cannot overwrite.

### Storage and manifest

- Tracked inventory: [`dataset/manifests/chess-v1.json`](../dataset/manifests/chess-v1.json).
  Paths are relative to the repository root; artifact paths are relative to its
  `storage` directory. It records SHA-256 hashes and byte sizes for every source,
  tool, and artifact; row counts and class counts for source/prepared FEN files;
  settings, provenance limitations, commands, and the preparation environment.
- Retained local files: `dataset/references/chess-v1/` (ignored by Git).
  Keep this directory for experiments and the defense, along with the tracked
  manifest and raw sources. Copy it when moving workspaces, or recreate it below.
- Disposable notebook outputs remain in `dataset/processed/` and `dataset/splits/`.
  Experiments no longer use those paths.

```text
dataset/references/chess-v1/
├── processed/
│   ├── cleaned.txt
│   ├── provenance.jsonl
│   └── report.json
└── splits/
    ├── train.txt
    ├── validation.txt
    ├── test.txt
    └── split_info.json
```

### Preparation report: chess-v1

All ten `.txt` files under `dataset/raw/` were selected, including
`check/10_pieces_SHORT.txt`. Their exact paths, hashes, per-source class counts,
and invalid-row counts are in the manifest. Sources are preserved unchanged.

| Stage | Records | Nothing | Check | Checkmate |
| --- | ---: | ---: | ---: | ---: |
| Raw (including one invalid row) | 723,025 | 300,000 | 249,305 | 173,719 |
| Cleaned | 715,555 | 297,563 | 248,133 | 169,859 |
| Train | 572,445 | 238,051 | 198,507 | 135,887 |
| Validation | 71,555 | 29,756 | 24,813 | 16,986 |
| Test | 71,555 | 29,756 | 24,813 | 16,986 |

Cleaning removes **7,469 duplicate rows** and **one malformed row**, with **zero
conflicting records**. The malformed row is line 1 of
`raw/check/10_pieces_SHORT.txt`. See the retained `processed/report.json` for
details and `provenance.jsonl` for every cleaned row's source locations.

The [cleaning rules](#cleaning-rules) validate six-field FENs
and labels, normalize color suffixes, retain the first duplicate, and exclude
all rows in conflicting groups. Source paths are traversed in sorted order.
Model inputs are identified by board, side to move, and en passant target;
castling rights and move counters do not distinguish inputs.

Splitting uses **seed 42**, **80% / 10% / 10%**, class stratification, and **no
balancing**. All cleaned positions occur exactly once across the three splits,
with **zero shared model-visible inputs**. Counts include rounding per class.
These files reproduce the existing notebook outputs; no split algorithm changed.

The external source, generator, license, and usage grant are **unknown**. The
repository filenames and Git history do not establish them. Local retention for
project experiments and the defense does not establish redistribution rights.
FEN syntax and label format are checked; chess legality and label correctness
have not been audited. These limitations are also recorded in the manifest.

### Commands and reproduction

Run from the repository root. Preparation and verification need only the Python
standard library. The recorded environment is **CPython 3.12.3**. Recreation
warns when the Python version/implementation or preparation tool hashes differ,
but accepts the result only if all artifact hashes and integrity checks match.
The manifest also records the base Git revision and each tool's exact SHA-256,
which identifies the preparation code even if it was uncommitted at that revision.
The platform string is informational.

Initial creation, already performed for this reference:

```sh
python3 -m scripts.reference_dataset prepare chess-v1
```

Verify the retained reference without modifying any file:

```sh
python3 -m scripts.reference_dataset verify
```

Verification exits nonzero on missing/modified sources or artifacts, mismatched
counts or metadata, duplicate/overlapping inputs, changed labels, incomplete
split coverage, or inconsistent provenance row counts. It compares the current
files with the tracked manifest; it does not rewrite a changed inventory.

On a fresh checkout, restore the ignored reference from the **recorded sources**:

```sh
python3 -m scripts.reference_dataset recreate
python3 -m scripts.reference_dataset verify
```

To independently reproduce it while the retained reference already exists:

```sh
python3 -m scripts.reference_dataset recreate --output /tmp/mytorch-chess-v1-recreated
python3 -m scripts.reference_dataset verify --storage /tmp/mytorch-chess-v1-recreated
```

The output directory must be new. Recreation stages only the selected sources,
runs the existing cleaner and splitter, and checks **all seven artifact hashes**
and integrity constraints before installing the generated files. Later additions
to `dataset/raw/` are ignored for this reference. The manifest remains unchanged.
Small-fixture tests exercise reproduction and verification failures in `make test`.

To intentionally change sources, cleaning rules, or settings, create a new
reference version and manifest, document it, then explicitly update experiment
paths. `prepare` refuses an existing reference name and `recreate` refuses an
existing output directory. Never regenerate a manifest to conceal a changed file.

### Experiment paths

[`configs/experiment.json`](../configs/experiment.json) resolves train, validation,
and metadata to `dataset/references/chess-v1/splits/`. The standalone comparison
notebook defaults to this validation file too. Tests check the supplied config
against a small frozen reference; the real paths and hashes were also checked
against this manifest.

Train only on `train.txt`, choose settings/models on `validation.txt`, and reserve
`test.txt` for the final selected model. The experiment config does not include a
test dataset. After verification, the usual preparation command is:

```sh
make experiment OUTPUT=results/experiments/chess-v1-preparation OPTIONS=--prepare
```

The runner captures the resolved dataset paths and hashes in its experiment
manifest and detects changes during a campaign. This issue adds no training,
model selection, or final test evaluation.
