# Chess datasets

This guide covers the input format, reproducible splits and training-only
enrichment. The workflow is **raw data → cleaning → frozen train/validation/test
→ optional train augmentation**. Validation and test assignments stay fixed.

## Contents

1. [FEN encoding and C++ dataset contract](#1-fen-encoding-and-c-dataset-contract)
2. [Dataset cleaning](#2-dataset-cleaning)
3. [Dataset splitting](#3-dataset-splitting)
4. [Frozen reference datasets](#4-frozen-reference-datasets) and [color mirrors](#color-mirror-augmentation)
5. [Analyze balance and diversity](#5-analyze-balance-and-diversity)
6. [Generated game data](#6-generated-game-data)
7. [Left-right mirror augmentation](#7-left-right-mirror-augmentation)
8. [Data rights and predictive limitations](#8-data-rights-and-predictive-limitations)

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

This one-hot layout avoids assigning an artificial numeric order to piece types.
Side to move identifies the king whose status is classified; en passant can
affect the legal replies to a check. For legal positions, castling cannot escape
check, so castling rights do not change these three labels. Move counters are
irrelevant to this classification, which has no draw class. This compact
contract preserves the relevant board information, but a dense network must
still learn interactions such as blocked attack lines and escape squares.
The [label/error audit](report/2026-11-08-label-and-error-audit/README.md) investigates
these limits; full chess legality is not enforced by the encoder.

This is a justified baseline, not a demonstrated optimal encoding. Adding an
explicit empty-square channel would raise placement inputs from 768 to 832
without adding information. Integer piece IDs would be smaller but introduce
an artificial numeric ordering. Attack maps or spatial architectures might
help model interactions, but have no controlled quality benchmark here. Changing
the contract would require consistent preprocessing/deduplication changes and
fresh training; no improvement should be claimed without that comparison.

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

Training uses `readDataset` to consume samples incrementally and stores active
binary feature indices, avoiding a second full dense dataset in memory. This is
an exact storage optimization; the feature order and model format are unchanged.

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
./my_torch_analyzer --predict models/basic-network_1.nn positions.txt
```

The command prints one prediction per nonblank line in the same order on stdout:
`Nothing`, `Check White`, `Check Black`, `Checkmate White`, or `Checkmate Black`.
The color identifies the attacking side, opposite to the FEN side to move.
On success, stderr is empty and the command returns 0. Invalid arguments, models,
or datasets return 84 with a diagnostic on stderr. A newly generated model is untrained; its class outputs have no
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

### Why there are several preparation directories

Preparation keeps intermediate outputs so that a transformation can be audited
and rebuilt without editing raw sources. They are not all independent datasets
that should be concatenated into training.

| Location / output | Purpose | Retention rule |
| --- | --- | --- |
| `dataset/raw/` | Original labeled source files | Preserve unchanged |
| `dataset/processed/cleaned.txt`, `report.json`, `provenance.jsonl` | Cleaned rows, rejected/duplicate counts and source mapping | Rebuildable preparation outputs; retain while needed to audit a run |
| `dataset/processed/chess-v1-color-mirror/` | Symmetry-derived training data | Keep while referenced by protocols/runs; validation/test remain separate |
| `dataset/processed/chess-v1-generated-*` | Different generation budgets, pools and mirrored variants | Inspect each manifest/report before using it; prepared data alone does not demonstrate a learning gain |
| Hidden `.generated-*` staging folders, when present | Generation workspace or interrupted preparation | Inspect status and references before cleanup; a name alone does not prove it is safe to delete |
| `dataset/splits/` | Notebook's current train/validation/test output | Rebuildable; use a frozen reference for comparable experiments |
| `dataset/references/` and tracked `dataset/manifests/` | Named frozen bytes and their identities/reconstruction settings | Preserve references used by recorded experiments and the final model |

New runs should reference shared datasets from `dataset/` and keep metrics,
histories, checkpoints and logs under `results/`. Historical one-off campaigns
also retain frozen data inside local results (including the large Adam
protocol's source campaign). Keep those archives intact until a verified
migration updates dependencies without rewriting historical evidence. Neither
this guide nor a normal Git clone moves or restores those local files.

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

The original dataset was **provided by Epitech for this project**. It is used
for training, evaluation and the defense in that educational context.
Preparation checks FEN syntax and label format. A subsequent
[rules-based audit](report/2026-11-08-label-and-error-audit/README.md) found all original
train and validation labels consistent with python-chess 1.11.2, while flagging
9 train and 2 validation positions as invalid. It did not audit the test split.
The frozen manifest records the limitations known at preparation time; see the
[current summary](#8-data-rights-and-predictive-limitations) below.

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

[`configs/protocols/compare-network-sizes.yml`](../configs/protocols/compare-network-sizes.yml) resolves train, validation,
and metadata to `dataset/references/chess-v1/splits/`. The standalone comparison
notebook defaults to this validation file too. Tests check the supplied config
against a small frozen reference; the real paths and hashes were also checked
against this manifest.

Train only on `train.txt`, choose settings/models on `validation.txt`, and reserve
`test.txt` for the final selected model. The experiment config does not include a
test dataset. After verification, launch a configured comparison with:

```sh
make experiment OUTPUT=results/experiments/YYYY-MM-DD-compare-network-sizes-01
```

The runner captures the resolved dataset paths and hashes in its experiment
manifest and detects changes during a campaign. Dataset verification does not
train, select a model or evaluate it on test.

### Color-mirror augmentation

[`scripts/augment_dataset.py`](../scripts/augment_dataset.py) reflects ranks
(`a1` ↔ `a8`, files unchanged), exchanges piece colors and the side to move,
maps castling colors and the en passant rank, and preserves move counters.
The three-class label is preserved; optional White/Black suffixes are exchanged.
A left-right reflection is a separate step ([section 7](#7-left-right-mirror-augmentation)).

The transformation creates another view of a training example without adding a
rules-derived answer to the network inputs. Applying it twice recovers the
canonical original row. Original rows are retained first; mirrors follow in
source order. Original duplicate inputs are collapsed, contradictory original
labels are rejected, and conflicting or duplicate added inputs are excluded.
Any mirror matching validation or test inputs is excluded; held-out labels
are not parsed for augmentation. Those splits are never augmented or reshuffled.

```sh
python3 -m scripts.augment_dataset \
  --splits dataset/references/chess-v1/splits \
  --output dataset/processed/chess-v1-color-mirror
```

The destination must be new. It contains `train.fen`, `additions.fen`,
`additions-source-lines.csv` and `manifest.json` with source/output hashes,
class counts and exclusions. Generation uses only the Python standard library.
The original train produces **1,141,143 unique rows** after exclusions. The
same transform can be applied to the generated-data reference described below;
use that reference's `splits/` as the input.

[`scripts/verify_augmentation.py`](../scripts/verify_augmentation.py) provides
additional checks against the optional Python chess oracle and the production
C++ encoder. The [augmentation report](report/2026-11-08-color-mirror-augmentation/README.md#augmentation-and-integrity)
records the independent checks and measured outcomes. More rows also mean more
updates per epoch; compare quality and total training cost, not epoch count alone.

## 5. Analyze balance and diversity

Open [`notebooks/dataset_analysis.ipynb`](../notebooks/dataset_analysis.ipynb) and
run all cells with the dependencies from `notebooks/requirements.txt`. It reads
the frozen `chess-v1` train and validation files, checks their manifest hashes,
and leaves the reserved test file unopened. Recreate the reference first on a
fresh checkout using the commands above.

The notebook shows class proportions, material-count distributions, side to
move, piece frequencies and board coverage. It checks duplicate input groups,
conflicting labels and train/validation overlap using the encoder's input key.
Missing or multiple kings are flagged, but this is not a full chess-legality
or label audit.

Two train-fitted diagnostic rules (global majority and majority by side to
move/piece count) quantify simple sampling shortcuts on validation. These rules
do not replace the supervised neural model. Numerical findings and charts stay
in the notebook; no files are exported into `results/`. Its conclusion section
records the observations for the checked dataset version.

Use the findings alongside model errors to justify a controlled experiment,
such as training-only balancing, material-coverage enrichment or a longer
baseline run. Compare validation macro-F1, per-class recall and runtime before
keeping a change. Do not infer real-game representativeness from a balanced
split, or tune settings on the final test set.

## 6. Generated game data

[`scripts/generate_dataset.py`](../scripts/generate_dataset.py) adds positions
from new legal games to a separate training file. Generation uses the optional,
pinned `python-chess` rules library. It is downloaded by pip, not bundled in the
repository. Install it only for generation and its tests:

```sh
python3 -m pip install -r scripts/requirements.txt
```

`make` and both C++ executables do not use Python or this library. Without it,
`make test` skips the generation tests and still runs the existing suite (whose
Python dependencies remain in root `requirements.txt`). CI installs both sets
so generation tests are exercised. The Makefile and C++ dependencies are unchanged.

```sh
python3 -m scripts.generate_dataset \
  --splits dataset/references/chess-v1/splits \
  --output dataset/processed/chess-v1-generated-games-60k-mate-heavy \
  --policy mixed --targets 15000 20000 25000 --games 160000 --seed 43 --workers 8
```

The trial targets exactly **60,000 new unique positions**: 15,000 `Nothing`,
20,000 `Check`, and 25,000 `Checkmate`. These are added to the original
572,445-row train, giving 632,445 rows. Earlier local trials are preserved.
`--workers` controls parallel game generation using the Python standard library.
Each game has its own seeded random stream; worker count does not affect the
resulting games or selected data. Set it to 1 for serial generation.

All games begin at the standard starting board and run for at most 240 plies
(half-moves). `mixed` alternates three policies: uniformly random legal moves,
capture-biased play (80% preference when captures exist), and check-biased play
(70% preference when checks exist, otherwise 40% capture preference). The two
biased policies take an immediate mate when available. Choices use sorted UCI
moves for reproducibility. This is a rules-based sampler, not engine self-play.
Games stop at a terminal result, including claimable draws, or the ply budget.

Sample every fourth ply, rotating the first sampled ply from 16 to 19 between
games so both sides to move are covered. Keep every checkmate even before
that threshold. Labels come from the resulting board: checkmate first, then
check, otherwise `Nothing` (including stalemate). Mixed play randomly retains
at most 12 candidates per class per game, across the full trajectory rather
than its first few moves. Adjust this limit with `--per-game`.

`--targets NOTHING CHECK CHECKMATE` sets exact class counts. Within each class,
reservoir sampling selects uniformly among eligible unique inputs from the
games processed so far. Common classes therefore continue drawing from later
games while rare mates accumulate. Stop when all targets are met, or fail
without installing an incomplete dataset if the `--games` budget runs out.
No rows are repeated to fill a target. The manifest records both the maximum
budget and the actual number of games played.

Without `--targets`, generation retains the first admissible inputs, with a
ceiling of `--per-class` (default 10,000) and runs the entire game budget.
The original defaults remain `--policy random --games 1000 --seed 42`.
Change `--max-plies`, `--min-ply`, or `--stride` to alter trajectory sampling.

The existing train rows stay first, followed by unique additions. Deduplication
uses the same 777-input key as color-mirror augmentation. Inputs already in
train, validation or test are excluded; held-out labels are never parsed.
Existing train/held-out overlap or contradictory original labels abort the run.
Validation and test remain fixed; do not mix the additions into a new split.

The new output directory contains:

| File | Purpose |
| --- | --- |
| `train.fen` | Original training rows plus additions; ready for the analyzer. |
| `additions.fen` | New positions only. |
| `additions-source.csv` | Output line, game number and ply for every addition. |
| `games.jsonl` | Full UCI move sequence and ending of each generated game. |
| `manifest.json` | Source/output hashes, generator hash, environment, settings, exclusions and class counts. |

An existing output directory is refused. Files are staged, generated labels and
legality are checked, then the directory is installed. Repeating the command
with a fresh destination and the same sources, settings and Python/chess versions
reproduces the FEN and provenance files. Outputs stay local under the ignored
`dataset/processed/` directory.

To change sampling without playing the games again, add
`--replay-games PATH/TO/games.jsonl` with the original game budget and ply limit
and a fresh output directory. Moves and game endings are checked during replay;
the source game file is hashed in the manifest. With unchanged sampling settings,
replay produces identical FEN and provenance files to fresh generation.

To compare against `train-reference-b128.yml` with matching initialization seed,
architecture, batch size, learning rate and stopping settings:

```sh
python3 scripts/experiment.py configs/protocols/train-generated-games.yml \
  --train-only --output results/training/generated-games-seed43-01
```

### Frozen training references

Prepared data is retained as separate references, each with copied held-outs
that are byte-identical to `chess-v1`. The generated-only training protocol
above now uses the frozen files, not disposable `dataset/processed/` outputs.

| Reference | Training rows | Contents |
| --- | ---: | --- |
| `chess-v1` | 572,445 | Original baseline. |
| `chess-v1-generated-60k` | 632,445 | Original train + 15k Nothing, 20k Check, 25k Checkmate. |
| `chess-v1-generated-120k-mirror` | 1,380,805 | Latest expanded generated train with color mirrors; not evaluated in the 60k trial. |
| `chess-v1-mirror-x4` | 2,275,774 | The color-mirror train + 1,134,631 left-right mirrors ([section 7](#7-left-right-mirror-augmentation)); trains the delivered model. |

`chess-v1-generated-60k-mirror` (1,260,989 rows) is archived: local dataset
copies were removed after its completed experiment. Its manifest, protocol,
model and measurements remain available. See the
[experiment and retention decision](report/2026-11-08-generated-60k/README.md).

Each reference lives under `dataset/references/NAME/`, with `splits/train.txt`,
`splits/validation.txt`, `splits/test.txt` and retained provenance files.
The tracked `dataset/manifests/NAME.json` records artifact hashes, counts,
source paths and hashes, and the parent reference. Large data stays local.
The original generated games and their source mapping are retained in the
generated-only reference; the mirror reference records its source line mapping
and links to that parent. No new chess games were generated to freeze these.

Verify a frozen variant without the optional chess library:

```sh
python3 -m scripts.freeze_dataset verify dataset/manifests/chess-v1-generated-60k.json
python3 -m scripts.freeze_dataset verify dataset/manifests/chess-v1-generated-120k-mirror.json
python3 -m scripts.freeze_dataset verify dataset/manifests/chess-v1-mirror-x4.json
```

Verification checks retained file hashes, class counts, unique model inputs,
split disjointness and unchanged held-outs. It does not require the disposable
preparation outputs. `scripts.freeze_dataset freeze` creates a new reference
from a prepared train and a parent manifest; existing references are refused.

Historical command for the completed 60k-mirror run (restore its archived
dataset before rerunning; this does not train on the newer 120k variant):

```sh
python3 scripts/experiment.py configs/protocols/train-generated-games-mirror.yml \
  --train-only --output results/training/generated-games-mirror-seed43-01
```

Inspect validation macro-F1, per-class recall, loss and runtime before retaining
the augmentation. An epoch now contains more batches, so equal epoch counts
also mean more parameter updates. Random play adds legal board diversity but
does not establish human-game coverage or improved model quality. Positions
from one game remain correlated despite sampling. No training benefit is
claimed by the generation checks; the reserved test set is used only to exclude
input overlap, never to score or select this augmentation.

### 60k trial (15k / 20k / 25k)

The seed-43 run generated **118,044 games**. The retained positions come from
51,083 distinct games. An independent audit replayed those games' 6,813,946
legal moves and matched every addition to its source game and ply.
All 60,000 inputs are unique and absent from the original train and held-outs;
the original rows and source hashes are unchanged. The C++ encoder agreed on
all added inputs, and the analyzer read all 60,000 positions successfully.

| Class | Added | White to move | Black to move |
| --- | ---: | ---: | ---: |
| Nothing | 15,000 | 7,414 | 7,586 |
| Check | 20,000 | 10,111 | 9,889 |
| Checkmate | 25,000 | 12,324 | 12,676 |

Material coverage includes 20,918 positions with 2–10 pieces, 21,389 with
11–20 pieces, and 17,693 with 21–32 pieces. All three game policies contribute:
16,899 random, 17,128 capture-biased, and 25,973 check-biased samples.
The local `verification.json` records the detailed distributions and replay
checks. No model was trained or selected in this data-generation run.

Compatibility checks also passed: a fresh portable C++ build and prediction
without Python in the build command, and `make test` both with the generation
library and in a fresh environment containing only root `requirements.txt`.
The latter skips the eight optional generation tests.

---

## 7. Left-right mirror augmentation

[`scripts/mirror_files.py`](../scripts/mirror_files.py) reflects every position
across the board's vertical axis: the a-file becomes the h-file, b becomes g,
and so on. The delivered model is trained on this ×4 train
([report](report/2026-11-08-mirror-view-ensemble.md)).

**Why labels stay correct.** Pieces move the same way toward either side of the
board, so a king in check stays in check, and a mate stays a mate. Castling is
the only rule that treats the two sides differently, and castling is never a
legal way out of check. The script therefore drops castling rights (`-`) from
mirrored rows; the 777 inputs do not encode them anyway. The en passant square
moves to the mirrored file. Side to move and both counters are unchanged.

| Example | FEN |
| --- | --- |
| Original | `8/6p1/6kp/6P1/6K1/5q2/8/7q w - - 3 57` (Checkmate) |
| Mirror | `8/1p6/pk6/1P6/1K6/2q5/8/q7 w - - 3 57` (Checkmate) |

**Input.** The color-mirror train
(`dataset/processed/chess-v1-color-mirror/train.fen`, 1,141,143 rows), which
already contains the original train and its color mirrors.

```sh
python3 scripts/mirror_files.py dataset/processed/chess-v1-color-mirror/train.fen \
  dataset/processed/chess-v1-mirror-x4/train.fen \
  --heldout dataset/references/chess-v1/splits/validation.txt \
            dataset/references/chess-v1/splits/test.txt
```

| Rows | Count |
| --- | ---: |
| Input rows (original + color mirrors) | 1,141,143 |
| Left-right mirrors added | 1,134,631 |
| Skipped: mirror already present in the train | 5,630 |
| Skipped: mirror inputs occur in validation or test | 882 |
| **Output train** | **2,275,774** |

| Class | Output train | Added mirrors |
| --- | ---: | ---: |
| Nothing | 945,299 | 471,024 |
| Check | 790,925 | 394,740 |
| Checkmate | 539,550 | 268,867 |

The output is frozen as `chess-v1-mirror-x4`
([manifest](../dataset/manifests/chess-v1-mirror-x4.json)):

```sh
python3 -m scripts.freeze_dataset freeze chess-v1-mirror-x4 \
  --train dataset/processed/chess-v1-mirror-x4/train.fen
```

Original rows come first and are unchanged; mirrors follow in source order.
Side to move stays balanced (1,137,882 White, 1,137,892 Black). The output
SHA-256 is `b03123716029236ed8f2…`; the full hash is in the
[report data](report/data/2026-11-08-mirror-view-ensemble.json).

**Checks.** python-chess 1.11.2 recomputes Check / Checkmate / Nothing for all
1,134,631 added rows and agrees with every label. Skipping held-out inputs keeps
validation and test independent of the train.

**Why it helps.** A network trained on the color-mirror train alone scores 89%
on validation but only 81% on the left-right mirrors of the same positions: it
had learned file-specific patterns. On the ×4 train, the 256/128 ReLU + Adam
network rises from 89.25% to 92.89% validation accuracy.

---

## 8. Data rights and predictive limitations

> **Key takeaway:** Epitech provided the original dataset for this project.
> Validation scores do not guarantee correct predictions on arbitrary boards.

### Data rights — what is established?

| Topic | Current status | Consequence |
| --- | --- | --- |
| Original dataset | **Provided by Epitech for the project** | Used for training, evaluation and the defense. |
| Scope of use | Educational project | This documentation does not assert an open license or unrestricted redistribution rights. |
| Locally generated positions | Generation provenance recorded | Kept distinct from the Epitech-provided reference data. |

Details: [frozen reference dataset](#4-frozen-reference-datasets).

### Predictive limitations — what can go wrong?

| Risk | Evidence or limitation | How we assess it |
| --- | --- | --- |
| Wrong predictions | The audited ReLU checkpoint confuses Check and Checkmate; blocked attack lines and rare mating patterns are difficult. | Inspect the confusion matrix, per-class recall and macro F1, rather than a global score alone. |
| Overstated generalization | Related positions can remain across splits; repeated validation tuning can make scores optimistic. | Exclude shared model-visible inputs; compare seeds on validation and reserve test for final assessment. |
| Different input distribution | New positions may differ from the training examples. | Scope claims to the evaluated data; do not promise reliability on arbitrary boards. |
| Confusing label quality with model reliability | Original train/validation labels match the rules in the audit; the test split was not audited. | Keep label checks separate from prediction measurements. A correct target does not guarantee a correct prediction. |

**Scope:** the network classifies positions; it does not play chess or guarantee
an exact chess-rules verdict. The error findings above concern the audited
checkpoint, not every model in the project.

**Evaluation rule:** compare configurations and seeds on validation → freeze the
model-selection rule → assess the selected model on the reserved test set.
Validation scores alone do not establish final test performance.

| Find the evidence | Document |
| --- | --- |
| Label checks and detailed error patterns | [Label and error audit](report/2026-11-08-label-and-error-audit/README.md) |
| Experiments, including unsuccessful approaches | [Training and evaluation reports](report/README.md) |
| Split, seed and final-test rules | [Evaluation protocol](evaluation.md#comparison-protocol-v1) |
