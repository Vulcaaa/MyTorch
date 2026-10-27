# Chess data encoding

`encodeFEN(fen)` converts a six-field FEN into a `std::vector<double>` of
**777 binary values** (`0.0` or `1.0`). The layout is fixed:

| Input | Count | Indices |
| --- | --- | --- |
| Piece placement | 64 × 12 = 768 | 0–767 |
| Side to move | 1 | 768 |
| En passant target file | 8 | 769–776 |

Castling rights and move counters are validated but not encoded. All indices
are zero-based.

## Piece placement

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

## Side to move and en passant

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

## Validation

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

## Dataset and model contract

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

The generator remains generic. FEN encoding is implemented; analyzer integration,
training, and prediction are pending. The analyzer must reject incompatible
model dimensions when integrated.
