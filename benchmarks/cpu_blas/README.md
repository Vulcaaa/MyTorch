# CPU BLAS evidence

This directory is intentionally independent of experiment runners and notebooks.
Keep it and its output directory when reorganizing the project for the defense.

The harness measures the actual C++ forward/backpropagation, plain SGD updates,
and complete train/validation loss evaluation each epoch. Parsing, initialization,
warm-up, serialization and final accuracy computation are outside the timed loop.
It preserves FP64, network `777 → 128 → 64 → 3`, sigmoid hidden layers, softmax
output, Xavier initialization, seed 43 and learning rate 0.01. It does not increase
the batch size or change the optimizer to manufacture a throughput gain.
The BLAS backend retains sparse first-layer work when inputs are mostly zero;
dense layer operations and dataset loss evaluation use blocked matrix products.

## Prepare while another experiment runs

Use a clean checkout of the baseline commit, or an exact pre-edit source snapshot.
The preparation step only copies sources/data and compiles separate binaries; it
never rebuilds the analyzer used by the running experiment.

```sh
python3 benchmarks/cpu_blas/run.py --prepare-only \
  --baseline-root /path/to/baseline \
  --train /path/to/train.txt --validation /path/to/validation.txt \
  --output /path/to/evidence/cpu-blas-01
```

Linux requires `libopenblas-dev` and `pkg-config` (Ubuntu/Debian); macOS uses
the system Accelerate framework. For a local OpenBLAS installation, supply
`--blas-cflags='-I/path/include'` and
`--blas-libs='-L/path/lib -Wl,-rpath,/path/lib -lopenblas'`.

Default fixtures are the first 8,192 nonempty train rows and 2,048 validation rows;
the exact bytes and SHA-256 hashes are saved. Default training is five epochs,
batch 32. `--train-rows`, `--validation-rows`, `--epochs`, and `--batch-size` can
change these settings at preparation time. Use a new output directory per protocol.

## Measure after the experiment

```sh
python3 benchmarks/cpu_blas/run.py \
  --run-prepared /path/to/evidence/cpu-blas-01 \
  --threads 1 2 4 8 --repeats 5
```

Optional `--wait-pid PID` waits for the entire experiment runner to exit before
timing. The runner refuses to time while a visible analyzer is training; launch
it where all relevant host processes are visible. Stop other training campaigns
and avoid concurrent CPU-heavy jobs. The runner does not kill or suspend them.

Each configuration runs sequentially in a fresh process. Run order is shuffled
with a fixed seed, and the relevant OpenBLAS/OpenMP/Accelerate thread environment
variables are set explicitly. Thread counts are requested limits: libraries may
choose fewer worker threads for small products. A matched baseline scalar build,
candidate scalar build and candidate BLAS build distinguish backend gains from
other source changes. Both use GCC/Clang `-O2`, without fast-math.

## Retained artifacts

- `sources/`, `Benchmark.cpp`, `run.py`: exact inputs to the build.
- `manifest.json`: source/data/binary hashes, compiler, CPU, flags, linked libraries.
- `measurements/`: run order, environment, every epoch/phase duration, losses,
  validation accuracy, stderr and each final serialized model.
- `summary.json`, `report.md`: median/range, speedup against baseline, positions
  per second including epoch loss evaluation, and numerical agreement.

Success requires maximum absolute final parameter and validation-loss differences
below `1e-10` and identical validation accuracy. Small rounding differences are
expected from reordered additions. A failed comparison retains its raw artifacts
and is not accepted as a performance result.

`--training-doc /path/to/docs/training.md` can replace only the reserved
`cpu-blas-measurements` section after successful measurements. Missing or ambiguous
markers cause a failure without changing the document. It never commits or pushes.

These are fixed-epoch throughput measurements on a subset. They do not establish
full-dataset speed, peak memory, or time to reach a validation-quality target.
Run a separate matched full training comparison before making those claims.

For a shareable figure (requires the project's optional matplotlib dependency):

```sh
python3 benchmarks/cpu_blas/plot.py /path/to/evidence/cpu-blas-01 \
  --output /path/to/evidence/cpu-blas-01
```

PNG and SVG exports show median epoch time and the observed timing range.
