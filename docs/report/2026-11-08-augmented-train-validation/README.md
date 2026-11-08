# Augmented checkpoints: train versus validation

Evaluate the saved seeds 44–46 on all **1,141,143 augmented training positions**, using the same frozen analyzer as their existing **71,555-position validation** evaluations. The selected checkpoints are unchanged. This is inference only: no retraining and no test evaluation.

## Results

| Seed | Selected epoch | Train accuracy | Validation accuracy | Train macro F1 | Validation macro F1 | F1 gap (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 44 | 227 | 91.19% | 87.63% | 90.53% | 86.92% | 3.61 |
| 45 | 240 | 91.66% | 88.02% | 90.97% | 87.27% | 3.70 |
| 46 | 281 | 91.87% | 88.06% | 91.27% | 87.40% | 3.86 |

Mean macro F1: **90.92% train**, **87.20% validation**; mean paired gap **3.72 percentage points**.

Means below give equal weight to each seed. The portable JSON also retains every per-seed score, support and confusion matrix.

| Class | Train precision | Validation precision | Train recall | Validation recall | Train F1 | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nothing | 96.82% | 93.77% | 95.74% | 93.26% | 96.27% | 93.51% |
| Check | 88.48% | 83.54% | 88.20% | 83.35% | 88.33% | 83.43% |
| Checkmate | 87.14% | 84.14% | 89.21% | 85.18% | 88.16% | 84.65% |

## Interpretation

Both residual training errors and a generalization gap remain. Check and Checkmate are already the weakest classes on train, so a validation-only explanation is insufficient. These measurements do not prove that increasing model size or changing activations will help; optimization, capacity, representation and labels are not separated by this diagnostic. A controlled capacity experiment is a hypothesis to test, not an established remedy.

Training scores are in-sample and optimistic. Augmented color-equivalent pairs are correlated, not independent observations. Train and validation have different positions, sizes and slightly different class proportions. Validation was used to select the checkpoints; no fresh holdout claim is made.

## Notebook and reproduction

- [Generic comparison notebook](../../../notebooks/comparison.ipynb)
- [Portable measurements and fingerprints](data.json)
- Raw evaluations: `results/evaluation/2026-11-08-augmented-train-seed-{44,45,46}-01/` (one metrics JSON and SVG confusion matrix per seed).
- Validation measurements are reused from the frozen color-mirror campaign; no duplicate validation inference is needed.

Reproduce one train evaluation with the existing script, a new output directory and the frozen executable; repeat for all three seeds:

```sh
. .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DEFAULT_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m scripts.evaluate \
  results/experiments/2026-11-08-color-mirror-augmentation-01/runs/seed-44/model.nn \
  results/experiments/2026-11-08-color-mirror-augmentation-01/data/train.fen \
  --analyzer results/experiments/2026-11-08-color-mirror-augmentation-01/tools/my_torch_analyzer \
  --output results/evaluation/NEW-TRAIN-SEED-44
```

The comparison notebook remains a generic tool for selected saved runs; this report and its JSON retain the experiment-specific train/validation diagnostic. Same-model and same-analyzer hashes were checked for every train/validation pair. Frozen train/model/analyzer hashes were checked before and after inference.
