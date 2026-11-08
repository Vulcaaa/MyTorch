# Benchmarks

Recorded results — **8 November 2026**. Scores below use the same **71,555-position
validation split**. Original train: **572,445** positions; color-mirror train:
**1,141,143**. Saved checkpoints minimize validation loss.

## Model comparison

Reference: **777 → 128 → 64 → 3** (108,035 parameters).
Large: **777 → 256 → 128 → 3** (232,451 parameters).
**±** denotes sample standard deviation across three seeds; **pp** means percentage points.

| Model / training | Train data | Seeds | Accuracy | Macro F1 |
| --- | --- | --- | ---: | ---: |
| Compact sigmoid / SGD 0.01¹ | Original | 43 | 80.41% | 79.78% |
| Reference sigmoid / SGD 0.01¹ | Original | 43 | 82.57% | 81.96% |
| Large sigmoid / SGD 0.01¹ | Original | 43 | 82.36% | 81.84% |
| Reference sigmoid / SGD 0.08 | Original | 44–46 | 82.87% | 82.28 ± 0.39% |
| Reference sigmoid / SGD 0.08 | Color-mirror | 44–46 | 87.90% | 87.20 ± 0.25% |
| Reference ReLU / SGD 0.16 | Color-mirror | 44–46 | 88.97% | **88.29 ± 0.22%** |
| Reference sigmoid / SGD 0.008 continuation | Color-mirror | 46 | 88.23% | 87.57% |
| Reference sigmoid / Adam 0.003 | Color-mirror | 46 | 87.98% | 87.46% |
| Reference ReLU / Adam 0.001 | Color-mirror | 46 | 88.74% | 88.13% |
| **Large ReLU / Adam 0.001** | Color-mirror | 46 | **89.25%** | **88.58%** |

¹ Historical checkpoints have unequal training histories. Compact: 777 → 64 → 3,
49,987 parameters. Three-seed rows report means; the others report one checkpoint.

Sources: [architecture](report/2026-11-07-three-models.md),
[seeds](report/2026-11-08-reference-seed-robustness.md),
[augmentation](report/2026-11-08-color-mirror-augmentation.md),
[ReLU](report/2026-11-08-relu-color-mirror.md),
[continuation](report/2026-11-08-sgd-checkpoint-resume.md),
[Adam](report/2026-11-08-adam-optimizer.md).

## What each experiment changed

| Comparison | Numeric result | Scope / conclusion |
| --- | --- | --- |
| Color-mirror augmentation | **+4.91 pp F1**; 82.28% → 87.20% | Three paired seeds; historical epoch compute **2.69×** higher, no equal-compute control |
| ReLU 0.16 vs sigmoid 0.08, augmented train | **+1.09 pp F1**; **6.6× fewer updates** to checkpoint | Three seeds each; Linux vs Mac, so no time-speedup claim |
| Sigmoid → tanh / ReLU, original train, 100-epoch maximum | F1 **77.94% → 81.98% / 82.28%** | Three seeds; tanh/ReLU ahead, sigmoid still improving at epoch 100 |
| ReLU initialization: Xavier vs He | F1 **82.28% vs 81.75%** | Three seeds; no separation under the experiment's seed-range rule |
| Batch 32 / LR 0.02 → batch 128 / LR 0.08, 100 epochs | **1,007.19 → 754.05 s**; F1 **77.72% → 77.71%** | **25.1% less epoch compute**; one seed, both batch and LR change |
| SGD continuation at LR 0.008 | F1 **87.40% → 87.57%** | One seed; selects first additional epoch, modest gain |
| Reference → large ReLU + Adam | **+0.44 pp F1**, **365 fewer errors** | One seed; roughly **57% longer training** |

Sources: model reports above, [activations / initialization](report/2026-11-08-hidden-activations.md),
[matched batch/LR comparison](report/2026-11-07-batch-learning-rate-100epochs.md).

## Cost and remaining errors

| ReLU + Adam, augmented train, seed 46 | Reference | Large |
| --- | ---: | ---: |
| Full training process time | 10.21 min | 15.99 min |
| Selected / completed epochs | 22 / 42 | 9 / 29 |
| Serialized checkpoint, including Adam state | 10.92 MB | 23.54 MB |
| Train / validation macro F1 | 92.27% / 88.13% | 94.33% / 88.58% |
| Validation Check F1 | 84.66% | 85.43% |
| Validation Checkmate F1 | 85.90% | 85.86% |

Large improves the aggregate score, while Checkmate F1 stays essentially unchanged.
It still makes **7,694 / 71,555 validation errors**. Source: [Adam report](report/2026-11-08-adam-optimizer.md).

| CPU backend | Threads | Median s/epoch | Speedup |
| --- | ---: | ---: | ---: |
| Baseline scalar | 1 | 18.994 | 1.00× |
| BLAS | **1** | **11.062** | **1.72×** |
| BLAS | 4 | 11.404 | 1.67× |

Full original dataset, Ryzen 7 7840HS / OpenBLAS, batch 32, three timing repeats.
Includes updates and full train/validation losses; excludes loading/saving and
classification. Maximum parameter difference: **7.77 × 10⁻¹⁶**, identical accuracy.
Source: [BLAS benchmark](../benchmarks/cpu_blas/results/linux-2026-11-07/README.md).

## Reading the results

- **Highest recorded validation candidate:** large ReLU + Adam; tested on one seed.
- **Most consistent data gain:** augmentation, positive on all three paired seeds.
- Validation is reused for tuning; hardware and budgets differ across campaigns.
  Timings across unrelated campaigns do not establish speedups. No model above
  demonstrates 95% recall for every class.
- Historical reference test: **82.13% accuracy / 81.49% F1**. Newer candidates have
  no reported test score; the project test set has already been consulted.
- **Not benchmarked:** inference latency and peak RAM. File sizes do not measure RAM.

[Detailed reports](report/README.md) · [Measurements](report/data/) · [Figures](report/figures/)
