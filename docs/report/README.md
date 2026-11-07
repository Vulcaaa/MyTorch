# Training and evaluation reports

Reports interpret recorded results. Portable measurements in `data/` and figures
in `figures/` are versioned with these reports. Checkpoints, full histories, raw
logs and frozen executables/data remain local in the ignored `models/` and
`results/` directories; links into those directories require the local artifacts.
The comparison notebook also requires those local result directories.

| Date | Report | Main finding |
| --- | --- | --- |
| 2026-11-07 | [Reference checkpoints: epochs 100 and 338](2026-11-07-reference-checkpoints.md) | Continued training improves validation macro F1 by 9.69 percentage points |
| 2026-11-07 | [Compact, reference and large](2026-11-07-three-models.md) | Reference retains the best validation macro F1; doubling its parameters adds no overall gain |
| 2026-11-07 | [Batch-size screening: speed and early learning](2026-11-07-batch-size-screen.md) | Larger batches shorten epochs but learn more slowly per compute time at learning rate 0.01 |
| 2026-11-07 | [Batch 128 with learning rate 0.04](2026-11-07-batch-learning-rate-screen.md) | Five epochs take 26% less compute time with similar early classification; later convergence remains untested |
| 2026-11-07 | [Batch/LR follow-up: twenty epochs](2026-11-07-batch-learning-rate-followup.md) | Batch 128 / LR 0.08 uses 25.7% less epoch compute time and improves early validation F1 by 4.71 points; final convergence remains unmeasured |
| 2026-11-07 | [Batch 32 / LR 0.02 control](2026-11-07-batch-learning-rate-control.md) | Retuning the smaller batch recovers quality; batch 128 remains faster but scores 1.23 F1 points lower at twenty epochs |
| 2026-11-07 | [Matched 100-epoch comparison](2026-11-07-batch-learning-rate-100epochs.md) | Batch 128: 25.1% less epoch compute; -0.01 F1 points versus batch 32 at the 100-epoch horizon |

## Next run

The selected full-training protocol is
[`train-reference-b128.yml`](../../configs/protocols/train-reference-b128.yml):
batch 128, learning rate 0.08, seed 43, maximum 500 epochs and patience 20.
Its run was launched on 2026-11-08 and is not part of the completed results above.
The original reference protocol is retained for comparison.
