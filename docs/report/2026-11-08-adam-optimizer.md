# Adam: similar quality with less training work

**Conclusion: retain Adam as the preferred training setup for the next experiments.**
Its validation quality is almost unchanged, while it needs far fewer updates.
The 0.08-point accuracy difference is not enough evidence to reject Adam.
This is an observed tradeoff on one seed, not proof of statistical equivalence.

Same network, sigmoid/softmax, Xavier, seed 46, augmented training data,
batch 128 and 71,555 validation positions. SGD uses LR 0.08; Adam uses LR 0.003,
selected from three ten-minute trials. Both full runs use a 500-epoch maximum,
patience 20 and min_delta 0.0001.

| Measure | SGD | Adam |
| --- | ---: | ---: |
| Validation accuracy | **88.06%** | 87.98% |
| Validation macro F1 | 87.40% | **87.46%** |
| Selected epoch | 281 | **20** |
| Completed epochs | 301 | **40** |
| Full-run wall time | 72.90 min | **11.25 min** |

Adam makes 56 additional errors overall, improves Checkmate F1 from 85.03% to
85.78%, and reduces Nothing F1 from 93.50% to 92.89%.

**The benefit is learning efficiency, not higher accuracy.** The full run uses
356,640 updates versus 2,683,716 for SGD: about 7.5 times fewer. Adam updates are
more expensive individually, so this ratio is not a compute-speedup measurement.
SGD timing is historical and used dense storage; Adam used compact binary storage.
The Adam LR search adds 30.88 minutes, bringing screening plus full training to
42.13 minutes, excluding evaluation.

Next: compare against Gustave's ReLU run on augmented data before choosing the
final delivered model. The delivered SGD checkpoint remains unchanged; the test
set has not been evaluated.

Local evidence: [campaign](../../results/experiments/2026-11-08-adam-screen-02/README.md),
[Adam curve](../../results/experiments/2026-11-08-adam-screen-02/full/history.svg),
[SGD curve](../../results/experiments/2026-11-08-color-mirror-augmentation-01/runs/seed-46/history.svg).
The full Adam checkpoint is byte-identical to the selected LR-0.003 screening
checkpoint; both have SHA-256 `038ab7b47c861dd6c2f46f10b42f10e3ddc886a706f799c7d1653cb5360ec052`.
