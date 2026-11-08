# Training and evaluation reports

Reports interpret recorded results. Portable measurements in `data/` and figures
in `figures/` are versioned with these reports. Checkpoints, full histories, raw
logs and frozen executables/data generally remain local in `models/` and
`results/`; links into ignored artifacts require the local files. The selected
large ReLU + Adam candidate is explicitly shared in `models/` with its README.
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
| 2026-11-08 | [Selected batch-128 training](2026-11-08-reference-b128-lr008.md) | Validation macro F1 82.78%, +0.81 points versus the previous reference |
| 2026-11-08 | [Seed robustness](2026-11-08-reference-seed-robustness.md) | Additional seeds 44–46: mean macro F1 82.28%, sample standard deviation 0.39 points |
| 2026-11-08 | [Error analysis](2026-11-08-reference-error-analysis.md) | Check recall 77.36% on validation; shared errors and crowded Checkmate positions remain weaknesses |
| 2026-11-08 | [Color-mirror augmentation](2026-11-08-color-mirror-augmentation.md) | Paired seeds 44–46: +4.91 pp macro F1; 2.69× historical epoch compute |
| 2026-11-08 | [Augmented train versus validation](2026-11-08-augmented-train-validation.md) | Mean macro F1 90.92% train vs 87.20% validation; residual training errors and generalization gap |
| 2026-11-08 | [SGD checkpoint continuation](2026-11-08-sgd-checkpoint-resume.md) | LR 0.008: 87.57% validation macro F1, +0.165 pp; both reduced-rate checkpoints selected at additional epoch 1 |
| 2026-11-08 | [Hidden activations: sigmoid, tanh and ReLU over three seeds](2026-11-08-hidden-activations.md) | At 100 epochs, tanh and ReLU lead sigmoid by 3.8–4.3 macro-F1 points on every seed and are not separated from each other; ReLU + Xavier (82.28%) matches the sigmoid 500-epoch seed mean with about 8× fewer epochs |
| 2026-11-08 | [ReLU on the color-mirror train: learning rate and seeds](2026-11-08-relu-color-mirror.md) | ReLU + Xavier at LR 0.16: 88.29% mean validation macro F1 on seeds 44–46 versus 87.20% for the augmented sigmoid, ahead on every seed, with 6.6× fewer updates to the checkpoint |
| 2026-11-08 | [Label and error audit](2026-11-08-label-and-error-audit.md) | All 644,000 train and validation labels match the chess rules; the audited ReLU model reaches 92.66% on its own train, and 94% of validation errors are Check↔Checkmate or Nothing↔Check, concentrated on blocked lines and checks with few replies |
| 2026-11-08 | [Adam optimizer](2026-11-08-adam-optimizer.md) | Larger ReLU + Adam: 89.25% accuracy / 88.58% macro F1 in 15.99 min; +0.51 accuracy points versus the small model on one seed |
| 2026-11-08 | [Improving the delivered model: four changes](2026-11-08-mirror-view-ensemble.md) | ReLU + Adam at 256/128, doubled training data and dropout: 93.66% test accuracy / 93.15% macro F1 at 2.4 MB; four views reach 95.05% validation at 5.7× the computation |

## Training protocols

The selected batch-128 run and the three additional seed runs are complete.
Their protocol uses network 777 → 128 → 64 → 3, sigmoid/softmax, Xavier,
learning rate 0.08, maximum 500 epochs, patience 20 and one thread.
See [`train-reference-b128.yml`](../../configs/protocols/train-reference-b128.yml).
The original reference protocol is retained for comparison. Reports distinguish
that old baseline from the selected batch-128 checkpoints.
