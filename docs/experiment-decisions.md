# Trial, failure and decision log

This retrospective index records what the experiments ruled in or out. Each row
links to recorded evidence; explanations are hypotheses unless the comparison
isolates them. It does not rewrite failed manifests or invent missing histories.
Snapshot: **8 November 2026**. Scores are validation scores unless specified.

## Learning and performance decisions

| Observation / question | Test and result | Decision and limit | Evidence |
| --- | --- | --- | --- |
| Could always predicting Nothing look acceptable? | Its support gives 41.58% validation accuracy; Check and Checkmate recall are zero. With the evaluator's zero convention its macro F1 is about 19.58% | Show accuracy together with macro F1, per-class precision/recall and the confusion matrix. This baseline is calculated from counts, not a new inference run | [Class counts](datasets.md#preparation-report-chess-v1), [metrics](evaluation.md#comparison-protocol-v1) |
| Would a larger sigmoid network solve the errors? | Compact/reference/large F1: 79.78 / 81.96 / 81.84%; larger network fits train better but does not improve overall validation | Keep the reference as that experiment's compromise. Unequal histories and one seed prevent a general conclusion about capacity | [Architecture report](report/2026-11-07-three-models/README.md) |
| Would faster epochs mean faster learning? | At fixed LR 0.01, batches 128/256 have faster epochs but worse early loss per compute time. Matched 100-epoch batch/LR retuning gives 77.72 / 77.71% F1 and 25.1% less compute for batch 128 | Retain batch 128 with its tuned LR; distinguish throughput from time to quality | [Screen](report/2026-11-07-batch-size-screen/README.md), [follow-up](report/2026-11-07-batch-learning-rate-100epochs/README.md) |
| Would more threads speed up training? | Full-data BLAS medians: 11.062 s at one thread, 11.404 s at four, versus 18.994 s baseline | Use one thread for the measured Linux reference. BLAS helps; additional threads do not help this workload | [Benchmark](../benchmarks/cpu_blas/results/linux-2026-11-07/README.md) |
| Could a chess symmetry improve class quality? | Color mirrors raise mean F1 from 82.28 to 87.20%, all three paired seeds improve; historical epoch compute increases 2.69× | Retain as a supported data alternative, while acknowledging additional updates and near-duplicate relationships | [Augmentation](report/2026-11-08-color-mirror-augmentation/README.md) |
| Would simply continuing the augmented sigmoid help? | Same-LR continuation retains epoch zero. Lower LR 0.008 selects additional epoch 1: 87.40 → 87.57% F1, then validation worsens | Small local improvement; extending those trajectories has no observed benefit. Check recall slightly falls despite aggregate F1 rising | [Continuation](report/2026-11-08-sgd-checkpoint-resume/README.md) |
| Did ReLU need a lower LR to avoid validation spikes? | LR 0.02–0.32 tested; 0.16 scores 88.47% F1 on tuning seed 46, above lower rates. Three-seed mean: 88.29%, versus 87.20% sigmoid | Retain the measured 0.16 setting; a lower LR is not automatically better. Cross-platform and reused-validation limits remain | [ReLU study](report/2026-11-08-relu-color-mirror/README.md) |
| Did remaining errors indicate bad labels? | All 644,000 original train/validation labels match the offline rules oracle; errors concentrate on blocked lines and checks with few replies | Label noise is not supported as the explanation on those audited rows. Label agreement does not certify full legality or source diversity | [Label/error audit](report/2026-11-08-label-and-error-audit/README.md) |
| Would widening ReLU + Adam help? | Same-seed width comparison: 88.13 → 88.58% F1, 88.74 → 89.25% accuracy, about 57% longer training. Checkmate F1 does not improve | Keep as a candidate, not a proven winner across seeds; more capacity widens the train/validation gap | [Adam capacity report](report/2026-11-08-adam-optimizer/README.md) |

## Execution failures and corrections

| Incident | What was checked / corrected | What the retained evidence supports | Source |
| --- | --- | --- | --- |
| Historical compact/large runner detects analyzer modification | Saved models are independently evaluated with one frozen analyzer | Valid checkpoint comparison, but no retroactive guarantee that training used a single executable. Keep original failed manifests | [Three-model report](report/2026-11-07-three-models/README.md#what-the-evidence-supports) |
| Oracle rejects mirrored FEN castling-field order before training | Emit standard KQkq order; rebuild candidate data and rerun integrity/oracle/C++ checks | Rejected data never become evidence for a completed training gain. Preserve rejected pre-training artifacts separately | [Augmentation integrity](report/2026-11-08-color-mirror-augmentation/README.md#augmentation-and-integrity) |
| Continuation driver cannot parse completion diagnostics from stderr | Recover already completed first arm without retraining; fix parsing and run remaining arms; isolation/recovery check passes | Orchestration failure, not failed neural learning. The original run and recovery remain traceable | [Continuation diagnostics](report/2026-11-08-sgd-checkpoint-resume/README.md#durations-and-limits) |

## Retention and remaining work

Preserve full histories, logs, failed status, frozen-tool/data hashes and
checkpoints locally. Portable measurements and SVGs remain under `docs/report/`;
their [index](report/README.md) identifies each experiment. Never fill missing
timings, settings or epochs by inference. A manifest's last state does not prove
that its process is still running.

For a new significant result, add the observed problem, the hypothesis, the
controlled change, the measured before/after and the resulting decision. Keep
the original dated report authoritative. Final checkpoint selection, generated
data training gains and inference/peak-memory measurements remain separate
work until recorded; no such result is implied by this log.
