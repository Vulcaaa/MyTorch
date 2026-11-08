# Training performance history and experiment archive

For a compact numeric comparison, see the [benchmark summary](../benchmarks.md).

**Start with the [training performance history](training-history.md).** It is the
main defense reading path: starting point → controlled tuning → augmentation →
activations/optimizer → remaining errors and delivery. Each stage explains the
question, the measurements, their interpretation and the resulting decision,
with selected figures. The archive below supplies the detailed evidence.

Each dated folder contains its report as `README.md`, portable measurements as
`data.json` when available, and its SVG figures alongside them.

**Final project choice: Gustave's model.** The reports preserve the completed
local experiments; no further training is planned for the generated-60k trial.

## Main findings

- Continuing the first reference improves validation macro F1 from 72.28 to
  81.96%; its original training metadata are incomplete.
- A matched 100-epoch comparison supports batch 128 / LR 0.08 as a time-saving
  setting: 25.1% less epoch compute, −0.01 F1 points versus batch 32 / LR 0.02.
- Color-mirror augmentation improves mean F1 from 82.28 to 87.20% over three
  paired seeds; the larger training-work budget is part of the result.
- Tuned augmented ReLU/SGD reaches 88.29% mean F1 over three seeds. Large
  ReLU/Adam reaches 88.58% on one seed, with no established robust superiority.

These are validation development results with different experimental scopes,
not a single controlled leaderboard. The delivered root model remains distinct
from the shared Adam candidate. Check [current model provenance](2026-11-08-mirror-view-ensemble.md)
and [candidate identity](../../models/README.md) before presenting a score.

## Detailed reports by investigation

The dates below are report dates. The history identifies the older initial
training and continuation separately; it does not infer exact execution order
between parallel investigations.

### 1. Establish the reference and test capacity — 7 November

| Report | Why it matters |
| --- | --- |
| [Epochs 100 and 338](2026-11-07-reference-checkpoints/README.md) | Initial learning, continuation, class gains and the older model's separate test result |
| [Compact, reference and large](2026-11-07-three-models/README.md) | Capacity/quality compromise; unequal histories and changed-executable incident |

### 2. Tune batch/LR, then verify the longer run — 7–8 November

| Report | Role in the investigation |
| --- | --- |
| [Five-epoch batch screen](2026-11-07-batch-size-screen/README.md) | Faster epochs do not imply better learning per compute time |
| [Batch 128 / LR 0.04](2026-11-07-batch-learning-rate-screen/README.md) | First attempt to recover learning with a larger batch |
| [Neighbor LR screen and twenty epochs](2026-11-07-batch-learning-rate-followup/README.md) | Select LR 0.08 and compare against the original smaller-batch control |
| [Stronger batch-32 / LR-0.02 control](2026-11-07-batch-learning-rate-control/README.md) | Qualifies the early gain: LR tuning also helps the smaller batch |
| [Matched 100 epochs](2026-11-07-batch-learning-rate-100epochs/README.md) | Compare selected settings at a longer common horizon and against time |
| [Full batch-128 run](2026-11-08-reference-b128-lr008/README.md) | Selected epoch 210, 82.78% F1 on tuning seed 43 |
| [Additional seeds](2026-11-08-reference-seed-robustness/README.md) | Mean 82.28%, sample SD 0.39 pp over seeds 44–46 |

### 3. Improve data and diagnose the plateau — 8 November

| Report | Role in the investigation |
| --- | --- |
| [Reference error analysis](2026-11-08-reference-error-analysis/README.md) | Check recall and recurring error patterns |
| [Color-mirror augmentation](2026-11-08-color-mirror-augmentation/README.md) | Paired gains, data integrity, class scores and increased training cost |
| [Augmented train versus validation](2026-11-08-augmented-train-validation/README.md) | Residual training errors and a 3.72-point mean F1 gap |
| [SGD continuation](2026-11-08-sgd-checkpoint-resume/README.md) | No gain at unchanged LR; small reduced-rate gains followed by overfitting |
| [Generated positions: 60k trial](2026-11-08-generated-60k/README.md) | +60k with mirrors: 88.15% macro F1 / 88.83% accuracy in 68m 54s; below the local large Adam candidate; intermediate mirrored dataset archived |

### 4. Change learning dynamics and revisit capacity — 8 November

| Report | Role in the investigation |
| --- | --- |
| [Hidden activations and initialization](2026-11-08-hidden-activations/README.md) | Common-budget sigmoid/tanh/ReLU comparison over three seeds |
| [Augmented ReLU rates and seeds](2026-11-08-relu-color-mirror/README.md) | LR tuning, repeated quality gain and fewer updates |
| [Label and error audit](2026-11-08-label-and-error-audit/README.md) | Rules-based label agreement and difficult position patterns |
| [Adam and larger ReLU network](2026-11-08-adam-optimizer/README.md) | Single-seed optimizer candidates, controlled width follow-up and overfitting |

### 5. Deliver the final model — 8 November

| Report | Role in the investigation |
| --- | --- |
| [Improving the delivered model: four changes](2026-11-08-mirror-view-ensemble.md) | ReLU + Adam at 256/128, doubled training data and dropout: 93.66% test accuracy / 93.15% macro F1 at 2.4 MB; four views reach 95.05% validation at 5.7× the computation |

## Settings, decisions and retained evidence

- [Hyperparameter guide](../hyperparameters.md): role, tested alternatives and
  evidence for model/training settings; CLI defaults differ from later protocols.
- [Trial and failure log](../experiment-decisions.md): unsuccessful trials,
  corrections and decisions, with the original failed records preserved.
- [CPU benchmarks](../../benchmarks/cpu_blas/README.md): execution efficiency
  and numerical equivalence, separate from trained-model quality.

Portable measurements and figures are versioned in each report folder.
Full histories, raw logs, frozen executables/data and most checkpoints remain in
local ignored `results/` and `models/`; a clone does not restore them. The
comparison notebook needs those local runs. The explicitly shared Adam candidate
is tracked with its [README](../../models/README.md).

Dated reports retain their original conclusions and planned next steps. Those
historical recommendations do not supersede current delivery provenance. Missing
metadata stay missing; no new training or test evaluation was run for this history.
