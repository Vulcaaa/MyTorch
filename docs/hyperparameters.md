# Model and training hyperparameters

This page connects configurable settings to the measurements that justify them.
It distinguishes implemented options from tested choices; no globally optimal
configuration has been established. Evidence snapshot: **8 November 2026**.

## Model settings versus training settings

**Model hyperparameters** define representation and structure: input features,
layer widths/depth and activations. **Training hyperparameters** define learning:
optimizer, learning rate, batch size, initialization and stopping. Where a value
is stored is separate from its role: learning rate and optimizer are training
choices even though this implementation stores them in the model file.

| Setting | What it controls | Where it is set | Evidence and decision |
| --- | --- | --- | --- |
| Input/output widths | Information received and classes predicted | Network `layers`; chess requires 777 / 3 | [Input rationale](datasets.md#1-fen-encoding-and-c-dataset-contract), [class contract](output-encoding.md); keep the same contract across comparisons |
| Hidden widths/depth | Capacity and parameter count | Network `layers` | Compact/reference/large sigmoid: 79.78 / 81.96 / 81.84% validation macro F1, with unequal histories; later Adam widening gains 0.44 pp on one seed. Capacity interacts with training, so more neurons are not an automatic improvement |
| Hidden activation | Nonlinearity and gradient propagation | `hidden_activation` | Sigmoid, tanh and ReLU tested. Tuned augmented ReLU/SGD scores 88.29% mean F1 versus 87.20% for augmented sigmoid, three seeds each |
| Output/loss | Three mutually exclusive state scores and training objective | `output_activation: softmax`; categorical cross-entropy in code | Softmax/CE matches the three-state target. Sigmoid outputs are supported by the module, but no chess quality advantage is demonstrated |
| Initialization | Initial weight scale | `initialization` | Xavier and He compared with ReLU; the activation study does not separate their quality under its decision rule. Xavier remains the measured baseline, not a universal winner |
| Optimizer | How gradients change parameters | `optimizer` | SGD baseline; Adam adds gradient moments and saved state. Large ReLU/Adam is the highest individual recorded candidate, but only one seed; optimizer, activation and LR comparisons must identify all changed settings |
| Learning rate | Size of updates | Model `learning_rate` | Batch-128 sigmoid uses 0.08; augmented ReLU/SGD retains 0.16 after testing 0.02–0.32; large ReLU/Adam uses 0.001. Rates are specific to the optimizer/data/architecture |
| Random seed | Initialization and training order | Model `seed`; protocol `seeds` overrides it | Repeat finalists rather than report only the best seed. Seeds do not guarantee identical weights/shuffles across platforms |
| Batch size | Examples averaged per update | `--batch-size`; protocol `batch_size` | Batch 128 / LR 0.08 uses 25.1% less epoch compute than batch 32 / LR 0.02 with almost equal F1 at 100 epochs; this tests two complete settings |
| Maximum epochs | Upper bound on passes through train | `--epochs`; protocol `epochs` | Later full protocols allow 500. This is a ceiling, not a target: the large Adam run selects epoch 9 and stops at 29 |
| Patience | Delay before stopping without sufficient validation improvement | `--patience`; protocol `patience` | Later full protocols use 20; training retains the lowest-loss checkpoint, not the last epoch. No patience sweep establishes 20 as optimal |
| Minimum improvement | Validation-loss decrease needed to reset patience | `--min-delta`; protocol `min_delta` | 0.0001 in the later protocols. It controls stopping, not whether a newly lower-loss checkpoint is saved; no sensitivity study was performed |
| Thread limit | CPU resource use | `--threads`; BLAS environment variables | One thread is the measured Linux choice for the reference network; increasing threads did not improve its benchmark |

The generator reads model settings from YAML and stores them in `.nn`. The
training CLI reads learning rate, optimizer and seed from that model; it has no
`--learning-rate` or `--optimizer` switch. The experiment runner applies network
overrides before generation. For the full schemas and defaults, see
[network configuration](architecture.md#network-configuration) and
[training options](training.md#options).

## Delivered checkpoint and measured alternatives

All three settings below use the same 1,141,143-position color-mirror train and
71,555-position validation. They use softmax, Xavier, batch 128, maximum 500
epochs, patience 20 and min_delta 0.0001. Those protocol settings differ from the
CLI defaults (100 epochs, batch 32, patience 10).

| Item | Delivered sigmoid | Repeated ReLU/SGD setting | Shared large ReLU/Adam candidate |
| --- | --- | --- | --- |
| Hidden layers | 128 / 64 | 128 / 64 | 256 / 128 |
| Trainable parameters | 108,035 | 108,035 | 232,451 |
| Activation / optimizer / LR | Sigmoid / SGD / 0.08 | ReLU / SGD / 0.16 | ReLU / Adam / 0.001 |
| Evidence | Seed-46 delivered checkpoint; three-seed campaign available | Seeds 44–46 | Seed 46 only |
| Validation macro F1 | 87.40% delivered; 87.20% campaign mean | 88.29% mean, sample SD 0.22 pp | 88.58% |
| Selected / completed epoch | 281 / 301 | Selected 35–42 across seeds | 9 / 29 |
| Status | Root `my_torch_network.nn` | Measured alternative, not promoted | Shared in `models/`, not promoted |

Do not rank a three-seed mean and a best individual score as equally strong
evidence. These results support specific improvements, not a completed final
model decision or a fresh test estimate. The test split has historical scores
for older models; it must not be described as never consulted.

## How gains were validated

1. Keep data fixed within a comparison and record hashes, initialization,
   settings, executable, environment and stopping rules.
2. Separate checkpoint selection by **minimum validation loss** from comparing
   the saved checkpoints by **validation macro F1**, accuracy and per-class scores.
3. Report selected and completed epochs, train/validation curves and costs.
   Augmentation changes examples per epoch; compare updates/time as well.
4. Keep negative results and execution failures. Repeat finalists where feasible;
   do not turn single-seed or historical timing differences into robust gains.

### Supporting experiments

- [Architecture comparison](report/2026-11-07-three-models.md) and
  [matched batch/LR comparison](report/2026-11-07-batch-learning-rate-100epochs.md).
- [Activation/initialization study](report/2026-11-08-hidden-activations.md),
  [augmentation](report/2026-11-08-color-mirror-augmentation.md) and
  [ReLU learning rates/seeds](report/2026-11-08-relu-color-mirror.md).
- [Adam and capacity](report/2026-11-08-adam-optimizer.md),
  [thread/BLAS benchmark](../benchmarks/cpu_blas/results/linux-2026-11-07/README.md)
  and [trial and failure log](experiment-decisions.md).

## What remains to decide

After the remaining declared trials, explicitly retain or replace the delivered
checkpoint, using validation and the quality/cost tradeoff. Freeze that decision
before its final test evaluation and update model provenance together with the
artifact. No additional broad search, encoding rewrite or new regularizer is
required to explain the completed work. Inference latency and peak memory are
still unmeasured; parameter count and serialized bytes are not substitutes.
