# Evaluation and evidence plan

This document defines the work, measurements, and evidence needed to assess
MyTorch and explain its design choices at the defense. It is a working plan,
not a report of completed experiments. An unchecked item is not completed;
proposed settings are not measured results.

## 1. Objectives and priorities

The goal is to deliver a reproducible supervised-learning solution that
distinguishes `Nothing`, `Check`, and `Checkmate`, and to explain its quality,
learning behavior, and computational cost with evidence.

Priority order:

1. Establish reliable datasets and an evaluation script.
2. Validate training and model saving on small deterministic examples.
3. Train a reference model and inspect learning and validation curves.
4. Compare a limited set of architectures and training settings.
5. Measure one practical alternative to an existing choice.
6. Select on validation data, evaluate on test data, and package the model and
   supporting report.

Parallel training is optional. It should not delay a working trained model,
correct evaluation, overfitting control, or reproducible delivery.

## 2. Starting point and dependencies

At preparation time, generation, serialization, FEN encoding, dataset loading,
and prediction are implemented. Training orchestration is still incomplete.
Cleaning and splitting work is available in separate pull requests and must
be integrated before using it as the shared experimental reference.

Work that can proceed before backpropagation is available:

- Implement and verify prediction metrics and plots using known examples.
- Test the evaluator against an untrained model as a pipeline check only.
- Record dataset composition, provenance, and split identity.
- Prepare experiment settings, result formats, and reporting templates.

Work that requires functioning training:

- Compare learned models, learning curves, and stopping behavior.
- Measure training cost and the impact of learning parameters.
- Select and deliver a trained model with reproducible results.

The neural computation interface and epoch metrics must be agreed before
connecting the experiment runner. Classification output remains one class per
position on stdout; diagnostics and training history need separate outputs.

## 3. Data protocol

### Preparation and traceability

- [ ] Inventory sources, versions or retrieval dates, licenses, and permitted
      use or redistribution. Mark unknown permissions explicitly.
- [ ] Record raw, accepted, rejected, duplicate, and conflicting-label counts.
- [ ] Preserve rejection reasons and source references for inspection.
- [ ] Record the cleaned input hash, split seed, ratios, and class counts.
- [ ] Verify that identical model-visible inputs never cross split boundaries.
- [ ] Retain the data used for training so it can be shown at the defense.

### Separation of responsibilities

| Dataset | Purpose | Permitted influence |
| --- | --- | --- |
| Train | Update weights and biases | Fit learned parameters; apply optional training-only balancing |
| Validation | Compare settings and monitor generalization | Choose architecture, hyperparameters, stopping point, and model |
| Test | Estimate final predictive quality | Evaluate after the choices are frozen |

Validation and test retain their original class distributions. Never update
weights from either set. Define the majority-class baseline from train counts.
Keep dataset files fixed across comparable runs and record their hashes.

Do not repeatedly inspect test results while tuning. If a test result motivates
further tuning, disclose that the test has influenced development and no longer
provides a fully independent final estimate.

**Evidence:** a dataset summary table, preparation commands, split manifest,
source/license notes, and an explanation of remaining sampling or labeling bias.

## 4. Prediction quality

### Required measurements

| Measurement | Question answered |
| --- | --- |
| Accuracy | What fraction of all positions is classified correctly? |
| Per-class precision | When a class is predicted, how often is it correct? |
| Per-class recall | What fraction of the true examples of a class is detected? |
| Per-class F1 | How well are precision and recall combined for each class? |
| Macro-F1 | How well does the model perform when each class has equal weight? |
| Class support | How many examples support each class result? |
| Confusion matrix | Which true classes are confused with which predictions? |

Use the fixed order `Nothing`, `Check`, `Checkmate`. For a class, precision is
`TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is
`2 * TP / (2 * TP + FP + FN)`. Macro-F1 is the mean of the three class F1 scores.
For undefined ratios, report a score of zero with an explicit warning and the
class support; do not silently omit the class from the macro average.

Accuracy alone is insufficient when classes are imbalanced. Use validation
macro-F1 as the primary selection metric, then recall for `Checkmate` and
prediction cost as declared tie-breakers. Record any change to this rule before
examining the affected experiment results.

### Baselines and evaluator checks

- [ ] Evaluate an always-majority-class predictor to expose class imbalance.
- [ ] Verify calculations on a small hand-computed confusion matrix, including
      absent classes, missed classes, and incorrect predictions.
- [ ] Reject prediction count mismatches, unexpected labels, and failed analyzer
      processes; never truncate or silently pair incomplete results.
- [ ] Confirm that one prediction corresponds to each input position in order.
- [ ] Keep labels out of the model inputs.

**Evidence:** a metrics table including baselines, a confusion matrix with raw
counts and row-normalized percentages, and a small documented error analysis.
Inspect examples of `Check`/`Checkmate` confusion and avoid claiming that a few
examples establish a general pattern. Do not present an untrained model's
pipeline-check results as evidence of learned quality.

## 5. Training procedure and encountered problems

### Establish correctness before scaling

- [ ] Document the loss function, compatible output activation, optimizer,
      gradient aggregation, and parameter update rule.
- [ ] Check gradients against finite differences on small networks.
- [ ] Verify expected weight and bias updates and reject invalid parameters.
- [ ] Demonstrate decreasing loss on a small deterministic learning problem.
- [ ] Confirm save/reload preserves predictions and both training save modes
      behave correctly.

These checks support training correctness. They do not replace an independent
measurement of chess classification quality.

### Capture a training history

For each epoch, record training and validation loss, comparable classification
metrics, elapsed time, and the checkpoint or stopping decision. State whether
training loss is accumulated during updates or recomputed on a fixed model at
the end of the epoch: the interpretation differs.

Validation inference must not update weights. Record sample ordering, shuffle
seed, batch handling, and how per-batch losses are aggregated. Aggregate by
sample count when batches have different sizes.

### Experiment and problem journal

For every meaningful failure or unexpected result, record the following in
the experiment journal. Leave fields pending rather than inventing findings.

| Field | Content to record |
| --- | --- |
| Run and context | Run ID, date, commit, configuration, data manifest |
| Observation | Actual error, curve, metric, or runtime behavior |
| Hypothesis | Suspected cause, explicitly separated from the observation |
| Investigation | Small check or controlled experiment used to test it |
| Change | One intervention and its reason |
| Result | Before/after evidence, including unsuccessful attempts |
| Decision | Keep, reject, or investigate further; remaining limitations |

Possible problems to investigate only if observed include numerical instability,
loss stagnation, majority-class collapse, slow epochs, and divergence between
training and validation. Failed runs are part of the record, not successful
measurements.

**Evidence:** learning curves, correctness-check results, an experiment journal,
and a concise explanation of the main problems encountered and resolved.

## 6. Hyperparameters and controlled comparisons

Separate model structure from the learning procedure.

| Family | Candidates to investigate | Evidence expected |
| --- | --- | --- |
| Model | Hidden-layer count and width; supported activation choices | Validation quality versus parameter count and runtime |
| Training | Learning rate, batch size, epoch budget, initialization and shuffle seeds | Convergence, stability, validation quality, and training time |

The current example uses `777 → 128 → 64 → 3`, sigmoid hidden activations,
softmax output, Xavier initialization, learning rate `0.01`, and seed `42`.
This is a starting configuration, not a demonstrated optimum. Epoch and batch
controls still need to be exposed by the training workflow.

### Bounded experiment sequence

1. Train the reference configuration to validate the complete workflow.
2. Compare a smaller architecture with the reference while holding the training
   settings and dataset fixed.
3. On the selected architecture, compare a small declared set of learning
   rates; for example `0.001`, `0.01`, and `0.1` if compatible with the chosen
   loss and update rule. These are proposed trials, not recommendations based
   on results.
4. Compare two batch sizes if supported and time permits.
5. Repeat finalists with the same predefined seeds, ideally three, and report
   individual results plus mean and standard deviation. If only one seed is
   feasible, disclose that variability has not been assessed.

Set a maximum run or time budget before starting. Compare candidates under a
declared common budget and stopping policy; record early-stopped epochs. Change
one factor at a time for an interpretable first comparison. Do not attribute a
gain to architecture if the data or training procedure changed too.

**Evidence:** full attempted settings, not just the winner; validation metrics,
costs, seed variability, and a written selection rationale. A small controlled
comparison is more useful than an unexplained large parameter search.

## 7. Overfitting and prevention

- [ ] Plot training and validation loss against epoch, with the best checkpoint
      and stopping epoch identified.
- [ ] Inspect training/validation metric gaps alongside the loss curves.
- [ ] Implement and document a concrete prevention mechanism, initially
      validation-loss-based early stopping with best-checkpoint restoration.
- [ ] Declare patience, minimum improvement, maximum epochs, and checkpoint
      selection rule before the run.
- [ ] Verify that stopping and restoration actually work and that validation
      samples never contribute gradients.

A lower training loss with deteriorating validation loss suggests overfitting;
it does not prove a single cause. Poor results on both sets may instead suggest
underfitting, a learning problem, or a data problem. A split detects a gap but is
not itself an operational prevention mechanism.

For early stopping, keep the lowest-validation-loss checkpoint within each run.
Compare the resulting candidate models using the declared validation macro-F1
selection rule. Report the difference between the stopping epoch and the
restored checkpoint's epoch.

**Evidence:** curves and a demonstrated stopping/restoration example, with the
policy and its limitations explained.

## 8. Practical alternative and impact assessment

Test at least one alternative that can improve the current approach without
rewriting the project. The first candidate is training-only class balancing,
compared with the unchanged training distribution.

Hold architecture, validation/test data, seeds, optimizer, and stopping policy
fixed. Balancing by repetition changes samples processed per epoch: report
processed samples and elapsed time as well as epochs so the comparison does not
hide extra computation.

Record the hypothesis before testing, then compare macro-F1, per-class recall,
confusions, and cost. Keep the alternative only if the measured tradeoff is
useful; a documented negative result still supports a decision. Other options,
such as a supported initialization or batch strategy, can replace this
experiment when better justified by an observed problem.

**Evidence:** a before/after table, a chart where useful, and an explicit
accept/reject decision grounded in measurements.

## 9. Computational costs

| Cost | Measurement protocol |
| --- | --- |
| Training | Total wall time, epoch times, epochs completed, samples processed |
| Prediction | Fixed file and model; report wall time and positions per second |
| Model | Trainable parameter count and serialized file size |
| Memory, if measured | Peak resident memory and the measurement method |

Record CPU, available memory, operating system, compiler/version, build flags,
and thread count. Repeat short prediction measurements and report the repeat
count and median. State whether loading the model, parsing FEN, and process
startup are included; command-level timing includes all three unless explicitly
isolated. Average time per position from a batch is not an individual latency
distribution.

For dense layers, parameter count is the sum of
`(input_width + 1) * output_width`, including biases. Distinguish measured memory
from estimates. Avoid timing comparisons across different hardware without
disclosing that difference.

**Evidence:** a quality-versus-cost comparison and a reasoned choice of the
delivered model. Parallelization, if attempted later, needs its own baseline,
thread counts, speedup measurements, and correctness checks.

## 10. Reproducibility and final delivery

Each experiment record must contain:

- Run ID and source commit; disclose any uncommitted code changes.
- Exact generation, training, and evaluation commands and tool versions.
- Configuration values, all relevant seeds, and stopping policy.
- Dataset hashes, split identity, class counts, and balancing settings.
- Epoch history, final metrics, costs, failures, and artifact locations.
- Model hash and the checkpoint-selection reason.

Suggested artifacts to create during implementation:

| Artifact | Purpose |
| --- | --- |
| Evaluation script | Compute consistent metrics and export plots/results |
| Run manifest | Preserve commands, settings, environment, and data identity |
| Epoch history | Store numerical learning curves, not only chart images |
| Experiment journal | Preserve observations, interventions, and decisions |
| Final evaluation report | Explain protocol, comparisons, limits, and model choice |
| `my_torch_network…` model | Deliver a trained model usable by the analyzer |

Final delivery checks:

- [ ] Freeze the chosen settings using validation results.
- [ ] Run and record the final test evaluation without retuning on test.
- [ ] Commit and push the selected trained model with the required filename.
- [ ] Load that exact delivered file and verify ordered predictions.
- [ ] Document a working prediction command and reproducible training commands.
- [ ] Preserve scripts, configurations, data references, numerical results,
      plots, and access to training data for the defense.
- [ ] Ensure plots have titles, labeled axes, units, legends where needed, and
      dataset/run identities. Keep the underlying numerical data.

## 11. Defense evidence checklist

| Assessment area | Evidence to prepare |
| --- | --- |
| Choice justification | Hypotheses, controlled comparisons, final rationale |
| Reusable save file | Delivered trained model and save/load/predict demonstration |
| Generation and module independence | Generated concrete models; neural code independent of chess encoding |
| Algorithm hyperparameters | Learning-rate/batch/epoch trials and convergence evidence |
| Model hyperparameters | Architecture comparison with quality and cost |
| Alternative proposal | Measured practical alternative and accept/reject decision |
| Overfitting | Independent validation, curves, operational prevention and checks |
| Documentation | Reproduction instructions, numerical results, readable plots |
| Choice impact | Quality, runtime, model size, data permissions, predictive limits |
| Optional parallelization | Only claim if implemented and measured |

Discuss limitations explicitly: synthetic or narrow data coverage, rare classes,
label reliability, unmeasured seed variability, source/license uncertainty, and
the difference between classifying a position and playing chess. Softmax scores
are not established confidence probabilities without calibration evaluation.

The supplied assessment is a snapshot of an earlier state. Use these artifacts
to demonstrate the current implementation; do not assume a new score or claim
credit for planned work. Update this plan as evidence is produced and keep the
final report focused on actual findings.

## References

- [Project architecture](architecture.md)
- [Input and prediction contract](data-encoding.md)
- [Class and output contract](output-encoding.md)
- The project brief and the supplied defense assessment define delivery and
  presentation expectations; retain them alongside the defense preparation.
