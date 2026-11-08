# Project requirements and defense checklist

Use this page to locate evidence, not as a predicted grade. The project subject
v2.6.0 requires a supervised-learning solution, a generic neural module, an
analyzer, reproducible training and a delivered pre-trained network. The defense
competencies also ask for justified choices, alternatives and measured impact.

## Requirements and evidence

| Requirement / competency | What to demonstrate | Where to look |
| --- | --- | --- |
| Generating neural networks | Create a concrete network from configurable layers, initialization and seed | [Architecture](architecture.md#network-configuration), generator example in the [README](../README.md) |
| Save file | Save, load, predict, then continue training; preserve Adam state where applicable | [Model format](architecture.md#saved-models-and-continuation), `tests/ModelSerializer.cpp`, `tests/Adam.cpp` |
| Generic my_torch module | Numeric network independent of FEN; compiled computation plus Python tooling | [Component boundaries](architecture.md#components-and-data-flow), `src/neural/` |
| Algorithm hyperparameters | Explain learning rate, batch size, epochs, optimizer and stopping rules | [Measured setting choices](hyperparameters.md), [training](training.md), protocols in `configs/protocols/` |
| Model hyperparameters | Compare widths, hidden activations and initialization with a declared protocol | [Model choices and alternatives](hyperparameters.md), [configuration](architecture.md#network-configuration) |
| Overfitting | Separate updates from validation, inspect both loss curves, restore the best checkpoint | [Stopping policy](training.md#3-checkpoints-and-stopping), [comparison protocol](evaluation.md#comparison-protocol-v1) |
| Choice justification | Explain the 777 inputs, class mapping and parameter choices using evidence | [Datasets](datasets.md), [outputs](output-encoding.md), experiment reports |
| Alternative proposal | Compare a concrete alternative, such as augmentation, ReLU/Adam or BLAS, against a stated baseline | [Data transformations](datasets.md#color-mirror-augmentation), [reports](report/README.md), [CPU benchmark](../benchmarks/cpu_blas/README.md) |
| Choice impact assessment | Assess quality by class, runtime, memory/model size, portability and data limitations | [Evaluation](evaluation.md), [data provenance](datasets.md#preparation-report-chess-v1), recorded benchmark conditions |
| Data rights and predictive risks | Identify Epitech as the dataset provider; explain scope of use, prediction errors and generalization limits | [Data rights and predictive limitations](datasets.md#8-data-rights-and-predictive-limitations) |
| Documentation | Provide repeatable commands, visual benchmarks, retained failures and conclusions | [Trial/failure log](experiment-decisions.md), [reports](report/README.md), [run records](evaluation.md#where-files-belong) |
| Parallelization / other bonuses | Show the actual backend, thread counts and measured impact; distinguish optional features from required behavior | [CPU backend](training.md#cpu-blas-backend), [measurements](../benchmarks/cpu_blas/results/linux-2026-11-07/README.md), CI workflow |

Do not equate more threads with a speedup, a single seed with robustness, or a
working training pipeline with good predictive quality. Keep negative results
and explain what they ruled out. Bonus credit remains the jury's decision.

## Delivery and demonstration

- Build through `make`; retain `re`, `clean` and `fclean`. No neural-network framework.
- Demonstrate generation → training → saving → reloading → ordered predictions.
- Check errors on stderr with exit code `84`, success `0`, and both training save modes.
- Push a loadable pre-trained file whose name starts with `my_torch_network`.
- Retain every training script, protocol and dataset; large datasets need not be in Git but must be available at the defense.
- Verify the chosen checkpoint's data, settings and fingerprints. Present train/validation loss curves and per-class metrics alongside accuracy.
- Freeze final choices before final test evaluation. Historical reference test results already exist; do not describe that split as entirely unseen.
- Check the delivered model and documented commands from a fresh checkout with only the declared dependencies.

## Remaining closing work

- The delivered checkpoint is chosen and tested once (93.66% accuracy,
  [provenance](report/2026-11-08-mirror-view-ensemble.md)); update the artifact and
  its provenance together if it changes again.
- Verify model loading, ordered predictions and documented reproduction steps
  from a fresh checkout; keep the required local data available for the defense.
- Verify CI covers explicit OpenBLAS and scalar builds; backend auto-detection
  alone does not establish coverage of both paths.
- Set the evaluation notebook to the chosen run and label its dataset role;
  export the selected model's figures and rehearse the short demonstration.
- If reporting inference latency or peak memory, measure them separately;
  parameter count, file size and training time do not establish those costs.

The [setting synthesis](hyperparameters.md) and [decision log](experiment-decisions.md)
cover completed work; the [dated reports](report/README.md) remain the evidence.
A release can package the final artifact but is optional and does not replace
these delivery checks.
