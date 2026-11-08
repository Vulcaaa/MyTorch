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
| Algorithm hyperparameters | Explain learning rate, batch size, epochs, optimizer and stopping rules | [Training](training.md), protocols in `configs/protocols/`, [reports](report/README.md) |
| Model hyperparameters | Compare widths, hidden activations and initialization with a declared protocol | [Configuration](architecture.md#network-configuration), architecture/activation reports in the [report index](report/README.md) |
| Overfitting | Separate updates from validation, inspect both loss curves, restore the best checkpoint | [Stopping policy](training.md#3-checkpoints-and-stopping), [comparison protocol](evaluation.md#comparison-protocol-v1) |
| Choice justification | Explain the 777 inputs, class mapping and parameter choices using evidence | [Datasets](datasets.md), [outputs](output-encoding.md), experiment reports |
| Alternative proposal | Compare a concrete alternative, such as augmentation, ReLU/Adam or BLAS, against a stated baseline | [Data transformations](datasets.md#color-mirror-augmentation), [reports](report/README.md), [CPU benchmark](../benchmarks/cpu_blas/README.md) |
| Choice impact assessment | Assess quality by class, runtime, memory/model size, portability and data limitations | [Evaluation](evaluation.md), [data provenance](datasets.md#preparation-report-chess-v1), recorded benchmark conditions |
| Documentation | Provide repeatable commands, visual benchmarks, retained failures and conclusions | [Guides](../README.md#documentation), [run records](evaluation.md#where-files-belong), [logs](logging.md) |
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

The current guides describe implemented behavior. Selecting the final model and
writing the cross-experiment synthesis are separate closing tasks; the existing
[dated reports](report/README.md) remain the evidence for completed experiments.
