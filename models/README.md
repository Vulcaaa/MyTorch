# Shared candidate model

[`large-relu-adam-seed46-lr0.001.nn`](large-relu-adam-seed46-lr0.001.nn) is the
selected epoch-9 checkpoint from the 29-epoch capacity run (15 min 59 s).

- Network: 777 → 256 → 128 → 3; ReLU/softmax, Xavier, Adam LR 0.001, seed 46.
- Data: 1,141,143 color-mirror-augmented training positions; 71,555 validation positions.
- Validation: **89.25% accuracy / 88.58% macro F1** (one seed, reused validation).
- SHA-256: `55944eb130c71cbcee29fddd6dc1e2239848fd9ec4697f5fc8017f6f8f0d911b`.
- Exact training checkpoint, including Adam state; requires the Adam-capable code in this branch.

From the repository root:

```sh
make
./my_torch_analyzer --predict models/large-relu-adam-seed46-lr0.001.nn positions.fen
```

[Training protocol](../configs/protocols/train-large-relu-adam.yml) ·
[Comparison report](../docs/report/2026-11-08-adam-optimizer.md) ·
[Portable metrics and provenance](../docs/report/data/2026-11-08-adam-optimizer.json).
The original local run is `results/training/2026-11-08-large-relu-adam-01/`.

This candidate is shared for comparison; the delivered `../my_torch_network.nn`
remains unchanged. Other generated and historical models remain ignored locally:
`relu-adam-seed46-lr0.001.nn` (small baseline),
`legacy-sigmoid-seed43-epoch100.nn`, and `legacy-sigmoid-seed42-untrained.nn`.
