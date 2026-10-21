# CALR Research Reproduction

This project contains a source-informed reimplementation of four experiments from Gayan Maduranga's CALR repository. It was developed with AI assistance. The original method and reference code are credited below; the original MIT license is retained.

## Scope

The implemented experiments are Himmelblau optimization, toy binary classification, tiny-MNIST gradient/Hessian comparison, and full MNIST optimizer comparison. This is a reproduction of these supplied experiments, not every experiment in the paper. CIFAR-10, Rosenbrock, saddle-point experiments, and the full cosine-annealing study are not included.

## Setup and execution

The recorded run used Python 3.13.3, CPU execution with one thread, and the package versions in requirements-lock.txt. From the repository root, run:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m reproduction.run
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. MNIST downloads automatically into data/. Internet access is required for the first download. CALR_DATA_DIR can select another dataset location. Each execution writes a new results directory. No sibling project is required.

The lock file records the original environment. Package availability can vary across platforms. requirements.txt lists direct dependencies without version pins; using different versions may change numerical results.

Defaults are 350 Himmelblau steps, 500 toy epochs, 10 tiny-MNIST epochs, and 50 epochs per optimizer for full MNIST. Use `python -m reproduction.run --help` for options.

## Included files

- reproduction/optimizers.py: optimizer implementations.
- reproduction/experiments.py: models, datasets, and training loops.
- reproduction/run.py: execution, plots, metrics, and run metadata.
- reproduction/compare.py: numerical comparison with reference outputs.
- results/reimplementation-20261019-112256/: completed reproduction run, including CSV histories, JSON metrics, plots, and status.json.
- reference-results/small-experiments/: captured numerical outputs from rerunning the original small-experiment scripts.
- reference-results/mnist/: original-code MNIST summary, printed epoch log, and plots.
- reference-results/README.md: provenance and comparison limitations.

## Verify the supplied results

After installing dependencies, run from the repository root:

```bash
python -m reproduction.compare --new results/reimplementation-20261019-112256 --reference reference-results/small-experiments --original-mnist reference-results/mnist
```

This writes comparison.json into the selected reproduction run directory. The supplied outputs pass 289 of 289 comparison groups, with exact agreement at the recorded precision. Each group may contain multiple values. Small-task histories and final coordinates are checked, as are full-precision MNIST summaries and printed epoch metrics. Original epoch losses were printed to four decimal places and accuracies to two. Original batch losses and corrected Himmelblau trajectories are not compared.

To compare a fresh run, replace the --new path with its generated directory. Exact agreement is not guaranteed on a different environment. The comparison checks implementation agreement with recorded reference outputs; it does not establish that every published result is reproduced.

## Interpretation and limitations

Models initialize sequentially, matching the original scripts. Himmelblau CALR uses the original Adam-style update, while MNIST uses gradient-based curvature scaling. The implementation copies Himmelblau coordinates when recording positions, correcting the original overwritten trajectory history. Consequently, trajectory images can differ even when checked losses and final coordinates match.

The main MNIST reproduction obtained best test accuracies of 96.15% for Adam, 95.97% for Nadam, 93.66% for RMSProp, 98.37% for SGD, and 98.21% for CALR. These match the included original-code run. They are not all identical to the published paper's values. There is one recorded seed/configuration, and no statistical significance claim is made.

The environment, downloaded data, and caches are excluded from version control. Run metadata preserves its original timestamps and local output path for traceability; that path is historical and is not required to execute the project.

## Sources

- Paper: *Curvature-Adaptive Learning Rate Optimizer: Theoretical Insights and Empirical Evaluation on Neural Network Training*. https://doi.org/10.32473/flairs.38.1.138986
- Original repository: https://github.com/Gayan225/CALR-Optimizer
