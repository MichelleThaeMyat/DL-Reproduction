# Controlled CALR Improvement Study

This project extends the existing CALR reproduction with an empirical evaluation of smoothed gradient magnitudes and cosine learning-rate decay. It includes ablations and matched SGD. These are proposed modifications, not a claim of a newly invented optimizer or guaranteed improvement. Code was developed with AI assistance. Original CALR source attribution is retained in LICENSE and reference_optimizers.py.

## Research question
Does replacing the instantaneous absolute gradient with a bias-corrected exponential moving average, together with cosine decay, improve validation-selected MNIST test performance under matched initialization and batch order?

## Setup
Use Python 3.13. From this folder:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m calr_improvement.test_optimizer
python -m calr_improvement.run --epochs 30 --seeds 42 43 44 --output results/rerun
python -m calr_improvement.analyze --results results/rerun
```

On Windows, activate with `.venv\Scripts\activate`. The lock file records the tested environment. Package availability and exact floating-point agreement can vary across platforms. `requirements.txt` lists the direct dependencies if version pins are unavailable, but changing versions is a departure from the recorded environment.

MNIST downloads automatically into `data/`. To reuse an existing copy, append `--data /absolute/path/to/data` to the run command, or set CALR_DATA_DIR. No sibling project is needed. The output directory must be empty or new to prevent overwriting completed evidence. A full rerun trains 15 models, each for 30 epochs. This can take several minutes or longer depending on hardware.

## Fixed protocol

- Dataset: 50,000 training images and 10,000 validation images from the official MNIST training set. Split seed 12345. Official 10,000-image test set remains separate.
- Model: 784, 256, 128, 10 fully connected units, ReLU, no dropout, no weight decay.
- Input: pixel / 255, then normalization with mean and standard deviation 0.5. Float32 CPU tensors, one thread, deterministic PyTorch operations.
- Training: cross-entropy, batch 64, 30 epochs. Preloaded tensors avoid repeated image conversion. Metrics use sample-weighted loss, including the shorter final batch.
- For each seed, every method loads identical initial weights and uses a separately reset shuffle generator with seed + 10000. Initial weight hashes and shuffle seeds are recorded.
- Methods: CALR, EMA-CALR, Cosine-CALR, EMA-Cosine-CALR, SGD with momentum 0.9.
- CALR: learning rate 0.01, alpha 0.1, beta1 0.9, epsilon 1e-8. EMA rho 0.9. Cosine rate decreases from 0.01 at epoch 1 to 0.001 at epoch 30. There are no restarts or cumulative multiplications.
- Checkpoint: lowest validation cross-entropy, with earliest epoch breaking ties. Every method trains the full budget. Test evaluation starts after all 15 training runs finish. No tuning or method selection uses these test scores.
- Primary contrast: EMA-Cosine-CALR minus CALR in paired test accuracy. Separate variants isolate smoothing and scheduling. SGD is a contextual baseline, not a hyperparameter-tuned optimum.

## Evidence

`results/controlled/` contains the measured run, histories, summaries, analysis, figures, split indices, and selected model checkpoints. `config.json` records software versions, protocol and source hashes. `runs.csv` is the compact per-run result table. `analysis.json` includes sample standard deviations and paired differences. The 95% t interval uses only three seed pairs, so it is descriptive and assumption-sensitive, not proof of a general advantage. All seeds share one data split.

## Difference from the earlier RR

The earlier run used 60,000 training images, 50 epochs, sequential model initializations and best test scores across epochs. This controlled study uses validation selection, 50,000 training images, 30 epochs and matched seeds. Its values cannot be directly subtracted from the old report as a method effect. The valid method comparisons are within this controlled study.

## Sources

- CALR paper: https://doi.org/10.32473/flairs.38.1.138986
- Original code: https://github.com/Gayan225/CALR-Optimizer
- Earlier reproduction: https://github.com/MichelleThaeMyat/DL-Reproduction
- Cosine annealing background: https://arxiv.org/abs/1608.03983

The original paper already discusses cosine variants. This project's contribution is the implemented magnitude-smoothing variant and the controlled ablation study, not novelty of cosine scheduling. The gradient-magnitude scaling remains a first-order heuristic and does not calculate a Hessian.

## Measured outcome

All 15 runs completed. CALR averaged 97.8433% test accuracy and 0.073587 test loss. EMA-Cosine-CALR averaged 97.7800% and 0.070917. The combined proposal reduced mean loss by about 3.63% but reduced accuracy in every seed pair. It did not improve the prespecified primary accuracy outcome. EMA alone changed mean accuracy by only +0.0067 percentage points. These are conditional results from three seeds, not a general superiority claim.

## Verify the included checkpoints

```bash
python -m calr_improvement.verify --data data
python -m calr_improvement.diagnose --data data
```

The verifier checks the included training-source hashes, matching initialization and shuffle metadata, validation selections, and test metrics recomputed from all 15 saved checkpoints. The diagnostic measures gradients on the first matched training batch. It does not choose hyperparameters or retrain models. The original config retains its historical dataset path; --data overrides that location on another computer. Checkpoint loading uses PyTorch weights_only=True.

The report is kept locally and excluded from this repository. The protocol above explains why these scores cannot be compared directly with the earlier RR's best-test scores.
