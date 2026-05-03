# Source-Optimal Training is Transfer-Suboptimal

Code for reproducing the experiments in "Source-Optimal Training is Transfer-Suboptimal" published in Transactions on Machine Learning Research (TMLR), May 2026. Paper available at https://openreview.net/forum?id=CMlpokFXfA

This repository contains three scripts. `simple_phase_experiment.py` validates the alignment-dependent phase transition in synthetic ridge regression, including sweeps over overparameterization ratios and non-isotropic covariance structures (Figures 2a, 2b). `unified_transfer_experiments.py` runs the nonlinear transfer learning experiments on MNIST, CIFAR-10, and 20 Newsgroups with both standard and explicit L2-SP fine-tuning, and produces the source-vs-transfer accuracy curves and parameter distance plots (Figures 3, 4). `target_independence_experiment.py` runs the target-side independence ablation on MNIST and CIFAR-10 (Figure 5).

Requirements: numpy, pandas, matplotlib, scipy, torch, torchvision, scikit-learn, tqdm.
