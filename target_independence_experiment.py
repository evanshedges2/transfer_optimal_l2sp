"""
Target-Side Independence Ablation (A3)
=======================================

Tests the theoretical prediction that in the isotropic regime, whether transfer
helps (and the ranking of source weight decays for transfer) is independent of
target sample size.

We sweep target_fraction across {1%, 5%, 10%, 20%, 50%} for MNIST and CIFAR-10
and show that the transfer-optimal source weight decay ranking is preserved.

Usage:
    python target_independence_experiment.py
    python target_independence_experiment.py --experiments mnist --seeds 5
    python target_independence_experiment.py --experiments mnist cifar10 --fractions 0.01 0.05 0.1 0.2 0.5

Author: C. Evans Hedges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
from tqdm import tqdm

from unified_transfer_experiments import (
    set_seed, get_device, RemapLabels, MLP, SmallCNN,
    train_model, evaluate_model, WEIGHT_DECAYS
)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.50]

CONFIG = {
    'mnist': {
        'weight_decays': WEIGHT_DECAYS,
        'source_epochs': 10,
        'transfer_epochs': 5,
        'lr': 0.001,
        'batch_size': 256,
        'n_seeds': 10,
        'transfer_wd': 1e-4,
    },
    'cifar10': {
        'weight_decays': WEIGHT_DECAYS,
        'source_epochs': 15,
        'transfer_epochs': 8,
        'lr': 0.001,
        'batch_size': 128,
        'n_seeds': 10,
        'transfer_wd': 1e-4,
    },
}

# =============================================================================
# Data Loading (with variable target fraction)
# =============================================================================

def load_mnist_split(target_fraction, seed):
    """Load MNIST Round vs Angular split with specified target fraction."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(root='./data/MNIST', train=True,
                                           download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data/MNIST', train=False,
                                          download=True, transform=transform)

    def split(dataset, digits):
        idx = [i for i, (_, l) in enumerate(dataset) if l in digits]
        return Subset(dataset, idx)

    src_digits, src_map = [0, 3, 6, 8, 9], {0:0, 3:1, 6:2, 8:3, 9:4}
    tgt_digits, tgt_map = [1, 2, 4, 5, 7], {1:0, 2:1, 4:2, 5:3, 7:4}

    train_source = RemapLabels(split(train_ds, src_digits), src_map)
    test_source = RemapLabels(split(test_ds, src_digits), src_map)
    train_tgt_full = RemapLabels(split(train_ds, tgt_digits), tgt_map)
    test_target = RemapLabels(split(test_ds, tgt_digits), tgt_map)

    set_seed(seed)
    n = max(1, int(len(train_tgt_full) * target_fraction))
    idx = torch.randperm(len(train_tgt_full))[:n].tolist()
    train_target = Subset(train_tgt_full, idx)

    return train_source, test_source, train_target, test_target


def load_cifar10_split(target_fraction, seed):
    """Load CIFAR-10 Animals vs Vehicles+ split with specified target fraction."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True,
                                              download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False,
                                             download=True, transform=transform)

    def split(dataset, classes):
        idx = [i for i, (_, l) in enumerate(dataset) if l in classes]
        return Subset(dataset, idx)

    src_cls, src_map = [2,3,4,5,6], {2:0, 3:1, 4:2, 5:3, 6:4}
    tgt_cls, tgt_map = [0,1,7,8,9], {0:0, 1:1, 7:2, 8:3, 9:4}

    train_source = RemapLabels(split(train_ds, src_cls), src_map)
    test_source = RemapLabels(split(test_ds, src_cls), src_map)
    train_tgt_full = RemapLabels(split(train_ds, tgt_cls), tgt_map)
    test_target = RemapLabels(split(test_ds, tgt_cls), tgt_map)

    set_seed(seed)
    n = max(1, int(len(train_tgt_full) * target_fraction))
    idx = torch.randperm(len(train_tgt_full))[:n].tolist()
    train_target = Subset(train_tgt_full, idx)

    return train_source, test_source, train_target, test_target

# =============================================================================
# Experiment Runner
# =============================================================================

def run_target_independence(experiment, fractions, device, output_dir):
    """Sweep target_fraction for one experiment."""
    cfg = CONFIG[experiment]
    model_cls = MLP if experiment == 'mnist' else SmallCNN
    load_fn = load_mnist_split if experiment == 'mnist' else load_cifar10_split

    print(f"\n{'='*70}")
    print(f"Target Independence: {experiment.upper()} | fractions = {fractions}")
    print(f"{'='*70}")

    all_results = []

    for frac in fractions:
        print(f"\n--- target_fraction = {frac:.0%} ---")
        for seed in tqdm(range(cfg['n_seeds']), desc=f'Seeds (frac={frac})'):
            set_seed(seed)
            train_source, test_source, train_target, test_target = load_fn(frac, seed)

            scratch_acc = None
            for wd in cfg['weight_decays']:
                # Train source
                source_model = model_cls()
                source_model = train_model(source_model, train_source, cfg['source_epochs'],
                                           cfg['lr'], wd, cfg['batch_size'], device)
                source_acc = evaluate_model(source_model, test_source, cfg['batch_size'], device)

                # Transfer
                transfer_model = model_cls()
                transfer_model.load_state_dict(source_model.state_dict())
                transfer_model = train_model(transfer_model, train_target, cfg['transfer_epochs'],
                                             cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device)
                transfer_acc = evaluate_model(transfer_model, test_target, cfg['batch_size'], device)

                # Scratch baseline
                if scratch_acc is None:
                    scratch_model = model_cls()
                    scratch_model = train_model(scratch_model, train_target, cfg['transfer_epochs'],
                                                cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device)
                    scratch_acc = evaluate_model(scratch_model, test_target, cfg['batch_size'], device)

                all_results.append({
                    'target_fraction': frac,
                    'seed': seed,
                    'weight_decay': wd,
                    'source_acc': source_acc,
                    'transfer_acc': transfer_acc,
                    'scratch_acc': scratch_acc,
                    'transfer_benefit': transfer_acc - scratch_acc,
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, f'{experiment}_target_independence_raw.csv'), index=False)

    summary = df.groupby(['target_fraction', 'weight_decay']).agg({
        'source_acc': ['mean', 'std'],
        'transfer_acc': ['mean', 'std'],
        'transfer_benefit': ['mean', 'std'],
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(os.path.join(output_dir, f'{experiment}_target_independence_summary.csv'),
                   index=False)

    return df, summary

# =============================================================================
# Plotting
# =============================================================================

def plot_target_independence(experiment, summary_df, output_dir):
    """Plot transfer accuracy vs source WD for different target fractions."""
    fractions = sorted(summary_df['target_fraction'].unique())
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(fractions)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for frac, color in zip(fractions, colors):
        sub = summary_df[summary_df['target_fraction'] == frac]
        wds = sub['weight_decay'].values
        wds_plot = np.array([max(w, 1e-7) for w in wds])

        ax1.plot(wds_plot, sub['transfer_acc_mean'], 'o-', color=color,
                 label=f'{frac:.0%}', linewidth=1.5, markersize=4)
        ax1.fill_between(wds_plot,
                         sub['transfer_acc_mean'] - sub['transfer_acc_std'],
                         sub['transfer_acc_mean'] + sub['transfer_acc_std'],
                         alpha=0.10, color=color)

        # Mark transfer-optimal WD
        idx_opt = sub['transfer_acc_mean'].idxmax()
        wd_opt = sub.loc[idx_opt, 'weight_decay']
        ax1.axvline(max(wd_opt, 1e-7), color=color, linestyle=':', alpha=0.5)

    ax1.set_xscale('log')
    ax1.set_xlabel('Source Weight Decay', fontsize=11)
    ax1.set_ylabel('Transfer Accuracy (%)', fontsize=11)
    ax1.set_title(f'{experiment.upper()}: Transfer Acc vs Source WD', fontsize=12,
                  fontweight='bold')
    ax1.legend(title='Target %', fontsize=8, title_fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: transfer-optimal WD vs fraction
    opt_wds = []
    for frac in fractions:
        sub = summary_df[summary_df['target_fraction'] == frac]
        idx_opt = sub['transfer_acc_mean'].idxmax()
        opt_wds.append(max(sub.loc[idx_opt, 'weight_decay'], 1e-7))

    ax2.plot(fractions, opt_wds, 'o-', color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('Target Fraction', fontsize=11)
    ax2.set_ylabel('Transfer-Optimal Source WD', fontsize=11)
    ax2.set_yscale('log')
    ax2.set_title(f'{experiment.upper()}: Optimal WD Stability', fontsize=12,
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Target-Side Independence of Transfer-Optimal Source Regularization',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{experiment}_target_independence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Target Independence Ablation')
    parser.add_argument('--experiments', nargs='+', default=['mnist', 'cifar10'],
                        choices=['mnist', 'cifar10'])
    parser.add_argument('--fractions', nargs='+', type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seeds', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    if args.seeds:
        for cfg in CONFIG.values():
            cfg['n_seeds'] = args.seeds

    start_time = time.time()

    for exp in args.experiments:
        df, summary = run_target_independence(exp, args.fractions, device, args.output_dir)
        plot_target_independence(exp, summary, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
