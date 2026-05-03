"""
Synthetic Phase Transition Experiment
=====================================

Validates the alignment-dependent phase transition (Corollary 3.9) in linear
ridge regression. Supports:
- Multiple overparameterization ratios gamma_0 (A4 ablation)
- Non-isotropic covariance with power-law spectrum (A5 ablation)

Usage:
    python simple_phase_experiment.py                          # default isotropic, gamma=2
    python simple_phase_experiment.py --gammas 1.5 2.0 3.0 5.0 # gamma sweep
    python simple_phase_experiment.py --covariance power_law --alphas 0.5 1.0 2.0

Author: C. Evans Hedges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import argparse
import os


def generate_covariance(p, cov_type='isotropic', alpha=1.0):
    """Generate a p x p covariance matrix.

    cov_type='isotropic': identity
    cov_type='power_law': diagonal with eigenvalues k^{-alpha}, k=1..p
    """
    if cov_type == 'isotropic':
        return np.eye(p)

    eigenvalues = np.arange(1, p + 1, dtype=float) ** (-alpha)
    eigenvalues /= eigenvalues.mean()  # normalize so trace/p = 1
    return np.diag(eigenvalues)


def generate_data(n, p, sigma_sq, w, Sigma=None):
    """Generate X, y = X w + noise with optional covariance Sigma."""
    Z = np.random.randn(n, p)
    if Sigma is not None and not np.allclose(Sigma, np.eye(p)):
        sqrt_Sigma = np.diag(np.sqrt(np.diag(Sigma)))  # diagonal case
        X = Z @ sqrt_Sigma
    else:
        X = Z
    noise = np.sqrt(sigma_sq) * np.random.randn(n)
    y = X @ w + noise
    return X, y


def solve_ridge(X, y, lam):
    n, p = X.shape
    K = X @ X.T
    K_chol = cho_factor(K + lam * n * np.eye(n), lower=True)
    alpha = cho_solve(K_chol, y)
    w_hat = X.T @ alpha
    return w_hat, K_chol


def solve_transfer(X_target, y_target, w_source, lam, alpha_scratch=None, K_chol=None):
    n, p = X_target.shape
    lam_scaled = lam * n

    if K_chol is None:
        K = X_target @ X_target.T
        K_chol = cho_factor(K + lam_scaled * np.eye(n), lower=True)

    if alpha_scratch is None:
        alpha_scratch = cho_solve(K_chol, y_target)

    Xw_source = X_target @ w_source
    beta = cho_solve(K_chol, Xw_source)
    w_hat = X_target.T @ (alpha_scratch - beta) + w_source
    return w_hat


def compute_risk(w_hat, w_true, Sigma=None):
    diff = w_hat - w_true
    if Sigma is not None and not np.allclose(Sigma, np.eye(len(diff))):
        return diff @ Sigma @ diff
    return np.dot(diff, diff)


def run_single_experiment(p, gamma, n_target, sigma_source_sq, sigma_target_sq,
                          rhos, lambdas, lambda_target, n_seeds,
                          Sigma_source=None, Sigma_target=None):
    """Run the phase transition experiment for one (gamma, covariance) setting."""
    n_source = int(p / gamma)
    all_results = []

    for seed in range(n_seeds):
        np.random.seed(42 + seed)

        w0 = np.random.randn(p)
        w0 = w0 / np.linalg.norm(w0)

        X0, y0 = generate_data(n_source, p, sigma_source_sq, w0, Sigma_source)

        source_models = []
        best_source_risk = float('inf')
        best_lambda_source = None

        for lam in lambdas:
            w0_hat, _ = solve_ridge(X0, y0, lam)
            risk = compute_risk(w0_hat, w0, Sigma_source)
            source_models.append((lam, w0_hat))
            if risk < best_source_risk:
                best_source_risk = risk
                best_lambda_source = lam

        for rho in rhos:
            w1 = rho * w0

            X1, y1 = generate_data(n_target, p, sigma_target_sq, w1, Sigma_target)

            K1 = X1 @ X1.T
            K1_chol = cho_factor(K1 + lambda_target * n_target * np.eye(n_target), lower=True)
            alpha_scratch = cho_solve(K1_chol, y1)

            best_transfer_risk = float('inf')
            best_lambda_transfer = None

            for lam_source, w0_hat in source_models:
                w1_hat = solve_transfer(X1, y1, w0_hat, lambda_target,
                                        alpha_scratch, K1_chol)
                risk = compute_risk(w1_hat, w1, Sigma_target)

                if risk < best_transfer_risk:
                    best_transfer_risk = risk
                    best_lambda_transfer = lam_source

            ratio = best_lambda_transfer / best_lambda_source

            all_results.append({
                'seed': seed,
                'rho': rho,
                'ratio': ratio,
                'lambda_source': best_lambda_source,
                'lambda_transfer': best_lambda_transfer
            })

    return pd.DataFrame(all_results)


def summarize(df):
    """Compute mean, std, 95% CI by rho."""
    summary = df.groupby('rho')['ratio'].agg(['mean', 'std', 'count']).reset_index()
    summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])
    return summary


# =============================================================================
# Plotting
# =============================================================================

def plot_gamma_sweep(all_summaries, output_dir):
    """Plot phase transition curves for multiple gamma values (A4)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_summaries)))

    for (gamma, summary), color in zip(all_summaries.items(), colors):
        ax.plot(summary['rho'], summary['mean'], marker='o', linestyle='-',
                color=color, label=rf'$\gamma_0 = {gamma}$', markersize=5)
        ax.fill_between(summary['rho'],
                        summary['mean'] - summary['ci'],
                        summary['mean'] + summary['ci'],
                        color=color, alpha=0.15)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Source Optimal')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect Alignment')

    ax.text(0.7, max(ax.get_ylim()[1] * 0.8, 1.3),
            'Over-Regularization\n(Imperfect Alignment)',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(1.3, min(ax.get_ylim()[0] * 1.2, 0.7),
            'Under-Regularization\n(Super-Alignment)',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel(r'Alignment ($\rho = \langle w_0, w_1 \rangle / \|w_0\|^2$)', fontsize=12)
    ax.set_ylabel(r'Optimal Regularization Ratio ($\lambda_{TL}^* / \lambda_{S}^*$)', fontsize=12)
    ax.set_title('Phase Transition Across Overparameterization Levels', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'phase_transition_gamma_sweep.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved gamma sweep plot to {path}")


def plot_covariance_sweep(all_summaries, output_dir):
    """Plot phase transition curves for non-isotropic covariance (A5)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(all_summaries)))

    for (label, summary), color in zip(all_summaries.items(), colors):
        ax.plot(summary['rho'], summary['mean'], marker='s', linestyle='-',
                color=color, label=label, markersize=5)
        ax.fill_between(summary['rho'],
                        summary['mean'] - summary['ci'],
                        summary['mean'] + summary['ci'],
                        color=color, alpha=0.15)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Source Optimal')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect Alignment')

    ax.set_xlabel(r'Alignment ($\rho = \langle w_0, w_1 \rangle / \|w_0\|^2$)', fontsize=12)
    ax.set_ylabel(r'Optimal Regularization Ratio ($\lambda_{TL}^* / \lambda_{S}^*$)', fontsize=12)
    ax.set_title('Phase Transition with Non-Isotropic Covariance', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'phase_transition_covariance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved covariance sweep plot to {path}")


def plot_single(summary, output_dir, suffix=''):
    """Plot a single phase transition curve (backward-compatible)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(summary['rho'], summary['mean'], marker='o', linestyle='-',
            color='blue', label='Mean Ratio')
    ax.fill_between(summary['rho'],
                    summary['mean'] - summary['ci'],
                    summary['mean'] + summary['ci'],
                    color='blue', alpha=0.2, label='95% CI')

    ax.axhline(1.0, color='red', linestyle='--', label='Source Optimal')
    ax.axvline(1.0, color='gray', linestyle=':', label='Perfect Alignment')

    ax.text(0.7, 1.5, 'Over-Regularization Regime\n(Imperfect Alignment)',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))
    ax.text(1.3, 0.5, 'Under-Regularization Regime\n(Super-Alignment)',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Alignment ($\rho = \langle w_0, w_1 \rangle / \|w_0\|^2$)')
    ax.set_ylabel(r'Optimal Regularization Ratio ($\lambda_{TL}^* / \lambda_{S}^*$)')
    ax.set_title('Phase Transition: Transfer-Optimal Regularization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'phase_boundary_plot{suffix}.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Synthetic Phase Transition Experiment')
    parser.add_argument('--p', type=int, default=500, help='Dimensionality')
    parser.add_argument('--gammas', nargs='+', type=float, default=[2.0],
                        help='Overparameterization ratios p/n_source')
    parser.add_argument('--n_target', type=int, default=50)
    parser.add_argument('--sigma_source', type=float, default=1.0)
    parser.add_argument('--sigma_target', type=float, default=0.1)
    parser.add_argument('--n_rhos', type=int, default=21,
                        help='Number of alignment values in [0.5, 1.5]')
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--lambda_target', type=float, default=0.1)
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--covariance', choices=['isotropic', 'power_law'],
                        default='isotropic')
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.0],
                        help='Power-law exponents (only used with --covariance power_law)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results and plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rhos = np.linspace(0.5, 1.5, args.n_rhos)
    lambdas = np.logspace(-2, 2, args.n_lambdas)

    if args.covariance == 'isotropic' and len(args.gammas) > 1:
        # A4: gamma sweep
        print(f"Running gamma sweep: {args.gammas}")
        all_summaries = {}
        all_raw = []

        for gamma in args.gammas:
            print(f"\n--- gamma = {gamma} ---")
            df = run_single_experiment(
                p=args.p, gamma=gamma, n_target=args.n_target,
                sigma_source_sq=args.sigma_source**2,
                sigma_target_sq=args.sigma_target**2,
                rhos=rhos, lambdas=lambdas, lambda_target=args.lambda_target,
                n_seeds=args.n_seeds)
            df['gamma'] = gamma
            all_raw.append(df)
            all_summaries[gamma] = summarize(df)

        pd.concat(all_raw).to_csv(
            os.path.join(args.output_dir, 'phase_transition_gamma_sweep.csv'), index=False)
        plot_gamma_sweep(all_summaries, args.output_dir)

    elif args.covariance == 'power_law':
        # A5: non-isotropic covariance sweep
        print(f"Running power-law covariance sweep: alphas = {args.alphas}")
        all_summaries = {}
        all_raw = []

        # Always include isotropic baseline
        print("\n--- Isotropic baseline ---")
        df_iso = run_single_experiment(
            p=args.p, gamma=args.gammas[0], n_target=args.n_target,
            sigma_source_sq=args.sigma_source**2,
            sigma_target_sq=args.sigma_target**2,
            rhos=rhos, lambdas=lambdas, lambda_target=args.lambda_target,
            n_seeds=args.n_seeds)
        df_iso['cov_type'] = 'isotropic'
        all_raw.append(df_iso)
        all_summaries['Isotropic'] = summarize(df_iso)

        for alpha in args.alphas:
            print(f"\n--- Power-law alpha = {alpha} ---")
            Sigma = generate_covariance(args.p, 'power_law', alpha)
            df = run_single_experiment(
                p=args.p, gamma=args.gammas[0], n_target=args.n_target,
                sigma_source_sq=args.sigma_source**2,
                sigma_target_sq=args.sigma_target**2,
                rhos=rhos, lambdas=lambdas, lambda_target=args.lambda_target,
                n_seeds=args.n_seeds,
                Sigma_source=Sigma, Sigma_target=Sigma)
            df['cov_type'] = f'power_law_alpha={alpha}'
            all_raw.append(df)
            all_summaries[rf'Power-law $\alpha$={alpha}'] = summarize(df)

        pd.concat(all_raw).to_csv(
            os.path.join(args.output_dir, 'phase_transition_covariance.csv'), index=False)
        plot_covariance_sweep(all_summaries, args.output_dir)

    else:
        # Default: single isotropic run
        gamma = args.gammas[0]
        print(f"Running single experiment: gamma={gamma}, isotropic")
        df = run_single_experiment(
            p=args.p, gamma=gamma, n_target=args.n_target,
            sigma_source_sq=args.sigma_source**2,
            sigma_target_sq=args.sigma_target**2,
            rhos=rhos, lambdas=lambdas, lambda_target=args.lambda_target,
            n_seeds=args.n_seeds)

        summary = summarize(df)
        print("\nSummary Results:")
        print(summary)

        df.to_csv(os.path.join(args.output_dir, 'phase_transition_full_results.csv'),
                  index=False)
        plot_single(summary, args.output_dir)


if __name__ == "__main__":
    main()
