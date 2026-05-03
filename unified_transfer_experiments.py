"""
Unified Transfer Learning Experiments
=====================================

This script reproduces three standard transfer learning experiments to demonstrate
that source-optimal regularization differs from transfer-optimal regularization.

Experiments:
1. MNIST: MLP, Round digits (0,3,6,8,9) -> Angular digits (1,2,4,5,7)
2. CIFAR-10: CNN, Animals -> Vehicles+
3. NLP: MLP on TF-IDF, 20 Newsgroups Tech -> Non-Tech categories

Structure (same for all):
- Sweep weight decay on source model training
- Transfer using both standard fine-tuning (Adam + WD) and explicit L2-SP
- Track parameter distance from initialization during transfer
- Compare source-optimal WD vs transfer-optimal WD

Author: C. Evans Hedges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

WEIGHT_DECAYS = [0.0] + np.logspace(-6, -1, 18).tolist()

CONFIG = {
    'mnist': {
        'weight_decays': WEIGHT_DECAYS,
        'source_epochs': 10,
        'transfer_epochs': 5,
        'lr': 0.001,
        'batch_size': 256,
        'n_seeds': 10,
        'target_fraction': 0.10,
        'transfer_wd': 1e-4,
        'l2sp_lambda': 1e-4,
    },
    'cifar10': {
        'weight_decays': WEIGHT_DECAYS,
        'source_epochs': 15,
        'transfer_epochs': 8,
        'lr': 0.001,
        'batch_size': 128,
        'n_seeds': 10,
        'target_fraction': 0.05,
        'transfer_wd': 1e-4,
        'l2sp_lambda': 1e-4,
    },
    'nlp': {
        'weight_decays': WEIGHT_DECAYS,
        'source_epochs': 15,
        'transfer_epochs': 8,
        'lr': 0.001,
        'batch_size': 64,
        'n_seeds': 10,
        'target_fraction': 0.10,
        'max_features': 5000,
        'transfer_wd': 1e-4,
        'l2sp_lambda': 1e-4,
    }
}

# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class RemapLabels(Dataset):
    """Wrapper to remap labels to [0, num_classes)."""
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, self.label_mapping[label]

class NoisyLabelDataset(Dataset):
    """Wrapper to add label noise to a dataset."""
    def __init__(self, dataset, noise_rate=0.0, num_classes=5, seed=42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes

        np.random.seed(seed)
        self.noisy_labels = []

        for i in range(len(dataset)):
            _, original_label = dataset[i]
            if np.random.random() < noise_rate:
                possible_labels = [l for l in range(num_classes) if l != original_label]
                noisy_label = np.random.choice(possible_labels)
            else:
                noisy_label = original_label
            self.noisy_labels.append(noisy_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.noisy_labels[idx]

# =============================================================================
# Models
# =============================================================================

class MLP(nn.Module):
    """2-layer MLP for MNIST."""
    def __init__(self, input_dim=784, hidden1=128, hidden2=64, num_classes=5):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10."""
    def __init__(self, num_classes=5):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

class TextMLP(nn.Module):
    """MLP for text classification on TF-IDF features."""
    def __init__(self, input_dim, hidden1=256, hidden2=128, num_classes=9):
        super(TextMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

# =============================================================================
# Training & Evaluation
# =============================================================================

def _param_distance(model, ref_params, device):
    """Compute ||theta - theta_ref||_2."""
    total = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ref_params:
                total += torch.sum((param - ref_params[name].to(device)) ** 2).item()
    return np.sqrt(total)


def train_model(model, train_dataset, epochs, lr, weight_decay, batch_size, device):
    """Train a model with Adam + weight decay (used for source training and scratch baselines)."""
    model = model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return model


def train_standard_transfer(model, train_dataset, epochs, lr, weight_decay,
                            batch_size, device, reference_params):
    """Standard fine-tuning (Adam + WD toward zero), tracking ||theta - theta_0|| per epoch."""
    model = model.to(device)
    model.train()

    ref = {n: p.clone().detach() for n, p in reference_params.items()}
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    distances = [_param_distance(model, ref, device)]
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        distances.append(_param_distance(model, ref, device))

    return model, distances


def train_l2sp_transfer(model, train_dataset, epochs, lr, l2sp_lambda,
                        reference_params, batch_size, device):
    """Fine-tuning with explicit L2-SP penalty toward source parameters."""
    model = model.to(device)
    model.train()

    ref = {n: p.clone().detach().to(device) for n, p in reference_params.items()}
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    distances = [_param_distance(model, ref, device)]
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)

            l2sp_penalty = sum(
                torch.sum((p - ref[n]) ** 2)
                for n, p in model.named_parameters() if n in ref
            )
            loss = loss + (l2sp_lambda / 2.0) * l2sp_penalty

            loss.backward()
            optimizer.step()
        distances.append(_param_distance(model, ref, device))

    return model, distances


def evaluate_model(model, test_dataset, batch_size, device):
    """Evaluate model accuracy."""
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_data(target_fraction=0.10, seed=42):
    """Load MNIST split into Round vs Angular digits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=True, transform=transform
    )

    def split_by_digits(dataset, digits):
        indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
        return Subset(dataset, indices)

    source_digits = [0, 3, 6, 8, 9]
    source_mapping = {0: 0, 3: 1, 6: 2, 8: 3, 9: 4}
    train_source = RemapLabels(split_by_digits(train_dataset, source_digits), source_mapping)
    test_source = RemapLabels(split_by_digits(test_dataset, source_digits), source_mapping)

    target_digits = [1, 2, 4, 5, 7]
    target_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 7: 4}
    train_target_full = RemapLabels(split_by_digits(train_dataset, target_digits), target_mapping)
    test_target = RemapLabels(split_by_digits(test_dataset, target_digits), target_mapping)

    set_seed(seed)
    n_target_samples = int(len(train_target_full) * target_fraction)
    target_indices = torch.randperm(len(train_target_full))[:n_target_samples].tolist()
    train_target = Subset(train_target_full, target_indices)

    return train_source, test_source, train_target, test_target


def load_cifar10_data(target_fraction=0.05, seed=42):
    """Load CIFAR-10 split into Animals vs Vehicles+."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=False, download=True, transform=transform
    )

    def split_by_classes(dataset, class_indices):
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_indices]
        return Subset(dataset, indices)

    source_classes = [2, 3, 4, 5, 6]
    source_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    train_source = RemapLabels(split_by_classes(train_dataset, source_classes), source_mapping)
    test_source = RemapLabels(split_by_classes(test_dataset, source_classes), source_mapping)

    target_classes = [0, 1, 7, 8, 9]
    target_mapping = {0: 0, 1: 1, 7: 2, 8: 3, 9: 4}
    train_target_full = RemapLabels(split_by_classes(train_dataset, target_classes), target_mapping)
    test_target = RemapLabels(split_by_classes(test_dataset, target_classes), target_mapping)

    set_seed(seed)
    n_target_samples = int(len(train_target_full) * target_fraction)
    target_indices = torch.randperm(len(train_target_full))[:n_target_samples].tolist()
    train_target = Subset(train_target_full, target_indices)

    return train_source, test_source, train_target, test_target


class TensorDatasetWrapper(Dataset):
    """Simple wrapper for tensor data."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_newsgroups_data(max_features=5000, target_fraction=0.10, seed=42):
    """Load 20 Newsgroups split into Tech vs Non-Tech categories."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    source_categories = [
        'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x',
        'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'
    ]
    target_categories = [
        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'talk.politics.misc', 'talk.politics.guns', 'talk.religion.misc',
        'misc.forsale', 'alt.atheism'
    ]

    print("Loading 20 Newsgroups dataset...")

    source_train = fetch_20newsgroups(subset='train', categories=source_categories,
                                       remove=('headers', 'footers', 'quotes'))
    source_test = fetch_20newsgroups(subset='test', categories=source_categories,
                                      remove=('headers', 'footers', 'quotes'))
    target_train_full = fetch_20newsgroups(subset='train', categories=target_categories,
                                            remove=('headers', 'footers', 'quotes'))
    target_test = fetch_20newsgroups(subset='test', categories=target_categories,
                                      remove=('headers', 'footers', 'quotes'))

    print(f"Fitting TF-IDF vectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    X_source_train = vectorizer.fit_transform(source_train.data).toarray()
    X_source_test = vectorizer.transform(source_test.data).toarray()
    X_target_train_full = vectorizer.transform(target_train_full.data).toarray()
    X_target_test = vectorizer.transform(target_test.data).toarray()

    source_label_map = {old: new for new, old in enumerate(sorted(set(source_train.target)))}
    target_label_map = {old: new for new, old in enumerate(sorted(set(target_train_full.target)))}

    y_source_train = np.array([source_label_map[l] for l in source_train.target])
    y_source_test = np.array([source_label_map[l] for l in source_test.target])
    y_target_train_full = np.array([target_label_map[l] for l in target_train_full.target])
    y_target_test = np.array([target_label_map[l] for l in target_test.target])

    np.random.seed(seed)
    n_target_samples = int(len(y_target_train_full) * target_fraction)
    target_indices = np.random.permutation(len(y_target_train_full))[:n_target_samples]
    X_target_train = X_target_train_full[target_indices]
    y_target_train = y_target_train_full[target_indices]

    train_source = TensorDatasetWrapper(torch.FloatTensor(X_source_train),
                                        torch.LongTensor(y_source_train))
    test_source = TensorDatasetWrapper(torch.FloatTensor(X_source_test),
                                       torch.LongTensor(y_source_test))
    train_target = TensorDatasetWrapper(torch.FloatTensor(X_target_train),
                                        torch.LongTensor(y_target_train))
    test_target = TensorDatasetWrapper(torch.FloatTensor(X_target_test),
                                       torch.LongTensor(y_target_test))

    n_source_classes = len(source_label_map)
    n_target_classes = len(target_label_map)

    print(f"  Source (Tech): {len(train_source)} train, {len(test_source)} test, "
          f"{n_source_classes} classes")
    print(f"  Target (Non-Tech): {len(train_target)} train ({target_fraction:.0%}), "
          f"{len(test_target)} test, {n_target_classes} classes")

    return (train_source, test_source, train_target, test_target,
            max_features, n_source_classes, n_target_classes)

# =============================================================================
# Experiment Runners
# =============================================================================

def _record_distances(dist_records, seed, wd, std_dists, l2sp_dists, run_l2sp):
    """Append per-epoch distance records."""
    for epoch in range(len(std_dists)):
        d = {'seed': seed, 'weight_decay': wd, 'epoch': epoch,
             'distance_standard': std_dists[epoch]}
        if run_l2sp and epoch < len(l2sp_dists):
            d['distance_l2sp'] = l2sp_dists[epoch]
        dist_records.append(d)


def _build_result_row(seed, wd, source_acc, std_acc, scratch_acc,
                      l2sp_acc, run_l2sp):
    """Build a result row dict."""
    row = {
        'seed': seed, 'weight_decay': wd,
        'source_acc': source_acc,
        'transfer_acc_standard': std_acc,
        'scratch_acc': scratch_acc,
        'transfer_benefit_standard': std_acc - scratch_acc,
    }
    if run_l2sp:
        row['transfer_acc_l2sp'] = l2sp_acc
        row['transfer_benefit_l2sp'] = l2sp_acc - scratch_acc
    return row


def run_mnist_experiment(device, output_dir, run_l2sp=True):
    """Run MNIST transfer experiment with standard and optionally L2-SP transfer."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: MNIST (Round -> Angular digits)")
    print("="*70)

    cfg = CONFIG['mnist']
    results, dist_records = [], []

    for seed in tqdm(range(cfg['n_seeds']), desc='Seeds'):
        set_seed(seed)
        train_source, test_source, train_target, test_target = load_mnist_data(
            target_fraction=cfg['target_fraction'], seed=seed
        )

        scratch_acc = None
        for wd in tqdm(cfg['weight_decays'], desc='Weight Decay', leave=False):
            source_model = MLP()
            source_model = train_model(source_model, train_source, cfg['source_epochs'],
                                       cfg['lr'], wd, cfg['batch_size'], device)
            source_acc = evaluate_model(source_model, test_source, cfg['batch_size'], device)
            source_params = {n: p.clone().cpu() for n, p in source_model.named_parameters()}

            # Standard transfer
            std_model = MLP()
            std_model.load_state_dict(source_model.state_dict())
            std_model, std_dists = train_standard_transfer(
                std_model, train_target, cfg['transfer_epochs'],
                cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device, source_params)
            std_acc = evaluate_model(std_model, test_target, cfg['batch_size'], device)

            # L2-SP transfer
            l2sp_acc, l2sp_dists = None, []
            if run_l2sp:
                l2sp_model = MLP()
                l2sp_model.load_state_dict(source_model.state_dict())
                l2sp_model, l2sp_dists = train_l2sp_transfer(
                    l2sp_model, train_target, cfg['transfer_epochs'],
                    cfg['lr'], cfg['l2sp_lambda'], source_params,
                    cfg['batch_size'], device)
                l2sp_acc = evaluate_model(l2sp_model, test_target, cfg['batch_size'], device)

            # Scratch baseline (once per seed)
            if scratch_acc is None:
                scratch_model = MLP()
                scratch_model = train_model(scratch_model, train_target, cfg['transfer_epochs'],
                                            cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device)
                scratch_acc = evaluate_model(scratch_model, test_target, cfg['batch_size'], device)

            results.append(_build_result_row(seed, wd, source_acc, std_acc,
                                             scratch_acc, l2sp_acc, run_l2sp))
            _record_distances(dist_records, seed, wd, std_dists, l2sp_dists, run_l2sp)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'mnist_raw.csv'), index=False)
    pd.DataFrame(dist_records).to_csv(
        os.path.join(output_dir, 'mnist_param_distances.csv'), index=False)

    summary = analyze_results(df, 'MNIST', run_l2sp=run_l2sp)
    summary.to_csv(os.path.join(output_dir, 'mnist_summary.csv'), index=False)
    return df, summary


def run_cifar10_experiment(device, output_dir, run_l2sp=True):
    """Run CIFAR-10 transfer experiment."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: CIFAR-10 (Animals -> Vehicles+)")
    print("="*70)

    cfg = CONFIG['cifar10']
    results, dist_records = [], []

    for seed in tqdm(range(cfg['n_seeds']), desc='Seeds'):
        set_seed(seed)
        train_source, test_source, train_target, test_target = load_cifar10_data(
            target_fraction=cfg['target_fraction'], seed=seed
        )

        scratch_acc = None
        for wd in tqdm(cfg['weight_decays'], desc='Weight Decay', leave=False):
            source_model = SmallCNN()
            source_model = train_model(source_model, train_source, cfg['source_epochs'],
                                       cfg['lr'], wd, cfg['batch_size'], device)
            source_acc = evaluate_model(source_model, test_source, cfg['batch_size'], device)
            source_params = {n: p.clone().cpu() for n, p in source_model.named_parameters()}

            # Standard transfer
            std_model = SmallCNN()
            std_model.load_state_dict(source_model.state_dict())
            std_model, std_dists = train_standard_transfer(
                std_model, train_target, cfg['transfer_epochs'],
                cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device, source_params)
            std_acc = evaluate_model(std_model, test_target, cfg['batch_size'], device)

            # L2-SP transfer
            l2sp_acc, l2sp_dists = None, []
            if run_l2sp:
                l2sp_model = SmallCNN()
                l2sp_model.load_state_dict(source_model.state_dict())
                l2sp_model, l2sp_dists = train_l2sp_transfer(
                    l2sp_model, train_target, cfg['transfer_epochs'],
                    cfg['lr'], cfg['l2sp_lambda'], source_params,
                    cfg['batch_size'], device)
                l2sp_acc = evaluate_model(l2sp_model, test_target, cfg['batch_size'], device)

            # Scratch baseline
            if scratch_acc is None:
                scratch_model = SmallCNN()
                scratch_model = train_model(scratch_model, train_target, cfg['transfer_epochs'],
                                            cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device)
                scratch_acc = evaluate_model(scratch_model, test_target, cfg['batch_size'], device)

            results.append(_build_result_row(seed, wd, source_acc, std_acc,
                                             scratch_acc, l2sp_acc, run_l2sp))
            _record_distances(dist_records, seed, wd, std_dists, l2sp_dists, run_l2sp)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'cifar10_raw.csv'), index=False)
    pd.DataFrame(dist_records).to_csv(
        os.path.join(output_dir, 'cifar10_param_distances.csv'), index=False)

    summary = analyze_results(df, 'CIFAR-10', run_l2sp=run_l2sp)
    summary.to_csv(os.path.join(output_dir, 'cifar10_summary.csv'), index=False)
    return df, summary


def run_nlp_experiment(device, output_dir, run_l2sp=True):
    """Run NLP transfer experiment (20 Newsgroups: Tech -> Non-Tech)."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: NLP (20 Newsgroups: Tech -> Non-Tech)")
    print("="*70)

    cfg = CONFIG['nlp']

    train_source, test_source, _, _, input_dim, n_source_classes, n_target_classes = \
        load_newsgroups_data(max_features=cfg['max_features'],
                             target_fraction=cfg['target_fraction'], seed=42)

    results, dist_records = [], []

    for seed in tqdm(range(cfg['n_seeds']), desc='Seeds'):
        set_seed(seed)
        _, _, train_target, test_target, _, _, _ = load_newsgroups_data(
            max_features=cfg['max_features'],
            target_fraction=cfg['target_fraction'], seed=seed)

        scratch_acc = None
        for wd in tqdm(cfg['weight_decays'], desc='Weight Decay', leave=False):
            source_model = TextMLP(input_dim, num_classes=n_source_classes)
            source_model = train_model(source_model, train_source, cfg['source_epochs'],
                                       cfg['lr'], wd, cfg['batch_size'], device)
            source_acc = evaluate_model(source_model, test_source, cfg['batch_size'], device)

            # Only backbone (fc1, fc2) is transferred for NLP
            backbone_params = {}
            for n, p in source_model.named_parameters():
                if n.startswith('fc1.') or n.startswith('fc2.'):
                    backbone_params[n] = p.clone().cpu()

            # Standard transfer
            std_model = TextMLP(input_dim, num_classes=n_target_classes)
            std_model.fc1.load_state_dict(source_model.fc1.state_dict())
            std_model.fc2.load_state_dict(source_model.fc2.state_dict())
            std_model, std_dists = train_standard_transfer(
                std_model, train_target, cfg['transfer_epochs'],
                cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device, backbone_params)
            std_acc = evaluate_model(std_model, test_target, cfg['batch_size'], device)

            # L2-SP transfer (penalty only on backbone layers)
            l2sp_acc, l2sp_dists = None, []
            if run_l2sp:
                l2sp_model = TextMLP(input_dim, num_classes=n_target_classes)
                l2sp_model.fc1.load_state_dict(source_model.fc1.state_dict())
                l2sp_model.fc2.load_state_dict(source_model.fc2.state_dict())
                l2sp_model, l2sp_dists = train_l2sp_transfer(
                    l2sp_model, train_target, cfg['transfer_epochs'],
                    cfg['lr'], cfg['l2sp_lambda'], backbone_params,
                    cfg['batch_size'], device)
                l2sp_acc = evaluate_model(l2sp_model, test_target, cfg['batch_size'], device)

            # Scratch baseline
            if scratch_acc is None:
                scratch_model = TextMLP(input_dim, num_classes=n_target_classes)
                scratch_model = train_model(scratch_model, train_target, cfg['transfer_epochs'],
                                            cfg['lr'], cfg['transfer_wd'], cfg['batch_size'], device)
                scratch_acc = evaluate_model(scratch_model, test_target, cfg['batch_size'], device)

            results.append(_build_result_row(seed, wd, source_acc, std_acc,
                                             scratch_acc, l2sp_acc, run_l2sp))
            _record_distances(dist_records, seed, wd, std_dists, l2sp_dists, run_l2sp)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'nlp_raw.csv'), index=False)
    pd.DataFrame(dist_records).to_csv(
        os.path.join(output_dir, 'nlp_param_distances.csv'), index=False)

    summary = analyze_results(df, 'NLP (20 Newsgroups)', run_l2sp=run_l2sp)
    summary.to_csv(os.path.join(output_dir, 'nlp_summary.csv'), index=False)
    return df, summary

# =============================================================================
# Analysis & Plotting
# =============================================================================

def analyze_results(df, experiment_name, run_l2sp=True):
    """Analyze results and find optimal weight decays."""
    agg_cols = {
        'source_acc': ['mean', 'std'],
        'transfer_acc_standard': ['mean', 'std'],
        'scratch_acc': ['mean', 'std'],
        'transfer_benefit_standard': ['mean', 'std'],
    }
    if run_l2sp and 'transfer_acc_l2sp' in df.columns:
        agg_cols['transfer_acc_l2sp'] = ['mean', 'std']
        agg_cols['transfer_benefit_l2sp'] = ['mean', 'std']

    df_agg = df.groupby('weight_decay').agg(agg_cols).reset_index()
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

    idx_source = df_agg['source_acc_mean'].idxmax()
    idx_std = df_agg['transfer_acc_standard_mean'].idxmax()

    wd_source_opt = df_agg.loc[idx_source, 'weight_decay']
    wd_std_opt = df_agg.loc[idx_std, 'weight_decay']

    print(f"\n{experiment_name} Results:")
    print(f"  Source-optimal WD:            {wd_source_opt:.2e} "
          f"(acc: {df_agg.loc[idx_source, 'source_acc_mean']:.2f}%)")
    print(f"  Standard-transfer-optimal WD: {wd_std_opt:.2e} "
          f"(acc: {df_agg.loc[idx_std, 'transfer_acc_standard_mean']:.2f}%)")

    if run_l2sp and 'transfer_acc_l2sp_mean' in df_agg.columns:
        idx_l2sp = df_agg['transfer_acc_l2sp_mean'].idxmax()
        wd_l2sp_opt = df_agg.loc[idx_l2sp, 'weight_decay']
        print(f"  L2-SP-transfer-optimal WD:    {wd_l2sp_opt:.2e} "
              f"(acc: {df_agg.loc[idx_l2sp, 'transfer_acc_l2sp_mean']:.2f}%)")

    direction = ("MORE" if wd_std_opt > wd_source_opt else
                 "LESS" if wd_std_opt < wd_source_opt else "~SAME")
    print(f"  -> Transfer needs {direction} regularization than source")

    return df_agg


def plot_results(results_dict, output_dir, run_l2sp=True):
    """Create visualization of all experiments showing both transfer protocols."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    experiments = ['MNIST', 'CIFAR-10', '20 Newsgroups']

    for i, (name, df) in enumerate(results_dict.items()):
        if df is None:
            continue

        ax = axes[i]
        wds = df['weight_decay'].values
        wds_plot = np.array([max(w, 1e-7) for w in wds])

        ax.plot(wds_plot, df['source_acc_mean'], 'o-', label='Source',
                color='#2E86AB', linewidth=2, markersize=4)
        ax.fill_between(wds_plot,
                        df['source_acc_mean'] - df['source_acc_std'],
                        df['source_acc_mean'] + df['source_acc_std'],
                        alpha=0.15, color='#2E86AB')

        ax.plot(wds_plot, df['transfer_acc_standard_mean'], 's--',
                label='Transfer (standard)', color='#A23B72', linewidth=2, markersize=4)
        ax.fill_between(wds_plot,
                        df['transfer_acc_standard_mean'] - df['transfer_acc_standard_std'],
                        df['transfer_acc_standard_mean'] + df['transfer_acc_standard_std'],
                        alpha=0.10, color='#A23B72')

        if run_l2sp and 'transfer_acc_l2sp_mean' in df.columns:
            ax.plot(wds_plot, df['transfer_acc_l2sp_mean'], '^:',
                    label='Transfer (L2-SP)', color='#F18F01', linewidth=2, markersize=4)
            ax.fill_between(wds_plot,
                            df['transfer_acc_l2sp_mean'] - df['transfer_acc_l2sp_std'],
                            df['transfer_acc_l2sp_mean'] + df['transfer_acc_l2sp_std'],
                            alpha=0.10, color='#F18F01')

        idx_src = df['source_acc_mean'].idxmax()
        idx_std = df['transfer_acc_standard_mean'].idxmax()
        ax.axvline(max(df.loc[idx_src, 'weight_decay'], 1e-7), color='green',
                   linestyle=':', alpha=0.7, label='Source Opt')
        ax.axvline(max(df.loc[idx_std, 'weight_decay'], 1e-7), color='red',
                   linestyle=':', alpha=0.7, label='Std Transfer Opt')

        if run_l2sp and 'transfer_acc_l2sp_mean' in df.columns:
            idx_l2sp = df['transfer_acc_l2sp_mean'].idxmax()
            ax.axvline(max(df.loc[idx_l2sp, 'weight_decay'], 1e-7), color='#F18F01',
                       linestyle='-.', alpha=0.7, label='L2-SP Transfer Opt')

        ax.set_xscale('log')
        ax.set_xlabel('Source Weight Decay', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.set_title(experiments[i], fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Source-Optimal vs Transfer-Optimal Weight Decay',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unified_results.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'unified_results.png')}")


def plot_param_distances(output_dir):
    """Plot parameter distance from initialization during fine-tuning."""
    files = {
        'MNIST': 'mnist_param_distances.csv',
        'CIFAR-10': 'cifar10_param_distances.csv',
        '20 Newsgroups': 'nlp_param_distances.csv',
    }
    available = {k: v for k, v in files.items()
                 if os.path.exists(os.path.join(output_dir, v))}
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (exp_name, fname) in zip(axes, available.items()):
        df = pd.read_csv(os.path.join(output_dir, fname))

        all_wds = sorted(df['weight_decay'].unique())
        n = len(all_wds)
        pick = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        selected = [all_wds[i] for i in pick if i < n]

        cmap = plt.cm.viridis(np.linspace(0, 1, len(selected)))
        for wd, c in zip(selected, cmap):
            sub = df[df['weight_decay'] == wd]
            mean_std = sub.groupby('epoch')['distance_standard'].mean()
            ax.plot(mean_std.index, mean_std.values, '-', color=c,
                    label=f'Std WD={wd:.1e}', linewidth=1.5)
            if 'distance_l2sp' in sub.columns and sub['distance_l2sp'].notna().any():
                mean_l2 = sub.groupby('epoch')['distance_l2sp'].mean()
                ax.plot(mean_l2.index, mean_l2.values, '--', color=c,
                        linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\|\theta - \theta_0\|_2$')
        ax.set_title(exp_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter Distance from Initialization\n(solid = standard, dashed = L2-SP)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'param_distances.png'),
                dpi=150, bbox_inches='tight')
    print(f"Param distance plot saved to {os.path.join(output_dir, 'param_distances.png')}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Transfer Learning Experiments')
    parser.add_argument('--experiments', nargs='+', default=['mnist', 'cifar10', 'nlp'],
                        choices=['mnist', 'cifar10', 'nlp'],
                        help='Which experiments to run')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of seeds')
    parser.add_argument('--no-l2sp', action='store_true',
                        help='Skip L2-SP transfer runs (faster, standard-only)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    run_l2sp = not args.no_l2sp
    if args.seeds:
        for cfg in CONFIG.values():
            cfg['n_seeds'] = args.seeds

    summaries = {}
    start_time = time.time()

    if 'mnist' in args.experiments:
        _, summaries['mnist'] = run_mnist_experiment(device, args.output_dir, run_l2sp)

    if 'cifar10' in args.experiments:
        _, summaries['cifar10'] = run_cifar10_experiment(device, args.output_dir, run_l2sp)

    if 'nlp' in args.experiments:
        _, summaries['nlp'] = run_nlp_experiment(device, args.output_dir, run_l2sp)

    plot_results(summaries, args.output_dir, run_l2sp=run_l2sp)
    plot_param_distances(args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
