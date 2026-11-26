"""
Main training script for MLP model.
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, parse_args
from models.mlp import MLP
from utils.data_utils import prepare_data, prepare_data_by_drug_count, DataProcessor


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['aupr'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = 0.0
        metrics['aupr'] = 0.0

    # Add prediction distribution
    metrics['actual_pos'] = int(np.sum(y_true == 1))
    metrics['actual_neg'] = int(np.sum(y_true == 0))
    metrics['pred_pos'] = int(np.sum(y_pred == 1))
    metrics['pred_neg'] = int(np.sum(y_pred == 0))

    # Confusion matrix elements
    metrics['tp'] = int(np.sum((y_true == 1) & (y_pred == 1)))
    metrics['tn'] = int(np.sum((y_true == 0) & (y_pred == 0)))
    metrics['fp'] = int(np.sum((y_true == 0) & (y_pred == 1)))
    metrics['fn'] = int(np.sum((y_true == 1) & (y_pred == 0)))

    return metrics


def train_epoch(model, drug_features, se_indices, labels, optimizer, criterion, batch_size, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = drug_features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Shuffle indices
    perm = torch.randperm(num_samples, device=device)
    drug_features = drug_features[perm]
    se_indices = se_indices[perm]
    labels = labels[perm]

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_drug = drug_features[start_idx:end_idx]
        batch_se = se_indices[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        optimizer.zero_grad()
        logits = model(batch_drug, batch_se)
        loss = criterion(logits, batch_labels)

        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * (end_idx - start_idx)

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, drug_features, se_indices, labels, batch_size, device):
    """Evaluate model."""
    model.eval()
    num_samples = drug_features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_preds = []
    all_probs = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_drug = drug_features[start_idx:end_idx]
        batch_se = se_indices[start_idx:end_idx]

        logits = model(batch_drug, batch_se)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_probs.append(probs[:, 1].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    labels_np = labels.cpu().numpy()

    metrics = compute_metrics(labels_np, all_preds, all_probs)
    return metrics


def train(config):
    """Main training function."""
    print("=" * 60)
    print("MLP Training for DDI Side Effect Prediction")
    print("=" * 60)
    print(f"\nDevice: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"Aggregation: {config.aggregation}")

    # Set seed
    set_seed(config.seed)

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, config.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Load and prepare data
    print("\n" + "=" * 60)
    print(f"Loading Data (split_mode: {config.split_mode})...")
    print("=" * 60)

    if config.split_mode == 'by_drug_count':
        data = prepare_data_by_drug_count(
            data_dir=config.data_dir,
            train_val_drug_range=(config.train_val_min_drugs, config.train_val_max_drugs),
            test_drug_range=(config.test_min_drugs, config.test_max_drugs),
            train_val_ratio=tuple(config.train_val_split),
            balance=config.balance,
            device=config.device,
            random_state=config.seed,
            max_samples_per_file=config.max_samples_per_file,
            drug_emb_method=config.drug_emb_method,
            drug_emb_dim=config.drug_emb_dim,
            aggregation=config.aggregation
        )
    else:
        data = prepare_data(
            data_dir=config.data_dir,
            min_drugs=config.min_drugs,
            max_drugs=config.max_drugs,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            balance=config.balance,
            device=config.device,
            random_state=config.seed,
            max_samples_per_file=config.max_samples_per_file,
            drug_emb_method=config.drug_emb_method,
            drug_emb_dim=config.drug_emb_dim,
            aggregation=config.aggregation
        )

    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing Model...")
    print("=" * 60)

    model = MLP(
        num_se=data['num_se'],
        drug_emb_dim=config.drug_emb_dim,
        se_emb_dim=config.se_emb_dim,
        drug_out_dim=config.drug_out_dim,
        se_out_dim=config.se_out_dim,
        mlp_hidden_dims=config.mlp_hidden_dims,
        num_classes=2,
        dropout=config.dropout
    ).to(config.device)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    if config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
        scheduler_type = 'epoch'
    elif config.scheduler == 'cosine_warmrestart':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.min_lr
        )
        scheduler_type = 'epoch'
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        scheduler_type = 'metric'

    print(f"Scheduler: {config.scheduler}")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='max'
    )

    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        run_name = config.wandb_run_name or f"mlp_{config.aggregation}_{timestamp}"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=vars(config),
            dir=output_dir
        )
        wandb.watch(model, log='all', log_freq=100)
        print(f"Wandb initialized: {config.wandb_project}/{run_name}")
    elif config.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available.")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'test_metrics': []
    }

    best_val_f1 = 0

    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(
            model,
            train_data['drug_features'],
            train_data['se_indices'],
            train_data['labels'],
            optimizer,
            criterion,
            config.batch_size,
            config.device
        )

        # Evaluate on training set
        train_metrics = evaluate(
            model,
            train_data['drug_features'],
            train_data['se_indices'],
            train_data['labels'],
            config.batch_size,
            config.device
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_data['drug_features'],
            val_data['se_indices'],
            val_data['labels'],
            config.batch_size,
            config.device
        )

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler_type == 'epoch':
            scheduler.step()
        else:
            scheduler.step(val_metrics['f1'])

        # Early stopping
        early_stopping(val_metrics['f1'], model)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

        # Wandb logging
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/lr': current_lr,
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/auc': train_metrics['auc'],
                'train/aupr': train_metrics['aupr'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1': val_metrics['f1'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/auc': val_metrics['auc'],
                'val/aupr': val_metrics['aupr'],
            })

        # Logging
        if epoch % config.log_interval == 0 or epoch == 1 or early_stopping.early_stop:
            print(f"\nEpoch {epoch}/{config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
            print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}, AUPR: {train_metrics['aupr']:.4f}")
            print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            if config.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(output_dir, 'best_model.pt'))

        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    early_stopping.load_best_model(model)

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_metrics = evaluate(
        model,
        test_data['drug_features'],
        test_data['se_indices'],
        test_data['labels'],
        config.batch_size,
        config.device
    )

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  AUPR:      {test_metrics['aupr']:.4f}")

    print(f"\nPrediction Distribution:")
    print(f"  Actual Positive:    {test_metrics['actual_pos']}")
    print(f"  Actual Negative:    {test_metrics['actual_neg']}")
    print(f"  Predicted Positive: {test_metrics['pred_pos']}")
    print(f"  Predicted Negative: {test_metrics['pred_neg']}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {test_metrics['tp']}, FP: {test_metrics['fp']}")
    print(f"  FN: {test_metrics['fn']}, TN: {test_metrics['tn']}")

    # Log test metrics to wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'test/accuracy': test_metrics['accuracy'],
            'test/f1': test_metrics['f1'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/auc': test_metrics['auc'],
            'test/aupr': test_metrics['aupr'],
        })
        wandb.run.summary['best_val_f1'] = best_val_f1
        wandb.run.summary['test_accuracy'] = test_metrics['accuracy']
        wandb.run.summary['test_f1'] = test_metrics['f1']
        wandb.run.summary['test_auc'] = test_metrics['auc']
        wandb.run.summary['test_aupr'] = test_metrics['aupr']

    # Save results
    results = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'history': history
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Finish wandb run
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished")

    return model, results


def main():
    """Main entry point."""
    args = parse_args()
    config = Config(args)
    print(config)

    model, results = train(config)

    return model, results


if __name__ == '__main__':
    main()
