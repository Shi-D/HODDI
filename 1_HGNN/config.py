"""
Configuration for HGNN model training.
"""

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HGNN for DDI Side Effect Prediction')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='dataset/HODDI_v4',
                        help='Path to HODDI dataset directory')
    parser.add_argument('--min_drugs', type=int, default=2,
                        help='Minimum number of drugs per hyperedge')
    parser.add_argument('--max_drugs', type=int, default=42,
                        help='Maximum number of drugs per hyperedge')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--balance', action='store_true', default=True,
                        help='Balance positive and negative samples')

    # Split mode arguments
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['random', 'by_drug_count'],
                        help='Data split mode: random or by_drug_count')
    parser.add_argument('--train_val_min_drugs', type=int, default=2,
                        help='Min drugs for train/val in by_drug_count mode')
    parser.add_argument('--train_val_max_drugs', type=int, default=5,
                        help='Max drugs for train/val in by_drug_count mode')
    parser.add_argument('--test_min_drugs', type=int, default=6,
                        help='Min drugs for test in by_drug_count mode')
    parser.add_argument('--test_max_drugs', type=int, default=42,
                        help='Max drugs for test in by_drug_count mode')
    parser.add_argument('--train_val_split', type=float, nargs=2, default=[0.7, 0.2],
                        help='Train:Val ratio within train_val data (default: 0.7 0.2)')

    # Model arguments
    parser.add_argument('--model_version', type=str, default='v1',
                        choices=['v1', 'v2', 'v3', 'v5', 'dit', 'dit_fast', 'residual'],
                        help='Model version: v1, v2, v3, v5 (cross-modal), dit, dit_fast, residual')
    parser.add_argument('--drug_emb_dim', type=int, default=128,
                        help='Drug embedding dimension')
    parser.add_argument('--se_emb_dim', type=int, default=128,
                        help='Side effect embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for HGNNv2')
    parser.add_argument('--drug_out_dim', type=int, default=64,
                        help='Drug encoder output dimension')
    parser.add_argument('--se_out_dim', type=int, default=64,
                        help='SE projector output dimension')
    parser.add_argument('--num_hgnn_layers', type=int, default=2,
                        help='Number of hypergraph conv layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[64, 32],
                        help='MLP hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention in hypergraph conv')

    # Drug embedding method
    parser.add_argument('--drug_emb_method', type=str, default='random',
                        choices=['random', 'morgan', 'chemberta'],
                        help='Drug embedding method')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum improvement for early stopping')

    # Device arguments
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use (auto will select best available)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save the best model')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval (epochs)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['cosine', 'cosine_warmrestart', 'plateau', None],
                        help='Learning rate scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--T_0', type=int, default=10,
                        help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--T_mult', type=int, default=2,
                        help='T_mult for CosineAnnealingWarmRestarts')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use wandb for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='HGNN-DDI',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')

    args = parser.parse_args()

    return args


def get_device(args):
    """Get the device to use for training."""
    import torch

    if args.no_cuda:
        return 'cpu'

    if args.device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    return args.device


class Config:
    """Configuration class for easier access."""

    def __init__(self, args=None):
        if args is None:
            args = parse_args()

        # Copy all arguments
        for key, value in vars(args).items():
            setattr(self, key, value)

        # Set device
        self.device = get_device(args)

    def __repr__(self):
        lines = ['Config:']
        for key, value in vars(self).items():
            lines.append(f'  {key}: {value}')
        return '\n'.join(lines)
