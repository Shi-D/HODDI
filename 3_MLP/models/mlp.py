"""
MLP model for drug-drug interaction side effect prediction.
Uses average drug features concatenated with side effect features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DrugProjector(nn.Module):
    """
    Projects pre-aggregated drug features.
    Drug aggregation is done during data preprocessing for efficiency.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize Drug Projector.

        Args:
            in_dim: Input dimension (drug embedding dim)
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, drug_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug_features: Pre-aggregated drug features (batch_size, in_dim)

        Returns:
            Projected drug features (batch_size, out_dim)
        """
        return self.projector(drug_features)


class SEProjector(nn.Module):
    """Projector for side effect embeddings."""

    def __init__(
        self,
        num_se: int,
        emb_dim: int,
        out_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize SE Projector.

        Args:
            num_se: Number of unique side effects
            emb_dim: Embedding dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(num_se, emb_dim)
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, se_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            se_indices: Side effect indices (batch_size,)

        Returns:
            SE features (batch_size, out_dim)
        """
        se_emb = self.embedding(se_indices)
        return self.projector(se_emb)


class MLPClassifier(nn.Module):
    """MLP classifier for final prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize MLP Classifier.

        Args:
            in_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)


class MLP(nn.Module):
    """
    MLP model for DDI side effect prediction.

    Architecture:
    1. Pre-aggregated drug features -> Projector -> Drug features (64-dim)
    2. SE indices -> Embedding + Projector -> SE features (64-dim)
    3. Concatenate Drug features + SE features (128-dim)
    4. MLP classifier -> Binary classification
    """

    def __init__(
        self,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        drug_out_dim: int = 64,
        se_out_dim: int = 64,
        mlp_hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize MLP model.

        Args:
            num_se: Number of unique side effects
            drug_emb_dim: Drug embedding dimension (input)
            se_emb_dim: SE embedding dimension
            drug_out_dim: Drug projector output dimension
            se_out_dim: SE projector output dimension
            mlp_hidden_dims: MLP hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.num_se = num_se
        self.drug_out_dim = drug_out_dim
        self.se_out_dim = se_out_dim

        # Drug projector (input is pre-aggregated features)
        self.drug_projector = DrugProjector(
            in_dim=drug_emb_dim,
            hidden_dim=drug_emb_dim,
            out_dim=drug_out_dim,
            dropout=dropout
        )

        # SE projector
        self.se_projector = SEProjector(
            num_se=num_se,
            emb_dim=se_emb_dim,
            out_dim=se_out_dim,
            dropout=dropout
        )

        # MLP classifier
        concat_dim = drug_out_dim + se_out_dim
        self.classifier = MLPClassifier(
            in_dim=concat_dim,
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        drug_features: torch.Tensor,
        se_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug_features: Pre-aggregated drug features (batch_size, drug_emb_dim)
            se_indices: Side effect indices (batch_size,)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Project drug features
        drug_out = self.drug_projector(drug_features)  # (batch_size, drug_out_dim)

        # Project side effects
        se_out = self.se_projector(se_indices)  # (batch_size, se_out_dim)

        # Concatenate
        combined = torch.cat([drug_out, se_out], dim=-1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict(self, drug_features: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        """Predict probabilities."""
        logits = self.forward(drug_features, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, drug_features: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        logits = self.forward(drug_features, se_indices)
        return torch.argmax(logits, dim=-1)
