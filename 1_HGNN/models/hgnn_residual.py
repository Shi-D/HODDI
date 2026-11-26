"""
HGNN with Residual Connections (HGNNv2)

Key improvements:
1. Original embedding residual: raw drug/SE embeddings are added to processed features
2. MLP residual: skip connections in the classifier MLP
3. HGNN layer residual: skip connections between hypergraph conv layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import Optional


class ResidualHypergraphConv(nn.Module):
    """Hypergraph convolution layer with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = HypergraphConv(
            in_channels,
            out_channels,
            use_attention=use_attention,
            heads=heads if use_attention else 1,
            concat=False,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Store input for residual
        identity = x

        # Hypergraph convolution
        out = self.conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
        out = self.norm(out)

        # Residual connection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        out = out + identity

        out = F.relu(out)
        out = self.dropout(out)
        return out


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.norm(out + identity)  # Residual + LayerNorm
        return out


class ResidualMLPClassifier(nn.Module):
    """MLP classifier with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.input_norm(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        return self.output(x)


class SEProjectorWithResidual(nn.Module):
    """SE projector with residual from original embedding."""

    def __init__(
        self,
        num_se: int,
        emb_dim: int,
        out_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(num_se, emb_dim)

        # Projection layers
        self.proj1 = nn.Linear(emb_dim, emb_dim)
        self.proj2 = nn.Linear(emb_dim, out_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection for original embedding
        self.emb_residual_proj = nn.Linear(emb_dim, out_dim)

    def forward(self, se_indices: torch.Tensor) -> torch.Tensor:
        # Original embedding
        se_emb = self.embedding(se_indices)  # (batch, emb_dim)

        # First projection with residual
        x = self.proj1(se_emb)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x + se_emb)  # Residual

        # Second projection
        x = self.proj2(x)
        x = F.relu(x)

        # Add residual from original embedding
        emb_residual = self.emb_residual_proj(se_emb)
        x = self.norm2(x + emb_residual)

        return x


class DrugEncoderWithResidual(nn.Module):
    """Drug encoder with residual connections at multiple levels."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Hypergraph convolution layers with residual
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                ResidualHypergraphConv(
                    hidden_dim, hidden_dim, heads=heads,
                    dropout=dropout, use_attention=use_attention
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.output_norm = nn.LayerNorm(out_dim)

        # Residual projection from input to output (skip all HGNN layers)
        self.global_residual_proj = nn.Linear(in_dim, out_dim)

        # Hyperedge attribute projection
        self.hyperedge_attr_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_emb: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, in_dim)
            incidence_matrix: (num_drugs, num_hyperedges)

        Returns:
            hyperedge_features: (num_hyperedges, out_dim)
        """
        # Convert incidence matrix to edge index
        hyperedge_index = self._incidence_to_edge_index(incidence_matrix)

        # Store original embedding for global residual
        drug_emb_original = drug_emb

        # Input projection
        x = self.input_proj(drug_emb)
        x = F.relu(x)
        x = self.input_norm(x)
        x = self.dropout(x)

        # Hypergraph convolutions with residual (inside each layer)
        for conv in self.convs:
            hyperedge_attr = self._compute_hyperedge_attr(x, incidence_matrix)
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)

        # Output projection
        x = self.output_proj(x)

        # Aggregate node features to hyperedge features
        hyperedge_features = torch.mm(incidence_matrix.t(), x)
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_features = hyperedge_features / degree

        # Global residual: add aggregated original embeddings
        original_hyperedge = torch.mm(incidence_matrix.t(), drug_emb_original)
        original_hyperedge = original_hyperedge / degree
        original_hyperedge = self.global_residual_proj(original_hyperedge)

        hyperedge_features = self.output_norm(hyperedge_features + original_hyperedge)

        return hyperedge_features

    def _incidence_to_edge_index(self, incidence_matrix: torch.Tensor) -> torch.Tensor:
        node_idx, edge_idx = torch.where(incidence_matrix > 0)
        return torch.stack([node_idx, edge_idx], dim=0)

    def _compute_hyperedge_attr(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        hyperedge_attr = torch.mm(incidence_matrix.t(), x)
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_attr = hyperedge_attr / degree
        hyperedge_attr = self.hyperedge_attr_proj(hyperedge_attr)
        return hyperedge_attr


class HGNNResidual(nn.Module):
    """
    HGNN with Residual Connections.

    Architecture:
    1. Drug embeddings -> HGNN layers (with layer residuals) -> Drug features
       + Global residual from original drug embeddings
    2. SE indices -> Embedding -> Projection (with embedding residual) -> SE features
    3. Concatenate [Drug features, SE features, Original drug agg, Original SE emb]
    4. Residual MLP classifier -> Binary classification

    The model combines:
    - Processed features (capture complex interactions)
    - Original embeddings (preserve raw information)
    """

    def __init__(
        self,
        num_drugs: int,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        drug_out_dim: int = 64,
        se_out_dim: int = 64,
        num_hgnn_layers: int = 2,
        num_heads: int = 4,
        mlp_hidden_dim: int = 128,
        mlp_num_blocks: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        drug_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.drug_emb_dim = drug_emb_dim
        self.se_emb_dim = se_emb_dim
        self.drug_out_dim = drug_out_dim
        self.se_out_dim = se_out_dim

        # Drug embedding
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_emb_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_emb_dim = drug_emb_dim

        # Drug encoder with residual
        self.drug_encoder = DrugEncoderWithResidual(
            in_dim=actual_drug_emb_dim,
            hidden_dim=drug_emb_dim,
            out_dim=drug_out_dim,
            num_layers=num_hgnn_layers,
            heads=num_heads,
            dropout=dropout,
            use_attention=use_attention
        )

        # SE projector with residual
        self.se_projector = SEProjectorWithResidual(
            num_se=num_se,
            emb_dim=se_emb_dim,
            out_dim=se_out_dim,
            dropout=dropout
        )

        # Additional projection for original embeddings to add as final residual
        self.drug_emb_final_proj = nn.Linear(actual_drug_emb_dim, drug_out_dim)
        self.se_emb_final_proj = nn.Linear(se_emb_dim, se_out_dim)

        # Classifier input: processed drug + SE features + original embedding residuals
        # We concatenate: drug_out_dim + se_out_dim
        # Then add residual from: drug_out_dim + se_out_dim
        concat_dim = drug_out_dim + se_out_dim

        # Residual MLP classifier
        self.classifier = ResidualMLPClassifier(
            in_dim=concat_dim,
            hidden_dim=mlp_hidden_dim,
            num_blocks=mlp_num_blocks,
            num_classes=num_classes,
            dropout=dropout
        )

        # Final feature norm
        self.feature_norm = nn.LayerNorm(concat_dim)

    def forward(
        self,
        incidence_matrix: torch.Tensor,
        se_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            incidence_matrix: (num_drugs, num_hyperedges)
            se_indices: (num_hyperedges,)

        Returns:
            logits: (num_hyperedges, num_classes)
        """
        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        # Encode drugs (includes internal residuals)
        drug_features = self.drug_encoder(drug_emb, incidence_matrix)

        # Project SE (includes embedding residual)
        se_features = self.se_projector(se_indices)

        # Concatenate processed features
        combined = torch.cat([drug_features, se_features], dim=-1)

        # Compute original embedding aggregation for final residual
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        orig_drug_agg = torch.mm(incidence_matrix.t(), drug_emb) / degree
        orig_drug_agg = self.drug_emb_final_proj(orig_drug_agg)

        orig_se_emb = self.se_projector.embedding(se_indices)
        orig_se_emb = self.se_emb_final_proj(orig_se_emb)

        # Add final residual from original embeddings
        orig_combined = torch.cat([orig_drug_agg, orig_se_emb], dim=-1)
        combined = self.feature_norm(combined + orig_combined)

        # Classify (with internal MLP residuals)
        logits = self.classifier(combined)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
