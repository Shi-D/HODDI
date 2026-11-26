"""
Hypergraph Neural Network model v3 for drug-drug interaction side effect prediction.

Based on HGNN v1 with added residual connections for better gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import Optional, List


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

        # Hypergraph convolution from PyG
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
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()

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
        out = F.relu(out)
        out = self.dropout(out)

        # Add residual connection
        out = out + self.residual_proj(identity)

        return out


class SEProjectorV3(nn.Module):
    """Projector for side effect embeddings."""

    def __init__(
        self,
        num_se: int,
        emb_dim: int,
        out_dim: int = 64,
        dropout: float = 0.1
    ):
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
        se_emb = self.embedding(se_indices)
        return self.projector(se_emb)


class DrugEncoderV3(nn.Module):
    """Encoder for drug features using hypergraph convolution with residual connections."""

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
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Hypergraph convolution layers with residual connections
        # All intermediate layers use hidden_dim, last layer outputs out_dim
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            self.convs.append(
                ResidualHypergraphConv(
                    in_ch, out_ch, heads=heads,
                    dropout=dropout, use_attention=use_attention
                )
            )

        # Hyperedge attribute projection for attention
        self.hyperedge_attr_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        drug_emb: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        # Convert incidence matrix to edge index
        hyperedge_index = self._incidence_to_edge_index(incidence_matrix)

        # Input projection
        x = self.input_proj(drug_emb)

        # Hypergraph convolutions with residual connections
        for i, conv in enumerate(self.convs):
            # Compute hyperedge attributes for attention
            hyperedge_attr = self._compute_hyperedge_attr(x, incidence_matrix, i)
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)

        # Aggregate node features to hyperedge features
        hyperedge_features = torch.mm(incidence_matrix.t(), x)

        # Normalize by node degree
        degree = incidence_matrix.sum(dim=0, keepdim=True).t()
        degree = degree.clamp(min=1)
        hyperedge_features = hyperedge_features / degree

        return hyperedge_features

    def _incidence_to_edge_index(self, incidence_matrix: torch.Tensor) -> torch.Tensor:
        node_idx, edge_idx = torch.where(incidence_matrix > 0)
        edge_index = torch.stack([node_idx, edge_idx], dim=0)
        return edge_index

    def _compute_hyperedge_attr(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        # Aggregate node features to hyperedges
        hyperedge_attr = torch.mm(incidence_matrix.t(), x)
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_attr = hyperedge_attr / degree

        # Project
        hyperedge_attr = self.hyperedge_attr_projs[layer_idx](hyperedge_attr)

        return hyperedge_attr


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual_proj(x)


class MLPClassifierV3(nn.Module):
    """MLP classifier with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualMLPBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.output(x)


class HGNNv3(nn.Module):
    """
    Hypergraph Neural Network v3 for DDI side effect prediction.

    Based on HGNN v1 with added residual connections:
    1. Residual connections in hypergraph conv layers
    2. Residual connections in MLP classifier
    3. LayerNorm for stability
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
        mlp_hidden_dims: List[int] = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        drug_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.drug_out_dim = drug_out_dim
        self.se_out_dim = se_out_dim

        # Drug embedding layer
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_emb_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_emb_dim = drug_emb_dim

        # Drug encoder with residual hypergraph convolution
        self.drug_encoder = DrugEncoderV3(
            in_dim=actual_drug_emb_dim,
            hidden_dim=drug_emb_dim,
            out_dim=drug_out_dim,
            num_layers=num_hgnn_layers,
            heads=num_heads,
            dropout=dropout,
            use_attention=use_attention
        )

        # SE projector
        self.se_projector = SEProjectorV3(
            num_se=num_se,
            emb_dim=se_emb_dim,
            out_dim=se_out_dim,
            dropout=dropout
        )

        # MLP classifier with residual connections
        concat_dim = drug_out_dim + se_out_dim
        self.classifier = MLPClassifierV3(
            in_dim=concat_dim,
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        incidence_matrix: torch.Tensor,
        se_indices: torch.Tensor
    ) -> torch.Tensor:
        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        # Encode drugs through hypergraph conv
        drug_features = self.drug_encoder(drug_emb, incidence_matrix)

        # Project side effects
        se_features = self.se_projector(se_indices)

        # Concatenate
        combined = torch.cat([drug_features, se_features], dim=-1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
