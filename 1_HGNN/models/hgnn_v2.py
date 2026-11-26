"""
Improved Hypergraph Neural Network model (v2) for drug-drug interaction side effect prediction.

Improvements over v1:
1. Residual connections for better gradient flow
2. Multi-scale feature aggregation from different layers
3. Pre-LayerNorm for more stable training
4. Gated aggregation for hyperedge features
5. Dropout scheduling
6. Optional edge attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import Optional, List


class ResidualHypergraphConv(nn.Module):
    """Hypergraph convolution layer with residual connection and pre-norm."""

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

        # Pre-norm
        self.norm = nn.LayerNorm(in_channels)

        # Hypergraph convolution
        self.conv = HypergraphConv(
            in_channels,
            out_channels,
            use_attention=use_attention,
            heads=heads if use_attention else 1,
            concat=False,
            dropout=dropout
        )

        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm
        h = self.norm(x)

        # Convolution
        h = self.conv(h, hyperedge_index, hyperedge_attr=hyperedge_attr)
        h = self.activation(h)
        h = self.dropout(h)

        # Residual connection
        return h + self.residual_proj(x)


class GatedAggregation(nn.Module):
    """Gated aggregation for combining node features to hyperedge features."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        # Apply gate
        gated_x = x * self.gate(x)

        # Aggregate to hyperedges
        hyperedge_features = torch.mm(incidence_matrix.t(), gated_x)

        # Normalize by degree
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_features = hyperedge_features / degree

        # Transform
        return self.transform(hyperedge_features)


class MultiScaleDrugEncoder(nn.Module):
    """Multi-scale drug encoder that concatenates features from all layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 3,
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
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Hypergraph convolution layers (all same dimension for residual)
        self.convs = nn.ModuleList([
            ResidualHypergraphConv(
                hidden_dim, hidden_dim, heads=heads,
                dropout=dropout, use_attention=use_attention
            )
            for _ in range(num_layers)
        ])

        # Hyperedge attribute projections
        self.hyperedge_attr_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Multi-scale aggregation (combines features from all layers)
        # Total features = hidden_dim * (num_layers + 1) (input + each layer output)
        total_dim = hidden_dim * (num_layers + 1)

        # Gated aggregation for each scale
        self.gated_aggs = nn.ModuleList([
            GatedAggregation(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers + 1)
        ])

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(
        self,
        drug_emb: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        # Convert incidence matrix to edge index
        hyperedge_index = self._incidence_to_edge_index(incidence_matrix)

        # Input projection
        x = self.input_proj(drug_emb)

        # Collect multi-scale features
        all_features = [self.gated_aggs[0](x, incidence_matrix)]

        # Hypergraph convolutions
        for i, conv in enumerate(self.convs):
            # Compute hyperedge attributes
            hyperedge_attr = self._compute_hyperedge_attr(x, incidence_matrix, i)
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)

            # Collect features at this scale
            all_features.append(self.gated_aggs[i + 1](x, incidence_matrix))

        # Concatenate multi-scale features
        multi_scale = torch.cat(all_features, dim=-1)

        # Final projection
        return self.output_proj(multi_scale)

    def _incidence_to_edge_index(self, incidence_matrix: torch.Tensor) -> torch.Tensor:
        node_idx, edge_idx = torch.where(incidence_matrix > 0)
        return torch.stack([node_idx, edge_idx], dim=0)

    def _compute_hyperedge_attr(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        hyperedge_attr = torch.mm(incidence_matrix.t(), x)
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_attr = hyperedge_attr / degree
        return self.hyperedge_attr_projs[layer_idx](hyperedge_attr)


class ImprovedSEProjector(nn.Module):
    """Improved SE projector with deeper network."""

    def __init__(
        self,
        num_se: int,
        emb_dim: int,
        out_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_se, emb_dim)

        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, se_indices: torch.Tensor) -> torch.Tensor:
        se_emb = self.embedding(se_indices)
        return self.projector(se_emb)


class ImprovedMLPClassifier(nn.Module):
    """Improved MLP classifier with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block for MLP."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class HGNNv2(nn.Module):
    """
    Improved Hypergraph Neural Network (v2) for DDI side effect prediction.

    Key improvements:
    1. Multi-scale feature aggregation
    2. Residual connections throughout
    3. Gated aggregation
    4. Deeper architecture
    5. Pre-LayerNorm for stability
    """

    def __init__(
        self,
        num_drugs: int,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        hidden_dim: int = 256,
        drug_out_dim: int = 128,
        se_out_dim: int = 128,
        num_hgnn_layers: int = 3,
        num_heads: int = 8,
        mlp_hidden_dims: List[int] = [256, 128, 64],
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

        # Drug embedding
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_emb_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_emb_dim = drug_emb_dim

        # Multi-scale drug encoder
        self.drug_encoder = MultiScaleDrugEncoder(
            in_dim=actual_drug_emb_dim,
            hidden_dim=hidden_dim,
            out_dim=drug_out_dim,
            num_layers=num_hgnn_layers,
            heads=num_heads,
            dropout=dropout,
            use_attention=use_attention
        )

        # Improved SE projector
        self.se_projector = ImprovedSEProjector(
            num_se=num_se,
            emb_dim=se_emb_dim,
            out_dim=se_out_dim,
            dropout=dropout
        )

        # Feature fusion with attention
        concat_dim = drug_out_dim + se_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.LayerNorm(concat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Improved classifier
        self.classifier = ImprovedMLPClassifier(
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

        # Encode drugs
        drug_features = self.drug_encoder(drug_emb, incidence_matrix)

        # Project SE
        se_features = self.se_projector(se_indices)

        # Fuse features
        combined = torch.cat([drug_features, se_features], dim=-1)
        combined = self.fusion(combined)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
