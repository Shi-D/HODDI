"""
Hypergraph Neural Network model for drug-drug interaction side effect prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import Optional


class AttentionHypergraphConv(nn.Module):
    """Hypergraph convolution layer with attention mechanism."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize AttentionHypergraphConv layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            heads: Number of attention heads
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.use_attention = use_attention

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

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            hyperedge_index: Hyperedge indices (2, num_connections)
            hyperedge_attr: Optional hyperedge attributes

        Returns:
            Updated node features (num_nodes, out_channels)
        """
        out = self.conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


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


class DrugEncoder(nn.Module):
    """Encoder for drug features using hypergraph convolution."""

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
        """
        Initialize Drug Encoder.

        Args:
            in_dim: Input dimension (drug embedding dim)
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension
            num_layers: Number of hypergraph conv layers
            heads: Number of attention heads
            dropout: Dropout rate
            use_attention: Whether to use attention in hypergraph conv
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Hypergraph convolution layers
        # All layers take hidden_dim as input, only last layer outputs out_dim
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            self.convs.append(
                AttentionHypergraphConv(
                    in_ch, out_ch, heads=heads,
                    dropout=dropout, use_attention=use_attention
                )
            )

        # Hyperedge attribute projection for attention
        # Input is always hidden_dim (x before any conv is hidden_dim)
        # Output should match conv's in_channels = hidden_dim
        self.hyperedge_attr_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_emb: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug_emb: Drug embeddings (num_drugs, in_dim)
            incidence_matrix: Incidence matrix (num_drugs, num_hyperedges)

        Returns:
            Hyperedge features (num_hyperedges, out_dim)
        """
        # Convert incidence matrix to edge index
        hyperedge_index = self._incidence_to_edge_index(incidence_matrix)

        # Input projection
        x = self.input_proj(drug_emb)
        x = F.relu(x)
        x = self.dropout(x)

        # Hypergraph convolutions
        for i, conv in enumerate(self.convs):
            # Compute hyperedge attributes for attention
            hyperedge_attr = self._compute_hyperedge_attr(x, incidence_matrix, i)
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)

        # Aggregate node features to hyperedge features
        # H^T @ X: (num_hyperedges, num_drugs) @ (num_drugs, out_dim)
        hyperedge_features = torch.mm(incidence_matrix.t(), x)

        # Normalize by node degree
        degree = incidence_matrix.sum(dim=0, keepdim=True).t()  # (num_hyperedges, 1)
        degree = degree.clamp(min=1)
        hyperedge_features = hyperedge_features / degree

        return hyperedge_features

    def _incidence_to_edge_index(self, incidence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert incidence matrix to edge index format.

        Args:
            incidence_matrix: (num_nodes, num_edges)

        Returns:
            edge_index: (2, num_connections)
        """
        # Find non-zero entries
        node_idx, edge_idx = torch.where(incidence_matrix > 0)
        edge_index = torch.stack([node_idx, edge_idx], dim=0)
        return edge_index

    def _compute_hyperedge_attr(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute hyperedge attributes for attention.

        Args:
            x: Node features (num_nodes, dim)
            incidence_matrix: (num_nodes, num_edges)
            layer_idx: Layer index

        Returns:
            Hyperedge attributes (num_edges, dim)
        """
        # Aggregate node features to hyperedges
        hyperedge_attr = torch.mm(incidence_matrix.t(), x)
        degree = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        hyperedge_attr = hyperedge_attr / degree

        # Project
        hyperedge_attr = self.hyperedge_attr_projs[layer_idx](hyperedge_attr)

        return hyperedge_attr


class MLPClassifier(nn.Module):
    """MLP classifier for final prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [64, 32],
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
        """
        Forward pass.

        Args:
            x: Input features (batch_size, in_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        return self.mlp(x)


class HGNN(nn.Module):
    """
    Hypergraph Neural Network for DDI side effect prediction.

    Architecture:
    1. Drug embeddings -> N layers AttentionHypergraphConv -> Drug features (64-dim)
    2. SE indices -> Embedding + Projector -> SE features (64-dim)
    3. Concatenate Drug features + SE features (128-dim)
    4. MLP classifier -> Binary classification
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
        mlp_hidden_dims: list = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        drug_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize HGNN model.

        Args:
            num_drugs: Number of unique drugs
            num_se: Number of unique side effects
            drug_emb_dim: Drug embedding dimension
            se_emb_dim: SE embedding dimension
            drug_out_dim: Drug encoder output dimension
            se_out_dim: SE projector output dimension
            num_hgnn_layers: Number of hypergraph conv layers
            num_heads: Number of attention heads
            mlp_hidden_dims: MLP hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            use_attention: Whether to use attention in hypergraph conv
            drug_embeddings: Pre-computed drug embeddings (optional)
        """
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.drug_out_dim = drug_out_dim
        self.se_out_dim = se_out_dim

        # Drug embedding layer (if no pre-computed embeddings)
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_emb_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_emb_dim = drug_emb_dim

        # Drug encoder with hypergraph convolution
        self.drug_encoder = DrugEncoder(
            in_dim=actual_drug_emb_dim,
            hidden_dim=drug_emb_dim,
            out_dim=drug_out_dim,
            num_layers=num_hgnn_layers,
            heads=num_heads,
            dropout=dropout,
            use_attention=use_attention
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
        incidence_matrix: torch.Tensor,
        se_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            incidence_matrix: Incidence matrix (num_drugs, num_hyperedges)
            se_indices: Side effect indices (num_hyperedges,)

        Returns:
            Logits (num_hyperedges, num_classes)
        """
        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        # Encode drugs through hypergraph conv
        drug_features = self.drug_encoder(drug_emb, incidence_matrix)  # (num_hyperedges, drug_out_dim)

        # Project side effects
        se_features = self.se_projector(se_indices)  # (num_hyperedges, se_out_dim)

        # Concatenate
        combined = torch.cat([drug_features, se_features], dim=-1)  # (num_hyperedges, drug_out_dim + se_out_dim)

        # Classify
        logits = self.classifier(combined)  # (num_hyperedges, num_classes)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.

        Args:
            incidence_matrix: Incidence matrix (num_drugs, num_hyperedges)
            se_indices: Side effect indices (num_hyperedges,)

        Returns:
            Probabilities (num_hyperedges, num_classes)
        """
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            incidence_matrix: Incidence matrix (num_drugs, num_hyperedges)
            se_indices: Side effect indices (num_hyperedges,)

        Returns:
            Predicted labels (num_hyperedges,)
        """
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
